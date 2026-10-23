# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 12 of 197

## 目录 (Table of Contents)

1. [Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning](#paper-0364)
2. [dnaHNet: A Scalable and Hierarchical Foundation Model for Genomic Sequence Learning](#paper-0365)
3. [Even Faster Kernel Matrix Linear Algebra via Density Estimation](#paper-0366)
4. [Modeling Hierarchical Thinking in Large Reasoning Models](#paper-0367)
5. [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](#paper-0368)
6. [DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions](#paper-0369)
7. [MIRA: A Score for Conditional Distribution Accuracy and Model Comparison](#paper-0370)
8. [From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs](#paper-0371)
9. [Self-Distillation Enables Continual Learning](#paper-0372)
10. [S$^3$GNN: Efficient Global Mixing and Local Message Passing for Long-Range Graph Learning](#paper-0373)
11. [Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination](#paper-0374)
12. [Path-dependent Discrete Amortized Inference](#paper-0375)
13. [Incentivizing Truthfulness and Collaborative Fairness in Bayesian Learning](#paper-0376)
14. [RED-HDP-HMM: Observation-Dependent Durations for Bayesian Nonparametric Sequential Models](#paper-0377)
15. [Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math](#paper-0378)
16. [WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution](#paper-0379)
17. [TG-RAG: A Retrieval-Augmented Framework for Reasoning Guidance in Specialized Domains](#paper-0380)
18. [Multimodal Latent Language Modeling with Next-Token Diffusion](#paper-0381)
19. [Training Diffusion Language Models for Black-Box Optimization](#paper-0382)
20. [Learning with Admissibility: Robust Fuzzy Hashing for Cross-Modal Retrieval with Noisy Labels](#paper-0383)
21. [ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools — From Consensus Learning to Ambiguity-Driven Emotion Reasoning](#paper-0384)
22. [Orthogonal Concept Erasure for Diffusion Models](#paper-0385)
23. [Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability](#paper-0386)
24. [Maximum Likelihood Reinforcement Learning](#paper-0387)
25. [The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization](#paper-0388)
26. [NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](#paper-0389)
27. [The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes](#paper-0390)
28. [Is Data Shapley Not Better than Random in Data Selection? Ask NASH](#paper-0391)
29. [Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment](#paper-0392)
30. [Beyond Language Modeling: An Exploration of Multimodal Pretraining](#paper-0393)
31. [Conservation Laws for Modern Neural Architectures](#paper-0394)
32. [Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors](#paper-0395)
33. [Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings](#paper-0396)

---


<a id="paper-0364"></a>

## Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning

### Images

![10c2d93708c005ee6c82883ffa62b5103a1e15bee9c7809f9f948d25bfc6bf1a.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/10c2d93708c005ee6c82883ffa62b5103a1e15bee9c7809f9f948d25bfc6bf1a.jpg)

![1d08ae0da945073509ba99e2bca960a261e896469ae4ea53405b1fe9f0f20665.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/1d08ae0da945073509ba99e2bca960a261e896469ae4ea53405b1fe9f0f20665.jpg)

![1e1ff04b3ad764ba43ff0179d1d5b24534745f0fe10d88423d0751811f639a77.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/1e1ff04b3ad764ba43ff0179d1d5b24534745f0fe10d88423d0751811f639a77.jpg)

![22ea9bf2589ccc76849401e3c5288ae019fb1cedf4bf382c94e8e7232269ed6d.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/22ea9bf2589ccc76849401e3c5288ae019fb1cedf4bf382c94e8e7232269ed6d.jpg)

![2551b3e176a2bb42bdee5d395c0e3e3e50a4401d3f07d828629b70740f1ddd7a.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/2551b3e176a2bb42bdee5d395c0e3e3e50a4401d3f07d828629b70740f1ddd7a.jpg)

![3e567cb031b2d2e3df35f569cdf3e1c0483d59567889c4d80ed62b852970de18.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/3e567cb031b2d2e3df35f569cdf3e1c0483d59567889c4d80ed62b852970de18.jpg)

![3eb3b4fbc4a9e89487edf34efb0d0c89e5ed79fbb312f94ae44457406e71eccf.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/3eb3b4fbc4a9e89487edf34efb0d0c89e5ed79fbb312f94ae44457406e71eccf.jpg)

![8941855e48c45d9584b8be32a7a628bf233b664017e7f6d1072f77740f8030ad.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/8941855e48c45d9584b8be32a7a628bf233b664017e7f6d1072f77740f8030ad.jpg)

![8e1a07c725d8bc057bc21e499aaaabf7608b8544a1d60621e726214c7d034f4f.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/8e1a07c725d8bc057bc21e499aaaabf7608b8544a1d60621e726214c7d034f4f.jpg)

![9c1b0f9bfbc5f344cfe0d0c1a2ef8fbdf70c248cb2b5aaf56c2593d9f203b3b3.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/9c1b0f9bfbc5f344cfe0d0c1a2ef8fbdf70c248cb2b5aaf56c2593d9f203b3b3.jpg)

![ad2caead3d6babe3852b6b1567585e548a81b505d61ea67461493e29a8941752.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/ad2caead3d6babe3852b6b1567585e548a81b505d61ea67461493e29a8941752.jpg)

![b50409924395d513370913848f7f1a22cd2ae97e6b618dffd570e932a45b81c3.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/b50409924395d513370913848f7f1a22cd2ae97e6b618dffd570e932a45b81c3.jpg)

![c6522a4f71f8a4c1c1c5221f36e5eb813d68d2fbb2f3ce14b55ff61df0311719.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/c6522a4f71f8a4c1c1c5221f36e5eb813d68d2fbb2f3ce14b55ff61df0311719.jpg)

![c8e5ebf6a7a7dd5e827d1b3b8632cb0c1ef1a6f257663d305c75b410881100e5.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/c8e5ebf6a7a7dd5e827d1b3b8632cb0c1ef1a6f257663d305c75b410881100e5.jpg)

![d817d827cc7eb91360e1977cabab79f582234d3512281dc7ef8386f24c8aa040.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/d817d827cc7eb91360e1977cabab79f582234d3512281dc7ef8386f24c8aa040.jpg)

![dd0ee0a82500a97457f68af00af8f5d9aefc83e1196537fd6a556161942e9048.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/images/dd0ee0a82500a97457f68af00af8f5d9aefc83e1196537fd6a556161942e9048.jpg)

### Tables

![887560381defcedb5bef1ff5126e5da187e34704ec10b4c377cd05c2ff463618.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/tables/887560381defcedb5bef1ff5126e5da187e34704ec10b4c377cd05c2ff463618.jpg)

![c8b1edd9e31e128a3d4ef31a9f80ddf27b51acc4287bff93763b2fcfcb1e61f1.jpg](../icml_results/370_Many%20Experiments%2C%20Few%20Repetitions%2C%20Unpaired%20Data%2C%20and%20Sparse%20Effects_%20Is%20Causal%20Inference%20Possible_/tables/c8b1edd9e31e128a3d4ef31a9f80ddf27b51acc4287bff93763b2fcfcb1e61f1.jpg)

## Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning


### Images

![02347cf18b1bdcca10f10c1725789e3b751b0cb2fd9daaf484f08ec615ecd37a.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/02347cf18b1bdcca10f10c1725789e3b751b0cb2fd9daaf484f08ec615ecd37a.jpg)

![0a09d1b2986dfb4a1b46d2589a19043202a75bbbc71a0e87734fa93454fef117.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/0a09d1b2986dfb4a1b46d2589a19043202a75bbbc71a0e87734fa93454fef117.jpg)

![0cca36ce271e71ca373a7c21d84d749201d89d312c04cef59a4d20f25147af9a.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/0cca36ce271e71ca373a7c21d84d749201d89d312c04cef59a4d20f25147af9a.jpg)

![168c1e0cc8330bfa8ab0babdad4c472120281a4de3d7a011db348889ca73c48f.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/168c1e0cc8330bfa8ab0babdad4c472120281a4de3d7a011db348889ca73c48f.jpg)

![224ed2d1e0bad690526fbc06c89d57d07cc7973b73af9ff2a9b37ea78d0dec99.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/224ed2d1e0bad690526fbc06c89d57d07cc7973b73af9ff2a9b37ea78d0dec99.jpg)

![2aa8918312b12b63a14536cbbe63f06544226da4baafd5ca485035b7ac895720.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/2aa8918312b12b63a14536cbbe63f06544226da4baafd5ca485035b7ac895720.jpg)

![2b1478fa4d0a3a099d7d24662fe094c77a7f62524f0835c49dfd0faefe2ead32.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/2b1478fa4d0a3a099d7d24662fe094c77a7f62524f0835c49dfd0faefe2ead32.jpg)

![5e49825c539aec77cc5af4c518ce07f3a403eb36aa7f0aa3128743c23a415743.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/5e49825c539aec77cc5af4c518ce07f3a403eb36aa7f0aa3128743c23a415743.jpg)

![5fe6b788e45b8f72709dd7f7bf684aac931a29a4665ab741e47ee07c2874a505.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/5fe6b788e45b8f72709dd7f7bf684aac931a29a4665ab741e47ee07c2874a505.jpg)

![693bb3c157940e03615748be9a2d741f449350bdef3ac4d2f955ace53db921a8.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/693bb3c157940e03615748be9a2d741f449350bdef3ac4d2f955ace53db921a8.jpg)

![6ff5381e9c8d3e98b979c1e58ab2fc988ea297ba81c71077be719f189b409f58.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/6ff5381e9c8d3e98b979c1e58ab2fc988ea297ba81c71077be719f189b409f58.jpg)

![80d6d7a2c8b9911b8d1f55c5b5b6f44818d6ebbf753325695f2bc8ec4ea0fc56.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/80d6d7a2c8b9911b8d1f55c5b5b6f44818d6ebbf753325695f2bc8ec4ea0fc56.jpg)

![97d8bad8ce7fafbc51bb1074c092d14b08898fa6eff154a74fa1379685ff7fa6.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/97d8bad8ce7fafbc51bb1074c092d14b08898fa6eff154a74fa1379685ff7fa6.jpg)

![b28eac8597bbd1f9485624725fee90c3920763cae15681d36e9f96780bcd2599.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/b28eac8597bbd1f9485624725fee90c3920763cae15681d36e9f96780bcd2599.jpg)

![d832301afbf628bb978efbde2f7d8dfbf59e84cd14ab7989ce939678e0a17af0.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/d832301afbf628bb978efbde2f7d8dfbf59e84cd14ab7989ce939678e0a17af0.jpg)

![e2510494e4608f57ed62b1aec0f0b46db5a4b3e04de1427b8b08fa5985daff38.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/e2510494e4608f57ed62b1aec0f0b46db5a4b3e04de1427b8b08fa5985daff38.jpg)

![f80e16f5b53958b948a76b700959cbd8442374c2b88780b151785a721b19613a.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/images/f80e16f5b53958b948a76b700959cbd8442374c2b88780b151785a721b19613a.jpg)

### Tables

![3c89cbbb1b5016306093ca0bbb8ea937b2d8e7c6bfeb0d7177e253912bfa90a5.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/3c89cbbb1b5016306093ca0bbb8ea937b2d8e7c6bfeb0d7177e253912bfa90a5.jpg)

![445c8f4c29a4f08851bc902922c995379d86397aa3f574d890cd68a114b55438.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/445c8f4c29a4f08851bc902922c995379d86397aa3f574d890cd68a114b55438.jpg)

![841242aaed40c401c99545246228ff4e899ccb642c3acd477ca3318ad14dc1b2.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/841242aaed40c401c99545246228ff4e899ccb642c3acd477ca3318ad14dc1b2.jpg)

![8de2beba3dc1027d71ed3514e7e6af36013ff450c0af2e046cc48fcaf651c1ad.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/8de2beba3dc1027d71ed3514e7e6af36013ff450c0af2e046cc48fcaf651c1ad.jpg)

![8e705d8d0706bb0ca22fac09616ffc5d1da6c460c2b234f880aa4ac1bd3340df.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/8e705d8d0706bb0ca22fac09616ffc5d1da6c460c2b234f880aa4ac1bd3340df.jpg)

![931c080599c8a65e05c1884150168335888688e66e0bb79c7a87eee9a115e8f6.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/931c080599c8a65e05c1884150168335888688e66e0bb79c7a87eee9a115e8f6.jpg)

![a5cdd77f65bd885128b2c781de76da419ec419c04dbdac0955e6ef99b9508319.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/a5cdd77f65bd885128b2c781de76da419ec419c04dbdac0955e6ef99b9508319.jpg)

![ab4768af929a516b645940f7b6e5d4fce1075e3c67232ff3728ebf3542290ff3.jpg](../icml_results/371_Trojan-Speak_%20Bypassing%20Constitutional%20Classifiers%20with%20No%20Jailbreak%20Tax%20via%20Adversarial%20Finetuning/tables/ab4768af929a516b645940f7b6e5d4fce1075e3c67232ff3728ebf3542290ff3.jpg)

<a id="paper-0365"></a>

## dnaHNet: A Scalable and Hierarchical Foundation Model for Genomic Sequence Learning


### Images

![069f87b32157a88436e19121d2c71a3d06021e9f08193e47a781f71a238379a9.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/069f87b32157a88436e19121d2c71a3d06021e9f08193e47a781f71a238379a9.jpg)

![24950219c6289ee3f3e571c78010833013e6a8d62a611434161f044d0206461a.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/24950219c6289ee3f3e571c78010833013e6a8d62a611434161f044d0206461a.jpg)

![44ebc882d1d9c76b44fb8ae92c207d12c235780cad67744e40751dada840c76f.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/44ebc882d1d9c76b44fb8ae92c207d12c235780cad67744e40751dada840c76f.jpg)

![6bbfce5d1451d1a97c5f97a110d4d0198d9ff68ba6446b486eba9ae020e1bbe0.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/6bbfce5d1451d1a97c5f97a110d4d0198d9ff68ba6446b486eba9ae020e1bbe0.jpg)

![841ff09141a3f49bb16edcb3c8d0236c4de658af42df01289e8b8080daaca044.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/841ff09141a3f49bb16edcb3c8d0236c4de658af42df01289e8b8080daaca044.jpg)

![88ec309a85f2d5d125e5508dddd55e1f02eaa56465dcfca0745d47420afe2cb4.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/88ec309a85f2d5d125e5508dddd55e1f02eaa56465dcfca0745d47420afe2cb4.jpg)

![bba95442f0e2d88a1497986e2d16a0ed65c876cb63342873d216957f172c361b.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/bba95442f0e2d88a1497986e2d16a0ed65c876cb63342873d216957f172c361b.jpg)

![f2d47abc687719822f4ebe2c134caed92ae8471b839d3a2070954817dea373d1.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/images/f2d47abc687719822f4ebe2c134caed92ae8471b839d3a2070954817dea373d1.jpg)

### Tables

![2ca2dbb5088256b3f96521f58de05ed6bab0855d62c63f07d2a2625823c75344.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/2ca2dbb5088256b3f96521f58de05ed6bab0855d62c63f07d2a2625823c75344.jpg)

![4e50f741a5ddec504474449c8a5d1575554e6375c189ae2f789e365d66b4d990.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/4e50f741a5ddec504474449c8a5d1575554e6375c189ae2f789e365d66b4d990.jpg)

![776cc6391329c8f17f7de9918edf8717c45590da325ff4220db311a88fedc05a.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/776cc6391329c8f17f7de9918edf8717c45590da325ff4220db311a88fedc05a.jpg)

![79975971d8905a7a603033e276c7b60f2963f9f8592a9fc121b83c81e40be29b.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/79975971d8905a7a603033e276c7b60f2963f9f8592a9fc121b83c81e40be29b.jpg)

![7b99bd9ac40da577aa66408edcd85cfd5fb7bbd37c266ee473364479daa0b234.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/7b99bd9ac40da577aa66408edcd85cfd5fb7bbd37c266ee473364479daa0b234.jpg)

![b155ddb38bf51281537ac0d0400c321c168f15a523e238d83000be3a1124f507.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/b155ddb38bf51281537ac0d0400c321c168f15a523e238d83000be3a1124f507.jpg)

![d0fbb6a08a5c535e909c77ecdf6c71bffba382aaa52227551d62e4d86a481ec6.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/d0fbb6a08a5c535e909c77ecdf6c71bffba382aaa52227551d62e4d86a481ec6.jpg)

![f80d9f8bfa5aa8fabe561cec03b895fbf46444127fa01fce9cc72c16fc331e19.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/f80d9f8bfa5aa8fabe561cec03b895fbf46444127fa01fce9cc72c16fc331e19.jpg)

![fd58674ae848b40ef13d7f4062e8e0ca2d1a0a72eb0e92ce7dc1f50358383c3f.jpg](../icml_results/372_dnaHNet_%20A%20Scalable%20and%20Hierarchical%20Foundation%20Model%20for%20Genomic%20Sequence%20Learning/tables/fd58674ae848b40ef13d7f4062e8e0ca2d1a0a72eb0e92ce7dc1f50358383c3f.jpg)

<a id="paper-0366"></a>

## Even Faster Kernel Matrix Linear Algebra via Density Estimation


### Images

![1727847562a46f71da565371ab46df5885c8b89a42ccf8f68c33a9553b7e32fe.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/1727847562a46f71da565371ab46df5885c8b89a42ccf8f68c33a9553b7e32fe.jpg)

![173aa2f4d7743cd1808e9f6a25d6f14e8529075e04d3207e2391b98acc7bbdbd.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/173aa2f4d7743cd1808e9f6a25d6f14e8529075e04d3207e2391b98acc7bbdbd.jpg)

![33b8af4cd069664f087df1f461dc2673f4c75c0bb6752cb1394c307279cecb96.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/33b8af4cd069664f087df1f461dc2673f4c75c0bb6752cb1394c307279cecb96.jpg)

![687f81bb4377a83694881281e26956fa0ee8ad9f02bbc76e17070687d39681d1.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/687f81bb4377a83694881281e26956fa0ee8ad9f02bbc76e17070687d39681d1.jpg)

![6fdb607808342bb20586c48cc69770eb5b304bf5aef57eb559aa907b7b88fa82.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/6fdb607808342bb20586c48cc69770eb5b304bf5aef57eb559aa907b7b88fa82.jpg)

![877a6003f495c2cb3b6315ddeab9d3051f838c3f361cb81f1122f43c6ab6e702.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/877a6003f495c2cb3b6315ddeab9d3051f838c3f361cb81f1122f43c6ab6e702.jpg)

![b36317ee79a91a2e2adff11193f61bda79880c5a5ad6e36266c69c93ea92d69e.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/b36317ee79a91a2e2adff11193f61bda79880c5a5ad6e36266c69c93ea92d69e.jpg)

![b54143f12ddfb440ab06de937eba8b0ad70cd1118b4f9462c29990673c906c14.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/b54143f12ddfb440ab06de937eba8b0ad70cd1118b4f9462c29990673c906c14.jpg)

![e92a27f1faa13b895a9e7ecd88e07a41ce70e6e13a31d1d9d714698ac44b07e0.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/e92a27f1faa13b895a9e7ecd88e07a41ce70e6e13a31d1d9d714698ac44b07e0.jpg)

![f3d8c5bb75f2b34def75f7c0791a02030f5e335af9914616a6325b04ae32f0ed.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/images/f3d8c5bb75f2b34def75f7c0791a02030f5e335af9914616a6325b04ae32f0ed.jpg)

### Tables

![46b67a89611cd3abb13f5ec0808d5647680daf8a29c04108307c1da77921f5f1.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/tables/46b67a89611cd3abb13f5ec0808d5647680daf8a29c04108307c1da77921f5f1.jpg)

![963a354f89b0be6dfc1722cefdfb3753554a724a80a4f45b64e333985f309a1c.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/tables/963a354f89b0be6dfc1722cefdfb3753554a724a80a4f45b64e333985f309a1c.jpg)

![b5bd254e2f03e28824a815aabd9067826f26bf3db68e77d7c3da71b667f7ed0e.jpg](../icml_results/373_Even%20Faster%20Kernel%20Matrix%20Linear%20Algebra%20via%20Density%20Estimation/tables/b5bd254e2f03e28824a815aabd9067826f26bf3db68e77d7c3da71b667f7ed0e.jpg)

<a id="paper-0367"></a>

## Modeling Hierarchical Thinking in Large Reasoning Models


### Images

![025705a93c558e9805a8e92c76309eb17ed15a9f355e854b6b91956a26414b51.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/images/025705a93c558e9805a8e92c76309eb17ed15a9f355e854b6b91956a26414b51.jpg)

![4e41fcffe9e08b16ba818d9ff9813b033bab34e50bebdea449041a3a0a690183.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/images/4e41fcffe9e08b16ba818d9ff9813b033bab34e50bebdea449041a3a0a690183.jpg)

![a22f44dfddb4e8ccdfe877c2df78adba2b9765119ba5a7eb16bdec8843df2686.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/images/a22f44dfddb4e8ccdfe877c2df78adba2b9765119ba5a7eb16bdec8843df2686.jpg)

![e53b5ca0f60b1c4dd0ce40e14a93155e918c1cdf5084667b6cfe887b5b9ef741.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/images/e53b5ca0f60b1c4dd0ce40e14a93155e918c1cdf5084667b6cfe887b5b9ef741.jpg)

### Tables

![611d7bf2e827c6d71c94dee8d5835ea3bd1276cc517892a86f55b4c82c3d9bb2.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/611d7bf2e827c6d71c94dee8d5835ea3bd1276cc517892a86f55b4c82c3d9bb2.jpg)

![6f62b0cef452c196343bcc3b3dbc05a9d7e55f41cbbdab1111e3ceba2702479f.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/6f62b0cef452c196343bcc3b3dbc05a9d7e55f41cbbdab1111e3ceba2702479f.jpg)

![8057f4916637675bbb23828927f0001c5c0cc25ec8204d97455decc796d4474f.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/8057f4916637675bbb23828927f0001c5c0cc25ec8204d97455decc796d4474f.jpg)

![a4b73dbb64ab4f109ecacc470fb8d1d86018eb7abccf4711f86687827f87e6e6.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/a4b73dbb64ab4f109ecacc470fb8d1d86018eb7abccf4711f86687827f87e6e6.jpg)

![b1947e7833aff7cefe9f3b501c3db7f7c4d235dbac524c4e06012f15fce23df5.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/b1947e7833aff7cefe9f3b501c3db7f7c4d235dbac524c4e06012f15fce23df5.jpg)

![d4e67e3dc3009ad99881afa91c961a4ed85bf48a070657563487001365d3d5dc.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/d4e67e3dc3009ad99881afa91c961a4ed85bf48a070657563487001365d3d5dc.jpg)

![dae124aae21a37668e0f926f4b776b69c9edb0077dee2b41de0069b2eb78a96b.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/dae124aae21a37668e0f926f4b776b69c9edb0077dee2b41de0069b2eb78a96b.jpg)

![e0359c46044d156287adfe429edbc0f67f68595511a1e2a7bffb7de235876441.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/e0359c46044d156287adfe429edbc0f67f68595511a1e2a7bffb7de235876441.jpg)

![e57cfb5566306830744ec5170f4094deac6838550f2ab89972cad516c8ffb5a7.jpg](../icml_results/374_Modeling%20Hierarchical%20Thinking%20in%20Large%20Reasoning%20Models/tables/e57cfb5566306830744ec5170f4094deac6838550f2ab89972cad516c8ffb5a7.jpg)

<a id="paper-0368"></a>

## DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research


### Images

![03c67171e0cb20bc90247f98c07f0f0d6594045af900711cc9ea63233b86c963.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/03c67171e0cb20bc90247f98c07f0f0d6594045af900711cc9ea63233b86c963.jpg)

![0b8156b9df72f3affacda66cd679314ced26bb8f554e0dfce9c8e6a169716d33.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/0b8156b9df72f3affacda66cd679314ced26bb8f554e0dfce9c8e6a169716d33.jpg)

![125cc40cdfa44258b17b5be4948498aab55ceca8d6f74b44f9853e3d454b5a67.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/125cc40cdfa44258b17b5be4948498aab55ceca8d6f74b44f9853e3d454b5a67.jpg)

![22b5458b69d983c1ab3be2e36421506343f9aad92d516e955d9718db49ed9522.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/22b5458b69d983c1ab3be2e36421506343f9aad92d516e955d9718db49ed9522.jpg)

![31d3119190a83daedd8652f8844fc11e21583c8cdf2780c47b451dbed8d04ec8.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/31d3119190a83daedd8652f8844fc11e21583c8cdf2780c47b451dbed8d04ec8.jpg)

![32ebdee97e20369210304e76a85defd75a26b00de6ac99b2bf571e969809e432.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/32ebdee97e20369210304e76a85defd75a26b00de6ac99b2bf571e969809e432.jpg)

![63f60e9f8e5a95bea9fcffd39e06f826d910d479c0676038b7faadf544a7885f.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/63f60e9f8e5a95bea9fcffd39e06f826d910d479c0676038b7faadf544a7885f.jpg)

![68a27e7eb09501f497e06104a3ed77e4179bb2986a9051638059a4c66abeb414.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/68a27e7eb09501f497e06104a3ed77e4179bb2986a9051638059a4c66abeb414.jpg)

![76fb14d102111b1e162bbb76f4fdc89366ba8364b2580c1a7e96031e415efc56.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/76fb14d102111b1e162bbb76f4fdc89366ba8364b2580c1a7e96031e415efc56.jpg)

![a5415560f857f46d0d3813b296b5963588514f9d9c4e9367d436c01901e2f181.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/a5415560f857f46d0d3813b296b5963588514f9d9c4e9367d436c01901e2f181.jpg)

![be8367db7b2d1b0d89098c3628644c2347836b9b825dd8507446dc9d3caec32d.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/be8367db7b2d1b0d89098c3628644c2347836b9b825dd8507446dc9d3caec32d.jpg)

![db9b0c2940121ee7530cd4275938c66f4cba674b87cb12ff27d5ec28346db35b.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/db9b0c2940121ee7530cd4275938c66f4cba674b87cb12ff27d5ec28346db35b.jpg)

![dc5e4e44355303d868a05965b61f0a6771922799563545c873a46970d48b4361.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/dc5e4e44355303d868a05965b61f0a6771922799563545c873a46970d48b4361.jpg)

![e7b3bcd86937aa71653cbfc0e4e89fa9a3ae7f9e67d371139527995da8f8e95e.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/e7b3bcd86937aa71653cbfc0e4e89fa9a3ae7f9e67d371139527995da8f8e95e.jpg)

![e8368b70c54a8e0e63da10d2146ea1a32c676a798ff8667156adc8a65a822d96.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/e8368b70c54a8e0e63da10d2146ea1a32c676a798ff8667156adc8a65a822d96.jpg)

![f234fbf544d02d8553da07628c94a512740ab3616eeb277768c751fe428094f0.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/images/f234fbf544d02d8553da07628c94a512740ab3616eeb277768c751fe428094f0.jpg)

### Tables

![0bbdd54490f692c6b1248fab01b2b08f9b2a5e656115ac9ed1ddf6018c36c62a.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/0bbdd54490f692c6b1248fab01b2b08f9b2a5e656115ac9ed1ddf6018c36c62a.jpg)

![0fd9c64b1e869be9caadf4baab9ece60b57173659d768bdbed386ed520988fa0.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/0fd9c64b1e869be9caadf4baab9ece60b57173659d768bdbed386ed520988fa0.jpg)

![1efe29ebf5faa1168eb53d8231bfd67fc0c770e4e34d986a4cdc998474fdd92b.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/1efe29ebf5faa1168eb53d8231bfd67fc0c770e4e34d986a4cdc998474fdd92b.jpg)

![1fcd32b6eb2b58d9cdc78e5d7a8b860469d051044e9d722e9478cf715b4922dd.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/1fcd32b6eb2b58d9cdc78e5d7a8b860469d051044e9d722e9478cf715b4922dd.jpg)

![20acb846d054c633f10d674271a0370c474a00c737de63088f7f564bdfe0019c.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/20acb846d054c633f10d674271a0370c474a00c737de63088f7f564bdfe0019c.jpg)

![315f142f735f015c5677b74b5444fa4bf3103867282544eb495961e3060c476e.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/315f142f735f015c5677b74b5444fa4bf3103867282544eb495961e3060c476e.jpg)

![3f6898a91a609a8881e81e2df3bca986c123c15e2bc210dd467b6f5196125641.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/3f6898a91a609a8881e81e2df3bca986c123c15e2bc210dd467b6f5196125641.jpg)

![639a0a6700ad9432774a94ad739187949ce4b831e608efdc5e9d0208cdb2422a.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/639a0a6700ad9432774a94ad739187949ce4b831e608efdc5e9d0208cdb2422a.jpg)

![70a3fd21dd7f2d70231007dd373ea1f69fddd18f7a0d51a33f7abacdfc7dd805.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/70a3fd21dd7f2d70231007dd373ea1f69fddd18f7a0d51a33f7abacdfc7dd805.jpg)

![73457671f93016c3267bfb87ea1fe2c785ab627fb7fbedccd876c2535ebd537f.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/73457671f93016c3267bfb87ea1fe2c785ab627fb7fbedccd876c2535ebd537f.jpg)

![74cc2f2ce3b15bba2f6b500829f56cb65bbeaf51eb05e56a23be0b3e9d4cd42b.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/74cc2f2ce3b15bba2f6b500829f56cb65bbeaf51eb05e56a23be0b3e9d4cd42b.jpg)

![84229e6ac6799cd8e26b395916d57951c6a653ee5c4be3654917249e4faa0f7d.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/84229e6ac6799cd8e26b395916d57951c6a653ee5c4be3654917249e4faa0f7d.jpg)

![9983f4ebda8c83e274656764d4d3f6fa34adc1ca35d11d9c284f0105a6a6da6a.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/9983f4ebda8c83e274656764d4d3f6fa34adc1ca35d11d9c284f0105a6a6da6a.jpg)

![c6b83e9ec2fee537c326c9958fcbf800d6d3a4ce75c41911cb2c5f6cc499bd13.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/c6b83e9ec2fee537c326c9958fcbf800d6d3a4ce75c41911cb2c5f6cc499bd13.jpg)

![cb29b5a6fe41f3a6f19c8e0c5a5cd45aa777053a219621ed0e9a01ad3a2dc7b5.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/cb29b5a6fe41f3a6f19c8e0c5a5cd45aa777053a219621ed0e9a01ad3a2dc7b5.jpg)

![e110764d32131cfe220506d5ca343171dbeea8137ed8c7679126f21f5bf9b502.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/e110764d32131cfe220506d5ca343171dbeea8137ed8c7679126f21f5bf9b502.jpg)

![ebda6d15641ff81ecf203cfcb0d319cb59ffea0f4724f4aa50321efc15b2d7d7.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/ebda6d15641ff81ecf203cfcb0d319cb59ffea0f4724f4aa50321efc15b2d7d7.jpg)

![f8b3d0f11015e672515e1f1aa4f05afc6d97d9f256bf07bf7a7bf833cb18b8bd.jpg](../icml_results/375_DR%20Tulu_%20Reinforcement%20Learning%20with%20Evolving%20Rubrics%20for%20Deep%20Research/tables/f8b3d0f11015e672515e1f1aa4f05afc6d97d9f256bf07bf7a7bf833cb18b8bd.jpg)

<a id="paper-0369"></a>

## DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions


### Images

![01c8666af5270d7977f0598757a4b98709226d62dc66e34342ae909b2db7ecf0.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/01c8666af5270d7977f0598757a4b98709226d62dc66e34342ae909b2db7ecf0.jpg)

![0bd844a6edb72ef5825b9dbe315e22506abef5530935e027c31d2fe2ac47162a.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/0bd844a6edb72ef5825b9dbe315e22506abef5530935e027c31d2fe2ac47162a.jpg)

![1729623172ca6736fde1a84b22371a083304244021ee6df104d9a4e084b2b652.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/1729623172ca6736fde1a84b22371a083304244021ee6df104d9a4e084b2b652.jpg)

![22dacfa28ede0308a0f40f93624868eed0f5df3dd50f02b509cea1d49071d1e4.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/22dacfa28ede0308a0f40f93624868eed0f5df3dd50f02b509cea1d49071d1e4.jpg)

![30c9a4d822526fcbd5a7d71da561a2b19db613a645eab5d06cf65139c182331e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/30c9a4d822526fcbd5a7d71da561a2b19db613a645eab5d06cf65139c182331e.jpg)

![4996ebd99eab1dfc63aad57ac894d6fd41b6046b0e2dde4b5f23a14047fd094c.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/4996ebd99eab1dfc63aad57ac894d6fd41b6046b0e2dde4b5f23a14047fd094c.jpg)

![7ce0f3e7346bb7e5fb929ec510e71ca985139b97eb7ed63d494f82bb75cd3e4f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/7ce0f3e7346bb7e5fb929ec510e71ca985139b97eb7ed63d494f82bb75cd3e4f.jpg)

![7e1db691dd3fa05aa6bf457dc71a37445d389d8c29a3809ee22fce5b1e844878.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/7e1db691dd3fa05aa6bf457dc71a37445d389d8c29a3809ee22fce5b1e844878.jpg)

![8346f6267368f47c4ba75654440ae9500e4afd27159465314f8dc0cd41f147f6.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/8346f6267368f47c4ba75654440ae9500e4afd27159465314f8dc0cd41f147f6.jpg)

![9334729434ea129deb77da069153e15fd592ced1fd1624b4a26db5df1a836707.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/9334729434ea129deb77da069153e15fd592ced1fd1624b4a26db5df1a836707.jpg)

![c06247bb3e1728b30a4718c56c4aa731f94c36b1c03cd1a3bd54a7193153a451.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/c06247bb3e1728b30a4718c56c4aa731f94c36b1c03cd1a3bd54a7193153a451.jpg)

![d45ba8aacd8d8064dd59c042f04116fbfb2591493566ea0097386a6208d52e92.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/d45ba8aacd8d8064dd59c042f04116fbfb2591493566ea0097386a6208d52e92.jpg)

![ea7675a9af5292f9997371b36ead051aad50f4964f13b95918a182a0374630b1.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/ea7675a9af5292f9997371b36ead051aad50f4964f13b95918a182a0374630b1.jpg)

![fd91ef657d58cb03d151df746edeeae9d02d6a3b6ca144c9b9e89b8204d54b40.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/fd91ef657d58cb03d151df746edeeae9d02d6a3b6ca144c9b9e89b8204d54b40.jpg)

![fe1cdf3eef959eb23c6290adf36f83ce273b55ecb7d8ab2bf7f00954c723a726.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/images/fe1cdf3eef959eb23c6290adf36f83ce273b55ecb7d8ab2bf7f00954c723a726.jpg)

### Tables

![06548edbc147358ab8eb40166a567fa09908776014893e4e9d24a70c86acdf3e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/06548edbc147358ab8eb40166a567fa09908776014893e4e9d24a70c86acdf3e.jpg)

![187cd9360137bb81fcd0f5536433f37b30bec7592d8ff326dc9031785f52238e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/187cd9360137bb81fcd0f5536433f37b30bec7592d8ff326dc9031785f52238e.jpg)

![1f0cabcc97bc6431dee2a21a5bec9d5dd0b36d1232c3a8b3ac85e2161f2d0656.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/1f0cabcc97bc6431dee2a21a5bec9d5dd0b36d1232c3a8b3ac85e2161f2d0656.jpg)

![2615532d5974e25800db9d08db6e038949f798705e613fb506654d312203aa29.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/2615532d5974e25800db9d08db6e038949f798705e613fb506654d312203aa29.jpg)

![38edcd22438a288ea370386c3055b1fb5dd65203dce76a5c8afe2f36517e7435.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/38edcd22438a288ea370386c3055b1fb5dd65203dce76a5c8afe2f36517e7435.jpg)

![3a1546ff06afb0e750e4232d422f5844137dc91ab9a3f0d8cc6c249a59da9404.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/3a1546ff06afb0e750e4232d422f5844137dc91ab9a3f0d8cc6c249a59da9404.jpg)

![3d03bd24ee2fca6e1b2100fd7a1d81ac3187fd17e06970405c6a724d1b91197e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/3d03bd24ee2fca6e1b2100fd7a1d81ac3187fd17e06970405c6a724d1b91197e.jpg)

![3d38055535dd7d81808944a9b9fa853e0ab71be5e2fcd7e0bc06dbc60400e690.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/3d38055535dd7d81808944a9b9fa853e0ab71be5e2fcd7e0bc06dbc60400e690.jpg)

![476b5dc60ab97a78c381b154818a68fd5fcc40a4459e770fa52f1086aa6b0a3e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/476b5dc60ab97a78c381b154818a68fd5fcc40a4459e770fa52f1086aa6b0a3e.jpg)

![4d0986ab2860f629443501e9c6f0ac425b949571af90753f790f1866f34924dc.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/4d0986ab2860f629443501e9c6f0ac425b949571af90753f790f1866f34924dc.jpg)

![55a2eb82674fe080451b2c0c6c5ea67c47b06f3aac8e11b36a941d1d6093ff84.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/55a2eb82674fe080451b2c0c6c5ea67c47b06f3aac8e11b36a941d1d6093ff84.jpg)

![57583f781eb91a4a9d9d8dca5780c6458609b2c2032089d3897162c200b7aa28.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/57583f781eb91a4a9d9d8dca5780c6458609b2c2032089d3897162c200b7aa28.jpg)

![58174de89214aa08a5e5af7c425d2cd76d0ee7be99693a3e7cedbcc67df23762.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/58174de89214aa08a5e5af7c425d2cd76d0ee7be99693a3e7cedbcc67df23762.jpg)

![58f50deedbabc06f9b02ae741b11d73454a43f15444f98e42a5f1c4087ba9a77.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/58f50deedbabc06f9b02ae741b11d73454a43f15444f98e42a5f1c4087ba9a77.jpg)

![59fee2953bf497bb7b791fbcc0624b42eef80f5e473264eaf170f56713ff968b.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/59fee2953bf497bb7b791fbcc0624b42eef80f5e473264eaf170f56713ff968b.jpg)

![5a1ddb741e76d4e6f7ff63fc79bc6ba9daa5e0f3e9669177abebe7250090aaf0.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/5a1ddb741e76d4e6f7ff63fc79bc6ba9daa5e0f3e9669177abebe7250090aaf0.jpg)

![702680844534a133e7836fe6f6926a922f94f1e4ca793f82774a44d3dc41bfd1.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/702680844534a133e7836fe6f6926a922f94f1e4ca793f82774a44d3dc41bfd1.jpg)

![758753a8a19a9751b85ff74fe1573ac1da71a65e8f870b27f7c38f27b2f15104.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/758753a8a19a9751b85ff74fe1573ac1da71a65e8f870b27f7c38f27b2f15104.jpg)

![79f846926ec604a973bc58010ea9f0337e9deec6c11f4be482894ad1052c2df2.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/79f846926ec604a973bc58010ea9f0337e9deec6c11f4be482894ad1052c2df2.jpg)

![8dd50d1140322be50b891c9720455ec4ecd69fed4010d02110c808d4d4632e81.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/8dd50d1140322be50b891c9720455ec4ecd69fed4010d02110c808d4d4632e81.jpg)

![987db6c56093e0624e1ed071efe251c2b6e60c228a29ab46c7cb833e368f0587.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/987db6c56093e0624e1ed071efe251c2b6e60c228a29ab46c7cb833e368f0587.jpg)

![9d3172c0b4b66510493e891833b1d3002a18c78b757f150704f69a61672a44d4.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/9d3172c0b4b66510493e891833b1d3002a18c78b757f150704f69a61672a44d4.jpg)

![a75458cc63bf137e700cb7508d8498d26c4f47da3f075d0708b58783801d565d.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/a75458cc63bf137e700cb7508d8498d26c4f47da3f075d0708b58783801d565d.jpg)

![a919ea87f2103fb8683449c60f93eacb8f0f510bee68bdf7be22bcc05090813b.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/a919ea87f2103fb8683449c60f93eacb8f0f510bee68bdf7be22bcc05090813b.jpg)

![b0ab9cb1a8cea1c648a68744dc2915df61b58fd49a55be68f94e54bc4f78248d.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/b0ab9cb1a8cea1c648a68744dc2915df61b58fd49a55be68f94e54bc4f78248d.jpg)

![b6bdf69b6a33cef7b98e8b9b380d931ee5c13380fe973552c1ff348a0a1708ad.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/b6bdf69b6a33cef7b98e8b9b380d931ee5c13380fe973552c1ff348a0a1708ad.jpg)

![c1cff0a4826e4a4111c9ad89c17e6599ce6c7b43a299aa4c5866895cf0140a74.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/c1cff0a4826e4a4111c9ad89c17e6599ce6c7b43a299aa4c5866895cf0140a74.jpg)

![c20519f9aee73987b861b7fc5d6e05c1deee4e73e0b018d781907ffef296a36f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/c20519f9aee73987b861b7fc5d6e05c1deee4e73e0b018d781907ffef296a36f.jpg)

![cbbe29fd2da6bcc9acc82de38e49cb2a21b26c343de811999eb652353a25fc3f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/cbbe29fd2da6bcc9acc82de38e49cb2a21b26c343de811999eb652353a25fc3f.jpg)

![cd74150a8faf7880915770f256d732f3f9e6ad6efeec5b37e22f73de03433f30.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/cd74150a8faf7880915770f256d732f3f9e6ad6efeec5b37e22f73de03433f30.jpg)

![ce7f3b1ce21ca4b9c2cfdaaf9a74bd6f513c719789559306b1ce282131757665.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/ce7f3b1ce21ca4b9c2cfdaaf9a74bd6f513c719789559306b1ce282131757665.jpg)

![cf69f01c8b051c054d785d5a006cc7d041831210c08c383b822bd71f82c488ab.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/cf69f01c8b051c054d785d5a006cc7d041831210c08c383b822bd71f82c488ab.jpg)

![d1d1cac8f96bfe67de273024620ff6ac03d7e4b3ab3a50de8ccde989abba8d5e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/d1d1cac8f96bfe67de273024620ff6ac03d7e4b3ab3a50de8ccde989abba8d5e.jpg)

![d2b13da0d84f20cd43ba44bfa8757dea6049bce06c82c83b1e7185434d151753.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/d2b13da0d84f20cd43ba44bfa8757dea6049bce06c82c83b1e7185434d151753.jpg)

![d78a898b9095b3e03b60dc266f3a0073d6f6e5024e8d9ec53b0ff51cd702fd79.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/d78a898b9095b3e03b60dc266f3a0073d6f6e5024e8d9ec53b0ff51cd702fd79.jpg)

![d9729463092a32a175cf7c558ca8b3516307c2a58077d173879cfb10c24a002f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/d9729463092a32a175cf7c558ca8b3516307c2a58077d173879cfb10c24a002f.jpg)

![db6c6911a37a162bd07594e9014f9388beb76ee0073c763a06d74847af9d801f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/db6c6911a37a162bd07594e9014f9388beb76ee0073c763a06d74847af9d801f.jpg)

![e01712a5683803e19e74dd362e0db7aa03a7082df915697a83995cc549c6e92e.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/e01712a5683803e19e74dd362e0db7aa03a7082df915697a83995cc549c6e92e.jpg)

![e30bdd4e900e4ca6780a4cc3fb2e703c2b180ad0899f35db49903adf0a03d940.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/e30bdd4e900e4ca6780a4cc3fb2e703c2b180ad0899f35db49903adf0a03d940.jpg)

![e7629fbf95e39e9b10c4f4ac8c3e0c467c07c04d2a025962f7e2c1a023750ddb.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/e7629fbf95e39e9b10c4f4ac8c3e0c467c07c04d2a025962f7e2c1a023750ddb.jpg)

![efe95df746b6da7fa6c1040487cdaeb5cef7595c806296ec5e172e8d739bd0fa.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/efe95df746b6da7fa6c1040487cdaeb5cef7595c806296ec5e172e8d739bd0fa.jpg)

![fea08e9d50284a107c6996fe2ad097fb3ecd95693bec7f7a856e5fbae236eb7f.jpg](../icml_results/376_DecodeShare_%20Tracing%20the%20Shared%20Subspace%20of%20LLM%20Decode-Time%20Decisions/tables/fea08e9d50284a107c6996fe2ad097fb3ecd95693bec7f7a856e5fbae236eb7f.jpg)

<a id="paper-0370"></a>

## MIRA: A Score for Conditional Distribution Accuracy and Model Comparison


### Images

![103f906b86d0c529a068f7f0d399dbf5081c194911002f9077dc8b91025c5abc.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/103f906b86d0c529a068f7f0d399dbf5081c194911002f9077dc8b91025c5abc.jpg)

![119bc8dd58ef21cd3867bceb64992d175c7e49d0d356a3367cb807ef43123179.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/119bc8dd58ef21cd3867bceb64992d175c7e49d0d356a3367cb807ef43123179.jpg)

![20fce300a27f42a99562a736d783d83ee473f27248dca0d7e6b8668d68032360.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/20fce300a27f42a99562a736d783d83ee473f27248dca0d7e6b8668d68032360.jpg)

![2ffff3f11d1e3c91ba56f15106337e698e220a4ffeaf00efda9e82490abab510.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/2ffff3f11d1e3c91ba56f15106337e698e220a4ffeaf00efda9e82490abab510.jpg)

![3364b2be2280b4bef0837c3d9041b7a1f825b03dca68edaa5c02f92c37aba01f.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/3364b2be2280b4bef0837c3d9041b7a1f825b03dca68edaa5c02f92c37aba01f.jpg)

![60bdfa0b6a6f063d8ae904a1a64cfc495c085c98a3e4032cfd029079f87a15c2.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/60bdfa0b6a6f063d8ae904a1a64cfc495c085c98a3e4032cfd029079f87a15c2.jpg)

![76e60362dfde9764d1ba2f391772c79378a7cf62c1285747d07f5cefea444d41.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/76e60362dfde9764d1ba2f391772c79378a7cf62c1285747d07f5cefea444d41.jpg)

![877e41cc96816c961f6ba551739f5e4e58fb2b9423a2f17d7f86c04e3f526ca0.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/877e41cc96816c961f6ba551739f5e4e58fb2b9423a2f17d7f86c04e3f526ca0.jpg)

![9c4521d87703473e56700e837d1f5f3b6bac8fe961b5c811d86e734ff74ab2fe.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/9c4521d87703473e56700e837d1f5f3b6bac8fe961b5c811d86e734ff74ab2fe.jpg)

![9d4e72a99c08deb232ca98c316155c738be6e467673b3a651503c4f18e7aceb5.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/9d4e72a99c08deb232ca98c316155c738be6e467673b3a651503c4f18e7aceb5.jpg)

![a09acde6a6fe147f77e494df48a1778b73185386382e2b1b9ca20310aeb587d0.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/a09acde6a6fe147f77e494df48a1778b73185386382e2b1b9ca20310aeb587d0.jpg)

![a13524ba4ece7cf4f896878be03674b2fb892cb530974e4b3fcdf2c48c9a5e3a.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/a13524ba4ece7cf4f896878be03674b2fb892cb530974e4b3fcdf2c48c9a5e3a.jpg)

![b13f0f5ee768de91014bbdbcbe415c8b59bc3cce35cb125567059737d611e103.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/b13f0f5ee768de91014bbdbcbe415c8b59bc3cce35cb125567059737d611e103.jpg)

![de7a868fb5b580c50c91c35502d500cfab10fba62fdfc91a6616f3160f5505f4.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/de7a868fb5b580c50c91c35502d500cfab10fba62fdfc91a6616f3160f5505f4.jpg)

![e044be9fa3a94db3733cebf23a82ff987048be7f9edbbb194d0870a205b84c92.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/e044be9fa3a94db3733cebf23a82ff987048be7f9edbbb194d0870a205b84c92.jpg)

![ea8a3ac6b5ba4e89f1693bad43e895aecfaa21b133a1c1560f96538176a52dfb.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/images/ea8a3ac6b5ba4e89f1693bad43e895aecfaa21b133a1c1560f96538176a52dfb.jpg)

### Tables

![0ff9a7cce6b61efbee9fc5c0c2f8dbd0fb6bda1687fedac6d8523018e9e847da.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/0ff9a7cce6b61efbee9fc5c0c2f8dbd0fb6bda1687fedac6d8523018e9e847da.jpg)

![134a2786da0fe28e899c3e40d87f035e99d28607d562d5c1f95d92491dfad3d9.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/134a2786da0fe28e899c3e40d87f035e99d28607d562d5c1f95d92491dfad3d9.jpg)

![3e1876b24bcd9da771798d639c2e6e012a1431088c707f9bacb5d71da35c9005.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/3e1876b24bcd9da771798d639c2e6e012a1431088c707f9bacb5d71da35c9005.jpg)

![53d4d0e9fa2112675c0f0d1cdb9b85710087e66646d3324b9d7e0c7d5a408681.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/53d4d0e9fa2112675c0f0d1cdb9b85710087e66646d3324b9d7e0c7d5a408681.jpg)

![6a7053835a276109dfce36eab90c5db5ac833a4b1286260135fc043f9b20dfd3.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/6a7053835a276109dfce36eab90c5db5ac833a4b1286260135fc043f9b20dfd3.jpg)

![7a493a9c25e8afd4b86e8a0c926e4c1474a860deb2287485e21a928e409dc12d.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/7a493a9c25e8afd4b86e8a0c926e4c1474a860deb2287485e21a928e409dc12d.jpg)

![81fb60d5b12e816adcb4a9ac224f2fea998cfc52475fae2e4e1fe13a24c25d0d.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/81fb60d5b12e816adcb4a9ac224f2fea998cfc52475fae2e4e1fe13a24c25d0d.jpg)

![93bc31ec0702c96ba264fe0a5977b843803c9c43fd5e0b4046d968f659d97083.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/93bc31ec0702c96ba264fe0a5977b843803c9c43fd5e0b4046d968f659d97083.jpg)

![94b2501e968fe55a24151688b644c0330e755caeb1f85bdd2cd653fddc2208f7.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/94b2501e968fe55a24151688b644c0330e755caeb1f85bdd2cd653fddc2208f7.jpg)

![99991d6660f6da9b2450c10d0afd39d7d2b51d0bccb6259c6f7026eb54275794.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/99991d6660f6da9b2450c10d0afd39d7d2b51d0bccb6259c6f7026eb54275794.jpg)

![b8f42caa628093d538871ded3a4b75ba607892ff5eba290774b121f4d0062a81.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/b8f42caa628093d538871ded3a4b75ba607892ff5eba290774b121f4d0062a81.jpg)

![bbbd90cebc8780dd9742ed0423f1ac1f6766c8499c12b1ca8d94cdcfddc296b1.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/bbbd90cebc8780dd9742ed0423f1ac1f6766c8499c12b1ca8d94cdcfddc296b1.jpg)

![bf1278a1b9413d88df88988551390cfbe25a36a2da81b898137fdbbe678d68cd.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/bf1278a1b9413d88df88988551390cfbe25a36a2da81b898137fdbbe678d68cd.jpg)

![d15363ee05e747d812388a328dd3696eda0bb71d289606f079e7f7b6975925b4.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/d15363ee05e747d812388a328dd3696eda0bb71d289606f079e7f7b6975925b4.jpg)

![d8b0775ac7528064084f01c0ab238d4bbba9f72afc402b059f51f783640fc5e6.jpg](../icml_results/377_MIRA_%20A%20Score%20for%20Conditional%20Distribution%20Accuracy%20and%20Model%20Comparison/tables/d8b0775ac7528064084f01c0ab238d4bbba9f72afc402b059f51f783640fc5e6.jpg)

<a id="paper-0371"></a>

## From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs


### Images

![23bd48994705f8f3c1d0d0edcb3cb745fdb53795e9ad972e1ec34d8aba7b0329.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/23bd48994705f8f3c1d0d0edcb3cb745fdb53795e9ad972e1ec34d8aba7b0329.jpg)

![2486a34d4474d2cef766c52cd1758a6a646644ed95789a853f82fe5a74447775.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/2486a34d4474d2cef766c52cd1758a6a646644ed95789a853f82fe5a74447775.jpg)

![39a554ecb32a66ea7cdcb6179c29d14e5287ed5f8880c59840b9104051f11ff4.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/39a554ecb32a66ea7cdcb6179c29d14e5287ed5f8880c59840b9104051f11ff4.jpg)

![3d3785d55bb19e79bd7b74b5bb91640dc163f1900b879586451582fa5b061232.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/3d3785d55bb19e79bd7b74b5bb91640dc163f1900b879586451582fa5b061232.jpg)

![6cd4aa8a2dee48fa23fb2cc54f05972fa907960dc3d2b8aa15e6bf4aec64e169.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/6cd4aa8a2dee48fa23fb2cc54f05972fa907960dc3d2b8aa15e6bf4aec64e169.jpg)

![7174f670aa32a82d08f66815b2f561f691de1930472739edc91f6c372217060e.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/7174f670aa32a82d08f66815b2f561f691de1930472739edc91f6c372217060e.jpg)

![982492b02e449ab46ac4e09ec3f68e0203c242cdcf1afc3a5414d1c7ec91d2b0.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/982492b02e449ab46ac4e09ec3f68e0203c242cdcf1afc3a5414d1c7ec91d2b0.jpg)

![b1c25c939af836f6eaf400152f1adb5449da875930c654fc7b56f8a37b0863bd.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/b1c25c939af836f6eaf400152f1adb5449da875930c654fc7b56f8a37b0863bd.jpg)

![c8e3c9435d08c35e6a43b4e4f7e33e90c406ee7d51a4afd5119ddea7175dfda8.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/images/c8e3c9435d08c35e6a43b4e4f7e33e90c406ee7d51a4afd5119ddea7175dfda8.jpg)

### Tables

![1c11ab6e12694b10129bf1c6310cc717033a9295232e70f6b19f2255fe2e0ec5.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/1c11ab6e12694b10129bf1c6310cc717033a9295232e70f6b19f2255fe2e0ec5.jpg)

![33138801768b6f2257104552387753119332490ab32bef8c7bf76d6ad5c6576e.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/33138801768b6f2257104552387753119332490ab32bef8c7bf76d6ad5c6576e.jpg)

![3b6f55b543e13fcd2bcfa4a3f157d9bca954c5e9386b2d53a639a14e809dc2ea.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/3b6f55b543e13fcd2bcfa4a3f157d9bca954c5e9386b2d53a639a14e809dc2ea.jpg)

![4e9a5c429adbb7bf488552efc54fe9e8abd126c20ff75c9ed3f7c4d1dcbc2bf7.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/4e9a5c429adbb7bf488552efc54fe9e8abd126c20ff75c9ed3f7c4d1dcbc2bf7.jpg)

![7039460bbccfab91e46f4d85e2ae4cf97b547e68fa45d5421782e4995e5d5b2d.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/7039460bbccfab91e46f4d85e2ae4cf97b547e68fa45d5421782e4995e5d5b2d.jpg)

![727aa6395e1bbadfae4e4381dce3a9d2bab802a44664260d02901f23fcb08262.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/727aa6395e1bbadfae4e4381dce3a9d2bab802a44664260d02901f23fcb08262.jpg)

![7ee1d1a273ca3a831791d6c95d661554ccc5738fbb828349bfc45eb94d368a90.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/7ee1d1a273ca3a831791d6c95d661554ccc5738fbb828349bfc45eb94d368a90.jpg)

![ae79477908d38e7f837f090699705ffb1a46b912e9cdf5081af32e7924015362.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/ae79477908d38e7f837f090699705ffb1a46b912e9cdf5081af32e7924015362.jpg)

![f4669308b18d1c55ade86bed45117fa26263d697bc2165a3487268609fdd5713.jpg](../icml_results/378_From%20Poisoned%20to%20Aware_%20Fostering%20Backdoor%20Self-Awareness%20in%20LLMs/tables/f4669308b18d1c55ade86bed45117fa26263d697bc2165a3487268609fdd5713.jpg)

<a id="paper-0372"></a>

## Self-Distillation Enables Continual Learning


### Images

![06c0604ca74155cb04a69f232a9a4d13771376abe3b8c1066149e2d06381649d.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/06c0604ca74155cb04a69f232a9a4d13771376abe3b8c1066149e2d06381649d.jpg)

![4bec871165db75f2df9dcb777fad555aa5d17dceb7d6f392a94108544d10068f.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/4bec871165db75f2df9dcb777fad555aa5d17dceb7d6f392a94108544d10068f.jpg)

![4f8a1f3d6b0631974d4ecbe449084af4dd7682dcdd05072f2117bc2c6c9c4a95.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/4f8a1f3d6b0631974d4ecbe449084af4dd7682dcdd05072f2117bc2c6c9c4a95.jpg)

![8f3f88ec50da2f259a69eedd5a15f90c9ab39b401869d376128275166aa57ac3.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/8f3f88ec50da2f259a69eedd5a15f90c9ab39b401869d376128275166aa57ac3.jpg)

![947601520bdbe057c380c0707d3b727e0c5bb45c34e9a79b5eada042266e74c4.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/947601520bdbe057c380c0707d3b727e0c5bb45c34e9a79b5eada042266e74c4.jpg)

![a2452a3c2d1a1770f8628e88b5f6cd1dbf0761a2d9fc5cfaa9b752593cc2b754.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/a2452a3c2d1a1770f8628e88b5f6cd1dbf0761a2d9fc5cfaa9b752593cc2b754.jpg)

![aa4593820e7ec6c71039f3675fbb9628bcc174386560c1636b51beef506ef9a9.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/aa4593820e7ec6c71039f3675fbb9628bcc174386560c1636b51beef506ef9a9.jpg)

![be86a23b1f0f60fe4b27f5850d0c16104375b27e3c6c5dd185cae437307b584e.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/images/be86a23b1f0f60fe4b27f5850d0c16104375b27e3c6c5dd185cae437307b584e.jpg)

### Tables

![05f93a894b3a826b2952f8070860ed6cc4c663730139c1ed9c73ad82d71145a6.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/tables/05f93a894b3a826b2952f8070860ed6cc4c663730139c1ed9c73ad82d71145a6.jpg)

![0de6035ad50a42aa9f0ee6fb4ad7062230095e060be990a33865d7a192c8797d.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/tables/0de6035ad50a42aa9f0ee6fb4ad7062230095e060be990a33865d7a192c8797d.jpg)

![2c846409047c7bc4334a564b1cbf09bdfa98b1e8f23f1396c355b93c6ce240e3.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/tables/2c846409047c7bc4334a564b1cbf09bdfa98b1e8f23f1396c355b93c6ce240e3.jpg)

![414ab803b6d8826f4e6f36ee0aa1f504deae37d0bd6facf745daba18040a1b01.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/tables/414ab803b6d8826f4e6f36ee0aa1f504deae37d0bd6facf745daba18040a1b01.jpg)

![5be1fe36ae96e519e930b12abe11cc1c31fd77baa8f7ce19d02d127c4958ee69.jpg](../icml_results/379_Self-Distillation%20Enables%20Continual%20Learning/tables/5be1fe36ae96e519e930b12abe11cc1c31fd77baa8f7ce19d02d127c4958ee69.jpg)

<a id="paper-0373"></a>

## S$^3$GNN: Efficient Global Mixing and Local Message Passing for Long-Range Graph Learning


### Images

![06397207c7a814dceb40e495c5d5c6d7f4fa2481f1de8bc363f34ee19754efad.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/06397207c7a814dceb40e495c5d5c6d7f4fa2481f1de8bc363f34ee19754efad.jpg)

![1822865723ee4a92a826ae7c5388a8502407c367f66e2482869ece9b047902e4.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/1822865723ee4a92a826ae7c5388a8502407c367f66e2482869ece9b047902e4.jpg)

![230417de0a18fd30fb1ba0a8c9581f45d9d6d3c887f1c94d8a08e09bf3a69afb.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/230417de0a18fd30fb1ba0a8c9581f45d9d6d3c887f1c94d8a08e09bf3a69afb.jpg)

![d77e7c21bbe5ea4d7ee5e498a46bef3e2270c18cb9623eb59d13afed241dc7cf.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/d77e7c21bbe5ea4d7ee5e498a46bef3e2270c18cb9623eb59d13afed241dc7cf.jpg)

![dabda0ee8067b6aef37d1dbb8a08b2a2be634ead0db3125e99d2314009f4a508.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/dabda0ee8067b6aef37d1dbb8a08b2a2be634ead0db3125e99d2314009f4a508.jpg)

![f8d7bb122fdcbd986ca7ed291ce5408cb6033923108f495da7667093d729ffb6.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/f8d7bb122fdcbd986ca7ed291ce5408cb6033923108f495da7667093d729ffb6.jpg)

![ff87f37555528eba9e0df32e4bed6cb23dc675984848e838893e022c1833496b.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/ff87f37555528eba9e0df32e4bed6cb23dc675984848e838893e022c1833496b.jpg)

![ffdad25cbbc32129c54f556ab139a086832ce143feeec23ba9ac5b509a9281cb.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/images/ffdad25cbbc32129c54f556ab139a086832ce143feeec23ba9ac5b509a9281cb.jpg)

### Tables

![12b787ea41ff2af168130b9ddc891b0e4eef655f3abbf037406849046e5f897b.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/12b787ea41ff2af168130b9ddc891b0e4eef655f3abbf037406849046e5f897b.jpg)

![236a5c63e3a1df8c4456a8a70f75ca10ad8d22e69cd51d57737af0c5bc3c8478.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/236a5c63e3a1df8c4456a8a70f75ca10ad8d22e69cd51d57737af0c5bc3c8478.jpg)

![3c4db1425b5969c249610304e28ab2bf859db9b10e3cc1d81facd275d05f6c2e.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/3c4db1425b5969c249610304e28ab2bf859db9b10e3cc1d81facd275d05f6c2e.jpg)

![414ba2b0385edb0ffea6bc48a0602e3ad1c3cc823092db077abeea943a872974.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/414ba2b0385edb0ffea6bc48a0602e3ad1c3cc823092db077abeea943a872974.jpg)

![4ed51162626af1fb262918188164adbf32a776ef753e7416591067eb03270121.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/4ed51162626af1fb262918188164adbf32a776ef753e7416591067eb03270121.jpg)

![6087a4815c7407166cf4fdbeb9e75aa5d5a1777a01b727212fd7e2ba15a16659.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/6087a4815c7407166cf4fdbeb9e75aa5d5a1777a01b727212fd7e2ba15a16659.jpg)

![7c20a922adf53ada1f602073660da2d856e4395f82f8fa2e18882882ef06b345.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/7c20a922adf53ada1f602073660da2d856e4395f82f8fa2e18882882ef06b345.jpg)

![881c7108e6a050c971d2efae3bed8b7ddf367cd219726d7b7e8f09e33d80c68f.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/881c7108e6a050c971d2efae3bed8b7ddf367cd219726d7b7e8f09e33d80c68f.jpg)

![a7d9b94374f5f97bc948a32c68d3e76bad7f1f65f3ee856bbe4bfebda3a575aa.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/a7d9b94374f5f97bc948a32c68d3e76bad7f1f65f3ee856bbe4bfebda3a575aa.jpg)

![ba6b8580c95c02efd221a85fdb71e979b743dd21ca5bee8c7b0a42681b9c6719.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/ba6b8580c95c02efd221a85fdb71e979b743dd21ca5bee8c7b0a42681b9c6719.jpg)

![baf7c2403418e0c797410cfc50b8c4f5d860ab00347ee3f1c4e73655153cfc1a.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/baf7c2403418e0c797410cfc50b8c4f5d860ab00347ee3f1c4e73655153cfc1a.jpg)

![f773ac4f0b719908b72ee2cfcec86ed0f82eaf80cb268957da1d1336f24cdafa.jpg](../icml_results/380_S%24%5E3%24GNN_%20Efficient%20Global%20Mixing%20and%20Local%20Message%20Passing%20for%20Long-Range%20Graph%20Learning/tables/f773ac4f0b719908b72ee2cfcec86ed0f82eaf80cb268957da1d1336f24cdafa.jpg)

<a id="paper-0374"></a>

## Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination


### Images

![03d8ab3d9f56c39a342538584b799e6d30d501922b6ec7f065d41938ce7cba38.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/03d8ab3d9f56c39a342538584b799e6d30d501922b6ec7f065d41938ce7cba38.jpg)

![1c6d65f30f3a37e2f1317555d373ac63253017b6a198f524a99c04d5e5045122.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/1c6d65f30f3a37e2f1317555d373ac63253017b6a198f524a99c04d5e5045122.jpg)

![2b8848bf25275b3fd0da93bcba4278491ee000e6552f04409d78011b045fff90.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/2b8848bf25275b3fd0da93bcba4278491ee000e6552f04409d78011b045fff90.jpg)

![4824f1bf655e9052e2ed720b5066eedc4ce6cc2655af0b670c896ce01c4e320c.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/4824f1bf655e9052e2ed720b5066eedc4ce6cc2655af0b670c896ce01c4e320c.jpg)

![51f2127ee6dc20ad6077a9c63ace33e94e3afc82d42f0cc280465eb94275c446.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/51f2127ee6dc20ad6077a9c63ace33e94e3afc82d42f0cc280465eb94275c446.jpg)

![5ee6b04d08bd738128af0e44457bf26133dac9550ef261782173d89bf700ce5b.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/5ee6b04d08bd738128af0e44457bf26133dac9550ef261782173d89bf700ce5b.jpg)

![830c541cbb6a6205de686c1a143f8ccd67cc0bcf4b5ebd87b84cd3de835bb4b4.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/830c541cbb6a6205de686c1a143f8ccd67cc0bcf4b5ebd87b84cd3de835bb4b4.jpg)

![8343bedb1c6362a168f424a64eacdfde145d87d3c22cee5a8e30cd80bab216a5.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/8343bedb1c6362a168f424a64eacdfde145d87d3c22cee5a8e30cd80bab216a5.jpg)

![8e405790a8f5e81ad8ae9cf1495889f04092f4eecda4fc11043b0c7faafd233b.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/8e405790a8f5e81ad8ae9cf1495889f04092f4eecda4fc11043b0c7faafd233b.jpg)

![9381cd7dfc064aee3fb61ea50862716a4fb6a0f5e1a39f5864c1088516a45941.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/9381cd7dfc064aee3fb61ea50862716a4fb6a0f5e1a39f5864c1088516a45941.jpg)

![9cab5889ba8dd68e3615ea95891f8d0080223c8d03e073c2372d7a786f299ad7.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/9cab5889ba8dd68e3615ea95891f8d0080223c8d03e073c2372d7a786f299ad7.jpg)

![a0e006ecbd94a323eb622bb9c961174e774beb363d21092003cbcd12824a82fd.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/a0e006ecbd94a323eb622bb9c961174e774beb363d21092003cbcd12824a82fd.jpg)

![b0ceb04119871a6abc2ffe873d9b00829d26fd70dd60a1dec35ad4c10ef547dd.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/b0ceb04119871a6abc2ffe873d9b00829d26fd70dd60a1dec35ad4c10ef547dd.jpg)

![b3ed16c198c37d2f08a2d43d6339530682023628da3de9a428b07d3ceac56b18.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/b3ed16c198c37d2f08a2d43d6339530682023628da3de9a428b07d3ceac56b18.jpg)

![bb6948a2100b4ab4b564679faa21d3576c896572b222a900dec29d91e55560d5.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/bb6948a2100b4ab4b564679faa21d3576c896572b222a900dec29d91e55560d5.jpg)

![bceb9fc4771661a599a5bfe58041826c10ed22e610ccd46462ad04ca93b587ae.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/bceb9fc4771661a599a5bfe58041826c10ed22e610ccd46462ad04ca93b587ae.jpg)

![ca780f0f25e24e561aba5911cf6e0b7468826ee009c4fa66935774a62a9a8a68.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/ca780f0f25e24e561aba5911cf6e0b7468826ee009c4fa66935774a62a9a8a68.jpg)

![ee46ec5946a3f1697151724e053d2a94d8e5a25bfdee5f52ca00232e878891b7.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/ee46ec5946a3f1697151724e053d2a94d8e5a25bfdee5f52ca00232e878891b7.jpg)

![f1f95b1b22cf81be610e7a7e7aa4c857455789996340c4a2c062773039441b24.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/images/f1f95b1b22cf81be610e7a7e7aa4c857455789996340c4a2c062773039441b24.jpg)

### Tables

![0b185d0ebf114646cc6a6107b0619b1aa5dab0e3a130033b47a36181f57cc7ad.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/0b185d0ebf114646cc6a6107b0619b1aa5dab0e3a130033b47a36181f57cc7ad.jpg)

![1d043f7c6e1bbb21ed67b5779f5ad5323502d7225567bd70c3ecd7c86a5e98b4.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/1d043f7c6e1bbb21ed67b5779f5ad5323502d7225567bd70c3ecd7c86a5e98b4.jpg)

![250aedcc5a6e0d4cfa706dc1a2cbf50bc8be850379fb31fcd5e1198a26d9c804.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/250aedcc5a6e0d4cfa706dc1a2cbf50bc8be850379fb31fcd5e1198a26d9c804.jpg)

![5631b7363abb6d8fbb4189192c6293fc7fb9cc38bd15b13a1fcdd7e83f64bd35.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/5631b7363abb6d8fbb4189192c6293fc7fb9cc38bd15b13a1fcdd7e83f64bd35.jpg)

![63668a66110af956189e677c9337f3b6be614370bc2b4e55d95c4d52c550e302.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/63668a66110af956189e677c9337f3b6be614370bc2b4e55d95c4d52c550e302.jpg)

![7bc6e51f144ab655fcb60aa461b0265baf4249bf75440c4916f43b8f0876c29c.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/7bc6e51f144ab655fcb60aa461b0265baf4249bf75440c4916f43b8f0876c29c.jpg)

![818af4e4cfd65d82f7f274901dd8be6be25404b7497df3a4ee16a511ca187955.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/818af4e4cfd65d82f7f274901dd8be6be25404b7497df3a4ee16a511ca187955.jpg)

![8cc7b174af7ed57c822b9e547cd26a040ccfe05e60fcbe04498184352d9e7fe2.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/8cc7b174af7ed57c822b9e547cd26a040ccfe05e60fcbe04498184352d9e7fe2.jpg)

![97d5d8359d588c8553cc19fdec3768f0ddf0e824538a99d1a77280a0dc7f2fa6.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/97d5d8359d588c8553cc19fdec3768f0ddf0e824538a99d1a77280a0dc7f2fa6.jpg)

![9a1609da74cce846b8f95fb79c4054305402d0f76ed3ebf4e2e3db9edd3cfb5e.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/9a1609da74cce846b8f95fb79c4054305402d0f76ed3ebf4e2e3db9edd3cfb5e.jpg)

![a476c1f525e4e6a91e2c87f46d90629f95241cd8a42bebcc22e318ad4ed77e50.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/a476c1f525e4e6a91e2c87f46d90629f95241cd8a42bebcc22e318ad4ed77e50.jpg)

![b06207b93015ffc173133f390288e29c0441cb0ff6318beb40393efd851b1afe.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/b06207b93015ffc173133f390288e29c0441cb0ff6318beb40393efd851b1afe.jpg)

![b7a968ae80e96d40ffc54b423e24860ab69e78253dd0e194b75362e0d75869f6.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/b7a968ae80e96d40ffc54b423e24860ab69e78253dd0e194b75362e0d75869f6.jpg)

![c9458721d9db951e8a1d33666c0ac0bebf1f5bc8518a0f3b20de43c1c35c290b.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/c9458721d9db951e8a1d33666c0ac0bebf1f5bc8518a0f3b20de43c1c35c290b.jpg)

![c9ae77457f4ad4790ab89783d53634059290583a1243afaa6401c9466ce68742.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/c9ae77457f4ad4790ab89783d53634059290583a1243afaa6401c9466ce68742.jpg)

![ca29d464000605e5d979b1c126bc9277ee664574a5d74aef1055da0a55223ad2.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/ca29d464000605e5d979b1c126bc9277ee664574a5d74aef1055da0a55223ad2.jpg)

![d1bcd41a8a1a89f223d87cb16553f5119ba116317026d39aca4ac5d95306658f.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/d1bcd41a8a1a89f223d87cb16553f5119ba116317026d39aca4ac5d95306658f.jpg)

![d3851fb035c6371291d01618aaa6eb33887fc0ae7e18fcede6537d9ad22d0525.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/d3851fb035c6371291d01618aaa6eb33887fc0ae7e18fcede6537d9ad22d0525.jpg)

![fa47d0ee593eecc940b12fee25b409d4f5aa541ebbc189634292d62f01522dcf.jpg](../icml_results/381_Are%20VLMs%20Seeing%20or%20Just%20Saying_%20Uncovering%20the%20Illusion%20of%20Visual%20Re-examination/tables/fa47d0ee593eecc940b12fee25b409d4f5aa541ebbc189634292d62f01522dcf.jpg)

<a id="paper-0375"></a>

## Path-dependent Discrete Amortized Inference


### Images

![0d427134ea1f1d83452c2a520be7ca20477a883a8367fba2ba25e99b4a8312ea.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/0d427134ea1f1d83452c2a520be7ca20477a883a8367fba2ba25e99b4a8312ea.jpg)

![2ce861a231be496238f59b3768edf82794462e497e3f8b7d660e3a40df6eed52.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/2ce861a231be496238f59b3768edf82794462e497e3f8b7d660e3a40df6eed52.jpg)

![4878adf8556332eea3e3526935419a4fd8484dbcc6610890217da84b5d416cf3.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/4878adf8556332eea3e3526935419a4fd8484dbcc6610890217da84b5d416cf3.jpg)

![4bcc1caef74e5103a8ff710e0f8e82441e8c9da429a2fc2c3006422a0eeafcb2.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/4bcc1caef74e5103a8ff710e0f8e82441e8c9da429a2fc2c3006422a0eeafcb2.jpg)

![4f1812c09179a89f5554265db513cc19f46913cd5386385ef30f5c1093245894.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/4f1812c09179a89f5554265db513cc19f46913cd5386385ef30f5c1093245894.jpg)

![607303bc4d37223735927bd2660bb644a110dcc16d477901613e4df26d1b2bb6.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/607303bc4d37223735927bd2660bb644a110dcc16d477901613e4df26d1b2bb6.jpg)

![6db1f28297a3ecf23122f479c30c3da573a6178c829ab175760371e612f2134e.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/6db1f28297a3ecf23122f479c30c3da573a6178c829ab175760371e612f2134e.jpg)

![8edf09f9a71257d28df4b11f09c6a7ee23eead4679c02660b040a2ab9d404bd4.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/8edf09f9a71257d28df4b11f09c6a7ee23eead4679c02660b040a2ab9d404bd4.jpg)

![9201e559ad7b87e0fe39309b3240d7aadc742b29ee96d5ffeedc169305cb988a.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/9201e559ad7b87e0fe39309b3240d7aadc742b29ee96d5ffeedc169305cb988a.jpg)

![99d303997fafd67b3000f16469d6ab40d0522b3e78eebfebee4e2218256823a6.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/99d303997fafd67b3000f16469d6ab40d0522b3e78eebfebee4e2218256823a6.jpg)

![aa0771a0a9146014be8531bbb00152fa64760c871c56d304623d82fc30585990.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/aa0771a0a9146014be8531bbb00152fa64760c871c56d304623d82fc30585990.jpg)

![be190495d569c9123fe7a5082a3c1b2b35535e2bcab3624e744aae4b788062a9.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/be190495d569c9123fe7a5082a3c1b2b35535e2bcab3624e744aae4b788062a9.jpg)

![c41028bab677947ed0bd2b7944770e5659fd79ef8103f8c63e93454f548f729b.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/c41028bab677947ed0bd2b7944770e5659fd79ef8103f8c63e93454f548f729b.jpg)

![d7c3d5ce09cc08b2db72b6baa5839f1817605d27285f92936ecec46e16c92c73.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/d7c3d5ce09cc08b2db72b6baa5839f1817605d27285f92936ecec46e16c92c73.jpg)

![f4d0d3db59289bde005d3140d9b949d8a521e831d1ad91f5babdd4219cb6ee9e.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/images/f4d0d3db59289bde005d3140d9b949d8a521e831d1ad91f5babdd4219cb6ee9e.jpg)

### Tables

![09e9b2f732245f446ba2e04e342d682fd71ace98ec4c071bc918b4aa0414cdfb.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/tables/09e9b2f732245f446ba2e04e342d682fd71ace98ec4c071bc918b4aa0414cdfb.jpg)

![1ac1c36f6a797e2c0ae97cf272bee5d4d94cb148bd146d7afe8ba2229ceff7d3.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/tables/1ac1c36f6a797e2c0ae97cf272bee5d4d94cb148bd146d7afe8ba2229ceff7d3.jpg)

![6ca443f47dda0a6e462cef4cf7cf519af620dc738c76b2cf3a348804b60cb2f3.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/tables/6ca443f47dda0a6e462cef4cf7cf519af620dc738c76b2cf3a348804b60cb2f3.jpg)

![ee9a2bd2fabd8fadbf0fac1cf3c734199c727e8c7210f0db7e291f2bf7a71366.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/tables/ee9a2bd2fabd8fadbf0fac1cf3c734199c727e8c7210f0db7e291f2bf7a71366.jpg)

![f42f8760e7811a0ae440d2ba4184a3e17effa8ce81147829907e321d09d028c2.jpg](../icml_results/382_Path-dependent%20Discrete%20Amortized%20Inference/tables/f42f8760e7811a0ae440d2ba4184a3e17effa8ce81147829907e321d09d028c2.jpg)

<a id="paper-0376"></a>

## Incentivizing Truthfulness and Collaborative Fairness in Bayesian Learning


### Images

![0175b00f3441fa754e2c7b2a1f20751fc35d3ca41e04af4b75a9f405c5f3af51.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/0175b00f3441fa754e2c7b2a1f20751fc35d3ca41e04af4b75a9f405c5f3af51.jpg)

![0221d10911101cfee16b73398b5fc4b098d2c2c088ef9c734166d711336e6bd9.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/0221d10911101cfee16b73398b5fc4b098d2c2c088ef9c734166d711336e6bd9.jpg)

![0a8e96acb571dc4eb51ae07b17a68ba4ec61d77b5a6e0c6ee7854dcb7f3d842e.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/0a8e96acb571dc4eb51ae07b17a68ba4ec61d77b5a6e0c6ee7854dcb7f3d842e.jpg)

![1391f82ad067db9478ac1a313fc7fb51150b8397383f2f238b648996d488ef10.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/1391f82ad067db9478ac1a313fc7fb51150b8397383f2f238b648996d488ef10.jpg)

![2da6e306c725072698794f7fccad0e9f329c803a7bfbc072d8d90dd9412de46f.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/2da6e306c725072698794f7fccad0e9f329c803a7bfbc072d8d90dd9412de46f.jpg)

![4f7fc5a9f2e517c5aae8d7c7ecc370de593d8e8a3a5c02fcea5f9bd5dcbd7f49.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/4f7fc5a9f2e517c5aae8d7c7ecc370de593d8e8a3a5c02fcea5f9bd5dcbd7f49.jpg)

![50b8ca26c0d3c1403b8290dcdcdebdc778027e20b96b7c439aaf232917409f9f.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/50b8ca26c0d3c1403b8290dcdcdebdc778027e20b96b7c439aaf232917409f9f.jpg)

![67741f8f54bd9e42d528d84645902cf04a53d2523e4f520a89c65f347d965a5a.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/67741f8f54bd9e42d528d84645902cf04a53d2523e4f520a89c65f347d965a5a.jpg)

![8dfc39d29b4b31de4be09e11ad3018d2f34dfe306d1b1ca6908dd2842423d40a.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/8dfc39d29b4b31de4be09e11ad3018d2f34dfe306d1b1ca6908dd2842423d40a.jpg)

![9b741e85033a05a3af519c94e80fa630f6218f16386031da746753668847ef30.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/9b741e85033a05a3af519c94e80fa630f6218f16386031da746753668847ef30.jpg)

![9c565675682ca8179e4a8181b1ffcc5ce588d172349f7c5636c2eb6d9fa8ec8a.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/9c565675682ca8179e4a8181b1ffcc5ce588d172349f7c5636c2eb6d9fa8ec8a.jpg)

![a09bb56d2816946d2cd4e83f087f1c4b7eb5f764631f8653624f03d89692503a.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/a09bb56d2816946d2cd4e83f087f1c4b7eb5f764631f8653624f03d89692503a.jpg)

![af0cc227e8114c97f839fdf41e6991233282c4aaa407bb6c3b6e6e1824967644.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/af0cc227e8114c97f839fdf41e6991233282c4aaa407bb6c3b6e6e1824967644.jpg)

![b378601277efea540b29e66f1cb74028aab89e39971e473bf6c29c66922e2775.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/b378601277efea540b29e66f1cb74028aab89e39971e473bf6c29c66922e2775.jpg)

![b628c7446a74c45702842e53686c471ab0773e2de4bc5478ad5f16d7b34d943e.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/b628c7446a74c45702842e53686c471ab0773e2de4bc5478ad5f16d7b34d943e.jpg)

![bed6c33ed254d4dbc11017f13802bb99bb7fe0d27faf5cdb0fafefe7efaa939a.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/bed6c33ed254d4dbc11017f13802bb99bb7fe0d27faf5cdb0fafefe7efaa939a.jpg)

![cb79e2440c6a184bfa9bbcafdc0058fa4910452bd080b9469331d79f2f16a5e5.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/cb79e2440c6a184bfa9bbcafdc0058fa4910452bd080b9469331d79f2f16a5e5.jpg)

![d4d310c315bc0d54aa138e7ac20a1e22c0c77200db46657acee25320fbb13b29.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/d4d310c315bc0d54aa138e7ac20a1e22c0c77200db46657acee25320fbb13b29.jpg)

![eddd802cbfff78be7ee6d7f9773ab2382b12199619dabf68bc5f18786677d9f3.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/images/eddd802cbfff78be7ee6d7f9773ab2382b12199619dabf68bc5f18786677d9f3.jpg)

### Tables

![630e7c6d9e7324408709260d830f814b10d0411e39f2aa7463a6ae80104f3ffc.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/tables/630e7c6d9e7324408709260d830f814b10d0411e39f2aa7463a6ae80104f3ffc.jpg)

![68a8faddf69194d95c06f8b272adc4f0bf515fedf29eab5a252c39c5842c428e.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/tables/68a8faddf69194d95c06f8b272adc4f0bf515fedf29eab5a252c39c5842c428e.jpg)

![e76236eb1128aad7fd4479f65a48766ee6c3f23e11433e96a604cb8a94bdebd8.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/tables/e76236eb1128aad7fd4479f65a48766ee6c3f23e11433e96a604cb8a94bdebd8.jpg)

![ecdc45c8981e2f5d0485a7e9b2ee4f03f42468f2a433a4e7143ef506a68411c2.jpg](../icml_results/384_Incentivizing%20Truthfulness%20and%20Collaborative%20Fairness%20in%20Bayesian%20Learning/tables/ecdc45c8981e2f5d0485a7e9b2ee4f03f42468f2a433a4e7143ef506a68411c2.jpg)

<a id="paper-0377"></a>

## RED-HDP-HMM: Observation-Dependent Durations for Bayesian Nonparametric Sequential Models


### Images

![09433ae44e3ee40a3cc2f3877c10566984084fa44710433a82fba926bf33a12b.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/09433ae44e3ee40a3cc2f3877c10566984084fa44710433a82fba926bf33a12b.jpg)

![78d27482438bd19ffea9e00d2327db8730c01a9627d66032a8d49e2bcc32b318.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/78d27482438bd19ffea9e00d2327db8730c01a9627d66032a8d49e2bcc32b318.jpg)

![9b85e59a8b16b5c44e54ec78d126d5556adccfa80a801584c2b46bbc6d752ba0.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/9b85e59a8b16b5c44e54ec78d126d5556adccfa80a801584c2b46bbc6d752ba0.jpg)

![a06670b66393d3cac62f670569514f5d40e110ab1f7b39c4b486b63a93a83bcb.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/a06670b66393d3cac62f670569514f5d40e110ab1f7b39c4b486b63a93a83bcb.jpg)

![ac1b2f1b8475763a43d9608898f7ad75fc7fe217630a9378d7cf7d27b3e8d6f8.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/ac1b2f1b8475763a43d9608898f7ad75fc7fe217630a9378d7cf7d27b3e8d6f8.jpg)

![c38cecc0a1a0505873a388f7d942ba0a1e8aaab58100996537d247692c3f95d3.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/c38cecc0a1a0505873a388f7d942ba0a1e8aaab58100996537d247692c3f95d3.jpg)

![c7f53581fd7fd6ce07c35a9115a51e5f7ced4f118327d680730795f9d0fc5a0f.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/images/c7f53581fd7fd6ce07c35a9115a51e5f7ced4f118327d680730795f9d0fc5a0f.jpg)

### Tables

![1d86de4e13d559fe2e8f481131adbb935ca5ea01357f49afbc8259a1e7fb9c5c.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/1d86de4e13d559fe2e8f481131adbb935ca5ea01357f49afbc8259a1e7fb9c5c.jpg)

![2c1087d3fe5b35dbdd52d1289961e88c72fe08af74d0ac90e13f4d414d783b3f.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/2c1087d3fe5b35dbdd52d1289961e88c72fe08af74d0ac90e13f4d414d783b3f.jpg)

![2ccdc8c78bbca28e399f1d0ab314424d604fa7e26b5034e48d2996bc8fc99bcd.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/2ccdc8c78bbca28e399f1d0ab314424d604fa7e26b5034e48d2996bc8fc99bcd.jpg)

![386849548a11ef4d45e90ee352a03ad8ee0bad740fd2ef184f5d0908c40ad385.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/386849548a11ef4d45e90ee352a03ad8ee0bad740fd2ef184f5d0908c40ad385.jpg)

![3be37643f4d152c03d7487de65728b9d1dab00f28e087a5af18b98a115bc6164.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/3be37643f4d152c03d7487de65728b9d1dab00f28e087a5af18b98a115bc6164.jpg)

![55506ee28ea515b402e5d2251e380e0adf0337fc4696d33bf7e8114ad8b9dc9f.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/55506ee28ea515b402e5d2251e380e0adf0337fc4696d33bf7e8114ad8b9dc9f.jpg)

![7e83228c0f3a1b573edb1cba9f87535e3a96b29bc7e5d1d52eb1d6818bba7e20.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/7e83228c0f3a1b573edb1cba9f87535e3a96b29bc7e5d1d52eb1d6818bba7e20.jpg)

![8da44d0a5ffecec0138937373ea4f069a2f4f3c66f1845a4a9518695b70631dd.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/8da44d0a5ffecec0138937373ea4f069a2f4f3c66f1845a4a9518695b70631dd.jpg)

![b19a9974a1e3f389d341eb600fd236f3a6b349dc2468e5ec200756b657afa6ff.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/b19a9974a1e3f389d341eb600fd236f3a6b349dc2468e5ec200756b657afa6ff.jpg)

![ddd111f56b99f4b56c057c06042a7a930c48fbe9b0dac156c87e9f0efa5d98b5.jpg](../icml_results/385_RED-HDP-HMM_%20Observation-Dependent%20Durations%20for%20Bayesian%20Nonparametric%20Sequential%20Models/tables/ddd111f56b99f4b56c057c06042a7a930c48fbe9b0dac156c87e9f0efa5d98b5.jpg)

<a id="paper-0378"></a>

## Judging What We Cannot Solve: A Consequence-Based Approach for Oracle-Free Evaluation of Research-Level Math


### Images

![1b6f52b31215ef37f1e5c2694676baa42a49f40f9d7418a19f278aa0b382550e.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/images/1b6f52b31215ef37f1e5c2694676baa42a49f40f9d7418a19f278aa0b382550e.jpg)

![5ab937d52da32d7dcd6c2778970039bb7905ab137690f9880e61675361adf925.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/images/5ab937d52da32d7dcd6c2778970039bb7905ab137690f9880e61675361adf925.jpg)

![6f1ef54b8059c335aa18826292d7345ebc02b2722015a639b1b7d3232eac561a.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/images/6f1ef54b8059c335aa18826292d7345ebc02b2722015a639b1b7d3232eac561a.jpg)

![a48205f95d62fdcf5cc5907c98ae8433d57333599f41a520a824b0b3ad9dacf7.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/images/a48205f95d62fdcf5cc5907c98ae8433d57333599f41a520a824b0b3ad9dacf7.jpg)

![da998861f7d9b0e2ca31952187464fab03d641413e31f5210e46b46c2f380092.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/images/da998861f7d9b0e2ca31952187464fab03d641413e31f5210e46b46c2f380092.jpg)

### Tables

![69a3e328f04d184da7ace3070c3972b0674688edf042f23d4a5d9d8f2b713728.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/69a3e328f04d184da7ace3070c3972b0674688edf042f23d4a5d9d8f2b713728.jpg)

![7b1c1ad4eabad419bba9e66dfa2a9d15d74ef639b644f27e53dd60a5c8ad0b6b.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/7b1c1ad4eabad419bba9e66dfa2a9d15d74ef639b644f27e53dd60a5c8ad0b6b.jpg)

![7e76b57acbb8322d74d611b07214ef567cdb2202c11b7e69fb6a55a15c825b8a.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/7e76b57acbb8322d74d611b07214ef567cdb2202c11b7e69fb6a55a15c825b8a.jpg)

![907b61f0a1382019ef441f9a468d95485ed798184970962af97dc67721183bd2.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/907b61f0a1382019ef441f9a468d95485ed798184970962af97dc67721183bd2.jpg)

![c71cdeefbdd1a7692d3f35d72e783f31550456b450a77e2c3b29c3dfbb10bb27.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/c71cdeefbdd1a7692d3f35d72e783f31550456b450a77e2c3b29c3dfbb10bb27.jpg)

![d9212185e2fc663fead657fda11e5ae445b93613d00fca8ed3ab4abbfc6445af.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/d9212185e2fc663fead657fda11e5ae445b93613d00fca8ed3ab4abbfc6445af.jpg)

![e795dc2c40282095b846b45bd5526a2f7a319787098d289fe0807f75c0b973c1.jpg](../icml_results/386_Judging%20What%20We%20Cannot%20Solve_%20A%20Consequence-Based%20Approach%20for%20Oracle-Free%20Evaluation%20of%20Research-Le/tables/e795dc2c40282095b846b45bd5526a2f7a319787098d289fe0807f75c0b973c1.jpg)

<a id="paper-0379"></a>

## WBMM: Windowed Batch Matrix Multiplication for Efficient Large Receptive Field Convolution


### Images

![8b6a36fd1088dce7e3eb59a1a0aa640adbe3486574f5a127f5fb1e0e3a2984ba.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/images/8b6a36fd1088dce7e3eb59a1a0aa640adbe3486574f5a127f5fb1e0e3a2984ba.jpg)

![a214ae4cb172f431ec545086431f91add90ce80b7498cfa3d6cd24554abd00f4.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/images/a214ae4cb172f431ec545086431f91add90ce80b7498cfa3d6cd24554abd00f4.jpg)

![e860febca8fa47e0703deae569f90973559270e7aed0f2e4100403a0031236ef.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/images/e860febca8fa47e0703deae569f90973559270e7aed0f2e4100403a0031236ef.jpg)

![f805ab1bbfa900d60d519cf41a9b4413c65359fab507286f37a36ba87996a63c.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/images/f805ab1bbfa900d60d519cf41a9b4413c65359fab507286f37a36ba87996a63c.jpg)

### Tables

![001a5a01601415b74e874909f7b58778573402b8854d06c0e3279757a4310cc4.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/001a5a01601415b74e874909f7b58778573402b8854d06c0e3279757a4310cc4.jpg)

![2b19c06e557c1e21e1ffb3fe0883171cb91ee5631bda07ac57f754cf1e14e276.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/2b19c06e557c1e21e1ffb3fe0883171cb91ee5631bda07ac57f754cf1e14e276.jpg)

![3e0bee3f0d292c2fbe914fc0faaa0cbbd5390180d6ec66faca65e8906bad6cd3.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/3e0bee3f0d292c2fbe914fc0faaa0cbbd5390180d6ec66faca65e8906bad6cd3.jpg)

![3e14a4fcd7e93e0629b438aacbd46eb76d7fc612aa39660707b2580235c82991.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/3e14a4fcd7e93e0629b438aacbd46eb76d7fc612aa39660707b2580235c82991.jpg)

![3f9b5fafa679cd2f8bcf64f8417f2f41848279f9299ca383a7b574910166e0d2.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/3f9b5fafa679cd2f8bcf64f8417f2f41848279f9299ca383a7b574910166e0d2.jpg)

![4f9bdf200dda1c9e9a6d3aedd2d213d0f9317f7aa8a509d85e65b95729aee1d9.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/4f9bdf200dda1c9e9a6d3aedd2d213d0f9317f7aa8a509d85e65b95729aee1d9.jpg)

![55796d523865ace91cda1d87333794a5f74b18bb8d33d1e0c44fda9c89f8da23.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/55796d523865ace91cda1d87333794a5f74b18bb8d33d1e0c44fda9c89f8da23.jpg)

![56525f90d088e5e648d50d35aed236d31c220382d812be0c9abac21b01d73102.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/56525f90d088e5e648d50d35aed236d31c220382d812be0c9abac21b01d73102.jpg)

![65896fd4e2e1b3a0615e657464975ec13cd05cca31fc76626c7986d087d96e0a.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/65896fd4e2e1b3a0615e657464975ec13cd05cca31fc76626c7986d087d96e0a.jpg)

![6824c7261a8e1b5b1e29950cd09e650fa6a6591decbaba3974fa5213de2c5e6f.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/6824c7261a8e1b5b1e29950cd09e650fa6a6591decbaba3974fa5213de2c5e6f.jpg)

![9a8274cdbc09b4a9cdcf5adc5b04ccaf8d74558ab39f478436bf38f872fa8ea8.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/9a8274cdbc09b4a9cdcf5adc5b04ccaf8d74558ab39f478436bf38f872fa8ea8.jpg)

![a162bf08e01421f8c96c9c4db02be8607adea5eec0888d9571db9205c4a039a0.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/a162bf08e01421f8c96c9c4db02be8607adea5eec0888d9571db9205c4a039a0.jpg)

![a4337416e906a8b428cfa7c2cfdad9364aa227c1f25d4756da41a2fad3c8ff0d.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/a4337416e906a8b428cfa7c2cfdad9364aa227c1f25d4756da41a2fad3c8ff0d.jpg)

![aead7cb4234d7c03ff1c5e20b7810695cda525f34a6e2a469b4cd0e0ec240e57.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/aead7cb4234d7c03ff1c5e20b7810695cda525f34a6e2a469b4cd0e0ec240e57.jpg)

![b357a9e65e11abc1e5e796b870fc1b731e87065f5834cf5d846280222f2a0af6.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/b357a9e65e11abc1e5e796b870fc1b731e87065f5834cf5d846280222f2a0af6.jpg)

![bf8adf896205a5fbce55ec52897869b2b522171e775b70f72028104a148f5092.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/bf8adf896205a5fbce55ec52897869b2b522171e775b70f72028104a148f5092.jpg)

![ce9ee47113e42321e1c67a9d0fdb4e844e1aceb06fee1d6bad15f0f8efe1a01b.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/ce9ee47113e42321e1c67a9d0fdb4e844e1aceb06fee1d6bad15f0f8efe1a01b.jpg)

![d6cc1d7148ca5574b5f75966a1ec7865465ce1004a96fe85c5574e71f5364d7f.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/d6cc1d7148ca5574b5f75966a1ec7865465ce1004a96fe85c5574e71f5364d7f.jpg)

![f00e42ba331f82d789098defa74439dc8dccc6631be4437f82f97cdced7d5b47.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/f00e42ba331f82d789098defa74439dc8dccc6631be4437f82f97cdced7d5b47.jpg)

![fec83e8d90b6e0f094c76ca233a3e9157a0215caf447ef9dad181fd89db67152.jpg](../icml_results/387_WBMM_%20Windowed%20Batch%20Matrix%20Multiplication%20for%20Efficient%20Large%20Receptive%20Field%20Convolution/tables/fec83e8d90b6e0f094c76ca233a3e9157a0215caf447ef9dad181fd89db67152.jpg)

<a id="paper-0380"></a>

## TG-RAG: A Retrieval-Augmented Framework for Reasoning Guidance in Specialized Domains


### Images

![10f37a4f9c3ff8cff459fc7bd92564231fb443db6bca5a4dabe7b84dbd3e0ce6.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/10f37a4f9c3ff8cff459fc7bd92564231fb443db6bca5a4dabe7b84dbd3e0ce6.jpg)

![417ffcea5752bc738050adc9b4a4032e5d8ab460e8abcec68f43c19c4ffc2f6b.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/417ffcea5752bc738050adc9b4a4032e5d8ab460e8abcec68f43c19c4ffc2f6b.jpg)

![493d3b78588e78499f074b4d568545de4741103a3c86487ebe234f2e9abe7995.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/493d3b78588e78499f074b4d568545de4741103a3c86487ebe234f2e9abe7995.jpg)

![4f108a623bb1da4abaf59120a025bf2cc2e988ee92a4fe6cab3b8d64dfa36bf5.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/4f108a623bb1da4abaf59120a025bf2cc2e988ee92a4fe6cab3b8d64dfa36bf5.jpg)

![64442c593b757b013a3df73d7e3574f5fee231af5809e00eb0d705d23a8a5a6a.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/64442c593b757b013a3df73d7e3574f5fee231af5809e00eb0d705d23a8a5a6a.jpg)

![e57f9e740587c4519b3c9f1bb2d1863245bbbc1ce178f598cd6d435e967a18e9.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/images/e57f9e740587c4519b3c9f1bb2d1863245bbbc1ce178f598cd6d435e967a18e9.jpg)

### Tables

![1dd14b484f443b653c3f2997117d2a9c724fcdd1aa87a01a3c4d944a3330ebcc.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/1dd14b484f443b653c3f2997117d2a9c724fcdd1aa87a01a3c4d944a3330ebcc.jpg)

![657034d1c6e2e4c2e5dfe62247ea1a73f0f82e3163721904e315bdffc72ea2a3.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/657034d1c6e2e4c2e5dfe62247ea1a73f0f82e3163721904e315bdffc72ea2a3.jpg)

![6982e7f6325922fb46a5b925921d07a7c6c0a3dc4eb9a300f988492bb7f71718.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/6982e7f6325922fb46a5b925921d07a7c6c0a3dc4eb9a300f988492bb7f71718.jpg)

![81a1704350cebe413cf306697201da759c7b1f3f68b0c0011884a10a3cc3ba3e.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/81a1704350cebe413cf306697201da759c7b1f3f68b0c0011884a10a3cc3ba3e.jpg)

![e2f8db1fc8f0d695b97052b2ab145ca815e00b02cd372655f2f480e01000cc7d.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/e2f8db1fc8f0d695b97052b2ab145ca815e00b02cd372655f2f480e01000cc7d.jpg)

![f533ffefad547baf923d145e716565e60586f99ff306021935bd77f48d836dea.jpg](../icml_results/388_TG-RAG_%20A%20Retrieval-Augmented%20Framework%20for%20Reasoning%20Guidance%20in%20Specialized%20Domains/tables/f533ffefad547baf923d145e716565e60586f99ff306021935bd77f48d836dea.jpg)

<a id="paper-0381"></a>

## Multimodal Latent Language Modeling with Next-Token Diffusion


### Images

![0e87579fccdb3a5a8dd2ae5e327e8b7f382c5c878266eab0b41752e5c1cf7ff0.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/0e87579fccdb3a5a8dd2ae5e327e8b7f382c5c878266eab0b41752e5c1cf7ff0.jpg)

![0f392e55cb793e2a2a1385990e8b8d11879ba54bd4590068219e59714b87d012.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/0f392e55cb793e2a2a1385990e8b8d11879ba54bd4590068219e59714b87d012.jpg)

![281dbc09318198d1b9d3c65448f61d091906775f55c6aaa25c3e5969664ff032.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/281dbc09318198d1b9d3c65448f61d091906775f55c6aaa25c3e5969664ff032.jpg)

![3132e517346cac8c66cd6ce1a612f8750472e84785584ba75812e362cb55078c.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/3132e517346cac8c66cd6ce1a612f8750472e84785584ba75812e362cb55078c.jpg)

![3349f4d153bec4bcdb22bb32a82004b490fe5dc33a2fd64d18de83ac498057c0.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/3349f4d153bec4bcdb22bb32a82004b490fe5dc33a2fd64d18de83ac498057c0.jpg)

![362f1de2113f6e1afe439d3df8128b3ff8154565f6033943f82ed1aa48d37a7c.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/362f1de2113f6e1afe439d3df8128b3ff8154565f6033943f82ed1aa48d37a7c.jpg)

![37da3a8a794e4e8e877c1ccd05747304b4089d63576d2ab19e462bd18c435a31.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/37da3a8a794e4e8e877c1ccd05747304b4089d63576d2ab19e462bd18c435a31.jpg)

![84e051863a5f74b1f1ab99be83ae7a9acd37d7fb32ecac58f7330340c42645dd.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/84e051863a5f74b1f1ab99be83ae7a9acd37d7fb32ecac58f7330340c42645dd.jpg)

![94e8e44f78e42deb92af2f81b6e366f1126e62c5a052c80babaa30891ac2c3cd.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/94e8e44f78e42deb92af2f81b6e366f1126e62c5a052c80babaa30891ac2c3cd.jpg)

![99a238a1094ca52163182025116d859f87f772da0090bcef5b5e70e7d05bfa43.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/99a238a1094ca52163182025116d859f87f772da0090bcef5b5e70e7d05bfa43.jpg)

![9b4806353d323c3471ecf8b02f56ee95cbe50a1c1c2cc696a18ec17f53f3eaba.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/9b4806353d323c3471ecf8b02f56ee95cbe50a1c1c2cc696a18ec17f53f3eaba.jpg)

![bcb0eb1b6aad86b73801c2949188c3aa8f8dab6f90c03d4ee5d6f87b72f6a874.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/bcb0eb1b6aad86b73801c2949188c3aa8f8dab6f90c03d4ee5d6f87b72f6a874.jpg)

![ed77ee266b292281057d9d751f0ff00a496259f05fd17bf9b0e95521015c7d3e.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/images/ed77ee266b292281057d9d751f0ff00a496259f05fd17bf9b0e95521015c7d3e.jpg)

### Tables

![1a58c4c988159caaae957c330b0192e2bc355a599467aecbcd8e290722036d63.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/1a58c4c988159caaae957c330b0192e2bc355a599467aecbcd8e290722036d63.jpg)

![2225760a1b57cc8a37855074def96deaadc2fc74431280ce849a25f565215039.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/2225760a1b57cc8a37855074def96deaadc2fc74431280ce849a25f565215039.jpg)

![4364fd2f243cd6d5ed62a04c1cbf843529a7d1500cb43583e40fe9af3eae9781.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/4364fd2f243cd6d5ed62a04c1cbf843529a7d1500cb43583e40fe9af3eae9781.jpg)

![7676b0392abf98565079c62d41ebc017ab14a3ecf960335f8dcc1f9e257e6c9d.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/7676b0392abf98565079c62d41ebc017ab14a3ecf960335f8dcc1f9e257e6c9d.jpg)

![82dd9740bed157e3265f8b8a7c88d2e43ac1257728dfacf6a39df92a39d88f72.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/82dd9740bed157e3265f8b8a7c88d2e43ac1257728dfacf6a39df92a39d88f72.jpg)

![90135d81d7143234c44995cf6dab44c0d69834622e4f8a0fc39d3a9c242dfbc4.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/90135d81d7143234c44995cf6dab44c0d69834622e4f8a0fc39d3a9c242dfbc4.jpg)

![a02900db1f5f8092ecdbb9476dcb3edfa4b03398547344d7f8430b8336e14001.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/a02900db1f5f8092ecdbb9476dcb3edfa4b03398547344d7f8430b8336e14001.jpg)

![b1137935b6e4b76851f7ec2533b204a5a43106fb2b13ea8935c3f0274e2924b3.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/b1137935b6e4b76851f7ec2533b204a5a43106fb2b13ea8935c3f0274e2924b3.jpg)

![b81edb5ee506df2a7938483d2c59f66e92cd3220276ecf22d00c7090e4d6b46a.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/b81edb5ee506df2a7938483d2c59f66e92cd3220276ecf22d00c7090e4d6b46a.jpg)

![cd29c076268cce86c32b1cfab5a0e82f48472d8085c192c4dd3ebde3f2fe708a.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/cd29c076268cce86c32b1cfab5a0e82f48472d8085c192c4dd3ebde3f2fe708a.jpg)

![d01678b2aa38d23bf4d2e1f17e072aad61c5838515d86cfbf2fb271316628b24.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/d01678b2aa38d23bf4d2e1f17e072aad61c5838515d86cfbf2fb271316628b24.jpg)

![d8e9441bb6aadb689b2b0e17b65f7ae234c359036f844289a712fca451f29ddb.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/d8e9441bb6aadb689b2b0e17b65f7ae234c359036f844289a712fca451f29ddb.jpg)

![e5652117176b8ce601989c773dbb30bbf95f86b5a17b9c9ec853e4e6f2c0aa53.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/e5652117176b8ce601989c773dbb30bbf95f86b5a17b9c9ec853e4e6f2c0aa53.jpg)

![eebf429b850f6d48720dd0da8fadd7f248f2132df729e7313572b5ad15dcff7d.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/eebf429b850f6d48720dd0da8fadd7f248f2132df729e7313572b5ad15dcff7d.jpg)

![fe7ee4e5498c9e71961e0a2cd8ea08829615381178a32c70801a6b7182c1afb0.jpg](../icml_results/389_Multimodal%20Latent%20Language%20Modeling%20with%20Next-Token%20Diffusion/tables/fe7ee4e5498c9e71961e0a2cd8ea08829615381178a32c70801a6b7182c1afb0.jpg)

<a id="paper-0382"></a>

## Training Diffusion Language Models for Black-Box Optimization


### Images

![3560317e8b6303ee1540e63d45a80c1397b1530b8b0cfc4af49959b1353a5c5c.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/images/3560317e8b6303ee1540e63d45a80c1397b1530b8b0cfc4af49959b1353a5c5c.jpg)

![95c6145c39e45d44bcda5042559f0c933fb2d9bfa997daeb1405354c28189215.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/images/95c6145c39e45d44bcda5042559f0c933fb2d9bfa997daeb1405354c28189215.jpg)

### Tables

![17e76e41ef0b62582349bed221bc5b2ae128219b167d516d8454bbf38e3b9811.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/17e76e41ef0b62582349bed221bc5b2ae128219b167d516d8454bbf38e3b9811.jpg)

![2b8d024ea11c095b3376a33502a6db5c7d196504e0c8c18fb2375129878a11b9.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/2b8d024ea11c095b3376a33502a6db5c7d196504e0c8c18fb2375129878a11b9.jpg)

![51399a9d1890c8eba32ffe06b2dd2141e8b45ac4c925a36be92c4676d8afda27.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/51399a9d1890c8eba32ffe06b2dd2141e8b45ac4c925a36be92c4676d8afda27.jpg)

![5e46563fd0f83039bedd83db5a8fdd8ff8c96e2fb0999d17233082c3999047f9.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/5e46563fd0f83039bedd83db5a8fdd8ff8c96e2fb0999d17233082c3999047f9.jpg)

![64dddce1b23cedf8c96708cb7b9b4ed2e5d323ae5bc647055ffaaadea85d7103.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/64dddce1b23cedf8c96708cb7b9b4ed2e5d323ae5bc647055ffaaadea85d7103.jpg)

![6f9a43b53f952ec58b7df6ecae3b7a36d439918f315bd5ee40694739373e3e74.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/6f9a43b53f952ec58b7df6ecae3b7a36d439918f315bd5ee40694739373e3e74.jpg)

![8a50eaeea35e0e4eaa521fe0102619c3c2e2e0e096ae9341cfa02a2d49b7f26f.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/8a50eaeea35e0e4eaa521fe0102619c3c2e2e0e096ae9341cfa02a2d49b7f26f.jpg)

![9af8f478b61872f18df0ecd01ea5e1771717f1287a9e5d8ed58a838891aff9ad.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/9af8f478b61872f18df0ecd01ea5e1771717f1287a9e5d8ed58a838891aff9ad.jpg)

![a4353b96eefb2aba68a6b06fdd763a5708228072f9970fb2f0e3eda98a8956f5.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/a4353b96eefb2aba68a6b06fdd763a5708228072f9970fb2f0e3eda98a8956f5.jpg)

![a9ba11ba7185885cf0b803d4bc9dd037d28c21eb6096c4718d690b626fc1ad29.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/a9ba11ba7185885cf0b803d4bc9dd037d28c21eb6096c4718d690b626fc1ad29.jpg)

![bff91830b1045022a0bae9e4378f38e38019406e7beea00c49c3c9012609e1b5.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/bff91830b1045022a0bae9e4378f38e38019406e7beea00c49c3c9012609e1b5.jpg)

![c333cfaf55e9290bdf47fe29339cbb29053fe0bd407ef0c050af6e7e8e452ccb.jpg](../icml_results/390_Training%20Diffusion%20Language%20Models%20for%20Black-Box%20Optimization/tables/c333cfaf55e9290bdf47fe29339cbb29053fe0bd407ef0c050af6e7e8e452ccb.jpg)

<a id="paper-0383"></a>

## Learning with Admissibility: Robust Fuzzy Hashing for Cross-Modal Retrieval with Noisy Labels


### Images

![0ccd3929eed7b630aa603e944ca74a3d5ae71ae85e3832be2db2e88c6f4a6ae4.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/0ccd3929eed7b630aa603e944ca74a3d5ae71ae85e3832be2db2e88c6f4a6ae4.jpg)

![1afb59e8f028a74e0a7ef7782fab42d05a564cb9c6c020ad3ed7068315ebef77.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/1afb59e8f028a74e0a7ef7782fab42d05a564cb9c6c020ad3ed7068315ebef77.jpg)

![29eabe17cc2d933f0b6318c37bfbd474628037d5eed38ea2629e65e70fd61051.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/29eabe17cc2d933f0b6318c37bfbd474628037d5eed38ea2629e65e70fd61051.jpg)

![52f17c800aaf7ea54dfc5da018c696e51c333f83a3f60ec6f2a007d953561593.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/52f17c800aaf7ea54dfc5da018c696e51c333f83a3f60ec6f2a007d953561593.jpg)

![5bf7881e105a657626fc7087b55b00e7f4f02a048fe8e3b8c071716c41a35fc5.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/5bf7881e105a657626fc7087b55b00e7f4f02a048fe8e3b8c071716c41a35fc5.jpg)

![64c74db1dcb0dc20376cf221ec4f160f3fc3fc70a600b93ba860781fde158dfb.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/64c74db1dcb0dc20376cf221ec4f160f3fc3fc70a600b93ba860781fde158dfb.jpg)

![70cc5994b0f786c0c64a484ea4e37e1afcab1b42e0d1a11e7f81356016cad0f0.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/70cc5994b0f786c0c64a484ea4e37e1afcab1b42e0d1a11e7f81356016cad0f0.jpg)

![70f74ec6c670852276e897795ec7afe400692d2a407cdf5ac61de835ff75c5f4.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/70f74ec6c670852276e897795ec7afe400692d2a407cdf5ac61de835ff75c5f4.jpg)

![8b06e06b456785ed8bd41a570c3c450888f418b8d39f26bbb289497b13139a13.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/images/8b06e06b456785ed8bd41a570c3c450888f418b8d39f26bbb289497b13139a13.jpg)

### Tables

![0da4012aade84bc4f2841b60ee41c9b4003a30ab46302996a45c6ca182876a12.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/0da4012aade84bc4f2841b60ee41c9b4003a30ab46302996a45c6ca182876a12.jpg)

![1a2e6a3f6cf92f91952a89e2a3558571a2daf7e1a6331d0e5ea56b5b32892bd4.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/1a2e6a3f6cf92f91952a89e2a3558571a2daf7e1a6331d0e5ea56b5b32892bd4.jpg)

![20bb0999abc4db8a08a68aabed0168ef5c3999fdba67f113b4062e2babf29c4f.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/20bb0999abc4db8a08a68aabed0168ef5c3999fdba67f113b4062e2babf29c4f.jpg)

![875d0ab9b650ede08fc61142a3cf387545a3ba65c5aa10e20543b362c52a5933.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/875d0ab9b650ede08fc61142a3cf387545a3ba65c5aa10e20543b362c52a5933.jpg)

![b50cd4f9c5362b7b8470d731a751891c63abf75d47dcf2e97857d75ce77e480f.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/b50cd4f9c5362b7b8470d731a751891c63abf75d47dcf2e97857d75ce77e480f.jpg)

![c6b05ebb4f00e196a5f511818d0d03bbbbabbc4aa51069225d310925ec538a6d.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/c6b05ebb4f00e196a5f511818d0d03bbbbabbc4aa51069225d310925ec538a6d.jpg)

![e211457e7c4e66c1664c3fb12520d83d3ca828433984bba155533cd5e0ce0116.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/e211457e7c4e66c1664c3fb12520d83d3ca828433984bba155533cd5e0ce0116.jpg)

![e61059922c976485082325ee88bc1f7dc902a1412bb6a1b9d372d289a89600b5.jpg](../icml_results/391_Learning%20with%20Admissibility_%20Robust%20Fuzzy%20Hashing%20for%20Cross-Modal%20Retrieval%20with%20Noisy%20Labels/tables/e61059922c976485082325ee88bc1f7dc902a1412bb6a1b9d372d289a89600b5.jpg)

<a id="paper-0384"></a>

## ADEPT: RL-Aligned Agentic Decoding of Emotion via Evidence Probing Tools — From Consensus Learning to Ambiguity-Driven Emotion Reasoning


### Images

![06f624c72992606ab99bf750745b25fcf2dd5fb39da1c55549f5e6687caffbe0.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/06f624c72992606ab99bf750745b25fcf2dd5fb39da1c55549f5e6687caffbe0.jpg)

![111df133b304f8e5faec61264a032ce5ce646af7719378239ff831573b9d8740.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/111df133b304f8e5faec61264a032ce5ce646af7719378239ff831573b9d8740.jpg)

![11c0cf3885e3cb1093b7542ef1f78570ec78e49529ae47b0c2af1c01e5db200c.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/11c0cf3885e3cb1093b7542ef1f78570ec78e49529ae47b0c2af1c01e5db200c.jpg)

![627710041bed19a3527cc1bd3e901cd7df11d5ce1775decbf3719fca332b08c5.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/627710041bed19a3527cc1bd3e901cd7df11d5ce1775decbf3719fca332b08c5.jpg)

![640b12663c1aaea758dc76aa542d223973d137446a805dfd1111796e6699d424.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/640b12663c1aaea758dc76aa542d223973d137446a805dfd1111796e6699d424.jpg)

![70754f117a94e8d1c182a803c65a3dcd4a47914a98eb6d6c91ebbe032196e4bd.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/70754f117a94e8d1c182a803c65a3dcd4a47914a98eb6d6c91ebbe032196e4bd.jpg)

![b26d3d40f4b0d28bedd77caedb8f33c09bd754b8a9fdaeba39d07daf9397b59d.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/b26d3d40f4b0d28bedd77caedb8f33c09bd754b8a9fdaeba39d07daf9397b59d.jpg)

![b6fc5d170e85ed6314b41e5c58325ba854ed3125101ff2530c418b48d22244c3.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/b6fc5d170e85ed6314b41e5c58325ba854ed3125101ff2530c418b48d22244c3.jpg)

![c2cd95ce9e84b7b4f1cf23b2b30e47e44375480b114b9bd3194b3c0993a487c9.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/c2cd95ce9e84b7b4f1cf23b2b30e47e44375480b114b9bd3194b3c0993a487c9.jpg)

![dc5a5789edd75380da41c364acdffcbfd435d44077019243068c420e94ba4342.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/dc5a5789edd75380da41c364acdffcbfd435d44077019243068c420e94ba4342.jpg)

![deda328212815fee6cf452a5c3f6d39c33a1f073a833eda8f380eba4cc752c98.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/deda328212815fee6cf452a5c3f6d39c33a1f073a833eda8f380eba4cc752c98.jpg)

![f509ae9b06d78fe7459cf973d6e8b61dc27918898640d390d993e9e05e864f49.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/f509ae9b06d78fe7459cf973d6e8b61dc27918898640d390d993e9e05e864f49.jpg)

![fc888572433d7f49ec53832dd774e11cdb997c18c8985f49fb6f2d2310ef6bd0.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/images/fc888572433d7f49ec53832dd774e11cdb997c18c8985f49fb6f2d2310ef6bd0.jpg)

### Tables

![0e60e34f2a2d6fde6e5748bebcad5c25de096c4cdc7dab83b4d3bc49370e0c51.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/0e60e34f2a2d6fde6e5748bebcad5c25de096c4cdc7dab83b4d3bc49370e0c51.jpg)

![106afdb258b344c94dd734df1f95d973b4131e848d42d347b967ed603a0d5f65.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/106afdb258b344c94dd734df1f95d973b4131e848d42d347b967ed603a0d5f65.jpg)

![3ceb25b1186b5cbc5a8ef12b2db819df0860eba54a7977f23f4f97c8655581a6.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/3ceb25b1186b5cbc5a8ef12b2db819df0860eba54a7977f23f4f97c8655581a6.jpg)

![496415ec6d4e2eb2ed3645b88d3e86772fb608a630a558f2091785f361b288cb.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/496415ec6d4e2eb2ed3645b88d3e86772fb608a630a558f2091785f361b288cb.jpg)

![5c245a978e291d97549a9b53a502af884fe6b682184320de776027d7d78f75de.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/5c245a978e291d97549a9b53a502af884fe6b682184320de776027d7d78f75de.jpg)

![6912c2b8d7ec96827577086a7c95aa536ad918d4a7b6613a7c52269efe62db7f.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/6912c2b8d7ec96827577086a7c95aa536ad918d4a7b6613a7c52269efe62db7f.jpg)

![9acde0e6fff84b92304a4c2beb8ed17fd1c7361c1c80ad8565eb8c27a34d5156.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/9acde0e6fff84b92304a4c2beb8ed17fd1c7361c1c80ad8565eb8c27a34d5156.jpg)

![9c4cc54aa4c79caaeba9b830ce50454bec5c40ec055e5b76b0d0d6f33100b650.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/9c4cc54aa4c79caaeba9b830ce50454bec5c40ec055e5b76b0d0d6f33100b650.jpg)

![9e8092760eb2fe872f1336604b570e64933743dbac17819d9d241522401946a2.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/9e8092760eb2fe872f1336604b570e64933743dbac17819d9d241522401946a2.jpg)

![a65c46aa1800a0f69982c6cb424046e4d1b5c111df7c7d470f59cad909e387b2.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/a65c46aa1800a0f69982c6cb424046e4d1b5c111df7c7d470f59cad909e387b2.jpg)

![d2d424c4eb8c5b1434417d3e471af52e52ab383eafb2796ad705724eb54d66fe.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/d2d424c4eb8c5b1434417d3e471af52e52ab383eafb2796ad705724eb54d66fe.jpg)

![fce77cff7d6781859cec5e52a4267660852255764e7cd1b2b657f2933b056d3a.jpg](../icml_results/392_ADEPT_%20RL-Aligned%20Agentic%20Decoding%20of%20Emotion%20via%20Evidence%20Probing%20Tools%20%E2%80%94%20From%20Consensus%20Learning%20t/tables/fce77cff7d6781859cec5e52a4267660852255764e7cd1b2b657f2933b056d3a.jpg)

<a id="paper-0385"></a>

## Orthogonal Concept Erasure for Diffusion Models


### Images

![52b6f0f8c2a0a902d7269fbaa5c0a522a0872c4fb7cefee821b19098791be0e2.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/52b6f0f8c2a0a902d7269fbaa5c0a522a0872c4fb7cefee821b19098791be0e2.jpg)

![5af26ab7060a89a2277fce6aed02a34e342c638d9fdb3a559a87aca77cbc4a81.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/5af26ab7060a89a2277fce6aed02a34e342c638d9fdb3a559a87aca77cbc4a81.jpg)

![5dd52c4125db8aabc863ee8058d08c35344709c029da111fdaa0adc6d572119e.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/5dd52c4125db8aabc863ee8058d08c35344709c029da111fdaa0adc6d572119e.jpg)

![8c62b2284f5e2508ccb6af2527a4abdef0768bb929d7f2d3f147263c3f9d01c1.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/8c62b2284f5e2508ccb6af2527a4abdef0768bb929d7f2d3f147263c3f9d01c1.jpg)

![cd51ea930124277ea2eefc53615717849ce11329911fc9f1157546b182a98da4.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/cd51ea930124277ea2eefc53615717849ce11329911fc9f1157546b182a98da4.jpg)

![dbc558c51b9199d458d9a6e247a4069d8a4d530140668f783ae0f8d91819a95a.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/dbc558c51b9199d458d9a6e247a4069d8a4d530140668f783ae0f8d91819a95a.jpg)

![f080d829ffc68fa9b958e0acd5f2e822392dce607e4676151c7921c7be6ea0ea.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/f080d829ffc68fa9b958e0acd5f2e822392dce607e4676151c7921c7be6ea0ea.jpg)

![f1af68fa4a5b44124a3f91d915640db57ae7c414c6e74359272ae5e439b01b94.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/f1af68fa4a5b44124a3f91d915640db57ae7c414c6e74359272ae5e439b01b94.jpg)

![f505b56d21c0dd07b118a975dba9f33f90144c63e45905edda737e93c891f991.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/images/f505b56d21c0dd07b118a975dba9f33f90144c63e45905edda737e93c891f991.jpg)

### Tables

![20f26236d79c0050a7f1784c866844d9553d273cea1c04624d752744b2f6484c.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/20f26236d79c0050a7f1784c866844d9553d273cea1c04624d752744b2f6484c.jpg)

![3c32d6778cecab5a6e9227f368205d8aee9c52d9d7c5cdaec2d4809585c86c62.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/3c32d6778cecab5a6e9227f368205d8aee9c52d9d7c5cdaec2d4809585c86c62.jpg)

![4c972fce3e636df40237110bbb9bbacf782ced3cb4f5a4355c120c6aacc9c199.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/4c972fce3e636df40237110bbb9bbacf782ced3cb4f5a4355c120c6aacc9c199.jpg)

![60100b37ef4f66731995330bbbdfa79297d509eaa28eb06e294963438f2d4486.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/60100b37ef4f66731995330bbbdfa79297d509eaa28eb06e294963438f2d4486.jpg)

![727c649ef8c5e3a3347b15fd8a7f81a1db6f96e48b1c86b891cbf999cdc8cd89.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/727c649ef8c5e3a3347b15fd8a7f81a1db6f96e48b1c86b891cbf999cdc8cd89.jpg)

![8e5b7668c86e1751580234bd24270cc77fa5299b4ea772c9d0b79683ad908a99.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/8e5b7668c86e1751580234bd24270cc77fa5299b4ea772c9d0b79683ad908a99.jpg)

![9af3e34839644d4ffc6850ef76ccbd0eba0ca644f3768039b2883dec58c92dc5.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/9af3e34839644d4ffc6850ef76ccbd0eba0ca644f3768039b2883dec58c92dc5.jpg)

![a354f568939192dfc285ef2971ab29cf10da8b27bcc7458b02fe10f9718527f9.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/a354f568939192dfc285ef2971ab29cf10da8b27bcc7458b02fe10f9718527f9.jpg)

![ca992e75ac5f04dadbe8269b0a62157316b3a0e35d2ee1d82786b824ad6f2784.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/ca992e75ac5f04dadbe8269b0a62157316b3a0e35d2ee1d82786b824ad6f2784.jpg)

![d270147725c15101bb1c2a1fe08ac592a430e65d2ff460fcca21dc71ffbcd821.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/d270147725c15101bb1c2a1fe08ac592a430e65d2ff460fcca21dc71ffbcd821.jpg)

![d8d76325c94bf7c8c6706b58cd95a68fca1eba8cf3430fccae28d1e5ebd8ee2d.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/d8d76325c94bf7c8c6706b58cd95a68fca1eba8cf3430fccae28d1e5ebd8ee2d.jpg)

![e5eb2795032a8cf57c68372318fc7d12c9b6f91aa2121bff5918e42308980234.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/e5eb2795032a8cf57c68372318fc7d12c9b6f91aa2121bff5918e42308980234.jpg)

![f6131a2c930532cfa97a04ec7f8b582a290e944fdc5f5e91c74ff75c46cd7dcd.jpg](../icml_results/393_Orthogonal%20Concept%20Erasure%20for%20Diffusion%20Models/tables/f6131a2c930532cfa97a04ec7f8b582a290e944fdc5f5e91c74ff75c46cd7dcd.jpg)

<a id="paper-0386"></a>

## Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability


### Images

![28686c3c02efb4c5ae551020e27d2b8c5a342f6056861250796e6e4a06f0503b.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/28686c3c02efb4c5ae551020e27d2b8c5a342f6056861250796e6e4a06f0503b.jpg)

![39ed2c44bd31d58a003b3849a0be9ba7515fec967f907f8d6381c37068aa1f84.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/39ed2c44bd31d58a003b3849a0be9ba7515fec967f907f8d6381c37068aa1f84.jpg)

![3a54c2ce50ef90e0bda5964d1ad9dda58766dcba8f674ad6c075d35fc0c621c5.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/3a54c2ce50ef90e0bda5964d1ad9dda58766dcba8f674ad6c075d35fc0c621c5.jpg)

![470a27e6d47aac8e7ee386b8a85377d60a6ac25e226d8378ed968c7f6aaa1df7.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/470a27e6d47aac8e7ee386b8a85377d60a6ac25e226d8378ed968c7f6aaa1df7.jpg)

![4a357318518ec20ba6b2f10de930bc540224da8ab5903c4cd2a1b70b4defe162.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/4a357318518ec20ba6b2f10de930bc540224da8ab5903c4cd2a1b70b4defe162.jpg)

![4b3ede943179702afc8c0d421197c5ec0c95d482762e573e785b2b65f6566688.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/4b3ede943179702afc8c0d421197c5ec0c95d482762e573e785b2b65f6566688.jpg)

![55bced81f800470a830479643afee020c5d06ee9f08db25f27f53959fbe0f22d.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/55bced81f800470a830479643afee020c5d06ee9f08db25f27f53959fbe0f22d.jpg)

![6c9c2c2292c757928de3c2698813832a6dedacb0fcf824f7820e83b675aa671a.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/6c9c2c2292c757928de3c2698813832a6dedacb0fcf824f7820e83b675aa671a.jpg)

![6fa7fe999271220381e3b3b9f339bc0bc596aa335c93c563fd1faff0a82a124a.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/6fa7fe999271220381e3b3b9f339bc0bc596aa335c93c563fd1faff0a82a124a.jpg)

![73897db118159be5d209fda3a917321c11629dbc35ce2953fbd4e276a4380e98.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/73897db118159be5d209fda3a917321c11629dbc35ce2953fbd4e276a4380e98.jpg)

![7d7412f4b346273b9ef34613ba501252048595dcd6390f6f86ca0d72e6069988.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/7d7412f4b346273b9ef34613ba501252048595dcd6390f6f86ca0d72e6069988.jpg)

![9057066d4f98684fdb17e4f1955c2321f8d0460d9a8522648e8ee9279ea100e7.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/9057066d4f98684fdb17e4f1955c2321f8d0460d9a8522648e8ee9279ea100e7.jpg)

![b08c9ade9638dcd1f92f9b945d04f89960a31e68bc906010562ef5aecdf357dc.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/b08c9ade9638dcd1f92f9b945d04f89960a31e68bc906010562ef5aecdf357dc.jpg)

![b3470df4c8291a6115ec0bebc94ccd055180277ca6307b48e66aee1fd34b7bb6.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/b3470df4c8291a6115ec0bebc94ccd055180277ca6307b48e66aee1fd34b7bb6.jpg)

![b3eddf5a47549ee6ab57d3bd607e70687845a86202604db9616ee5c51b08e695.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/b3eddf5a47549ee6ab57d3bd607e70687845a86202604db9616ee5c51b08e695.jpg)

![b5a577e4b4bd13ba5c1887be7aec9e3e3c9a9e9f33aa799f9bf4cc0a1ec1a437.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/b5a577e4b4bd13ba5c1887be7aec9e3e3c9a9e9f33aa799f9bf4cc0a1ec1a437.jpg)

![bfd58bb67685999631c8efde9a45f8016632325cdbe769d26ce256d37ea8a713.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/bfd58bb67685999631c8efde9a45f8016632325cdbe769d26ce256d37ea8a713.jpg)

![cd93553cf6e7ab7bd631bfa2487204b9c8d346388f0e0edfd6bad427d67d5cb3.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/cd93553cf6e7ab7bd631bfa2487204b9c8d346388f0e0edfd6bad427d67d5cb3.jpg)

![e51b9e8bced0a58b32f3fe7ce7b6680ae610db7a528be94cbbd8cdbf64a29061.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/e51b9e8bced0a58b32f3fe7ce7b6680ae610db7a528be94cbbd8cdbf64a29061.jpg)

![ed0f47dccb1d01a7057bb69e826c9d73320da6ca8e66f031db6aaf1843c0ffb3.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/images/ed0f47dccb1d01a7057bb69e826c9d73320da6ca8e66f031db6aaf1843c0ffb3.jpg)

### Tables

![0a26f0a87f80e046a434c7e81e77e30d7944565a510358fbc306db8e68b963db.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/0a26f0a87f80e046a434c7e81e77e30d7944565a510358fbc306db8e68b963db.jpg)

![4a52f0e935a9e2742ca17c809121db781f7aecacce8d1696dd47c7fb650f3b3b.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/4a52f0e935a9e2742ca17c809121db781f7aecacce8d1696dd47c7fb650f3b3b.jpg)

![4b241f2c7d1fafa3fe098e2adcec04c8ff2630e4d7687501da6f42342c5b7377.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/4b241f2c7d1fafa3fe098e2adcec04c8ff2630e4d7687501da6f42342c5b7377.jpg)

![5224937c629b384ed005340b94fe69a9f649cf7c88a3d0dd682093a3bd5a2b84.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/5224937c629b384ed005340b94fe69a9f649cf7c88a3d0dd682093a3bd5a2b84.jpg)

![62bc41bc4eb0fd7536fcca3c238e9a727733567c6a478a2f3e08772e28241f93.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/62bc41bc4eb0fd7536fcca3c238e9a727733567c6a478a2f3e08772e28241f93.jpg)

![68c2b923773a3f6db4b29a2a20d3e777cdef7fd5d7e84eb93814cca4f7cbf3f1.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/68c2b923773a3f6db4b29a2a20d3e777cdef7fd5d7e84eb93814cca4f7cbf3f1.jpg)

![71cd26db210c28964bbf549f04bb4983fbf101639479a74c74aa2af270f1fefa.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/71cd26db210c28964bbf549f04bb4983fbf101639479a74c74aa2af270f1fefa.jpg)

![9f55f108dc371bd3ce22468e63307d0ed065ad274c8d5857ebb29743525ce6c1.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/9f55f108dc371bd3ce22468e63307d0ed065ad274c8d5857ebb29743525ce6c1.jpg)

![e4813a67659517722bcbcfe1e1857783d9e320f0fbe18709c7ae64284dbb8dda.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/e4813a67659517722bcbcfe1e1857783d9e320f0fbe18709c7ae64284dbb8dda.jpg)

![eb9a765570ec990a638d1c15796a62dbd75c373e8ccbfa8c59c259deedb17aa0.jpg](../icml_results/394_Teaching%20Models%20to%20Teach%20Themselves_%20Reasoning%20at%20the%20Edge%20of%20Learnability/tables/eb9a765570ec990a638d1c15796a62dbd75c373e8ccbfa8c59c259deedb17aa0.jpg)

<a id="paper-0387"></a>

## Maximum Likelihood Reinforcement Learning


### Images

![19b48b268cfec520d501ed5c59fea825c9bcb447b7389259f6671e01c1e68044.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/19b48b268cfec520d501ed5c59fea825c9bcb447b7389259f6671e01c1e68044.jpg)

![2fef659d5622e0d05a746f70f9ab335e5490382230055613db6d97539e8402d7.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/2fef659d5622e0d05a746f70f9ab335e5490382230055613db6d97539e8402d7.jpg)

![4da923e9a8871463417c03433e129824ac0891e212667e8a7faa363cf23fcac5.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/4da923e9a8871463417c03433e129824ac0891e212667e8a7faa363cf23fcac5.jpg)

![594330915ca914fefffb56915ee905ca9a6d95f4a60515a92a079d78742b47ed.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/594330915ca914fefffb56915ee905ca9a6d95f4a60515a92a079d78742b47ed.jpg)

![5cf9cbce3e8e32afe861ceba6f3f27a4b7a50910b012d67d7d1dbb093e348b7a.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/5cf9cbce3e8e32afe861ceba6f3f27a4b7a50910b012d67d7d1dbb093e348b7a.jpg)

![5da2712f73d87b4fe6f5d8c09b293f29a96c7f1f0f154d02ff8399bccbc23ada.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/5da2712f73d87b4fe6f5d8c09b293f29a96c7f1f0f154d02ff8399bccbc23ada.jpg)

![62e5886a0c075ebc70f64a3f2a9a35ef9578a0d9e1757a18c28eb5155a25e0cd.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/62e5886a0c075ebc70f64a3f2a9a35ef9578a0d9e1757a18c28eb5155a25e0cd.jpg)

![6de9cdaccd4203b236d995aa9ef9b8d64942fb5f43a8d05652f828f239737526.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/6de9cdaccd4203b236d995aa9ef9b8d64942fb5f43a8d05652f828f239737526.jpg)

![70fd8bb3a1698178b6caba5fd3a42885713160dc29e735aafee5a240af1fdc11.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/70fd8bb3a1698178b6caba5fd3a42885713160dc29e735aafee5a240af1fdc11.jpg)

![7396062be28d97d496c9339f3b11dd0b30573874ff4d60d723e7bb56d5e15ce9.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/7396062be28d97d496c9339f3b11dd0b30573874ff4d60d723e7bb56d5e15ce9.jpg)

![7ce51e13ae630894dcfb5621e2e47045d8852008bf4e4b740a745951f58a9971.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/7ce51e13ae630894dcfb5621e2e47045d8852008bf4e4b740a745951f58a9971.jpg)

![87117dd4b71c6f4d9dadda5bf481dcdf5e16b283cae40acb4b4218368e776cf7.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/87117dd4b71c6f4d9dadda5bf481dcdf5e16b283cae40acb4b4218368e776cf7.jpg)

![8818c006669167378ead685c49aaf9f27372dbed4530cc1affa455134ec32863.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/8818c006669167378ead685c49aaf9f27372dbed4530cc1affa455134ec32863.jpg)

![9525b2736397afa518e7f4e3a4eba0f4c5974245753a0e17893776141895f6cc.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/9525b2736397afa518e7f4e3a4eba0f4c5974245753a0e17893776141895f6cc.jpg)

![9a2060d70cedf5f47953f5a16e622b3c57b8797f3b8481a90273a0f30b5aa72d.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/9a2060d70cedf5f47953f5a16e622b3c57b8797f3b8481a90273a0f30b5aa72d.jpg)

![a225be7408831ff9b20d10d11fd8a4af0ade39647b44452611fc5f9d1ed3807f.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/a225be7408831ff9b20d10d11fd8a4af0ade39647b44452611fc5f9d1ed3807f.jpg)

![a5e15206eafd8c9015187106059c2aa746d2c5b5e683842763dc8a275e6cde5e.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/a5e15206eafd8c9015187106059c2aa746d2c5b5e683842763dc8a275e6cde5e.jpg)

![b1ddcb3d3d6fc18c93e77bd5ace7cd5ea0bb96dddccff7ca884ac41a114fa04b.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/b1ddcb3d3d6fc18c93e77bd5ace7cd5ea0bb96dddccff7ca884ac41a114fa04b.jpg)

![c2fed4d76ed1a2f892188bd8035b6cc6c158a3dd307385bb264d97e3071f9484.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/c2fed4d76ed1a2f892188bd8035b6cc6c158a3dd307385bb264d97e3071f9484.jpg)

![d3b3a580ce20c6988cdca6fc9b421ca7b8a44c1ee57e9b13d02985c231fbec29.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/d3b3a580ce20c6988cdca6fc9b421ca7b8a44c1ee57e9b13d02985c231fbec29.jpg)

![d7cc5952c6099bde29abdf2cfc63f538c4718b8cbf385272e304449b76556b91.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/d7cc5952c6099bde29abdf2cfc63f538c4718b8cbf385272e304449b76556b91.jpg)

![dd329b0f6b5d43e8f1bbdaf9ae3298757c53a7c9af9083962ae3271d0f9170df.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/dd329b0f6b5d43e8f1bbdaf9ae3298757c53a7c9af9083962ae3271d0f9170df.jpg)

![dffb9c9da0b9a4b6764def9ec5a592215f971ac8862b680bc8f2f974428eca4a.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/dffb9c9da0b9a4b6764def9ec5a592215f971ac8862b680bc8f2f974428eca4a.jpg)

![e5e1ad830d2b218a4738a308cb302056f2687592ba926cf87685cda0d379f78b.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/e5e1ad830d2b218a4738a308cb302056f2687592ba926cf87685cda0d379f78b.jpg)

![ec4ae63cb08f40d30015c42ebd24906c3bb1f0eb3d830142c5f497ff395ee545.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/ec4ae63cb08f40d30015c42ebd24906c3bb1f0eb3d830142c5f497ff395ee545.jpg)

![f01ac21c92efb802048623c1376938c2fd42ba707a7b8a3c13d1dd811f20eb4a.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/f01ac21c92efb802048623c1376938c2fd42ba707a7b8a3c13d1dd811f20eb4a.jpg)

![f049aaa6323aef43c38ae1654c892d3dcda3da4ea38dbde516fab837079fea68.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/f049aaa6323aef43c38ae1654c892d3dcda3da4ea38dbde516fab837079fea68.jpg)

![f488e8c54b82ab8c694c08ba201997beb14eff59b3a02e644d2df5b87c8ee8d7.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/f488e8c54b82ab8c694c08ba201997beb14eff59b3a02e644d2df5b87c8ee8d7.jpg)

![f62e6bef3dfcb4b3497fb588eb07b6c6671bbb95f913dc268b38ef197b503e67.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/f62e6bef3dfcb4b3497fb588eb07b6c6671bbb95f913dc268b38ef197b503e67.jpg)

![f93923ce6a125ceec94a76e944275b0c41603f1e94753b8056988a7dd578efb1.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/f93923ce6a125ceec94a76e944275b0c41603f1e94753b8056988a7dd578efb1.jpg)

![fb8f81fc19b4794efde7704b16c292dcb58b0d6ed0ed6fe7d367c46e37c759ff.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/images/fb8f81fc19b4794efde7704b16c292dcb58b0d6ed0ed6fe7d367c46e37c759ff.jpg)

### Tables

![079687e78038da364eabeb90acc68ac61a5c2342face0505798b6a6cb0ad5962.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/079687e78038da364eabeb90acc68ac61a5c2342face0505798b6a6cb0ad5962.jpg)

![1a8d9fcb493ae2861c0280bb91f0924a4920084b7786a3c90bb59bbca23aa377.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/1a8d9fcb493ae2861c0280bb91f0924a4920084b7786a3c90bb59bbca23aa377.jpg)

![33c6839f5a85420fe722770490df173e288f55534c0e6db843c6d63da1123377.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/33c6839f5a85420fe722770490df173e288f55534c0e6db843c6d63da1123377.jpg)

![55cf4f1a95bc0a3b294a8df1c17f622082b0d15a905ecba3f0d6538b32395644.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/55cf4f1a95bc0a3b294a8df1c17f622082b0d15a905ecba3f0d6538b32395644.jpg)

![653ae4a16e122c6eaa1635a0b3a4baddd7206f15be85db47bcd7a6a37f9e5d73.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/653ae4a16e122c6eaa1635a0b3a4baddd7206f15be85db47bcd7a6a37f9e5d73.jpg)

![8a1a879e8d3621f2aaa6a86ae308c42ddb156b31bc0fb9adeaf7d8cbf087625f.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/8a1a879e8d3621f2aaa6a86ae308c42ddb156b31bc0fb9adeaf7d8cbf087625f.jpg)

![b24b66fe195786f71df69402c474ca7bc27d9bbf3ca9b80a35c1a43e35f7b60b.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/b24b66fe195786f71df69402c474ca7bc27d9bbf3ca9b80a35c1a43e35f7b60b.jpg)

![b7e6831d50547d11a46db56d0d9810f4cf4f9c5d26b39bb1f0c995caece841ef.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/b7e6831d50547d11a46db56d0d9810f4cf4f9c5d26b39bb1f0c995caece841ef.jpg)

![c94d80272dba065ba093bff93028b16d675ff639de5a4c2a2f33fa781f580ccc.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/c94d80272dba065ba093bff93028b16d675ff639de5a4c2a2f33fa781f580ccc.jpg)

![de8bde8d0f37e433319a07a3d1f07d3c27086d8fafc4bd6823ce4f8a2dd57963.jpg](../icml_results/395_Maximum%20Likelihood%20Reinforcement%20Learning/tables/de8bde8d0f37e433319a07a3d1f07d3c27086d8fafc4bd6823ce4f8a2dd57963.jpg)

<a id="paper-0388"></a>

## The Value of Variance: Mitigating Debate Collapse in Multi-Agent Systems via Uncertainty-Driven Policy Optimization


### Images

![2c59c64d07001e06719ef3408de662da83269a9d1ce4c7ed7351d5c2f4a416f5.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/2c59c64d07001e06719ef3408de662da83269a9d1ce4c7ed7351d5c2f4a416f5.jpg)

![38475b539f59428d05bbfaf79dd6920ff5f4ff560567ca731f892a6d01099e77.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/38475b539f59428d05bbfaf79dd6920ff5f4ff560567ca731f892a6d01099e77.jpg)

![39b47aedf8a0a17024fa21ed2fa841f2852e4855baa7f8e3913d814094cfca8f.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/39b47aedf8a0a17024fa21ed2fa841f2852e4855baa7f8e3913d814094cfca8f.jpg)

![4af6f2615b1c479f3d30a404cb377c1d99e5cc10cab6e3b19e3e7a7f5a920539.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/4af6f2615b1c479f3d30a404cb377c1d99e5cc10cab6e3b19e3e7a7f5a920539.jpg)

![77d2d14dbb026dff687fd66642041047e75418460b21f8383262ee1adf37e4aa.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/77d2d14dbb026dff687fd66642041047e75418460b21f8383262ee1adf37e4aa.jpg)

![799fa7d661aa1ac4bae91cccb187836323f60a0a96c7e5fe0af92e5d1d273cea.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/799fa7d661aa1ac4bae91cccb187836323f60a0a96c7e5fe0af92e5d1d273cea.jpg)

![a5ae3fbf2727362fde69636ab4c4f4dd9a73879afe24efb0aa12d8deb4219163.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/a5ae3fbf2727362fde69636ab4c4f4dd9a73879afe24efb0aa12d8deb4219163.jpg)

![b69c1bc08421ecc955535746a1ab92f3bcf6f8985adc5876b52311fd67a52c8a.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/b69c1bc08421ecc955535746a1ab92f3bcf6f8985adc5876b52311fd67a52c8a.jpg)

![cc248776045ea51ab44e4ad537761d322d82fbccd9055e901890be967394451f.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/cc248776045ea51ab44e4ad537761d322d82fbccd9055e901890be967394451f.jpg)

![d1f5624d51c5ff4f1a2ae7651662cb43adc1c84eca62c97472812fe0b2c295fb.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/d1f5624d51c5ff4f1a2ae7651662cb43adc1c84eca62c97472812fe0b2c295fb.jpg)

![f8f19ec4f666ad7ae940b708391e9c75734f280701a9cfcddbbf5f272b65b002.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/images/f8f19ec4f666ad7ae940b708391e9c75734f280701a9cfcddbbf5f272b65b002.jpg)

### Tables

![0e6413044b142184ff1b93e78afa8e3b7abf284d20bc342864213c6d6e24d2b3.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/0e6413044b142184ff1b93e78afa8e3b7abf284d20bc342864213c6d6e24d2b3.jpg)

![5436df332e5a82d0c0127109919c4c0216f112e70bb10433f2bc9345245f4131.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/5436df332e5a82d0c0127109919c4c0216f112e70bb10433f2bc9345245f4131.jpg)

![71a41e8c78c934a752956ae166e858b050192fe957784580662fdcbd8a447dd0.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/71a41e8c78c934a752956ae166e858b050192fe957784580662fdcbd8a447dd0.jpg)

![a79cf8bb708f505780623ed4dce22ddbe8035b077d10faa64db7fe1552c840c6.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/a79cf8bb708f505780623ed4dce22ddbe8035b077d10faa64db7fe1552c840c6.jpg)

![d251d42a5aff1c4c891bdc30e75730cb6afe60b4a7d10af84d277e744a5b432b.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/d251d42a5aff1c4c891bdc30e75730cb6afe60b4a7d10af84d277e744a5b432b.jpg)

![e663a0cdc744c3e52bca7951a401c9916249a8d082683c28ad891d5f19dc760d.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/e663a0cdc744c3e52bca7951a401c9916249a8d082683c28ad891d5f19dc760d.jpg)

![e8f4d13d536e6b991168d231329b145c8c886afc17bdfb7347fa4ae063a0bf92.jpg](../icml_results/396_The%20Value%20of%20Variance_%20Mitigating%20Debate%20Collapse%20in%20Multi-Agent%20Systems%20via%20Uncertainty-Driven%20Poli/tables/e8f4d13d536e6b991168d231329b145c8c886afc17bdfb7347fa4ae063a0bf92.jpg)

<a id="paper-0389"></a>

## NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search


### Images

![47903756be8516579c2c6c89c1eb77ebbdf9facf8820f65c3499741b894bd0c2.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/images/47903756be8516579c2c6c89c1eb77ebbdf9facf8820f65c3499741b894bd0c2.jpg)

![6e050a527a32801e4447ec8201e6eb5d31e83842afa51e9d787e9be223a09bcc.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/images/6e050a527a32801e4447ec8201e6eb5d31e83842afa51e9d787e9be223a09bcc.jpg)

![a591ba7df7f6cb31c774a9e07b4dd29190ce1162185338ea9d57228a4bb4eada.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/images/a591ba7df7f6cb31c774a9e07b4dd29190ce1162185338ea9d57228a4bb4eada.jpg)

### Tables

![36e40a16a922231cebd86b299de34dd26c6ecac22181598a4f6b31738944ba11.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/36e40a16a922231cebd86b299de34dd26c6ecac22181598a4f6b31738944ba11.jpg)

![765756ad56d28d20fb36cea227c6d94ad5c0a721ec061437820107f30c5cd70e.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/765756ad56d28d20fb36cea227c6d94ad5c0a721ec061437820107f30c5cd70e.jpg)

![8375339d7da2b406cb3d2428622c929743f63016a0e4108f6bd7f510f310f020.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/8375339d7da2b406cb3d2428622c929743f63016a0e4108f6bd7f510f310f020.jpg)

![9b636ddc10666c204e10de86d7a527574bc6a0ed564287fa06441fdb2d929b76.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/9b636ddc10666c204e10de86d7a527574bc6a0ed564287fa06441fdb2d929b76.jpg)

![9b9166e4dc8d97f167008e1413704be07724ab28cb2762e5553fa7a20fc2daf7.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/9b9166e4dc8d97f167008e1413704be07724ab28cb2762e5553fa7a20fc2daf7.jpg)

![d865aef2ea8165ccdf7229aaed3246a99778ee8cd22cc8f8b6a27b25b89d3c81.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/d865aef2ea8165ccdf7229aaed3246a99778ee8cd22cc8f8b6a27b25b89d3c81.jpg)

![f49f1ba53fc310161595b427fbd151a51282a6dff4fde9907404924289e9185f.jpg](../icml_results/397_NonZero_%20Interaction-Guided%20Exploration%20for%20Multi-Agent%20Monte%20Carlo%20Tree%20Search/tables/f49f1ba53fc310161595b427fbd151a51282a6dff4fde9907404924289e9185f.jpg)

<a id="paper-0390"></a>

## The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes


### Images

![14173c4a3f7642562d9f40a234e2f0084668a94f1b4bbab41c1beb23c9a687da.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/14173c4a3f7642562d9f40a234e2f0084668a94f1b4bbab41c1beb23c9a687da.jpg)

![1d86005e728db816912e88f63f157ac85a9d47dec025d40e31a95cfb20ab2517.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/1d86005e728db816912e88f63f157ac85a9d47dec025d40e31a95cfb20ab2517.jpg)

![22062fcd0f01179a413fc5861349a7d60b3ec860919e14bc2f719fd249da47b4.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/22062fcd0f01179a413fc5861349a7d60b3ec860919e14bc2f719fd249da47b4.jpg)

![2a8b1ce32a6405ee86016a0cc7d74a88215f7f656e9a6e1a0aaa02bc4806e58a.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/2a8b1ce32a6405ee86016a0cc7d74a88215f7f656e9a6e1a0aaa02bc4806e58a.jpg)

![347f10b391c25d48014f1ad0457bf4ec5afca55e22a93ba4dff80b769c244e08.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/347f10b391c25d48014f1ad0457bf4ec5afca55e22a93ba4dff80b769c244e08.jpg)

![8443836f76e8d8db331bdff82ab952294db84a4ca4c7591c25a169853cd06d4e.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/8443836f76e8d8db331bdff82ab952294db84a4ca4c7591c25a169853cd06d4e.jpg)

![94ad606bd278ee228d3518916ce4c64409388f20a30febcca8a62f4b7bd42e08.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/94ad606bd278ee228d3518916ce4c64409388f20a30febcca8a62f4b7bd42e08.jpg)

![a561a78631914b2f2d2c10b87f8726c4c70512207b2c2d6b7fa28010d3c414b5.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/a561a78631914b2f2d2c10b87f8726c4c70512207b2c2d6b7fa28010d3c414b5.jpg)

![a757643847799909d5270debb9254627484135f81ea60b0abf605439a053a041.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/a757643847799909d5270debb9254627484135f81ea60b0abf605439a053a041.jpg)

![b04eea5253e398967a4fb2409221d5378403db999a4dcab99d3e08c284dd66f7.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/b04eea5253e398967a4fb2409221d5378403db999a4dcab99d3e08c284dd66f7.jpg)

![b0a28246f654fefdb6e3906bfd1545fe555d28031c191e76781c6f97fe0fd3cf.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/b0a28246f654fefdb6e3906bfd1545fe555d28031c191e76781c6f97fe0fd3cf.jpg)

![c1f17e17871f94bd03c328f33a1525be167b741009f485d89cf0a01a43bdd4af.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/c1f17e17871f94bd03c328f33a1525be167b741009f485d89cf0a01a43bdd4af.jpg)

![cde32f0c8040d514a2d8420da3b0e03611eefaa9bf8ee60a923b202c09c4c4e8.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/cde32f0c8040d514a2d8420da3b0e03611eefaa9bf8ee60a923b202c09c4c4e8.jpg)

![cf6bf1e6b6518905689ed910b7264c88f4adc29bee93b0fddd2ae1e781e270a8.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/cf6bf1e6b6518905689ed910b7264c88f4adc29bee93b0fddd2ae1e781e270a8.jpg)

![d50738d4ec3e3c706f57d74690ec839c1931e8178481ade8eac86a7c1257da5f.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/d50738d4ec3e3c706f57d74690ec839c1931e8178481ade8eac86a7c1257da5f.jpg)

![dd08105b58fac157617804955c4338705fa8f012985d630d5ed41949a37ff36f.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/dd08105b58fac157617804955c4338705fa8f012985d630d5ed41949a37ff36f.jpg)

![e194b21a9931e78317292f8e2ec2f98799e7803f1d9e12b573f4fa99a940d0ad.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/e194b21a9931e78317292f8e2ec2f98799e7803f1d9e12b573f4fa99a940d0ad.jpg)

![fa237e838d2694126f46ce4a3006dd9423316a049d496a2219840445ab8ea975.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/images/fa237e838d2694126f46ce4a3006dd9423316a049d496a2219840445ab8ea975.jpg)

### Tables

![05df95f4ecd127d799e0a7e29252108e11367cb7a82279a3b8d5aa9a241b914d.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/05df95f4ecd127d799e0a7e29252108e11367cb7a82279a3b8d5aa9a241b914d.jpg)

![0b5ffd1126c5f05af94e27eb1e68e15d9bdc843eec6bf00ecc91c536a8444f72.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/0b5ffd1126c5f05af94e27eb1e68e15d9bdc843eec6bf00ecc91c536a8444f72.jpg)

![1623dd2ecc35a1c9b8d65151cca39b5fb8c1bc829ee9e04e9bd63d06e1fa599e.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/1623dd2ecc35a1c9b8d65151cca39b5fb8c1bc829ee9e04e9bd63d06e1fa599e.jpg)

![1b14b8339a0f8d7fa91c3b42edf0855c3a6a384aaa82aa0947019c6e332fd27a.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/1b14b8339a0f8d7fa91c3b42edf0855c3a6a384aaa82aa0947019c6e332fd27a.jpg)

![20efa975bd596464c8393fc571a6bc23ba7535ca69a038a196f141c29b04106d.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/20efa975bd596464c8393fc571a6bc23ba7535ca69a038a196f141c29b04106d.jpg)

![3b7a192abd631e3340d8122934dbefb61df5f255a38951e770c08e763e2c4647.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/3b7a192abd631e3340d8122934dbefb61df5f255a38951e770c08e763e2c4647.jpg)

![3bb5bcd4a7a4fbeb0bb77a02c5682f5136a197904ce0293cadd0ebd97d129b0e.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/3bb5bcd4a7a4fbeb0bb77a02c5682f5136a197904ce0293cadd0ebd97d129b0e.jpg)

![3edfa725d68506b1675a7525f0090788e5f4766ef36f310ba92e7baee6817f76.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/3edfa725d68506b1675a7525f0090788e5f4766ef36f310ba92e7baee6817f76.jpg)

![4d6e8bb671fb584c1fdb1919fe3e4e9fa77688d6240b40a7d193a9f0d89785fd.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/4d6e8bb671fb584c1fdb1919fe3e4e9fa77688d6240b40a7d193a9f0d89785fd.jpg)

![532cbcfa521d08ca8096e34250b6d9530ba1483dae12b65edc9eaf61d49a7e6c.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/532cbcfa521d08ca8096e34250b6d9530ba1483dae12b65edc9eaf61d49a7e6c.jpg)

![5a92accf899bc336a9463ce300312ee8c1f5849e60bace6de025d93fd2866c59.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/5a92accf899bc336a9463ce300312ee8c1f5849e60bace6de025d93fd2866c59.jpg)

![667a64b06ae35e9ce2035bc32007d0055f6000ed51555d75901351569d6f46c5.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/667a64b06ae35e9ce2035bc32007d0055f6000ed51555d75901351569d6f46c5.jpg)

![675c9a1430c7447824a4ba1e1f61779796c3752554169d47f7792b24c3b53b69.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/675c9a1430c7447824a4ba1e1f61779796c3752554169d47f7792b24c3b53b69.jpg)

![6ebfff36d2e0a5e577e20f55dc3dbf6c39ca9a4375b14d29fccf6251b1748d07.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/6ebfff36d2e0a5e577e20f55dc3dbf6c39ca9a4375b14d29fccf6251b1748d07.jpg)

![80925e5caee540164ba98e48a225ad617fc78fcfb1f0d14201f40f15f4f615b2.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/80925e5caee540164ba98e48a225ad617fc78fcfb1f0d14201f40f15f4f615b2.jpg)

![80a806773401b9571c5f133dea33aed9afd23b9d539ab545c49d66b799dea299.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/80a806773401b9571c5f133dea33aed9afd23b9d539ab545c49d66b799dea299.jpg)

![88a76fd45f737512019978f333bd9904eb9bc2903bbf13a1986e8ec2d1a5bead.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/88a76fd45f737512019978f333bd9904eb9bc2903bbf13a1986e8ec2d1a5bead.jpg)

![8b939897a3e6c375426dde9ed01734457b0ea1483f714bf45029806e6b6f1839.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/8b939897a3e6c375426dde9ed01734457b0ea1483f714bf45029806e6b6f1839.jpg)

![985c0bce4db7a340a1c08d49dd29324559e47f140e930fec0614af2398d21fc3.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/985c0bce4db7a340a1c08d49dd29324559e47f140e930fec0614af2398d21fc3.jpg)

![a72b4e99895a90816f2725b891ea7f1d12738fdfd156a5516a8d0ed9187c95ea.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/a72b4e99895a90816f2725b891ea7f1d12738fdfd156a5516a8d0ed9187c95ea.jpg)

![a7988dcbe10c27f2f51d8db495c437076289d43d75dff48e4670f09634b9d522.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/a7988dcbe10c27f2f51d8db495c437076289d43d75dff48e4670f09634b9d522.jpg)

![bf741d146b2e0410e25b6df518d4ee8653979543e358dfdd68e8608377f0b612.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/bf741d146b2e0410e25b6df518d4ee8653979543e358dfdd68e8608377f0b612.jpg)

![c5773b2e317694e43f18792f34d55543feca941bd5a9a8ecb569df7aa9c8b969.jpg](../icml_results/398_The%20Obfuscation%20Atlas_%20Mapping%20Where%20Honesty%20Emerges%20in%20RLVR%20with%20Deception%20Probes/tables/c5773b2e317694e43f18792f34d55543feca941bd5a9a8ecb569df7aa9c8b969.jpg)

<a id="paper-0391"></a>

## Is Data Shapley Not Better than Random in Data Selection? Ask NASH


### Images

![0116fdf7abb4cc9918ecc1bafff554c7393bc2811c80f8945f74030cf7e2179f.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/0116fdf7abb4cc9918ecc1bafff554c7393bc2811c80f8945f74030cf7e2179f.jpg)

![134be4a69dbc948e2317c40648c09315729783a53b848bba138c2f9dba45e72d.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/134be4a69dbc948e2317c40648c09315729783a53b848bba138c2f9dba45e72d.jpg)

![1aecb32230fa75e0f40e599d7a10fc2b552d0e7c3063a8ec6dc23a12d66b0ebf.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/1aecb32230fa75e0f40e599d7a10fc2b552d0e7c3063a8ec6dc23a12d66b0ebf.jpg)

![285ddb1b2dcbdf5221538f2ee7ae2796d0c0f32268445ce1d8ab01ae82419fc8.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/285ddb1b2dcbdf5221538f2ee7ae2796d0c0f32268445ce1d8ab01ae82419fc8.jpg)

![2e0e6de1ba51a5b93b96322d51c66bc910864d3852c9e9de4967f845879a9f7b.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/2e0e6de1ba51a5b93b96322d51c66bc910864d3852c9e9de4967f845879a9f7b.jpg)

![30abe70b884091f0f5ecb210428c236e11d4c2ef665e81e1b3805f07f4023f86.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/30abe70b884091f0f5ecb210428c236e11d4c2ef665e81e1b3805f07f4023f86.jpg)

![413a9a086be525bcaf18e9523f11ee36a763544e9682bd88d7eaede365ab442d.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/413a9a086be525bcaf18e9523f11ee36a763544e9682bd88d7eaede365ab442d.jpg)

![48aa057b80457b4d3e7df009ce565bc2e37547d84febf4961b456c67cd7822a1.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/48aa057b80457b4d3e7df009ce565bc2e37547d84febf4961b456c67cd7822a1.jpg)

![843ff12121e2009641c071abfe26c872cb3513d42b309809f1486d9e70fbe259.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/843ff12121e2009641c071abfe26c872cb3513d42b309809f1486d9e70fbe259.jpg)

![90e36b9440e2c9b2c4de04f49f7e5476ddacb9f9e2be4f7212016114c3ddd8c8.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/90e36b9440e2c9b2c4de04f49f7e5476ddacb9f9e2be4f7212016114c3ddd8c8.jpg)

![98caca26536effeaf5ed138f2c02552baa1e90293014390de1c0f1c1252422ac.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/98caca26536effeaf5ed138f2c02552baa1e90293014390de1c0f1c1252422ac.jpg)

![b480aa2c8b7195491c15a848c2c2a6e9ee87d2b67bb0e905d0f0a418c29d93e3.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/b480aa2c8b7195491c15a848c2c2a6e9ee87d2b67bb0e905d0f0a418c29d93e3.jpg)

![d7d31a1934499e9380bfc88f8bc9159eec8f94475acfb3956d7ad4ba4c69b25a.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/d7d31a1934499e9380bfc88f8bc9159eec8f94475acfb3956d7ad4ba4c69b25a.jpg)

![d9efc4b8334f04be7728e0d62d878378735127c6148562be80865e003d583e1d.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/d9efc4b8334f04be7728e0d62d878378735127c6148562be80865e003d583e1d.jpg)

![ddb45829f86cbf142de42dda169cd8fe058373be62952f78fada62d300e07384.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/ddb45829f86cbf142de42dda169cd8fe058373be62952f78fada62d300e07384.jpg)

![e6d1420fb24548e9091f2196944c3ac35a34561809de26d6064e959b8458ffd1.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/e6d1420fb24548e9091f2196944c3ac35a34561809de26d6064e959b8458ffd1.jpg)

![fccddb7790612235dfddfbf8f9aea92bbcc44473a1ac5c66bc71c0dc932f1038.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/images/fccddb7790612235dfddfbf8f9aea92bbcc44473a1ac5c66bc71c0dc932f1038.jpg)

### Tables

![03db993bbbcc921916ef950502132471661b7ffd9fb606520bd13cfcaf7dc435.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/03db993bbbcc921916ef950502132471661b7ffd9fb606520bd13cfcaf7dc435.jpg)

![2a8bccc72d1c1b7a475221eee66c3d2ed323de1308a0462f0c90da572fe3f11e.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/2a8bccc72d1c1b7a475221eee66c3d2ed323de1308a0462f0c90da572fe3f11e.jpg)

![2b4fcb90cf497b21bc8bff2674ebdb730b9cfd09a7ef4aae6992db30ca366015.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/2b4fcb90cf497b21bc8bff2674ebdb730b9cfd09a7ef4aae6992db30ca366015.jpg)

![4f6dd7e1815b1c808c52fe17c87ab9e22c309bcbd20cbbd4fc29bf1e6c999525.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/4f6dd7e1815b1c808c52fe17c87ab9e22c309bcbd20cbbd4fc29bf1e6c999525.jpg)

![58568be4ac2b0892887192fea90ca321b68d05b6b4632207522376db0d2c16e5.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/58568be4ac2b0892887192fea90ca321b68d05b6b4632207522376db0d2c16e5.jpg)

![5b09b3416fc14a343273abccc9b7e4f8d73c60984f1882172c2f0df534cab4e2.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/5b09b3416fc14a343273abccc9b7e4f8d73c60984f1882172c2f0df534cab4e2.jpg)

![7e8239e7d8f5a0f2b0d089c65854fa0127c1cba0eac21eb34794ea3741ca7ab1.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/7e8239e7d8f5a0f2b0d089c65854fa0127c1cba0eac21eb34794ea3741ca7ab1.jpg)

![9cb3d1de62fbd3a86e00c9895605dcb52570ab027ccae76b8f3bac7d92dfa291.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/9cb3d1de62fbd3a86e00c9895605dcb52570ab027ccae76b8f3bac7d92dfa291.jpg)

![cdf78301d89fc5d197773a5a9140ae554086ec4b77898a38e595c0e38fdc4131.jpg](../icml_results/399_Is%20Data%20Shapley%20Not%20Better%20than%20Random%20in%20Data%20Selection_%20Ask%20NASH/tables/cdf78301d89fc5d197773a5a9140ae554086ec4b77898a38e595c0e38fdc4131.jpg)

<a id="paper-0392"></a>

## Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment


### Images

![0fad3ed4142385cecae979e17e3a73f5b38e884bf4a61acb4777d8704b1688eb.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/0fad3ed4142385cecae979e17e3a73f5b38e884bf4a61acb4777d8704b1688eb.jpg)

![124cd62253200d0634a6eafcf69e2fc83d363f190e47aa02f51b7b310274cc92.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/124cd62253200d0634a6eafcf69e2fc83d363f190e47aa02f51b7b310274cc92.jpg)

![2078a41649a18390b86a4a264734386b2324321f5dc47343f6ea634efb6b215d.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/2078a41649a18390b86a4a264734386b2324321f5dc47343f6ea634efb6b215d.jpg)

![2129a06be80dec7dff082eb9ada8bce236a8c960cb0847e8cf5436cae8a6e0fb.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/2129a06be80dec7dff082eb9ada8bce236a8c960cb0847e8cf5436cae8a6e0fb.jpg)

![3495aa66e1eb762f9ae7d3807ea827ae5ec4458b79787edf10a76f69e0468328.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/3495aa66e1eb762f9ae7d3807ea827ae5ec4458b79787edf10a76f69e0468328.jpg)

![41a3064d9d05924f3b93b9c8173e5c0868474f88b81a157290d8e098fa811c92.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/41a3064d9d05924f3b93b9c8173e5c0868474f88b81a157290d8e098fa811c92.jpg)

![4f2c4419b334c25eb529a8102a7e2293bc2c08da331ace287f12b47639764272.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/4f2c4419b334c25eb529a8102a7e2293bc2c08da331ace287f12b47639764272.jpg)

![5b1fbe43746006a40cc67599ad1549029790999f47d40a25e430439ee18ee98a.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/5b1fbe43746006a40cc67599ad1549029790999f47d40a25e430439ee18ee98a.jpg)

![61ea730be11c504e7f873c4526229e971d3de21fec3d1be9b6a39e10b926730e.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/61ea730be11c504e7f873c4526229e971d3de21fec3d1be9b6a39e10b926730e.jpg)

![7081dce47f41cd9ab05f526eaba1f53e3ba12d19887ed9af97b4b18a8306499a.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/7081dce47f41cd9ab05f526eaba1f53e3ba12d19887ed9af97b4b18a8306499a.jpg)

![831d01ee59d985a39964f2cce8a111b851ba810428b2049b6c1f8a50fd1b695e.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/831d01ee59d985a39964f2cce8a111b851ba810428b2049b6c1f8a50fd1b695e.jpg)

![8a4410b30067a7ac3e72e10d5e0e990c2effcc8110d22882b5635ab36b8f00e0.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/8a4410b30067a7ac3e72e10d5e0e990c2effcc8110d22882b5635ab36b8f00e0.jpg)

![8bbbf117835c8078ed38986f6c8e232a82f928d906c9f1033944d8f77f01d7b3.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/8bbbf117835c8078ed38986f6c8e232a82f928d906c9f1033944d8f77f01d7b3.jpg)

![90325df82db38b90837919e84a67de9d0a49634fd54c097b8368d5d9569e9171.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/90325df82db38b90837919e84a67de9d0a49634fd54c097b8368d5d9569e9171.jpg)

![9ad590cce062be68dc392240dfef4cbf8ed68edc9bd8a481768cae1eebde37a2.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/9ad590cce062be68dc392240dfef4cbf8ed68edc9bd8a481768cae1eebde37a2.jpg)

![a3cc0d48a6a74611c0c7bf34b662121606fc024a3339059cb597a105a6bc70eb.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/a3cc0d48a6a74611c0c7bf34b662121606fc024a3339059cb597a105a6bc70eb.jpg)

![a52453a971d90a5235d5a1e59ed58a394f7444aabf3ed5caf8c8e9c49a5fa1a0.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/a52453a971d90a5235d5a1e59ed58a394f7444aabf3ed5caf8c8e9c49a5fa1a0.jpg)

![a90524b0f076aee7de73d21ad6aac42d1e2038c80a1b0f3ec59efe655448f28f.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/a90524b0f076aee7de73d21ad6aac42d1e2038c80a1b0f3ec59efe655448f28f.jpg)

![abdae6e5089df7b5629027d8aea43698bb2896ea4e33e0a6923aed8d36d9df94.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/abdae6e5089df7b5629027d8aea43698bb2896ea4e33e0a6923aed8d36d9df94.jpg)

![aeded847b5c6574bd559e3fc6456a4a1d9d2c1f11c894ba5a859518f51fc6629.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/aeded847b5c6574bd559e3fc6456a4a1d9d2c1f11c894ba5a859518f51fc6629.jpg)

![bceb3a468a811def7e608684ade18d4361dd7bbc20c96274279178cbf52d40b5.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/bceb3a468a811def7e608684ade18d4361dd7bbc20c96274279178cbf52d40b5.jpg)

![c4b7624d8e73e92035396dd7abc7c9ef0a2947215e9934c53367f9d7282834a2.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/c4b7624d8e73e92035396dd7abc7c9ef0a2947215e9934c53367f9d7282834a2.jpg)

![d2d29d3d1b02bdab112a3e64f707d6aec3d96ca0dbd4a7320ba8390d1fff5f8c.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/d2d29d3d1b02bdab112a3e64f707d6aec3d96ca0dbd4a7320ba8390d1fff5f8c.jpg)

![e3a054586584397ea566b7dba576c0e80182da56c3f194aeb1cc9593a31cdf9c.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/e3a054586584397ea566b7dba576c0e80182da56c3f194aeb1cc9593a31cdf9c.jpg)

![f1e2b2dc6802dc211d765cd10c1412fa72c402953e0b218e7abaa5805ad4caea.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/f1e2b2dc6802dc211d765cd10c1412fa72c402953e0b218e7abaa5805ad4caea.jpg)

![f6d670b1fb69cbf43d9449f3fe07f2d3bc69023f39b5ca8440adc996aeb25d3b.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/f6d670b1fb69cbf43d9449f3fe07f2d3bc69023f39b5ca8440adc996aeb25d3b.jpg)

![fe705b61415519e170f626cd8a50c0801c0ec2cb1797beb01dec936aa7ca89cd.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/images/fe705b61415519e170f626cd8a50c0801c0ec2cb1797beb01dec936aa7ca89cd.jpg)

### Tables

![0d417b45ce6851cee5727f5afd39d75d31714ec659c91cc391b8481bc505ac13.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/0d417b45ce6851cee5727f5afd39d75d31714ec659c91cc391b8481bc505ac13.jpg)

![10e061b12c07b59494b2eef931f65a1190b7416659a7cb856ed77dde903c1514.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/10e061b12c07b59494b2eef931f65a1190b7416659a7cb856ed77dde903c1514.jpg)

![1e6fa6f7812edd43fdd8c51708ba57c314a8d5c7c2473390e3ae32bc82f9a7d0.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/1e6fa6f7812edd43fdd8c51708ba57c314a8d5c7c2473390e3ae32bc82f9a7d0.jpg)

![1f6f2390d7ecc35589e566245b3af9e7290fd07776a7ac20a1adc6ffeda7a795.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/1f6f2390d7ecc35589e566245b3af9e7290fd07776a7ac20a1adc6ffeda7a795.jpg)

![20b5c40fc94f5093d93145a1aa74c49f1efd349266f52937ee06f671a84d49e9.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/20b5c40fc94f5093d93145a1aa74c49f1efd349266f52937ee06f671a84d49e9.jpg)

![24e48fd71aaf49485f7c6ee342af82d267b455212f84d5eb8eb262a6a5021b45.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/24e48fd71aaf49485f7c6ee342af82d267b455212f84d5eb8eb262a6a5021b45.jpg)

![3574f2eb17d4e5835914d85c8e61cc8553e55bc687e2f91b9280fecfd3b15dd1.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/3574f2eb17d4e5835914d85c8e61cc8553e55bc687e2f91b9280fecfd3b15dd1.jpg)

![3ade9efdcd36d01aff8ac9f15e0df3c6f47d8fc163b433bb35a1c0f76a621f03.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/3ade9efdcd36d01aff8ac9f15e0df3c6f47d8fc163b433bb35a1c0f76a621f03.jpg)

![4ad57d6e253ea5235b289c0666dccef0b119b21e56318adfd2b0cb1a92fbedc4.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/4ad57d6e253ea5235b289c0666dccef0b119b21e56318adfd2b0cb1a92fbedc4.jpg)

![590d39a20e592800fd4e0afb0d5c1115a9c3ae75e0f826ac75d9dbde4e028375.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/590d39a20e592800fd4e0afb0d5c1115a9c3ae75e0f826ac75d9dbde4e028375.jpg)

![66ca99f40e3400fb2a7665bba36632c98f527c71cf87c47150f62f8e36a6adae.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/66ca99f40e3400fb2a7665bba36632c98f527c71cf87c47150f62f8e36a6adae.jpg)

![700eed652a2437bd224e96088628ffab7fbaf6282a4642033e456288299db638.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/700eed652a2437bd224e96088628ffab7fbaf6282a4642033e456288299db638.jpg)

![7a6ce9f5802a9d047d3d96976755f13f81a79e6fdf63b52b8f93c3b61b66d407.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/7a6ce9f5802a9d047d3d96976755f13f81a79e6fdf63b52b8f93c3b61b66d407.jpg)

![7a7ddde218f7fd6cc0dfb056850a5029d5cf07c948106aa8347e0da3316abb55.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/7a7ddde218f7fd6cc0dfb056850a5029d5cf07c948106aa8347e0da3316abb55.jpg)

![92e0d83c57f6bca89861b066271d5b34b40bfa9a22261509a0ecd3cb8f729178.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/92e0d83c57f6bca89861b066271d5b34b40bfa9a22261509a0ecd3cb8f729178.jpg)

![9640a48019a42b813dff46b47a3723515855c54107a6b793755e7bc08b40b375.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/9640a48019a42b813dff46b47a3723515855c54107a6b793755e7bc08b40b375.jpg)

![9b9b4400534376f9a78be8a3c52e5d49db84c56da3460dbd3d23609ab482bee1.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/9b9b4400534376f9a78be8a3c52e5d49db84c56da3460dbd3d23609ab482bee1.jpg)

![a99a746587404c2573222f97d4f2ccdf8a56c34fe435dd27768e9c1261d48f15.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/a99a746587404c2573222f97d4f2ccdf8a56c34fe435dd27768e9c1261d48f15.jpg)

![b9f6046c0f7e567439ddb40acd6142fb0ae4e72b93a3734362a79ebed168808b.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/b9f6046c0f7e567439ddb40acd6142fb0ae4e72b93a3734362a79ebed168808b.jpg)

![baf0b52d4b0229def29baaf76084fe131e40b4ec3d060dc683f90c5b7fad9250.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/baf0b52d4b0229def29baaf76084fe131e40b4ec3d060dc683f90c5b7fad9250.jpg)

![bc0346eb238419aab5359b3c6833cc0a9422b78d16c6266075234e323bfc8d56.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/bc0346eb238419aab5359b3c6833cc0a9422b78d16c6266075234e323bfc8d56.jpg)

![bfd28dff62f62ac32646d8aa34db5c9669f6ce8cabbd13f5557dfe275f91db5e.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/bfd28dff62f62ac32646d8aa34db5c9669f6ce8cabbd13f5557dfe275f91db5e.jpg)

![d050b70e48dcb7838c831fcea67212f58b36c0fa0ae36637bc2cd772cf2d6e8c.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/d050b70e48dcb7838c831fcea67212f58b36c0fa0ae36637bc2cd772cf2d6e8c.jpg)

![e0c4acf964c8dcfaced8d1f12267cfe4582ddf276a74d906f561b338fd523310.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/e0c4acf964c8dcfaced8d1f12267cfe4582ddf276a74d906f561b338fd523310.jpg)

![e32207e3acd8931b22943ba23550c9bfae13de6501af3f6c19501c02d729fa91.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/e32207e3acd8931b22943ba23550c9bfae13de6501af3f6c19501c02d729fa91.jpg)

![f48dcbed8704f031722ba4c1467b8bef4208fa18fa7dc8d5cf88401c5bf4fb85.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/f48dcbed8704f031722ba4c1467b8bef4208fa18fa7dc8d5cf88401c5bf4fb85.jpg)

![ff24de16c35ffc4a78a0b64c4410aefb87b6709aff8594528024891765c7a3b0.jpg](../icml_results/400_Alignment%20Pretraining_%20AI%20Discourse%20Causes%20Self-Fulfilling%20%28Mis%29alignment/tables/ff24de16c35ffc4a78a0b64c4410aefb87b6709aff8594528024891765c7a3b0.jpg)

<a id="paper-0393"></a>

## Beyond Language Modeling: An Exploration of Multimodal Pretraining


### Images

![0abf7f1cc771aafab4a30b8f48ada6509ef5d2c0088d65478921f08532a21818.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/0abf7f1cc771aafab4a30b8f48ada6509ef5d2c0088d65478921f08532a21818.jpg)

![0c8ba14c450a7d04636c11890784fd0413a80e64ac45fe8cc129aa3b1a61f759.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/0c8ba14c450a7d04636c11890784fd0413a80e64ac45fe8cc129aa3b1a61f759.jpg)

![125212fd218915f579f53d706eb54e899862663fc1260c9d7e280a2eea52de8e.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/125212fd218915f579f53d706eb54e899862663fc1260c9d7e280a2eea52de8e.jpg)

![147a4fa0efad0594b5f36c07c2084f5875f58990206a849291d9c5e51cd1aefb.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/147a4fa0efad0594b5f36c07c2084f5875f58990206a849291d9c5e51cd1aefb.jpg)

![1a862e757ab86abe2219fb11c7f16f3fdf54610616da1b5632a4df67cc692931.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/1a862e757ab86abe2219fb11c7f16f3fdf54610616da1b5632a4df67cc692931.jpg)

![1c700cb31e767148a6188fd210cbc667a457f3804104fd6a1bba1b41f2092d95.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/1c700cb31e767148a6188fd210cbc667a457f3804104fd6a1bba1b41f2092d95.jpg)

![25a5fb0fa928731da9b3adbc421f1811d7a48e15fedcb1b2feceaf1d25972d28.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/25a5fb0fa928731da9b3adbc421f1811d7a48e15fedcb1b2feceaf1d25972d28.jpg)

![28bb20b45e3fb03a01abb6d186d1c169ad11a3c855640a00da764351e8304d83.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/28bb20b45e3fb03a01abb6d186d1c169ad11a3c855640a00da764351e8304d83.jpg)

![29e5493d877f20fd2d302cbd4e33da1f375f1fb860bd8a9e32db501236f1328c.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/29e5493d877f20fd2d302cbd4e33da1f375f1fb860bd8a9e32db501236f1328c.jpg)

![2da126c43663d1089c18a4df17ec54432ed2eef13e65411c5f1a3a50c05bb10d.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/2da126c43663d1089c18a4df17ec54432ed2eef13e65411c5f1a3a50c05bb10d.jpg)

![32a096483e53355fded6758863f50a15b50f5575962a67b7e9c27022be20adcc.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/32a096483e53355fded6758863f50a15b50f5575962a67b7e9c27022be20adcc.jpg)

![354b6c66e3d7d388453fbad82c2d8ac2448bf34cc39b55e914a90dc454615721.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/354b6c66e3d7d388453fbad82c2d8ac2448bf34cc39b55e914a90dc454615721.jpg)

![379536aa98ad4069e206a539ed4b4e14b9f625b7d9c80b6020439afa1d121713.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/379536aa98ad4069e206a539ed4b4e14b9f625b7d9c80b6020439afa1d121713.jpg)

![44ace92c51539eb0ea35570dfe462c3db4bda7eb642dd43be6e0911f09b58523.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/44ace92c51539eb0ea35570dfe462c3db4bda7eb642dd43be6e0911f09b58523.jpg)

![45bdb12d975e8da56352a420f9ad81b3939d69b1136b6a30621ddfaca0e46de8.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/45bdb12d975e8da56352a420f9ad81b3939d69b1136b6a30621ddfaca0e46de8.jpg)

![5112143e0e3397200bdafdaaeef429af2d44ff35d8a7702f838a0cd7aa2521df.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/5112143e0e3397200bdafdaaeef429af2d44ff35d8a7702f838a0cd7aa2521df.jpg)

![6be600b5c6697e34337b2f0e544a808de0f81d9e27ee3b21052e38ab1467172b.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/6be600b5c6697e34337b2f0e544a808de0f81d9e27ee3b21052e38ab1467172b.jpg)

![7cf43912f00e0d028f2f37ec52771220910052af8b77048af7332d5d252d2114.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/7cf43912f00e0d028f2f37ec52771220910052af8b77048af7332d5d252d2114.jpg)

![810e34675b8fbc1508bbd391f927d5bc834d0cef8ca64d85e6b1d5ae4f89b060.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/810e34675b8fbc1508bbd391f927d5bc834d0cef8ca64d85e6b1d5ae4f89b060.jpg)

![95a0f4d758f809355d9f8aa9190a667b005efa58df6e5985129daf49892a231b.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/95a0f4d758f809355d9f8aa9190a667b005efa58df6e5985129daf49892a231b.jpg)

![968b53bfa11f4491c10619424245e3787a1be7273e968eb41d85786896dc755e.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/968b53bfa11f4491c10619424245e3787a1be7273e968eb41d85786896dc755e.jpg)

![9fb09289101cadce4bad9e1db33ba7323adaa6a1f9e970add3e9feb105630418.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/9fb09289101cadce4bad9e1db33ba7323adaa6a1f9e970add3e9feb105630418.jpg)

![b00fa66516b6b1812264b5bc909b1a201bacbcdfa9a799b5acc8eb63589ac235.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/b00fa66516b6b1812264b5bc909b1a201bacbcdfa9a799b5acc8eb63589ac235.jpg)

![b0f16a2004ba1ebaa157e80a5cec6ad480997f7968e13b096210ae24c51b4850.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/b0f16a2004ba1ebaa157e80a5cec6ad480997f7968e13b096210ae24c51b4850.jpg)

![b6b0245003a491fb7c83591faf8515999633fe303c41575372492fc03ff332a4.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/b6b0245003a491fb7c83591faf8515999633fe303c41575372492fc03ff332a4.jpg)

![b6bbffe017bb5151ee5ed81ada84f0bd04b4851ca1d465d17f10ec2c9b82d5fd.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/b6bbffe017bb5151ee5ed81ada84f0bd04b4851ca1d465d17f10ec2c9b82d5fd.jpg)

![bc6a6c81ba00a1eebd4b0db3deff131092a38253a97cb673c61f462ae47b36a6.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/bc6a6c81ba00a1eebd4b0db3deff131092a38253a97cb673c61f462ae47b36a6.jpg)

![c6371fce7c4cd4a773aa973ba1304ef69bf195f46cbfa3c3b810d54bdf6046bc.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/c6371fce7c4cd4a773aa973ba1304ef69bf195f46cbfa3c3b810d54bdf6046bc.jpg)

![d274bb82904b02b4ed5883e474863d78fdc5d9e41638c5069c36c7fa9ef0900c.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/d274bb82904b02b4ed5883e474863d78fdc5d9e41638c5069c36c7fa9ef0900c.jpg)

![dd3486037a99fae5befb1fb7529ad71567aef8f839814fdf72f7d27320cce004.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/dd3486037a99fae5befb1fb7529ad71567aef8f839814fdf72f7d27320cce004.jpg)

![e3c90d61bedbf695f074c1740f7f396030c3d4bc515a2abc6fd715735bc2dd5a.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/e3c90d61bedbf695f074c1740f7f396030c3d4bc515a2abc6fd715735bc2dd5a.jpg)

![e6cc01f4888c0282098d337b2765eb68d29bece1b8067cdf2249efba38d28331.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/e6cc01f4888c0282098d337b2765eb68d29bece1b8067cdf2249efba38d28331.jpg)

![e9a76442c9e9edc08172737e744bfe1e47fcb393bf0491163efd86f7511601fd.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/e9a76442c9e9edc08172737e744bfe1e47fcb393bf0491163efd86f7511601fd.jpg)

![f3c4aeecdb362671a3a14551fa166e277d22d5fa7fdf6e7b0b2cad55de32496a.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/f3c4aeecdb362671a3a14551fa166e277d22d5fa7fdf6e7b0b2cad55de32496a.jpg)

![f527e33d15ac456b881dbef778d2eaf4c700070d92e38e686b47166e12b73085.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/images/f527e33d15ac456b881dbef778d2eaf4c700070d92e38e686b47166e12b73085.jpg)

### Tables

![09950d3534073c424f8fdf61c19a020cabe2c129bf157c4604f30ddfb64275cc.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/09950d3534073c424f8fdf61c19a020cabe2c129bf157c4604f30ddfb64275cc.jpg)

![5a4b2d9be866b2f0e3efd3c12033f0e110af0be6bf2d9f4ddbb1cdbf7493603c.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/5a4b2d9be866b2f0e3efd3c12033f0e110af0be6bf2d9f4ddbb1cdbf7493603c.jpg)

![736de0db31d7bfae18c68b2b061ad8803391081f524faac6a0c50716e4b25e70.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/736de0db31d7bfae18c68b2b061ad8803391081f524faac6a0c50716e4b25e70.jpg)

![9fa52bbe94909f17ef03cfb0b5e97d520e65a997e498412e361bd0afea6b40f8.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/9fa52bbe94909f17ef03cfb0b5e97d520e65a997e498412e361bd0afea6b40f8.jpg)

![aa6bcd54b03ca8f63feaf5edeedcd1ad9f34aaa9bb3a4604d5ea2525f1ed2971.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/aa6bcd54b03ca8f63feaf5edeedcd1ad9f34aaa9bb3a4604d5ea2525f1ed2971.jpg)

![e85bea26e69875115f405ae1418358d78bd9c5c95432514fa5a604ace1bde7ea.jpg](../icml_results/401_Beyond%20Language%20Modeling_%20An%20Exploration%20of%20Multimodal%20Pretraining/tables/e85bea26e69875115f405ae1418358d78bd9c5c95432514fa5a604ace1bde7ea.jpg)

<a id="paper-0394"></a>

## Conservation Laws for Modern Neural Architectures


### Images

![2e9b9c66b62105977238a7f7e169e37d315d77e3d1c21cb6bc88fe616cc42976.jpg](../icml_results/402_Conservation%20Laws%20for%20Modern%20Neural%20Architectures/images/2e9b9c66b62105977238a7f7e169e37d315d77e3d1c21cb6bc88fe616cc42976.jpg)

![89927c00bd6e77fa3bc190ccd5ff022589e2f0ccaa3ecbdfacc39b92f879b79b.jpg](../icml_results/402_Conservation%20Laws%20for%20Modern%20Neural%20Architectures/images/89927c00bd6e77fa3bc190ccd5ff022589e2f0ccaa3ecbdfacc39b92f879b79b.jpg)

![cc1bfa615defbdddd5b2bba95973da812554399da3de25fd79250d16f32accfc.jpg](../icml_results/402_Conservation%20Laws%20for%20Modern%20Neural%20Architectures/images/cc1bfa615defbdddd5b2bba95973da812554399da3de25fd79250d16f32accfc.jpg)

![cf3376433382ae2cdf6fdc84924602f4cd07bb842afcf14afb6bee9e30a18717.jpg](../icml_results/402_Conservation%20Laws%20for%20Modern%20Neural%20Architectures/images/cf3376433382ae2cdf6fdc84924602f4cd07bb842afcf14afb6bee9e30a18717.jpg)

![f991ca340d276f7fc8e510ecbaef67accaa6d7032dffaab1c56d3ebca61eaee4.jpg](../icml_results/402_Conservation%20Laws%20for%20Modern%20Neural%20Architectures/images/f991ca340d276f7fc8e510ecbaef67accaa6d7032dffaab1c56d3ebca61eaee4.jpg)

<a id="paper-0395"></a>

## Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors


### Images

![2cd6937eeb5afda852b00019f27660226319a89194ddfe86a974f2bf8fe0c311.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/images/2cd6937eeb5afda852b00019f27660226319a89194ddfe86a974f2bf8fe0c311.jpg)

![548de38bca86461e56c232bf125c1dcddf7851a4d28556c64355b0ec7a1a3c3f.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/images/548de38bca86461e56c232bf125c1dcddf7851a4d28556c64355b0ec7a1a3c3f.jpg)

![7333294bd3317ff6b9586a2f01ec57adaf28bf47c7f0d6e6c276f8c5fe26fe65.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/images/7333294bd3317ff6b9586a2f01ec57adaf28bf47c7f0d6e6c276f8c5fe26fe65.jpg)

![d380f0dd515ecae7a41eb7b7a05aabcf38ffd91875f05b5683e02cc36e6968c9.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/images/d380f0dd515ecae7a41eb7b7a05aabcf38ffd91875f05b5683e02cc36e6968c9.jpg)

![eac644f2123890fcd986f474118bc70e3f0bf024c05e550e76c39916704fb399.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/images/eac644f2123890fcd986f474118bc70e3f0bf024c05e550e76c39916704fb399.jpg)

### Tables

![1280bcb95bbbfa6ae416d41de59ac87f02529f54bacda04acb14329324b4cc3a.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/1280bcb95bbbfa6ae416d41de59ac87f02529f54bacda04acb14329324b4cc3a.jpg)

![13b647001ef73f880e3b15cc69fc8ee3ec2afb3f591795260d29fdca21ae2ebe.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/13b647001ef73f880e3b15cc69fc8ee3ec2afb3f591795260d29fdca21ae2ebe.jpg)

![325a4f4bfbba5e76da9a16558f1379bf00709ae86a4f3f4383c0021beea45f2f.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/325a4f4bfbba5e76da9a16558f1379bf00709ae86a4f3f4383c0021beea45f2f.jpg)

![67461cebf2b95ffcc35fd1cc10de326774654828d5164bef6bdf5cc3ed95ca62.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/67461cebf2b95ffcc35fd1cc10de326774654828d5164bef6bdf5cc3ed95ca62.jpg)

![84495004da00fa14a093d89b641be4f16758f317267e3416d7f11aced939793e.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/84495004da00fa14a093d89b641be4f16758f317267e3416d7f11aced939793e.jpg)

![9feea5ed503acd2845f98efe3da03b99ac60f3044e3d48d0e180ecaab5bc25eb.jpg](../icml_results/403_Beyond%20ReLU_%20Bifurcation%2C%20Oversmoothing%2C%20and%20Topological%20Priors/tables/9feea5ed503acd2845f98efe3da03b99ac60f3044e3d48d0e180ecaab5bc25eb.jpg)

<a id="paper-0396"></a>

## Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings

### Images

![3748b1e0d2462e9f0b714bf6832be0fa04c628777b117b89afb1436dd37fc373.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/images/3748b1e0d2462e9f0b714bf6832be0fa04c628777b117b89afb1436dd37fc373.jpg)

![7be9d9b40a3969686f10a6773324fbc9418cb62cb84ef2d960f9860bd02f0978.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/images/7be9d9b40a3969686f10a6773324fbc9418cb62cb84ef2d960f9860bd02f0978.jpg)

![c51a5a6c69b3871ad03e2b59281377deb061e4742d3bcd57f1206c66067f9a93.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/images/c51a5a6c69b3871ad03e2b59281377deb061e4742d3bcd57f1206c66067f9a93.jpg)

![cf4874ae148e6e4afc09d5f280e927b19da8c26ae3d06aa30f21542f8b97780a.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/images/cf4874ae148e6e4afc09d5f280e927b19da8c26ae3d06aa30f21542f8b97780a.jpg)

![e328d2b4bc588ea0f792d347f0a8374346bc435821c770e66e6686d86d45e71f.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/images/e328d2b4bc588ea0f792d347f0a8374346bc435821c770e66e6686d86d45e71f.jpg)

### Tables

![06a4d45f6885c1eb1bfe0ad55e190eb1f409f9a91472ae514edc1e80164e69c5.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/06a4d45f6885c1eb1bfe0ad55e190eb1f409f9a91472ae514edc1e80164e69c5.jpg)

![200974cebf5f0c2493d7b091f5441ba0b56850f2cb2e910528cdee70e71800e5.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/200974cebf5f0c2493d7b091f5441ba0b56850f2cb2e910528cdee70e71800e5.jpg)

![248cd084c06daa40e23bb5e62b643414586c4656efe0cab1f28ecb44606ce069.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/248cd084c06daa40e23bb5e62b643414586c4656efe0cab1f28ecb44606ce069.jpg)

![43c59a2a2a183b4fb72c569d15a159a5f2132cd5482071b96ebbbca273415562.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/43c59a2a2a183b4fb72c569d15a159a5f2132cd5482071b96ebbbca273415562.jpg)

![8b37b33f1cec048c2a28dcb7cffe6481aa01084d9ac14fefc2e32c421458cf47.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/8b37b33f1cec048c2a28dcb7cffe6481aa01084d9ac14fefc2e32c421458cf47.jpg)

![8ddc227f04231d08c25cac58ecc60629a5da8c340640d450d55407bc48ecd515.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/8ddc227f04231d08c25cac58ecc60629a5da8c340640d450d55407bc48ecd515.jpg)

![9397a643b2ba14208bee9a914d6ac7c2006f5d2cba4192ba1cc7a31ff7927f66.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/9397a643b2ba14208bee9a914d6ac7c2006f5d2cba4192ba1cc7a31ff7927f66.jpg)

![970412a742f3a6b1927dee12958106cee69ee7a3836bcb026602bb83dd443333.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/970412a742f3a6b1927dee12958106cee69ee7a3836bcb026602bb83dd443333.jpg)

![abb94964ffa313745fef58cc08e4cf7afd9e49a1e6390aaeef82a938b7983b71.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/abb94964ffa313745fef58cc08e4cf7afd9e49a1e6390aaeef82a938b7983b71.jpg)

![cacd9f870e6fc0a4d037f2048ebb74e5e07232a09c2fb81ef8fdfe8f150da102.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/cacd9f870e6fc0a4d037f2048ebb74e5e07232a09c2fb81ef8fdfe8f150da102.jpg)

![f67091dd066fec0dd828e660393c19e5c81e9aa9944331ed72d89384c0f9d207.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/f67091dd066fec0dd828e660393c19e5c81e9aa9944331ed72d89384c0f9d207.jpg)

![fcf447ffe73e8a41384630ed06a73e0967a11c9de098c1cded14380b048f2f5a.jpg](../icml_results/404_Neural%20Concept%20Verifier_%20Scaling%20Prover-Verifier%20Games%20via%20Concept%20Encodings/tables/fcf447ffe73e8a41384630ed06a73e0967a11c9de098c1cded14380b048f2f5a.jpg)
