# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 122 of 197

## 目录 (Table of Contents)

1. [FedSDR: Federated Self-Distillation with Rectification](#paper-3994)
2. [More Sail than Ballast: Addressing Harmful Knowledge Leakage in the Expansive Reasoning Space of LRMs](#paper-3995)
3. [Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](#paper-3996)
4. [Attention Hijacking: Backdooring Text Dataset Distillation via Semantic Anchors](#paper-3997)
5. [FedSSM: State Space Model-based Proactive Inference for Heterogeneous Multimodal Federated Learning](#paper-3998)
6. [Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO](#paper-3999)
7. [Edit-Based Refinement for Parallel Masked Diffusion Language Models](#paper-4000)
8. [When Simple Problems Wear Complex Costumes: Improving Efficiency in LRM’s Adaptive Reasoning](#paper-4001)
9. [A Minimal Agent for Automated Theorem Proving](#paper-4002)
10. [Who Evaluates AI’s Social Impacts? Mapping Coverage and Gaps in First and Third Party Evaluations](#paper-4003)
11. [Use What You Know: Causal Foundation Models with Partial Graphs](#paper-4004)
12. [Online Rubrics Elicitation from Pairwise Comparisons](#paper-4005)
13. [Weight-Space Learning for Certifiable Few-shot Transfer Learning](#paper-4006)
14. [From Muon to Gluon: Bridging Theory and Practice of LMO-based Optimizers for LLMs](#paper-4007)
15. [A Fourier perspective on the learning dynamics of neural networks: from sample complexities to mechanistic insights](#paper-4008)
16. [A Geometric Analysis of Small-sized Language Model Hallucinations](#paper-4009)
17. [Improved Bounds for Reward-Agnostic and Reward-Free Exploration](#paper-4010)
18. [Optimal conversion from Rényi Differential Privacy to $f$-Differential Privacy](#paper-4011)
19. [Transporting Task Vectors across Different Architectures without Training](#paper-4012)
20. [Attacks on Machine-Text Detectors Retain Stylistic Fingerprints](#paper-4013)
21. [ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models](#paper-4014)
22. [Practical and Scalable Hamiltonian Monte Carlo Without the Metropolis Test](#paper-4015)
23. [Model-Free Robust Average-Reward Reinforcement Learning with Sample Complexity Analysis](#paper-4016)
24. [Distributionally Robust Markov Games with Average Reward](#paper-4017)
25. [Performative Learning Theory](#paper-4018)
26. [FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations](#paper-4019)
27. [Diffusion Models Preferentially Memorize Prototypical Examples or: Why Does My Diffusion Model Love Slop?](#paper-4020)
28. [Ambient Dataloops: Generative Models for Dataset Refinement](#paper-4021)
29. [PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning](#paper-4022)
30. [Categorical Flow Maps](#paper-4023)
31. [SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport](#paper-4024)
32. [Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data](#paper-4025)
33. [Preserving Plasticity in Continual Learning via Dynamical Isometry](#paper-4026)

---


<a id="paper-3994"></a>

## FedSDR: Federated Self-Distillation with Rectification

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

## FedSDR: Federated Self-Distillation with Rectification


### Images

![06cf6dc391ead42276a66af20ae9c7e762b226f0c4525e8c7da783f69df0274f.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/06cf6dc391ead42276a66af20ae9c7e762b226f0c4525e8c7da783f69df0274f.jpg)

![2a8d01a52ecb22b74b9ba23181de5ca3a6dbfe218aa82228343d2df43fd60025.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/2a8d01a52ecb22b74b9ba23181de5ca3a6dbfe218aa82228343d2df43fd60025.jpg)

![2d9b5bbbada8022d34cf1f551268580fc0b64c38199a47532011412705755d31.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/2d9b5bbbada8022d34cf1f551268580fc0b64c38199a47532011412705755d31.jpg)

![2da386f6396748c36601063e981cdf14a4b58fe9ff68bbf94acd07a58dd1dc00.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/2da386f6396748c36601063e981cdf14a4b58fe9ff68bbf94acd07a58dd1dc00.jpg)

![8b545335da3f8daa61450e0cdb274a75276a327ed2815dd16ee5326c3d84fe55.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/8b545335da3f8daa61450e0cdb274a75276a327ed2815dd16ee5326c3d84fe55.jpg)

![c38e2345afbd9611bcbbd74d30193a1eccb48ef4f8067cffd681fadaca0782fb.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/images/c38e2345afbd9611bcbbd74d30193a1eccb48ef4f8067cffd681fadaca0782fb.jpg)

### Tables

![0843202639029dbb760529288048e06802830762600f9e7ec3d876aee534f05d.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/0843202639029dbb760529288048e06802830762600f9e7ec3d876aee534f05d.jpg)

![0eb6afea8dff212edce958b27008c1dbe60e078a534dde76b7c4cad90c13a86a.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/0eb6afea8dff212edce958b27008c1dbe60e078a534dde76b7c4cad90c13a86a.jpg)

![1dde139bcdc5a2318ebcc0cf8a4e9d6eca9462c8311ebc30c770cb41ef4e34d1.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/1dde139bcdc5a2318ebcc0cf8a4e9d6eca9462c8311ebc30c770cb41ef4e34d1.jpg)

![1edd3c97087ea46ed5555963c42d64cd8e570d38c0f8b139391480f2d547592f.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/1edd3c97087ea46ed5555963c42d64cd8e570d38c0f8b139391480f2d547592f.jpg)

![3e5fb62b350f84ba903c3efc50d5cb896fb540e91bfce20c981bd460515f0c84.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/3e5fb62b350f84ba903c3efc50d5cb896fb540e91bfce20c981bd460515f0c84.jpg)

![44f0010329cc9f50e29b37aed677c8c0d91acb153003e76854c5c3a430872fcd.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/44f0010329cc9f50e29b37aed677c8c0d91acb153003e76854c5c3a430872fcd.jpg)

![53183cd8f4b0198771203d3767a3e600427a6035493047d67ae85e45cc5a1183.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/53183cd8f4b0198771203d3767a3e600427a6035493047d67ae85e45cc5a1183.jpg)

![6554dc0e72c052c89404dd2a745b3d0ded82e8b6ff02fe8800c0cbf8dda6b4af.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/6554dc0e72c052c89404dd2a745b3d0ded82e8b6ff02fe8800c0cbf8dda6b4af.jpg)

![b39fc125c67a6728eac5d2be506b01d2f07f8a3698a3f1ce89c7c44fec879666.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/b39fc125c67a6728eac5d2be506b01d2f07f8a3698a3f1ce89c7c44fec879666.jpg)

![c7727063ea00a4631fd6a10ec8824bc0df9cd3358909b04febc5db32fe90c7a1.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/c7727063ea00a4631fd6a10ec8824bc0df9cd3358909b04febc5db32fe90c7a1.jpg)

![f10a12556d7a063d0f5cf4788b9545a9893d52dbd68d4b11f0abb541536f1dac.jpg](../icml_results/4035_FedSDR_%20Federated%20Self-Distillation%20with%20Rectification/tables/f10a12556d7a063d0f5cf4788b9545a9893d52dbd68d4b11f0abb541536f1dac.jpg)

<a id="paper-3995"></a>

## More Sail than Ballast: Addressing Harmful Knowledge Leakage in the Expansive Reasoning Space of LRMs


### Images

![00b89dea87b275d8202a1cf2185a47822950006d2fb161bcf96b2aca84e0e3f2.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/00b89dea87b275d8202a1cf2185a47822950006d2fb161bcf96b2aca84e0e3f2.jpg)

![48ef4aab1c9a587131147ba1920fc9fb89e2399422f7961fcd4b7d05ff3a38d1.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/48ef4aab1c9a587131147ba1920fc9fb89e2399422f7961fcd4b7d05ff3a38d1.jpg)

![65fce10de79e17a363d8ae4519819f05bc9f12a6d18428864d0448c38718a647.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/65fce10de79e17a363d8ae4519819f05bc9f12a6d18428864d0448c38718a647.jpg)

![c85c02c131e73693c94f1abe3e58aa4bf7b5727deab5df5d80b8c4c6050ffce7.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/c85c02c131e73693c94f1abe3e58aa4bf7b5727deab5df5d80b8c4c6050ffce7.jpg)

![cd65c80f75bc1bdd116e67db12d09bc45568427afe06a9150df68c1d4074d43c.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/cd65c80f75bc1bdd116e67db12d09bc45568427afe06a9150df68c1d4074d43c.jpg)

![f67c8301c9435d99d72a8c5a31a36ebe686a5feba8656b19e55efe83446a62de.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/images/f67c8301c9435d99d72a8c5a31a36ebe686a5feba8656b19e55efe83446a62de.jpg)

### Tables

![1d9b2e227991661ad479460027c76b88cd1acd0a056712ec304be6cf868645be.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/1d9b2e227991661ad479460027c76b88cd1acd0a056712ec304be6cf868645be.jpg)

![22471d068a6e4033d1de57790b23c56f377e01dca2cee631370deaa14ca7b89a.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/22471d068a6e4033d1de57790b23c56f377e01dca2cee631370deaa14ca7b89a.jpg)

![277cb3dbfdcf25ee0ff41cf6c44cbdd5d694bd9c9cda87e585ae9395d3d715ce.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/277cb3dbfdcf25ee0ff41cf6c44cbdd5d694bd9c9cda87e585ae9395d3d715ce.jpg)

![36b1a7f321f050fde5e2e1a1a43151a3123eb6b598e96f93f2c7deedba28d0dd.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/36b1a7f321f050fde5e2e1a1a43151a3123eb6b598e96f93f2c7deedba28d0dd.jpg)

![515a20d12b832e50d9432cccc223b05ea3abe5863aaa0fa108cc03067d0527d8.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/515a20d12b832e50d9432cccc223b05ea3abe5863aaa0fa108cc03067d0527d8.jpg)

![606f0a9069b5c0bb6a80073963c705665b3d93f3f442e83026d1f221a24c3f82.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/606f0a9069b5c0bb6a80073963c705665b3d93f3f442e83026d1f221a24c3f82.jpg)

![63e80e1093e598379ba27bffa29a3cfbc5574611b799531349d9b1f678b9e8de.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/63e80e1093e598379ba27bffa29a3cfbc5574611b799531349d9b1f678b9e8de.jpg)

![71b5ca252da7094bcb2613ea8e01f5fa169307e4f78616ecf939606c44256bee.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/71b5ca252da7094bcb2613ea8e01f5fa169307e4f78616ecf939606c44256bee.jpg)

![80f7fa17e8fbc39d2eb7110c641fe565e9d367831ee466ff91b9433fa4748af8.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/80f7fa17e8fbc39d2eb7110c641fe565e9d367831ee466ff91b9433fa4748af8.jpg)

![8c423b2f24bf372bd3f5bdaee4e8f087ab8cb5ef396719277047f47cb0b130a2.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/8c423b2f24bf372bd3f5bdaee4e8f087ab8cb5ef396719277047f47cb0b130a2.jpg)

![8c799e2a06b2bc13c0bb90f4a60d8f11103ef180698fdbe5e672c88c284ccf19.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/8c799e2a06b2bc13c0bb90f4a60d8f11103ef180698fdbe5e672c88c284ccf19.jpg)

![9a1beab12b9532b63a4e9a857a5d228ecdfc2a44017cbc129ce358a7ed0570ae.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/9a1beab12b9532b63a4e9a857a5d228ecdfc2a44017cbc129ce358a7ed0570ae.jpg)

![abb2319c5735a2caa628f551a7eb1b6de3f78467e2b6816861e61ff625e2b6a7.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/abb2319c5735a2caa628f551a7eb1b6de3f78467e2b6816861e61ff625e2b6a7.jpg)

![acdd420238406359646ada19dc5d47318eda1ce3f50198784ec10bc93eb46f5a.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/acdd420238406359646ada19dc5d47318eda1ce3f50198784ec10bc93eb46f5a.jpg)

![ca7633fb40562c6c40ddd4c4a9da157d422027f59dc094919f56009149c6bba1.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/ca7633fb40562c6c40ddd4c4a9da157d422027f59dc094919f56009149c6bba1.jpg)

![d0e8655f0bdb1ac71a28003c1985bb923c6ab553d08f67472feecebfaab5a4eb.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/d0e8655f0bdb1ac71a28003c1985bb923c6ab553d08f67472feecebfaab5a4eb.jpg)

![d60afa92171cc0ded746fbab1f94a5088aa13b83160f9807ebd3e8895778edcc.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/d60afa92171cc0ded746fbab1f94a5088aa13b83160f9807ebd3e8895778edcc.jpg)

![fd5632fbb607ec9de282e328ae24dd2a0a8df4e956efd5cc1bc39f3abef76274.jpg](../icml_results/4036_More%20Sail%20than%20Ballast_%20Addressing%20Harmful%20Knowledge%20Leakage%20in%20the%20Expansive%20Reasoning%20Space%20of%20LRM/tables/fd5632fbb607ec9de282e328ae24dd2a0a8df4e956efd5cc1bc39f3abef76274.jpg)

<a id="paper-3996"></a>

## Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence


### Images

![0ffb8835a838e3a94fdfb0720b1ccab735761e2eceea757df7309dbca1bd18b3.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/0ffb8835a838e3a94fdfb0720b1ccab735761e2eceea757df7309dbca1bd18b3.jpg)

![123cf7f9ac553505d5d8d3f4d9eca0806a39b5ec5b3bdea5bb4028c5f13c27b8.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/123cf7f9ac553505d5d8d3f4d9eca0806a39b5ec5b3bdea5bb4028c5f13c27b8.jpg)

![37360df3a1125b9b83ac127f532f347455df45543b4e403edf94358c6adb21ac.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/37360df3a1125b9b83ac127f532f347455df45543b4e403edf94358c6adb21ac.jpg)

![3bfc896d600da275a1e657271541b226f88ad4d71e9ae9274097d969c7c69561.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/3bfc896d600da275a1e657271541b226f88ad4d71e9ae9274097d969c7c69561.jpg)

![79c7921700c530fbe3aa19d7a703402771e50bf9e1e6afb2f9763777301fce21.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/79c7921700c530fbe3aa19d7a703402771e50bf9e1e6afb2f9763777301fce21.jpg)

![7ffb4fabb04e125dd10a7f6fca322ce67844132c91fb756ae6a449a059f7af80.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/7ffb4fabb04e125dd10a7f6fca322ce67844132c91fb756ae6a449a059f7af80.jpg)

![9be93b9245fe6d4a886bc44921bc15ee0048e2a8a675cb71a34fa9e193633bc4.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/9be93b9245fe6d4a886bc44921bc15ee0048e2a8a675cb71a34fa9e193633bc4.jpg)

![e37cf62e779eeac64212dcb9dc0bbe1fbe1cdf575cd01a464594d309a44c1206.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/e37cf62e779eeac64212dcb9dc0bbe1fbe1cdf575cd01a464594d309a44c1206.jpg)

![f251a9ae2b85e9076c3940ef88d3711d0dca481fa5497ea1b6c48d1a21875f2d.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/f251a9ae2b85e9076c3940ef88d3711d0dca481fa5497ea1b6c48d1a21875f2d.jpg)

![f4dea726f9c224a03b3a8a728448cb01d3d22401732c33eececb8cf661bba9b0.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/images/f4dea726f9c224a03b3a8a728448cb01d3d22401732c33eececb8cf661bba9b0.jpg)

### Tables

![0269a6690aa709aa64c725167279c0d9de8f6f050506405151528bfc3b12cd1b.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/0269a6690aa709aa64c725167279c0d9de8f6f050506405151528bfc3b12cd1b.jpg)

![06f8f95b6adfd57aefe9eaa09ae7a2a76b5edd2e01e654e4eb78eeceb7401cc7.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/06f8f95b6adfd57aefe9eaa09ae7a2a76b5edd2e01e654e4eb78eeceb7401cc7.jpg)

![09b8dc2d8651ad9af2d86d8c00649fd6c60de3bd2afda77504fefa27416ac744.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/09b8dc2d8651ad9af2d86d8c00649fd6c60de3bd2afda77504fefa27416ac744.jpg)

![0b3f4ca7a29b79f580fe21a22e55aed528ebc42de39d17f33f029ca9c8e6177b.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/0b3f4ca7a29b79f580fe21a22e55aed528ebc42de39d17f33f029ca9c8e6177b.jpg)

![1684cfc895f382c1d3ae657e62a6947b5fc5f2b3d7fdbb1d2f39687c457c3cef.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/1684cfc895f382c1d3ae657e62a6947b5fc5f2b3d7fdbb1d2f39687c457c3cef.jpg)

![1c48e50013432ef28e15a3403bd26020613a7d6fc14a0a9219e55c55c8bc1fb0.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/1c48e50013432ef28e15a3403bd26020613a7d6fc14a0a9219e55c55c8bc1fb0.jpg)

![1e0a28a0776d53287db9b22b4ca46883bf0d5e8dc4d0da6341d68060cb1180d1.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/1e0a28a0776d53287db9b22b4ca46883bf0d5e8dc4d0da6341d68060cb1180d1.jpg)

![27a88377be422de3376410a8ef1e30b66a97c9397b3e024242ddb48093e2b1ff.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/27a88377be422de3376410a8ef1e30b66a97c9397b3e024242ddb48093e2b1ff.jpg)

![2ddd77726410e4fc4802cc162213d335aa5152259a7a73ae8b9db42bb172595b.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/2ddd77726410e4fc4802cc162213d335aa5152259a7a73ae8b9db42bb172595b.jpg)

![3666dc4417ad32c18ab544b68c9b76a62be9d684fe737a64626a38ba632ace4e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/3666dc4417ad32c18ab544b68c9b76a62be9d684fe737a64626a38ba632ace4e.jpg)

![40033879bfc599266d558d35749811eff241624bef8aba8fc8f347b7854d002e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/40033879bfc599266d558d35749811eff241624bef8aba8fc8f347b7854d002e.jpg)

![409edb12a30039970a88af15651d24378b856ccf60fabb4700ca04e5cbb977a1.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/409edb12a30039970a88af15651d24378b856ccf60fabb4700ca04e5cbb977a1.jpg)

![4732524a257a5a24e79dbff3a1bac32b5062b8d42a3a96ca9e39f73c214d9f9b.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/4732524a257a5a24e79dbff3a1bac32b5062b8d42a3a96ca9e39f73c214d9f9b.jpg)

![4a465f39db3aae4c9b42c253856954092608d46829bbff06456b4929331497b4.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/4a465f39db3aae4c9b42c253856954092608d46829bbff06456b4929331497b4.jpg)

![4b3cfc822ce3fc41b7742646dc260c52d2a49f2ce6f43083b889eb88f3237e3e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/4b3cfc822ce3fc41b7742646dc260c52d2a49f2ce6f43083b889eb88f3237e3e.jpg)

![4cde928cdb77a434cc37ee9ee58b3f54d3074157bbc6acf47272f4d6039fe53a.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/4cde928cdb77a434cc37ee9ee58b3f54d3074157bbc6acf47272f4d6039fe53a.jpg)

![517b662fb9c25c0a4e0b3b2daf3b0ed8f98d9cc68797b177873310b57a3b4933.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/517b662fb9c25c0a4e0b3b2daf3b0ed8f98d9cc68797b177873310b57a3b4933.jpg)

![58dc6ccb40c3a395cb91ef29be7ae1771e2f4da7f139933a3859af6a38e913ce.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/58dc6ccb40c3a395cb91ef29be7ae1771e2f4da7f139933a3859af6a38e913ce.jpg)

![61756c41d5ebf4b076e7cee7790de727eb127f0a6d97edfccc5182cad07255f9.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/61756c41d5ebf4b076e7cee7790de727eb127f0a6d97edfccc5182cad07255f9.jpg)

![634be0b0e35663c186b9d5bfc2e0033b94ab3d2e3634e52bc55c665d4b1dac0e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/634be0b0e35663c186b9d5bfc2e0033b94ab3d2e3634e52bc55c665d4b1dac0e.jpg)

![699839bde9abc4c85e4c4ed3d49b24bd7905e8b772d6eeb8b8965f5643f4d994.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/699839bde9abc4c85e4c4ed3d49b24bd7905e8b772d6eeb8b8965f5643f4d994.jpg)

![72408bf323fa80001cc8b3faaa9201cd887c34e9b9b9eb4815b35e89308ee0a3.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/72408bf323fa80001cc8b3faaa9201cd887c34e9b9b9eb4815b35e89308ee0a3.jpg)

![72a1fdf489edd6a218a63903621eba930b86869d284d2d8ce3e8c79f2fdd9cb6.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/72a1fdf489edd6a218a63903621eba930b86869d284d2d8ce3e8c79f2fdd9cb6.jpg)

![748bb59b7ac5adcb0a2828fcba3ceb868f0829481201a8e77d30117753ae973e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/748bb59b7ac5adcb0a2828fcba3ceb868f0829481201a8e77d30117753ae973e.jpg)

![7711c76819fc492f87bd378f4b53837a069f038e7d9c131cd0d6bf8ee387221e.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/7711c76819fc492f87bd378f4b53837a069f038e7d9c131cd0d6bf8ee387221e.jpg)

![78e83962d0b9bfe3d9e4c352ed6a84568751300782b56a09c02aef8025e88f03.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/78e83962d0b9bfe3d9e4c352ed6a84568751300782b56a09c02aef8025e88f03.jpg)

![79dea8b63134f3d953d23cc8accbb459a31b0cec6d8e18e8e4044e9229739d81.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/79dea8b63134f3d953d23cc8accbb459a31b0cec6d8e18e8e4044e9229739d81.jpg)

![80e2e3e20efd61ad8b8427a3932240ff6d54f0daeec56db4e017ebc995829af8.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/80e2e3e20efd61ad8b8427a3932240ff6d54f0daeec56db4e017ebc995829af8.jpg)

![812e444ca77991ce16612f6a9d6f14626c62cba9684fdb9b036d7f30680c9762.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/812e444ca77991ce16612f6a9d6f14626c62cba9684fdb9b036d7f30680c9762.jpg)

![8b0cb445e8fa6dd284251136e38da66bd0d2a9da8a1e208c0656f3bd24d526b4.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/8b0cb445e8fa6dd284251136e38da66bd0d2a9da8a1e208c0656f3bd24d526b4.jpg)

![91bb1f13dc88e80799317f37f64e072d7119204d95825aa906a1e9e3a5267ac0.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/91bb1f13dc88e80799317f37f64e072d7119204d95825aa906a1e9e3a5267ac0.jpg)

![9425bea8e4870e93354f4195ab3391371a0f223c97c289e3460521f379886cb2.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/9425bea8e4870e93354f4195ab3391371a0f223c97c289e3460521f379886cb2.jpg)

![9db0c4dc93a6fafa48faf86b25a8d34ef5d8dbf95ece1035291e536c6ab0f0ed.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/9db0c4dc93a6fafa48faf86b25a8d34ef5d8dbf95ece1035291e536c6ab0f0ed.jpg)

![a487540ac7b23e2d83fe9e8241e07bd845cd1ecb1dfcccd2f643e48f26b51ac1.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/a487540ac7b23e2d83fe9e8241e07bd845cd1ecb1dfcccd2f643e48f26b51ac1.jpg)

![ab35913a40e3ede8fc3676370e45b38c9b12df363ddb4689aec4cdfdd24a46be.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/ab35913a40e3ede8fc3676370e45b38c9b12df363ddb4689aec4cdfdd24a46be.jpg)

![b326a2dc5e06bcc91187b26982140ad06b984f10da70e56be535dc6a18b1a1ac.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/b326a2dc5e06bcc91187b26982140ad06b984f10da70e56be535dc6a18b1a1ac.jpg)

![b9398a5169eae4e2ceaf3ef83c6c6d72f7faa3298295ecf2cdb44f02afd38c1d.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/b9398a5169eae4e2ceaf3ef83c6c6d72f7faa3298295ecf2cdb44f02afd38c1d.jpg)

![bbedcc5262b00bd9dcd2d55f0b863ac525021bc3553874fd50236ddfa58cd8fb.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/bbedcc5262b00bd9dcd2d55f0b863ac525021bc3553874fd50236ddfa58cd8fb.jpg)

![c4115c70b0c8b7c7ff0eb68d1b49ece890ebd67c0a038ab83adec9f3c8040930.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/c4115c70b0c8b7c7ff0eb68d1b49ece890ebd67c0a038ab83adec9f3c8040930.jpg)

![c652b57fc140c3bd2ddb4e710f262a0d3138dc2e426b57735d2916ace49758c4.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/c652b57fc140c3bd2ddb4e710f262a0d3138dc2e426b57735d2916ace49758c4.jpg)

![cb11492c56844a17a21921d23608d18edc7d7fbdca1f5a59055360b91b3d8bdb.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/cb11492c56844a17a21921d23608d18edc7d7fbdca1f5a59055360b91b3d8bdb.jpg)

![d0c17b96e7abea27c191a28fb6601b5b8b3ec8356140bdfd29fcf2140bc8da04.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/d0c17b96e7abea27c191a28fb6601b5b8b3ec8356140bdfd29fcf2140bc8da04.jpg)

![e02aa6523c61fba95c8990b62a487c0431fbef4c0750d1cba48d7718f1ed0ee7.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/e02aa6523c61fba95c8990b62a487c0431fbef4c0750d1cba48d7718f1ed0ee7.jpg)

![e05eb32956ae31566dbfd816768f1b4f5862de3e5e552474fd6d0cc7ecc18a34.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/e05eb32956ae31566dbfd816768f1b4f5862de3e5e552474fd6d0cc7ecc18a34.jpg)

![e40224bca451994caf1eb560f6b43ec7f167ba3dc3ea60ce8b9d1d1204d522f5.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/e40224bca451994caf1eb560f6b43ec7f167ba3dc3ea60ce8b9d1d1204d522f5.jpg)

![ea3aa9e9551e973a323469ff4ab3569918ea3e638a5c543e82596c7b613b4acb.jpg](../icml_results/4037_Revisiting%20Parameter-Based%20Knowledge%20Editing%20in%20Large%20Language%20Models_%20Theoretical%20Limits%20and%20Empiri/tables/ea3aa9e9551e973a323469ff4ab3569918ea3e638a5c543e82596c7b613b4acb.jpg)

<a id="paper-3997"></a>

## Attention Hijacking: Backdooring Text Dataset Distillation via Semantic Anchors


### Images

![665b9cfdc9e83e01ebec9f38ec11bd47ca43e68c2358c95f8053de086e780cdc.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/images/665b9cfdc9e83e01ebec9f38ec11bd47ca43e68c2358c95f8053de086e780cdc.jpg)

![80c4069e785ae1d4572db56d06c7e81bbe648cfe01bf6fad4a6782ae6f025258.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/images/80c4069e785ae1d4572db56d06c7e81bbe648cfe01bf6fad4a6782ae6f025258.jpg)

![b810d0e2d79eb806952e3c01854800dc3af280bb35f623e0ba8a335fcdf84b73.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/images/b810d0e2d79eb806952e3c01854800dc3af280bb35f623e0ba8a335fcdf84b73.jpg)

### Tables

![160a335700a15f43e2f3df505248ab86634ff40f4f79b6bb4860ad4142826428.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/160a335700a15f43e2f3df505248ab86634ff40f4f79b6bb4860ad4142826428.jpg)

![275f98499e1f6089cfcd64a5746178c8bad4ebe3ae451c582e8886f88ec542f2.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/275f98499e1f6089cfcd64a5746178c8bad4ebe3ae451c582e8886f88ec542f2.jpg)

![53068058daead99571d8c010bc748b8945b104abb98df5cae10ddf6087062db2.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/53068058daead99571d8c010bc748b8945b104abb98df5cae10ddf6087062db2.jpg)

![57e9d4603bf3d9c9f215162f807e59a7929e9bdd1ced214f9558e6e1810fa7ec.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/57e9d4603bf3d9c9f215162f807e59a7929e9bdd1ced214f9558e6e1810fa7ec.jpg)

![7c3f31f8bd42a6094ddc657e23d92efb2efb29e8535d32f6e241c67e7c8ffb56.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/7c3f31f8bd42a6094ddc657e23d92efb2efb29e8535d32f6e241c67e7c8ffb56.jpg)

![9208a9237f37ef05f87ee74374e948587d29568144f2d3883f61cb22338434ea.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/9208a9237f37ef05f87ee74374e948587d29568144f2d3883f61cb22338434ea.jpg)

![cf92658060ee687059d6d18ec1cb18adf9a9d90564ada3c17b802cd949e227cd.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/cf92658060ee687059d6d18ec1cb18adf9a9d90564ada3c17b802cd949e227cd.jpg)

![d39cce3004040adff13de3e8dc6e873c53bc34411f89b41dd03da94083edf339.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/d39cce3004040adff13de3e8dc6e873c53bc34411f89b41dd03da94083edf339.jpg)

![dac8556387ea9b8a9c222ab681defca41ede022a70f529353a63e6ae9ab20604.jpg](../icml_results/4038_Attention%20Hijacking_%20Backdooring%20Text%20Dataset%20Distillation%20via%20Semantic%20Anchors/tables/dac8556387ea9b8a9c222ab681defca41ede022a70f529353a63e6ae9ab20604.jpg)

<a id="paper-3998"></a>

## FedSSM: State Space Model-based Proactive Inference for Heterogeneous Multimodal Federated Learning


### Images

![2433769153b747011a08652fa70e86d158ec4dd0452c7d7cff636529d9473a92.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/2433769153b747011a08652fa70e86d158ec4dd0452c7d7cff636529d9473a92.jpg)

![5c63c4e75276b21543ddb2c2df7367050cd88845b974261a9d533022b3cc28c0.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/5c63c4e75276b21543ddb2c2df7367050cd88845b974261a9d533022b3cc28c0.jpg)

![5d0dd1e06c636e4599f633eca4156e57441147b1e72dab2039b71be11a38d28e.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/5d0dd1e06c636e4599f633eca4156e57441147b1e72dab2039b71be11a38d28e.jpg)

![7ac4196a3a9224c1116da415eb03624de049f8f8a5cc9c83abfc90649319f803.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/7ac4196a3a9224c1116da415eb03624de049f8f8a5cc9c83abfc90649319f803.jpg)

![801d375ffcc2f8366ebb72874c95fb6f27ef344f5901d9622b7e1a6b6d588500.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/801d375ffcc2f8366ebb72874c95fb6f27ef344f5901d9622b7e1a6b6d588500.jpg)

![ddda514346245ec3d3e019ddc162cf891dbc761d5d642bcc67b7723f8299cd34.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/ddda514346245ec3d3e019ddc162cf891dbc761d5d642bcc67b7723f8299cd34.jpg)

![ff99aee7c4aa21e494f6cdeb46cedbe045b1696b05983ec4ff9a16c1851a9856.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/images/ff99aee7c4aa21e494f6cdeb46cedbe045b1696b05983ec4ff9a16c1851a9856.jpg)

### Tables

![3a6ef5afec8af1e156c9591a8c9db027f0d15c7e23c824b7a062488d1d70aa12.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/3a6ef5afec8af1e156c9591a8c9db027f0d15c7e23c824b7a062488d1d70aa12.jpg)

![3e1a2cd1584f5513e9c480df8dd091e4c9965f2c1f88be2d8010ebc45d33cc4e.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/3e1a2cd1584f5513e9c480df8dd091e4c9965f2c1f88be2d8010ebc45d33cc4e.jpg)

![3f55188bc304231eec0530b275ae9de2e25e333fc1decff0728b376015d8d0f3.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/3f55188bc304231eec0530b275ae9de2e25e333fc1decff0728b376015d8d0f3.jpg)

![55f7f80c31170e1e5dd8681300cc27e9784395f2bb5f3677e2ecc13253375f16.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/55f7f80c31170e1e5dd8681300cc27e9784395f2bb5f3677e2ecc13253375f16.jpg)

![59adb15c08da2d2a08e2c0e77bc23a079ff1ab4fa6a7e423dbc64fbad54e0f1d.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/59adb15c08da2d2a08e2c0e77bc23a079ff1ab4fa6a7e423dbc64fbad54e0f1d.jpg)

![5e98534bde69bdcdeb7df97a3d9de04ca92e7cebc479026db968ce668dc4b0d0.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/5e98534bde69bdcdeb7df97a3d9de04ca92e7cebc479026db968ce668dc4b0d0.jpg)

![69f7dd32537822cf67c16d843f6a1296b22ac12bc225e1cd30d62ad92b2f6f25.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/69f7dd32537822cf67c16d843f6a1296b22ac12bc225e1cd30d62ad92b2f6f25.jpg)

![95c134e0a84040166245348937938976f2c91a236e62887b42c6649a4b3563c7.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/95c134e0a84040166245348937938976f2c91a236e62887b42c6649a4b3563c7.jpg)

![a88268247c0f9f88e86e8b9fbb973337e535ee315497b8b906213f22431c318e.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/a88268247c0f9f88e86e8b9fbb973337e535ee315497b8b906213f22431c318e.jpg)

![d40a28ec2cd8705c69fa406a9faa043b26eb32f216ec8a07eca5681b7b07b603.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/d40a28ec2cd8705c69fa406a9faa043b26eb32f216ec8a07eca5681b7b07b603.jpg)

![edba7496bcec31c82c47b3bedc09ed1f435345e23a14c682b5e8c901cf417ded.jpg](../icml_results/4039_FedSSM_%20State%20Space%20Model-based%20Proactive%20Inference%20for%20Heterogeneous%20Multimodal%20Federated%20Learning/tables/edba7496bcec31c82c47b3bedc09ed1f435345e23a14c682b5e8c901cf417ded.jpg)

<a id="paper-3999"></a>

## Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO


### Images

![0f8d32c5770d8775fb6e36422164b4a3d598d61a5d7d016793fff8f6f1ee25b9.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/0f8d32c5770d8775fb6e36422164b4a3d598d61a5d7d016793fff8f6f1ee25b9.jpg)

![10722be723a066321805ce28f98a484dab7cdcc22e8ff6cc7a96009f055a7e8d.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/10722be723a066321805ce28f98a484dab7cdcc22e8ff6cc7a96009f055a7e8d.jpg)

![a4bc9eadbda4b5109bc290ca54d72d9e755f4cec0d6805d2fb608d6d4fc6a62c.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/a4bc9eadbda4b5109bc290ca54d72d9e755f4cec0d6805d2fb608d6d4fc6a62c.jpg)

![b081fbaa8cff090dbe3ae583daf66be554e15e58a7807c735dd60a8daf205a9a.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/b081fbaa8cff090dbe3ae583daf66be554e15e58a7807c735dd60a8daf205a9a.jpg)

![ca677a8a5aa0424dba176b4781601c787f604bf297ea09486a6779fe525240fd.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/ca677a8a5aa0424dba176b4781601c787f604bf297ea09486a6779fe525240fd.jpg)

![d1fea42aa82be87c05dfc3f0984e081024fd0a07f984f61eca6d3a4efb60de3f.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/d1fea42aa82be87c05dfc3f0984e081024fd0a07f984f61eca6d3a4efb60de3f.jpg)

![d3b58a7ccbfcca2d991f07695342594d8d89eb4581a4088bffe704d92f5e5504.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/images/d3b58a7ccbfcca2d991f07695342594d8d89eb4581a4088bffe704d92f5e5504.jpg)

### Tables

![1be0ea34857b85410522640cb36f6ec4ab8a664b0dd458f5f444572b94af6387.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/tables/1be0ea34857b85410522640cb36f6ec4ab8a664b0dd458f5f444572b94af6387.jpg)

![40b31a2f26304556c4b95b16f54a3d9d4441ba12dbf10c1f1aa0c8ee6f19e2d4.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/tables/40b31a2f26304556c4b95b16f54a3d9d4441ba12dbf10c1f1aa0c8ee6f19e2d4.jpg)

![40cef509a61215ae01c9235e704e0b01646f01b9e3f379cbdf6a70f8d169baa7.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/tables/40cef509a61215ae01c9235e704e0b01646f01b9e3f379cbdf6a70f8d169baa7.jpg)

![62f2b96c449075ee9a5be6170e503b514a9bedf7cb59fa4b8375b0611a8ab228.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/tables/62f2b96c449075ee9a5be6170e503b514a9bedf7cb59fa4b8375b0611a8ab228.jpg)

![9fa789bf814f68f61cb2f9cfe9578cb2d80d1bed9e49555b3d15fc7beb830ae9.jpg](../icml_results/4040_Smaller%20Models%20are%20Natural%20Explorers%20for%20Policy-Level%20Diversity%20in%20GRPO/tables/9fa789bf814f68f61cb2f9cfe9578cb2d80d1bed9e49555b3d15fc7beb830ae9.jpg)

<a id="paper-4000"></a>

## Edit-Based Refinement for Parallel Masked Diffusion Language Models


### Images

![19a8343910ee88b46caab975c739fd4fe253543ca9e5673890638d0ffa6d365c.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/19a8343910ee88b46caab975c739fd4fe253543ca9e5673890638d0ffa6d365c.jpg)

![2c27237ae1b51d407cb80efeba1722ee0fa60dff8a0fe1e5933a14f47647f96d.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/2c27237ae1b51d407cb80efeba1722ee0fa60dff8a0fe1e5933a14f47647f96d.jpg)

![2d89e9a5a0a409fb86a2b09e3a0fa20bee6d2da1305784e82c4492333357b6a7.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/2d89e9a5a0a409fb86a2b09e3a0fa20bee6d2da1305784e82c4492333357b6a7.jpg)

![c5832e81c3d5452efd042c1c875feebcb7019f7d3e4f1cba8b6249533a39538a.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/c5832e81c3d5452efd042c1c875feebcb7019f7d3e4f1cba8b6249533a39538a.jpg)

![fb91d79899943a65cc68686011506d922681731bc0d975d07adf9fff8798c676.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/fb91d79899943a65cc68686011506d922681731bc0d975d07adf9fff8798c676.jpg)

![fca369fde811dccf33d3b07900fd8942f72bd02569e1b4f9e978a4f91d10a35e.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/images/fca369fde811dccf33d3b07900fd8942f72bd02569e1b4f9e978a4f91d10a35e.jpg)

### Tables

![02731aa149643ef585f699e718b4d107761b0cf55d270feb09992074d75610da.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/02731aa149643ef585f699e718b4d107761b0cf55d270feb09992074d75610da.jpg)

![368763e9367f525e688503ba8529c94a5a1e62da6d9f0775d9cea772bae3f7c5.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/368763e9367f525e688503ba8529c94a5a1e62da6d9f0775d9cea772bae3f7c5.jpg)

![39002ef4eff8c2eb48d9a40e821611ce985cc46b84ba6d7aa11c9c7a1e20d1ee.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/39002ef4eff8c2eb48d9a40e821611ce985cc46b84ba6d7aa11c9c7a1e20d1ee.jpg)

![422dcc2db1632032a2f9a8b156828b034fb3813d7db6bf918df93e34dc4cb162.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/422dcc2db1632032a2f9a8b156828b034fb3813d7db6bf918df93e34dc4cb162.jpg)

![64946507eaa4b584e2448c8da82a0f71d3dd4d0a513d58546b45bb32e19dc885.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/64946507eaa4b584e2448c8da82a0f71d3dd4d0a513d58546b45bb32e19dc885.jpg)

![68a71286a5d12cc28ba696606986d841f93191ae8598afbfae8f58d7b7da0b97.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/68a71286a5d12cc28ba696606986d841f93191ae8598afbfae8f58d7b7da0b97.jpg)

![f73e172b6877eb700148f1326630c8b65ad5f7c5f198b7b4c84d2c60375a2d70.jpg](../icml_results/4041_Edit-Based%20Refinement%20for%20Parallel%20Masked%20Diffusion%20Language%20Models/tables/f73e172b6877eb700148f1326630c8b65ad5f7c5f198b7b4c84d2c60375a2d70.jpg)

<a id="paper-4001"></a>

## When Simple Problems Wear Complex Costumes: Improving Efficiency in LRM’s Adaptive Reasoning


### Images

![3034dde8cf3d0ee7340ccf5192644b8112c19bdc803fbf71a4c68be56f6733e1.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/images/3034dde8cf3d0ee7340ccf5192644b8112c19bdc803fbf71a4c68be56f6733e1.jpg)

![4d1f16df524655377b66dfb8989a545463d3eefe77bca1d56612f0ad6df3930b.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/images/4d1f16df524655377b66dfb8989a545463d3eefe77bca1d56612f0ad6df3930b.jpg)

![79818c8586b38e2fbce4a043af817489466ab2bac44f72bb747729246e5f62c0.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/images/79818c8586b38e2fbce4a043af817489466ab2bac44f72bb747729246e5f62c0.jpg)

![a98797f48b5e3d6875104e174013c2d3ec48d20f21f9c6f6045db41f02477b98.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/images/a98797f48b5e3d6875104e174013c2d3ec48d20f21f9c6f6045db41f02477b98.jpg)

![c5b9079bae3e64d9c77c0ddcc28ed619bda16d42ec5d2b7d11c51bb0dfe39d30.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/images/c5b9079bae3e64d9c77c0ddcc28ed619bda16d42ec5d2b7d11c51bb0dfe39d30.jpg)

### Tables

![1a861d0c0cc8db48dddd0eaaa87c8d58f7b059f1e2b54b77ef9d65f27c46d499.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/1a861d0c0cc8db48dddd0eaaa87c8d58f7b059f1e2b54b77ef9d65f27c46d499.jpg)

![205f820acaac961cd5fbf7255058d2a9dc82074c12c7c3a710c44ceeae9cc8c8.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/205f820acaac961cd5fbf7255058d2a9dc82074c12c7c3a710c44ceeae9cc8c8.jpg)

![ace990db26d7ad16a39e4479d86d31e0efe9b88aebb53784e22647f30655e297.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/ace990db26d7ad16a39e4479d86d31e0efe9b88aebb53784e22647f30655e297.jpg)

![ced53cecd5cd277e6bccda66521d4be340bc3e2a98cfa13e747a53ef45164f51.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/ced53cecd5cd277e6bccda66521d4be340bc3e2a98cfa13e747a53ef45164f51.jpg)

![e9f9e751095977d4781cba4f9868b43b6093f9571e0f59060b59a8c7dc5edee8.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/e9f9e751095977d4781cba4f9868b43b6093f9571e0f59060b59a8c7dc5edee8.jpg)

![faf9ddd9033ec33a197dece9e4fda79f5839437ed9378d9f511242eb99d3f177.jpg](../icml_results/4042_When%20Simple%20Problems%20Wear%20Complex%20Costumes_%20Improving%20Efficiency%20in%20LRM%E2%80%99s%20Adaptive%20Reasoning/tables/faf9ddd9033ec33a197dece9e4fda79f5839437ed9378d9f511242eb99d3f177.jpg)

<a id="paper-4002"></a>

## A Minimal Agent for Automated Theorem Proving


### Images

![01cb3fdbc5501121f633b10725527142fc0a304f0b076afeb2f413741b4329a6.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/01cb3fdbc5501121f633b10725527142fc0a304f0b076afeb2f413741b4329a6.jpg)

![0266a84c7c2d5f11e7c44d7cee967e0caa5b424de3f4f4d090a5a32be1df3c03.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/0266a84c7c2d5f11e7c44d7cee967e0caa5b424de3f4f4d090a5a32be1df3c03.jpg)

![088e775a709adc13dbfc60143218ec9fe3602ab2b45b378c3b91897aae8487db.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/088e775a709adc13dbfc60143218ec9fe3602ab2b45b378c3b91897aae8487db.jpg)

![1d1747584b328012064e426a39dea4815025937fd680003f080140835732038e.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/1d1747584b328012064e426a39dea4815025937fd680003f080140835732038e.jpg)

![1ef301f01b6eef5dd23ce05cdcac19f4983c3f58aecdfec052afdea2393f4ca6.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/1ef301f01b6eef5dd23ce05cdcac19f4983c3f58aecdfec052afdea2393f4ca6.jpg)

![5aa44096aeccfc4933b71fe4fbcbf18e5a4210ae7580611fd8cfec59012c195b.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/5aa44096aeccfc4933b71fe4fbcbf18e5a4210ae7580611fd8cfec59012c195b.jpg)

![98655e3fc4d221f5c4f344673a74eb20f608bff1ce397379b088d2ef3a0e4fe6.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/98655e3fc4d221f5c4f344673a74eb20f608bff1ce397379b088d2ef3a0e4fe6.jpg)

![b070f220227e9ec9f1a5279d482cab7479814610d15cfda70cc41fd2d87bacc9.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/b070f220227e9ec9f1a5279d482cab7479814610d15cfda70cc41fd2d87bacc9.jpg)

![c2e3a3c3de7d2578d4cead14ed4ea895868b87fd37a4d7470b5cecc8c7143aa8.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/c2e3a3c3de7d2578d4cead14ed4ea895868b87fd37a4d7470b5cecc8c7143aa8.jpg)

![f86ad451e70e965e609d809ec36a0e1af0b39b2f1af761b6e79b4f8177d24dcb.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/images/f86ad451e70e965e609d809ec36a0e1af0b39b2f1af761b6e79b4f8177d24dcb.jpg)

### Tables

![08d15cac102610dafc2312dea548c6361ab90207edd48acc5b12ce3e63059cf8.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/tables/08d15cac102610dafc2312dea548c6361ab90207edd48acc5b12ce3e63059cf8.jpg)

![1c984eb92db9051b97840531fc49f4021fdcb7756e646bf2cbcb1f3007043381.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/tables/1c984eb92db9051b97840531fc49f4021fdcb7756e646bf2cbcb1f3007043381.jpg)

![74bb641fa31cb727ee426ed08edb3183e07676c7cacce1d1d5f512a58a21135c.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/tables/74bb641fa31cb727ee426ed08edb3183e07676c7cacce1d1d5f512a58a21135c.jpg)

![b0633c60d76f9d32595d70e6710ab6e574934994ba3a6f1d3e96aaa6b2f29874.jpg](../icml_results/4043_A%20Minimal%20Agent%20for%20Automated%20Theorem%20Proving/tables/b0633c60d76f9d32595d70e6710ab6e574934994ba3a6f1d3e96aaa6b2f29874.jpg)

<a id="paper-4003"></a>

## Who Evaluates AI’s Social Impacts? Mapping Coverage and Gaps in First and Third Party Evaluations


### Images

![0762a941a0e701ad30a1107b3d8d4ccbee42a961d57099b270c601f82ff654b8.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/0762a941a0e701ad30a1107b3d8d4ccbee42a961d57099b270c601f82ff654b8.jpg)

![0e480a66e4ce34d9b9568855e38fc4b903480339f13f70a413c42b556f077421.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/0e480a66e4ce34d9b9568855e38fc4b903480339f13f70a413c42b556f077421.jpg)

![18d20ad04d4af842f9d81a221130991200118734d96691b22ea4037f0c6ea97e.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/18d20ad04d4af842f9d81a221130991200118734d96691b22ea4037f0c6ea97e.jpg)

![1922341f063f4e747b0943a9cf13a7cc974eafde65b528cfa46d83782927341b.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/1922341f063f4e747b0943a9cf13a7cc974eafde65b528cfa46d83782927341b.jpg)

![1a084ab67f212313364408c295a3a44447143f24368b2eaef855b7a526649863.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/1a084ab67f212313364408c295a3a44447143f24368b2eaef855b7a526649863.jpg)

![1dc872dba5828fae63a29cfc5caef48803d1cc06143502a6c8e8e9287e3447d9.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/1dc872dba5828fae63a29cfc5caef48803d1cc06143502a6c8e8e9287e3447d9.jpg)

![20fc72a441050d5e8435cc15e46753716d3129c8bd9148de9efb522711859284.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/20fc72a441050d5e8435cc15e46753716d3129c8bd9148de9efb522711859284.jpg)

![27f5b27ebb29d01a5f3acbfcc2469feb7230617cf2719b818979a623e1c1a14d.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/27f5b27ebb29d01a5f3acbfcc2469feb7230617cf2719b818979a623e1c1a14d.jpg)

![289921bcf89dc08ee98c3fc627b5084f1c4679799b2f032622302893b38562e5.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/289921bcf89dc08ee98c3fc627b5084f1c4679799b2f032622302893b38562e5.jpg)

![2ad3077024e69b3bd354d1a3130cc1289af4da57e3dc1a52d8d5c8aeb5f6aa9e.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/2ad3077024e69b3bd354d1a3130cc1289af4da57e3dc1a52d8d5c8aeb5f6aa9e.jpg)

![376e22af6cdfa4b1b79e38effc4ef6742509478d23708d0ecdbc76160c83cc02.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/376e22af6cdfa4b1b79e38effc4ef6742509478d23708d0ecdbc76160c83cc02.jpg)

![43a22343bf64735d04f86fb5ba1c796c104e3f19f07cc6ebc260c9b229f6d733.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/43a22343bf64735d04f86fb5ba1c796c104e3f19f07cc6ebc260c9b229f6d733.jpg)

![49c509b08e7229ac1f71cf3404c4120f4eab65a0578864289a9599104190ec55.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/49c509b08e7229ac1f71cf3404c4120f4eab65a0578864289a9599104190ec55.jpg)

![4dc923ca08da459a45e6b8fd91e32aa8c52d373bc510dff36f7df7654ae709fb.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/4dc923ca08da459a45e6b8fd91e32aa8c52d373bc510dff36f7df7654ae709fb.jpg)

![514be2466916ea20b7c2f6eed788b8df43d464a84ee886367b55fe608e6b8d6a.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/514be2466916ea20b7c2f6eed788b8df43d464a84ee886367b55fe608e6b8d6a.jpg)

![596e9e6f00d257d21a3d0e54bc18bfa9c1afb8804c5a4413bea2be95bcf25773.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/596e9e6f00d257d21a3d0e54bc18bfa9c1afb8804c5a4413bea2be95bcf25773.jpg)

![6587ce1e287b344cb2e7611f9478eea0d76192b95421386dbf77bf8a0dac4bd6.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/6587ce1e287b344cb2e7611f9478eea0d76192b95421386dbf77bf8a0dac4bd6.jpg)

![6775899cd830e097ed20c9df8c1d922648f13c4b59ec6141ee2a4187041e7f17.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/6775899cd830e097ed20c9df8c1d922648f13c4b59ec6141ee2a4187041e7f17.jpg)

![73ff7028be9edaca71917ec5f591f39aae242b6f6bbda4d887720e0516617504.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/73ff7028be9edaca71917ec5f591f39aae242b6f6bbda4d887720e0516617504.jpg)

![7ef28548b8b9180b9a958dadc6aa5aa1c2499d50220d7972749483f6d813095f.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/7ef28548b8b9180b9a958dadc6aa5aa1c2499d50220d7972749483f6d813095f.jpg)

![96075bfc547b136e26c9ae6f3e7821c0ea5b19b46f853f7455846026955afb93.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/96075bfc547b136e26c9ae6f3e7821c0ea5b19b46f853f7455846026955afb93.jpg)

![9ddb91d2542d135beeee870fa6d915b96baadabd7bf8579c4ec564d18cb4b59f.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/9ddb91d2542d135beeee870fa6d915b96baadabd7bf8579c4ec564d18cb4b59f.jpg)

![aeacb3f86b93c0f08175a406bee34d029450c0bf2265e03e06b13de5da582143.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/aeacb3f86b93c0f08175a406bee34d029450c0bf2265e03e06b13de5da582143.jpg)

![b0bafd58cff0d5d6ae47afc3e81b3c844b02680670bb17d45f9bb5935f849451.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/b0bafd58cff0d5d6ae47afc3e81b3c844b02680670bb17d45f9bb5935f849451.jpg)

![bb8ef9bc6514daf1706cf9edb361b82ef6d6acf19c5e83a2bd37e16676ef5f5f.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/bb8ef9bc6514daf1706cf9edb361b82ef6d6acf19c5e83a2bd37e16676ef5f5f.jpg)

![bc05a4f1c6c73ea9c3b9a984cb242aeff13f6e59b572ea9398e8564c1a8b83ff.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/bc05a4f1c6c73ea9c3b9a984cb242aeff13f6e59b572ea9398e8564c1a8b83ff.jpg)

![e71d8f861b80ca0e954087ee881d9400f7ba6e0f07809afa8973b84f72370962.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/e71d8f861b80ca0e954087ee881d9400f7ba6e0f07809afa8973b84f72370962.jpg)

![ee75a1b93aad343c468db27e97ddfda1a0bfe03be27e40f55fdeb4b3964be83c.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/ee75a1b93aad343c468db27e97ddfda1a0bfe03be27e40f55fdeb4b3964be83c.jpg)

![f23ba44244a3e138e8739047c231872eff475ac993de20a40ad5f040d7fca6cd.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/f23ba44244a3e138e8739047c231872eff475ac993de20a40ad5f040d7fca6cd.jpg)

![f3255c622db83ee356981080caadad7bd63a6e1730d877b2371c9aad80d6b41a.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/f3255c622db83ee356981080caadad7bd63a6e1730d877b2371c9aad80d6b41a.jpg)

![f4d1371eb9880473924ef816415bb670fff14823c01bc0e61ab6be6717865fac.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/f4d1371eb9880473924ef816415bb670fff14823c01bc0e61ab6be6717865fac.jpg)

![f773ffe94eeeb2511a5970c350a5e9531c3d4cab6dc5d568472abae4a3ee1a47.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/f773ffe94eeeb2511a5970c350a5e9531c3d4cab6dc5d568472abae4a3ee1a47.jpg)

![fad2a916f3234da93a904b4a91eb396624b1d491bdb3648236578f922f5eb369.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/fad2a916f3234da93a904b4a91eb396624b1d491bdb3648236578f922f5eb369.jpg)

![fbdeb5eed8c3c6678638c2d24bd00b9cc122def29edbef30e09da832af4d6fe9.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/images/fbdeb5eed8c3c6678638c2d24bd00b9cc122def29edbef30e09da832af4d6fe9.jpg)

### Tables

![00c152ffc5a4438f6616f96eae00db2ae8158fce6b387bcce3ca09d849550d38.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/00c152ffc5a4438f6616f96eae00db2ae8158fce6b387bcce3ca09d849550d38.jpg)

![343228e966f0bb5e36d58fa0e67c991928a8f8746ffdf7e75ea1bf75e1717fe8.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/343228e966f0bb5e36d58fa0e67c991928a8f8746ffdf7e75ea1bf75e1717fe8.jpg)

![39f7449dc53e67edfce74acf28f74333c32f742dd47f209c035cd07d8e97da27.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/39f7449dc53e67edfce74acf28f74333c32f742dd47f209c035cd07d8e97da27.jpg)

![44f31dab7fbe2bf286600c095c3e4d42127c622d7b1a1b04e36c7b560d94a6e6.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/44f31dab7fbe2bf286600c095c3e4d42127c622d7b1a1b04e36c7b560d94a6e6.jpg)

![51ed90ee1dac285527cd47c1f3079045020ff55aca0a1bc27e002b7722d061e4.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/51ed90ee1dac285527cd47c1f3079045020ff55aca0a1bc27e002b7722d061e4.jpg)

![528af3e84ff1910e4d55930c9e3aa69f00cbda6f05db29efbc710a0c0a90cb09.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/528af3e84ff1910e4d55930c9e3aa69f00cbda6f05db29efbc710a0c0a90cb09.jpg)

![5a8e40d52bb126938b31f71502f5c2f681255b4e96230e45c1b316249d8071e3.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/5a8e40d52bb126938b31f71502f5c2f681255b4e96230e45c1b316249d8071e3.jpg)

![5bcef4e9a9f7218e729b3ae6d2eeacb1190ec935365d4367ac147253ee4aaae7.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/5bcef4e9a9f7218e729b3ae6d2eeacb1190ec935365d4367ac147253ee4aaae7.jpg)

![5fd3cf70c699bc16c1dc36f1ef127bf0343039b193f41d0a177367ec4b368add.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/5fd3cf70c699bc16c1dc36f1ef127bf0343039b193f41d0a177367ec4b368add.jpg)

![8aa9984bbd3ba1d6ade781de346a526cbb0c65f00410bd17a62c176d259d7d3f.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/8aa9984bbd3ba1d6ade781de346a526cbb0c65f00410bd17a62c176d259d7d3f.jpg)

![a0364c79373ee91ce6d34799d0ae69c4b8e5839d4f979792a1ed25ab61f0fa9f.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/a0364c79373ee91ce6d34799d0ae69c4b8e5839d4f979792a1ed25ab61f0fa9f.jpg)

![a2f09850f14a8c5975967841a38c89ffe02dd4e322862104961cf1cff8a6dec6.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/a2f09850f14a8c5975967841a38c89ffe02dd4e322862104961cf1cff8a6dec6.jpg)

![a57e3f5f1f23d6d80fe2b4ab6553633648173dc079b42715d34f8b34eda791f0.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/a57e3f5f1f23d6d80fe2b4ab6553633648173dc079b42715d34f8b34eda791f0.jpg)

![ca5d6c9af11f40e3929d724968ba57445b88e9cb8c7195583d6a6ba9360885c2.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/ca5d6c9af11f40e3929d724968ba57445b88e9cb8c7195583d6a6ba9360885c2.jpg)

![d94cde8bb54552e84947c36842550a3ff60f04ef04bf82b2a8557d0405d5f629.jpg](../icml_results/4044_Who%20Evaluates%20AI%E2%80%99s%20Social%20Impacts_%20Mapping%20Coverage%20and%20Gaps%20in%20First%20and%20Third%20Party%20Evaluations/tables/d94cde8bb54552e84947c36842550a3ff60f04ef04bf82b2a8557d0405d5f629.jpg)

<a id="paper-4004"></a>

## Use What You Know: Causal Foundation Models with Partial Graphs


### Images

![059900e8f37c33ad4d73211b079d835ab79a2c6a47eb6c64d8d7af2e4820a6cb.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/059900e8f37c33ad4d73211b079d835ab79a2c6a47eb6c64d8d7af2e4820a6cb.jpg)

![08c55adc5147763581460ba3adbd378c4ea092bacaa070663aaf67c4c4cbf54f.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/08c55adc5147763581460ba3adbd378c4ea092bacaa070663aaf67c4c4cbf54f.jpg)

![099ebf17a0e3a489d4890c3537092d2d97f1c0790133bdca895433d3224c0f57.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/099ebf17a0e3a489d4890c3537092d2d97f1c0790133bdca895433d3224c0f57.jpg)

![27e0e65232fdf0b69eddd43ce6670399cb5b42617a27a7d28f20234875fe5d14.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/27e0e65232fdf0b69eddd43ce6670399cb5b42617a27a7d28f20234875fe5d14.jpg)

![28b62a62dcca35fba919f1087c3bd54812500e90f8ecf974efb0449202159bdf.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/28b62a62dcca35fba919f1087c3bd54812500e90f8ecf974efb0449202159bdf.jpg)

![2fac36ca4690868229130241b956e8438d58bd9d03d2a9be26aefbbd70f60218.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/2fac36ca4690868229130241b956e8438d58bd9d03d2a9be26aefbbd70f60218.jpg)

![49a1c6a8b431c98bea124ad80c179a5a16022d324c3109a22691e5ed9d8a74d9.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/49a1c6a8b431c98bea124ad80c179a5a16022d324c3109a22691e5ed9d8a74d9.jpg)

![4d17eb4945946950b38107699e7787f264a7c0160e3debc7cc94ffade7c51e87.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/4d17eb4945946950b38107699e7787f264a7c0160e3debc7cc94ffade7c51e87.jpg)

![4dd5e5a695d0b446b7f89055f1fe572280a2906ab6e86a53cfc4749a72edd3d0.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/4dd5e5a695d0b446b7f89055f1fe572280a2906ab6e86a53cfc4749a72edd3d0.jpg)

![51fa3c47d79b52211fa2cdbc6732bf26a3a473d432489b76cd6ba9f56918481f.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/51fa3c47d79b52211fa2cdbc6732bf26a3a473d432489b76cd6ba9f56918481f.jpg)

![558e63fa19924f182326deabfbae325440618b3ad8357c0ea4d2d3d0d30f5c8e.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/558e63fa19924f182326deabfbae325440618b3ad8357c0ea4d2d3d0d30f5c8e.jpg)

![6470e32ec99000b2ba22315e76aa29593c6532c6e239b891b26b6999219789af.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/6470e32ec99000b2ba22315e76aa29593c6532c6e239b891b26b6999219789af.jpg)

![6c16857b32e4817fc42f0538be923849f2bb3b48df67469eedce1e659789323a.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/6c16857b32e4817fc42f0538be923849f2bb3b48df67469eedce1e659789323a.jpg)

![6d3a8654fd69c2cde2bd07389f651d649250d6e3fe70f6d854357d7913afe3a3.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/6d3a8654fd69c2cde2bd07389f651d649250d6e3fe70f6d854357d7913afe3a3.jpg)

![72c766ef89c341b51248a10eb1ab1a5b3c1cfc937946f0ddf3272f3da20a30fc.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/72c766ef89c341b51248a10eb1ab1a5b3c1cfc937946f0ddf3272f3da20a30fc.jpg)

![75132064911998b5ef13a34b9ab1c1fd11e36267aa44c391653fdc3e282d62a7.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/75132064911998b5ef13a34b9ab1c1fd11e36267aa44c391653fdc3e282d62a7.jpg)

![76846ae8e29ca60067c4ca21c288f67ccd142923c877ef11820b57963fc699ec.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/76846ae8e29ca60067c4ca21c288f67ccd142923c877ef11820b57963fc699ec.jpg)

![8258422d87471c7ee1b7b8e4100f03bd9e21ac54f05e2b62bae7160267498f45.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/8258422d87471c7ee1b7b8e4100f03bd9e21ac54f05e2b62bae7160267498f45.jpg)

![8bd77f9805b3376a039488e51524771038919c6f5408ebb3c51a691d664fb63f.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/8bd77f9805b3376a039488e51524771038919c6f5408ebb3c51a691d664fb63f.jpg)

![a3fdeaf08a1489b5bd7787f96214d8d5d82d26af87ff25781ab273512142e689.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/a3fdeaf08a1489b5bd7787f96214d8d5d82d26af87ff25781ab273512142e689.jpg)

![a78e7f7d9733864ae3fc4eb37d7e76fd930d4d9b7bbd247186232bdf08a02e24.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/a78e7f7d9733864ae3fc4eb37d7e76fd930d4d9b7bbd247186232bdf08a02e24.jpg)

![b4c3b1265e31a307aefa4753216681708741909020aaff38f37782cf2f59f1c8.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/b4c3b1265e31a307aefa4753216681708741909020aaff38f37782cf2f59f1c8.jpg)

![ec88ae4d60dd81f6ca188e6d3e273b0886b1adceda11e2760653fa0d3c71fe7f.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/images/ec88ae4d60dd81f6ca188e6d3e273b0886b1adceda11e2760653fa0d3c71fe7f.jpg)

### Tables

![11392c781feccdcf8edaca04b8588e5397dcdddd8ac964ff685c372da0f5ab79.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/tables/11392c781feccdcf8edaca04b8588e5397dcdddd8ac964ff685c372da0f5ab79.jpg)

![91e7d0fd52448e7d0c762ec8aa780153bf39b8c8f9ce851ed43fba52e652f039.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/tables/91e7d0fd52448e7d0c762ec8aa780153bf39b8c8f9ce851ed43fba52e652f039.jpg)

![e497ee05a67af386bf604b29245407ebe3723c84efb6e1c5db5db1c8168a5170.jpg](../icml_results/4045_Use%20What%20You%20Know_%20Causal%20Foundation%20Models%20with%20Partial%20Graphs/tables/e497ee05a67af386bf604b29245407ebe3723c84efb6e1c5db5db1c8168a5170.jpg)

<a id="paper-4005"></a>

## Online Rubrics Elicitation from Pairwise Comparisons


### Images

![2bc86d13229d44bf32dacceeb93d3c2e2106e507fd17598703d035a2b8524154.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/2bc86d13229d44bf32dacceeb93d3c2e2106e507fd17598703d035a2b8524154.jpg)

![7004bd995934828e13970e0f0f86b0dfb919632d554f39372e66a79a07926e6a.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/7004bd995934828e13970e0f0f86b0dfb919632d554f39372e66a79a07926e6a.jpg)

![7e9c76f077a69b76b8d99dba4032e10f9fd9b45d16dc41ad6610e774afc7fe90.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/7e9c76f077a69b76b8d99dba4032e10f9fd9b45d16dc41ad6610e774afc7fe90.jpg)

![cdce08d24bee579d503348671f81f7de3178f31dacf1d03ef02153eb56087d0f.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/cdce08d24bee579d503348671f81f7de3178f31dacf1d03ef02153eb56087d0f.jpg)

![f5e4a3d32e1fd56ae84c4e0694ffc2aed26b4fda861e77c48eb28050ec0f2eb5.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/f5e4a3d32e1fd56ae84c4e0694ffc2aed26b4fda861e77c48eb28050ec0f2eb5.jpg)

![faa09f26191083fdeaebacfab6ca38a35173e5eefc01180252e5385dbb3f2f40.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/images/faa09f26191083fdeaebacfab6ca38a35173e5eefc01180252e5385dbb3f2f40.jpg)

### Tables

![175710fa76abb231b6ca2142d683047f3b4b2efa2874a5e8142c77ffd2c13687.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/175710fa76abb231b6ca2142d683047f3b4b2efa2874a5e8142c77ffd2c13687.jpg)

![179b8c2749769258e84cdca0cf3e5c09943a8d202b6e5a4193503720be354061.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/179b8c2749769258e84cdca0cf3e5c09943a8d202b6e5a4193503720be354061.jpg)

![182866afd8da83d421a7b484592ebf9f057614878aeef7ec9d34eb93796e902a.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/182866afd8da83d421a7b484592ebf9f057614878aeef7ec9d34eb93796e902a.jpg)

![2f919d0110dc995150879037a01872f6ce68a1fb034ae9ecd431b1ae0e46e1b2.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/2f919d0110dc995150879037a01872f6ce68a1fb034ae9ecd431b1ae0e46e1b2.jpg)

![3a3479c3393dfbedea77db5bf908ff46c8f4aae087fddf3b08f489e71a973352.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/3a3479c3393dfbedea77db5bf908ff46c8f4aae087fddf3b08f489e71a973352.jpg)

![64d57b2e13f9752f48e4d815cbde42e6a6ee9a3a7193668b5112cceac3a471a0.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/64d57b2e13f9752f48e4d815cbde42e6a6ee9a3a7193668b5112cceac3a471a0.jpg)

![69237fe10fd97c455028be7f06b169f6abfdcff2f6eded6fae82234231c3b6cd.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/69237fe10fd97c455028be7f06b169f6abfdcff2f6eded6fae82234231c3b6cd.jpg)

![6e329474c0ec24689926b6bd26dd059d1f446d214aaba616fdf74d6e5c3b6a89.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/6e329474c0ec24689926b6bd26dd059d1f446d214aaba616fdf74d6e5c3b6a89.jpg)

![711cbb03b7e85b7587ee9a0513e3b321deec50c36a8d6b75e0117c32964f3c61.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/711cbb03b7e85b7587ee9a0513e3b321deec50c36a8d6b75e0117c32964f3c61.jpg)

![7170a0e2de2889972b15013a6c27cdb62fef6e62d221914aee731acd738ea6ad.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/7170a0e2de2889972b15013a6c27cdb62fef6e62d221914aee731acd738ea6ad.jpg)

![7f8470823dc2a63af4629a13eedd199ba353454f2a50d367f8b0dc2a2cccc66c.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/7f8470823dc2a63af4629a13eedd199ba353454f2a50d367f8b0dc2a2cccc66c.jpg)

![8da64450d9dd562c1b556a68806ce9d0608e1361cd09c12374524a31f1f5d306.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/8da64450d9dd562c1b556a68806ce9d0608e1361cd09c12374524a31f1f5d306.jpg)

![99912de795cf9f7e766efe5153e914f4ccc579f94e9bb052e7da89bfb2ba7edb.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/99912de795cf9f7e766efe5153e914f4ccc579f94e9bb052e7da89bfb2ba7edb.jpg)

![b0119869325063cdbe50cec1a2dd08a8dcb16515792f356d80b511f284cc250c.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/b0119869325063cdbe50cec1a2dd08a8dcb16515792f356d80b511f284cc250c.jpg)

![dd1e7e02b7813e25015fc5b602961c29b282bf0bcdcf924c4ae8851eff67291d.jpg](../icml_results/4046_Online%20Rubrics%20Elicitation%20from%20Pairwise%20Comparisons/tables/dd1e7e02b7813e25015fc5b602961c29b282bf0bcdcf924c4ae8851eff67291d.jpg)

<a id="paper-4006"></a>

## Weight-Space Learning for Certifiable Few-shot Transfer Learning


### Images

![267f5c750d00506fdf7d75ab8ca14bdb71f0bef705ffe999656b6c8b9f93987a.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/267f5c750d00506fdf7d75ab8ca14bdb71f0bef705ffe999656b6c8b9f93987a.jpg)

![31b65ecfc18d959a81ec3056a5b0595434b7bd0367b09e7124d4217385483298.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/31b65ecfc18d959a81ec3056a5b0595434b7bd0367b09e7124d4217385483298.jpg)

![70d0223122d3bf0f6eb9aa8e848cccaa2950032eccac746456d4d228d403d502.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/70d0223122d3bf0f6eb9aa8e848cccaa2950032eccac746456d4d228d403d502.jpg)

![dd13f263c79de83c405488e8c56478b0ac735bbc4bf66950b61fc04be7fde82e.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/dd13f263c79de83c405488e8c56478b0ac735bbc4bf66950b61fc04be7fde82e.jpg)

![e39bef32ebbe7c2410d0f2a0f318101b932c72dbec75b91583fec8319f54b084.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/e39bef32ebbe7c2410d0f2a0f318101b932c72dbec75b91583fec8319f54b084.jpg)

![ff23734c98e8a8cda18a0ca99758147c58fc1e888a13cf584e5ab57598804a3b.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/images/ff23734c98e8a8cda18a0ca99758147c58fc1e888a13cf584e5ab57598804a3b.jpg)

### Tables

![029dc307e15dd8fbb5bd7ea198185d08385d699c99ef564d02e5baf9827a2800.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/029dc307e15dd8fbb5bd7ea198185d08385d699c99ef564d02e5baf9827a2800.jpg)

![32954d2f63bc6fe2f44fe7bad8bcc113839eb0fb8e19dcf34477fdc93c48d227.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/32954d2f63bc6fe2f44fe7bad8bcc113839eb0fb8e19dcf34477fdc93c48d227.jpg)

![48fdf29774dc197611df2b7cbd059adedc0a5160b4df07a590fe396cae503412.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/48fdf29774dc197611df2b7cbd059adedc0a5160b4df07a590fe396cae503412.jpg)

![4902a86a6218b839a678f6b0f05757742895295f7417ccd3368743547a409211.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/4902a86a6218b839a678f6b0f05757742895295f7417ccd3368743547a409211.jpg)

![4aaefe2e2ddf31ab82d1cbdfa5b9c0f705374cc8c5411847a03d6417370dd3d2.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/4aaefe2e2ddf31ab82d1cbdfa5b9c0f705374cc8c5411847a03d6417370dd3d2.jpg)

![508ac06edc08147df2f65296cc3198dfd4a437b2aa7d41e6bef993aeaa521c11.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/508ac06edc08147df2f65296cc3198dfd4a437b2aa7d41e6bef993aeaa521c11.jpg)

![6647554f99ed60e60c7c136acb2386a5029203cf4f318ec5d529ec673d18afdf.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/6647554f99ed60e60c7c136acb2386a5029203cf4f318ec5d529ec673d18afdf.jpg)

![93ecb2500c021b2de8f391235069ef0c4f2e117162b81178f0e9513dbcb81c1f.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/93ecb2500c021b2de8f391235069ef0c4f2e117162b81178f0e9513dbcb81c1f.jpg)

![a4185805d2c3585642fd08fe8f1b94621f529e554a6396837d30086756375924.jpg](../icml_results/4047_Weight-Space%20Learning%20for%20Certifiable%20Few-shot%20Transfer%20Learning/tables/a4185805d2c3585642fd08fe8f1b94621f529e554a6396837d30086756375924.jpg)

<a id="paper-4007"></a>

## From Muon to Gluon: Bridging Theory and Practice of LMO-based Optimizers for LLMs


### Images

![03f8e0b2b6d5ad0193293da67f84c8546f979712804d4b8340b14e2306116693.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/03f8e0b2b6d5ad0193293da67f84c8546f979712804d4b8340b14e2306116693.jpg)

![06dcbb646c7232d4134958a97d25d4d45631bd81223ed3a7cc1111af1a65b4fe.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/06dcbb646c7232d4134958a97d25d4d45631bd81223ed3a7cc1111af1a65b4fe.jpg)

![10044ee76f701d317b327125266b86ec814da7306bfb2db8577d36e1e70ffb49.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/10044ee76f701d317b327125266b86ec814da7306bfb2db8577d36e1e70ffb49.jpg)

![19027f39552525fd9fffefb60fbf51422728facb2088d3ae25342e720d325346.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/19027f39552525fd9fffefb60fbf51422728facb2088d3ae25342e720d325346.jpg)

![2327deed8b1c23e5afa29c579f190aee3c771ed5c93ca42952aa0dc12452fdf3.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/2327deed8b1c23e5afa29c579f190aee3c771ed5c93ca42952aa0dc12452fdf3.jpg)

![2bb3a418b27ca6eba24091034c2551330e2f4be30927e53b487055d419bcaf04.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/2bb3a418b27ca6eba24091034c2551330e2f4be30927e53b487055d419bcaf04.jpg)

![49a14f54c968e167226890044298a4f79091aa30847ea34473093a6459f3979e.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/49a14f54c968e167226890044298a4f79091aa30847ea34473093a6459f3979e.jpg)

![4b9d78d10c52693775074fd6292350137139d0560e64334330c58ded75d9824d.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/4b9d78d10c52693775074fd6292350137139d0560e64334330c58ded75d9824d.jpg)

![697bcae3605bfa80d9004364c7eceb7220fa40c03f57d1cc5e444b9d2f3510e5.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/697bcae3605bfa80d9004364c7eceb7220fa40c03f57d1cc5e444b9d2f3510e5.jpg)

![71694523beda05ee77442b775569931035c9580f36812ab4ebe1333cd6c85e39.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/71694523beda05ee77442b775569931035c9580f36812ab4ebe1333cd6c85e39.jpg)

![86ba74e937684847d8194c2c5509d7be69a1723a53a7f26a21bd8fbaab3dcece.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/86ba74e937684847d8194c2c5509d7be69a1723a53a7f26a21bd8fbaab3dcece.jpg)

![9fe59d799e610317081814bb8aedd06810a91c044a69bb945c53b12782978e52.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/9fe59d799e610317081814bb8aedd06810a91c044a69bb945c53b12782978e52.jpg)

![d9c510abd7e987da60da4361e2807e57adb0e85c42559912c81cef4a4aa43884.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/d9c510abd7e987da60da4361e2807e57adb0e85c42559912c81cef4a4aa43884.jpg)

![db5e7bacef4b08853b2b6d6af2ecc55b0381e8f352e8f0bc2c2f1f6637e45747.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/db5e7bacef4b08853b2b6d6af2ecc55b0381e8f352e8f0bc2c2f1f6637e45747.jpg)

![dd1862d1acebbf258a42bf0f66dad4d21e1a3b2c7941842d234f4bb6b0112cbe.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/dd1862d1acebbf258a42bf0f66dad4d21e1a3b2c7941842d234f4bb6b0112cbe.jpg)

![dd7fbde615c2628a3e5dde5fb22875a4c967649fffe56cca3b2a5a46a4f0ab2b.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/dd7fbde615c2628a3e5dde5fb22875a4c967649fffe56cca3b2a5a46a4f0ab2b.jpg)

![e30939d9858f083425187baa13d6530bb2bd67e56d3d90ad8894227b850c72cc.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/e30939d9858f083425187baa13d6530bb2bd67e56d3d90ad8894227b850c72cc.jpg)

![f13017f497c7790166649d8b0f83fb216fc1f9dccd77e539e76516451ee6eda9.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/images/f13017f497c7790166649d8b0f83fb216fc1f9dccd77e539e76516451ee6eda9.jpg)

### Tables

![afd497b2a2357920a4f7e4706282570084045ee7d1b7e1e888f4bf9060aa9bbb.jpg](../icml_results/4048_From%20Muon%20to%20Gluon_%20Bridging%20Theory%20and%20Practice%20of%20LMO-based%20Optimizers%20for%20LLMs/tables/afd497b2a2357920a4f7e4706282570084045ee7d1b7e1e888f4bf9060aa9bbb.jpg)

<a id="paper-4008"></a>

## A Fourier perspective on the learning dynamics of neural networks: from sample complexities to mechanistic insights


### Images

![0daf437ecb05f5a8b45a8477ad761abc791a1aaa870be2af11f7ba7e14e0e3fb.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/0daf437ecb05f5a8b45a8477ad761abc791a1aaa870be2af11f7ba7e14e0e3fb.jpg)

![27e6a56e263956e18252e7f8598dedad7c9a7c4fdaba4fe17552c35e2d7b41ce.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/27e6a56e263956e18252e7f8598dedad7c9a7c4fdaba4fe17552c35e2d7b41ce.jpg)

![433b855907c9b71628d991e046f42342eaabedae33073cb57bc8a6e8da720552.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/433b855907c9b71628d991e046f42342eaabedae33073cb57bc8a6e8da720552.jpg)

![813daa9115e2c59c7bebcb0547cd8b7d13d46a70d0c3892a16864789c6be615e.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/813daa9115e2c59c7bebcb0547cd8b7d13d46a70d0c3892a16864789c6be615e.jpg)

![840a89aa32ddab6435cfe66fb9f0e8ac2ba0d57bd96a961b7b0f1ae4fd840366.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/840a89aa32ddab6435cfe66fb9f0e8ac2ba0d57bd96a961b7b0f1ae4fd840366.jpg)

![88fbdb22fcf9665afb1cb66ed282fd10155919b0c5091e6c02048e14b6f27f94.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/88fbdb22fcf9665afb1cb66ed282fd10155919b0c5091e6c02048e14b6f27f94.jpg)

![921ad009b8959bc3562acaf87d03de23d6f36503030b33e863552ee18ba41095.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/921ad009b8959bc3562acaf87d03de23d6f36503030b33e863552ee18ba41095.jpg)

![96360f4d57b348dd150dab961920e8095d51e478712ce0705cc313dafda0196d.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/96360f4d57b348dd150dab961920e8095d51e478712ce0705cc313dafda0196d.jpg)

![9f5e669a74e72267af7ab918268cba253a7b4ed79f256fb52fc6f2d18b7d0cea.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/9f5e669a74e72267af7ab918268cba253a7b4ed79f256fb52fc6f2d18b7d0cea.jpg)

![d902c6be48536d203452b740570a3874d3d49ca6cb8e304ef77e446773c84b13.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/d902c6be48536d203452b740570a3874d3d49ca6cb8e304ef77e446773c84b13.jpg)

![f0fccbd4435ed2765db8469e5fc72d0045613d69b1ac3eba9e0ba4836de3ed8b.jpg](../icml_results/4049_A%20Fourier%20perspective%20on%20the%20learning%20dynamics%20of%20neural%20networks_%20from%20sample%20complexities%20to%20mecha/images/f0fccbd4435ed2765db8469e5fc72d0045613d69b1ac3eba9e0ba4836de3ed8b.jpg)

<a id="paper-4009"></a>

## A Geometric Analysis of Small-sized Language Model Hallucinations


### Images

![25e47db5bddb4151d0834a2e69559c8b3bb711179bf574ae96e155b781891822.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/25e47db5bddb4151d0834a2e69559c8b3bb711179bf574ae96e155b781891822.jpg)

![7e0a87a30c5490e71e39743cae18c6c174590dd6e01c5d9c5db9060a51d08782.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/7e0a87a30c5490e71e39743cae18c6c174590dd6e01c5d9c5db9060a51d08782.jpg)

![7f199b351fa9da950d3e4df2ca9201bdbfab70007b0472bccfa767b5365ef126.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/7f199b351fa9da950d3e4df2ca9201bdbfab70007b0472bccfa767b5365ef126.jpg)

![87165df6acc9ebb31ddc2b598e965a97f48a48139e029bdf5facd963b3429217.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/87165df6acc9ebb31ddc2b598e965a97f48a48139e029bdf5facd963b3429217.jpg)

![8f6cf04b803ebc1a36d46d827427ac203a692dd83ae51263c77e2d9ef51bcbe9.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/8f6cf04b803ebc1a36d46d827427ac203a692dd83ae51263c77e2d9ef51bcbe9.jpg)

![9c990077bebd000b451dfd861c1371d04c76159130d3a153dee48f630a38a252.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/9c990077bebd000b451dfd861c1371d04c76159130d3a153dee48f630a38a252.jpg)

![bac4e7d584c8fddf3f092c3a5aba523050fd6418fc63f8af9480ae11fb6125b7.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/bac4e7d584c8fddf3f092c3a5aba523050fd6418fc63f8af9480ae11fb6125b7.jpg)

![c19cb5584c7a55d4b5e351f197c474f54c8bd9a8d919a3b36229f1a0a73be769.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/c19cb5584c7a55d4b5e351f197c474f54c8bd9a8d919a3b36229f1a0a73be769.jpg)

![c3e63a5cf99c1d6139b97055ce509811f2a9ad296f026eb7b7074860cf5c47a9.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/c3e63a5cf99c1d6139b97055ce509811f2a9ad296f026eb7b7074860cf5c47a9.jpg)

![c8e9a9bde00e71a1659e8d5dec96efca0e00e06c7b33e2125b1d9bc2b85345a8.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/c8e9a9bde00e71a1659e8d5dec96efca0e00e06c7b33e2125b1d9bc2b85345a8.jpg)

![de33ca071c9b34d540a5b727fc80fd29c3bcf192c0d1d17d30317242e816d13f.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/de33ca071c9b34d540a5b727fc80fd29c3bcf192c0d1d17d30317242e816d13f.jpg)

![ed2f8c29e8f57484c5e2bc21b66596368a35c248e11c07238a2195e2b00f79f9.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/ed2f8c29e8f57484c5e2bc21b66596368a35c248e11c07238a2195e2b00f79f9.jpg)

![fd838d629dec120a04016a216a7dbd3b829401b18b7c2b2a7f3dd2e037892526.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/images/fd838d629dec120a04016a216a7dbd3b829401b18b7c2b2a7f3dd2e037892526.jpg)

### Tables

![00e7e8da93444fdba8ccdeae04bb9587043b1fd2069afe0609bc3ec2ea416d32.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/00e7e8da93444fdba8ccdeae04bb9587043b1fd2069afe0609bc3ec2ea416d32.jpg)

![0486f7442f0e9a31c592e4768768da277f25a0a216710dfd4f76f9459dabe153.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/0486f7442f0e9a31c592e4768768da277f25a0a216710dfd4f76f9459dabe153.jpg)

![0d8f7f485192996f389746e201acace3b028746aa305d402748b67b63ad0c90d.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/0d8f7f485192996f389746e201acace3b028746aa305d402748b67b63ad0c90d.jpg)

![15222de35cb9793c75f530a692df096222133d2600e67054bdf06cf734e9c639.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/15222de35cb9793c75f530a692df096222133d2600e67054bdf06cf734e9c639.jpg)

![26e5ea9f8c531491766f193d2e77cceb63ff53b6658c6b7a475ee1c226115666.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/26e5ea9f8c531491766f193d2e77cceb63ff53b6658c6b7a475ee1c226115666.jpg)

![3728bb186432aabef352df36070afe7c078ebba3db01012440dcfdd1e3427e06.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/3728bb186432aabef352df36070afe7c078ebba3db01012440dcfdd1e3427e06.jpg)

![373c077e02b9312dbfe29d3ff25d0eeaa88045779ae6fa8b8dd4336893e56c4d.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/373c077e02b9312dbfe29d3ff25d0eeaa88045779ae6fa8b8dd4336893e56c4d.jpg)

![3942585aad260dad0b660fbdf5d94153083ebacf4da7de2552ded7df637a6fad.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/3942585aad260dad0b660fbdf5d94153083ebacf4da7de2552ded7df637a6fad.jpg)

![3da35b0d4bf33ff2a1c6b4ca64c1b66f27ebbdb8e4b1b392342a8c3b858e30c7.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/3da35b0d4bf33ff2a1c6b4ca64c1b66f27ebbdb8e4b1b392342a8c3b858e30c7.jpg)

![76baefa4cbb6a1035c7eb6a94cc3a9f76c10fdee163f54c9a3fa91548a92983b.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/76baefa4cbb6a1035c7eb6a94cc3a9f76c10fdee163f54c9a3fa91548a92983b.jpg)

![82ecaf3611fc1df9176e3b5fb6831072fdfaf6acf63ab8a5feb327d90ff94017.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/82ecaf3611fc1df9176e3b5fb6831072fdfaf6acf63ab8a5feb327d90ff94017.jpg)

![8a1f8718b7122f8ab9be035a91453296f50c3a4e8cce666161ee651c669f8c40.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/8a1f8718b7122f8ab9be035a91453296f50c3a4e8cce666161ee651c669f8c40.jpg)

![95b1437bcdc82b2973aba372de4b23e645c7e4169fe58ec3d33270e60f533d37.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/95b1437bcdc82b2973aba372de4b23e645c7e4169fe58ec3d33270e60f533d37.jpg)

![a91364e8fcf951c80068ac7314edaf5331f157f4fb0a589ca6dc0813cd2b8dd5.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/a91364e8fcf951c80068ac7314edaf5331f157f4fb0a589ca6dc0813cd2b8dd5.jpg)

![b3619cb9a3c3b08f559f0f230292936623fb4b0fa8ee598504ac85b928cc7839.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/b3619cb9a3c3b08f559f0f230292936623fb4b0fa8ee598504ac85b928cc7839.jpg)

![bad3dbd04524d48e0f870eea8faf67c5fd6efad8deeb22a9347f0de8ca68192b.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/bad3dbd04524d48e0f870eea8faf67c5fd6efad8deeb22a9347f0de8ca68192b.jpg)

![bce951c3b47817c126e541f1386f08926198bc05711c5e1b73d68b0c499da2d0.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/bce951c3b47817c126e541f1386f08926198bc05711c5e1b73d68b0c499da2d0.jpg)

![c8365cc41db7eaf8d11de38e8dd461e5d05075f07bd0ae94d783673e90c70b2b.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/c8365cc41db7eaf8d11de38e8dd461e5d05075f07bd0ae94d783673e90c70b2b.jpg)

![cbe9f64a3eefe3152df31b6f3746ffd7244b5b44549c22363c00ef61849c61ab.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/cbe9f64a3eefe3152df31b6f3746ffd7244b5b44549c22363c00ef61849c61ab.jpg)

![ec24d817f38fec7ea32f0e649cb4659b492dbe9f4919de6e9e332172cd4b8edb.jpg](../icml_results/4050_A%20Geometric%20Analysis%20of%20Small-sized%20Language%20Model%20Hallucinations/tables/ec24d817f38fec7ea32f0e649cb4659b492dbe9f4919de6e9e332172cd4b8edb.jpg)

<a id="paper-4010"></a>

## Improved Bounds for Reward-Agnostic and Reward-Free Exploration


### Images

![057aa8bcbd732052b36b960ffdedec0a8837c9aef9dc7828239ef0e4c6816620.jpg](../icml_results/4051_Improved%20Bounds%20for%20Reward-Agnostic%20and%20Reward-Free%20Exploration/images/057aa8bcbd732052b36b960ffdedec0a8837c9aef9dc7828239ef0e4c6816620.jpg)

![068628eebeb7cefa45d4586b79744758c85084a1f68e587fc81d33dac7c41e13.jpg](../icml_results/4051_Improved%20Bounds%20for%20Reward-Agnostic%20and%20Reward-Free%20Exploration/images/068628eebeb7cefa45d4586b79744758c85084a1f68e587fc81d33dac7c41e13.jpg)

### Tables

![df763c094a63e475e4bf30c621f876ebc30167e73c5a0bd28affc034cd74f1dd.jpg](../icml_results/4051_Improved%20Bounds%20for%20Reward-Agnostic%20and%20Reward-Free%20Exploration/tables/df763c094a63e475e4bf30c621f876ebc30167e73c5a0bd28affc034cd74f1dd.jpg)

<a id="paper-4011"></a>

## Optimal conversion from Rényi Differential Privacy to $f$-Differential Privacy


### Images

![2f420e920457411c5c42412e2f0e07bb654b5b0d361346df7a0e96a4a6a3ec71.jpg](../icml_results/4052_Optimal%20conversion%20from%20R%C3%A9nyi%20Differential%20Privacy%20to%20%24f%24-Differential%20Privacy/images/2f420e920457411c5c42412e2f0e07bb654b5b0d361346df7a0e96a4a6a3ec71.jpg)

![d454c14ebf1fdc2591457813009700f511a55e276b3bd4239c669fc16b4efc83.jpg](../icml_results/4052_Optimal%20conversion%20from%20R%C3%A9nyi%20Differential%20Privacy%20to%20%24f%24-Differential%20Privacy/images/d454c14ebf1fdc2591457813009700f511a55e276b3bd4239c669fc16b4efc83.jpg)

![eec3a59e33b9731578affa57961bd2a8e9501d2750a645f4741703fc91a1f48a.jpg](../icml_results/4052_Optimal%20conversion%20from%20R%C3%A9nyi%20Differential%20Privacy%20to%20%24f%24-Differential%20Privacy/images/eec3a59e33b9731578affa57961bd2a8e9501d2750a645f4741703fc91a1f48a.jpg)

<a id="paper-4012"></a>

## Transporting Task Vectors across Different Architectures without Training


### Images

![047be3c118e90bca9c38988bbb7fdca4c992423bb12592d0b4a91ef0bf9ebd0e.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/images/047be3c118e90bca9c38988bbb7fdca4c992423bb12592d0b4a91ef0bf9ebd0e.jpg)

![18d821b489636e21e048e851a57e4fb3aa87b03ba8d6bf8e98e7b04b3f0b10f7.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/images/18d821b489636e21e048e851a57e4fb3aa87b03ba8d6bf8e98e7b04b3f0b10f7.jpg)

![48d25cf30b33bc8f17f5f2e8726323050b7801ddce02686945b18d73a7cdeaa5.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/images/48d25cf30b33bc8f17f5f2e8726323050b7801ddce02686945b18d73a7cdeaa5.jpg)

![b1c614aa3b692e7594a27497e796d189888cfaffb1e96a96cc2cb9055c23f897.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/images/b1c614aa3b692e7594a27497e796d189888cfaffb1e96a96cc2cb9055c23f897.jpg)

![f4a6a3f04cb144021a67fcf7cbfc8305ba9b2c8c19767cb3d12806651df350a2.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/images/f4a6a3f04cb144021a67fcf7cbfc8305ba9b2c8c19767cb3d12806651df350a2.jpg)

### Tables

![259ba95fd0a01b8d0304b4af3bae9f07e2db55a2363c3c11ca45be8ceb62294a.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/259ba95fd0a01b8d0304b4af3bae9f07e2db55a2363c3c11ca45be8ceb62294a.jpg)

![2f8854abcf085c36fcf658247062472f28cb2cbd5a3e803eb69823ecd4ca12a3.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/2f8854abcf085c36fcf658247062472f28cb2cbd5a3e803eb69823ecd4ca12a3.jpg)

![51cb21393a4e81e47ecd2caa58cf8ecb3ea21e0b94ba87fe543225c79a6cf3d8.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/51cb21393a4e81e47ecd2caa58cf8ecb3ea21e0b94ba87fe543225c79a6cf3d8.jpg)

![5e1b9df98e9c9790371fbfaca19ef048fa90ae1dcb21f09457ab2abf8a97ec7b.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/5e1b9df98e9c9790371fbfaca19ef048fa90ae1dcb21f09457ab2abf8a97ec7b.jpg)

![6080c391a94e2f865b74f1154abbc044ea67c63cf2bd38213c90e0e784268243.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/6080c391a94e2f865b74f1154abbc044ea67c63cf2bd38213c90e0e784268243.jpg)

![6a33ba5048c7ef9b93750ed4ac86851538e6339e1401ff054fe038081a46eb45.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/6a33ba5048c7ef9b93750ed4ac86851538e6339e1401ff054fe038081a46eb45.jpg)

![785d5feefe06d6df912a29f4e5c7c115576eccae26b77e417b09c9c81be57038.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/785d5feefe06d6df912a29f4e5c7c115576eccae26b77e417b09c9c81be57038.jpg)

![977c7577570a5934507ac372f696766d117b87020bd147e18925eae20d7c22ae.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/977c7577570a5934507ac372f696766d117b87020bd147e18925eae20d7c22ae.jpg)

![a2f59f0f7d2732844074a050f7e6a5801489fd44f0a26dccd965fe176b96d2b9.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/a2f59f0f7d2732844074a050f7e6a5801489fd44f0a26dccd965fe176b96d2b9.jpg)

![aaabb2850c0508adc891194a483a9a57a997a99011b79c89dba9afe4934260b5.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/aaabb2850c0508adc891194a483a9a57a997a99011b79c89dba9afe4934260b5.jpg)

![ad0a3d14bc5027faca445314b7f036be32930400b52fad7caf1b8c202269f2c2.jpg](../icml_results/4053_Transporting%20Task%20Vectors%20across%20Different%20Architectures%20without%20Training/tables/ad0a3d14bc5027faca445314b7f036be32930400b52fad7caf1b8c202269f2c2.jpg)

<a id="paper-4013"></a>

## Attacks on Machine-Text Detectors Retain Stylistic Fingerprints


### Images

![088a1be55852cff4c0c35b4a077cf9439ad740dddc1145775cfa459e533a0301.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/088a1be55852cff4c0c35b4a077cf9439ad740dddc1145775cfa459e533a0301.jpg)

![6b3e9241db8164540106c5a49a2f3de49efceeb721dd25f0be6b842e47c7b153.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/6b3e9241db8164540106c5a49a2f3de49efceeb721dd25f0be6b842e47c7b153.jpg)

![731e5a3deebcc119a15e001b4cc354e6afd8d6b6a765c689fd76477a14b206a9.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/731e5a3deebcc119a15e001b4cc354e6afd8d6b6a765c689fd76477a14b206a9.jpg)

![81eb8bac19bd619f83ea806cbee392d49edb3d58647b45fc307b4c08752ae115.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/81eb8bac19bd619f83ea806cbee392d49edb3d58647b45fc307b4c08752ae115.jpg)

![c3b6c18ba8a49a0d2203de2b8be5cc5fd067ddb2830134b6a94bcfa47483a718.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/c3b6c18ba8a49a0d2203de2b8be5cc5fd067ddb2830134b6a94bcfa47483a718.jpg)

![d56ebed38264a5e0825280f572a6ed612cb8858d5174a0b3a54d9fd31c19f66c.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/d56ebed38264a5e0825280f572a6ed612cb8858d5174a0b3a54d9fd31c19f66c.jpg)

![e0baff9e7b11652d9705691799eea2068d572f2b58ac2b7b491d2d0328ebda77.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/e0baff9e7b11652d9705691799eea2068d572f2b58ac2b7b491d2d0328ebda77.jpg)

![fddf2079bc4e9668547f3e959b541381e7c9988b3d841e078c7cee3bace43840.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/images/fddf2079bc4e9668547f3e959b541381e7c9988b3d841e078c7cee3bace43840.jpg)

### Tables

![050320a1f523f1e50f13843da6ff177c982017c7bfb8c72a02d3bceaf60e17cc.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/050320a1f523f1e50f13843da6ff177c982017c7bfb8c72a02d3bceaf60e17cc.jpg)

![10d3455983bd6e01793231e6f04761e5d756c774ab6c5d4b0c82f3c9cad77ee4.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/10d3455983bd6e01793231e6f04761e5d756c774ab6c5d4b0c82f3c9cad77ee4.jpg)

![129e01eefa29f1d5c755170259ba4d8fa045c3122cef7ea294e26c1cc39fb525.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/129e01eefa29f1d5c755170259ba4d8fa045c3122cef7ea294e26c1cc39fb525.jpg)

![2269037229178bee04aef6eff0cec47051fc72f03b50b1c7fdf7bc8d5d277311.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/2269037229178bee04aef6eff0cec47051fc72f03b50b1c7fdf7bc8d5d277311.jpg)

![29689653f276da8d3d655e14d6e08e0f6f3210ec5e21bc940e41961db37d0568.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/29689653f276da8d3d655e14d6e08e0f6f3210ec5e21bc940e41961db37d0568.jpg)

![37cd9b106cee7d7b4854574f91c6442461441eb7e48b5f3883ea7f195d94d076.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/37cd9b106cee7d7b4854574f91c6442461441eb7e48b5f3883ea7f195d94d076.jpg)

![4126bcccbb4b0ece00a1ff01034531abf355e50cb3257f5f12faddacd735adb9.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/4126bcccbb4b0ece00a1ff01034531abf355e50cb3257f5f12faddacd735adb9.jpg)

![4431181f092d7c6335a177eca1e52cd68fd8806d054b22008f6a3b8338597d6b.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/4431181f092d7c6335a177eca1e52cd68fd8806d054b22008f6a3b8338597d6b.jpg)

![500913b37add57af1f59a0d95918ce9c56053ee058884ce1d378cb8d531000c9.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/500913b37add57af1f59a0d95918ce9c56053ee058884ce1d378cb8d531000c9.jpg)

![5af585878106176ed03fb49b65425f6fb2b3ce26f34d8b3bffea90e83df60192.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/5af585878106176ed03fb49b65425f6fb2b3ce26f34d8b3bffea90e83df60192.jpg)

![5b4bad116daf5ec11ed9063ac861ea22c3e787493620dd68563d50e239989781.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/5b4bad116daf5ec11ed9063ac861ea22c3e787493620dd68563d50e239989781.jpg)

![862ad14a856260243811c44ba7d721807c35657d6e240e2b17a143ab8c04eea1.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/862ad14a856260243811c44ba7d721807c35657d6e240e2b17a143ab8c04eea1.jpg)

![9316bc3117e5e8f7b0ad22804b58af1b4942cf3bb370dbda32b9ac74986a5372.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/9316bc3117e5e8f7b0ad22804b58af1b4942cf3bb370dbda32b9ac74986a5372.jpg)

![97af9bf4f21b08ac3caad8cafdd7826054a285366a3c1f7592a6c3ccc1e23882.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/97af9bf4f21b08ac3caad8cafdd7826054a285366a3c1f7592a6c3ccc1e23882.jpg)

![9f91c393c7935e88aaa9125c42c394861686197ea74729f8bbd27c3140266cb0.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/9f91c393c7935e88aaa9125c42c394861686197ea74729f8bbd27c3140266cb0.jpg)

![ad192358227303ef1d6997a24bc674498474bc7c6fa54683573488e214b0e2c0.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/ad192358227303ef1d6997a24bc674498474bc7c6fa54683573488e214b0e2c0.jpg)

![ae4cafacb713e13578ac1b6aae9c4827f210341891423fa902d42c108eb46ecf.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/ae4cafacb713e13578ac1b6aae9c4827f210341891423fa902d42c108eb46ecf.jpg)

![f9e6eef75ca8c7855a7c0186798128deea8fdd6aeb502247b4a5cf96df4ae163.jpg](../icml_results/4054_Attacks%20on%20Machine-Text%20Detectors%20Retain%20Stylistic%20Fingerprints/tables/f9e6eef75ca8c7855a7c0186798128deea8fdd6aeb502247b4a5cf96df4ae163.jpg)

<a id="paper-4014"></a>

## ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models


### Images

![07f57f99c73d2b1eaff72afed52ca693d58047d822d08b5c0823d2ed7521408b.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/07f57f99c73d2b1eaff72afed52ca693d58047d822d08b5c0823d2ed7521408b.jpg)

![18d11337d433b9b59e820087402e6dc58bf98ca244896012a1fa608cf5ec4fa5.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/18d11337d433b9b59e820087402e6dc58bf98ca244896012a1fa608cf5ec4fa5.jpg)

![18d3a047cd39adbad6a7ccacd4df4fe70174bc21023058b16ecb22239a28d4b5.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/18d3a047cd39adbad6a7ccacd4df4fe70174bc21023058b16ecb22239a28d4b5.jpg)

![44b71882bb966618211d5eead78921d7a294a527f022ee63cf39c514e9ffa965.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/44b71882bb966618211d5eead78921d7a294a527f022ee63cf39c514e9ffa965.jpg)

![55dde3a719e833851aa699c65a97a2daa50dd48678ffbad2eff0f7ac76c57298.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/55dde3a719e833851aa699c65a97a2daa50dd48678ffbad2eff0f7ac76c57298.jpg)

![5820a9577ebb43e8a6cfe96beb041541ad83ddb1a098f419720507a2b4af9c1b.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/5820a9577ebb43e8a6cfe96beb041541ad83ddb1a098f419720507a2b4af9c1b.jpg)

![622885ffb4755b2d0c3c9b5984e648bf950a7f53dcd938e54a36dc4ff9736729.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/622885ffb4755b2d0c3c9b5984e648bf950a7f53dcd938e54a36dc4ff9736729.jpg)

![729412c85bfd51cfccc1832d312fb0f58dd0e0a0735d4f211a76f9b1b38b7616.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/729412c85bfd51cfccc1832d312fb0f58dd0e0a0735d4f211a76f9b1b38b7616.jpg)

![7ae3f25879b79b3488d50eb615445bd0d001c66e742044b761e6c4f4f177870c.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/7ae3f25879b79b3488d50eb615445bd0d001c66e742044b761e6c4f4f177870c.jpg)

![837369d1a9f1241241023dbe414d712a1549df1613bca58991033117601ed253.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/837369d1a9f1241241023dbe414d712a1549df1613bca58991033117601ed253.jpg)

![856a04d888cdcf152be839b0d541860ef7e714c1ebfa2095e80d1b10f2038375.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/856a04d888cdcf152be839b0d541860ef7e714c1ebfa2095e80d1b10f2038375.jpg)

![89f997fe6e27dc116d774325b7c229c338e3128817ab395e6bc277807127c8c7.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/89f997fe6e27dc116d774325b7c229c338e3128817ab395e6bc277807127c8c7.jpg)

![8ccefd455c1a2568aa010b4cd557796f42fd977c5d4dc8f8315a7e71f20300e0.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/8ccefd455c1a2568aa010b4cd557796f42fd977c5d4dc8f8315a7e71f20300e0.jpg)

![8f64098628333f20ed75e5942e609134f7551c125e0bca0b3c79ccf61b0ed83c.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/8f64098628333f20ed75e5942e609134f7551c125e0bca0b3c79ccf61b0ed83c.jpg)

![93b3e8905c3a51c48ed6efeaa32aa0b5f79f721e062feb0c02ee909a5ff6f8d2.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/93b3e8905c3a51c48ed6efeaa32aa0b5f79f721e062feb0c02ee909a5ff6f8d2.jpg)

![946f049b5c82a70b11d17cec2fd64c416d4c2025d57ace4d4f235b4bcf405d5d.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/946f049b5c82a70b11d17cec2fd64c416d4c2025d57ace4d4f235b4bcf405d5d.jpg)

![948200bff0216fe971e2e9d9131b7593396b9548ab2722ae531e6feab2311451.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/948200bff0216fe971e2e9d9131b7593396b9548ab2722ae531e6feab2311451.jpg)

![980ae000745c17ea348ccf67a31a2f6e8c321761246c4da7eaade641261cc493.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/980ae000745c17ea348ccf67a31a2f6e8c321761246c4da7eaade641261cc493.jpg)

![9d621682f3ebd7020fdd28535e9fce3026ac58922c131356566516e46e387f3d.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/9d621682f3ebd7020fdd28535e9fce3026ac58922c131356566516e46e387f3d.jpg)

![ab8da60f967eb61ecb20e893fdc89cb8877211cb787fc1311c3c22736b709dcb.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/ab8da60f967eb61ecb20e893fdc89cb8877211cb787fc1311c3c22736b709dcb.jpg)

![af1d20a39135388124693638c526b8573f94a3aebac8a32da2296949c8155c70.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/af1d20a39135388124693638c526b8573f94a3aebac8a32da2296949c8155c70.jpg)

![bc19eb11794dc2dc7c854155a55f0d396f2b944eca1531ac92ca2d35da073963.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/bc19eb11794dc2dc7c854155a55f0d396f2b944eca1531ac92ca2d35da073963.jpg)

![ce848d9fd14eeae2490bae200675ac4fba3315a1b32944f0fad0d458f879ac51.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/ce848d9fd14eeae2490bae200675ac4fba3315a1b32944f0fad0d458f879ac51.jpg)

![db9710570f794d43cc01ee12d2ba7e24455b6cb46c342e5cc6ddb9755253bb3b.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/db9710570f794d43cc01ee12d2ba7e24455b6cb46c342e5cc6ddb9755253bb3b.jpg)

![f3e06d8d5c273098f5c6cbf3ae2eb8ac535a1b79c6bcd096c837e856ccf241ec.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/images/f3e06d8d5c273098f5c6cbf3ae2eb8ac535a1b79c6bcd096c837e856ccf241ec.jpg)

### Tables

![006d89bdca3f7a28038af26b67453d5c1c0caa3cddee6bc48703d3fd5dcc7c05.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/006d89bdca3f7a28038af26b67453d5c1c0caa3cddee6bc48703d3fd5dcc7c05.jpg)

![4514074a52e6f7e0eba5ec1b443fdfdf5faa6708270446589e735296c9aff598.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/4514074a52e6f7e0eba5ec1b443fdfdf5faa6708270446589e735296c9aff598.jpg)

![55ab461f3ead91652497a7ade7066c8ef5341d834728ad287de6a1447b3c50c0.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/55ab461f3ead91652497a7ade7066c8ef5341d834728ad287de6a1447b3c50c0.jpg)

![7e40b82cd50f51c1e9a365f86c30c572531814cbd050f04298a9a1d2610504d1.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/7e40b82cd50f51c1e9a365f86c30c572531814cbd050f04298a9a1d2610504d1.jpg)

![85d70f6a800ac2d06cf2320752461126215c8798cb8925c7b394c0adaaedba65.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/85d70f6a800ac2d06cf2320752461126215c8798cb8925c7b394c0adaaedba65.jpg)

![9c10bb600d68d5bb0c988a8f063389d267a44eb40928d35bae249bb8d22478bd.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/9c10bb600d68d5bb0c988a8f063389d267a44eb40928d35bae249bb8d22478bd.jpg)

![bf701d450c910f2474462cceedf816436f66fec2cce354322e67dfac163207f3.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/bf701d450c910f2474462cceedf816436f66fec2cce354322e67dfac163207f3.jpg)

![e04cea79abc18e03303f8ee6fdb8b9c0fb549027d48000a5087aa12bd28cc810.jpg](../icml_results/4055_ZeroBench_%20An%20Impossible%20Visual%20Benchmark%20for%20Contemporary%20Large%20Multimodal%20Models/tables/e04cea79abc18e03303f8ee6fdb8b9c0fb549027d48000a5087aa12bd28cc810.jpg)

<a id="paper-4015"></a>

## Practical and Scalable Hamiltonian Monte Carlo Without the Metropolis Test


### Images

![08b7439a88bf68ae36f2da31e58f15d3911e465368a6e7f97e37397d52b4dc65.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/08b7439a88bf68ae36f2da31e58f15d3911e465368a6e7f97e37397d52b4dc65.jpg)

![2143e9da83e25faed9b06847b43c9332b43a674122dad78dd9745096273ef571.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/2143e9da83e25faed9b06847b43c9332b43a674122dad78dd9745096273ef571.jpg)

![6dc2762fc0858b2fd02a5bdcadbb220a896473eb1499d2a3c75742bf9298af37.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/6dc2762fc0858b2fd02a5bdcadbb220a896473eb1499d2a3c75742bf9298af37.jpg)

![704528b8d2497a7713471243f23d080bc0339d130143d010545d1a132a1d01dc.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/704528b8d2497a7713471243f23d080bc0339d130143d010545d1a132a1d01dc.jpg)

![a161419be1644dd1c52bb34e6fc3e3b3dee9696e5edbab3e316532c262366fa9.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/a161419be1644dd1c52bb34e6fc3e3b3dee9696e5edbab3e316532c262366fa9.jpg)

![a4f8342d3219cf9c48428ab596f2e0264a57d0a85db441100d627720d0ae8e73.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/a4f8342d3219cf9c48428ab596f2e0264a57d0a85db441100d627720d0ae8e73.jpg)

![c55097108dc3cb4cf484a71e5b94d08dba51e2242e5df652aa6233fa56da4c2c.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/c55097108dc3cb4cf484a71e5b94d08dba51e2242e5df652aa6233fa56da4c2c.jpg)

![e36216e9d23245926bfa5afc0d08149b7eed7419ff940608f0c42468714f5a8d.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/e36216e9d23245926bfa5afc0d08149b7eed7419ff940608f0c42468714f5a8d.jpg)

![f0bf261ba922ed199b41dccfb937773b6644208b7e3a626ee4c2e538e73c763f.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/images/f0bf261ba922ed199b41dccfb937773b6644208b7e3a626ee4c2e538e73c763f.jpg)

### Tables

![31c7ca0680be898f7471047ec0e67057bed1130559a842cf3f46c9179a4feb02.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/tables/31c7ca0680be898f7471047ec0e67057bed1130559a842cf3f46c9179a4feb02.jpg)

![96eea088eb9888e0916002e3828739c56b9f57a243dd4e09d3f0b6463315ccd8.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/tables/96eea088eb9888e0916002e3828739c56b9f57a243dd4e09d3f0b6463315ccd8.jpg)

![a3e4bbdd7dc582988a48f79dc6ad8227a84e1ea03ac34e20e66c0d0cd3153a5b.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/tables/a3e4bbdd7dc582988a48f79dc6ad8227a84e1ea03ac34e20e66c0d0cd3153a5b.jpg)

![e1e58a518316096f4ae3a81012ed0979b8406541dc4097efab6deac60fcd800c.jpg](../icml_results/4056_Practical%20and%20Scalable%20Hamiltonian%20Monte%20Carlo%20Without%20the%20Metropolis%20Test/tables/e1e58a518316096f4ae3a81012ed0979b8406541dc4097efab6deac60fcd800c.jpg)

<a id="paper-4016"></a>

## Model-Free Robust Average-Reward Reinforcement Learning with Sample Complexity Analysis


### Images

![0be72692ac4bf08c0d098f12ee76f315bc666d412c0f18cd26472d2cf8ed795a.jpg](../icml_results/4057_Model-Free%20Robust%20Average-Reward%20Reinforcement%20Learning%20with%20Sample%20Complexity%20Analysis/images/0be72692ac4bf08c0d098f12ee76f315bc666d412c0f18cd26472d2cf8ed795a.jpg)

![2df43d8009e6002deda57bedb4d812aa719a95bf4db6a8536ccf38f1ccae7b55.jpg](../icml_results/4057_Model-Free%20Robust%20Average-Reward%20Reinforcement%20Learning%20with%20Sample%20Complexity%20Analysis/images/2df43d8009e6002deda57bedb4d812aa719a95bf4db6a8536ccf38f1ccae7b55.jpg)

### Tables

![bba0201c1792a70a6c4497735c15a9e0fe3a5754526edcd49b7ac47405c1432b.jpg](../icml_results/4057_Model-Free%20Robust%20Average-Reward%20Reinforcement%20Learning%20with%20Sample%20Complexity%20Analysis/tables/bba0201c1792a70a6c4497735c15a9e0fe3a5754526edcd49b7ac47405c1432b.jpg)

<a id="paper-4017"></a>

## Distributionally Robust Markov Games with Average Reward


### Images

![0e726a629044c94dddbfe4c8ceccba26db27d9fadd73cd4bf2d64e1bcb593f04.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/images/0e726a629044c94dddbfe4c8ceccba26db27d9fadd73cd4bf2d64e1bcb593f04.jpg)

![136d248541e70c8c471017e7b6c676ef46f7a6b8bcf713977453ca05993397b1.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/images/136d248541e70c8c471017e7b6c676ef46f7a6b8bcf713977453ca05993397b1.jpg)

![ff43ee9e2b50375f1a59b9868ebb471217741a3c7b3ae06b0adf4a4fcb9b77c3.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/images/ff43ee9e2b50375f1a59b9868ebb471217741a3c7b3ae06b0adf4a4fcb9b77c3.jpg)

### Tables

![1209e1419a29987fdf7efa6483e5510bba78eda0fcc4b04569fbe8314d7f4799.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/tables/1209e1419a29987fdf7efa6483e5510bba78eda0fcc4b04569fbe8314d7f4799.jpg)

![5722978d6ac58bb4fdb324a8493df0834ca2d54f0759b3a4918b1d5604bfe4d0.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/tables/5722978d6ac58bb4fdb324a8493df0834ca2d54f0759b3a4918b1d5604bfe4d0.jpg)

![cc8716e9594c4344eca910c9fa046ab4c41f6cb2e09b8d34b9a38df2cf959b94.jpg](../icml_results/4058_Distributionally%20Robust%20Markov%20Games%20with%20Average%20Reward/tables/cc8716e9594c4344eca910c9fa046ab4c41f6cb2e09b8d34b9a38df2cf959b94.jpg)

<a id="paper-4018"></a>

## Performative Learning Theory


### Images

![89e574351265fe504c09e69fa5d7505856408ad5eba88a44d43d1703e98c8fbb.jpg](../icml_results/4059_Performative%20Learning%20Theory/images/89e574351265fe504c09e69fa5d7505856408ad5eba88a44d43d1703e98c8fbb.jpg)

![9dc7be81e0a80c75161728eef771848e239f51490a71763d27c1d0ec42f86264.jpg](../icml_results/4059_Performative%20Learning%20Theory/images/9dc7be81e0a80c75161728eef771848e239f51490a71763d27c1d0ec42f86264.jpg)

### Tables

![cf23c90a4a7631865b7d2362eca708d0ad7100f0802b8e4246d852ef7b1f7a13.jpg](../icml_results/4059_Performative%20Learning%20Theory/tables/cf23c90a4a7631865b7d2362eca708d0ad7100f0802b8e4246d852ef7b1f7a13.jpg)

<a id="paper-4019"></a>

## FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations


### Images

![09b5d2677bef76f962770eb4bd29f37d93ccb46ecab7a867ca41cd91742e6c29.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/09b5d2677bef76f962770eb4bd29f37d93ccb46ecab7a867ca41cd91742e6c29.jpg)

![0f62e2c6a2f9fa55924ed8f3dab406bc5c317cb76135e7973e7dc0d1607e21ed.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/0f62e2c6a2f9fa55924ed8f3dab406bc5c317cb76135e7973e7dc0d1607e21ed.jpg)

![39a8161dd09ff9f52c18d1e0ff455f610854f863c033858f84e6363413abf35d.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/39a8161dd09ff9f52c18d1e0ff455f610854f863c033858f84e6363413abf35d.jpg)

![459ea78fb42e1287693d1c7e17b7e1cbd96a88d1369bf966fa7a055dd8007e5b.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/459ea78fb42e1287693d1c7e17b7e1cbd96a88d1369bf966fa7a055dd8007e5b.jpg)

![4863a2072fcd050117e2223630855a6640acc374cc6aa0188d0cb1b997a44257.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/4863a2072fcd050117e2223630855a6640acc374cc6aa0188d0cb1b997a44257.jpg)

![526a0a732c588fa9cb92349bb6389e8c930c0a6829b6a2dbca064eba33fe569d.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/526a0a732c588fa9cb92349bb6389e8c930c0a6829b6a2dbca064eba33fe569d.jpg)

![5953872d9d10ed31c1c6c3e8c087971ac461416e5cb315e3a829ecb240ee334d.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/5953872d9d10ed31c1c6c3e8c087971ac461416e5cb315e3a829ecb240ee334d.jpg)

![5c17f55c739f1426ab2da2d547e82e098aa5716bbb7f95abcffeb1e239482f66.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/5c17f55c739f1426ab2da2d547e82e098aa5716bbb7f95abcffeb1e239482f66.jpg)

![6a6d25f53ddcc73fa3b40d320f23b53f1c0473c5f29e293df0c84f3d67b8c9ec.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/6a6d25f53ddcc73fa3b40d320f23b53f1c0473c5f29e293df0c84f3d67b8c9ec.jpg)

![7a0ab7daaa8a709b53d2cbd72cb52f9b1d83e4e9f7ca83aba409041f674a2246.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/7a0ab7daaa8a709b53d2cbd72cb52f9b1d83e4e9f7ca83aba409041f674a2246.jpg)

![80eeccf0987de581cc23d1370f8538059a45b632032d69f821e69f88ddb7f816.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/80eeccf0987de581cc23d1370f8538059a45b632032d69f821e69f88ddb7f816.jpg)

![943fc44cd91e32066d42faafb4b3df57c0bff8ae739cde33f8248522554689c6.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/943fc44cd91e32066d42faafb4b3df57c0bff8ae739cde33f8248522554689c6.jpg)

![971d349024c502dcfb6146a04570de9dab49c4b61efa5228ce98ba57f2fd86b4.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/971d349024c502dcfb6146a04570de9dab49c4b61efa5228ce98ba57f2fd86b4.jpg)

![b2eb02e87987729aa42cf2673cc1c18193f96f379a0965d948483c28582b004c.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/b2eb02e87987729aa42cf2673cc1c18193f96f379a0965d948483c28582b004c.jpg)

![b7c33dedb27f44581a7da11a17b181a430098c23d380a22b7599bf917933fbbb.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/b7c33dedb27f44581a7da11a17b181a430098c23d380a22b7599bf917933fbbb.jpg)

![bb28dc07370cf1f184b08ef35dabacad2ae6b5db0dfae8f28ca1b45d0ee47452.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/bb28dc07370cf1f184b08ef35dabacad2ae6b5db0dfae8f28ca1b45d0ee47452.jpg)

![d0e6b765c70816dab687a4e638c3e2fbc321c993c6115c25bf564b06b0693784.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/d0e6b765c70816dab687a4e638c3e2fbc321c993c6115c25bf564b06b0693784.jpg)

![e783b8e424596a96c6b5dd0d401dde14e8a4d9ea5d330e3d067d2efd35e4ae95.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/e783b8e424596a96c6b5dd0d401dde14e8a4d9ea5d330e3d067d2efd35e4ae95.jpg)

![e83b1610c98f6e97179c7761fa3bce0e8a8bfe89e1fcfdd1d61b5a68f224e578.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/e83b1610c98f6e97179c7761fa3bce0e8a8bfe89e1fcfdd1d61b5a68f224e578.jpg)

![f553d5a60b5d42fd742d027de963af7dde03cc25065c7dc0a05b61b82b0f9798.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/f553d5a60b5d42fd742d027de963af7dde03cc25065c7dc0a05b61b82b0f9798.jpg)

![fcdd7fed1fce48645260d575369fc3bd7ddcd65212714a07d7e7fc7ef0f3acfb.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/fcdd7fed1fce48645260d575369fc3bd7ddcd65212714a07d7e7fc7ef0f3acfb.jpg)

![fd4f78fb90bd75b7a60650807577a8b3b7115dce4d2ce0edafa3880a49142484.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/fd4f78fb90bd75b7a60650807577a8b3b7115dce4d2ce0edafa3880a49142484.jpg)

![ff76f4c3b7f0a25db3d7525f9cb32a4db956418e538d7614a99479dc674cf235.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/images/ff76f4c3b7f0a25db3d7525f9cb32a4db956418e538d7614a99479dc674cf235.jpg)

### Tables

![1034ab3b8d2ffb15f24f2f608733730624eea5df093ea089b383c499d5511adc.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/1034ab3b8d2ffb15f24f2f608733730624eea5df093ea089b383c499d5511adc.jpg)

![11b3037cff29839fae3b1eaa735b3542e55aaead5978132e44ae9f814b1c4ba3.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/11b3037cff29839fae3b1eaa735b3542e55aaead5978132e44ae9f814b1c4ba3.jpg)

![21f762cf08ae4a3e8dd6d584011ff6c6abf753e8b65d3d9a9d0eaf5f95b209f6.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/21f762cf08ae4a3e8dd6d584011ff6c6abf753e8b65d3d9a9d0eaf5f95b209f6.jpg)

![2a73e1f463cd71db7e1e89d69f71c74e98ca12769cdf521aa1b3e3504176a580.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/2a73e1f463cd71db7e1e89d69f71c74e98ca12769cdf521aa1b3e3504176a580.jpg)

![460c66d690a7d204016f283cc9fdd3718b2781c6531aad15e1c95a065dc9dda4.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/460c66d690a7d204016f283cc9fdd3718b2781c6531aad15e1c95a065dc9dda4.jpg)

![51835350c570678d011ee3998df69b5521acd68aa571ee9217af6dceb0fb830a.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/51835350c570678d011ee3998df69b5521acd68aa571ee9217af6dceb0fb830a.jpg)

![5f88a8a008f075fbf43a9afa6f5a34ebf9969dbc5240c20dfc05ddfe77664989.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/5f88a8a008f075fbf43a9afa6f5a34ebf9969dbc5240c20dfc05ddfe77664989.jpg)

![764d8dc6233d3bc2386527abc80430ba8c8d80e574a94d5686522af1881aadab.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/764d8dc6233d3bc2386527abc80430ba8c8d80e574a94d5686522af1881aadab.jpg)

![7a11a78605115282f43b47e6b4a5d05266a6fb670b41e2a10d0af17e75d8375f.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/7a11a78605115282f43b47e6b4a5d05266a6fb670b41e2a10d0af17e75d8375f.jpg)

![7ab07fc967c6bfb5838db46685571a019a0cef0ce03bfae03614fae487ba2871.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/7ab07fc967c6bfb5838db46685571a019a0cef0ce03bfae03614fae487ba2871.jpg)

![84e641d4da24693a7b40710c9366a94e11a1c19e863765617c9bc95feb060136.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/84e641d4da24693a7b40710c9366a94e11a1c19e863765617c9bc95feb060136.jpg)

![8b1ad0624c6d14072e1e31ebbdee434129136cdfbf459c5648fddf9833e3ec03.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/8b1ad0624c6d14072e1e31ebbdee434129136cdfbf459c5648fddf9833e3ec03.jpg)

![9fca08d37873054a5799b220c539445d103daafdd11c24b2196d57cba63d7703.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/9fca08d37873054a5799b220c539445d103daafdd11c24b2196d57cba63d7703.jpg)

![a3beefd5bb7291b90bb8ae3154a5d3ba8ab076b763833c57ce2c85f35a625326.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/a3beefd5bb7291b90bb8ae3154a5d3ba8ab076b763833c57ce2c85f35a625326.jpg)

![b9fc81e2f1b7b0f8410ca71aaa3b8625119df9263da936f125f60e9609817da3.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/b9fc81e2f1b7b0f8410ca71aaa3b8625119df9263da936f125f60e9609817da3.jpg)

![c17e2ac5179835a16c585970e6562c874503dd7c8fc9dd47970c897819ec4a95.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/c17e2ac5179835a16c585970e6562c874503dd7c8fc9dd47970c897819ec4a95.jpg)

![eba9009c2a3999b808a7257a6f012a54a01c4a3f0ddd8c88c6d7ef351323a95c.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/eba9009c2a3999b808a7257a6f012a54a01c4a3f0ddd8c88c6d7ef351323a95c.jpg)

![f0383306e3e838c534470150d3e644a4c86de0d82faf0b09bf34855e17f8d934.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/f0383306e3e838c534470150d3e644a4c86de0d82faf0b09bf34855e17f8d934.jpg)

![f8c0997b412b9fc2850bfed5e9ec8076894235baf0df95470a492af9899c7d84.jpg](../icml_results/4060_FloorplanQA_%20A%20Benchmark%20for%20Spatial%20Reasoning%20in%20LLMs%20using%20Structured%20Representations/tables/f8c0997b412b9fc2850bfed5e9ec8076894235baf0df95470a492af9899c7d84.jpg)

<a id="paper-4020"></a>

## Diffusion Models Preferentially Memorize Prototypical Examples or: Why Does My Diffusion Model Love Slop?


### Images

![013a0020c317fa3c1ffb6364ba49be6767c7fcdc2dda7067ffe0659b6e5d56c2.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/013a0020c317fa3c1ffb6364ba49be6767c7fcdc2dda7067ffe0659b6e5d56c2.jpg)

![04c0bc12c9c3860d169e0abec9cadbf4f2497bd2991978dd532b417f55059466.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/04c0bc12c9c3860d169e0abec9cadbf4f2497bd2991978dd532b417f55059466.jpg)

![0c0ad7beeea5333e387716f0da1901692d337c9bf96e38185ada4dab9d6b0e2c.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/0c0ad7beeea5333e387716f0da1901692d337c9bf96e38185ada4dab9d6b0e2c.jpg)

![1e39199b529a64ac00e139e92b0b6ea54f9405accb6170d3c545e9c9361242aa.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/1e39199b529a64ac00e139e92b0b6ea54f9405accb6170d3c545e9c9361242aa.jpg)

![20f906a0e01d25bfc2bd42f808c38b9b3bbde9034b182f92a0636c847f8a31ea.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/20f906a0e01d25bfc2bd42f808c38b9b3bbde9034b182f92a0636c847f8a31ea.jpg)

![314b742fec38dee3fbb209d40a8d44fa4dd5f377f4ea044cc0a5da3651ff58ab.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/314b742fec38dee3fbb209d40a8d44fa4dd5f377f4ea044cc0a5da3651ff58ab.jpg)

![32b309d54e3770586800cea267e485bcf01bc618ea65e2dd4111cd756c5e34b6.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/32b309d54e3770586800cea267e485bcf01bc618ea65e2dd4111cd756c5e34b6.jpg)

![32b7599d941f44437aa9c5311a1932e146000f96633583df79806e6de825b93a.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/32b7599d941f44437aa9c5311a1932e146000f96633583df79806e6de825b93a.jpg)

![4b8718b57e33897a0c9dcee9b25b946b9f303d224a88e30059687f708ac8fe36.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/4b8718b57e33897a0c9dcee9b25b946b9f303d224a88e30059687f708ac8fe36.jpg)

![61f59bec8aedde9e841eabdd50c629d8649012846ba7740f5096a72d57a9c792.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/61f59bec8aedde9e841eabdd50c629d8649012846ba7740f5096a72d57a9c792.jpg)

![6245d0d4f1397e9babd9ac2f57dcce675a47c838d498be951b8bc5807802f94f.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/6245d0d4f1397e9babd9ac2f57dcce675a47c838d498be951b8bc5807802f94f.jpg)

![6294b60037e612056254aae5404ef71fb30a4c61971fe37409cb6b68aefbf894.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/6294b60037e612056254aae5404ef71fb30a4c61971fe37409cb6b68aefbf894.jpg)

![660b2fe74d8bf8f9128f51778bff85b2901cba26d1d45f36368b0499a002d4a7.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/660b2fe74d8bf8f9128f51778bff85b2901cba26d1d45f36368b0499a002d4a7.jpg)

![6bfa3a3645ade4a2ff91d5053c8d0b9b6455fbc62f80ba1f95d5024f9e2337d2.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/6bfa3a3645ade4a2ff91d5053c8d0b9b6455fbc62f80ba1f95d5024f9e2337d2.jpg)

![74fe5a0467576abeee71784b4b4bcaa382493aba96c8dfd4143ac36de581e05e.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/74fe5a0467576abeee71784b4b4bcaa382493aba96c8dfd4143ac36de581e05e.jpg)

![78651543cf0818ded90b5191f36684a325177532a57240b9dbdbdfc4cc106c41.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/78651543cf0818ded90b5191f36684a325177532a57240b9dbdbdfc4cc106c41.jpg)

![7a91e15ec9168d2741b791eec6fbdfac45b3c51270aeac72e6a5ff31d862e4cd.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/7a91e15ec9168d2741b791eec6fbdfac45b3c51270aeac72e6a5ff31d862e4cd.jpg)

![967823cd15454046f0a73803b9429440357035677b0ba6dffc4c48b037db204a.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/967823cd15454046f0a73803b9429440357035677b0ba6dffc4c48b037db204a.jpg)

![c2d5e96cf90d6e582a1670302b9fcc01008ea943b3bbc4740c227f577c31e80a.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/c2d5e96cf90d6e582a1670302b9fcc01008ea943b3bbc4740c227f577c31e80a.jpg)

![c42d760c8c2700851792424026186a3efe9871986a752bc05d0f2b686c1ff7f3.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/c42d760c8c2700851792424026186a3efe9871986a752bc05d0f2b686c1ff7f3.jpg)

![c7839ac988f10c80527bdcf29ef2caeb4f816a0b426933de3c82a28835d616a7.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/c7839ac988f10c80527bdcf29ef2caeb4f816a0b426933de3c82a28835d616a7.jpg)

![d177ed0be86c88d146df3806f4d0218544eff83e4337e2536e81cca1a348bb18.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/d177ed0be86c88d146df3806f4d0218544eff83e4337e2536e81cca1a348bb18.jpg)

![d21fe2cd883c15c8bf143f6515a8967c0dddd5a73b7a9ce3e115700cb1218a6b.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/d21fe2cd883c15c8bf143f6515a8967c0dddd5a73b7a9ce3e115700cb1218a6b.jpg)

![d8de8cc9b0e64eb4f620529e95985dacd240cbb05d40ca53bdbdb0bac06bc00b.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/d8de8cc9b0e64eb4f620529e95985dacd240cbb05d40ca53bdbdb0bac06bc00b.jpg)

![e939872469c307d7cf08c727063517ebbfe20c4ea0804a0c63909dbcc0420b6f.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/e939872469c307d7cf08c727063517ebbfe20c4ea0804a0c63909dbcc0420b6f.jpg)

![f08f2642f49ea26ee75804356776b90e7152c154a2f794559fc00ddf9f48fdea.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/f08f2642f49ea26ee75804356776b90e7152c154a2f794559fc00ddf9f48fdea.jpg)

![f38fb8ac0bbb5228fc5a0d28e2e4ca17b3263bb75cbce20f3e1e072d04692a6c.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/f38fb8ac0bbb5228fc5a0d28e2e4ca17b3263bb75cbce20f3e1e072d04692a6c.jpg)

![fa975a168eef26247ca6e969be128cee88b0d69f084ee5bc745c2fc04e8a1842.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/images/fa975a168eef26247ca6e969be128cee88b0d69f084ee5bc745c2fc04e8a1842.jpg)

### Tables

![af235bb3ac52785d2288f328c22c2d461884a5dc2974ff8a458c5dd8dd9d04bf.jpg](../icml_results/4061_Diffusion%20Models%20Preferentially%20Memorize%20Prototypical%20Examples%20or_%20Why%20Does%20My%20Diffusion%20Model%20Love%20/tables/af235bb3ac52785d2288f328c22c2d461884a5dc2974ff8a458c5dd8dd9d04bf.jpg)

<a id="paper-4021"></a>

## Ambient Dataloops: Generative Models for Dataset Refinement


### Images

![1bbd377b9bdeda07682b7836427b71ec03430ead5038e5f1ebf82adbef48c9d4.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/1bbd377b9bdeda07682b7836427b71ec03430ead5038e5f1ebf82adbef48c9d4.jpg)

![2c794941df3b3c3488f654c4179f0b33c3853638a03a5bb19dc542e2480a1e49.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/2c794941df3b3c3488f654c4179f0b33c3853638a03a5bb19dc542e2480a1e49.jpg)

![2d0562e0f7702bf5c2b9023432bcf9fa65dc6302c0be131816347f7f9b13b40f.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/2d0562e0f7702bf5c2b9023432bcf9fa65dc6302c0be131816347f7f9b13b40f.jpg)

![5fc3bfc06a3acd5a903327722af890789038020017e2998456765cd4508c4916.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/5fc3bfc06a3acd5a903327722af890789038020017e2998456765cd4508c4916.jpg)

![8c6ccd7c9cab1fbee020962f9a7b77ef4618e43a8a0a06da347d674c35d76093.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/8c6ccd7c9cab1fbee020962f9a7b77ef4618e43a8a0a06da347d674c35d76093.jpg)

![a3d85fc0e3112981e76d411cdb419f1ceee676f894d32ed3ab4ed854ea932091.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/a3d85fc0e3112981e76d411cdb419f1ceee676f894d32ed3ab4ed854ea932091.jpg)

![c27fa5f44dc8536f256577bdc0a8eedac4c0c8be8dc0aaa06542c8bd49f01e4e.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/c27fa5f44dc8536f256577bdc0a8eedac4c0c8be8dc0aaa06542c8bd49f01e4e.jpg)

![d92ed78a44f94e744e0aa2d0174b6ab284a98c8201f00c70dac99384f75d5d89.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/d92ed78a44f94e744e0aa2d0174b6ab284a98c8201f00c70dac99384f75d5d89.jpg)

![e666c2f546768dfb1c4f981aafa3428cbb04ea72f56b3b8823db39ebcc93ad53.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/e666c2f546768dfb1c4f981aafa3428cbb04ea72f56b3b8823db39ebcc93ad53.jpg)

![f8eaf99c02877c5a2b8e9e0a1018751a88b6a4b850f3fd6062f39515ae6101cd.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/images/f8eaf99c02877c5a2b8e9e0a1018751a88b6a4b850f3fd6062f39515ae6101cd.jpg)

### Tables

![03757c8e843607efa1b2d043bd687d9c8fd5ea298f40d99d9384dfa6283ce7ab.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/03757c8e843607efa1b2d043bd687d9c8fd5ea298f40d99d9384dfa6283ce7ab.jpg)

![0cc8810a905af01190a5071f0542509ccfe604e328fc4f86d973f4c909517af9.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/0cc8810a905af01190a5071f0542509ccfe604e328fc4f86d973f4c909517af9.jpg)

![28895482b2d869d201a1cd088f7980419b6d13084d5b846902d655ac558a9569.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/28895482b2d869d201a1cd088f7980419b6d13084d5b846902d655ac558a9569.jpg)

![5a9e56e276f1af44379522f59ca4e069f8cbd42ebe99312d60a84ece754ef8de.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/5a9e56e276f1af44379522f59ca4e069f8cbd42ebe99312d60a84ece754ef8de.jpg)

![5df757e93bfc30dad0c52816a4cf761c17f08eab55ba407cb0b9dc126858b83a.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/5df757e93bfc30dad0c52816a4cf761c17f08eab55ba407cb0b9dc126858b83a.jpg)

![5eb5a490cd934856b8903e8cbf61cbed94421c6a0c8bbc68e77e5f486c699193.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/5eb5a490cd934856b8903e8cbf61cbed94421c6a0c8bbc68e77e5f486c699193.jpg)

![60ecf1a3d9085bed729979026b8360caf3e606ce2448677d2a4d3600e175d816.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/60ecf1a3d9085bed729979026b8360caf3e606ce2448677d2a4d3600e175d816.jpg)

![6ceb94efc00f97295157af04f2253d51dffad6abd713ea15e3184096a61841e5.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/6ceb94efc00f97295157af04f2253d51dffad6abd713ea15e3184096a61841e5.jpg)

![726a088a444f34f14773b235390215a4fe8c6608fb0d7bca6110e56dc597a73f.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/726a088a444f34f14773b235390215a4fe8c6608fb0d7bca6110e56dc597a73f.jpg)

![77134c0455d0c391e3450c58636c49720dcab326eb44e7cb0f2fea37fed60ece.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/77134c0455d0c391e3450c58636c49720dcab326eb44e7cb0f2fea37fed60ece.jpg)

![9214d366a1e0a5e3e9f534c7a1255957a5040ffea2f23c46d405ec9cf4747417.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/9214d366a1e0a5e3e9f534c7a1255957a5040ffea2f23c46d405ec9cf4747417.jpg)

![958595695f19c71b24245f93369cf875f8cdb71f092005570cd3f895b80f41da.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/958595695f19c71b24245f93369cf875f8cdb71f092005570cd3f895b80f41da.jpg)

![9ab3aa3046a050883e9e9539295a81277c6d921d439db5b68c570ea4f65cecb9.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/9ab3aa3046a050883e9e9539295a81277c6d921d439db5b68c570ea4f65cecb9.jpg)

![9cd297103575a51e051fd62d0830b66fae6ea5d00c584c8509c5daf78623c963.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/9cd297103575a51e051fd62d0830b66fae6ea5d00c584c8509c5daf78623c963.jpg)

![d2540919b8e3a8fd6e45613acf12a3173c0727ec667cf66c42be394ddd91e07d.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/d2540919b8e3a8fd6e45613acf12a3173c0727ec667cf66c42be394ddd91e07d.jpg)

![ec578d51710c54ea0cf0f04b6e3e6ae62fed6cd7e16ceaaf653adf5c2c74d3d6.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/ec578d51710c54ea0cf0f04b6e3e6ae62fed6cd7e16ceaaf653adf5c2c74d3d6.jpg)

![f5fc1ee0e2f42dcbcaa4132460439447c26791c3f3c0dd2fb431b3b3299ff530.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/f5fc1ee0e2f42dcbcaa4132460439447c26791c3f3c0dd2fb431b3b3299ff530.jpg)

![ffecb873bd4d76c05e997439924f999d9ccd7ef706dc65bae7ae404818eb931f.jpg](../icml_results/4062_Ambient%20Dataloops_%20Generative%20Models%20for%20Dataset%20Refinement/tables/ffecb873bd4d76c05e997439924f999d9ccd7ef706dc65bae7ae404818eb931f.jpg)

<a id="paper-4022"></a>

## PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning


### Images

![0ee42f567f6385d714bbd2d9161e55fcdac7c02d7a3bd425109afdafff4cc81d.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/0ee42f567f6385d714bbd2d9161e55fcdac7c02d7a3bd425109afdafff4cc81d.jpg)

![46d215715de295680e6c9d8dc76bc4a7e4b1e6a539d74c7188bc0832d472568c.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/46d215715de295680e6c9d8dc76bc4a7e4b1e6a539d74c7188bc0832d472568c.jpg)

![5ea88a1609036711880a911bd563642bf8c18b7a47f5bbc871a8ab46d5174c1a.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/5ea88a1609036711880a911bd563642bf8c18b7a47f5bbc871a8ab46d5174c1a.jpg)

![68f4b826a2bee7e07644ac3b85f576db08dc00a0eb25ad9c18c4376ed5889246.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/68f4b826a2bee7e07644ac3b85f576db08dc00a0eb25ad9c18c4376ed5889246.jpg)

![a085be1bf5ddbfeba76dcdae5ee179f7eee95471a4931d0b1f91493388453c49.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/a085be1bf5ddbfeba76dcdae5ee179f7eee95471a4931d0b1f91493388453c49.jpg)

![a9d76af19eb91ba64e973078c2127b43af36aaf16030013ef6c73c3d567bba32.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/images/a9d76af19eb91ba64e973078c2127b43af36aaf16030013ef6c73c3d567bba32.jpg)

### Tables

![2049f16c8b78f8ebfa77545680e5d948af8cb908262fe46f5552a99640f9caf8.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/2049f16c8b78f8ebfa77545680e5d948af8cb908262fe46f5552a99640f9caf8.jpg)

![2c4cb4b13b208c406c5021424ce83a00958145089fdb14e97aaef78a5a175ca9.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/2c4cb4b13b208c406c5021424ce83a00958145089fdb14e97aaef78a5a175ca9.jpg)

![38e9918318027377581a07f1d933736a5164c7bf6e92fb9f558afadb6df6c035.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/38e9918318027377581a07f1d933736a5164c7bf6e92fb9f558afadb6df6c035.jpg)

![4c406633a2736aca98f62184e5d866cb9275035a4dd3ea46994b2375547ab332.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/4c406633a2736aca98f62184e5d866cb9275035a4dd3ea46994b2375547ab332.jpg)

![a4f2bee7b579f6850c1ff9d796ef2a40c2c77a994a86f8e546d8b2d83050079e.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/a4f2bee7b579f6850c1ff9d796ef2a40c2c77a994a86f8e546d8b2d83050079e.jpg)

![df488f3c7ae471e12ab8e5ce414a9d9c19e79016eee8026a1b2f18a7982e7c47.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/df488f3c7ae471e12ab8e5ce414a9d9c19e79016eee8026a1b2f18a7982e7c47.jpg)

![dff3ebf6779a50f2495a6db490b58f866cb2e4df4eebbdd7e8e60a29bc2e3798.jpg](../icml_results/4063_PAMD_%20Structured%20Adaptive%20Distances%20for%20Bisimulation%20Representations%20in%20Visual%20Reinforcement%20Learnin/tables/dff3ebf6779a50f2495a6db490b58f866cb2e4df4eebbdd7e8e60a29bc2e3798.jpg)

<a id="paper-4023"></a>

## Categorical Flow Maps


### Images

![16cc232e6ad55b35f2ffa3597ec221d0c13fc0eb30e838ac8feaf548048e900b.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/16cc232e6ad55b35f2ffa3597ec221d0c13fc0eb30e838ac8feaf548048e900b.jpg)

![1a46dc76bbb7826e20a4a60ffa94e86c75c8705d865fe3ede7592397398b921e.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/1a46dc76bbb7826e20a4a60ffa94e86c75c8705d865fe3ede7592397398b921e.jpg)

![234e357d0ab0b4b5ff962bdf114d93c32b79682e5c1f1c5269384556dca631f1.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/234e357d0ab0b4b5ff962bdf114d93c32b79682e5c1f1c5269384556dca631f1.jpg)

![5bdd6a546090b8f933e9b11b6c02fb08fdbbbc64a4c00d2621eaf630965bd6d9.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/5bdd6a546090b8f933e9b11b6c02fb08fdbbbc64a4c00d2621eaf630965bd6d9.jpg)

![6b78180e55cd48b81ce8c3a8aa2924fa7f11c4632029db51abc45c4688dbd1e7.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/6b78180e55cd48b81ce8c3a8aa2924fa7f11c4632029db51abc45c4688dbd1e7.jpg)

![85c25bd350e18319ae8174981583987bd0c6b5a28fb90463fdd5f6fb5f0f92e0.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/85c25bd350e18319ae8174981583987bd0c6b5a28fb90463fdd5f6fb5f0f92e0.jpg)

![930d2ea497cee21896ee0d76daa345c5b108250dad524b5fef37f195c7b84664.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/930d2ea497cee21896ee0d76daa345c5b108250dad524b5fef37f195c7b84664.jpg)

![b9eaef32acbfabfed14876aff08b26e5485484216640ec07557d215010c28bfc.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/b9eaef32acbfabfed14876aff08b26e5485484216640ec07557d215010c28bfc.jpg)

![c484c0c300cd69a0201e5520d66fa6147aa8541950084d3542fb69743e1d5003.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/c484c0c300cd69a0201e5520d66fa6147aa8541950084d3542fb69743e1d5003.jpg)

![e2396ec1a8de9373f15954e7babd182fd3519f33e77aee7632e216c3b9779a9a.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/e2396ec1a8de9373f15954e7babd182fd3519f33e77aee7632e216c3b9779a9a.jpg)

![f9bbc544edd92c9af1cf28b4cd617a1dd7adbea5bdf05b765315ed76316da0f7.jpg](../icml_results/4064_Categorical%20Flow%20Maps/images/f9bbc544edd92c9af1cf28b4cd617a1dd7adbea5bdf05b765315ed76316da0f7.jpg)

### Tables

![364178e28a97d85bc8c9506753777490fa710f62ff49eaf591c2db9f9494adc0.jpg](../icml_results/4064_Categorical%20Flow%20Maps/tables/364178e28a97d85bc8c9506753777490fa710f62ff49eaf591c2db9f9494adc0.jpg)

![886dd961cfc6e7fda77c62decc58a7a5909fac42b2a76657abc761962a997569.jpg](../icml_results/4064_Categorical%20Flow%20Maps/tables/886dd961cfc6e7fda77c62decc58a7a5909fac42b2a76657abc761962a997569.jpg)

![b6dc16802f26e613f82df1a372b90579a5e181812bf1c3c2c51c4961cb131d19.jpg](../icml_results/4064_Categorical%20Flow%20Maps/tables/b6dc16802f26e613f82df1a372b90579a5e181812bf1c3c2c51c4961cb131d19.jpg)

![dcc2439dac9a9047fe3cfe6c6a4d293fe347353ee04dff6d73625c68ec097b46.jpg](../icml_results/4064_Categorical%20Flow%20Maps/tables/dcc2439dac9a9047fe3cfe6c6a4d293fe347353ee04dff6d73625c68ec097b46.jpg)

<a id="paper-4024"></a>

## SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport


### Images

![02172a8f783c96543518b6fb7fc7acd83a5e290ae3ce732ba58ceef6a67428da.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/02172a8f783c96543518b6fb7fc7acd83a5e290ae3ce732ba58ceef6a67428da.jpg)

![16731a647bd85a48abc832c253d04c34c4eda7c1433ee0c6a98fdbb97cf52a50.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/16731a647bd85a48abc832c253d04c34c4eda7c1433ee0c6a98fdbb97cf52a50.jpg)

![24bce4b45b5eac9425e0b92ccd7470641f293867c5a1f24c89e7077e0bd18cda.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/24bce4b45b5eac9425e0b92ccd7470641f293867c5a1f24c89e7077e0bd18cda.jpg)

![27fd0e01445db2f5f5958ed95976a3f62beb3549fa5b9db88969eff429ca21a4.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/27fd0e01445db2f5f5958ed95976a3f62beb3549fa5b9db88969eff429ca21a4.jpg)

![46c8a8e519c8653c30abf8e741ce5aa3f7dd231ff9ea79b89d809dbda216b72f.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/46c8a8e519c8653c30abf8e741ce5aa3f7dd231ff9ea79b89d809dbda216b72f.jpg)

![601658ae837bb2390e46bdf3872b1927ad4e90a58faa2be255d01f5266b4b6d2.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/601658ae837bb2390e46bdf3872b1927ad4e90a58faa2be255d01f5266b4b6d2.jpg)

![709f5ec318551413ec0f4cbfa72cab78b30b091a635f09a004510229f4426981.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/709f5ec318551413ec0f4cbfa72cab78b30b091a635f09a004510229f4426981.jpg)

![720d5fa3feeb6c591da5cae41cf58b32f55a89beedaea94a83d52b710bdbb308.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/720d5fa3feeb6c591da5cae41cf58b32f55a89beedaea94a83d52b710bdbb308.jpg)

![8d206096481a7041a3e1c604c6d8af5a2360655d3111a269d36433cee66c118a.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/8d206096481a7041a3e1c604c6d8af5a2360655d3111a269d36433cee66c118a.jpg)

![938eb810c628203db01e5b2d39dad934bca68d9f5b80300017b04d315910eeb1.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/938eb810c628203db01e5b2d39dad934bca68d9f5b80300017b04d315910eeb1.jpg)

![9548ccdc4da107fbed3eb2b4f1ebb6c0b5591d1bbfd125c59357f64fec846025.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/9548ccdc4da107fbed3eb2b4f1ebb6c0b5591d1bbfd125c59357f64fec846025.jpg)

![99338d60db09728ab5c329fc0734dc50eb8ee94e34ac99f87dbd54bb41342aea.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/99338d60db09728ab5c329fc0734dc50eb8ee94e34ac99f87dbd54bb41342aea.jpg)

![9be687326ae4e1f34f38e9fc1ce8704377d487dfbab42b4503c451a7a2cf7a37.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/9be687326ae4e1f34f38e9fc1ce8704377d487dfbab42b4503c451a7a2cf7a37.jpg)

![b934abe9002eacf936eef2dedd650ed99997ac0736029ea3453bde769a2606e6.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/b934abe9002eacf936eef2dedd650ed99997ac0736029ea3453bde769a2606e6.jpg)

![bb21f160b05cf33cbd674f9b602465c21aed929f3db89a50771e2f44c8639139.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/bb21f160b05cf33cbd674f9b602465c21aed929f3db89a50771e2f44c8639139.jpg)

![c9ba3993f5e81f5a61542a190e7613d087af4f3fd7c9544b3574fb11e90cffb0.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/c9ba3993f5e81f5a61542a190e7613d087af4f3fd7c9544b3574fb11e90cffb0.jpg)

![d9e2702351bed761b28176bbdd739bebea4e541adc42ee79fa4b93dd2daef21b.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/d9e2702351bed761b28176bbdd739bebea4e541adc42ee79fa4b93dd2daef21b.jpg)

![fa86b5370ca1685f4c33c342ae746b4a17ae706fd55c2cb5f3babf0af00a11f0.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/images/fa86b5370ca1685f4c33c342ae746b4a17ae706fd55c2cb5f3babf0af00a11f0.jpg)

### Tables

![03ed24044561d01e80f6d67c5ab4cba82c4fbce2e008f3122fd071f04a2c955f.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/03ed24044561d01e80f6d67c5ab4cba82c4fbce2e008f3122fd071f04a2c955f.jpg)

![1acf81165c647aa8f39cf143ed9b8fd1c4b836d5e58e0ef97d080a57f5bd0494.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/1acf81165c647aa8f39cf143ed9b8fd1c4b836d5e58e0ef97d080a57f5bd0494.jpg)

![34ebcdc3c2d1923c7f11db4fab9e46c0897f3cf2e9c57c3f4fc7becdbb1efcd1.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/34ebcdc3c2d1923c7f11db4fab9e46c0897f3cf2e9c57c3f4fc7becdbb1efcd1.jpg)

![34ecf84350d79fa39f1c4c992d242db09cf8fbf4d9df17db2d13fbfb72e53aa8.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/34ecf84350d79fa39f1c4c992d242db09cf8fbf4d9df17db2d13fbfb72e53aa8.jpg)

![45b6d2607d3ef518583d0a480c60d6ac1df7fbd8f888a4578ce82743f4f82cf2.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/45b6d2607d3ef518583d0a480c60d6ac1df7fbd8f888a4578ce82743f4f82cf2.jpg)

![4776ba15075e55da34b4a6bad8029fd4a15275b293d2200d670df4200d6c26d0.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/4776ba15075e55da34b4a6bad8029fd4a15275b293d2200d670df4200d6c26d0.jpg)

![5875d3e45547e562bea6b4e81ab89aa83a765f288ba4a8283d01d78be0d2be3c.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/5875d3e45547e562bea6b4e81ab89aa83a765f288ba4a8283d01d78be0d2be3c.jpg)

![645ebb3c3e40e1d3479e36d1902083c0fe109e7763938d760bf76f29f73c637d.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/645ebb3c3e40e1d3479e36d1902083c0fe109e7763938d760bf76f29f73c637d.jpg)

![646e05ced7f512f9fa93035c55b5b50a52531e33c6f13d76c227656b428a6704.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/646e05ced7f512f9fa93035c55b5b50a52531e33c6f13d76c227656b428a6704.jpg)

![72b2208de0059e9f17c9d97b047447b0be10c6f938a7d5b9f06cb5d078a50f31.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/72b2208de0059e9f17c9d97b047447b0be10c6f938a7d5b9f06cb5d078a50f31.jpg)

![8fab450599fcd210c86ebc87a1fcd6791827769cf387fb4da80e2e7e9c4d10dc.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/8fab450599fcd210c86ebc87a1fcd6791827769cf387fb4da80e2e7e9c4d10dc.jpg)

![a18b7ecc6ed7aa425c84d90b19a333d0ea1bf8a3b96a8adb73dd2691ab4f0a0d.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/a18b7ecc6ed7aa425c84d90b19a333d0ea1bf8a3b96a8adb73dd2691ab4f0a0d.jpg)

![f5a152625c44a8784bdb75c4c1685a53c78e29154dba116b822efd9c04022f08.jpg](../icml_results/4065_SOTAlign_%20Semi-Supervised%20Alignment%20of%20Unimodal%20Vision%20and%20Language%20Models%20via%20Optimal%20Transport/tables/f5a152625c44a8784bdb75c4c1685a53c78e29154dba116b822efd9c04022f08.jpg)

<a id="paper-4025"></a>

## Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data


### Images

![01125a0557a9eac1eda084d08c7ab7286b81749f36b9a13c5dfb5640a9a54467.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/01125a0557a9eac1eda084d08c7ab7286b81749f36b9a13c5dfb5640a9a54467.jpg)

![01176bd568bfed9ee96e0a5c489b308dd252507cc656a8d7b36118152514c53e.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/01176bd568bfed9ee96e0a5c489b308dd252507cc656a8d7b36118152514c53e.jpg)

![02b1b118d6acf49461cea49781c6bc2bba2bb777979b30552475f6434f4fef43.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/02b1b118d6acf49461cea49781c6bc2bba2bb777979b30552475f6434f4fef43.jpg)

![0f10ea4d93769482e6e7ff6beb462d6c18b7afa062620406372aa28667a81039.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/0f10ea4d93769482e6e7ff6beb462d6c18b7afa062620406372aa28667a81039.jpg)

![151b5a8c1b04bb3487b0011fb6892a0ef6e5f0194d6ad7fd843e68ae64a3973c.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/151b5a8c1b04bb3487b0011fb6892a0ef6e5f0194d6ad7fd843e68ae64a3973c.jpg)

![17798f4454452831e75815791eadcc21a35952ceb1f24c70f045a2ad38bc6bcf.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/17798f4454452831e75815791eadcc21a35952ceb1f24c70f045a2ad38bc6bcf.jpg)

![18522997dc002407bd1553d2de0ad300c750f13b73e5e647b9ebca6e34f0b431.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/18522997dc002407bd1553d2de0ad300c750f13b73e5e647b9ebca6e34f0b431.jpg)

![18bd8fea2353ca3d0b06cf6d6b8a8b79a15d99bde0dfe26f1cbc9e61c9424bed.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/18bd8fea2353ca3d0b06cf6d6b8a8b79a15d99bde0dfe26f1cbc9e61c9424bed.jpg)

![19e33157dcbd4015bb256a354f70ab1eee3b54ea93a13e2d382a7fa3def03149.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/19e33157dcbd4015bb256a354f70ab1eee3b54ea93a13e2d382a7fa3def03149.jpg)

![1bb06e4f560da8566a3906c2c6eefaad48300c0ed77f715adfbceba9fb769e35.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/1bb06e4f560da8566a3906c2c6eefaad48300c0ed77f715adfbceba9fb769e35.jpg)

![25e2134f28d329a11e57d8d824a21d44201306d7a4adaa7626ba9b8fe02627e1.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/25e2134f28d329a11e57d8d824a21d44201306d7a4adaa7626ba9b8fe02627e1.jpg)

![2ab70bda90c6a3277a33205f84f8bbe90c0991e5fe4b5e2d9872b4f9437ffb65.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/2ab70bda90c6a3277a33205f84f8bbe90c0991e5fe4b5e2d9872b4f9437ffb65.jpg)

![41626ace316b0933100ac9ea59e69520e738568610214d75402519b424b53ad8.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/41626ace316b0933100ac9ea59e69520e738568610214d75402519b424b53ad8.jpg)

![4b37596a2c8df0e02235de8ea056bfbb378aaa7d0f1a13ff7a19f59159c1d6e5.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/4b37596a2c8df0e02235de8ea056bfbb378aaa7d0f1a13ff7a19f59159c1d6e5.jpg)

![531296b235b71f042cc966b94874f128f26263be81345128bef39de0a4fc3ab1.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/531296b235b71f042cc966b94874f128f26263be81345128bef39de0a4fc3ab1.jpg)

![5608cfd909e9360ea81ebf1ae6fe66d8f7952daedfba7ea12920c6b9c8f88760.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/5608cfd909e9360ea81ebf1ae6fe66d8f7952daedfba7ea12920c6b9c8f88760.jpg)

![56ffb605a5a9fc1b68155f7c1512278c63c2bd05883a592caa17bffe0ccbcfda.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/56ffb605a5a9fc1b68155f7c1512278c63c2bd05883a592caa17bffe0ccbcfda.jpg)

![5dce509517bc4a8840d3fa4b69f90c611266dd88d5fb3ae6cf4955b21f1a11be.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/5dce509517bc4a8840d3fa4b69f90c611266dd88d5fb3ae6cf4955b21f1a11be.jpg)

![6253cfc869be75f695b0f4c887e78d5c5133e445c7b8d2d90cc9e8dd5ad05f57.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/6253cfc869be75f695b0f4c887e78d5c5133e445c7b8d2d90cc9e8dd5ad05f57.jpg)

![68bc8e0d6cc2d2dd14ef85e023724abd2b87494261be8b2304cee3422381b4a7.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/68bc8e0d6cc2d2dd14ef85e023724abd2b87494261be8b2304cee3422381b4a7.jpg)

![7648246ee9516c2df90a21f06120d5e628ce58735ef03550e5718b60dd195ce0.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/7648246ee9516c2df90a21f06120d5e628ce58735ef03550e5718b60dd195ce0.jpg)

![784f614529a2239f120fc877974b50acd4ee4355931b6382e3eaf5fbc27203e3.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/784f614529a2239f120fc877974b50acd4ee4355931b6382e3eaf5fbc27203e3.jpg)

![8d93ef3a537956dfa8942f1d4fff77df9df7b558700640d0fe33714d61746445.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/8d93ef3a537956dfa8942f1d4fff77df9df7b558700640d0fe33714d61746445.jpg)

![a66ce509995c15710b51adddb8c60f5815f7e915accf17176da2aa87a1bb00e2.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/a66ce509995c15710b51adddb8c60f5815f7e915accf17176da2aa87a1bb00e2.jpg)

![c2c983d067c8bfbf445a8cb92cede2e3a30877819c278805d9531a450a734265.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/c2c983d067c8bfbf445a8cb92cede2e3a30877819c278805d9531a450a734265.jpg)

![d574a8e0a51c2a3255df159d0a8f9bfc69de263ac575992d5e6867127ea232d5.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/d574a8e0a51c2a3255df159d0a8f9bfc69de263ac575992d5e6867127ea232d5.jpg)

![da09ccc1b9f2c82c77e4616c534241ec98f87f9db00b2bf6bd4f09ebeb37931b.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/da09ccc1b9f2c82c77e4616c534241ec98f87f9db00b2bf6bd4f09ebeb37931b.jpg)

![f0953d6bab9ba6f52cfd38e6168954528cce23cf92cb223968f7ed25ed3222bb.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/f0953d6bab9ba6f52cfd38e6168954528cce23cf92cb223968f7ed25ed3222bb.jpg)

![f39245b54d954807d485c340469f874b69e009378b6cca5116b30bddbda069b3.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/f39245b54d954807d485c340469f874b69e009378b6cca5116b30bddbda069b3.jpg)

![fb8617ec7719e0de3a7b12e02cc21ee6fe039a4dd857e6d32f98d8741cf0f0f2.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/images/fb8617ec7719e0de3a7b12e02cc21ee6fe039a4dd857e6d32f98d8741cf0f0f2.jpg)

### Tables

![04611461416dee50a47c9afea2db09775f0a195ee483aaf0ff83ed9ff647a90e.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/04611461416dee50a47c9afea2db09775f0a195ee483aaf0ff83ed9ff647a90e.jpg)

![06e46cce33e7d03689f2619029c76afaa4642efce1fe25904ee08289594494fb.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/06e46cce33e7d03689f2619029c76afaa4642efce1fe25904ee08289594494fb.jpg)

![09ba43812e1a3fedefa70160dd9ba5813621b8fabce949649c3e1636c3415125.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/09ba43812e1a3fedefa70160dd9ba5813621b8fabce949649c3e1636c3415125.jpg)

![10ff0b3f97e5940224097ab69be7c945dc94851355721c1eb04c20017d7879a6.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/10ff0b3f97e5940224097ab69be7c945dc94851355721c1eb04c20017d7879a6.jpg)

![1acf6fb5b6210efae4fa7f0ddf048694d2ee716acbcaf078072c8b7486a9e9c7.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/1acf6fb5b6210efae4fa7f0ddf048694d2ee716acbcaf078072c8b7486a9e9c7.jpg)

![234e6301054f0dd5b41d593561d4283353bc7197d5ff9d0aade0189795dd50f0.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/234e6301054f0dd5b41d593561d4283353bc7197d5ff9d0aade0189795dd50f0.jpg)

![2a75421f8b43e94dbbf5cd80fc72f40e3cae805535a3ff5aa4d4b884ac9e0cc5.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/2a75421f8b43e94dbbf5cd80fc72f40e3cae805535a3ff5aa4d4b884ac9e0cc5.jpg)

![3c70780366a708d539599fd811ead6e8364b535353d00d7bcc8a5d91e775eac4.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/3c70780366a708d539599fd811ead6e8364b535353d00d7bcc8a5d91e775eac4.jpg)

![4b7611649e3e97e3950d98dad1c90ba89158ee4940b4414eb829a1c7abc660b0.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/4b7611649e3e97e3950d98dad1c90ba89158ee4940b4414eb829a1c7abc660b0.jpg)

![53be9bd911d81ccc157546cbcb8f517cceb2301f1ccf9288f1a4fd798c4999ee.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/53be9bd911d81ccc157546cbcb8f517cceb2301f1ccf9288f1a4fd798c4999ee.jpg)

![6de4ef951619b1f2c0e2384d0975af7bf1c14b5af39319a03950b70676e7f093.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/6de4ef951619b1f2c0e2384d0975af7bf1c14b5af39319a03950b70676e7f093.jpg)

![7d2df8ebcb4b41fdf5c42310fdf86564d3c9458ce1f2378015afc6c4f1174f17.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/7d2df8ebcb4b41fdf5c42310fdf86564d3c9458ce1f2378015afc6c4f1174f17.jpg)

![8e26ff55598d7d9e5f9304a1a99d6b06145a59aafba8128d745352b753bd77d7.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/8e26ff55598d7d9e5f9304a1a99d6b06145a59aafba8128d745352b753bd77d7.jpg)

![97e9c040d501ae5cd49ea817c75987a696409053c6609484b0dc29caa0a24939.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/97e9c040d501ae5cd49ea817c75987a696409053c6609484b0dc29caa0a24939.jpg)

![98f2fbe4851ce0d8c3fcfca8625045d4ed20aa7d2e5d7a89597be2d40cea2e64.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/98f2fbe4851ce0d8c3fcfca8625045d4ed20aa7d2e5d7a89597be2d40cea2e64.jpg)

![9be67ad98a0793f06e291b9e50eac50f8ee1043a02f643511c6eb6257e8712ad.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/9be67ad98a0793f06e291b9e50eac50f8ee1043a02f643511c6eb6257e8712ad.jpg)

![b960e38b2b54a8c566b08962fca8008a96802af1800c7cd55dccfd64a7ab069c.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/b960e38b2b54a8c566b08962fca8008a96802af1800c7cd55dccfd64a7ab069c.jpg)

![bc9808fd17449480e9c8dd7eda803ae8b022bbb243590a5db63e981b87d1ec8c.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/bc9808fd17449480e9c8dd7eda803ae8b022bbb243590a5db63e981b87d1ec8c.jpg)

![c365bd81cdc8dbd8b958416e5fb28330cf69912e209bb6a87655b08732919a0d.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/c365bd81cdc8dbd8b958416e5fb28330cf69912e209bb6a87655b08732919a0d.jpg)

![c435c3cb3a1f5d785a0d863081a7a5798aa6539e0cd564d95baa5014409843f0.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/c435c3cb3a1f5d785a0d863081a7a5798aa6539e0cd564d95baa5014409843f0.jpg)

![c8376864bc3e99efac006c15d9e807ccdd79e7f0ef5b24e4fde465efc49ba78a.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/c8376864bc3e99efac006c15d9e807ccdd79e7f0ef5b24e4fde465efc49ba78a.jpg)

![f41255343d71510ebd4caedd7a9098f7a95048502a13d1505361080afedb8faa.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/f41255343d71510ebd4caedd7a9098f7a95048502a13d1505361080afedb8faa.jpg)

![f5041a00976eefd27e21e8406f9c44377cc367d03049d6c52163263dd23bdd45.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/f5041a00976eefd27e21e8406f9c44377cc367d03049d6c52163263dd23bdd45.jpg)

![f9764f90620096ed608238965c9fdc138495581fe414e32f5bc8c05769f1df86.jpg](../icml_results/4066_Privately%20Fine-Tuned%20LLMs%20Preserve%20Temporal%20Dynamics%20in%20Tabular%20Data/tables/f9764f90620096ed608238965c9fdc138495581fe414e32f5bc8c05769f1df86.jpg)

<a id="paper-4026"></a>

## Preserving Plasticity in Continual Learning via Dynamical Isometry

### Images

![032daa9438de8098787ef4a57d725a9d2e1104d7e08746de5710c53e38296624.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/032daa9438de8098787ef4a57d725a9d2e1104d7e08746de5710c53e38296624.jpg)

![0d9b3802b983501618f664c27a1ac97b7f11a24116f5215c4aa0de888d927892.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/0d9b3802b983501618f664c27a1ac97b7f11a24116f5215c4aa0de888d927892.jpg)

![0f5958109896ed360c5e47c8c60f8257d0bb320e3e16646abe90af83ea6a6503.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/0f5958109896ed360c5e47c8c60f8257d0bb320e3e16646abe90af83ea6a6503.jpg)

![16481f4edea6c3d6617f7ab7d90e5167c943cec7026e2973d5a7d3f42533336f.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/16481f4edea6c3d6617f7ab7d90e5167c943cec7026e2973d5a7d3f42533336f.jpg)

![175a09ed000e8160b104babd64ea5c103d8839030240055fce804a97e74fe927.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/175a09ed000e8160b104babd64ea5c103d8839030240055fce804a97e74fe927.jpg)

![2031a2fcc3fb346a422c2c11c31b87e5fddaa12563597b44c5bd8e4316ce993e.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/2031a2fcc3fb346a422c2c11c31b87e5fddaa12563597b44c5bd8e4316ce993e.jpg)

![2453c8246fa52b5ac894f29ba7dfbf4604fc5731a8cc39bd4ecfbdfd6e078f72.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/2453c8246fa52b5ac894f29ba7dfbf4604fc5731a8cc39bd4ecfbdfd6e078f72.jpg)

![2a9b8a7bf7b8d6a8bcd26638f145aebcccd152be68b0953968a177b6094fdfe4.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/2a9b8a7bf7b8d6a8bcd26638f145aebcccd152be68b0953968a177b6094fdfe4.jpg)

![467a008d279b849b8fe575c0cdb7d54529e125af8086dfc25b0edd146a6b6876.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/467a008d279b849b8fe575c0cdb7d54529e125af8086dfc25b0edd146a6b6876.jpg)

![4db6b64372c50b69d213a674d430070bca28f84d5346e7a5953d82c76e021c76.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/4db6b64372c50b69d213a674d430070bca28f84d5346e7a5953d82c76e021c76.jpg)

![4e90231c9464e568ee5e09cb50a3a9dc4ae6ca34dde894fcc7c97ed7b82e2fcb.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/4e90231c9464e568ee5e09cb50a3a9dc4ae6ca34dde894fcc7c97ed7b82e2fcb.jpg)

![4f594558716f52e99197f8dfbceaf9842734a4b3c479c9cc5e7b39cdfdbe5135.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/4f594558716f52e99197f8dfbceaf9842734a4b3c479c9cc5e7b39cdfdbe5135.jpg)

![570bf3c6ebcda2356e240668f56213d71344997c791c2e3ecffd2f368d866e01.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/570bf3c6ebcda2356e240668f56213d71344997c791c2e3ecffd2f368d866e01.jpg)

![5929ac1d7b17151090de56fd58829b68d0f4220e7033cd8e0e248c80f7b2b485.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/5929ac1d7b17151090de56fd58829b68d0f4220e7033cd8e0e248c80f7b2b485.jpg)

![5a17d0a537522b536375ca57afe6fac20f524e6e9a95acaa2d18f3019339328c.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/5a17d0a537522b536375ca57afe6fac20f524e6e9a95acaa2d18f3019339328c.jpg)

![60ce9e4d948df9056561411d96f9a07452729a69b70fc8100b2012f5c032f23c.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/60ce9e4d948df9056561411d96f9a07452729a69b70fc8100b2012f5c032f23c.jpg)

![68944c4d611930df73a18e6c891127967ed663d361dc23c527e16507d526eb81.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/68944c4d611930df73a18e6c891127967ed663d361dc23c527e16507d526eb81.jpg)

![76254120b3aa99879ca14861a0e770af3630261cbb35dccea8f4209b291d6cec.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/76254120b3aa99879ca14861a0e770af3630261cbb35dccea8f4209b291d6cec.jpg)

![837593947f9bc81e3410cfdc5f8bf8a9e04cf65db5ad1c17e7eaa87b426ce940.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/837593947f9bc81e3410cfdc5f8bf8a9e04cf65db5ad1c17e7eaa87b426ce940.jpg)

![9743b4a2fb73afd133b753d49d327396392208b78c14803fb25f502262506e7c.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/9743b4a2fb73afd133b753d49d327396392208b78c14803fb25f502262506e7c.jpg)

![987bc7d25f48fc3a044b11b909cec309f3b8ecca4dbef3876e2c98f9a4912231.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/987bc7d25f48fc3a044b11b909cec309f3b8ecca4dbef3876e2c98f9a4912231.jpg)

![a79e9c241ac97d1686eba11ef412c83c209552d286fa9587d7bda21e46855a02.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/a79e9c241ac97d1686eba11ef412c83c209552d286fa9587d7bda21e46855a02.jpg)

![a9768cda3be80ee0c542c31c8ed45c6bfee5e92cb733d10153e92b401f75126a.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/a9768cda3be80ee0c542c31c8ed45c6bfee5e92cb733d10153e92b401f75126a.jpg)

![af31acb45b93d63be52d07ae2c3e5f74d15a41de74e8b6cfe0a3ff528ca5f907.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/af31acb45b93d63be52d07ae2c3e5f74d15a41de74e8b6cfe0a3ff528ca5f907.jpg)

![b0b2efd449e2d7cb7a840a7a4fb97d48449a5b0e7ce455c3236acf5ab2e025ec.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/b0b2efd449e2d7cb7a840a7a4fb97d48449a5b0e7ce455c3236acf5ab2e025ec.jpg)

![b53a1f937987c2387941e1051bd5820cd5238d628f88cd29c6b2e13731cd0ece.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/b53a1f937987c2387941e1051bd5820cd5238d628f88cd29c6b2e13731cd0ece.jpg)

![bed7989097a851dad5a67de89aa84ba1ef7d54d8adce040e2898ecbf39a94019.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/bed7989097a851dad5a67de89aa84ba1ef7d54d8adce040e2898ecbf39a94019.jpg)

![c0dc400e33180f4a07ad2d6ac04f5f775e15523c6ac89a28122fb779e43b0707.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/c0dc400e33180f4a07ad2d6ac04f5f775e15523c6ac89a28122fb779e43b0707.jpg)

![c34b0966b389230bc05d9fe87ba8135e7d0358ca1246d7756c0d11da82e0e9ca.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/c34b0966b389230bc05d9fe87ba8135e7d0358ca1246d7756c0d11da82e0e9ca.jpg)

![e4c25f614f5728af906c550566fb1d696f25f19d9907b13b4b5438086d1554e2.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/e4c25f614f5728af906c550566fb1d696f25f19d9907b13b4b5438086d1554e2.jpg)

![eb5b31730de2f70a4573899b43e4badf2ab770056169821012f01e6812bf473c.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/eb5b31730de2f70a4573899b43e4badf2ab770056169821012f01e6812bf473c.jpg)

![fb75d432023a0d1e2459e598b8afb3e1e56072e6cec35008b396302f3f792078.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/images/fb75d432023a0d1e2459e598b8afb3e1e56072e6cec35008b396302f3f792078.jpg)

### Tables

![4110ce747f180263aba2b7354420bc267b21f5d8a8a721e30c5f6564ee77c8fd.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/tables/4110ce747f180263aba2b7354420bc267b21f5d8a8a721e30c5f6564ee77c8fd.jpg)

![56f875782f53f842483698253228de7ca2cf9632f2ab47974918c5b491fb1325.jpg](../icml_results/4067_Preserving%20Plasticity%20in%20Continual%20Learning%20via%20Dynamical%20Isometry/tables/56f875782f53f842483698253228de7ca2cf9632f2ab47974918c5b491fb1325.jpg)
