# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 108 of 197

## 目录 (Table of Contents)

1. [On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](#paper-3532)
2. [Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models](#paper-3533)
3. [ToaSt: Token Channel Selection and Structured Pruning for Efficient ViT](#paper-3534)
4. [Understanding Dynamic Compute Allocation in Recurrent Transformers](#paper-3535)
5. [Future Dynamic 3D Reconstruction: Toward 3D World Modeling with Disentangled Ego-Motion](#paper-3536)
6. [IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](#paper-3537)
7. [Probing the Geometry of Diffusion Models with the String Method](#paper-3538)
8. [Spectral Guidance for Flexible and Efficient Control of Diffusion Models](#paper-3539)
9. [Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](#paper-3540)
10. [On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning](#paper-3541)
11. [Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](#paper-3542)
12. [Compositional Generative Modeling from Decentralized Data](#paper-3543)
13. [Incremental Transformer Neural Processes](#paper-3544)
14. [Learning to Evict from Key-Value Cache](#paper-3545)
15. [Fair Dataset Distillation via Cross-Group Barycenter Alignment](#paper-3546)
16. [Causal Detection of Multi-Step LLM Agent Attacks](#paper-3547)
17. [LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning](#paper-3548)
18. [SpatialJB: How Text Distribution Art Becomes The "Jailbreak Key" for LLM Guardrails](#paper-3549)
19. [Bayesian Rain Field Reconstruction using Commercial Microwave Links and Diffusion Model Priors](#paper-3550)
20. [Imposing Boundary Conditions on Neural Operators via Learned Function Extensions](#paper-3551)
21. [Spurious Correlation Learning in Preference Optimization: Mechanisms, Consequences, and Mitigation via Tie Training](#paper-3552)
22. [Calibrated Knowledge Aggregation in Bayesian Mixture-of-Experts for Continual VQA](#paper-3553)
23. [PepCompass: Navigating Peptide Embedding Spaces Using Riemannian Geometry](#paper-3554)
24. [Reinforcement Learning with Verifiable Rewards: GRPO’s Loss, Dynamics, and Success Amplification](#paper-3555)
25. [DRIFT: Decoupled Rollouts and Importance-Weighted Fine-Tuning for Efficient Multi-Turn Optimization](#paper-3556)
26. [Federated Manifold Learning (FML): Tackling Domain Heterogeneity with Structural Knowledge Transfer](#paper-3557)
27. [Multi-Integration of Labels Across Categories for Component Identification in Multi-trial Time Series](#paper-3558)
28. [Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](#paper-3559)
29. [UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations](#paper-3560)
30. [When Does Sparsity Mitigate the Curse of Depth in LLMs](#paper-3561)
31. [BOCLOAK: Optimal Transport-Guided Adversarial Attacks on Graph Neural Network-Based Bot Detection](#paper-3562)
32. [Latent Diffusion Pretraining for Crystal Property Prediction](#paper-3563)
33. [SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights](#paper-3564)

---


<a id="paper-3532"></a>

## On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach

### Images

![1db09df3b8aa8898f1f5e319744f6cfac4806009f0fde98022900e6164721b35.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/1db09df3b8aa8898f1f5e319744f6cfac4806009f0fde98022900e6164721b35.jpg)

![3b0845e58c007e1b448dc8615c332822f30239c1c3c3b6d0de581555831a2460.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/3b0845e58c007e1b448dc8615c332822f30239c1c3c3b6d0de581555831a2460.jpg)

![4f28d1da20ceb808b90437ca328e54ac5bae284719b95b0302802f045ce367d4.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/4f28d1da20ceb808b90437ca328e54ac5bae284719b95b0302802f045ce367d4.jpg)

![56d1615f98a2bdbee00b0e992dc1373ff48d5b7f4f0b9541463ab7275d26f2ec.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/56d1615f98a2bdbee00b0e992dc1373ff48d5b7f4f0b9541463ab7275d26f2ec.jpg)

![758033c2391f86396cda16d77ae4b02e22731fd4d237af58a5a499c563ac80bd.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/758033c2391f86396cda16d77ae4b02e22731fd4d237af58a5a499c563ac80bd.jpg)

![786d0705f5d0c89ad38f36cb5627f517e17c4c5309493b50880d30b5f72048b9.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/786d0705f5d0c89ad38f36cb5627f517e17c4c5309493b50880d30b5f72048b9.jpg)

![78df5a341cdd87beb6538b6cb0ec730788e10b1e43d176e3ac614dbbfad09055.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/78df5a341cdd87beb6538b6cb0ec730788e10b1e43d176e3ac614dbbfad09055.jpg)

![7af7be4d3c0d8b47fc26208fa8f19af3f49874318b7b8c71df08e26f92b1aa1d.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/7af7be4d3c0d8b47fc26208fa8f19af3f49874318b7b8c71df08e26f92b1aa1d.jpg)

![8010f968a275be7fad3d673bbeb7c41009111e0765ec1488e9c3b1289b81a9af.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/8010f968a275be7fad3d673bbeb7c41009111e0765ec1488e9c3b1289b81a9af.jpg)

![92d9f1d20c2e0345e012c8e023a3f7820d4a0947caf76e1fbd49f592b9e83906.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/92d9f1d20c2e0345e012c8e023a3f7820d4a0947caf76e1fbd49f592b9e83906.jpg)

![b056d1baed7f6bbe6753664e8a3039da3b52317d11ac9e309c1d0315510836ac.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/b056d1baed7f6bbe6753664e8a3039da3b52317d11ac9e309c1d0315510836ac.jpg)

![bc6139edc4ed0c38ed0f8c23942e0ef82f57809241b0e025ad531c5660751d9b.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/bc6139edc4ed0c38ed0f8c23942e0ef82f57809241b0e025ad531c5660751d9b.jpg)

![cb10f495da633d3de1fa906dc669657d9935d6af4043b4c42c44f6220ba91c45.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/cb10f495da633d3de1fa906dc669657d9935d6af4043b4c42c44f6220ba91c45.jpg)

![d9acb60245315af06ffaa69cc0ff23b62eb615daac979fbeaa0ab6b504d30e98.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/d9acb60245315af06ffaa69cc0ff23b62eb615daac979fbeaa0ab6b504d30e98.jpg)

![f32dca27f0465984412b48b16fbb3c00fb998a922dc8e2ff2d62374b9bd5cb01.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/images/f32dca27f0465984412b48b16fbb3c00fb998a922dc8e2ff2d62374b9bd5cb01.jpg)

### Tables

![0281c3b10ec545817778b35c1e083518ebdc0675ed9f998d1444d68a56c99b40.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/tables/0281c3b10ec545817778b35c1e083518ebdc0675ed9f998d1444d68a56c99b40.jpg)

![505373750f747adf3b8dc1d150bd39ff842b8ff4fe0663907ebebcaaab9d73bd.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/tables/505373750f747adf3b8dc1d150bd39ff842b8ff4fe0663907ebebcaaab9d73bd.jpg)

![88c9b464e871bdf1283b356f82a98cb0e5c6d250b3e72fef6b703062d28489e7.jpg](../icml_results/3571_How%20Should%20Transformers%20Encode%20Numeric%20Values%20in%20Electronic%20Health%20Records_/tables/88c9b464e871bdf1283b356f82a98cb0e5c6d250b3e72fef6b703062d28489e7.jpg)

## On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach


### Images

![44b9387f868c51b17abded7e4259d3515b93a015b0b74cbaf97185241d204c76.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/images/44b9387f868c51b17abded7e4259d3515b93a015b0b74cbaf97185241d204c76.jpg)

![84e1e076d4d31916d9f970afb5799ae1057f916151b68ac289c6eacb76389470.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/images/84e1e076d4d31916d9f970afb5799ae1057f916151b68ac289c6eacb76389470.jpg)

![92535d1663492bd28e4f8f71b1ed20996cad0bef87df0e62edf4f88816328c58.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/images/92535d1663492bd28e4f8f71b1ed20996cad0bef87df0e62edf4f88816328c58.jpg)

![ae3e5c34158d18567642973da1a8092ef4f08693ed1ecb474ad5990365213213.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/images/ae3e5c34158d18567642973da1a8092ef4f08693ed1ecb474ad5990365213213.jpg)

### Tables

![1140a92ff42497601ae5fda4823bd28d4599e02a21904672046df375894f735f.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/tables/1140a92ff42497601ae5fda4823bd28d4599e02a21904672046df375894f735f.jpg)

![3fb3c3f57a392e0f534f86e991f5079f548915fecf7c5259a2bac41f1e403bd1.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/tables/3fb3c3f57a392e0f534f86e991f5079f548915fecf7c5259a2bac41f1e403bd1.jpg)

![f5d635fdfe03f1c9cc26a87f12a920a5e8a7b77311ed8f5a36d5a1c2596abdd5.jpg](../icml_results/3572_On%20the%20Interaction%20of%20Batch%20Noise%2C%20Adaptivity%2C%20and%20Compression%2C%20under%20%24%28L_0%2CL_1%29%24-Smoothness_%20An%20SDE/tables/f5d635fdfe03f1c9cc26a87f12a920a5e8a7b77311ed8f5a36d5a1c2596abdd5.jpg)

<a id="paper-3533"></a>

## Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models


### Images

![01fc5c10323b660a514d6ac42b07df7dfacd33c275ac6455e4a4e06290f7ee06.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/01fc5c10323b660a514d6ac42b07df7dfacd33c275ac6455e4a4e06290f7ee06.jpg)

![050e3e6eee07ff84389801000b38b14224a767d7d202a3ce1c6cce3daabbafd1.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/050e3e6eee07ff84389801000b38b14224a767d7d202a3ce1c6cce3daabbafd1.jpg)

![132ea50823a99b65bf6c5ea65320b12f6cfcf701ad94a722e37e74b50c9b431a.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/132ea50823a99b65bf6c5ea65320b12f6cfcf701ad94a722e37e74b50c9b431a.jpg)

![28b102c007cbcd93ef26982206900655290f2131ca60298f2a0d6da1c8245579.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/28b102c007cbcd93ef26982206900655290f2131ca60298f2a0d6da1c8245579.jpg)

![7eeb8e2b0b6995bbc445e6e9a3d323be67b14651e3d360bc4aaff187e7320feb.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/7eeb8e2b0b6995bbc445e6e9a3d323be67b14651e3d360bc4aaff187e7320feb.jpg)

![84afb096b65ff63022f21a9ac47076fab9728522a814ecfc84f4f6577991d2c9.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/84afb096b65ff63022f21a9ac47076fab9728522a814ecfc84f4f6577991d2c9.jpg)

![d2dca20d0d3f2e4a2c969206d39d32e1ff8a02fdeac52f1a1df0b5057b8ce977.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/images/d2dca20d0d3f2e4a2c969206d39d32e1ff8a02fdeac52f1a1df0b5057b8ce977.jpg)

### Tables

![0d4e8cbea74ab870f8854b250ede9dfdbaaa2e022dba3f321a52aaa21e40482c.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/0d4e8cbea74ab870f8854b250ede9dfdbaaa2e022dba3f321a52aaa21e40482c.jpg)

![1f2853cf9897e86043a651b304f49ebaee221425b34e07dcef8b46ee622c86a0.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/1f2853cf9897e86043a651b304f49ebaee221425b34e07dcef8b46ee622c86a0.jpg)

![47dc1112eec88a746bac6531d2ffc793a2c15a3e17a395ad11b30b837de42476.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/47dc1112eec88a746bac6531d2ffc793a2c15a3e17a395ad11b30b837de42476.jpg)

![69e7b63346e32d0ee4aadfff2ce81518621dfd784875a8ce9db002133ab2bda0.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/69e7b63346e32d0ee4aadfff2ce81518621dfd784875a8ce9db002133ab2bda0.jpg)

![772d5e35d758438ecba34fffef252c3a0eeed94f281fb9544374d3d7a073e5bf.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/772d5e35d758438ecba34fffef252c3a0eeed94f281fb9544374d3d7a073e5bf.jpg)

![8f31944059832068ee1546a9a19afb8cbe4e77cf2a83cfb3feca87d720bb0bea.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/8f31944059832068ee1546a9a19afb8cbe4e77cf2a83cfb3feca87d720bb0bea.jpg)

![9724b84ba0079babf0664e2a7670b5d86b22953672579ad6ae2d058843bb0d75.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/9724b84ba0079babf0664e2a7670b5d86b22953672579ad6ae2d058843bb0d75.jpg)

![a74d7d09696465b474825ccddbe829ed16ca8a2b28b7f9978f3ea052e9c38748.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/a74d7d09696465b474825ccddbe829ed16ca8a2b28b7f9978f3ea052e9c38748.jpg)

![b80d2af34804477eeb1379cd58d2e5404bcbc52f75f948172fd2e2ba0c41fc97.jpg](../icml_results/3573_Beyond%20Perplexity_%20UTF-8%20Validity%20in%20Byte-aware%20Language%20Models/tables/b80d2af34804477eeb1379cd58d2e5404bcbc52f75f948172fd2e2ba0c41fc97.jpg)

<a id="paper-3534"></a>

## ToaSt: Token Channel Selection and Structured Pruning for Efficient ViT


### Images

![11fb20cb14c85d7673bb1598dced7818d0fe92e70db431e1d48218d852d964a1.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/11fb20cb14c85d7673bb1598dced7818d0fe92e70db431e1d48218d852d964a1.jpg)

![38bb4b65a32e44e9f7802014edd19ab56cad4ad103ab4f6ffaf893395de3fba0.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/38bb4b65a32e44e9f7802014edd19ab56cad4ad103ab4f6ffaf893395de3fba0.jpg)

![5d5a5c6acfca2cd90b12c7611535a735aaac52573c25115f872908d0f3e5efc5.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/5d5a5c6acfca2cd90b12c7611535a735aaac52573c25115f872908d0f3e5efc5.jpg)

![824b91595275f8741cdbf7c7ab49743bda4da906cb54605654eb9b3e0aabe7c9.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/824b91595275f8741cdbf7c7ab49743bda4da906cb54605654eb9b3e0aabe7c9.jpg)

![c46c5691d67b89395c6fddd545a61b6cf7c2aee70577a79d0b9a4661ff1dd0f9.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/c46c5691d67b89395c6fddd545a61b6cf7c2aee70577a79d0b9a4661ff1dd0f9.jpg)

![e732c4295dadc3295b046b6cdf0cf6e2cedba33e26fd2de9fab6ad428fe976df.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/images/e732c4295dadc3295b046b6cdf0cf6e2cedba33e26fd2de9fab6ad428fe976df.jpg)

### Tables

![162a197a03ee48029c398035ffbcacf4ddd5475d2662e02f219080d005e018ee.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/162a197a03ee48029c398035ffbcacf4ddd5475d2662e02f219080d005e018ee.jpg)

![32473ebb662dbc0cee46f9d9d32140f3dee36f37ace4b94c6447332697be9dcb.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/32473ebb662dbc0cee46f9d9d32140f3dee36f37ace4b94c6447332697be9dcb.jpg)

![3f2be8968953d6466fa86e2324734ce81fb801cb2e386a6bd7519fdf2479dac4.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/3f2be8968953d6466fa86e2324734ce81fb801cb2e386a6bd7519fdf2479dac4.jpg)

![42608d716abeb1e40a81c8a2088038a81ba3304e797aeca541a36a5c13166c23.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/42608d716abeb1e40a81c8a2088038a81ba3304e797aeca541a36a5c13166c23.jpg)

![42ef299123afe15187d4be34aae82148c03fe3c031a4ec845485f9082a829837.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/42ef299123afe15187d4be34aae82148c03fe3c031a4ec845485f9082a829837.jpg)

![511b1e590c32a102df2814bd98551227b0d1adc268a5ac27453712fede1a1964.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/511b1e590c32a102df2814bd98551227b0d1adc268a5ac27453712fede1a1964.jpg)

![58777ab0ec46998eee6656fd172f18e9269ea4f1704d512dbcf1b915d71cc5cb.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/58777ab0ec46998eee6656fd172f18e9269ea4f1704d512dbcf1b915d71cc5cb.jpg)

![682b14beca8dc17e63b2dc6cf20873043f8dc64d4c5156d37700c6f50fac8075.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/682b14beca8dc17e63b2dc6cf20873043f8dc64d4c5156d37700c6f50fac8075.jpg)

![695613708e5096e46ac02222bfb7587fb9b9d13a980fb5c0dee166091d70046b.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/695613708e5096e46ac02222bfb7587fb9b9d13a980fb5c0dee166091d70046b.jpg)

![69c1c5cba3801dc2648c82a4d221fe1a84ab403d266d31b8e2c8e5d46472ba36.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/69c1c5cba3801dc2648c82a4d221fe1a84ab403d266d31b8e2c8e5d46472ba36.jpg)

![7528b834fb55c1f31d36a24b18edd98e9e09f87469cf9655d8f1f45afc4cd49b.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/7528b834fb55c1f31d36a24b18edd98e9e09f87469cf9655d8f1f45afc4cd49b.jpg)

![8113bf30dab7d44ac1bd54febd0f78ef89b256ed18e8a060bc929920a17140eb.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/8113bf30dab7d44ac1bd54febd0f78ef89b256ed18e8a060bc929920a17140eb.jpg)

![820ba58befbd23d25907f672b29a78df39b753550640def2cb5e6ffb2f0db1bb.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/820ba58befbd23d25907f672b29a78df39b753550640def2cb5e6ffb2f0db1bb.jpg)

![9782732e386e987307b0a07717872656007347d28eed6e2efbda063c9c31b4e7.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/9782732e386e987307b0a07717872656007347d28eed6e2efbda063c9c31b4e7.jpg)

![9e0ab293160502447b98bf9fae9edf9ad6626e6cadff498d00919f3b294a2e67.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/9e0ab293160502447b98bf9fae9edf9ad6626e6cadff498d00919f3b294a2e67.jpg)

![ab0e576b28930fdcd86e7766addf2d6eadd480d05adb8b04541c71e4d9b4aefb.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/ab0e576b28930fdcd86e7766addf2d6eadd480d05adb8b04541c71e4d9b4aefb.jpg)

![af5fa3b568a407d8647128747ea6ae58cd2c83a857b88595351689bd85067bb4.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/af5fa3b568a407d8647128747ea6ae58cd2c83a857b88595351689bd85067bb4.jpg)

![bc38672a67b4b4cc1052bb53b6661734f519ec347a258da8589f7c5bfa556a76.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/bc38672a67b4b4cc1052bb53b6661734f519ec347a258da8589f7c5bfa556a76.jpg)

![bd6faaa8b1cea71cce2b3f7e0e1a1d96047ce296172bf760d7cead3156511362.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/bd6faaa8b1cea71cce2b3f7e0e1a1d96047ce296172bf760d7cead3156511362.jpg)

![cc1cb93ccbc2bfd36948471952df15cd93f5f1de2290b7892177134a3da3cf39.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/cc1cb93ccbc2bfd36948471952df15cd93f5f1de2290b7892177134a3da3cf39.jpg)

![cefb35d9dca9c0bd06e8e452a28729117e85cfd8ea5b119e27768c8c5b1a2280.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/cefb35d9dca9c0bd06e8e452a28729117e85cfd8ea5b119e27768c8c5b1a2280.jpg)

![d07c58c703b8bc565429e88a9bbdcbc8508f9e50a2bd0f9e4ce88d1953388303.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/d07c58c703b8bc565429e88a9bbdcbc8508f9e50a2bd0f9e4ce88d1953388303.jpg)

![d250649a20f7d7bbf509f8e0dd709ff56f0bc2f95fca47fec7d7d2f5a9cdbc9d.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/d250649a20f7d7bbf509f8e0dd709ff56f0bc2f95fca47fec7d7d2f5a9cdbc9d.jpg)

![ddf6c81ad6acdc76c12d9aee213e86364e9d0795c315c3e0af2a3a34da46c3ce.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/ddf6c81ad6acdc76c12d9aee213e86364e9d0795c315c3e0af2a3a34da46c3ce.jpg)

![df405ef18fb77704592c56cb56a4ef625039b633561e640f51fb8176ac94f544.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/df405ef18fb77704592c56cb56a4ef625039b633561e640f51fb8176ac94f544.jpg)

![e1c96b8119fe20817ca7dc6167febf9e11d8e7985bf5c0252590c8d022be886c.jpg](../icml_results/3574_ToaSt_%20Token%20Channel%20Selection%20and%20Structured%20Pruning%20for%20Efficient%20ViT/tables/e1c96b8119fe20817ca7dc6167febf9e11d8e7985bf5c0252590c8d022be886c.jpg)

<a id="paper-3535"></a>

## Understanding Dynamic Compute Allocation in Recurrent Transformers


### Images

![1cbdfbd48f2875fe6104031f0c2fc365145021698f28209b1d3f1f545b60df2c.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/1cbdfbd48f2875fe6104031f0c2fc365145021698f28209b1d3f1f545b60df2c.jpg)

![462564a6a5ff81eda9a45bf95dba7ac9e6e11300f74a9f6907cdfba2e1da3655.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/462564a6a5ff81eda9a45bf95dba7ac9e6e11300f74a9f6907cdfba2e1da3655.jpg)

![51cadf164ed8508ba2b3379f8443642e0f94ed8d647c56e59ae5d9d6d3d34f6c.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/51cadf164ed8508ba2b3379f8443642e0f94ed8d647c56e59ae5d9d6d3d34f6c.jpg)

![52d1c7f3955fe6d4ce3ee6f20637f3c8f171287f50cacb2bb9bb2310f439f4bb.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/52d1c7f3955fe6d4ce3ee6f20637f3c8f171287f50cacb2bb9bb2310f439f4bb.jpg)

![54d622aa595d1a327f4daebab22c46e7271ae4c4eae774c48f8704c4c779d2e4.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/54d622aa595d1a327f4daebab22c46e7271ae4c4eae774c48f8704c4c779d2e4.jpg)

![6e135017286b9c1a50d41cc66dee7bcedbb08db31086375a4ace069adae17a2d.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/6e135017286b9c1a50d41cc66dee7bcedbb08db31086375a4ace069adae17a2d.jpg)

![7b28894ebdb1e2785c7906391497adc20093b39c68c658e4951b08e734393465.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/7b28894ebdb1e2785c7906391497adc20093b39c68c658e4951b08e734393465.jpg)

![7c2df5649a55985edea3ad4cacacfa06e461fd55f2d5c7b9acba6ca430d40de4.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/7c2df5649a55985edea3ad4cacacfa06e461fd55f2d5c7b9acba6ca430d40de4.jpg)

![7c92f3dc7ec124a9eda8479166e8d6426a2199ef0bd0309bfb6446cad48bad06.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/7c92f3dc7ec124a9eda8479166e8d6426a2199ef0bd0309bfb6446cad48bad06.jpg)

![9176af3c2ccd04bbc0769590358834f45371b93edc255c9f0aa29d539d524bd0.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/9176af3c2ccd04bbc0769590358834f45371b93edc255c9f0aa29d539d524bd0.jpg)

![98e2a980759b6f10608f957d8ec85f7b7b857d47d813b51d6aa5eebd7957255a.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/98e2a980759b6f10608f957d8ec85f7b7b857d47d813b51d6aa5eebd7957255a.jpg)

![9d302662a828073c066b878db01bfaa0b391fd9f895f2e1953d873a3663f9ad9.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/9d302662a828073c066b878db01bfaa0b391fd9f895f2e1953d873a3663f9ad9.jpg)

![acefeae624e68215bfd70b76838a970a274870a1f855138ab9ec436d95bdd268.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/acefeae624e68215bfd70b76838a970a274870a1f855138ab9ec436d95bdd268.jpg)

![b67a5b49b5c17f298e8382247591b076ed54c7c2be9f68226d891da9fe2f404d.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/b67a5b49b5c17f298e8382247591b076ed54c7c2be9f68226d891da9fe2f404d.jpg)

![cb0f5c3222c2c0c522ba56637c07a2ab28f50baa8b82550df285c8c10bfc0fde.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/images/cb0f5c3222c2c0c522ba56637c07a2ab28f50baa8b82550df285c8c10bfc0fde.jpg)

### Tables

![3408101b67f7ec6b420ff9cf4772b07759fa050e0c38036c2ad658fcf8c55fee.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/3408101b67f7ec6b420ff9cf4772b07759fa050e0c38036c2ad658fcf8c55fee.jpg)

![43daafc2474ded5df69ab1113a64f5a6033c7e43f9c72d0066997da4889b1e19.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/43daafc2474ded5df69ab1113a64f5a6033c7e43f9c72d0066997da4889b1e19.jpg)

![5e8e638a34c675a264022d945a1a2ef4956e6b11286415eea3ec306b1eef608d.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/5e8e638a34c675a264022d945a1a2ef4956e6b11286415eea3ec306b1eef608d.jpg)

![6747c099b7ca4fbbbe9307bfd5d87d301ce1b2e0df5f49325d89bdab9738d7f0.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/6747c099b7ca4fbbbe9307bfd5d87d301ce1b2e0df5f49325d89bdab9738d7f0.jpg)

![8b1f176a8074eed42319b1fa9c41b4541136274aac763510cbd7ec3b71f122a3.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/8b1f176a8074eed42319b1fa9c41b4541136274aac763510cbd7ec3b71f122a3.jpg)

![e12eb8e53e6a68e12cfac8365c7a76abf44582f2d0664b6b1222e320b9007f54.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/e12eb8e53e6a68e12cfac8365c7a76abf44582f2d0664b6b1222e320b9007f54.jpg)

![e4eee57e5e3ec46332af90ebe9c7056f87deea2942552b6feb616de1a73c6293.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/e4eee57e5e3ec46332af90ebe9c7056f87deea2942552b6feb616de1a73c6293.jpg)

![e8bb5368e1a48158d58fbb4d79816824b35bea3720ae8a6967f11c0a85545d90.jpg](../icml_results/3575_Understanding%20Dynamic%20Compute%20Allocation%20in%20Recurrent%20Transformers/tables/e8bb5368e1a48158d58fbb4d79816824b35bea3720ae8a6967f11c0a85545d90.jpg)

<a id="paper-3536"></a>

## Future Dynamic 3D Reconstruction: Toward 3D World Modeling with Disentangled Ego-Motion


### Images

![516fe4bfb8b681b5cda6e4b24ebe65e31391570fed2b5abd656cf597392cd892.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/images/516fe4bfb8b681b5cda6e4b24ebe65e31391570fed2b5abd656cf597392cd892.jpg)

![95b67b03d14d04f36fa5d86f82ab0192ab4f66a86096261636a045618b304cc2.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/images/95b67b03d14d04f36fa5d86f82ab0192ab4f66a86096261636a045618b304cc2.jpg)

![bffb0ec06f9b8473165a7fee152fee1ab99266701b2be817fad6a7bc18d17092.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/images/bffb0ec06f9b8473165a7fee152fee1ab99266701b2be817fad6a7bc18d17092.jpg)

![d3b83791c5b4bb0a94bce8c62db8885d3ed86b97de66172bc8b4d2bb4bf3dfee.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/images/d3b83791c5b4bb0a94bce8c62db8885d3ed86b97de66172bc8b4d2bb4bf3dfee.jpg)

![e65a6fb0aa4856db367c7a519cc79cf21cb1cdaf763f8ba8c47795465c7ef24e.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/images/e65a6fb0aa4856db367c7a519cc79cf21cb1cdaf763f8ba8c47795465c7ef24e.jpg)

### Tables

![0ae72c0f78c7876cfa0e428ba2ddc900bb7384a5f4c5ce85fcae78c67ef58105.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/0ae72c0f78c7876cfa0e428ba2ddc900bb7384a5f4c5ce85fcae78c67ef58105.jpg)

![0bc0ca36049d0804129de57eea7bc3ea07d99c9b21380a751763092ff6e5f774.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/0bc0ca36049d0804129de57eea7bc3ea07d99c9b21380a751763092ff6e5f774.jpg)

![34a126fd2404da358f8c78c410c5615819b5af9d59a485ac7cce7b572db936b0.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/34a126fd2404da358f8c78c410c5615819b5af9d59a485ac7cce7b572db936b0.jpg)

![60cd6d1751f6cf6895651c34bc370b17fe74e694522c0b35d037fc16a18b758c.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/60cd6d1751f6cf6895651c34bc370b17fe74e694522c0b35d037fc16a18b758c.jpg)

![66ab45801d6694dfd755c6cb4bab5cdf97b0986c1abd558c2e76d9b8b38a65a9.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/66ab45801d6694dfd755c6cb4bab5cdf97b0986c1abd558c2e76d9b8b38a65a9.jpg)

![6ec67049c3598c5b7e11e5f383edfefb65d0187061f9d9fbe0016b7fcd6e6d42.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/6ec67049c3598c5b7e11e5f383edfefb65d0187061f9d9fbe0016b7fcd6e6d42.jpg)

![8be178fdd39cdf4aee5badf625eff5dffc1ae71913c9bfbcf82b7f98dc6d8ef3.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/8be178fdd39cdf4aee5badf625eff5dffc1ae71913c9bfbcf82b7f98dc6d8ef3.jpg)

![8d86c21beaf6d5e7a095a43ebd576ed72fcb537dd1a82a91ad7b1e9f2cf18011.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/8d86c21beaf6d5e7a095a43ebd576ed72fcb537dd1a82a91ad7b1e9f2cf18011.jpg)

![96da2d1a7cacd29f8250622b9f923122aec53993b78e2a6575d7f252b4541569.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/96da2d1a7cacd29f8250622b9f923122aec53993b78e2a6575d7f252b4541569.jpg)

![9f78cfdd4d58771d504e95c256bf135e9251010e28d7fa755461215316b08a52.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/9f78cfdd4d58771d504e95c256bf135e9251010e28d7fa755461215316b08a52.jpg)

![a0e99e3b7aa7d2e9d2860b1f7c959dfc77b391188c8057cf6a2fa02e5dd82064.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/a0e99e3b7aa7d2e9d2860b1f7c959dfc77b391188c8057cf6a2fa02e5dd82064.jpg)

![b0479dd199ab7a0b53d3e28f79afb5b241dcdc98b6af021dc4190239c9621ef3.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/b0479dd199ab7a0b53d3e28f79afb5b241dcdc98b6af021dc4190239c9621ef3.jpg)

![b2a2cb6c1f62445717db1ade6fa1777b180ec59d15e39b1f06238f3d400c5152.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/b2a2cb6c1f62445717db1ade6fa1777b180ec59d15e39b1f06238f3d400c5152.jpg)

![b9520c8ab86e2a9920fc55f94b83575bb66653becea08123ccf587a403b89a4e.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/b9520c8ab86e2a9920fc55f94b83575bb66653becea08123ccf587a403b89a4e.jpg)

![c43b3d5382c404f0f132675b494e6f9496499b150055c62d59341619f5177172.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/c43b3d5382c404f0f132675b494e6f9496499b150055c62d59341619f5177172.jpg)

![c80d6ee84b5ed5f0e94d736603d0912ad0560c6d0b60c15f62509ea9efca6706.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/c80d6ee84b5ed5f0e94d736603d0912ad0560c6d0b60c15f62509ea9efca6706.jpg)

![d6ef0a38cf06cf82f170ef4082f74d01f199c1c667b20c2827b171a2d500c11e.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/d6ef0a38cf06cf82f170ef4082f74d01f199c1c667b20c2827b171a2d500c11e.jpg)

![eb81811d3f7376392d3cb59d99506cdf683d2b555fe0ca379f11b88c089c5a04.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/eb81811d3f7376392d3cb59d99506cdf683d2b555fe0ca379f11b88c089c5a04.jpg)

![fa0d0f0d1fdba368e4067fee65f5e0d4fe94cbf944c4c976845928de6836ad13.jpg](../icml_results/3576_Future%20Dynamic%203D%20Reconstruction_%20Toward%203D%20World%20Modeling%20with%20Disentangled%20Ego-Motion/tables/fa0d0f0d1fdba368e4067fee65f5e0d4fe94cbf944c4c976845928de6836ad13.jpg)

<a id="paper-3537"></a>

## IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension


### Images

![1511b7a84187de3bfb3384974802ebf8526996c998ed34a7acd82bc8a81a878d.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/1511b7a84187de3bfb3384974802ebf8526996c998ed34a7acd82bc8a81a878d.jpg)

![18c62cab93aa7466c913330d5aee9d335ad5722c864e5304b2ec76460d7b8dca.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/18c62cab93aa7466c913330d5aee9d335ad5722c864e5304b2ec76460d7b8dca.jpg)

![22ab301c87a027db086425b923462e14014232698a9f35cd3a40ff599fe01b01.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/22ab301c87a027db086425b923462e14014232698a9f35cd3a40ff599fe01b01.jpg)

![3b1a2fc59515fe6ada8b988a81b53cd8a026ff5f7859fa8dfe27dfdd6651fe1e.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/3b1a2fc59515fe6ada8b988a81b53cd8a026ff5f7859fa8dfe27dfdd6651fe1e.jpg)

![504ad6956030b6e68fee780a57f5f74f968e065ccdc85e79898626e43e04c336.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/504ad6956030b6e68fee780a57f5f74f968e065ccdc85e79898626e43e04c336.jpg)

![589a05fd55d082ed8d579c626ff058d26ebaa999b54c5cab4302b4d7e75433b8.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/589a05fd55d082ed8d579c626ff058d26ebaa999b54c5cab4302b4d7e75433b8.jpg)

![5ee24b6fd2ad0a7a9c1b7968a951951b0bad8b163bb722249d8a5774f7d21e86.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/5ee24b6fd2ad0a7a9c1b7968a951951b0bad8b163bb722249d8a5774f7d21e86.jpg)

![78c6a6c65a372309b02acd9091dd724376400cd3130e659f3967b898e5872b61.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/78c6a6c65a372309b02acd9091dd724376400cd3130e659f3967b898e5872b61.jpg)

![bc37a3f61fd0aa1ab59a3ec34a64b361d43f9926b3a8f3ca13a7a556691a7c4c.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/bc37a3f61fd0aa1ab59a3ec34a64b361d43f9926b3a8f3ca13a7a556691a7c4c.jpg)

![f917d31694157dffda4151e5501051da8379c8e2ecd7d6f0f9793ce149b4e96a.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/images/f917d31694157dffda4151e5501051da8379c8e2ecd7d6f0f9793ce149b4e96a.jpg)

### Tables

![9b2f7dc886912cb219502b4091aa5cd99341b1323366df6bbe7dc7f0bbe5423c.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/tables/9b2f7dc886912cb219502b4091aa5cd99341b1323366df6bbe7dc7f0bbe5423c.jpg)

![aab4c1185abb3845300bfb47adca76aeefdcb2637ef4a10e1075f74595dbbc41.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/tables/aab4c1185abb3845300bfb47adca76aeefdcb2637ef4a10e1075f74595dbbc41.jpg)

![ba94b510955cd6c3a076c4025d140fe8e21fbb4128de2b66acd8f5cca291c9da.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/tables/ba94b510955cd6c3a076c4025d140fe8e21fbb4128de2b66acd8f5cca291c9da.jpg)

![ede4c3f5b871d8e6e2bc4a7bb7d1b12d30263541a206cfc13741db973db560ed.jpg](../icml_results/3577_IdEst_%20Assessing%20Self-Supervised%20Learning%20Representations%20via%20Intrinsic%20Dimension/tables/ede4c3f5b871d8e6e2bc4a7bb7d1b12d30263541a206cfc13741db973db560ed.jpg)

<a id="paper-3538"></a>

## Probing the Geometry of Diffusion Models with the String Method


### Images

![023695cb111a5fdb97d12ed7688dbfdbe2a878179f0429a2c6e958f215862879.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/023695cb111a5fdb97d12ed7688dbfdbe2a878179f0429a2c6e958f215862879.jpg)

![05ef24622cf0ce9c60bf28b20267be655b139b59d8b7e0c5dd846f43d200e612.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/05ef24622cf0ce9c60bf28b20267be655b139b59d8b7e0c5dd846f43d200e612.jpg)

![0883bd5036d2f915e4393c0100bd73d986119ccfeadc64fd48be53b1533e2581.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/0883bd5036d2f915e4393c0100bd73d986119ccfeadc64fd48be53b1533e2581.jpg)

![24a6a034ca8b570a9e5cde1c83f0afb91de5e63e44e73596d616019644e457af.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/24a6a034ca8b570a9e5cde1c83f0afb91de5e63e44e73596d616019644e457af.jpg)

![348055a0681a82b25990a666fbc0855b4801c96e10e5bb5d8d48ff999c7b1a39.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/348055a0681a82b25990a666fbc0855b4801c96e10e5bb5d8d48ff999c7b1a39.jpg)

![46eadfdc28367bcdb808537eaeca5d1bfcdde050bd915375bbd31c446c77d9c7.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/46eadfdc28367bcdb808537eaeca5d1bfcdde050bd915375bbd31c446c77d9c7.jpg)

![4b9dad5256e58c06c3ea6a5e31d5f4b790788ccb45dffbc212018f4d0d246a5b.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/4b9dad5256e58c06c3ea6a5e31d5f4b790788ccb45dffbc212018f4d0d246a5b.jpg)

![542a1c16c76e673fad72690e8950abc23b28c86d6b9569ac1b0928ded88f40de.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/542a1c16c76e673fad72690e8950abc23b28c86d6b9569ac1b0928ded88f40de.jpg)

![567bd9f9f80da27c545cdf7fa798512ef5b12c47423eca7244dce9bdcb21135b.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/567bd9f9f80da27c545cdf7fa798512ef5b12c47423eca7244dce9bdcb21135b.jpg)

![5efcefc69d25c52f7d3ade198f7ae94e80c474047fb7f4a8195700266faca71a.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/5efcefc69d25c52f7d3ade198f7ae94e80c474047fb7f4a8195700266faca71a.jpg)

![5f2d1a86b69004b6a97d9891ad2d085ed855994d47b79cadac28f234b3ad3b7a.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/5f2d1a86b69004b6a97d9891ad2d085ed855994d47b79cadac28f234b3ad3b7a.jpg)

![66a70196f985cd3695cc7d76f6da1d9026b0774b4e1fc1b040152255188a9661.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/66a70196f985cd3695cc7d76f6da1d9026b0774b4e1fc1b040152255188a9661.jpg)

![69d849bda81a1156717660d8b868c5cebda17009e7b3e02e650b26c05e874e42.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/69d849bda81a1156717660d8b868c5cebda17009e7b3e02e650b26c05e874e42.jpg)

![6df7176d93eab0e2e43c11f5284a884a21dfecdc19a0236ae61cdd497f2bad25.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/6df7176d93eab0e2e43c11f5284a884a21dfecdc19a0236ae61cdd497f2bad25.jpg)

![9ee2911833c43f8291bb06b0ad7e8176bd7a48e1526c85cc5cf6778bc70d0065.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/9ee2911833c43f8291bb06b0ad7e8176bd7a48e1526c85cc5cf6778bc70d0065.jpg)

![9fc3fc27469410dd908169b09e29d56e653e096c6555b261e913371f9e78e2bb.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/9fc3fc27469410dd908169b09e29d56e653e096c6555b261e913371f9e78e2bb.jpg)

![a0e412b4a2093dedf2d9f43c8291e5e36ffd853760bcaee5965a2faf20483739.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/a0e412b4a2093dedf2d9f43c8291e5e36ffd853760bcaee5965a2faf20483739.jpg)

![a52e1c626cbc112e01b7f598c676c5710dc386cbeb5df742cdd874f30905bc08.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/a52e1c626cbc112e01b7f598c676c5710dc386cbeb5df742cdd874f30905bc08.jpg)

![a64e75a5ee6caab93d80ba3709cd4d496844698cf0b5ab427b96d04300fd1664.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/a64e75a5ee6caab93d80ba3709cd4d496844698cf0b5ab427b96d04300fd1664.jpg)

![a95d7717f3d067d704979c510c1f50c38551333b1b2c730a6c696abfb2f93a06.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/a95d7717f3d067d704979c510c1f50c38551333b1b2c730a6c696abfb2f93a06.jpg)

![ac26c31cf1fcf214edbd4da32bd5ce4cace7aaefb6812f4947cc610e24aa6f80.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/ac26c31cf1fcf214edbd4da32bd5ce4cace7aaefb6812f4947cc610e24aa6f80.jpg)

![b1c735f0dd54727e93b44b2004d31cf75b4f063efb35a3f045996b22f7823008.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/b1c735f0dd54727e93b44b2004d31cf75b4f063efb35a3f045996b22f7823008.jpg)

![b410be51de84915c28a8810d6c38922c3d415ae8306eaec263be6570386b1f86.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/b410be51de84915c28a8810d6c38922c3d415ae8306eaec263be6570386b1f86.jpg)

![c81abc7b074a2d820383a7c0d935e148f050d66f759fac52e605d6fffdb3530d.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/c81abc7b074a2d820383a7c0d935e148f050d66f759fac52e605d6fffdb3530d.jpg)

![cd6e97293e0c1ab67b0808a06f194776620d40dc0a0750ce6f8adaa60c491e67.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/cd6e97293e0c1ab67b0808a06f194776620d40dc0a0750ce6f8adaa60c491e67.jpg)

![d4bc4ffbb533cba6bdbe01aa5b8cb6cad19c7822b92395f041c566777cead9c9.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/d4bc4ffbb533cba6bdbe01aa5b8cb6cad19c7822b92395f041c566777cead9c9.jpg)

![d4e2d00d27bd133638189f4a79554e1027e8cf554fda4d60261b5665ba9a7ab2.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/d4e2d00d27bd133638189f4a79554e1027e8cf554fda4d60261b5665ba9a7ab2.jpg)

![d62944fe5a75676c840f6db1b65ea290f639a499448d80148224049be42a3d79.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/d62944fe5a75676c840f6db1b65ea290f639a499448d80148224049be42a3d79.jpg)

![db657107e21a3576f57267c5ff1007270c9396e277ff7da44280aee08bfb4a91.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/db657107e21a3576f57267c5ff1007270c9396e277ff7da44280aee08bfb4a91.jpg)

![dd760bec8c6cc1e1ff88d6dd5fcb706102cc0d5dc5446e9cf8ec4a3d5ed08cae.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/dd760bec8c6cc1e1ff88d6dd5fcb706102cc0d5dc5446e9cf8ec4a3d5ed08cae.jpg)

![e7607beddc867c929b8bc745b47c56d37429a6b6ceab4ef28315b369ba475765.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/e7607beddc867c929b8bc745b47c56d37429a6b6ceab4ef28315b369ba475765.jpg)

![e7ebcc48076b291cf250b2fb3977a170df21345c12faa1633e0b6b51ecdcf545.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/e7ebcc48076b291cf250b2fb3977a170df21345c12faa1633e0b6b51ecdcf545.jpg)

![fc9c6fba24313f8a51f49e946a48162d7b30f32668f6439cf870ea221c295fc4.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/fc9c6fba24313f8a51f49e946a48162d7b30f32668f6439cf870ea221c295fc4.jpg)

![ffd776c1296dc16cf5fa1c6f864c0a845f5c2a4ef05091faf121cf20e22e9bc5.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/images/ffd776c1296dc16cf5fa1c6f864c0a845f5c2a4ef05091faf121cf20e22e9bc5.jpg)

### Tables

![0a9fad13f8a5b4fb7583dc68455a829c73f3a3321a270393aeefa1473c3da7c2.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/0a9fad13f8a5b4fb7583dc68455a829c73f3a3321a270393aeefa1473c3da7c2.jpg)

![182bfb4920d8c824f9d368b9cabb9a59696f519878356fb3b4236e8a497faa20.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/182bfb4920d8c824f9d368b9cabb9a59696f519878356fb3b4236e8a497faa20.jpg)

![186a02b70c0b01fbbe431ac1d6e33269c5d760feb52a3fe7298ea593235c5be0.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/186a02b70c0b01fbbe431ac1d6e33269c5d760feb52a3fe7298ea593235c5be0.jpg)

![1a21d4d84a14fa35906fab47b4ed743e9fe5adec39f2a0274b2d058f6010fc06.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/1a21d4d84a14fa35906fab47b4ed743e9fe5adec39f2a0274b2d058f6010fc06.jpg)

![298d0a807685129f73c22e12c4577c48f35dc3bc325aa617f1106811731a6759.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/298d0a807685129f73c22e12c4577c48f35dc3bc325aa617f1106811731a6759.jpg)

![b2144e4664bec40ace2621eb351f86a4ba75e4bdfe6e893faaeef5eeb9ead333.jpg](../icml_results/3578_Probing%20the%20Geometry%20of%20Diffusion%20Models%20with%20the%20String%20Method/tables/b2144e4664bec40ace2621eb351f86a4ba75e4bdfe6e893faaeef5eeb9ead333.jpg)

<a id="paper-3539"></a>

## Spectral Guidance for Flexible and Efficient Control of Diffusion Models


### Images

![0e3795c09ea3219866cf5a913819c29a72c56a55dfaa1fcd698efacc3b5b69e5.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/0e3795c09ea3219866cf5a913819c29a72c56a55dfaa1fcd698efacc3b5b69e5.jpg)

![0e4d1e241e18b11bb35ac63786f1d43e7c53a44f5494f9ade03d357627820183.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/0e4d1e241e18b11bb35ac63786f1d43e7c53a44f5494f9ade03d357627820183.jpg)

![19099d1d6e51c228cb9fb64a035589cc940e6af9b5a10050e6697af3094a2830.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/19099d1d6e51c228cb9fb64a035589cc940e6af9b5a10050e6697af3094a2830.jpg)

![5b347e8c75f743717395e57f2b311bb2c532580e5e9de376586644e3b550fe4d.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/5b347e8c75f743717395e57f2b311bb2c532580e5e9de376586644e3b550fe4d.jpg)

![6bc14ae42bc4be261ff2c67e29518e3b3c9a45925866334a7e28bde141e773ae.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/6bc14ae42bc4be261ff2c67e29518e3b3c9a45925866334a7e28bde141e773ae.jpg)

![7368bd2689a401d8b6303b5d79d20c62c0e158e6170e42887d186c02b0d89d1e.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/7368bd2689a401d8b6303b5d79d20c62c0e158e6170e42887d186c02b0d89d1e.jpg)

![9ec99b2341c7abd0c8a9acf439aba08c43b50748cccfd2ac32c630a4400b9c49.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/9ec99b2341c7abd0c8a9acf439aba08c43b50748cccfd2ac32c630a4400b9c49.jpg)

![af60527688e2f6bf088c58b3f68ef2d0724771c1de12a7e7f1054aa092db9639.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/af60527688e2f6bf088c58b3f68ef2d0724771c1de12a7e7f1054aa092db9639.jpg)

![b40ee5a099cdf61819991e95a6c53287353dda1f44906b78f1add889f1974809.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/b40ee5a099cdf61819991e95a6c53287353dda1f44906b78f1add889f1974809.jpg)

![c5eb22cfb8888628af8cd816e3435969504d723cc30a051c556422ad074a66d6.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/c5eb22cfb8888628af8cd816e3435969504d723cc30a051c556422ad074a66d6.jpg)

![e01063798625e7c1d1e9e7d6688d4e4aff054d874fee810a809dab26729f86f6.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/images/e01063798625e7c1d1e9e7d6688d4e4aff054d874fee810a809dab26729f86f6.jpg)

### Tables

![2265741f3dcf32c19a8ebfc6ff471b8a75b8c7b473d52d78a3db91f1eaf8ce1f.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/tables/2265741f3dcf32c19a8ebfc6ff471b8a75b8c7b473d52d78a3db91f1eaf8ce1f.jpg)

![610482a1a4bac9bd7f829fa18f5de9281ca8c08fd2235c18e9d37cc67ee1a35d.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/tables/610482a1a4bac9bd7f829fa18f5de9281ca8c08fd2235c18e9d37cc67ee1a35d.jpg)

![77b42c648be1576717192125f881dbea06baf7d330fcba9d8049c8b243856568.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/tables/77b42c648be1576717192125f881dbea06baf7d330fcba9d8049c8b243856568.jpg)

![e21c5900a8e30a794e1576dc6ea2cc2aa713b42915552e1f776b51b1584df299.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/tables/e21c5900a8e30a794e1576dc6ea2cc2aa713b42915552e1f776b51b1584df299.jpg)

![f5c5b842b417e7086b24fff611f59c1edcb7589707c850dabc13357a0f54b903.jpg](../icml_results/3579_Spectral%20Guidance%20for%20Flexible%20and%20Efficient%20Control%20of%20Diffusion%20Models/tables/f5c5b842b417e7086b24fff611f59c1edcb7589707c850dabc13357a0f54b903.jpg)

<a id="paper-3540"></a>

## Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation


### Images

![0a36f932b7f085b44654de23c4f76e0b4a427368fd93df1ed51f62aa56e65e7d.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/0a36f932b7f085b44654de23c4f76e0b4a427368fd93df1ed51f62aa56e65e7d.jpg)

![0d0ecd5d3752669d87feda6dccff3948cfa86c3259e3000a3f38ca9c67d49777.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/0d0ecd5d3752669d87feda6dccff3948cfa86c3259e3000a3f38ca9c67d49777.jpg)

![13604a22c77360ceed28f2cb281f711d5ee08f3dc0ef52683cdd189ab4e8ece2.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/13604a22c77360ceed28f2cb281f711d5ee08f3dc0ef52683cdd189ab4e8ece2.jpg)

![13cc5cb23d233479b5f693876243f8888adbf2d2edf80b905d4fedb73c2d9c8c.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/13cc5cb23d233479b5f693876243f8888adbf2d2edf80b905d4fedb73c2d9c8c.jpg)

![19eaaed98d325ab652cc3d844cdacae66efdc1d3b6091546e45abec769561ba2.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/19eaaed98d325ab652cc3d844cdacae66efdc1d3b6091546e45abec769561ba2.jpg)

![1d5ad2b080ef1887f3f363e0af5872202cdc64ff4b0bbbcc1f89263366bf1396.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/1d5ad2b080ef1887f3f363e0af5872202cdc64ff4b0bbbcc1f89263366bf1396.jpg)

![1f76c620bd6c3a53f7cce81e60116aaf9490c9cd1916e32919340ef4da20844f.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/1f76c620bd6c3a53f7cce81e60116aaf9490c9cd1916e32919340ef4da20844f.jpg)

![44a32e3a701e3e7d8979c073c93b314f4ada279668e9e0945b889d13b7105100.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/44a32e3a701e3e7d8979c073c93b314f4ada279668e9e0945b889d13b7105100.jpg)

![4749139371ef57243bcbfcd38c541e1ddef46f6f284a1a7ea9bddc085b11ea1c.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/4749139371ef57243bcbfcd38c541e1ddef46f6f284a1a7ea9bddc085b11ea1c.jpg)

![52a6dbfa1ee94c58f53a6fc6ab1fcf4758facc51142ec13de35505233158b70a.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/52a6dbfa1ee94c58f53a6fc6ab1fcf4758facc51142ec13de35505233158b70a.jpg)

![67e4e9e4dfc9e6a9ed7e3eb32589a311278a6adc34946b0cd48d74671b471ff0.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/67e4e9e4dfc9e6a9ed7e3eb32589a311278a6adc34946b0cd48d74671b471ff0.jpg)

![7830f7fd3d3c57b898b9692db0e4bdc6869ed7385a6910c7258094a4a55c6234.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/7830f7fd3d3c57b898b9692db0e4bdc6869ed7385a6910c7258094a4a55c6234.jpg)

![a41bd87ce9f4dd7866e9ba1b410615112d3c3c9b464914fc06aad2cae83dc308.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/a41bd87ce9f4dd7866e9ba1b410615112d3c3c9b464914fc06aad2cae83dc308.jpg)

![af3366c4b7b156dd4008058bd0f0c7066e304ef733a4f8976b1379f24ec372ce.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/af3366c4b7b156dd4008058bd0f0c7066e304ef733a4f8976b1379f24ec372ce.jpg)

![b657acb2f4a8e6841f33ac9c5cc9363290862814c49fed06f673071427a0e714.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/b657acb2f4a8e6841f33ac9c5cc9363290862814c49fed06f673071427a0e714.jpg)

![b94828b48a46681ad4d5c292f79a873c06e8e401b02b01defca0fca137740a08.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/b94828b48a46681ad4d5c292f79a873c06e8e401b02b01defca0fca137740a08.jpg)

![f7b2a0a3304f81496b5b7979d735743f7060fc0fef71ca787d94d5b6c7443472.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/f7b2a0a3304f81496b5b7979d735743f7060fc0fef71ca787d94d5b6c7443472.jpg)

![fb8aa3f7cebbb97f733456d13badce6f1dfbe1a3793f8924b7f98936130b877d.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/images/fb8aa3f7cebbb97f733456d13badce6f1dfbe1a3793f8924b7f98936130b877d.jpg)

### Tables

![106845f1db555abb5c777af598cc063dc4bb8ca2268c30b5d70c469c8d74df55.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/106845f1db555abb5c777af598cc063dc4bb8ca2268c30b5d70c469c8d74df55.jpg)

![10f2ff94b5c2e22f04084431d07f53341a67e5e16a1ae2428062c049d12feee6.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/10f2ff94b5c2e22f04084431d07f53341a67e5e16a1ae2428062c049d12feee6.jpg)

![31d891cdfb3299a0fcb9f5f0d8e4dc04489dc29bcfbb7858072ca668ce1b026b.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/31d891cdfb3299a0fcb9f5f0d8e4dc04489dc29bcfbb7858072ca668ce1b026b.jpg)

![3fef1e6b477b118ba95cf9edca4370fe0ec9d01f8ca5239413cfdd9036d76231.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/3fef1e6b477b118ba95cf9edca4370fe0ec9d01f8ca5239413cfdd9036d76231.jpg)

![4ec401fc7b7be15a856a56761c0208490fe9a8cbf4c78b6e47e477e45336ac02.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/4ec401fc7b7be15a856a56761c0208490fe9a8cbf4c78b6e47e477e45336ac02.jpg)

![77002edb84b19fb5b16744d3b7e55cc38b076d2eebeaf5ea04474bf358cb623a.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/77002edb84b19fb5b16744d3b7e55cc38b076d2eebeaf5ea04474bf358cb623a.jpg)

![b6750350fbff62ab20f563b5589e26be4b8ff03d81b8d1cc70fdd15f438fc544.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/b6750350fbff62ab20f563b5589e26be4b8ff03d81b8d1cc70fdd15f438fc544.jpg)

![cb6adc36f098988ab0cb4fc456f227390ef8c0d4c7ddac8b7f182f22fb8029e1.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/cb6adc36f098988ab0cb4fc456f227390ef8c0d4c7ddac8b7f182f22fb8029e1.jpg)

![eb095d73559f32cb9f0955d5745305c2e9d4dc3c9ee21894230bc1e5003db201.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/eb095d73559f32cb9f0955d5745305c2e9d4dc3c9ee21894230bc1e5003db201.jpg)

![f46b63bc1d9dbb7baab42f4a61b901a3b1ca6f62ea96e579067ed5c7ed7aa9cf.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/f46b63bc1d9dbb7baab42f4a61b901a3b1ca6f62ea96e579067ed5c7ed7aa9cf.jpg)

![fd92cb67d4b9a24f275af31e7e274c1ec7e6aa40351ea833e398f561d848ac77.jpg](../icml_results/3580_Partial%20Fusion%20of%20Neural%20Networks_%20Efficient%20Tradeoffs%20Between%20Ensembles%20and%20Weight%20Aggregation/tables/fd92cb67d4b9a24f275af31e7e274c1ec7e6aa40351ea833e398f561d848ac77.jpg)

<a id="paper-3541"></a>

## On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning


### Images

![1c336eb9550cfb15dee6674b31d071877a451699bbb817daec5cff6c31826590.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/1c336eb9550cfb15dee6674b31d071877a451699bbb817daec5cff6c31826590.jpg)

![2f411d64f911a323a8acefa07e6af106c85e8c51cb683a84f7d12e5dc0a5f72c.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/2f411d64f911a323a8acefa07e6af106c85e8c51cb683a84f7d12e5dc0a5f72c.jpg)

![3ea0de6ea26247b622679ea3e6b8cbe61625aa10011bdc11d7364be231a7a8cf.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/3ea0de6ea26247b622679ea3e6b8cbe61625aa10011bdc11d7364be231a7a8cf.jpg)

![59cce6b8427034b637ea2b87d1733f32ae182c412190962cec8f354705f1d1be.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/59cce6b8427034b637ea2b87d1733f32ae182c412190962cec8f354705f1d1be.jpg)

![8d878367cf8e01d739194192e689ac5e4ca7915e5fb39d5dd06ef6702fdf310a.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/8d878367cf8e01d739194192e689ac5e4ca7915e5fb39d5dd06ef6702fdf310a.jpg)

![92fb05728c207fa371e6e9eb59c34d93dca36a728a6a130a96e0ade67fe61f9b.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/92fb05728c207fa371e6e9eb59c34d93dca36a728a6a130a96e0ade67fe61f9b.jpg)

![d3db311aa027c07e86471cddb5e58cdac38502781384de5c910d6dbb8de71131.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/d3db311aa027c07e86471cddb5e58cdac38502781384de5c910d6dbb8de71131.jpg)

![e5a3e70d4c544c67a2df480dcf5d6b33aaf25952b494d091f2cf3c88c03ee417.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/images/e5a3e70d4c544c67a2df480dcf5d6b33aaf25952b494d091f2cf3c88c03ee417.jpg)

### Tables

![3dca29ccbbfe1edf07c6b6a855a0f4fb30ab3955daa656dc6bb14ab443806fe2.jpg](../icml_results/3581_On%20the%20Sample%20Efficiency%20of%20Inverse%20Dynamics%20Models%20for%20Semi-Supervised%20Imitation%20Learning/tables/3dca29ccbbfe1edf07c6b6a855a0f4fb30ab3955daa656dc6bb14ab443806fe2.jpg)

<a id="paper-3542"></a>

## Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks


### Images

![2ad294725831dd324a314eb2fa0bd46cdbe0beff183c76eb5ab3709a08b44234.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/2ad294725831dd324a314eb2fa0bd46cdbe0beff183c76eb5ab3709a08b44234.jpg)

![363e542c15aa27ffbb1a882f2e89bc73d7609cd09d449479673c3354321c2434.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/363e542c15aa27ffbb1a882f2e89bc73d7609cd09d449479673c3354321c2434.jpg)

![3f439bb78d82e12ac72acafbf53ebf62fc22d905abe8d9f4da07c3d1e6f95f07.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/3f439bb78d82e12ac72acafbf53ebf62fc22d905abe8d9f4da07c3d1e6f95f07.jpg)

![475334dee6b57ed5f8f2d7196e3af1061fe6ff97292ae56bb2a6633132afc2cf.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/475334dee6b57ed5f8f2d7196e3af1061fe6ff97292ae56bb2a6633132afc2cf.jpg)

![4ea4f50415f2f3259d7f644f06e879a5db0b4dc9e904c815b97b743860606b0f.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/4ea4f50415f2f3259d7f644f06e879a5db0b4dc9e904c815b97b743860606b0f.jpg)

![50d7144a0d5737f84ca277f4fe2460952b7f898cc204bce6d88e899358c4a277.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/50d7144a0d5737f84ca277f4fe2460952b7f898cc204bce6d88e899358c4a277.jpg)

![5add98106051bd463e9858a1e1af784f21e89954fd403fe3ea73ec8a512dc9e2.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/5add98106051bd463e9858a1e1af784f21e89954fd403fe3ea73ec8a512dc9e2.jpg)

![5b4080112bb680933f86d25297166c2ba78f53c7cc18c6aea9c312dbfd5bfa6e.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/5b4080112bb680933f86d25297166c2ba78f53c7cc18c6aea9c312dbfd5bfa6e.jpg)

![7e953ebfd5ab6502a106d49a1a1ea651c2ca319c92204589be39bf78081eb230.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/7e953ebfd5ab6502a106d49a1a1ea651c2ca319c92204589be39bf78081eb230.jpg)

![85f1f5238b578b10850572ee7981e9ebcfdad453c6ae5f77df1afce92635f631.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/85f1f5238b578b10850572ee7981e9ebcfdad453c6ae5f77df1afce92635f631.jpg)

![8c15e2682723ebedd1f89b153b1b24993e3e323b5a976d3bb44a60e51fa99921.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/8c15e2682723ebedd1f89b153b1b24993e3e323b5a976d3bb44a60e51fa99921.jpg)

![b12104821061ce193598737a5fad53e05e67b44aba2e3293c169f691d193a3f1.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/b12104821061ce193598737a5fad53e05e67b44aba2e3293c169f691d193a3f1.jpg)

![b42019cad3ab8955c62b7a2f8b38e8ae6e2431c4163f05efdbd9266de3cf2c2c.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/images/b42019cad3ab8955c62b7a2f8b38e8ae6e2431c4163f05efdbd9266de3cf2c2c.jpg)

### Tables

![1e995c4061c3da325786cc287607ee015c9135408880bad502d824ab687ac821.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/1e995c4061c3da325786cc287607ee015c9135408880bad502d824ab687ac821.jpg)

![28986fde7d2f1f594be83909298cf27bec201dd619e72dcd2a0062b06dd4e33b.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/28986fde7d2f1f594be83909298cf27bec201dd619e72dcd2a0062b06dd4e33b.jpg)

![3fced2d985dba481c25d7a111fb859efa96b3d0159d179f8d4e5dc4b990b911a.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/3fced2d985dba481c25d7a111fb859efa96b3d0159d179f8d4e5dc4b990b911a.jpg)

![8b46ee36e5cb3df1bdff9dd940e982408818cd25867a517e42c9d2d65ac7ae80.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/8b46ee36e5cb3df1bdff9dd940e982408818cd25867a517e42c9d2d65ac7ae80.jpg)

![ce9bc4cfae4de25ef857524b226b6196fcc907c43ad831b1b6a43cd90bb84ce2.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/ce9bc4cfae4de25ef857524b226b6196fcc907c43ad831b1b6a43cd90bb84ce2.jpg)

![ffecb3a483e104cdd1d0818560222307f1c22b770d9dc601f6ffffe73b00951a.jpg](../icml_results/3582_Bullet%20Trains_%20Parallelizing%20Training%20of%20Temporally%20Precise%20Spiking%20Neural%20Networks/tables/ffecb3a483e104cdd1d0818560222307f1c22b770d9dc601f6ffffe73b00951a.jpg)

<a id="paper-3543"></a>

## Compositional Generative Modeling from Decentralized Data


### Images

![03a7c8679c53f176fd27c90ba2132d32ccf237309a9e5cff0ae5f7e8b0415526.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/03a7c8679c53f176fd27c90ba2132d32ccf237309a9e5cff0ae5f7e8b0415526.jpg)

![14f31eb917e6a021338bd91feac9b6306bdc28477b2420cd7ee4b9b04fa49bba.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/14f31eb917e6a021338bd91feac9b6306bdc28477b2420cd7ee4b9b04fa49bba.jpg)

![3b70bbfbb3c30867e70e4d68f246a2af3108fe993bdf6789e288af26c2dbde5c.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/3b70bbfbb3c30867e70e4d68f246a2af3108fe993bdf6789e288af26c2dbde5c.jpg)

![3e349687a8faf72f9f036fc3b91d4639b584d4ab0b175c9fd8602e39eaa35512.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/3e349687a8faf72f9f036fc3b91d4639b584d4ab0b175c9fd8602e39eaa35512.jpg)

![7acea993137bfcb39aa57c8cefdc02445103e3c78e2f043960797fc4f36c2429.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/7acea993137bfcb39aa57c8cefdc02445103e3c78e2f043960797fc4f36c2429.jpg)

![9104cebfd0c83fc7848e99e1d146d66ff180771d1a98040c262cf35e5ac3e97a.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/9104cebfd0c83fc7848e99e1d146d66ff180771d1a98040c262cf35e5ac3e97a.jpg)

![c6b9620958520660c6b175f2ddfc24255eec15f85854a545ec95b4743650e78d.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/c6b9620958520660c6b175f2ddfc24255eec15f85854a545ec95b4743650e78d.jpg)

![d244131f1b94e04f3cc8fc91c2f636ab8f1bc0b5f2ee9cf039175c818fec4b83.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/d244131f1b94e04f3cc8fc91c2f636ab8f1bc0b5f2ee9cf039175c818fec4b83.jpg)

![e1e3a7caa5a8da891e1a8e3b9531d011c5076feb9608d5997a6aee8478e4702d.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/images/e1e3a7caa5a8da891e1a8e3b9531d011c5076feb9608d5997a6aee8478e4702d.jpg)

### Tables

![129699a029bf60a7c347039ee2d9f82c1955729f1b09d0ea6eff86f0a10628ee.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/tables/129699a029bf60a7c347039ee2d9f82c1955729f1b09d0ea6eff86f0a10628ee.jpg)

![376e278341d1bcaaa16601a0bf9f4dbfc047d0952df207a81167e9f10f09eae3.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/tables/376e278341d1bcaaa16601a0bf9f4dbfc047d0952df207a81167e9f10f09eae3.jpg)

![3b59699c5e962912b13742e33bdbbe639fa2fae732c9534a5ccb48d4d2f1fbee.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/tables/3b59699c5e962912b13742e33bdbbe639fa2fae732c9534a5ccb48d4d2f1fbee.jpg)

![466a56ae8ba5fa1d65a9d52156b2d7fff69761a75da9a66efdaa8428dde97c02.jpg](../icml_results/3583_Compositional%20Generative%20Modeling%20from%20Decentralized%20Data/tables/466a56ae8ba5fa1d65a9d52156b2d7fff69761a75da9a66efdaa8428dde97c02.jpg)

<a id="paper-3544"></a>

## Incremental Transformer Neural Processes


### Images

![0514bf9126401bcd1c0b1106c4422447661614408915e121f86c397b4614c9dd.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/0514bf9126401bcd1c0b1106c4422447661614408915e121f86c397b4614c9dd.jpg)

![0a90024cabb76dcb4e080ba39bd833a1c1e62563d33862af9320e9d8699cf8f5.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/0a90024cabb76dcb4e080ba39bd833a1c1e62563d33862af9320e9d8699cf8f5.jpg)

![110b8345ae8cf6c1e03cbe2dfcf2f38c953dc2d7b6eb243d1233c09ee89cc68b.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/110b8345ae8cf6c1e03cbe2dfcf2f38c953dc2d7b6eb243d1233c09ee89cc68b.jpg)

![1e8c592d387f4da941a84b0ffb285888dad93db2bc72e9bdc1fa143a26bb2cc0.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/1e8c592d387f4da941a84b0ffb285888dad93db2bc72e9bdc1fa143a26bb2cc0.jpg)

![311d3d944df9ba60935f3b975dece672deff3a66fe761ee54de815f61ae5853a.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/311d3d944df9ba60935f3b975dece672deff3a66fe761ee54de815f61ae5853a.jpg)

![359950e632f13db283cbc07e1be463f11aeffa01e923414efefbbae36acd8790.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/359950e632f13db283cbc07e1be463f11aeffa01e923414efefbbae36acd8790.jpg)

![36f6f1b0a2fde36e2abd8e70cc2c5a2408328bd933e4f4bc88335c0be17eeeec.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/36f6f1b0a2fde36e2abd8e70cc2c5a2408328bd933e4f4bc88335c0be17eeeec.jpg)

![3bd8b4de723c586b2f420f535680b1711d7b44e011d34a344769b7c4cd938cec.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/3bd8b4de723c586b2f420f535680b1711d7b44e011d34a344769b7c4cd938cec.jpg)

![3e35eaa8a90954f388567fb94261dd4ded7d066b5220c8d4e5f3ad05c9c4bf65.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/3e35eaa8a90954f388567fb94261dd4ded7d066b5220c8d4e5f3ad05c9c4bf65.jpg)

![48c17fe6d6d91c829f1e65330960b8cc82c77cbdcfc1a251165a11fe87821c08.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/48c17fe6d6d91c829f1e65330960b8cc82c77cbdcfc1a251165a11fe87821c08.jpg)

![77730418603563cb354747b58f61bc8beb06d8349239da495bd962dde46f8dfa.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/77730418603563cb354747b58f61bc8beb06d8349239da495bd962dde46f8dfa.jpg)

![7a7feb439344ccec01d3f4e3beb8b54f0d9b62b2872df82e09e9749ac46b1154.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/7a7feb439344ccec01d3f4e3beb8b54f0d9b62b2872df82e09e9749ac46b1154.jpg)

![7cc07c3a6de813552279a68a2ca01bed0a4b68d53308040d7968937324835bb6.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/7cc07c3a6de813552279a68a2ca01bed0a4b68d53308040d7968937324835bb6.jpg)

![7d36a022af96049106e2445989a9300f9a4c2881fbaede8d254307c6c4ef3c4e.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/7d36a022af96049106e2445989a9300f9a4c2881fbaede8d254307c6c4ef3c4e.jpg)

![981115821528149da5c0fb55ca2139cb40426acee521edd00564016772cfdff7.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/981115821528149da5c0fb55ca2139cb40426acee521edd00564016772cfdff7.jpg)

![a8c0437463dcc1e9f13e24961d899413b50f5679df0e019531de384e632ee82c.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/a8c0437463dcc1e9f13e24961d899413b50f5679df0e019531de384e632ee82c.jpg)

![b72cc92ea778fa87d80fadfb4417b9b02e158fdc3dff687b4d04451adc7d523f.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/b72cc92ea778fa87d80fadfb4417b9b02e158fdc3dff687b4d04451adc7d523f.jpg)

![bc3b2c0acc020b43903a2fc51dc79954d84d645ed20af9c884f5f0b13d5922ff.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/bc3b2c0acc020b43903a2fc51dc79954d84d645ed20af9c884f5f0b13d5922ff.jpg)

![be4849a6fa642c1f5a4ef17fdf06f8625469650ff2da25bb6601b737b34f40a5.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/be4849a6fa642c1f5a4ef17fdf06f8625469650ff2da25bb6601b737b34f40a5.jpg)

![be7bcf20cd1b031779c4bb533152c499a7a981aae90f77001db33dadd94ef822.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/be7bcf20cd1b031779c4bb533152c499a7a981aae90f77001db33dadd94ef822.jpg)

![c0e4f2f52bb3f2f8920b79f1af765b114b702af15925d77efe56eab0a249e36f.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/c0e4f2f52bb3f2f8920b79f1af765b114b702af15925d77efe56eab0a249e36f.jpg)

![c4feb1a1da29b17bac26458456e85e0ce0408aae958131be6c28ec84e2d83817.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/c4feb1a1da29b17bac26458456e85e0ce0408aae958131be6c28ec84e2d83817.jpg)

![db46b345466e901060703f8be410aedf8c21fb209015f19d8bee17b881507aeb.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/db46b345466e901060703f8be410aedf8c21fb209015f19d8bee17b881507aeb.jpg)

![db79c66c8d1036343a94edca28b7ed63d974105ef5a18aa9a2dc69755bf0f0bb.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/db79c66c8d1036343a94edca28b7ed63d974105ef5a18aa9a2dc69755bf0f0bb.jpg)

![e5e557d1c8a3d08f223bcde2fdfadb68eadfd65d80f9bd55ade20622b28a32c6.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/e5e557d1c8a3d08f223bcde2fdfadb68eadfd65d80f9bd55ade20622b28a32c6.jpg)

![ed167896e27a7fbf382cd2cc6868e04f5127cce4d9de91a898ba13dc10fa60d8.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/ed167896e27a7fbf382cd2cc6868e04f5127cce4d9de91a898ba13dc10fa60d8.jpg)

![f5742dc7a484cd4edbb1d8912688c9c3331cfb3fb29d0aebb539397cedfeb1eb.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/f5742dc7a484cd4edbb1d8912688c9c3331cfb3fb29d0aebb539397cedfeb1eb.jpg)

![fc7feabce47e119f82b52c3c16a4d7b9f84cb7e40a03488bd0fcfce3c37e37b3.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/images/fc7feabce47e119f82b52c3c16a4d7b9f84cb7e40a03488bd0fcfce3c37e37b3.jpg)

### Tables

![06f2eb77472e714079cc248423c25fa78732b97ebd4e6dc9f863888605daf03b.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/06f2eb77472e714079cc248423c25fa78732b97ebd4e6dc9f863888605daf03b.jpg)

![2c349a5e429f4da3ba0802ef14c3d13737710642fd5d34b7e657ed68ca2922c2.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/2c349a5e429f4da3ba0802ef14c3d13737710642fd5d34b7e657ed68ca2922c2.jpg)

![2e5a082ef11fc5e90ecb747815fc2d5e9567a70aeba72b9f6e1c358740211e72.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/2e5a082ef11fc5e90ecb747815fc2d5e9567a70aeba72b9f6e1c358740211e72.jpg)

![31efd8391764d18e38fb083df2e328213ffc3dd7bb39af1d92ab35d725124fe8.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/31efd8391764d18e38fb083df2e328213ffc3dd7bb39af1d92ab35d725124fe8.jpg)

![5fa30cda8c18a1b6d340607da20d6a2f06729d2ea7a01409d7c0be48e9f4afba.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/5fa30cda8c18a1b6d340607da20d6a2f06729d2ea7a01409d7c0be48e9f4afba.jpg)

![76b35abc8689c685b9f96b2f96ca0e3093217f130f70484ab73c2cef5633a23d.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/76b35abc8689c685b9f96b2f96ca0e3093217f130f70484ab73c2cef5633a23d.jpg)

![7c5b63715d606977af88fa3f2b7be697c3e2442392b2c4e6679291bf96911cca.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/7c5b63715d606977af88fa3f2b7be697c3e2442392b2c4e6679291bf96911cca.jpg)

![b2885d81da6f052452b7f662fae4f09bb1f5b760bdc1d412369d33f6301575f7.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/b2885d81da6f052452b7f662fae4f09bb1f5b760bdc1d412369d33f6301575f7.jpg)

![bece2578c486baacde2b091eece4266a60febe398f89cc4cc620add5cf8b4f4d.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/bece2578c486baacde2b091eece4266a60febe398f89cc4cc620add5cf8b4f4d.jpg)

![c853494eb98eaeed5c0eec54608f7031d775853f3ff5a26e4c3c587233d42b15.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/c853494eb98eaeed5c0eec54608f7031d775853f3ff5a26e4c3c587233d42b15.jpg)

![d01ff700f4d934965322599cd44ea163b5aaa5410cd4fbbd0bdeb330b08e4dc9.jpg](../icml_results/3584_Incremental%20Transformer%20Neural%20Processes/tables/d01ff700f4d934965322599cd44ea163b5aaa5410cd4fbbd0bdeb330b08e4dc9.jpg)

<a id="paper-3545"></a>

## Learning to Evict from Key-Value Cache


### Images

![0ed7824d6aef96cbdd40e314ec813b69c668b55f0df9372b8c3a8d56cec5834b.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/0ed7824d6aef96cbdd40e314ec813b69c668b55f0df9372b8c3a8d56cec5834b.jpg)

![0ee3294471bc2fcb14635f18ec9d3c077902b4e233b2720bf0811717b98a4b1b.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/0ee3294471bc2fcb14635f18ec9d3c077902b4e233b2720bf0811717b98a4b1b.jpg)

![12bc99f03ad5e329f71fc92efd7ec7a88eca15cb595743ff8742e86d76fe6bf7.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/12bc99f03ad5e329f71fc92efd7ec7a88eca15cb595743ff8742e86d76fe6bf7.jpg)

![2166b4b7144e0b5bd7ed6bd9814406b16c5ebc2096f4f0534bcb40a8cf1f35a9.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/2166b4b7144e0b5bd7ed6bd9814406b16c5ebc2096f4f0534bcb40a8cf1f35a9.jpg)

![2fabf02a1fff098418bd4c51b08ccad22b43b91bb9c3d3419166d1c8ad7fc98a.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/2fabf02a1fff098418bd4c51b08ccad22b43b91bb9c3d3419166d1c8ad7fc98a.jpg)

![370fc6698b54eec4dd38c11785b84df2bd9606fe31df45dd334120d1d8e07883.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/370fc6698b54eec4dd38c11785b84df2bd9606fe31df45dd334120d1d8e07883.jpg)

![38aa99d0d0cca588290efba4290a4ef4a65d1c5fa724928614d43fb2770f2b9c.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/38aa99d0d0cca588290efba4290a4ef4a65d1c5fa724928614d43fb2770f2b9c.jpg)

![4332188edef9aa721ce5cee821dc31516fe544fe1291e8e6284aa8786fe4e505.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/4332188edef9aa721ce5cee821dc31516fe544fe1291e8e6284aa8786fe4e505.jpg)

![50a4069813820171de03b74ca51fa1e7f154ca3f1936b7308f3826aa7db96bab.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/50a4069813820171de03b74ca51fa1e7f154ca3f1936b7308f3826aa7db96bab.jpg)

![54c25ccc64b9521f9b8a7d51cf16fbf8790acc6412ff61a4ae4cad6468f47f7c.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/54c25ccc64b9521f9b8a7d51cf16fbf8790acc6412ff61a4ae4cad6468f47f7c.jpg)

![6374d57214b6eb612dcb5008cf680cf23b0482fa1bb68ae9b8e5c377360bafa7.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/6374d57214b6eb612dcb5008cf680cf23b0482fa1bb68ae9b8e5c377360bafa7.jpg)

![6ec6b2b6785868ed7a8064e06946934ab44ac579e01eb7dda1b84fe5eb020b16.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/6ec6b2b6785868ed7a8064e06946934ab44ac579e01eb7dda1b84fe5eb020b16.jpg)

![8b296e9293443eac72baff6e5b1caf051e4b85c1f2264bdc933a1382346e2f30.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/8b296e9293443eac72baff6e5b1caf051e4b85c1f2264bdc933a1382346e2f30.jpg)

![8c8ae5d99716c8a72ce5333390edec953919db0abf073be5452329b01edda553.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/8c8ae5d99716c8a72ce5333390edec953919db0abf073be5452329b01edda553.jpg)

![8d2ef3f40362a5b508d462cf3b80833ee2680a12e7adc38f8d442af407a3624b.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/8d2ef3f40362a5b508d462cf3b80833ee2680a12e7adc38f8d442af407a3624b.jpg)

![bb45d2598a881cac6dd8861e32504b0849a1c59793b5e1dd5a3c0d72094851c1.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/bb45d2598a881cac6dd8861e32504b0849a1c59793b5e1dd5a3c0d72094851c1.jpg)

![c332757fc67707f5b92ca2936511e96a1fe3ae198cdf6b44468eb2c4ff187b1d.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/c332757fc67707f5b92ca2936511e96a1fe3ae198cdf6b44468eb2c4ff187b1d.jpg)

![c45936bcad0a3f7d50e61b454996083382a213c0640d05b526d8759df67d3138.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/c45936bcad0a3f7d50e61b454996083382a213c0640d05b526d8759df67d3138.jpg)

![cc3d474f00becd9d2ac825acbef1ec579c1bcbf7e4c1d4dcfe31145b2ad3f4cd.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/cc3d474f00becd9d2ac825acbef1ec579c1bcbf7e4c1d4dcfe31145b2ad3f4cd.jpg)

![d517525fdd4dd1fca206ec896d611db8b49731bca85c2be6e40f157abe229d93.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/d517525fdd4dd1fca206ec896d611db8b49731bca85c2be6e40f157abe229d93.jpg)

![da5dbf6c737886bc00784b1d4de602c09a92f6289d1103aee41a051effc11148.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/da5dbf6c737886bc00784b1d4de602c09a92f6289d1103aee41a051effc11148.jpg)

![e90910fc7ce9500bee8a38f72f9fd20565093a799bcdfb6328f8323df5cce91a.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/e90910fc7ce9500bee8a38f72f9fd20565093a799bcdfb6328f8323df5cce91a.jpg)

![ea8f9304e360662135bf5df20ed700531f4231ac15ad026dca2be7c4b18c5342.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/ea8f9304e360662135bf5df20ed700531f4231ac15ad026dca2be7c4b18c5342.jpg)

![f45948ae4a9773f437243b16be69dba5079fa433ad19f7e74043af2e91fc3529.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/f45948ae4a9773f437243b16be69dba5079fa433ad19f7e74043af2e91fc3529.jpg)

![f842a4f093e4410c9b7d32a651de2597015c5e87ea11e7b2fab54b8d15fda446.jpg](../icml_results/3585_Learning%20to%20Evict%20from%20Key-Value%20Cache/images/f842a4f093e4410c9b7d32a651de2597015c5e87ea11e7b2fab54b8d15fda446.jpg)

<a id="paper-3546"></a>

## Fair Dataset Distillation via Cross-Group Barycenter Alignment


### Images

![19f48afdfdda40ac15b6a7731ceea40c90160c566fb04ffb232fd87d0fb56cf5.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/19f48afdfdda40ac15b6a7731ceea40c90160c566fb04ffb232fd87d0fb56cf5.jpg)

![3d68391791c87fe22600b910919758362d511d05a5130fc738fa39e0b349d0a6.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/3d68391791c87fe22600b910919758362d511d05a5130fc738fa39e0b349d0a6.jpg)

![60eca46ee612de7565c346e8d6022c7ef817754f7bc11de18afbb7e5b7e5e87a.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/60eca46ee612de7565c346e8d6022c7ef817754f7bc11de18afbb7e5b7e5e87a.jpg)

![77921947f989f65113300ba0be49f84bef3a5b45dcd35d0a5f82b72ecf3b9d00.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/77921947f989f65113300ba0be49f84bef3a5b45dcd35d0a5f82b72ecf3b9d00.jpg)

![92f4ea71e008ca5e585b5722f1c34ec4b1315df976008a1766d94b3b191558ac.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/92f4ea71e008ca5e585b5722f1c34ec4b1315df976008a1766d94b3b191558ac.jpg)

![98dc749bb59fd78f2721f661dc14743acf7a082c0aeff1a4222e73c9f0e71a48.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/98dc749bb59fd78f2721f661dc14743acf7a082c0aeff1a4222e73c9f0e71a48.jpg)

![9e5e614f5c9573d7ba5cfe714306efa7cfae5c4e2965e3d127247b7f6a1f7559.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/9e5e614f5c9573d7ba5cfe714306efa7cfae5c4e2965e3d127247b7f6a1f7559.jpg)

![a07b167adc56b59bca3a94ea944ca1de98185bd9dba248f0287d15aa2c9baceb.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/a07b167adc56b59bca3a94ea944ca1de98185bd9dba248f0287d15aa2c9baceb.jpg)

![a3e29766c66e345d2f7b375e01f9ba43a9f8993f227d710601654310058725ae.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/a3e29766c66e345d2f7b375e01f9ba43a9f8993f227d710601654310058725ae.jpg)

![c4bd7838ea8130122839ffcc1e8389a108f2b389a2835f6fc8201914b7321900.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/c4bd7838ea8130122839ffcc1e8389a108f2b389a2835f6fc8201914b7321900.jpg)

![da64d05bb5bd876b338a908e5c53b2d62775a2ca0c2620638c2244e935eca25b.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/da64d05bb5bd876b338a908e5c53b2d62775a2ca0c2620638c2244e935eca25b.jpg)

![f01b510e4bc8775e5d8f1b154382bac31cacb3894ad8caddb429c34f573c591f.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/images/f01b510e4bc8775e5d8f1b154382bac31cacb3894ad8caddb429c34f573c591f.jpg)

### Tables

![0669127ffcccb5116b52f369da848c6f59242075eb388a00488558be1a502bfe.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/0669127ffcccb5116b52f369da848c6f59242075eb388a00488558be1a502bfe.jpg)

![179cefc1da7a0939219077a2214804348792b79794e9783977658a41425d9217.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/179cefc1da7a0939219077a2214804348792b79794e9783977658a41425d9217.jpg)

![1b68da52495e5922151770c03408f9911e8b7eeeba035d5bed059b39a168061e.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/1b68da52495e5922151770c03408f9911e8b7eeeba035d5bed059b39a168061e.jpg)

![2105b103f57c7004adf0550c9dc6f5830a38027937d6ed3f32c47eaccc50babe.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/2105b103f57c7004adf0550c9dc6f5830a38027937d6ed3f32c47eaccc50babe.jpg)

![2782ebf2d4b2cb1e99086f7b7285b0edc82ed06f0727f7abecab03b189b01ae8.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/2782ebf2d4b2cb1e99086f7b7285b0edc82ed06f0727f7abecab03b189b01ae8.jpg)

![2a45b4f33b5598ad1f5a7ef2abab879353fa1b124c208cbeb1870fa5f97f1622.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/2a45b4f33b5598ad1f5a7ef2abab879353fa1b124c208cbeb1870fa5f97f1622.jpg)

![2f51710a86b3707cf1946b1aa1c8a52b400160c543cb9e57dd0f2506917c79d5.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/2f51710a86b3707cf1946b1aa1c8a52b400160c543cb9e57dd0f2506917c79d5.jpg)

![32169101f05acd77bae4085644194cce41e56b56f033ee8832154a4bf0ac49cb.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/32169101f05acd77bae4085644194cce41e56b56f033ee8832154a4bf0ac49cb.jpg)

![38810e39325fc534943f2a2244ee15a3729f31f3f5c7b7d0dc3bd4a381b3c460.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/38810e39325fc534943f2a2244ee15a3729f31f3f5c7b7d0dc3bd4a381b3c460.jpg)

![4d84e7c660f1f4291bab3b669303274e380583fac8291aef7d6f1b3889ed2664.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/4d84e7c660f1f4291bab3b669303274e380583fac8291aef7d6f1b3889ed2664.jpg)

![507f3b756abe1c495756b30d5ea4f35ab691660f696ec3f1be024632d8cb7ca1.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/507f3b756abe1c495756b30d5ea4f35ab691660f696ec3f1be024632d8cb7ca1.jpg)

![51368c243eb76f00bb41eb0dda3b0056033aa727d7ebde22e34d79ea2633914d.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/51368c243eb76f00bb41eb0dda3b0056033aa727d7ebde22e34d79ea2633914d.jpg)

![5201c71ce6b41e3ce97b81230c34767f0b548e0259dae7d0aec8883d1b478af6.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/5201c71ce6b41e3ce97b81230c34767f0b548e0259dae7d0aec8883d1b478af6.jpg)

![859468459a2211dc0817af10702f6b13e50304e9ff8f106fad70f3c8ed227f5b.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/859468459a2211dc0817af10702f6b13e50304e9ff8f106fad70f3c8ed227f5b.jpg)

![cb64d31e4370c95179245cb0bcf46297cc9e7821eb68bffd1a1fc2fd772a6615.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/cb64d31e4370c95179245cb0bcf46297cc9e7821eb68bffd1a1fc2fd772a6615.jpg)

![cdf9c56f311d3f58dba1b3686d4bf965edaff8a77410e83b9c9606e9d79b95ce.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/cdf9c56f311d3f58dba1b3686d4bf965edaff8a77410e83b9c9606e9d79b95ce.jpg)

![d46c610f871e5b122b31a329d5c3d098e1ec713c8ea5c029d0bf9f6900a1fffb.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/d46c610f871e5b122b31a329d5c3d098e1ec713c8ea5c029d0bf9f6900a1fffb.jpg)

![e85610d18d1c629df39adecad1f041fb2e5bfb8f3d179b46c9233bf952e5262e.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/e85610d18d1c629df39adecad1f041fb2e5bfb8f3d179b46c9233bf952e5262e.jpg)

![efade144f2738f9d53615ea7bbceb9b1d361d2391bc3ab35a7a0bef8ee78f2dd.jpg](../icml_results/3586_Fair%20Dataset%20Distillation%20via%20Cross-Group%20Barycenter%20Alignment/tables/efade144f2738f9d53615ea7bbceb9b1d361d2391bc3ab35a7a0bef8ee78f2dd.jpg)

<a id="paper-3547"></a>

## Causal Detection of Multi-Step LLM Agent Attacks


### Images

![0ca89be9439543adcd0954730ae7de714d78e058bcfd2b5bce73c2d0ecb42a05.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/0ca89be9439543adcd0954730ae7de714d78e058bcfd2b5bce73c2d0ecb42a05.jpg)

![2f3e85cbc008ae8ef90aa4b41419f2a5e6bc9548977b5b10183320dcd8e06fc4.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/2f3e85cbc008ae8ef90aa4b41419f2a5e6bc9548977b5b10183320dcd8e06fc4.jpg)

![4b68909085f32e4c0201b9de7f5e588163264ee748e1bd9d9dafeca1be9d0cf6.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/4b68909085f32e4c0201b9de7f5e588163264ee748e1bd9d9dafeca1be9d0cf6.jpg)

![4fa5b17d381261811a037abc6a6513a6d8943020b08c67e9e755df0cf3bb0689.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/4fa5b17d381261811a037abc6a6513a6d8943020b08c67e9e755df0cf3bb0689.jpg)

![5aa032a0ef7389c7becabda81b620defd13d2bd5c58569c74007f3dd9207bff6.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/5aa032a0ef7389c7becabda81b620defd13d2bd5c58569c74007f3dd9207bff6.jpg)

![627d40d7da73df2feee716b3ec7b8f7cf6818d0cfd2d23c1face7f912f4e1bc5.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/627d40d7da73df2feee716b3ec7b8f7cf6818d0cfd2d23c1face7f912f4e1bc5.jpg)

![80d88139652d8db174ca3e7dc91dc4383b9f05f240364a498f4875c1ed55435b.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/80d88139652d8db174ca3e7dc91dc4383b9f05f240364a498f4875c1ed55435b.jpg)

![b0b72b1873d8d3618d87eaabfbfbec320db2213ca603f1e481ba1c3b2947c48f.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/b0b72b1873d8d3618d87eaabfbfbec320db2213ca603f1e481ba1c3b2947c48f.jpg)

![b62315f5324eeb54fd29a1672eb8aa0965abf335acf825835099c07719a30e49.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/b62315f5324eeb54fd29a1672eb8aa0965abf335acf825835099c07719a30e49.jpg)

![f39def24b8c5b15d8ee7fffec5a5fa92cf63bd82bd5286a0c1bff97e2992411c.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/images/f39def24b8c5b15d8ee7fffec5a5fa92cf63bd82bd5286a0c1bff97e2992411c.jpg)

### Tables

![103f15ce4e9381cf3367a1404d84750d09bbe3edcbf491270017dd2845200ccc.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/103f15ce4e9381cf3367a1404d84750d09bbe3edcbf491270017dd2845200ccc.jpg)

![13273c90a09268ffc650dc272ba5cbae804f92f8f9c57597ebe6929a7e3197f8.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/13273c90a09268ffc650dc272ba5cbae804f92f8f9c57597ebe6929a7e3197f8.jpg)

![1ef3e45d01700cf3f3d3c06ee97dd9d4924712497b1e41105d8389d7fa32b654.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/1ef3e45d01700cf3f3d3c06ee97dd9d4924712497b1e41105d8389d7fa32b654.jpg)

![39f03bbc4e45e4c366433cd9ba4d1bb9c0fb6927ef8c1720d21f3e04df2bfc3e.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/39f03bbc4e45e4c366433cd9ba4d1bb9c0fb6927ef8c1720d21f3e04df2bfc3e.jpg)

![4d2bd3bf358348d5adb96aeb751401a0dd79018f8dc715903cae726f4946bfe9.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/4d2bd3bf358348d5adb96aeb751401a0dd79018f8dc715903cae726f4946bfe9.jpg)

![75d40efb2984294a69184b6c3a36848e8556384ceb9ec8c041f5c8adb6b9ae9e.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/75d40efb2984294a69184b6c3a36848e8556384ceb9ec8c041f5c8adb6b9ae9e.jpg)

![88a66a8311e0a0dfa2697fc357b0ca82e30e36da2f2d2c50d03f882d77240990.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/88a66a8311e0a0dfa2697fc357b0ca82e30e36da2f2d2c50d03f882d77240990.jpg)

![ac91309a092db84d967900a4d2e4727a9a3962577f1728d9f266bd405acfd57e.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/ac91309a092db84d967900a4d2e4727a9a3962577f1728d9f266bd405acfd57e.jpg)

![b9860c66a66ca82ca571a8ed923858fdb5ced147781687b2b4cdfdbb8e357d28.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/b9860c66a66ca82ca571a8ed923858fdb5ced147781687b2b4cdfdbb8e357d28.jpg)

![da926cf895f218de455a47dec148ed3163b2305b3e8eb2daaa97289821975cf0.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/da926cf895f218de455a47dec148ed3163b2305b3e8eb2daaa97289821975cf0.jpg)

![f6e6922f3dd6cf2d0eaec56ca8f43cee48c6379969b03d1f97aeb6a3cd305a60.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/f6e6922f3dd6cf2d0eaec56ca8f43cee48c6379969b03d1f97aeb6a3cd305a60.jpg)

![fdcd6b471352471043da1560bab95a2464c1f89a0d6c64dbbacd98e187a7d2ca.jpg](../icml_results/3587_Causal%20Detection%20of%20Multi-Step%20LLM%20Agent%20Attacks/tables/fdcd6b471352471043da1560bab95a2464c1f89a0d6c64dbbacd98e187a7d2ca.jpg)

<a id="paper-3548"></a>

## LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning


### Images

![014acf9d81dcde28f273ee3c470085103fb46a8fe1b708f9b9f08759e58efb52.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/014acf9d81dcde28f273ee3c470085103fb46a8fe1b708f9b9f08759e58efb52.jpg)

![13382068bcd636f055a4bb2e25a45060534e714800f810857822a0b161c60c93.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/13382068bcd636f055a4bb2e25a45060534e714800f810857822a0b161c60c93.jpg)

![237589a67426ad0367a2ff9beae8f01886be0dceb3ece0ff5dab5c487b7fc56f.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/237589a67426ad0367a2ff9beae8f01886be0dceb3ece0ff5dab5c487b7fc56f.jpg)

![316eab68ee6a594a359c63547f2c128493c90d298cecbda6acb4c29040ab0a41.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/316eab68ee6a594a359c63547f2c128493c90d298cecbda6acb4c29040ab0a41.jpg)

![45d0623624168ffb6546950e4d6184d407b6d986dd8bad1e66a006727fe1628b.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/45d0623624168ffb6546950e4d6184d407b6d986dd8bad1e66a006727fe1628b.jpg)

![4f15d1b37333dde92d16a583fc0349911fc307d0cb86e06268b608be2705a68f.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/4f15d1b37333dde92d16a583fc0349911fc307d0cb86e06268b608be2705a68f.jpg)

![5049f7706a586775168ec61ee27eb5cd3e21435085df7563534ec230c6e49ab4.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/5049f7706a586775168ec61ee27eb5cd3e21435085df7563534ec230c6e49ab4.jpg)

![5aa4caaaad9be203a15f0b3c7dba0e8354070ce205c27a83ecdcd7a454f61566.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/5aa4caaaad9be203a15f0b3c7dba0e8354070ce205c27a83ecdcd7a454f61566.jpg)

![6070bd8e57af3fd2da1556e3f614f28b41dfca03689f2af30982868d042e98e5.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/6070bd8e57af3fd2da1556e3f614f28b41dfca03689f2af30982868d042e98e5.jpg)

![92a080754c5eaae78c639e2bf9b310b0022b538f479da4ce10c3460c30b581f3.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/92a080754c5eaae78c639e2bf9b310b0022b538f479da4ce10c3460c30b581f3.jpg)

![9455439d5453431864288229fcb90f6ebec5540683d032ba3e5e425e9f21cb48.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/9455439d5453431864288229fcb90f6ebec5540683d032ba3e5e425e9f21cb48.jpg)

![9ebaedf7d7443bb2598294e519b3ef1af96adfcf4e12b0ec4249db07394c5898.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/9ebaedf7d7443bb2598294e519b3ef1af96adfcf4e12b0ec4249db07394c5898.jpg)

![a1b351acd5ec3968d8ceb679a271e943e72d053a54396eb051f5ad75489a7e51.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/a1b351acd5ec3968d8ceb679a271e943e72d053a54396eb051f5ad75489a7e51.jpg)

![acf9aebb70cf5a35f508292146c1c66b3cc5259f9df4ecaa00a4396ec2792b37.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/acf9aebb70cf5a35f508292146c1c66b3cc5259f9df4ecaa00a4396ec2792b37.jpg)

![d83a18f35c1e39fa0bf3c851fd03a616e739504fb15ff6b653b0f19faf124622.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/d83a18f35c1e39fa0bf3c851fd03a616e739504fb15ff6b653b0f19faf124622.jpg)

![e0ef47ad5eb2f9c31e6450d1792472d817203e5e38376d7c039b7ffd43cf10e9.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/e0ef47ad5eb2f9c31e6450d1792472d817203e5e38376d7c039b7ffd43cf10e9.jpg)

![f443de63ce4f5d3b22fedd77b02e195dff289c8192e02c5b6dd6365be39ee5e6.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/f443de63ce4f5d3b22fedd77b02e195dff289c8192e02c5b6dd6365be39ee5e6.jpg)

![fc50c35e00ad691a9979acdcc87a25f2b7d038723f52e12d395371b88d1ddd5d.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/fc50c35e00ad691a9979acdcc87a25f2b7d038723f52e12d395371b88d1ddd5d.jpg)

![ffad19663d2b8b7272abb9d3f9b280bb8473618b6a618b147778c23e822aa060.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/images/ffad19663d2b8b7272abb9d3f9b280bb8473618b6a618b147778c23e822aa060.jpg)

### Tables

![059298ff308300abc2511f8f6e6ae93edb73cd6d048a42d73c6d9babde547529.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/059298ff308300abc2511f8f6e6ae93edb73cd6d048a42d73c6d9babde547529.jpg)

![0b7ec2838eafe68b756e2200808d858460fdd66808c385831e18cb85b7c4e038.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/0b7ec2838eafe68b756e2200808d858460fdd66808c385831e18cb85b7c4e038.jpg)

![39c7444ae228fe720d8cfea4da9a159ea720e5ab6d0de3494a1eed836d61a444.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/39c7444ae228fe720d8cfea4da9a159ea720e5ab6d0de3494a1eed836d61a444.jpg)

![57e95b75b68fb3aa63ca31576e2175a471794a4d0768de44f5b3852d113dc33d.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/57e95b75b68fb3aa63ca31576e2175a471794a4d0768de44f5b3852d113dc33d.jpg)

![624d4a99362ed5ec6cb6156738bbb8a7000971efe1b709bac69227bcf4bf2dab.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/624d4a99362ed5ec6cb6156738bbb8a7000971efe1b709bac69227bcf4bf2dab.jpg)

![9010ed0fe6e2ea836ea64ad58e000e005845985d12dee95b4bba398e79b362ef.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/9010ed0fe6e2ea836ea64ad58e000e005845985d12dee95b4bba398e79b362ef.jpg)

![99903a54a0ebb049a5c9c6d2f778d9949b95760e6fc271e9dea076fd01a41538.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/99903a54a0ebb049a5c9c6d2f778d9949b95760e6fc271e9dea076fd01a41538.jpg)

![9eb7ce9cf7e99bf608ecef91f93089fdd8ad4f33e24831eef6348e7ac73dcb7d.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/9eb7ce9cf7e99bf608ecef91f93089fdd8ad4f33e24831eef6348e7ac73dcb7d.jpg)

![b2487dadf301ed20d9fa5d49e91edc919005355b16e3131ebe9ab7bea23052dd.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/b2487dadf301ed20d9fa5d49e91edc919005355b16e3131ebe9ab7bea23052dd.jpg)

![c740710e4eb97847906ba9d7ba5728e84f160fea414986dd49dbba46e63d454d.jpg](../icml_results/3588_LongCoT_%20Benchmarking%20Long-Horizon%20Chain-of-Thought%20Reasoning/tables/c740710e4eb97847906ba9d7ba5728e84f160fea414986dd49dbba46e63d454d.jpg)

<a id="paper-3549"></a>

## SpatialJB: How Text Distribution Art Becomes The "Jailbreak Key" for LLM Guardrails


### Images

![1352f378ed2044023daebcc0c6eb526e0fc697e16f83c076fefb20a2659206f6.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/1352f378ed2044023daebcc0c6eb526e0fc697e16f83c076fefb20a2659206f6.jpg)

![252d5d083554fae0b88e2954dffa04ceb3fe1cd57732fe2a63e15ba83150b986.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/252d5d083554fae0b88e2954dffa04ceb3fe1cd57732fe2a63e15ba83150b986.jpg)

![39d4c24a97fb447ef0860d3f989b5315483c1df1fe3d2308abf436197cd100df.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/39d4c24a97fb447ef0860d3f989b5315483c1df1fe3d2308abf436197cd100df.jpg)

![3d4331cfab7136aff92129bec5b9bbd29178afbd67de4d6d3c4173a22ca1878f.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/3d4331cfab7136aff92129bec5b9bbd29178afbd67de4d6d3c4173a22ca1878f.jpg)

![4e05a10a7e4cc147677a76af308f4b4d0035d7637696f30732282c023dfaff40.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/4e05a10a7e4cc147677a76af308f4b4d0035d7637696f30732282c023dfaff40.jpg)

![53a767f437ed30014a7c62f82c51a26ac6c3414d96c744fbcc02883102478f91.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/53a767f437ed30014a7c62f82c51a26ac6c3414d96c744fbcc02883102478f91.jpg)

![5ab030569852951fb210ab935f8004d5f24fb6fe1842e955fbae29b6eda88e74.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/5ab030569852951fb210ab935f8004d5f24fb6fe1842e955fbae29b6eda88e74.jpg)

![659a2843e864b56a005ee5e2a9371a51ba19bc80e1f0deec91864da0aa590c26.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/659a2843e864b56a005ee5e2a9371a51ba19bc80e1f0deec91864da0aa590c26.jpg)

![9d5918255eb496f06a73fb418c8820854e14d310c4f4f86e4c9c08be1920636c.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/9d5918255eb496f06a73fb418c8820854e14d310c4f4f86e4c9c08be1920636c.jpg)

![9e5fef0637a36d1634f09eb5e91ebe17a108d454b54d137fe35deef765b41b67.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/9e5fef0637a36d1634f09eb5e91ebe17a108d454b54d137fe35deef765b41b67.jpg)

![a899c156e22de10e2ee9bc69d6d783838fb87f2011a8e2cb093a96d63dd59a05.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/a899c156e22de10e2ee9bc69d6d783838fb87f2011a8e2cb093a96d63dd59a05.jpg)

![b3393c51389ecb025aaee89db21dc5c08cb6a8fbd36e63f09b2ba2fc09521c15.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/b3393c51389ecb025aaee89db21dc5c08cb6a8fbd36e63f09b2ba2fc09521c15.jpg)

![c62fd4843b4861025858ee60a32e8f66bf2b0e997caacd0b6423b718a38da63a.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/c62fd4843b4861025858ee60a32e8f66bf2b0e997caacd0b6423b718a38da63a.jpg)

![d35cc38c399b9b6fb717d8e7df2d7721e6509038497d66be541166f394917ed2.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/d35cc38c399b9b6fb717d8e7df2d7721e6509038497d66be541166f394917ed2.jpg)

![d49553aa5349d1744ffc2cb1a88caa3c4297a7a55a46e72c7953eb4ee6fcdbf7.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/d49553aa5349d1744ffc2cb1a88caa3c4297a7a55a46e72c7953eb4ee6fcdbf7.jpg)

![fff4565c10dbfc3590ce85317ad57e4e902a67eb6c5fd896f1654d2f2d998c99.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/images/fff4565c10dbfc3590ce85317ad57e4e902a67eb6c5fd896f1654d2f2d998c99.jpg)

### Tables

![215c0293343ac6c65917f35d3dfa02fe91398aaeb98b34e8f08c7fecc023a647.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/215c0293343ac6c65917f35d3dfa02fe91398aaeb98b34e8f08c7fecc023a647.jpg)

![2ef92b3f172de2253c730459167b956fd8f7718c543e370db4db3ff88900286f.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/2ef92b3f172de2253c730459167b956fd8f7718c543e370db4db3ff88900286f.jpg)

![6fb657eb99600a79966081fcb7e281fd369f182032408fbc2e495d24474d7db1.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/6fb657eb99600a79966081fcb7e281fd369f182032408fbc2e495d24474d7db1.jpg)

![86a795882dadb96445437e5e507cd2b2963177fc8f899b55b9d95413c50e971d.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/86a795882dadb96445437e5e507cd2b2963177fc8f899b55b9d95413c50e971d.jpg)

![ad374148068ce667e73a6b419cce6fc30f034fe18273b3e7c0e9bb4eb6bd7be9.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/ad374148068ce667e73a6b419cce6fc30f034fe18273b3e7c0e9bb4eb6bd7be9.jpg)

![c1f5e78089718c840fd15eba0579349b551adff8b0dac6db9de490ffcfd7317f.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/c1f5e78089718c840fd15eba0579349b551adff8b0dac6db9de490ffcfd7317f.jpg)

![e956a8b9d524e71b9bf82cdab246ab37e8d14d89305b9fcd20b899f3858c830f.jpg](../icml_results/3589_SpatialJB_%20How%20Text%20Distribution%20Art%20Becomes%20The%20_Jailbreak%20Key_%20for%20LLM%20Guardrails/tables/e956a8b9d524e71b9bf82cdab246ab37e8d14d89305b9fcd20b899f3858c830f.jpg)

<a id="paper-3550"></a>

## Bayesian Rain Field Reconstruction using Commercial Microwave Links and Diffusion Model Priors


### Images

![11184440064431d922eed589fc80bac7686dc9792d1f954e2893ae4db0a1cd2e.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/11184440064431d922eed589fc80bac7686dc9792d1f954e2893ae4db0a1cd2e.jpg)

![257350c91f27c63ad305eb106a158b8dd91c33f955c16f3cd4062b72b659f64d.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/257350c91f27c63ad305eb106a158b8dd91c33f955c16f3cd4062b72b659f64d.jpg)

![5d87eb0fe35ccf020f31f70f8b7fda6161972ce5a12f5d30efaff7cf0d35e278.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/5d87eb0fe35ccf020f31f70f8b7fda6161972ce5a12f5d30efaff7cf0d35e278.jpg)

![82621e00403462218f7328071e5d1e1fd5a90fc94c698e7a235471733f09367e.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/82621e00403462218f7328071e5d1e1fd5a90fc94c698e7a235471733f09367e.jpg)

![958061199c483bc6b83bca153687e640fbfe733c7f8879722148381d82c11a55.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/958061199c483bc6b83bca153687e640fbfe733c7f8879722148381d82c11a55.jpg)

![ba1b12df38ea6bed49e79e1fe2e052348e64e39f0ec6d9887ddf43ce8e75109c.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/ba1b12df38ea6bed49e79e1fe2e052348e64e39f0ec6d9887ddf43ce8e75109c.jpg)

![c023c566dbd4bb67166e033b5ee28e7fab98cc528366a56507c54e8678cb577e.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/c023c566dbd4bb67166e033b5ee28e7fab98cc528366a56507c54e8678cb577e.jpg)

![e0069fd1f7e059c9408af979f2fbc991abef188c5e54407066b68cbb93e2d83e.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/e0069fd1f7e059c9408af979f2fbc991abef188c5e54407066b68cbb93e2d83e.jpg)

![fc96d3e53e08f84aecb4a8557a8d87e58d7b47f13e08b5faf8a79ef930cee5f9.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/images/fc96d3e53e08f84aecb4a8557a8d87e58d7b47f13e08b5faf8a79ef930cee5f9.jpg)

### Tables

![2f6b52b821287474a959268c87432ed1ed17364b4531886fa602ada62876430b.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/2f6b52b821287474a959268c87432ed1ed17364b4531886fa602ada62876430b.jpg)

![7a9144c4b410dd6a9c0513d595c68539a877b3bbd5ccc322acdefcbe8fb5ab02.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/7a9144c4b410dd6a9c0513d595c68539a877b3bbd5ccc322acdefcbe8fb5ab02.jpg)

![98bd84a3132da7ce791d7a7515e1acca3cc9ae8d26e4ee0288818516b8c0e1e3.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/98bd84a3132da7ce791d7a7515e1acca3cc9ae8d26e4ee0288818516b8c0e1e3.jpg)

![bcfd8bb37ca1a710c7c19a5e536fa07e5936a62f201d4e9afebefb81c744fb42.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/bcfd8bb37ca1a710c7c19a5e536fa07e5936a62f201d4e9afebefb81c744fb42.jpg)

![c363dffea3a0a75ba5a1940f02a6c4d3ea4aa52bfc5daf8342ec83c456873251.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/c363dffea3a0a75ba5a1940f02a6c4d3ea4aa52bfc5daf8342ec83c456873251.jpg)

![d4f99499a6e512d8b50e8f07de292aff4beb764ad13f52ceee438e036ded0c84.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/d4f99499a6e512d8b50e8f07de292aff4beb764ad13f52ceee438e036ded0c84.jpg)

![d9c459230ac5d4708d540bb08cc8f7047862e0a6246094c8fd1ae8447eec275a.jpg](../icml_results/3590_Bayesian%20Rain%20Field%20Reconstruction%20using%20Commercial%20Microwave%20Links%20and%20Diffusion%20Model%20Priors/tables/d9c459230ac5d4708d540bb08cc8f7047862e0a6246094c8fd1ae8447eec275a.jpg)

<a id="paper-3551"></a>

## Imposing Boundary Conditions on Neural Operators via Learned Function Extensions


### Images

![0922ca8e4fab5f0762eead57b0683569f021b6b4036ab8fc2cc9575b9b4f89c1.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/0922ca8e4fab5f0762eead57b0683569f021b6b4036ab8fc2cc9575b9b4f89c1.jpg)

![0df8fe3acb34123fc61af878054da6f9802a9761c9053bd263d6f0cfa74719f9.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/0df8fe3acb34123fc61af878054da6f9802a9761c9053bd263d6f0cfa74719f9.jpg)

![0f969a54ee95e7313d5af20546b2b6a73563eff533e396caacb3351cf048cdef.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/0f969a54ee95e7313d5af20546b2b6a73563eff533e396caacb3351cf048cdef.jpg)

![123e377960c03bc966753c13c686e67e37bb9e4faa4c90b11a67da963b569efc.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/123e377960c03bc966753c13c686e67e37bb9e4faa4c90b11a67da963b569efc.jpg)

![146fce7596c13ee8cb38e89f359b7afc200e1ba7c144a1c15b87bc52f284daa0.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/146fce7596c13ee8cb38e89f359b7afc200e1ba7c144a1c15b87bc52f284daa0.jpg)

![154788aea5d5d50a512d10d8ce341f7c3dd5cd3a82e994c81da8caa38e2c59ab.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/154788aea5d5d50a512d10d8ce341f7c3dd5cd3a82e994c81da8caa38e2c59ab.jpg)

![20911e275770a44e83b9d06f70898b9ce309fa9a81512a0180df0e8ae8449b12.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/20911e275770a44e83b9d06f70898b9ce309fa9a81512a0180df0e8ae8449b12.jpg)

![28a46bb50ca4bb040df0ce43cca51e0df95e2b15cdaf5b4ec8cb33b787aaf777.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/28a46bb50ca4bb040df0ce43cca51e0df95e2b15cdaf5b4ec8cb33b787aaf777.jpg)

![29a48d704ad898cfe05b33900cca0af5bc4d9c254262dc34468a7f60c758c188.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/29a48d704ad898cfe05b33900cca0af5bc4d9c254262dc34468a7f60c758c188.jpg)

![29ebc9b301658849be87b02a644f0109e04fbaa9b9f539a2f90e36471d72ce31.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/29ebc9b301658849be87b02a644f0109e04fbaa9b9f539a2f90e36471d72ce31.jpg)

![2a7792d752b09a0e5d2b77318e05093a534bf013db074cbd05614fb915818fa8.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/2a7792d752b09a0e5d2b77318e05093a534bf013db074cbd05614fb915818fa8.jpg)

![2f0faacbb40e691dcee20470709dfe233dd4775b75b218a652d4aaf24468e442.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/2f0faacbb40e691dcee20470709dfe233dd4775b75b218a652d4aaf24468e442.jpg)

![3266a87520e108956f22d05858d1b3b8081e7f172bdb534e2ebad298ae233f16.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/3266a87520e108956f22d05858d1b3b8081e7f172bdb534e2ebad298ae233f16.jpg)

![38b6eb32d119207c77d14b94bf88f5d8713940acfcb8658aeef37b52812a16d9.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/38b6eb32d119207c77d14b94bf88f5d8713940acfcb8658aeef37b52812a16d9.jpg)

![3b5f1f8a9e4c3889e63e2a7e0b33861388a14043f9bf1cea021752ac157581e5.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/3b5f1f8a9e4c3889e63e2a7e0b33861388a14043f9bf1cea021752ac157581e5.jpg)

![3dc8975b1b05876b77a95ba8ac79fb45e4b66dce5bc11faebd52738a7c7bf932.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/3dc8975b1b05876b77a95ba8ac79fb45e4b66dce5bc11faebd52738a7c7bf932.jpg)

![3f2a32b8230d445af963c53aa9b1b7827e6933162553ee67c3215bfb34ded6de.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/3f2a32b8230d445af963c53aa9b1b7827e6933162553ee67c3215bfb34ded6de.jpg)

![40aec582d599e2c948c54d23da3eafb9a50bb44deb52c89777a7b03ab42dbd15.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/40aec582d599e2c948c54d23da3eafb9a50bb44deb52c89777a7b03ab42dbd15.jpg)

![4c2d8ce5341eaa9588102720cf7720dc7e36cfcaf6b01bb20b0c4bce6c87c9b7.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/4c2d8ce5341eaa9588102720cf7720dc7e36cfcaf6b01bb20b0c4bce6c87c9b7.jpg)

![4ff5e995cfc8eef702996829ed9cb09a0654650623969407b110f55e7bcb5ef7.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/4ff5e995cfc8eef702996829ed9cb09a0654650623969407b110f55e7bcb5ef7.jpg)

![5282cfb9dcd7c9571d1163e8a6e37574f9c19b14ca511cc368dceee3a2c5c33a.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/5282cfb9dcd7c9571d1163e8a6e37574f9c19b14ca511cc368dceee3a2c5c33a.jpg)

![52c03c6dae1e945ba2de7a72b625f2dbd91902e40bd6910f57301248b2973e33.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/52c03c6dae1e945ba2de7a72b625f2dbd91902e40bd6910f57301248b2973e33.jpg)

![546aa972a212ca146c7e7116f6c2c8d1b674fdc7c4ed805f4b6069a163b56201.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/546aa972a212ca146c7e7116f6c2c8d1b674fdc7c4ed805f4b6069a163b56201.jpg)

![54f2e656118a246dcb1228499bd7fe488a8a0b5c9595d3e400bd9c3445e0a8c4.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/54f2e656118a246dcb1228499bd7fe488a8a0b5c9595d3e400bd9c3445e0a8c4.jpg)

![6101a82bdde23d866667840fa88883f54f3532dc011da490df74ef0ce2753cdc.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/6101a82bdde23d866667840fa88883f54f3532dc011da490df74ef0ce2753cdc.jpg)

![61b3902d14f53ff916888c58609be7ae211f1c0533834da6bbfa6ff815f0125f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/61b3902d14f53ff916888c58609be7ae211f1c0533834da6bbfa6ff815f0125f.jpg)

![65c08f04d62b0dfa4a1a057359990b973a2152e0d439f19f00b9f121d72a8677.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/65c08f04d62b0dfa4a1a057359990b973a2152e0d439f19f00b9f121d72a8677.jpg)

![6ed3a327819cef0c689639616dd94b3b118999a2311f32cd3a5adea87715d3f3.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/6ed3a327819cef0c689639616dd94b3b118999a2311f32cd3a5adea87715d3f3.jpg)

![6fe827824f545b037fcd87c2d2144c7e65aa238bf8db63dd5ac28265fa6481b4.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/6fe827824f545b037fcd87c2d2144c7e65aa238bf8db63dd5ac28265fa6481b4.jpg)

![7309b674f1a8875128613ff603ac07d54ee9b9b695500c4fa6516c6bd0e8aaf8.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7309b674f1a8875128613ff603ac07d54ee9b9b695500c4fa6516c6bd0e8aaf8.jpg)

![783823c7bb49b7b90d18b2292d289f77ced1c8e9cc1a91f4391dbd20319c1639.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/783823c7bb49b7b90d18b2292d289f77ced1c8e9cc1a91f4391dbd20319c1639.jpg)

![78c1877e2bc239586aebe5ccbe9d953c4271b9e9ed2702630ff15d53e6f997b8.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/78c1877e2bc239586aebe5ccbe9d953c4271b9e9ed2702630ff15d53e6f997b8.jpg)

![7992e28f03292cfe38fa0d86057942f636b08e7a9965c3e8c2c179184e37aff3.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7992e28f03292cfe38fa0d86057942f636b08e7a9965c3e8c2c179184e37aff3.jpg)

![7aa2f7dc59a2ae967864e8326dcf0dc714001ab476580980704b82005cbf71a8.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7aa2f7dc59a2ae967864e8326dcf0dc714001ab476580980704b82005cbf71a8.jpg)

![7beb6ea0978444b40bb4c30d859d1d969dd969ff2a7dd9682bb563366d5042a7.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7beb6ea0978444b40bb4c30d859d1d969dd969ff2a7dd9682bb563366d5042a7.jpg)

![7cd059b14edb34f553c8b97e6a4a90118bfcbb856a762417d1f523bd5a9dddc1.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7cd059b14edb34f553c8b97e6a4a90118bfcbb856a762417d1f523bd5a9dddc1.jpg)

![7e9ba95ebeeaa1d8a2895b10144f630cfa7388a58062e85b05071ae9a4b4032f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/7e9ba95ebeeaa1d8a2895b10144f630cfa7388a58062e85b05071ae9a4b4032f.jpg)

![81b059124969c0ee00d40a26cd87a3cb1c0894ccad47d57cfb9d7e77c8cb1186.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/81b059124969c0ee00d40a26cd87a3cb1c0894ccad47d57cfb9d7e77c8cb1186.jpg)

![8577300d69f492f2dd842e2fd1b1939ebc076c02c32a364534c0e06eb9a4f901.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/8577300d69f492f2dd842e2fd1b1939ebc076c02c32a364534c0e06eb9a4f901.jpg)

![85fc8ea7bc3a26b68c0894586690828d4b1a85598406d31780bd588d89671dbc.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/85fc8ea7bc3a26b68c0894586690828d4b1a85598406d31780bd588d89671dbc.jpg)

![89f8ece7745827f4777db050afd90a6f782e201c44b4ddf9b0d71d408d2b3b19.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/89f8ece7745827f4777db050afd90a6f782e201c44b4ddf9b0d71d408d2b3b19.jpg)

![8bbe1cea382265cf9bec32ebbd3274737aec49d0c8a161694884657abc00899a.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/8bbe1cea382265cf9bec32ebbd3274737aec49d0c8a161694884657abc00899a.jpg)

![925e2acc940422a9123163bb74723fd0bf12b073e02933b7a54de8f960645563.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/925e2acc940422a9123163bb74723fd0bf12b073e02933b7a54de8f960645563.jpg)

![9d68d2c09cea2345b0d8577990e9005c922fe22399c85837cacea4a45019314b.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/9d68d2c09cea2345b0d8577990e9005c922fe22399c85837cacea4a45019314b.jpg)

![9da856a6a184f33b9d0749b6d8eb3d5bf45c79e9c9cdb15b979088b8e9ffa26f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/9da856a6a184f33b9d0749b6d8eb3d5bf45c79e9c9cdb15b979088b8e9ffa26f.jpg)

![a110dfe0bdc3caf884f5018130b124279ee692cdc3538b67b9213b57c2cb245f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/a110dfe0bdc3caf884f5018130b124279ee692cdc3538b67b9213b57c2cb245f.jpg)

![a750fd9dca031789e2a6c331e070da3056d6bb224f51d10f88c158a08c867530.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/a750fd9dca031789e2a6c331e070da3056d6bb224f51d10f88c158a08c867530.jpg)

![ad7380f3ef25e6bb2df506d15379148e1cd0bbe8a7b16993f62afc3ba5c8f362.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/ad7380f3ef25e6bb2df506d15379148e1cd0bbe8a7b16993f62afc3ba5c8f362.jpg)

![aee4da5f61d18505722be0ee486fd40b3c7375e3a99f3808387b52855b654ca7.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/aee4da5f61d18505722be0ee486fd40b3c7375e3a99f3808387b52855b654ca7.jpg)

![b2adc5cdd86df76aea8b8140031f11c5e8161b0852dfb07732407bb39b84e49c.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/b2adc5cdd86df76aea8b8140031f11c5e8161b0852dfb07732407bb39b84e49c.jpg)

![b5b969de91f76382b8a9c71eef9ab10cbd33068fd587d9a3582011b4c30de376.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/b5b969de91f76382b8a9c71eef9ab10cbd33068fd587d9a3582011b4c30de376.jpg)

![b96339962321c0eb3d987c895c95d3b80adec42b9efa6095ae7cc4e74435e85e.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/b96339962321c0eb3d987c895c95d3b80adec42b9efa6095ae7cc4e74435e85e.jpg)

![bb833b72d8e01f569f684e21b7d6b46185ac64afc5f8f62cfa6c424e9b9869ef.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/bb833b72d8e01f569f684e21b7d6b46185ac64afc5f8f62cfa6c424e9b9869ef.jpg)

![bca388a66c76101efd8c65102e4a6bbc573d899910e1b464ff701a36c3b17d16.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/bca388a66c76101efd8c65102e4a6bbc573d899910e1b464ff701a36c3b17d16.jpg)

![bfd63f03e34e55a33b501ef5b91492c4bbe3bcd7589a86de731dcfb75c6f9f38.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/bfd63f03e34e55a33b501ef5b91492c4bbe3bcd7589a86de731dcfb75c6f9f38.jpg)

![c5809f9b80db2497e0e34eb3ca7a5c6f4359174eaad65fda8118376304d866b6.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/c5809f9b80db2497e0e34eb3ca7a5c6f4359174eaad65fda8118376304d866b6.jpg)

![c8c5813214d7de855672d650766454673e89c7317d55f593395fecaad0da27b9.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/c8c5813214d7de855672d650766454673e89c7317d55f593395fecaad0da27b9.jpg)

![d5643fbcbf286e0e8e791e529c6e9cb2232ead223bbb4c0340f0a29e5055b4e6.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/d5643fbcbf286e0e8e791e529c6e9cb2232ead223bbb4c0340f0a29e5055b4e6.jpg)

![dc1fa3f01a332778a22c7673285530ed91f4026f2a812da8b1d027a8f54097b0.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/dc1fa3f01a332778a22c7673285530ed91f4026f2a812da8b1d027a8f54097b0.jpg)

![dc69f651a3f30783f47117d4c4d3a72453f42f7211d9dcf5298f2f1a0142cef1.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/dc69f651a3f30783f47117d4c4d3a72453f42f7211d9dcf5298f2f1a0142cef1.jpg)

![deafdd051ec0f0de15dd71cc98870911b5d057bc6d9484a8c301baba20a04b39.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/deafdd051ec0f0de15dd71cc98870911b5d057bc6d9484a8c301baba20a04b39.jpg)

![e6809180827d05cb3bff92730312e0a4e0fb4c7bf2a951c71cc4a0367c3a9dd1.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/e6809180827d05cb3bff92730312e0a4e0fb4c7bf2a951c71cc4a0367c3a9dd1.jpg)

![eded26e2c4dbea72f3dd288cdfc364ac76db592ab9076e4b0443913765a6b3cb.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/eded26e2c4dbea72f3dd288cdfc364ac76db592ab9076e4b0443913765a6b3cb.jpg)

![eebac904f1d70122442d1c9cc7d022c5781fcc3b13d0b4f8346455a349e5f743.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/eebac904f1d70122442d1c9cc7d022c5781fcc3b13d0b4f8346455a349e5f743.jpg)

![f80805a6e981b66af380ae359e26e1c0fa307af657232483f5958ad0dc88e2f3.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/f80805a6e981b66af380ae359e26e1c0fa307af657232483f5958ad0dc88e2f3.jpg)

![f9c20df6adfef35cb4f37c7637f7edab3d5e82b691f0b99a43ec6c5c1b4482fa.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/f9c20df6adfef35cb4f37c7637f7edab3d5e82b691f0b99a43ec6c5c1b4482fa.jpg)

![fdde281b9dae3cc86ec7c56b61cf87f0237172566bf460b04f63e9e1c1c17ec2.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/fdde281b9dae3cc86ec7c56b61cf87f0237172566bf460b04f63e9e1c1c17ec2.jpg)

![fde6944eb5fedcc9e21383c9c1eeaec50951ca602df0d0639811d29bf712d37f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/images/fde6944eb5fedcc9e21383c9c1eeaec50951ca602df0d0639811d29bf712d37f.jpg)

### Tables

![0ea901c3ff406b2ca32e6bf0a6eb920c6ef6648965bededfb6c3708f5b26e73f.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/0ea901c3ff406b2ca32e6bf0a6eb920c6ef6648965bededfb6c3708f5b26e73f.jpg)

![210bc279e0c02b41b2886312b29ed0adfc606f4471062e51c748eaf41abe8a27.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/210bc279e0c02b41b2886312b29ed0adfc606f4471062e51c748eaf41abe8a27.jpg)

![6d0fa63e92298e7566e143857857dabba0390dd2c893e21e5f79cde1e7442390.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/6d0fa63e92298e7566e143857857dabba0390dd2c893e21e5f79cde1e7442390.jpg)

![6f19eb39d1e66df710250b9a8f99e599f3f8c50559f9b2b5b0da182621e90eea.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/6f19eb39d1e66df710250b9a8f99e599f3f8c50559f9b2b5b0da182621e90eea.jpg)

![c0f0d806118d5932763ab7064c1dc711b1fc914230ee4fb4919e10058fe6e171.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/c0f0d806118d5932763ab7064c1dc711b1fc914230ee4fb4919e10058fe6e171.jpg)

![e74baa50a73b25f62d7c5ad577c5b4905d113415256f7b57cce2ced7027e1d81.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/e74baa50a73b25f62d7c5ad577c5b4905d113415256f7b57cce2ced7027e1d81.jpg)

![ecc55eb27a1411a9334a9341b002c90f76e21c8d509526856b96806a16c04736.jpg](../icml_results/3591_Imposing%20Boundary%20Conditions%20on%20Neural%20Operators%20via%20Learned%20Function%20Extensions/tables/ecc55eb27a1411a9334a9341b002c90f76e21c8d509526856b96806a16c04736.jpg)

<a id="paper-3552"></a>

## Spurious Correlation Learning in Preference Optimization: Mechanisms, Consequences, and Mitigation via Tie Training


### Images

![05726e5c056c81ab16cda2f3be4781cdb0df17a78eacc48a2c751180aca445bd.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/05726e5c056c81ab16cda2f3be4781cdb0df17a78eacc48a2c751180aca445bd.jpg)

![1c9d084ea928f6293ee7750ca4b5c6986f2806fd70c8deade09583671e08dee1.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/1c9d084ea928f6293ee7750ca4b5c6986f2806fd70c8deade09583671e08dee1.jpg)

![308dc5e4bf19660bf01779bd7d643cd17bed1aa10928e8dc0dcd8139f7016e6d.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/308dc5e4bf19660bf01779bd7d643cd17bed1aa10928e8dc0dcd8139f7016e6d.jpg)

![3986efd770e6629d1d882ee3f17b4b21e4f1d5319c1c6c606b0e0935422a3c83.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/3986efd770e6629d1d882ee3f17b4b21e4f1d5319c1c6c606b0e0935422a3c83.jpg)

![4aa0de2c7413431a08cffdb2fa0d60f8a2dc290750880430887644255b28034a.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/4aa0de2c7413431a08cffdb2fa0d60f8a2dc290750880430887644255b28034a.jpg)

![4c50f5e35f1f3712f47377a6901bb4d222ea8fc434f0c7cb01f17d2503b121b3.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/4c50f5e35f1f3712f47377a6901bb4d222ea8fc434f0c7cb01f17d2503b121b3.jpg)

![6b900ad8ad58bc27ab8b2ed77e68c0bfcc51daf7393d8179f8d1102747b83adc.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/6b900ad8ad58bc27ab8b2ed77e68c0bfcc51daf7393d8179f8d1102747b83adc.jpg)

![7a2024c00850c2571449a8a19e2da25ba34e2e74a038ce45ad470b30832901cc.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/7a2024c00850c2571449a8a19e2da25ba34e2e74a038ce45ad470b30832901cc.jpg)

![85f931970b064515c7a11ba6e385534c2b7c29d085d0a0f6f8be540a790c0f53.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/85f931970b064515c7a11ba6e385534c2b7c29d085d0a0f6f8be540a790c0f53.jpg)

![9770f678651ec959f7f7e1510f9f15cdb5b5bbfa576e039358ece0c73c488709.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/9770f678651ec959f7f7e1510f9f15cdb5b5bbfa576e039358ece0c73c488709.jpg)

![9918e7159249f58310e6c08237715cc7fca59ecf6966c69ebbe34b491b044eae.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/9918e7159249f58310e6c08237715cc7fca59ecf6966c69ebbe34b491b044eae.jpg)

![a3cb467d08248a9fe459de105cedc918d4a1253c9ccb16956ec573bf40540375.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/a3cb467d08248a9fe459de105cedc918d4a1253c9ccb16956ec573bf40540375.jpg)

![b86fc0bdbf35593351888126a0ab16fe41f1e746f7b228dec90226b8603f9081.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/b86fc0bdbf35593351888126a0ab16fe41f1e746f7b228dec90226b8603f9081.jpg)

![d11d60831e4db39c033f504caf21271a8ab2b6306ccc700739da3745aae4f4b9.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/d11d60831e4db39c033f504caf21271a8ab2b6306ccc700739da3745aae4f4b9.jpg)

![d2377cf16d51e51bf7dbbc0e3f2a21ec2631ecbdc91b18cd62509744b4b833f1.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/d2377cf16d51e51bf7dbbc0e3f2a21ec2631ecbdc91b18cd62509744b4b833f1.jpg)

![d5184e44f0c59363ca1e4a4ce866bfd5df9cd9503dcc5008c356cd7351e62457.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/d5184e44f0c59363ca1e4a4ce866bfd5df9cd9503dcc5008c356cd7351e62457.jpg)

![db79acfe0e6a14162433f0e896b1fae5e53a969a55efaa54f45b16542e3bf9e1.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/db79acfe0e6a14162433f0e896b1fae5e53a969a55efaa54f45b16542e3bf9e1.jpg)

![f379801ec0b7e354513c4fade7f1af66a81adb7b2dab5e1e6fb4c8de72197356.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/images/f379801ec0b7e354513c4fade7f1af66a81adb7b2dab5e1e6fb4c8de72197356.jpg)

### Tables

![1563d648ad45600da720556a09c4b6f3fc92b4dbb8b85103815ec31a05c466b6.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/1563d648ad45600da720556a09c4b6f3fc92b4dbb8b85103815ec31a05c466b6.jpg)

![201f44f369f4600cc0c4520f817d9fd43125f7d8bee3de656a9fbad853ef69d2.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/201f44f369f4600cc0c4520f817d9fd43125f7d8bee3de656a9fbad853ef69d2.jpg)

![2458cc35a93b3ed8338d0617445ef27ea20fd3d61e5c9b0954ccef2283a6b637.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/2458cc35a93b3ed8338d0617445ef27ea20fd3d61e5c9b0954ccef2283a6b637.jpg)

![45f203b4a26fce9deebe81f201959da09a35c53ece72990c59a22d6eca278046.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/45f203b4a26fce9deebe81f201959da09a35c53ece72990c59a22d6eca278046.jpg)

![6fda92fe03394a797815c726ff978806627f22a885b481d45b9e1964e96b2755.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/6fda92fe03394a797815c726ff978806627f22a885b481d45b9e1964e96b2755.jpg)

![a88eb185a1000bbede5e915a4414738d7ba3f7a3e84a2dc1552af20024c64d89.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/a88eb185a1000bbede5e915a4414738d7ba3f7a3e84a2dc1552af20024c64d89.jpg)

![d120cb556746d4b05517fa3111c716d9345cc40b41de2ef16fe0d0001986d866.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/d120cb556746d4b05517fa3111c716d9345cc40b41de2ef16fe0d0001986d866.jpg)

![eb30ad44461c18fd2af0df6c669e633b9d97e74d51ae0bc705922942e59f70d4.jpg](../icml_results/3592_Spurious%20Correlation%20Learning%20in%20Preference%20Optimization_%20Mechanisms%2C%20Consequences%2C%20and%20Mitigation%20v/tables/eb30ad44461c18fd2af0df6c669e633b9d97e74d51ae0bc705922942e59f70d4.jpg)

<a id="paper-3553"></a>

## Calibrated Knowledge Aggregation in Bayesian Mixture-of-Experts for Continual VQA


### Images

![05575c42bd36fc6bd9dd49db8aa386a33a7bf72c852184d1b4e584f6f80ec849.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/05575c42bd36fc6bd9dd49db8aa386a33a7bf72c852184d1b4e584f6f80ec849.jpg)

![1143c766592c8e512654dbb2892f1ed026cb2682ffaa41a28c37b4f5cc4506f1.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/1143c766592c8e512654dbb2892f1ed026cb2682ffaa41a28c37b4f5cc4506f1.jpg)

![49df096b43b55c4392c6abaf11844941097014abb16b7db82ecf7bfd76882272.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/49df096b43b55c4392c6abaf11844941097014abb16b7db82ecf7bfd76882272.jpg)

![5755161d4d8038aeb67a0df17afffb58bc10d60fbc0ed050ebf647096cf1418d.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/5755161d4d8038aeb67a0df17afffb58bc10d60fbc0ed050ebf647096cf1418d.jpg)

![6110c127eee7975d5dbde00138e8801a1761389b0a697cd1af42555fa6e64226.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/6110c127eee7975d5dbde00138e8801a1761389b0a697cd1af42555fa6e64226.jpg)

![6b75497a6020ae7b4872e126fafef9b4f345864c2cb05a50de41d79bcf6ca595.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/6b75497a6020ae7b4872e126fafef9b4f345864c2cb05a50de41d79bcf6ca595.jpg)

![81d421772b12663423ae5b84bf24f657fd1c43bbc645b875cc5c5e45a6e9719d.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/81d421772b12663423ae5b84bf24f657fd1c43bbc645b875cc5c5e45a6e9719d.jpg)

![8a8293e4efeb5e237d848565c496dd82287dfec54c0b683f7431cc89dfdd3d10.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/8a8293e4efeb5e237d848565c496dd82287dfec54c0b683f7431cc89dfdd3d10.jpg)

![a4e9fe7c017af3a8b923c1a2fdefe2baae5e451b0e70137f8a3bae3b6aa815bc.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/a4e9fe7c017af3a8b923c1a2fdefe2baae5e451b0e70137f8a3bae3b6aa815bc.jpg)

![cf02539a5ac0187bea357adccfaee685a785c792bca284528ba1a17a3fd4a413.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/cf02539a5ac0187bea357adccfaee685a785c792bca284528ba1a17a3fd4a413.jpg)

![d0eae3eb163173fceb5edd82ed534f112a7fea256ee9746be507980a544fcf41.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/d0eae3eb163173fceb5edd82ed534f112a7fea256ee9746be507980a544fcf41.jpg)

![fd88d1c02c36a62b15b9766f53b138c93408f7063962f411d452ef81a1d1473c.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/images/fd88d1c02c36a62b15b9766f53b138c93408f7063962f411d452ef81a1d1473c.jpg)

### Tables

![0bf80620e3661b846cd3d26e0f42023f02a66ab2bd37e6ef42a6a013dd352343.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/0bf80620e3661b846cd3d26e0f42023f02a66ab2bd37e6ef42a6a013dd352343.jpg)

![0d37c0ed836fa348cb492e0a6dad6820f15e84c558cec201ecdc4b18bd3c2a7d.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/0d37c0ed836fa348cb492e0a6dad6820f15e84c558cec201ecdc4b18bd3c2a7d.jpg)

![2c7ac7763e588f49ff920be0f8557f34fbba0b0ac94914e9575892219dd4d4d5.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/2c7ac7763e588f49ff920be0f8557f34fbba0b0ac94914e9575892219dd4d4d5.jpg)

![2cfab4f5a790fbb7195ca4e53c5e9c117cbd686320cccc35329cb92fe2250ce2.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/2cfab4f5a790fbb7195ca4e53c5e9c117cbd686320cccc35329cb92fe2250ce2.jpg)

![2f422f34892b3a47e28dec29644fb4fb91f31232e1987fe0cc280dc248291ca1.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/2f422f34892b3a47e28dec29644fb4fb91f31232e1987fe0cc280dc248291ca1.jpg)

![464448295187af4249a311f2be21098fe41ac0b2d3db89740c69eb3d82959178.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/464448295187af4249a311f2be21098fe41ac0b2d3db89740c69eb3d82959178.jpg)

![46e772bfff8cf831d25e55287b8878314729e9b03693e5cec095489d0ab908e3.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/46e772bfff8cf831d25e55287b8878314729e9b03693e5cec095489d0ab908e3.jpg)

![c60dff330d7c33754fc7d5af835b921f42e28cddd118fc929e31315abbe134f5.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/c60dff330d7c33754fc7d5af835b921f42e28cddd118fc929e31315abbe134f5.jpg)

![d001041f6620b5fc0fe783aa2bc71be9d75675487a4dba7a2d67892c167133a0.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/d001041f6620b5fc0fe783aa2bc71be9d75675487a4dba7a2d67892c167133a0.jpg)

![e2154c771804efa61eeb19f82b222407fa24676e84012bb574b130037fc768e7.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/e2154c771804efa61eeb19f82b222407fa24676e84012bb574b130037fc768e7.jpg)

![efa62373707f68826b7b712757de384f64ecb3657c9094a01b3ae1a7af264495.jpg](../icml_results/3593_Calibrated%20Knowledge%20Aggregation%20in%20Bayesian%20Mixture-of-Experts%20for%20Continual%20VQA/tables/efa62373707f68826b7b712757de384f64ecb3657c9094a01b3ae1a7af264495.jpg)

<a id="paper-3554"></a>

## PepCompass: Navigating Peptide Embedding Spaces Using Riemannian Geometry


### Images

![391b91d4001b5c8ce8adae6ffc6a5c84efecbf29e1446354a6fa39d680ee8f23.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/391b91d4001b5c8ce8adae6ffc6a5c84efecbf29e1446354a6fa39d680ee8f23.jpg)

![407df132244b1be8a391f9061d2a6b2a11795a71de3b97f0160aa372a34ac4ff.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/407df132244b1be8a391f9061d2a6b2a11795a71de3b97f0160aa372a34ac4ff.jpg)

![44f22c0f3bf3c7a971ce9e3f9c16403c40a2b1ac95559ed3e9c5f93771ee82df.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/44f22c0f3bf3c7a971ce9e3f9c16403c40a2b1ac95559ed3e9c5f93771ee82df.jpg)

![465e713a35ca13f057eee6d6141dcd8fdd7948b674bab925d0b1856d940be751.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/465e713a35ca13f057eee6d6141dcd8fdd7948b674bab925d0b1856d940be751.jpg)

![50e35e9de42051c2af35ca9748984ab6188bf07c950c266e5737726ee1bffab2.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/50e35e9de42051c2af35ca9748984ab6188bf07c950c266e5737726ee1bffab2.jpg)

![53d7300be653c158deea0c26cc34cc4bc88c0481a8ed75b6411474aab1df9b9f.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/53d7300be653c158deea0c26cc34cc4bc88c0481a8ed75b6411474aab1df9b9f.jpg)

![5ee8b241bd70df44dd2e11e2fb450826ced9762bd4a3cecb171693b31114cb1e.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/5ee8b241bd70df44dd2e11e2fb450826ced9762bd4a3cecb171693b31114cb1e.jpg)

![6cb17415a7c6d50c50b208baa17abcff2b39c82c9a589124387228cdec874fa7.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/6cb17415a7c6d50c50b208baa17abcff2b39c82c9a589124387228cdec874fa7.jpg)

![8e0375fbf1a88758ac35086135afb19f4a89700bbd3c91bd3f34f4f780eca1bb.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/8e0375fbf1a88758ac35086135afb19f4a89700bbd3c91bd3f34f4f780eca1bb.jpg)

![ac44409d4ee98a76f534c1c67a04bb3cd2bf9565feb272c84151e2b9b34d384c.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/ac44409d4ee98a76f534c1c67a04bb3cd2bf9565feb272c84151e2b9b34d384c.jpg)

![bb49dc31a0037f2e61566acedd31be76f5f7c52d0305347c1155c60f5cc336c0.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/bb49dc31a0037f2e61566acedd31be76f5f7c52d0305347c1155c60f5cc336c0.jpg)

![c6d6983baf068dbd3fd222716e56721e8acf95a0288accd2c614f981aff4e5ba.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/c6d6983baf068dbd3fd222716e56721e8acf95a0288accd2c614f981aff4e5ba.jpg)

![cd25f36f7230465f706dc5532290eeff6f5b9733b947394ed80af58dd1d31208.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/cd25f36f7230465f706dc5532290eeff6f5b9733b947394ed80af58dd1d31208.jpg)

![ee257511cd6d45ccce0ce4f0fd0b4b3c4b8951eff1c5801198398844dfb35079.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/images/ee257511cd6d45ccce0ce4f0fd0b4b3c4b8951eff1c5801198398844dfb35079.jpg)

### Tables

![03f3a6215d241e3afa695c77d4af20e9f6d87392526b34816dd7d13906931bee.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/03f3a6215d241e3afa695c77d4af20e9f6d87392526b34816dd7d13906931bee.jpg)

![0d8831486fca46b486a86fb5777fc8bac2c4dec824861eec900e16722ae592a7.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/0d8831486fca46b486a86fb5777fc8bac2c4dec824861eec900e16722ae592a7.jpg)

![20bd8776f88eac396e5e28c89c96a0023249a860dd024d4023bd5d37696bc478.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/20bd8776f88eac396e5e28c89c96a0023249a860dd024d4023bd5d37696bc478.jpg)

![2db44a1f48e72335cf76e48b31d9210e17c6f513f408033086dd0cf21e1f0157.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/2db44a1f48e72335cf76e48b31d9210e17c6f513f408033086dd0cf21e1f0157.jpg)

![3d3c8ac2d5710df1c4b83ef9664e541d07129716356eb1acb897c5c9fdf0d99c.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/3d3c8ac2d5710df1c4b83ef9664e541d07129716356eb1acb897c5c9fdf0d99c.jpg)

![540ffd94af47fe45790e911f1ddb820888bb2dac202c6a2d6a4d7b067a21f360.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/540ffd94af47fe45790e911f1ddb820888bb2dac202c6a2d6a4d7b067a21f360.jpg)

![615ca25890d0c2a1813146be8291d69d76d3edac66e5d42568a778853c7dc0ee.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/615ca25890d0c2a1813146be8291d69d76d3edac66e5d42568a778853c7dc0ee.jpg)

![66b3b29ffe61ede3c2a4f102eee9ec6b13ecbe3cb5549ee312dccf85f53485ec.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/66b3b29ffe61ede3c2a4f102eee9ec6b13ecbe3cb5549ee312dccf85f53485ec.jpg)

![7536988b9c71e51776934ea315304e2112b37cfa68bfd2694951f60083a0a055.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/7536988b9c71e51776934ea315304e2112b37cfa68bfd2694951f60083a0a055.jpg)

![8081a391195e34ae681eaabf27b59b1918f273999a08febb1f630111eaab30f2.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/8081a391195e34ae681eaabf27b59b1918f273999a08febb1f630111eaab30f2.jpg)

![90df0ebecbbd618b5f4c92bd8c46db7b1f8f308744e69d05ee5437c630f5fb5f.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/90df0ebecbbd618b5f4c92bd8c46db7b1f8f308744e69d05ee5437c630f5fb5f.jpg)

![91b193289f8666eced213302e3c287f1a813f14ce35a11f2808f28821c6edf59.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/91b193289f8666eced213302e3c287f1a813f14ce35a11f2808f28821c6edf59.jpg)

![ad93e0caf0f82e341271af8fc3ecd28c65d53a475826c67be32fdbacdf6a4650.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/ad93e0caf0f82e341271af8fc3ecd28c65d53a475826c67be32fdbacdf6a4650.jpg)

![c4e4adc5a8db224398fa4523f499b4a9de1d1375afa944e0df9bfd2b6e1c5143.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/c4e4adc5a8db224398fa4523f499b4a9de1d1375afa944e0df9bfd2b6e1c5143.jpg)

![f610c9dfe0f660769b577afd9402687ed62fb2fc2048562cf8386dbab5b1a9d9.jpg](../icml_results/3594_PepCompass_%20Navigating%20Peptide%20Embedding%20Spaces%20Using%20Riemannian%20Geometry/tables/f610c9dfe0f660769b577afd9402687ed62fb2fc2048562cf8386dbab5b1a9d9.jpg)

<a id="paper-3555"></a>

## Reinforcement Learning with Verifiable Rewards: GRPO’s Loss, Dynamics, and Success Amplification


### Images

![324f4c5795b72de6db01607c81e8670b8e057c646dbdcf0c0d8197de073b583f.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/324f4c5795b72de6db01607c81e8670b8e057c646dbdcf0c0d8197de073b583f.jpg)

![45d3be73659fa1106e3fe9ce8f7dff820be6d4abaaf7c41eb2b9479216ae3dd4.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/45d3be73659fa1106e3fe9ce8f7dff820be6d4abaaf7c41eb2b9479216ae3dd4.jpg)

![9975b547d556a75ad4f9c39836c783d97027edb3372a511db80c8cce17b8c21b.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/9975b547d556a75ad4f9c39836c783d97027edb3372a511db80c8cce17b8c21b.jpg)

![9f602c2f32cb19593d83901f9a0175be2ef4b2213038ebcff19334f197f5ca27.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/9f602c2f32cb19593d83901f9a0175be2ef4b2213038ebcff19334f197f5ca27.jpg)

![b3847e71aecc884abf9fd32aaae3351110e6e1c7acabb4336c3d6e2b1370e9af.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/b3847e71aecc884abf9fd32aaae3351110e6e1c7acabb4336c3d6e2b1370e9af.jpg)

![bd6ee345dfd3cc41153ce8415a1ce372e85df973a130936ce84468b7c37e21b7.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/bd6ee345dfd3cc41153ce8415a1ce372e85df973a130936ce84468b7c37e21b7.jpg)

![d2b06c9ed55d96b909e1cd8fc31ed7a076dec85c3baddb60ccc83c62d90641c3.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/d2b06c9ed55d96b909e1cd8fc31ed7a076dec85c3baddb60ccc83c62d90641c3.jpg)

![fda6bbc5a3098694adfdef67122831c788dc4d32c1366cce60bf2a2522fbefa9.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/fda6bbc5a3098694adfdef67122831c788dc4d32c1366cce60bf2a2522fbefa9.jpg)

![fde77592380b3e1033ed2c7d1d81dbf2839781c01422310e9e9efcd0ce2a8394.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/images/fde77592380b3e1033ed2c7d1d81dbf2839781c01422310e9e9efcd0ce2a8394.jpg)

### Tables

![05b3a742bfa4929e67692ddb1e74bcaeb6da3a3828917e931816e770c001d594.jpg](../icml_results/3595_Reinforcement%20Learning%20with%20Verifiable%20Rewards_%20GRPO%E2%80%99s%20Loss%2C%20Dynamics%2C%20and%20Success%20Amplification/tables/05b3a742bfa4929e67692ddb1e74bcaeb6da3a3828917e931816e770c001d594.jpg)

<a id="paper-3556"></a>

## DRIFT: Decoupled Rollouts and Importance-Weighted Fine-Tuning for Efficient Multi-Turn Optimization


### Images

![2acceb54eef23e96b92e9b7c70f4885c69cdd9f89a13633088e923673493e9e4.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/2acceb54eef23e96b92e9b7c70f4885c69cdd9f89a13633088e923673493e9e4.jpg)

![517a49b104507178db99fd6083c165abd190c44f3bb90c40921815f3f645a4ed.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/517a49b104507178db99fd6083c165abd190c44f3bb90c40921815f3f645a4ed.jpg)

![614c67edad49e8d5727f525f5b9fcac00c3b7eeebdaf31f4cff57f877b2c0852.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/614c67edad49e8d5727f525f5b9fcac00c3b7eeebdaf31f4cff57f877b2c0852.jpg)

![7b59fe43e3f7509d58979b65c31edf26e70adc689af760381a2c9f4651a84f7a.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/7b59fe43e3f7509d58979b65c31edf26e70adc689af760381a2c9f4651a84f7a.jpg)

![ab1e84283eb566e5e5b674a893f63ad4999674eaab5b93e08083a4e62ba96882.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/ab1e84283eb566e5e5b674a893f63ad4999674eaab5b93e08083a4e62ba96882.jpg)

![cff66c3b529454ddc8d02ffe67dd556b396bd3c2feb37a228b71e889a26b6e8d.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/cff66c3b529454ddc8d02ffe67dd556b396bd3c2feb37a228b71e889a26b6e8d.jpg)

![d6fe3d1ec901af9f397147711f2c9283be4ffbe4c24ffa15510f47503d78d170.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/d6fe3d1ec901af9f397147711f2c9283be4ffbe4c24ffa15510f47503d78d170.jpg)

![ea59da432ae699202dc8eb91ec1e0aec2b98af59350c1dc2d2121f411639f8d7.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/ea59da432ae699202dc8eb91ec1e0aec2b98af59350c1dc2d2121f411639f8d7.jpg)

![ec16efc220da6faf8a9366901ceb401c9930f5ce09e39d0208f82c8425538ae5.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/images/ec16efc220da6faf8a9366901ceb401c9930f5ce09e39d0208f82c8425538ae5.jpg)

### Tables

![03031be52fb76062d4c4c3a0fcf4931fe7ebbfbc8a88768c8155ec71bb2509e6.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/03031be52fb76062d4c4c3a0fcf4931fe7ebbfbc8a88768c8155ec71bb2509e6.jpg)

![091a373dc593382513e80cadafc429b834a0587ac8fe805ec627150705d4e084.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/091a373dc593382513e80cadafc429b834a0587ac8fe805ec627150705d4e084.jpg)

![0ece98c13992c909d6ea49477d797d82a8c252d4ff12fd3f42207452538b3439.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/0ece98c13992c909d6ea49477d797d82a8c252d4ff12fd3f42207452538b3439.jpg)

![1c59089d55e683b0b9f9b9df29a3a66aa924508135d2abc3206637259efcbd2e.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/1c59089d55e683b0b9f9b9df29a3a66aa924508135d2abc3206637259efcbd2e.jpg)

![210a65c82ca290f0b3e44bcd4c372fbee5aeb61931f009e3810e6c14a0d1cf88.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/210a65c82ca290f0b3e44bcd4c372fbee5aeb61931f009e3810e6c14a0d1cf88.jpg)

![30cbf9b66f03f1b9346328c22368d3838b264ac3c4380079dc9f42ceb13f835c.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/30cbf9b66f03f1b9346328c22368d3838b264ac3c4380079dc9f42ceb13f835c.jpg)

![3d7b1cfaeefd6dc68d230d88aee943638f4950a46f93e70e67021248e6cdf3c1.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/3d7b1cfaeefd6dc68d230d88aee943638f4950a46f93e70e67021248e6cdf3c1.jpg)

![3ffe0304b2cc8cbcbc721c32aa8bedbb1c0c338baed045b3bf2aac3a03c28df7.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/3ffe0304b2cc8cbcbc721c32aa8bedbb1c0c338baed045b3bf2aac3a03c28df7.jpg)

![a02fc33cd6c9219f1c7cdb06139e37a46e1c96df0ae627a3aa2af914a4eadfab.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/a02fc33cd6c9219f1c7cdb06139e37a46e1c96df0ae627a3aa2af914a4eadfab.jpg)

![b37a05cbc0e1c53b145c58c9816d9e891b71e128643e2562fe7649d19eac702b.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/b37a05cbc0e1c53b145c58c9816d9e891b71e128643e2562fe7649d19eac702b.jpg)

![ceee058c2338f5d6564c14eb08c1956f6f0256029502507d8f1a238e3f8ce069.jpg](../icml_results/3596_DRIFT_%20Decoupled%20Rollouts%20and%20Importance-Weighted%20Fine-Tuning%20for%20Efficient%20Multi-Turn%20Optimization/tables/ceee058c2338f5d6564c14eb08c1956f6f0256029502507d8f1a238e3f8ce069.jpg)

<a id="paper-3557"></a>

## Federated Manifold Learning (FML): Tackling Domain Heterogeneity with Structural Knowledge Transfer


### Images

![5a48eeed43dee4bd04764c9f2c5946811bffa6372e3670504b52089ec2e70bd9.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/images/5a48eeed43dee4bd04764c9f2c5946811bffa6372e3670504b52089ec2e70bd9.jpg)

![8f8d324eafc7ca3841093e87c751f66bea547e94043762bbce08f6866e33ccfa.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/images/8f8d324eafc7ca3841093e87c751f66bea547e94043762bbce08f6866e33ccfa.jpg)

![a4da0245100fb8ec0816b67055716557b37662d4b962f5ec35ff36df8dd76e46.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/images/a4da0245100fb8ec0816b67055716557b37662d4b962f5ec35ff36df8dd76e46.jpg)

![c869c59e7e453bb008b04d3413f77200931f379b5f81b8c0a12749be32676dc4.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/images/c869c59e7e453bb008b04d3413f77200931f379b5f81b8c0a12749be32676dc4.jpg)

![d95a879e68bcfb6ce3eb8449067bd2de1a9f79a38637b3c29df9aa923448cd6e.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/images/d95a879e68bcfb6ce3eb8449067bd2de1a9f79a38637b3c29df9aa923448cd6e.jpg)

### Tables

![06337ccac444e7fb43268fbfc47888eb457e6ef536ea24ec80bd2565c7e54697.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/tables/06337ccac444e7fb43268fbfc47888eb457e6ef536ea24ec80bd2565c7e54697.jpg)

![2f4bd51e02555e2f33ac59f572bfa1609c34265eb9b09b47ea759d94ddca1b34.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/tables/2f4bd51e02555e2f33ac59f572bfa1609c34265eb9b09b47ea759d94ddca1b34.jpg)

![506bf573514500e9290fd58c7f7aedf848678368c2674cd90c81f4f92734b0b0.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/tables/506bf573514500e9290fd58c7f7aedf848678368c2674cd90c81f4f92734b0b0.jpg)

![c770bc26ab9d489c89fbfbb5ca22a7276101841c90dd68ea5fae58bc4367992a.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/tables/c770bc26ab9d489c89fbfbb5ca22a7276101841c90dd68ea5fae58bc4367992a.jpg)

![f859aee90b6db884febfb6abefe684fe126d5bfc1b3709ee130e2449b9b5ed94.jpg](../icml_results/3597_Federated%20Manifold%20Learning%20%28FML%29_%20Tackling%20Domain%20Heterogeneity%20with%20Structural%20Knowledge%20Transfer/tables/f859aee90b6db884febfb6abefe684fe126d5bfc1b3709ee130e2449b9b5ed94.jpg)

<a id="paper-3558"></a>

## Multi-Integration of Labels Across Categories for Component Identification in Multi-trial Time Series


### Images

![023fbf480d5343d6b125fa211bb20a037a4ee68211b0231f3700b71eac0c3108.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/023fbf480d5343d6b125fa211bb20a037a4ee68211b0231f3700b71eac0c3108.jpg)

![063c25510fe3a8765c7fccdb51224d26bbc36f7799af3284c36d52edb3aeb66d.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/063c25510fe3a8765c7fccdb51224d26bbc36f7799af3284c36d52edb3aeb66d.jpg)

![0e5296101d9fdd1474006a612641c2fa6b04fa096a0b82334c5977b4ff66f863.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/0e5296101d9fdd1474006a612641c2fa6b04fa096a0b82334c5977b4ff66f863.jpg)

![10340ca17052d1ab5ee74d141bced912c63d2b1ee962d01e8ca45ecedefdda73.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/10340ca17052d1ab5ee74d141bced912c63d2b1ee962d01e8ca45ecedefdda73.jpg)

![111673a4a5821e6b5aaafc524e5b52e87651671a30ca713fd6e841cf3ca77270.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/111673a4a5821e6b5aaafc524e5b52e87651671a30ca713fd6e841cf3ca77270.jpg)

![11184350c29d8ed7c2b7c897f4b51c7bd35cc99a3ec8567ee13f600d5e5a1216.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/11184350c29d8ed7c2b7c897f4b51c7bd35cc99a3ec8567ee13f600d5e5a1216.jpg)

![15f52d12ad99df492a9f440f0f934ba9760528e6184f055f605dec2b36c9543d.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/15f52d12ad99df492a9f440f0f934ba9760528e6184f055f605dec2b36c9543d.jpg)

![19ad98161d8059063e2cdce28cded8cbe348799b4bc17fd287e3d33f96e369d2.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/19ad98161d8059063e2cdce28cded8cbe348799b4bc17fd287e3d33f96e369d2.jpg)

![2c4a8539e8012cb02d2e0180604d7767adf79188d6138c436eaacfe89fc1b643.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/2c4a8539e8012cb02d2e0180604d7767adf79188d6138c436eaacfe89fc1b643.jpg)

![3dde8d9fd62f656409b443b460d5498c019d0bbe63b18daf0f182365268d948a.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/3dde8d9fd62f656409b443b460d5498c019d0bbe63b18daf0f182365268d948a.jpg)

![3eec76c3226112a9da12a84bff12e33e0dbf2e82a9385349f1b3208ab22872b9.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/3eec76c3226112a9da12a84bff12e33e0dbf2e82a9385349f1b3208ab22872b9.jpg)

![48fb2e38a0e4213d1364ecce87bafcc6cd4ee454ec8010080db7fe2a751fd2d4.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/48fb2e38a0e4213d1364ecce87bafcc6cd4ee454ec8010080db7fe2a751fd2d4.jpg)

![495fc01d191977e890ddc4a1df6aac6499fe4d10d53c2ec67fe70319be1f087d.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/495fc01d191977e890ddc4a1df6aac6499fe4d10d53c2ec67fe70319be1f087d.jpg)

![4b0183b91c5736d3be55ed09362654546cc7826c26ba2bb2e9ce8b7c6f281bfb.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/4b0183b91c5736d3be55ed09362654546cc7826c26ba2bb2e9ce8b7c6f281bfb.jpg)

![4df240c241ecf2c0a3cbc6e3ac592eb25bc3be0ad28f10233bd5fe29071cd398.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/4df240c241ecf2c0a3cbc6e3ac592eb25bc3be0ad28f10233bd5fe29071cd398.jpg)

![56b4ac27d7847338a0c51a47a680529bec43e6b364ae08c9ebe6038d7f7e272b.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/56b4ac27d7847338a0c51a47a680529bec43e6b364ae08c9ebe6038d7f7e272b.jpg)

![612e2e0e9707af34fad5bd58f2ee07d920e081e62b17205bc5b288cb880f303c.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/612e2e0e9707af34fad5bd58f2ee07d920e081e62b17205bc5b288cb880f303c.jpg)

![6137b02564a57ede34c9b935c6191249b77a7be6422829492b207af08d1be1d3.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/6137b02564a57ede34c9b935c6191249b77a7be6422829492b207af08d1be1d3.jpg)

![66e5415c0635905ee83e9f413a0ae1424cd7b9010b4477fed0e80f68b0b27899.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/66e5415c0635905ee83e9f413a0ae1424cd7b9010b4477fed0e80f68b0b27899.jpg)

![6f5d4a4a09b4933e75df689f194daac3eb551b580646996a644a8a71f3a2104f.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/6f5d4a4a09b4933e75df689f194daac3eb551b580646996a644a8a71f3a2104f.jpg)

![7328b7b30b86696d27f2cb1b3236b66dbbd9e34353726a2a8b704c6eb4d96b9b.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/7328b7b30b86696d27f2cb1b3236b66dbbd9e34353726a2a8b704c6eb4d96b9b.jpg)

![73e8a9c7bbd32b3268d7b8b7bfc1e0f699721ff8dc6cd40a56a2483a11952752.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/73e8a9c7bbd32b3268d7b8b7bfc1e0f699721ff8dc6cd40a56a2483a11952752.jpg)

![7735b147e944b2b60f9b14fb8b83b22db3326fb29bde4d7efac0d84d2222d17f.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/7735b147e944b2b60f9b14fb8b83b22db3326fb29bde4d7efac0d84d2222d17f.jpg)

![7d1df5bce65fa4775bf3b24f26b6203c0341882463252e404eec528fda9a5a2c.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/7d1df5bce65fa4775bf3b24f26b6203c0341882463252e404eec528fda9a5a2c.jpg)

![816b73534c39656870833269ee7e73a551e075a4f3971a4f7ae2490426a0a6a3.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/816b73534c39656870833269ee7e73a551e075a4f3971a4f7ae2490426a0a6a3.jpg)

![8b278ad5e961fa49a6a6d4afb007d227302563d08537bc0ee0d19b1e22dc173c.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/8b278ad5e961fa49a6a6d4afb007d227302563d08537bc0ee0d19b1e22dc173c.jpg)

![8e2fd8543743d2310c9528467586d2cd3339a98763ff8c35dafc88f485c4ac39.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/8e2fd8543743d2310c9528467586d2cd3339a98763ff8c35dafc88f485c4ac39.jpg)

![8f7dc4ab7f06a9fa939e05da689c9457394612ac6bc7c7a05945818ac308ccc8.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/8f7dc4ab7f06a9fa939e05da689c9457394612ac6bc7c7a05945818ac308ccc8.jpg)

![93c7499aa004cdc6406b7374eb3abeca469770f873f083021504d6bc5af6c085.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/93c7499aa004cdc6406b7374eb3abeca469770f873f083021504d6bc5af6c085.jpg)

![959b524665cef9afd7640c2150556afad7c4d6c6d1cb91491857e03f18af9ebd.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/959b524665cef9afd7640c2150556afad7c4d6c6d1cb91491857e03f18af9ebd.jpg)

![9db0c519405e4367f4f1cc63a5693e3ea52699812760bff7a7e3f5f17bb3610c.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/9db0c519405e4367f4f1cc63a5693e3ea52699812760bff7a7e3f5f17bb3610c.jpg)

![a69b6e8b567036db46993fd4c8082f952280a522bfec083b6bcebf5ac6e5b7cf.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/a69b6e8b567036db46993fd4c8082f952280a522bfec083b6bcebf5ac6e5b7cf.jpg)

![a9143a8d0e2e22fb205c8c896345007aef462b317eeb63c1a796922f54ea615b.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/a9143a8d0e2e22fb205c8c896345007aef462b317eeb63c1a796922f54ea615b.jpg)

![ae2f7169e89cb41b3d0c748fd577804abce26762c9aab33cae85d96c445d1db1.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/ae2f7169e89cb41b3d0c748fd577804abce26762c9aab33cae85d96c445d1db1.jpg)

![af139f8a173ccb01f0d00913b921950d871d416ca570e06ee97d9b9f6c424feb.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/af139f8a173ccb01f0d00913b921950d871d416ca570e06ee97d9b9f6c424feb.jpg)

![b685e2ae6ba05ec31baea05448a1e0ccf54b2ad297cbd809fac7c4e4aea0bd4f.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/b685e2ae6ba05ec31baea05448a1e0ccf54b2ad297cbd809fac7c4e4aea0bd4f.jpg)

![c1d615692612efcd917e372374a12f84313a83e252eb366a27540371b5e3154d.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/c1d615692612efcd917e372374a12f84313a83e252eb366a27540371b5e3154d.jpg)

![c55af2c42de0be5d2ecdce1e4978ed868d64810c80524ab5a551301ed980a904.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/c55af2c42de0be5d2ecdce1e4978ed868d64810c80524ab5a551301ed980a904.jpg)

![c998e5ab9076c5668ea8ae7d86ccbffb412e13701d0ba6a492b7478bb3296397.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/c998e5ab9076c5668ea8ae7d86ccbffb412e13701d0ba6a492b7478bb3296397.jpg)

![cc9584cb3e89e5c00551ba58cc1c69791f99f31bb7874d263e2f21d6b24d0434.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/cc9584cb3e89e5c00551ba58cc1c69791f99f31bb7874d263e2f21d6b24d0434.jpg)

![cd1583e77770570e94a5d7f810af508f2de5e398bf34e152faf602249f5ef153.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/cd1583e77770570e94a5d7f810af508f2de5e398bf34e152faf602249f5ef153.jpg)

![cd17c63d985c17eda5bed7190787b100c6a6db491dc43fa64eae571a555df5dd.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/cd17c63d985c17eda5bed7190787b100c6a6db491dc43fa64eae571a555df5dd.jpg)

![d3b2c7eab4feac136ffe5984ae77d1dc5fe0a6b143b1fad960e58df576939ce6.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/d3b2c7eab4feac136ffe5984ae77d1dc5fe0a6b143b1fad960e58df576939ce6.jpg)

![dd90f945e12b991ae1a195649effa5f87c5f38c13af7c9f9b02e87b8319c12e5.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/dd90f945e12b991ae1a195649effa5f87c5f38c13af7c9f9b02e87b8319c12e5.jpg)

![e1ae6a37386522097901ebf1828811ae0bc1dea7fd95489dee4588609aaf08a4.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/e1ae6a37386522097901ebf1828811ae0bc1dea7fd95489dee4588609aaf08a4.jpg)

![eb0241772b18ed026326bd008ff9be25c97d9543c1b49bd946db335fda99b3c8.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/images/eb0241772b18ed026326bd008ff9be25c97d9543c1b49bd946db335fda99b3c8.jpg)

### Tables

![3bbf05cfcac99cede767259bfb6d0be6e6df2c0a30a91a46d90def09188c8729.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/tables/3bbf05cfcac99cede767259bfb6d0be6e6df2c0a30a91a46d90def09188c8729.jpg)

![3c48b91e69c3f9e0e8857d525e32bda2246d11acd2ed08a6a2751ea620783b95.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/tables/3c48b91e69c3f9e0e8857d525e32bda2246d11acd2ed08a6a2751ea620783b95.jpg)

![883a8440151ebdeb5680cda7f1b34417d38fafe74648026ff41eb2db835e4bfc.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/tables/883a8440151ebdeb5680cda7f1b34417d38fafe74648026ff41eb2db835e4bfc.jpg)

![a132dfd05919864773787fb829fc62cd2673b0fd3495620b73adcd9b61ff5a28.jpg](../icml_results/3598_Multi-Integration%20of%20Labels%20Across%20Categories%20for%20Component%20Identification%20in%20Multi-trial%20Time%20Serie/tables/a132dfd05919864773787fb829fc62cd2673b0fd3495620b73adcd9b61ff5a28.jpg)

<a id="paper-3559"></a>

## Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features


### Images

![234abc78eeef4824612d34e0055cbc341ee07f37d1ac2d075a6c96ee9a629e8a.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/234abc78eeef4824612d34e0055cbc341ee07f37d1ac2d075a6c96ee9a629e8a.jpg)

![32c10d817c1a340793cae6c05fdab7d89c52d55a5475530881d98c5aef19b3e7.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/32c10d817c1a340793cae6c05fdab7d89c52d55a5475530881d98c5aef19b3e7.jpg)

![346f8b78802e1f827a2745acc19f0155bc038106f1b57a64f366d261e1cfed29.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/346f8b78802e1f827a2745acc19f0155bc038106f1b57a64f366d261e1cfed29.jpg)

![39364842d9c4cc9860b1a8c7bfb1a24e9f4d7adc280514ae03ef954bcf8c4876.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/39364842d9c4cc9860b1a8c7bfb1a24e9f4d7adc280514ae03ef954bcf8c4876.jpg)

![4874ce3457841b7561a0139168a6916153c705b12e395a1f122a1776888ad95a.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/4874ce3457841b7561a0139168a6916153c705b12e395a1f122a1776888ad95a.jpg)

![6bdc1aebee5f2857f77b85df72ddf6906f89e764b60e3b8d0335de9965090556.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/6bdc1aebee5f2857f77b85df72ddf6906f89e764b60e3b8d0335de9965090556.jpg)

![7090cd564cca534cc77527e54b6d68bcfd62c29192013793bccdac96b0a4f2f2.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/7090cd564cca534cc77527e54b6d68bcfd62c29192013793bccdac96b0a4f2f2.jpg)

![742e3be6ae480a1aaf466e08372e1edfe788f155aa312a4e48c06ba187ad4678.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/742e3be6ae480a1aaf466e08372e1edfe788f155aa312a4e48c06ba187ad4678.jpg)

![859f5e512dc2e2b0ba8b0eb13567c4319fd449567ddb570f4b07f2fddd43d3a2.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/859f5e512dc2e2b0ba8b0eb13567c4319fd449567ddb570f4b07f2fddd43d3a2.jpg)

![9785b4f1580e702944a0173bbbf8664d55dcad042c6992e8f0cb2bc316cb83db.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9785b4f1580e702944a0173bbbf8664d55dcad042c6992e8f0cb2bc316cb83db.jpg)

![9ae0ba7649a3b2eeb9048446e85dd2b04fa839d2af41cdc1ac7e7ee8920f1380.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9ae0ba7649a3b2eeb9048446e85dd2b04fa839d2af41cdc1ac7e7ee8920f1380.jpg)

![9b62c873ee06dd47c2c340a301a9dde6d0841cb0c42d196ef7e8aa47dc78f924.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9b62c873ee06dd47c2c340a301a9dde6d0841cb0c42d196ef7e8aa47dc78f924.jpg)

![9be4a47182a5e28bd194ec97f823108987862afd71e399b6c8988aebe0ce9775.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9be4a47182a5e28bd194ec97f823108987862afd71e399b6c8988aebe0ce9775.jpg)

![9d60ed2e512102280c17d0127f319fada95fa733a10039a08125e5d6f0558f3c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9d60ed2e512102280c17d0127f319fada95fa733a10039a08125e5d6f0558f3c.jpg)

![9d99466f40948df54456688a43cfdae3324a69cad56e0fa6fbfacb730fe5180e.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/9d99466f40948df54456688a43cfdae3324a69cad56e0fa6fbfacb730fe5180e.jpg)

![a0a5b8edaa80c8925c2a5c118c8b89c69f70c32e60f1fd0642475d25716a2903.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/a0a5b8edaa80c8925c2a5c118c8b89c69f70c32e60f1fd0642475d25716a2903.jpg)

![a7b441754fd527fbb98504c20b5a94e4157326e7712e52f1db089f6d4b894ecd.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/a7b441754fd527fbb98504c20b5a94e4157326e7712e52f1db089f6d4b894ecd.jpg)

![a8393cb10bfc342c6d017c48830cc6d91aaa11e40727e34b48c60bbd374a4840.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/a8393cb10bfc342c6d017c48830cc6d91aaa11e40727e34b48c60bbd374a4840.jpg)

![bb97277244c10d62659946b8f22a2b1b441aaade59fccc789d7d322b7806bd7d.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/bb97277244c10d62659946b8f22a2b1b441aaade59fccc789d7d322b7806bd7d.jpg)

![c5dc017cdb6de8173da1be25dee8937e819beef1bf7c38b447b40e0245da0e6c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/c5dc017cdb6de8173da1be25dee8937e819beef1bf7c38b447b40e0245da0e6c.jpg)

![d23285326e54c0cdbdd4ab31e5a35f2de4f64b8a5b0c45fd96698314fd5ec5f5.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/d23285326e54c0cdbdd4ab31e5a35f2de4f64b8a5b0c45fd96698314fd5ec5f5.jpg)

![d72dbd18c987aed17f2b3f275688e39e1b332094f849590c2b60ee1dbb9cdbc8.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/d72dbd18c987aed17f2b3f275688e39e1b332094f849590c2b60ee1dbb9cdbc8.jpg)

![d82fd09b84ffd3eda3084844e2ab2fa8be46a03a98a1ddc3193feb7440c4067c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/d82fd09b84ffd3eda3084844e2ab2fa8be46a03a98a1ddc3193feb7440c4067c.jpg)

![ed9443ef937fee97953e9631c1b6d5e7cf6cf0f5175c55e942f71c603459d951.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/ed9443ef937fee97953e9631c1b6d5e7cf6cf0f5175c55e942f71c603459d951.jpg)

![f259a9aefa07e19f5471628085f569aaf8505126d74ffec584f06f3b05feb5dc.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/images/f259a9aefa07e19f5471628085f569aaf8505126d74ffec584f06f3b05feb5dc.jpg)

### Tables

![07b6b4707e79153c33fc9ba467392a8c1580a17b42b9c569acf6ee190d1d6923.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/07b6b4707e79153c33fc9ba467392a8c1580a17b42b9c569acf6ee190d1d6923.jpg)

![163ec0e588f0780a56bc1ee37bdb7a5fa6f801df7b5318566591d586490d3470.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/163ec0e588f0780a56bc1ee37bdb7a5fa6f801df7b5318566591d586490d3470.jpg)

![1e9cffd7b196ad1cb726a8ea92a28c6d68ce40eb7b2f8343cd90aed2581b460e.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/1e9cffd7b196ad1cb726a8ea92a28c6d68ce40eb7b2f8343cd90aed2581b460e.jpg)

![2012cbb1556736c882b912518236a3868a7d1ef03bc4e1ac6462ac991fd8cf98.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/2012cbb1556736c882b912518236a3868a7d1ef03bc4e1ac6462ac991fd8cf98.jpg)

![2eda647bc5423584bfcad276a9c411d07e24fbb8a18e6893adcf6e48c0c32738.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/2eda647bc5423584bfcad276a9c411d07e24fbb8a18e6893adcf6e48c0c32738.jpg)

![489c210b922ac8a5fcbe06e721854330b4e542e93e4b7283702d4316311f6893.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/489c210b922ac8a5fcbe06e721854330b4e542e93e4b7283702d4316311f6893.jpg)

![4c2a51b70c1c2688ba41e41f048c98c1131f8a44f9350fa8aeb9d1b53a48add1.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/4c2a51b70c1c2688ba41e41f048c98c1131f8a44f9350fa8aeb9d1b53a48add1.jpg)

![50e2b17f82c06e11576ab3203a4b2a7ea98f189b39cccd333fbfdae27e30e213.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/50e2b17f82c06e11576ab3203a4b2a7ea98f189b39cccd333fbfdae27e30e213.jpg)

![5a94a031f7998ffa4788325e3623c139f4b0f346342b36495a40826e37e7485c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/5a94a031f7998ffa4788325e3623c139f4b0f346342b36495a40826e37e7485c.jpg)

![6499d4055782d15ea37d2a6c2d6abf4025cbb0b4c0d8e6a57d18ba458ca5b2f5.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/6499d4055782d15ea37d2a6c2d6abf4025cbb0b4c0d8e6a57d18ba458ca5b2f5.jpg)

![661393bd5f3dc83d9c1ad63f114284318d4d93b516e5eb0a9769600c76542e00.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/661393bd5f3dc83d9c1ad63f114284318d4d93b516e5eb0a9769600c76542e00.jpg)

![6b2e1b1457ac517da24c0bb2afb315748b7e04f50ba0d1188d4b05adb9e50f5c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/6b2e1b1457ac517da24c0bb2afb315748b7e04f50ba0d1188d4b05adb9e50f5c.jpg)

![707c8b7ea235bc07391691dfba319615920a558a176c079835791bcc6f84c921.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/707c8b7ea235bc07391691dfba319615920a558a176c079835791bcc6f84c921.jpg)

![7f6a1b1c0580bdc8e759e5c6633da3653d7d61265b822ee61b39e47e0734cf4f.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/7f6a1b1c0580bdc8e759e5c6633da3653d7d61265b822ee61b39e47e0734cf4f.jpg)

![91ecccf4e288ff638a06b6e2a107178db2a9ac5cb52823ecc6816d7b743df713.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/91ecccf4e288ff638a06b6e2a107178db2a9ac5cb52823ecc6816d7b743df713.jpg)

![9291a104d7abd34e264accf00e6c102182cf26ce29e7e1416490f2759377a920.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/9291a104d7abd34e264accf00e6c102182cf26ce29e7e1416490f2759377a920.jpg)

![9880c3cedb231bda13d4e4546cc8dfb2cb3ab5e2affe61941638a25b93e73b4c.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/9880c3cedb231bda13d4e4546cc8dfb2cb3ab5e2affe61941638a25b93e73b4c.jpg)

![a3bf0fc6734045c553ae7e9b8180f38a3eaa2f8db33b5bb21b2bc2883a8906b2.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/a3bf0fc6734045c553ae7e9b8180f38a3eaa2f8db33b5bb21b2bc2883a8906b2.jpg)

![becf6acff00e834c63c352778a574732f881d6eb432955770bb35941ca694c63.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/becf6acff00e834c63c352778a574732f881d6eb432955770bb35941ca694c63.jpg)

![bf6faa5ae1dd433a78b3967dca496507c37c3ff2e7b32dbc242b9a892b439a76.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/bf6faa5ae1dd433a78b3967dca496507c37c3ff2e7b32dbc242b9a892b439a76.jpg)

![d1df61329c4963ac5c8076c1018593db81355207861ead3b47c3a4f11f28b7a9.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/d1df61329c4963ac5c8076c1018593db81355207861ead3b47c3a4f11f28b7a9.jpg)

![d3748dafe7c111171d4cc06bfa47b9cc82a5a8dfc5f862a0e539463246a7d3df.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/d3748dafe7c111171d4cc06bfa47b9cc82a5a8dfc5f862a0e539463246a7d3df.jpg)

![dedd492aba6bc51ba8bb2f9379ed4aa9014fe8b945d370c822608792a2e94905.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/dedd492aba6bc51ba8bb2f9379ed4aa9014fe8b945d370c822608792a2e94905.jpg)

![df358f45363fd99b5c2e2a552b78b4abf0d6982cbe9f47176fe4d06faf27208e.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/df358f45363fd99b5c2e2a552b78b4abf0d6982cbe9f47176fe4d06faf27208e.jpg)

![e7427c8ef859b7d7725617474f68299e88cd20880447bef68534a1ccd2920850.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/e7427c8ef859b7d7725617474f68299e88cd20880447bef68534a1ccd2920850.jpg)

![f9c8c42f701e512b779a38f1f968a16b18e8cb3018e95d73e493d1b75390c121.jpg](../icml_results/3599_Cascaded%20Flow%20Matching%20for%20Heterogeneous%20Tabular%20Data%20with%20Mixed-Type%20Features/tables/f9c8c42f701e512b779a38f1f968a16b18e8cb3018e95d73e493d1b75390c121.jpg)

<a id="paper-3560"></a>

## UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations


### Images

![0c37d6936e6598f0c6fce5176e90aa4cb6aca1d8952bf397183dbad41cc6b793.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/0c37d6936e6598f0c6fce5176e90aa4cb6aca1d8952bf397183dbad41cc6b793.jpg)

![2b227be9ae9e64c7de57ffafc5e11e441bb95b85ce2e7bf47bbe95c92528fee2.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/2b227be9ae9e64c7de57ffafc5e11e441bb95b85ce2e7bf47bbe95c92528fee2.jpg)

![30a79ece0926b34cfb6029fe8da6eb2a84b7574d5ad302f8120f1236e57e5678.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/30a79ece0926b34cfb6029fe8da6eb2a84b7574d5ad302f8120f1236e57e5678.jpg)

![4097047f342bce0b347714bb3297140fb9a3beccd8613d2dab93ad90a48a9b60.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/4097047f342bce0b347714bb3297140fb9a3beccd8613d2dab93ad90a48a9b60.jpg)

![4770d8b404ac7cd964e0e9ad3c1bd61d933df57270eb3599ca76fd577700cbf6.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/4770d8b404ac7cd964e0e9ad3c1bd61d933df57270eb3599ca76fd577700cbf6.jpg)

![77b37445bba3e53480ac525e1c27e6610a95a7b314dfa4f422c9f8d4c893b559.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/77b37445bba3e53480ac525e1c27e6610a95a7b314dfa4f422c9f8d4c893b559.jpg)

![7e11cd7619f02b856f077a48855b2ca3f30f4d868b58a550d514e412324f7ae6.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/7e11cd7619f02b856f077a48855b2ca3f30f4d868b58a550d514e412324f7ae6.jpg)

![84f35e60da64d5aed5744c752557279bad057a0c36284b4a6d244028d3d8e684.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/84f35e60da64d5aed5744c752557279bad057a0c36284b4a6d244028d3d8e684.jpg)

![9145a7655f63307c032ed8b950b7598b1627d904c05a4c2b17d764d226ae21e4.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/9145a7655f63307c032ed8b950b7598b1627d904c05a4c2b17d764d226ae21e4.jpg)

![9187ea6ed6535c127a7a13be37f8e815905ae525288206aafbe45c5cebdcec0e.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/9187ea6ed6535c127a7a13be37f8e815905ae525288206aafbe45c5cebdcec0e.jpg)

![964dcad13f1af13f33f0d1be44c6f1b0c9b17f1ccd1dc763be6bd4b0c3784ff4.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/964dcad13f1af13f33f0d1be44c6f1b0c9b17f1ccd1dc763be6bd4b0c3784ff4.jpg)

![a2955fa5ffeccff9fe32627aa66b54351fe4900d62db345a5ef975e8d8dd26fc.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/a2955fa5ffeccff9fe32627aa66b54351fe4900d62db345a5ef975e8d8dd26fc.jpg)

![afe16d2e360523c93c158fa2dc28278ca8caaa0f4fda98c2295055ca6230369f.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/afe16d2e360523c93c158fa2dc28278ca8caaa0f4fda98c2295055ca6230369f.jpg)

![c2b3986cbbd30111be750d5daa3a0cb70c2652e41521252de6ce1d83eba6da28.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/c2b3986cbbd30111be750d5daa3a0cb70c2652e41521252de6ce1d83eba6da28.jpg)

![cdb7b786b9fdb2ad4140cfd2c833e959b9039717760828c948ba3354083f3e2a.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/cdb7b786b9fdb2ad4140cfd2c833e959b9039717760828c948ba3354083f3e2a.jpg)

![d265309e04f959913eda53fa023d6a28c2d7245f740cb128b2809571f27d71ef.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/d265309e04f959913eda53fa023d6a28c2d7245f740cb128b2809571f27d71ef.jpg)

![ef8718bc1c0b850ea640c87550b3246993681bab724a758f598ef997c646b747.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/ef8718bc1c0b850ea640c87550b3246993681bab724a758f598ef997c646b747.jpg)

![f6b55558aa058cfcf89fac39b1f52becfd99f065eb7917f7db6989eb04300dda.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/f6b55558aa058cfcf89fac39b1f52becfd99f065eb7917f7db6989eb04300dda.jpg)

![fedfc527e9549fbf047373af68dbc74fbd4c54e8cb904123db18f2f5229d6f60.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/images/fedfc527e9549fbf047373af68dbc74fbd4c54e8cb904123db18f2f5229d6f60.jpg)

### Tables

![001a40a43feac31ccd003fe8b6b32827d6d95fa2ee534adc101ac8b47cd7fb68.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/001a40a43feac31ccd003fe8b6b32827d6d95fa2ee534adc101ac8b47cd7fb68.jpg)

![083ab669b46e970ba00afdc463d96360226cbf42ded50b39936a8f97fa8c526b.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/083ab669b46e970ba00afdc463d96360226cbf42ded50b39936a8f97fa8c526b.jpg)

![0c622ad6dc6b673345e30b220a85471f9310c417c1622374c40e22a155c222e9.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/0c622ad6dc6b673345e30b220a85471f9310c417c1622374c40e22a155c222e9.jpg)

![2ccb0074fbff862fbe5224f64654f7e9bafa40763ff98c7c4a5f7f0fe3c6578b.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/2ccb0074fbff862fbe5224f64654f7e9bafa40763ff98c7c4a5f7f0fe3c6578b.jpg)

![2f9e2c4c3f06866a54a421254ffce86afdf7c51853e3dc5104ed50ef89ead8dc.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/2f9e2c4c3f06866a54a421254ffce86afdf7c51853e3dc5104ed50ef89ead8dc.jpg)

![44956c6f90c67d1bb7d088327519f0404d807e9ae1bf877bc5388412d5a3b120.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/44956c6f90c67d1bb7d088327519f0404d807e9ae1bf877bc5388412d5a3b120.jpg)

![4657f7cbdf5829173885bcb4b044f8ab8bcca9bf25acc4cb3ad78327244fe203.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/4657f7cbdf5829173885bcb4b044f8ab8bcca9bf25acc4cb3ad78327244fe203.jpg)

![46b0178fb4085b7197a1ab552d24cf1309e76c39cc76873abebc078baad47485.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/46b0178fb4085b7197a1ab552d24cf1309e76c39cc76873abebc078baad47485.jpg)

![4a7ac6e886a8c717d3df15bcb9de233d64ce8870a4b1ab08b45a5b013a52727c.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/4a7ac6e886a8c717d3df15bcb9de233d64ce8870a4b1ab08b45a5b013a52727c.jpg)

![56e8e022970b6e4a65df7a542b86524cbd17a510ed5e65ed5c9ef24302a8986a.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/56e8e022970b6e4a65df7a542b86524cbd17a510ed5e65ed5c9ef24302a8986a.jpg)

![5fe64c2e53219121d7c5b61bce7d6bd8bd67b6d899f232bae47c327190ab0eb7.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/5fe64c2e53219121d7c5b61bce7d6bd8bd67b6d899f232bae47c327190ab0eb7.jpg)

![62ec6d11e6500d6e69a879b073bce8b765ecc6fef41b50d47621253f18f3c8e6.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/62ec6d11e6500d6e69a879b073bce8b765ecc6fef41b50d47621253f18f3c8e6.jpg)

![63766ad14002bc56da723607600ba4771367714dd29f64e98825711fbc587aa8.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/63766ad14002bc56da723607600ba4771367714dd29f64e98825711fbc587aa8.jpg)

![66c72389f6a9f2c4ee242eb609b43d10be55319bad85ceaf748608324619744b.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/66c72389f6a9f2c4ee242eb609b43d10be55319bad85ceaf748608324619744b.jpg)

![6f8f3bf5cebee89d86bed8123df0921c9e9ef380a25596e8ac6927065c6c3b9b.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/6f8f3bf5cebee89d86bed8123df0921c9e9ef380a25596e8ac6927065c6c3b9b.jpg)

![78890d75ae8f834a43e98c8486c60cafae4a97c7f0787baabe803190b85c1921.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/78890d75ae8f834a43e98c8486c60cafae4a97c7f0787baabe803190b85c1921.jpg)

![7b98aa16fe460201424a13f9380c622195430dea8f44e7c48b6636116df51c5a.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/7b98aa16fe460201424a13f9380c622195430dea8f44e7c48b6636116df51c5a.jpg)

![810081f0afa7860c65dc95335603f4a84bc1440146851ec1cf48c9cd1ec9ff41.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/810081f0afa7860c65dc95335603f4a84bc1440146851ec1cf48c9cd1ec9ff41.jpg)

![8ac4f19ba7edd1ea91d0afdc27ab85c6ef5621e379c8f2d5f86b5f64c5c4b5aa.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/8ac4f19ba7edd1ea91d0afdc27ab85c6ef5621e379c8f2d5f86b5f64c5c4b5aa.jpg)

![915d5e49ba5a8381807e8e72216c294ae5696d79a9b0570f74300d1c677e09ad.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/915d5e49ba5a8381807e8e72216c294ae5696d79a9b0570f74300d1c677e09ad.jpg)

![933cda6f7572485177e85caec5e00d36730639bcdcab8ed5da1f4fbb7a2547bb.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/933cda6f7572485177e85caec5e00d36730639bcdcab8ed5da1f4fbb7a2547bb.jpg)

![a17da940e69c2ca087c26485cc453b47e6393f1f0dd6c1df52ce1b47bf985758.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/a17da940e69c2ca087c26485cc453b47e6393f1f0dd6c1df52ce1b47bf985758.jpg)

![a652eea7eb98e31a1ea7e8faafa8600e5c140cd8cec29c03bc70d4820e6f4525.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/a652eea7eb98e31a1ea7e8faafa8600e5c140cd8cec29c03bc70d4820e6f4525.jpg)

![ac28e878224c0588f730696254c2fd1c25dfad4cb131b857739296b8bc9ed024.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/ac28e878224c0588f730696254c2fd1c25dfad4cb131b857739296b8bc9ed024.jpg)

![b00cb67503ac10f757d4b6ccc59d4446b5583d2433d430dfdf4e7c0c90dcd3e8.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/b00cb67503ac10f757d4b6ccc59d4446b5583d2433d430dfdf4e7c0c90dcd3e8.jpg)

![b81ae16ed9690fcac83ff19a20548034027a53978fcc45058b19979178f4d353.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/b81ae16ed9690fcac83ff19a20548034027a53978fcc45058b19979178f4d353.jpg)

![b857c32d41a920241da38e7cecd10750790097d702633fbdf5122b8faf9e82d3.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/b857c32d41a920241da38e7cecd10750790097d702633fbdf5122b8faf9e82d3.jpg)

![d5cfdd62bf4ab797caa30759acc7a20737a6808822e16d967c7faeaff905f4fe.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/d5cfdd62bf4ab797caa30759acc7a20737a6808822e16d967c7faeaff905f4fe.jpg)

![daaa464bfa309f49746deba379d49aa72cd2d1d23ec1c035170ab8eed22f17c7.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/daaa464bfa309f49746deba379d49aa72cd2d1d23ec1c035170ab8eed22f17c7.jpg)

![fe8eca886ac93b8becae276fc1bc6f1707ee5f2ea4613577884a27fd78276b8d.jpg](../icml_results/3600_UrbanFusion_%20Stochastic%20Multimodal%20Fusion%20for%20Contrastive%20Learning%20of%20Robust%20Spatial%20Representations/tables/fe8eca886ac93b8becae276fc1bc6f1707ee5f2ea4613577884a27fd78276b8d.jpg)

<a id="paper-3561"></a>

## When Does Sparsity Mitigate the Curse of Depth in LLMs


### Images

![10e2e7b25904aa57a05aec04cf77985af4d9c38e01e3472194c847a23477dc47.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/10e2e7b25904aa57a05aec04cf77985af4d9c38e01e3472194c847a23477dc47.jpg)

![31665012fb2557cf81e0ff22b19afb612a021711ac4506e1290f6089f8c07289.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/31665012fb2557cf81e0ff22b19afb612a021711ac4506e1290f6089f8c07289.jpg)

![37c2f742915f67a59d265813ebf51c7ce2ee33de20536285216f742d3c7402c4.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/37c2f742915f67a59d265813ebf51c7ce2ee33de20536285216f742d3c7402c4.jpg)

![3c413ac43ad2c8e0ecb88e47fd5287dbc24884a4c7db834ea614b79624abfbca.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/3c413ac43ad2c8e0ecb88e47fd5287dbc24884a4c7db834ea614b79624abfbca.jpg)

![3feca002de506ab8e39f8f88d90d3dc08362320ce875821b676ce7aa478bcc9b.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/3feca002de506ab8e39f8f88d90d3dc08362320ce875821b676ce7aa478bcc9b.jpg)

![52cdc468b572f90f41dd4884a16d64796861517b689bef9c1e014e418eea08b5.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/52cdc468b572f90f41dd4884a16d64796861517b689bef9c1e014e418eea08b5.jpg)

![5a394d571e519422cd479cd2183f9ac10a9f812a5a7fb4dd8940a77ed0307d20.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/5a394d571e519422cd479cd2183f9ac10a9f812a5a7fb4dd8940a77ed0307d20.jpg)

![64beaa21e86243257d39f8f401d9663551c24fc2369ac0c0b9c35d251212f114.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/64beaa21e86243257d39f8f401d9663551c24fc2369ac0c0b9c35d251212f114.jpg)

![65068ab94883e2ba6b13298870522265c62e9bd34b1096ee6c3300af3e33f590.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/65068ab94883e2ba6b13298870522265c62e9bd34b1096ee6c3300af3e33f590.jpg)

![6aead990cb8fedab38c96c328df638c087266ec70599b7bf2ada33e1cf7415a5.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/6aead990cb8fedab38c96c328df638c087266ec70599b7bf2ada33e1cf7415a5.jpg)

![76c95f74c4f61a293c51f8f4aefc669273c898985b6c2fc4a2fcc7ccca7044e3.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/76c95f74c4f61a293c51f8f4aefc669273c898985b6c2fc4a2fcc7ccca7044e3.jpg)

![7ab5fc5e963eba654a80e8ae1865684ac80ac34b13e77c831eb97819d4bd7221.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/7ab5fc5e963eba654a80e8ae1865684ac80ac34b13e77c831eb97819d4bd7221.jpg)

![7e07c2b3405db40b36aa38c6838283a7fdfed699957d024a1ca193bd124e43a2.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/7e07c2b3405db40b36aa38c6838283a7fdfed699957d024a1ca193bd124e43a2.jpg)

![7f567ff85b204937bc81befa9916bb060ea2158f44c8f6ee54be8556f223a29e.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/7f567ff85b204937bc81befa9916bb060ea2158f44c8f6ee54be8556f223a29e.jpg)

![843520ca67f0946ff032e2c7d8339293b71e2c6c413d7e3cd2f77dc269dc7765.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/843520ca67f0946ff032e2c7d8339293b71e2c6c413d7e3cd2f77dc269dc7765.jpg)

![865cd029ecd0e3f30fdd017ae91e68cafdf0ecc2e5fd350e6e896533c31bf60c.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/865cd029ecd0e3f30fdd017ae91e68cafdf0ecc2e5fd350e6e896533c31bf60c.jpg)

![994c390db600007210fdf5b9381c8fce60f6b86df6407e9228edc1c9d2da8555.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/994c390db600007210fdf5b9381c8fce60f6b86df6407e9228edc1c9d2da8555.jpg)

![99962e3f72deb06f0e30b1a0e905875a4df8dbf26a15e30e53f251a8c032ef6c.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/99962e3f72deb06f0e30b1a0e905875a4df8dbf26a15e30e53f251a8c032ef6c.jpg)

![9f9f4c2835f199fcbaa970626094f3602e5dcac487a68ae0f954d033a5485f25.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/9f9f4c2835f199fcbaa970626094f3602e5dcac487a68ae0f954d033a5485f25.jpg)

![a2585bbdfc3a041708a0136615c624dbd257039811760847f36616cb5eaf76a9.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/a2585bbdfc3a041708a0136615c624dbd257039811760847f36616cb5eaf76a9.jpg)

![a3faad282fb8b0914600b18f9e6f14d632c6ced0a4fa497845c18606af877c3e.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/a3faad282fb8b0914600b18f9e6f14d632c6ced0a4fa497845c18606af877c3e.jpg)

![a9a7f15a898ab8d76e81b4209268e1de3501b1968dd2c344922860c89cb9f6b0.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/a9a7f15a898ab8d76e81b4209268e1de3501b1968dd2c344922860c89cb9f6b0.jpg)

![ac7210a9371cc60c8ddea77454a33d63ad472fb7d4bf758d935d63f927cf6409.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/ac7210a9371cc60c8ddea77454a33d63ad472fb7d4bf758d935d63f927cf6409.jpg)

![b043d9264553bf7a194b83b0400c1787fdd8ea29a854e0b8e554020ad8d90644.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/b043d9264553bf7a194b83b0400c1787fdd8ea29a854e0b8e554020ad8d90644.jpg)

![c57fce230d880e257b9c9cfa6974f477b41ea9749bd4792451859da901a4f614.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/c57fce230d880e257b9c9cfa6974f477b41ea9749bd4792451859da901a4f614.jpg)

![d4e75b41bad5cb971d0ad67931077d1d2cfb49bff77f847ca560fff8b5876262.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/d4e75b41bad5cb971d0ad67931077d1d2cfb49bff77f847ca560fff8b5876262.jpg)

![df86aed938c5002c2da2eb7f6639c347cb32b550143076d6e1fc4d31390485ca.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/df86aed938c5002c2da2eb7f6639c347cb32b550143076d6e1fc4d31390485ca.jpg)

![e16555616c217ca1d56ff0e5a96bd382ccf39ba52dcb87f021bf5ee8c586f1df.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/e16555616c217ca1d56ff0e5a96bd382ccf39ba52dcb87f021bf5ee8c586f1df.jpg)

![e2b3d3e8cd1fa9132a327511bbe358e620aa460c9a35632099b3630f9ec32495.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/e2b3d3e8cd1fa9132a327511bbe358e620aa460c9a35632099b3630f9ec32495.jpg)

![e9d96ade744eacce67364cb8fbf8cab4452fac31045465fba170ebfb0d16c9ce.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/images/e9d96ade744eacce67364cb8fbf8cab4452fac31045465fba170ebfb0d16c9ce.jpg)

### Tables

![531a50c67c1f89264d83ec204d0f994624aadf183f32b46cf72f6aed85fa1cfc.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/531a50c67c1f89264d83ec204d0f994624aadf183f32b46cf72f6aed85fa1cfc.jpg)

![5f7dc4bd0c9a9a898f97902bcd75d0e4936ee36c2d0f38f0fa5623d296717984.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/5f7dc4bd0c9a9a898f97902bcd75d0e4936ee36c2d0f38f0fa5623d296717984.jpg)

![6c41d9d5f8a1b6a61679e90a36d97c9b48891dc3a0496113365364089fcf7c1e.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/6c41d9d5f8a1b6a61679e90a36d97c9b48891dc3a0496113365364089fcf7c1e.jpg)

![a3d326e1a7a6a11ab1a992cf3bf9d9e64d8062e498165043e5d7261830af0216.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/a3d326e1a7a6a11ab1a992cf3bf9d9e64d8062e498165043e5d7261830af0216.jpg)

![aa8c50d4ee4760629eb52722950a2d4fbd0b60d374cfaef1d5656b5c7b88a402.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/aa8c50d4ee4760629eb52722950a2d4fbd0b60d374cfaef1d5656b5c7b88a402.jpg)

![ae4ea816117f5ba3bdbd625b7b6f1870f3fb17e1d6a0a81eea70670725cafca3.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/ae4ea816117f5ba3bdbd625b7b6f1870f3fb17e1d6a0a81eea70670725cafca3.jpg)

![b7daef63baa5225f37d0cd3f14ce8d6c92d20a3f474f3d289c09e15259a88d87.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/b7daef63baa5225f37d0cd3f14ce8d6c92d20a3f474f3d289c09e15259a88d87.jpg)

![d5b592fbf7c0b8ecb711f3d2d7af87dbc2e542bd77d398ec0914cd04ec13c4fc.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/d5b592fbf7c0b8ecb711f3d2d7af87dbc2e542bd77d398ec0914cd04ec13c4fc.jpg)

![e523f7a65dd686ee31912e3c9217c7a53c76a6cb3fb4eb41d0906b937f9b4212.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/e523f7a65dd686ee31912e3c9217c7a53c76a6cb3fb4eb41d0906b937f9b4212.jpg)

![f556b8e344c32503defc4bd3bffe65fcee374a9c8897be6d878e875dc89d5a2a.jpg](../icml_results/3601_When%20Does%20Sparsity%20Mitigate%20the%20Curse%20of%20Depth%20in%20LLMs/tables/f556b8e344c32503defc4bd3bffe65fcee374a9c8897be6d878e875dc89d5a2a.jpg)

<a id="paper-3562"></a>

## BOCLOAK: Optimal Transport-Guided Adversarial Attacks on Graph Neural Network-Based Bot Detection


### Images

![0cb2d208643a97b4a31d6015a72a15d187f6a35a7f28dacd8860d5b5164e2bd9.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/0cb2d208643a97b4a31d6015a72a15d187f6a35a7f28dacd8860d5b5164e2bd9.jpg)

![452ba8a0e41c2c46f1af2a535785cde1e656205e64d66cc5c0a6dc2aae2b7481.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/452ba8a0e41c2c46f1af2a535785cde1e656205e64d66cc5c0a6dc2aae2b7481.jpg)

![508ba58fd7a8dd83e260fce8b3a7505bd863e7aa81fc068ec7d1ee14559b7ce3.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/508ba58fd7a8dd83e260fce8b3a7505bd863e7aa81fc068ec7d1ee14559b7ce3.jpg)

![6c630fad9b2b74264d6811b6a11cbdd3bfa8dd2cffcbc0c95b6bf225393c58a2.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/6c630fad9b2b74264d6811b6a11cbdd3bfa8dd2cffcbc0c95b6bf225393c58a2.jpg)

![8c0c04d8de7054ea5c31442e53f6b3ab0afa0f315a59057ac90c999aed87ef4b.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/8c0c04d8de7054ea5c31442e53f6b3ab0afa0f315a59057ac90c999aed87ef4b.jpg)

![8ce00411315dac455c0b9b3cfaa334930a9f361009eb71c215bb3db4acddfa49.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/8ce00411315dac455c0b9b3cfaa334930a9f361009eb71c215bb3db4acddfa49.jpg)

![92ccb5815bb768d80ebda285bcac71973ba51d27981885d559ad5aebead7faa4.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/92ccb5815bb768d80ebda285bcac71973ba51d27981885d559ad5aebead7faa4.jpg)

![99637749553a12e209a84a732ea0f78aa3a6e23f96f0c3509e0194b391b1689c.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/99637749553a12e209a84a732ea0f78aa3a6e23f96f0c3509e0194b391b1689c.jpg)

![a9b2f288539dc8962ae43edbefdf672c66ae986ad228b08865d99f7509c17c5a.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/a9b2f288539dc8962ae43edbefdf672c66ae986ad228b08865d99f7509c17c5a.jpg)

![ca36bb8c518eec5b68fd6d17afa0b94cd7a59d1b82f110448ff58ca39324dacf.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/ca36bb8c518eec5b68fd6d17afa0b94cd7a59d1b82f110448ff58ca39324dacf.jpg)

![f6e9af944379adea0220d5cbdb0894f754c1675b407bb368b6e2d9681283b6ac.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/images/f6e9af944379adea0220d5cbdb0894f754c1675b407bb368b6e2d9681283b6ac.jpg)

### Tables

![143da20dea5777adb2aea52bd73fc1567afe47cd386a67bee5ea7752f704a669.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/143da20dea5777adb2aea52bd73fc1567afe47cd386a67bee5ea7752f704a669.jpg)

![16b5612a85e7639670cae4e2e63b9bdb9b341b75c6a75c01fc4a03d47115e39c.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/16b5612a85e7639670cae4e2e63b9bdb9b341b75c6a75c01fc4a03d47115e39c.jpg)

![1b2cf5f8ba0e876eeefa754c8a4a6127ae7324ea47f74ae3d36a23e86404a52b.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/1b2cf5f8ba0e876eeefa754c8a4a6127ae7324ea47f74ae3d36a23e86404a52b.jpg)

![3b71be3a198d6af9141df8c13ef707d3ebf2e6df5fd9cfd699787de162c9fb75.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/3b71be3a198d6af9141df8c13ef707d3ebf2e6df5fd9cfd699787de162c9fb75.jpg)

![435308d16589f413c66240c5496c74e40ce4ad4aac3862180b6e9a2c23aa1529.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/435308d16589f413c66240c5496c74e40ce4ad4aac3862180b6e9a2c23aa1529.jpg)

![6578d7ec349669616ca645be4c30377982d32db8437a7060c868b282de0eca08.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/6578d7ec349669616ca645be4c30377982d32db8437a7060c868b282de0eca08.jpg)

![6b6fc3c5c7ca4c09e2c1c38df0ab60650ece5f22002dd88db5b3e2314a91b68e.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/6b6fc3c5c7ca4c09e2c1c38df0ab60650ece5f22002dd88db5b3e2314a91b68e.jpg)

![79e4c7db662236fdc7a1918c4827b43223de4188eb7e4511339c0297e8e647fe.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/79e4c7db662236fdc7a1918c4827b43223de4188eb7e4511339c0297e8e647fe.jpg)

![87ddb68c1fd7ddba340c360236b4f7c85176e91764f5a0ff942861a8d9289ebb.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/87ddb68c1fd7ddba340c360236b4f7c85176e91764f5a0ff942861a8d9289ebb.jpg)

![8f9c87ac305a0f04802c9d79c59942fc699f31e7d9c0015980568185eaa836ab.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/8f9c87ac305a0f04802c9d79c59942fc699f31e7d9c0015980568185eaa836ab.jpg)

![9179daf248b38a0970b9e6557784217ae16713d51987af399b79d7734e8aab06.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/9179daf248b38a0970b9e6557784217ae16713d51987af399b79d7734e8aab06.jpg)

![9604498f7a371823b2f6c337f45f7088156249099d9c41f4342f400ad79ff2b3.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/9604498f7a371823b2f6c337f45f7088156249099d9c41f4342f400ad79ff2b3.jpg)

![9a69889235df53000b3997ad139538ec9acf35fb19fb8cf720bde683ef4874e8.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/9a69889235df53000b3997ad139538ec9acf35fb19fb8cf720bde683ef4874e8.jpg)

![adbc8729751daec283ee2120ee76354c99d9c37e684f8c95e74e4733a583657c.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/adbc8729751daec283ee2120ee76354c99d9c37e684f8c95e74e4733a583657c.jpg)

![b2fb9102b2f51860281aa00c30f8538ca916c80b3fadee3cc286abb88155cae4.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/b2fb9102b2f51860281aa00c30f8538ca916c80b3fadee3cc286abb88155cae4.jpg)

![b7a224dc251aede1fc38ffdf3766df2133f2352a1c2ac7523cc37632333920ed.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/b7a224dc251aede1fc38ffdf3766df2133f2352a1c2ac7523cc37632333920ed.jpg)

![bec6f65651d450937f4444038980b1c56dc35022bc8d6fc277a1715ea74e499f.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/bec6f65651d450937f4444038980b1c56dc35022bc8d6fc277a1715ea74e499f.jpg)

![c749c75a228726b22ff6d5a30ca2ec1900ca65a7a5cc4c657dcd1c5a793333a0.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/c749c75a228726b22ff6d5a30ca2ec1900ca65a7a5cc4c657dcd1c5a793333a0.jpg)

![c8217a9d087dfcc514cb7763cbc45f6656e55d07360bd2055886349411b4c2e0.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/c8217a9d087dfcc514cb7763cbc45f6656e55d07360bd2055886349411b4c2e0.jpg)

![cb0d117c3280764707f5462a77ba57183eba784ecfd581f1dc3acf902e140e48.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/cb0d117c3280764707f5462a77ba57183eba784ecfd581f1dc3acf902e140e48.jpg)

![cd98df505e25261af4e3a6b38ff9ee86ad1aa5df5b38d75e70547fd697373199.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/cd98df505e25261af4e3a6b38ff9ee86ad1aa5df5b38d75e70547fd697373199.jpg)

![dccced7f1822d3ed310a24b758ee234105f3f7d9e6c65583e835b487ea02e6d6.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/dccced7f1822d3ed310a24b758ee234105f3f7d9e6c65583e835b487ea02e6d6.jpg)

![ddb6906ead6f75087997975b96196a4410fb0bb9b220177126c1da03890e8071.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/ddb6906ead6f75087997975b96196a4410fb0bb9b220177126c1da03890e8071.jpg)

![e322a858c2667c9d146c90b0edece27cc8fd60519650fa0d66bd7c075a65a8db.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/e322a858c2667c9d146c90b0edece27cc8fd60519650fa0d66bd7c075a65a8db.jpg)

![f36e4245f30e6081c70f3ef8113f1fba9308af8eb5e11bd8cd4ccfa27aadce96.jpg](../icml_results/3602_BOCLOAK_%20Optimal%20Transport-Guided%20Adversarial%20Attacks%20on%20Graph%20Neural%20Network-Based%20Bot%20Detection/tables/f36e4245f30e6081c70f3ef8113f1fba9308af8eb5e11bd8cd4ccfa27aadce96.jpg)

<a id="paper-3563"></a>

## Latent Diffusion Pretraining for Crystal Property Prediction


### Images

![10e051e349632a9c97881a56bdf36df433c2262a0585b0c7c3b234cf8fd48bbf.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/10e051e349632a9c97881a56bdf36df433c2262a0585b0c7c3b234cf8fd48bbf.jpg)

![319e45b80845a21b5f2a0b57e1ceb579d936f8e5e826dc1016eca919e0a41af6.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/319e45b80845a21b5f2a0b57e1ceb579d936f8e5e826dc1016eca919e0a41af6.jpg)

![845bd931e80456a3682f996159cf977abc3b9b6c4dfcf095d6fda09b5cc93a37.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/845bd931e80456a3682f996159cf977abc3b9b6c4dfcf095d6fda09b5cc93a37.jpg)

![911569e5770fce29359d7af05b932cb626de5776f1a69d65f5ab31c3880028f5.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/911569e5770fce29359d7af05b932cb626de5776f1a69d65f5ab31c3880028f5.jpg)

![95d00c784db09d13f417c1873b39a089b30335c6adbfe625e3744ed9f2f5f762.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/95d00c784db09d13f417c1873b39a089b30335c6adbfe625e3744ed9f2f5f762.jpg)

![a76ebde40e65b018a494549f5e99bd9c25ea9795e3622ee527cf949e04357272.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/a76ebde40e65b018a494549f5e99bd9c25ea9795e3622ee527cf949e04357272.jpg)

![c16a9ccccdda9fb73beb5c67b0951653ec2af2f7f45faac52ad6e19b25d07da2.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/c16a9ccccdda9fb73beb5c67b0951653ec2af2f7f45faac52ad6e19b25d07da2.jpg)

![cb8ea9a3605f93adf07107aa3e0e6e0d4e63611d7f313a05a09042f371710a71.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/cb8ea9a3605f93adf07107aa3e0e6e0d4e63611d7f313a05a09042f371710a71.jpg)

![e2c009294a07dded13d4b206d2097dc998e3d2ab2164919876db569e6cd6b261.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/images/e2c009294a07dded13d4b206d2097dc998e3d2ab2164919876db569e6cd6b261.jpg)

### Tables

![1cacb640840ca76915464f30e1f932aa44a2b08b3ba07f605547ec631577ed95.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/1cacb640840ca76915464f30e1f932aa44a2b08b3ba07f605547ec631577ed95.jpg)

![2dc8ccb10f929846ab1554b174841e28d327728336b33151eedb82bb56cd80a4.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/2dc8ccb10f929846ab1554b174841e28d327728336b33151eedb82bb56cd80a4.jpg)

![336ced391e78554f3543a4ff878cfe533722589fada5913f74d9294c03718616.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/336ced391e78554f3543a4ff878cfe533722589fada5913f74d9294c03718616.jpg)

![4aa3dba4e9d05b9beab108d6e0960fc762e9ac1388aa5e31133d6378e417c0f2.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/4aa3dba4e9d05b9beab108d6e0960fc762e9ac1388aa5e31133d6378e417c0f2.jpg)

![55836d583285f4d83bceca09a8528dcae526b826228e1ea6068542f741b06d4e.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/55836d583285f4d83bceca09a8528dcae526b826228e1ea6068542f741b06d4e.jpg)

![57f17ce83d2476e06f67764fb488a7732b7311b98750780934bd76d4aac1710c.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/57f17ce83d2476e06f67764fb488a7732b7311b98750780934bd76d4aac1710c.jpg)

![ad06e222a145a5f7d344b29f2c2a1cb7d7a0f95b0a2ed0b01496c318cbfe959b.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/ad06e222a145a5f7d344b29f2c2a1cb7d7a0f95b0a2ed0b01496c318cbfe959b.jpg)

![cde15aafd8174f40532e6476471e2e798e20b223940670935f3927bcdf0c8138.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/cde15aafd8174f40532e6476471e2e798e20b223940670935f3927bcdf0c8138.jpg)

![d768dcb041a230f3068abf8454251504cddb9930566409b8aa26b31205343c57.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/d768dcb041a230f3068abf8454251504cddb9930566409b8aa26b31205343c57.jpg)

![deb0a187fa51fda463001e61e82b726b50a4da0b2f66ffe1444865e2c7ae14f8.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/deb0a187fa51fda463001e61e82b726b50a4da0b2f66ffe1444865e2c7ae14f8.jpg)

![fa41153b9b78c35e2ac8fd4211381fd0ea2f7535946642330d34f4523bfc31ed.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/fa41153b9b78c35e2ac8fd4211381fd0ea2f7535946642330d34f4523bfc31ed.jpg)

![fee9989f4a7cb71e250146d48b23d0b8aebf29999af50c6fcbc304fb860d0a0b.jpg](../icml_results/3603_Latent%20Diffusion%20Pretraining%20for%20Crystal%20Property%20Prediction/tables/fee9989f4a7cb71e250146d48b23d0b8aebf29999af50c6fcbc304fb860d0a0b.jpg)

<a id="paper-3564"></a>

## SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights

### Images

![2a857d7ee0a40426864a9a3c23db61c00e0f190f49b0bfeff194b72f9e92da75.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/2a857d7ee0a40426864a9a3c23db61c00e0f190f49b0bfeff194b72f9e92da75.jpg)

![6932deab912b205e15423a228f930b4455cf4be8bc3a67ce078fd19075226e71.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/6932deab912b205e15423a228f930b4455cf4be8bc3a67ce078fd19075226e71.jpg)

![98c360d93cc4b0ad339728895a6e0ccf1ff32c475f794f2d2930d9a1a9baef29.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/98c360d93cc4b0ad339728895a6e0ccf1ff32c475f794f2d2930d9a1a9baef29.jpg)

![a37f8719b8d4a5295c5e7a736e8d37062593992ae967cf9125e959527363bc4c.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/a37f8719b8d4a5295c5e7a736e8d37062593992ae967cf9125e959527363bc4c.jpg)

![ac42c15b8d404b5d189be53afb6bd85504a2ca67dffaa5cb5fa85b6d3889d8c7.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/ac42c15b8d404b5d189be53afb6bd85504a2ca67dffaa5cb5fa85b6d3889d8c7.jpg)

![b0587e155b4babb566c822700ef22630815b5cb27b8fe0de4d5ed9628aaf8e4e.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/b0587e155b4babb566c822700ef22630815b5cb27b8fe0de4d5ed9628aaf8e4e.jpg)

![cd2729d8f8cf1f7699236e17c69274a0104ad11206c6705be9243e4144949419.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/cd2729d8f8cf1f7699236e17c69274a0104ad11206c6705be9243e4144949419.jpg)

![d07b58891f876c7de73406b54e14b1c8cc5700b75b30f474f1a388f5ef6342d4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/d07b58891f876c7de73406b54e14b1c8cc5700b75b30f474f1a388f5ef6342d4.jpg)

![e0c7bba5cf4c28d58d66aa710bfdab3db445249bafb48eaeaadfa1a2bdc8203b.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/e0c7bba5cf4c28d58d66aa710bfdab3db445249bafb48eaeaadfa1a2bdc8203b.jpg)

![e2a31192cf4d62ce70d704ed1298397c85171cf819f17eab5b848dd3b4bb32dc.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/images/e2a31192cf4d62ce70d704ed1298397c85171cf819f17eab5b848dd3b4bb32dc.jpg)

### Tables

![0dcdb8a2d04a5ab22e5bbb69845b4dc23d2716e841acccc2ea626ad23e278c10.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/0dcdb8a2d04a5ab22e5bbb69845b4dc23d2716e841acccc2ea626ad23e278c10.jpg)

![2658302e66e85d572d43c5f20cc64d61a714b50c299ac5bed8b9e1849a8a7df0.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/2658302e66e85d572d43c5f20cc64d61a714b50c299ac5bed8b9e1849a8a7df0.jpg)

![30451beb988f5632e3b89f0761fad7b0973e5955c4706c51d25cd8e618db91c4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/30451beb988f5632e3b89f0761fad7b0973e5955c4706c51d25cd8e618db91c4.jpg)

![3205fe5fcf4944c4e84aa708e8c0c069fe29958428167a1c5553fb4f2e8d3e5a.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/3205fe5fcf4944c4e84aa708e8c0c069fe29958428167a1c5553fb4f2e8d3e5a.jpg)

![4195996f0b03dc356e8414722e5e7278593556fb47553c05daa4a752533ec0a0.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4195996f0b03dc356e8414722e5e7278593556fb47553c05daa4a752533ec0a0.jpg)

![4cb346f478ea87b9caa82636cd445831b55b4d632489494483e7409e1fee630b.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4cb346f478ea87b9caa82636cd445831b55b4d632489494483e7409e1fee630b.jpg)

![4d6c92bef348c671fd8ecb6d9ef48453ac91205508571a285db67d8a107ca147.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/4d6c92bef348c671fd8ecb6d9ef48453ac91205508571a285db67d8a107ca147.jpg)

![554d0cd0ba275af1dc8c734ccfd1277a4bd192c8a3065f9310b020ec93b65726.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/554d0cd0ba275af1dc8c734ccfd1277a4bd192c8a3065f9310b020ec93b65726.jpg)

![64276e3842d0c5011ea44f13b614b5d26074f4d4dfa8a8f79985f86b53f0b70e.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/64276e3842d0c5011ea44f13b614b5d26074f4d4dfa8a8f79985f86b53f0b70e.jpg)

![6fa31e5fb339fc0abfa6ca245e40bde1f4555d0e27a4bb5d93f1538f2bcc7ae2.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/6fa31e5fb339fc0abfa6ca245e40bde1f4555d0e27a4bb5d93f1538f2bcc7ae2.jpg)

![7113a649f47dda577fe18bc9c3733dca7d1fc85bcd91da6d5f49a19f91570498.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/7113a649f47dda577fe18bc9c3733dca7d1fc85bcd91da6d5f49a19f91570498.jpg)

![7b0aa4602d31e0185512f94a32739a27ab508e13867f86210affd7f61b0998bf.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/7b0aa4602d31e0185512f94a32739a27ab508e13867f86210affd7f61b0998bf.jpg)

![9bc0991cc9b2e859f704a8747f44e2643ebf07a9b4aa403db1d4b7b68bd83ddc.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/9bc0991cc9b2e859f704a8747f44e2643ebf07a9b4aa403db1d4b7b68bd83ddc.jpg)

![9f2891d62ee3175a0ba0d71aa586705d747bdbb9eb11c9ef90c804c731c19687.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/9f2891d62ee3175a0ba0d71aa586705d747bdbb9eb11c9ef90c804c731c19687.jpg)

![abf2e366b2f37299f8ffa004bbe957fce2fdb32bfa3d0ce8dee9e7d58dc09df9.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/abf2e366b2f37299f8ffa004bbe957fce2fdb32bfa3d0ce8dee9e7d58dc09df9.jpg)

![acb0c78abcef03e17b788e13096e66882b5b37a4ae32eeda54a0850c54b770df.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/acb0c78abcef03e17b788e13096e66882b5b37a4ae32eeda54a0850c54b770df.jpg)

![acc624e8d1a30c3f65579663838b33484f92d5f6cab6811999f44ab443f1f564.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/acc624e8d1a30c3f65579663838b33484f92d5f6cab6811999f44ab443f1f564.jpg)

![b03ff99d3f78e63f359d4e5deafef83c7955977dcfedc9cb67fae1ae85c51543.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b03ff99d3f78e63f359d4e5deafef83c7955977dcfedc9cb67fae1ae85c51543.jpg)

![b0ad46a6cd79bac3450a20f75692ec115ad2ee43180708cccd9ce133535ac196.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b0ad46a6cd79bac3450a20f75692ec115ad2ee43180708cccd9ce133535ac196.jpg)

![b123b3f18f168b6a040344b9a3aadf9f6582986f109cd927fe6ff7d81c2521e8.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b123b3f18f168b6a040344b9a3aadf9f6582986f109cd927fe6ff7d81c2521e8.jpg)

![b5574c39e81841a5f4f1f95902ccd4230ec2d4de34130195a95d7fec93cdd3ff.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/b5574c39e81841a5f4f1f95902ccd4230ec2d4de34130195a95d7fec93cdd3ff.jpg)

![bfb03a9ab1b3b5af532f82f87b24fef4dbdea663ed05500db50e28d370d1b1b4.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/bfb03a9ab1b3b5af532f82f87b24fef4dbdea663ed05500db50e28d370d1b1b4.jpg)

![c5f8190db7d0ad50afd93f4325f0ec9001059dba78fcc405b90220b915cde50c.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/c5f8190db7d0ad50afd93f4325f0ec9001059dba78fcc405b90220b915cde50c.jpg)

![d2d81989eb882c6db8a6c7658130d9ee6c6df34bb255c4ca80416df7816015c6.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/d2d81989eb882c6db8a6c7658130d9ee6c6df34bb255c4ca80416df7816015c6.jpg)

![ec2044678b90b7a3abb1e9939d9523e6a3e06d6293bb2d0f66a4862745854c40.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/ec2044678b90b7a3abb1e9939d9523e6a3e06d6293bb2d0f66a4862745854c40.jpg)

![f5caa8992afa8e9bf30f9d14a26193f1c0b7cb06ccb470ff5f93b3b1f9a36d3a.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/f5caa8992afa8e9bf30f9d14a26193f1c0b7cb06ccb470ff5f93b3b1f9a36d3a.jpg)

![f9bbac40e42acaa0e18b7f7c25dcd61c8c64d5bceadc709619dc9290af0bb219.jpg](../icml_results/3604_SINQ_%20Sinkhorn-Normalized%20Quantization%20for%20Calibration-Free%20Low-Precision%20LLM%20Weights/tables/f9bbac40e42acaa0e18b7f7c25dcd61c8c64d5bceadc709619dc9290af0bb219.jpg)
