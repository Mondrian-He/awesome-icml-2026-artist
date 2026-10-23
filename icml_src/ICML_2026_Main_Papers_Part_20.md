# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 20 of 197

## 目录 (Table of Contents)

1. [Adversarial Attacks and Robust Training for Hypergraph Neural Networks](#paper-0628)
2. [Biases in the Blind Spot: Detecting What LLMs Fail to Mention](#paper-0629)
3. [Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](#paper-0630)
4. [LLM Self-Recognition: Steering and Retrieving Activation Signatures](#paper-0631)
5. [A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](#paper-0632)
6. [Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](#paper-0633)
7. [How does information access affect LLM monitors’ ability to detect sabotage?](#paper-0634)
8. [High-Probability Convergence Guarantees of Decentralized SGD](#paper-0635)
9. [Efficient RL Training for LLMs with Experience Replay](#paper-0636)
10. [Rank-Learner: Orthogonal Ranking of Treatment Effects](#paper-0637)
11. [Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage](#paper-0638)
12. [Minimax-Optimal Policy Regret in Partially Observable Markov Games](#paper-0639)
13. [TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](#paper-0640)
14. [Learning Rewrite-Invariant Reasoning with Targeted Alternation Training](#paper-0641)
15. [Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding](#paper-0642)
16. [Exploiting weight-space symmetries for approximating curvature](#paper-0643)
17. [AutoBaxBuilder: Bootstrapping Code Security Benchmarking](#paper-0644)
18. [RedDebate: Safer Responses Through Multi-Agent Red Teaming Debates](#paper-0645)
19. [TRACER: Persistent Regularization for Robust Multimodal Finetuning](#paper-0646)
20. [Latent Reasoning in TRMs is Secretly a Policy Improvement Operator](#paper-0647)
21. [Zero-Shot Off-Policy Learning](#paper-0648)
22. [How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models](#paper-0649)
23. [WeightCLIP: Aligning Datasets and Models for Weight Space Learning](#paper-0650)
24. [Why DDIM Hallucinates More Than DDPM: A Theoretical Analysis of Reverse Dynamics](#paper-0651)
25. [Robust Bayes-Assisted Conformal Prediction](#paper-0652)
26. [The Crowded Embedding Space: A Mean-Field Mechanism for Emergent Marginalization in Retrieval-Augmented Agents](#paper-0653)
27. [Fast and Near-Optimal Algorithms for Private Hypothesis Selection](#paper-0654)
28. [PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs](#paper-0655)
29. [Grounding Functional Similarity by Invariance-Aware Model Stitching](#paper-0656)
30. [Learning Rate Annealing Improves Tuning Robustness in Stochastic Optimization](#paper-0657)
31. [Intrinsic Credit Assignment for Long Horizon Interaction](#paper-0658)
32. [SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs](#paper-0659)
33. [Reducing Per-Sample Interference in Stochastic Optimization](#paper-0660)

---


<a id="paper-0628"></a>

## Adversarial Attacks and Robust Training for Hypergraph Neural Networks

### Images

![1dee4ead6b9e8a6fe7f96ec9006694f1f4d9f8f2925efcb4eb0fad142b25f5d8.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/1dee4ead6b9e8a6fe7f96ec9006694f1f4d9f8f2925efcb4eb0fad142b25f5d8.jpg)

![309eb76dab05dd8f6e83c489b2dd3f8e16ac2c5337ad922a5b146bf4f0f58635.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/309eb76dab05dd8f6e83c489b2dd3f8e16ac2c5337ad922a5b146bf4f0f58635.jpg)

![4a076a7decb533307035abace95b4ace0356ec6ee241fb35a8a52f9a67ab7c5a.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/4a076a7decb533307035abace95b4ace0356ec6ee241fb35a8a52f9a67ab7c5a.jpg)

![64dc4364043fcecc6f3706be67d2c3cf239b203df1cc6da31aa41b4e6d98daff.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/64dc4364043fcecc6f3706be67d2c3cf239b203df1cc6da31aa41b4e6d98daff.jpg)

![ab0a91c5323890f06a0019d878996b69d4a89dfa662ff205192ea54a5e99f515.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/ab0a91c5323890f06a0019d878996b69d4a89dfa662ff205192ea54a5e99f515.jpg)

![b51b67c4acbe5949e5c24d531056ca43ef8325f16ea93c14145423853aa44957.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/b51b67c4acbe5949e5c24d531056ca43ef8325f16ea93c14145423853aa44957.jpg)

![be954963c276087d769afd596b900b35a36e038fba89978a4da5392a2d045c25.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/images/be954963c276087d769afd596b900b35a36e038fba89978a4da5392a2d045c25.jpg)

### Tables

![549b07ad64da44214f65c215ffb9c2afe878a5528fd8d4113b4df613bc5c5a2a.jpg](../icml_results/636_Optimal%20Learning%20from%20Label%20Proportions%20with%20General%20Loss%20Functions/tables/549b07ad64da44214f65c215ffb9c2afe878a5528fd8d4113b4df613bc5c5a2a.jpg)

## Adversarial Attacks and Robust Training for Hypergraph Neural Networks


### Images

![385bc6f5f15d6fc7103bc54c8d5d06a80fd79c8c0d45978c16395171bee5fb40.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/385bc6f5f15d6fc7103bc54c8d5d06a80fd79c8c0d45978c16395171bee5fb40.jpg)

![76140a3d825ee3b6a2d1b8681360f2278042ff5a88b957a38304545c1196aaef.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/76140a3d825ee3b6a2d1b8681360f2278042ff5a88b957a38304545c1196aaef.jpg)

![b383dcc0955a2b6f19b7734e6c44c552ba18f116eac4060c787162c3db6177a0.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/b383dcc0955a2b6f19b7734e6c44c552ba18f116eac4060c787162c3db6177a0.jpg)

![becda45222f14785b39cf0509864893b869af20ab5664ee0c165f235e16c9415.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/becda45222f14785b39cf0509864893b869af20ab5664ee0c165f235e16c9415.jpg)

![eb680641f25969ee24cec45a17d81f83460e86ccd2ab627cb43bca93483c3d94.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/eb680641f25969ee24cec45a17d81f83460e86ccd2ab627cb43bca93483c3d94.jpg)

![fb67c1106a710a6b4b5ed35ef0c1d935ee1312b0f984418880f27cb1e84508e8.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/images/fb67c1106a710a6b4b5ed35ef0c1d935ee1312b0f984418880f27cb1e84508e8.jpg)

### Tables

![01b6409b816f98c594f1da29c5e94ee1e83e46cc3e4c46ffc71dbed6926c4427.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/01b6409b816f98c594f1da29c5e94ee1e83e46cc3e4c46ffc71dbed6926c4427.jpg)

![0b8358dccc3d943ba565e5032cc79626669609593e527ca0ad8cef92de10660b.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/0b8358dccc3d943ba565e5032cc79626669609593e527ca0ad8cef92de10660b.jpg)

![0d054517adb6586421f4f76155a072a77fb7072385286378fd283e05ac0a0664.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/0d054517adb6586421f4f76155a072a77fb7072385286378fd283e05ac0a0664.jpg)

![1a25dbce7a9815eaee785ab71ba10bda991b126c85a9eca48b848d12170d55f2.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/1a25dbce7a9815eaee785ab71ba10bda991b126c85a9eca48b848d12170d55f2.jpg)

![1ea5cd958988ac05b150ecee1ff75c20a1670dd7a761f2c198f26f880824fde9.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/1ea5cd958988ac05b150ecee1ff75c20a1670dd7a761f2c198f26f880824fde9.jpg)

![2301e0203846b50f6c5af37c7d71e527542fe76f8c40aa31aa03a5b4c183c2e4.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/2301e0203846b50f6c5af37c7d71e527542fe76f8c40aa31aa03a5b4c183c2e4.jpg)

![252612a95c14e66cfc6cc92737b478817c9742e8e92876e7a8ed82ee76db8324.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/252612a95c14e66cfc6cc92737b478817c9742e8e92876e7a8ed82ee76db8324.jpg)

![4546e93de0f8678f1d2e8215b3f00bc43964289469989240444bcbb78b53d4af.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/4546e93de0f8678f1d2e8215b3f00bc43964289469989240444bcbb78b53d4af.jpg)

![4788ffe2c237f2381518a6bbdaef5d0e52be8425bd15627558dfab63e746891d.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/4788ffe2c237f2381518a6bbdaef5d0e52be8425bd15627558dfab63e746891d.jpg)

![54e491b41493d955e51263c327ae0988fd5978f0a829c20a3901056d044cd6ab.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/54e491b41493d955e51263c327ae0988fd5978f0a829c20a3901056d044cd6ab.jpg)

![558001e124f9eaa9c24c2c44782099a4a279ad357da0d4c568f690c43805ed70.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/558001e124f9eaa9c24c2c44782099a4a279ad357da0d4c568f690c43805ed70.jpg)

![577293660b30eec9bf68e68872b97d022a6197574841260623afb6a88c1c967b.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/577293660b30eec9bf68e68872b97d022a6197574841260623afb6a88c1c967b.jpg)

![6577e3aeb81770852200f8cd3ee28fdce1f83874a6ab156719a0eb2aa1efb3cd.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/6577e3aeb81770852200f8cd3ee28fdce1f83874a6ab156719a0eb2aa1efb3cd.jpg)

![73ed57fbd15b2c9ea5f2844caeed25e6d332a7faf8127b2cd8c078df9de24e39.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/73ed57fbd15b2c9ea5f2844caeed25e6d332a7faf8127b2cd8c078df9de24e39.jpg)

![9537381b507c9f16e81005dff35e9996a94b1b7619de2c246069c33fba383f39.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/9537381b507c9f16e81005dff35e9996a94b1b7619de2c246069c33fba383f39.jpg)

![bc3228f4490df174b0da164e4eba8f636a8b8a840ac47b6a0b39ddd93764b53f.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/bc3228f4490df174b0da164e4eba8f636a8b8a840ac47b6a0b39ddd93764b53f.jpg)

![d14ecb11fa132077de3ab5270d091c0782ca74f2c5bbf09820752628df09179f.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/d14ecb11fa132077de3ab5270d091c0782ca74f2c5bbf09820752628df09179f.jpg)

![d5f2a8c5b966c1887a097bcb23ba66a8d5823b9bd1c249bb9eb20e8c95dc02db.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/d5f2a8c5b966c1887a097bcb23ba66a8d5823b9bd1c249bb9eb20e8c95dc02db.jpg)

![dfee8ca3dfc6d88d019af473ab92eb7bebd0ac75538df40d40049aacebdcabf2.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/dfee8ca3dfc6d88d019af473ab92eb7bebd0ac75538df40d40049aacebdcabf2.jpg)

![fe5460d2101a03af1fc2040c552740b340b81ea741fd195b30f9739aed2a0711.jpg](../icml_results/637_Adversarial%20Attacks%20and%20Robust%20Training%20for%20Hypergraph%20Neural%20Networks/tables/fe5460d2101a03af1fc2040c552740b340b81ea741fd195b30f9739aed2a0711.jpg)

<a id="paper-0629"></a>

## Biases in the Blind Spot: Detecting What LLMs Fail to Mention


### Images

![3facde6f7df0c3bb13c288914dcac096c2259744967d45b6b0ee1e35335ccc8b.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/images/3facde6f7df0c3bb13c288914dcac096c2259744967d45b6b0ee1e35335ccc8b.jpg)

![7f3bad6cf150c4f85d0e562bc9203c30d85df10d7e218a1cd90eadaaba3895b2.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/images/7f3bad6cf150c4f85d0e562bc9203c30d85df10d7e218a1cd90eadaaba3895b2.jpg)

![c31d5a1d9d377fc031f29eddae8db2b3c465a3f32f0284cff5f20174990cd591.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/images/c31d5a1d9d377fc031f29eddae8db2b3c465a3f32f0284cff5f20174990cd591.jpg)

![ec9522066ab541aa285fa414a4ee443623d07bd3c8f07e9976dff7d902441824.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/images/ec9522066ab541aa285fa414a4ee443623d07bd3c8f07e9976dff7d902441824.jpg)

### Tables

![19f1605a8f7f6fe426e497725f5e36b09023813500d3cf2ede0ad1c65440b941.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/19f1605a8f7f6fe426e497725f5e36b09023813500d3cf2ede0ad1c65440b941.jpg)

![1ac598b02708a27950cd37dc5412fb35cddfa55da5e3a1c6d2e0eaff9424e806.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/1ac598b02708a27950cd37dc5412fb35cddfa55da5e3a1c6d2e0eaff9424e806.jpg)

![25a241ec2025cb9db82d04f531207638d1d375b7a15c5d9e132050a29b17fc27.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/25a241ec2025cb9db82d04f531207638d1d375b7a15c5d9e132050a29b17fc27.jpg)

![2952a6adec864b43bb58d8aaed2def8c5ecf2cf64219e118cbb546cb2b4244f1.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/2952a6adec864b43bb58d8aaed2def8c5ecf2cf64219e118cbb546cb2b4244f1.jpg)

![2c60ffe008aac79a153bafbd22546664dea3a8cae87991e47dd1c9b3e50cae25.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/2c60ffe008aac79a153bafbd22546664dea3a8cae87991e47dd1c9b3e50cae25.jpg)

![381f9fe922826ba7b96aaa04b22c75d8569b9fea4ff06acdcd8212e091da8009.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/381f9fe922826ba7b96aaa04b22c75d8569b9fea4ff06acdcd8212e091da8009.jpg)

![44518004d265a4f423d8675c6e8945d59fc43e3705a6a110ca78d8db6c3c2ac0.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/44518004d265a4f423d8675c6e8945d59fc43e3705a6a110ca78d8db6c3c2ac0.jpg)

![45e0ed018972cdda864d475470d8eac3af10d9a4aa87e2533079a9dbe51a1668.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/45e0ed018972cdda864d475470d8eac3af10d9a4aa87e2533079a9dbe51a1668.jpg)

![49f868ad79d405389214b7da928c0b96dc4fed885bb83be417c7e2b245ea66fb.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/49f868ad79d405389214b7da928c0b96dc4fed885bb83be417c7e2b245ea66fb.jpg)

![555877260349ab73ab96197ddc2743ca2ea718ea6b9da839d7f569015e09423b.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/555877260349ab73ab96197ddc2743ca2ea718ea6b9da839d7f569015e09423b.jpg)

![5f58c5ce1b33c8e9fe8645d36ee20f9a86a12bfe0f7789b7d62e7df121d646c8.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/5f58c5ce1b33c8e9fe8645d36ee20f9a86a12bfe0f7789b7d62e7df121d646c8.jpg)

![61eeb8f03621bc3f481d66a8ce4e9c99cd5f358b3c973f8338bd0e761474a3b4.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/61eeb8f03621bc3f481d66a8ce4e9c99cd5f358b3c973f8338bd0e761474a3b4.jpg)

![66b3360ac4fa01b08c011492f7835bcee15299aa47717ae48fb9a0fc7354f804.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/66b3360ac4fa01b08c011492f7835bcee15299aa47717ae48fb9a0fc7354f804.jpg)

![70bd7d5e59e97bc15149ba8bb1e1c09c2be0dde26805b6953853ca7bf7163b42.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/70bd7d5e59e97bc15149ba8bb1e1c09c2be0dde26805b6953853ca7bf7163b42.jpg)

![7574e610e5fbb3d8c96241c004f69ff449d7f0c9f25d54ce2600f8b55e79803c.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/7574e610e5fbb3d8c96241c004f69ff449d7f0c9f25d54ce2600f8b55e79803c.jpg)

![77aba6ee93e6c5a041293e29d11bd75f2148b320deb74c5cb6e0f178f104904a.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/77aba6ee93e6c5a041293e29d11bd75f2148b320deb74c5cb6e0f178f104904a.jpg)

![923e512f465223b43f8fce3491a788d2ae3158998009deede12180d7c99dd9e4.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/923e512f465223b43f8fce3491a788d2ae3158998009deede12180d7c99dd9e4.jpg)

![9267896b2e5077d37ce0c0b1617eb9153b53da42f8d523ad1b811f7faabf59af.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/9267896b2e5077d37ce0c0b1617eb9153b53da42f8d523ad1b811f7faabf59af.jpg)

![9ac36e8873aa861cd6a80ba04152272da793e87601acf0ca17dcfc543ed45d34.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/9ac36e8873aa861cd6a80ba04152272da793e87601acf0ca17dcfc543ed45d34.jpg)

![a5c1861de6ce73cd76e1232f6899236fa7152f9fdcebadaeeec82a76cb4b57ec.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/a5c1861de6ce73cd76e1232f6899236fa7152f9fdcebadaeeec82a76cb4b57ec.jpg)

![b1d88e5696315b35d8a0fec88091d76b1559225d492eac607f9c5f7cd409f4d4.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/b1d88e5696315b35d8a0fec88091d76b1559225d492eac607f9c5f7cd409f4d4.jpg)

![b442b2aff56f3ba8e21426f62f98aa67289c2d49785edca340ddd3d368dce3da.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/b442b2aff56f3ba8e21426f62f98aa67289c2d49785edca340ddd3d368dce3da.jpg)

![beffc0fe2b61b6a34aaf6644a7c5df7c69981dfb11ae534fb1e0f6cb04e46d28.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/beffc0fe2b61b6a34aaf6644a7c5df7c69981dfb11ae534fb1e0f6cb04e46d28.jpg)

![c0186eabdbc90cb27ec1e5fc6918aba39686f06a47276ef125ec60e249ed8d15.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/c0186eabdbc90cb27ec1e5fc6918aba39686f06a47276ef125ec60e249ed8d15.jpg)

![c192784886a42b775a53e38f1e8f25c0e6c5b16967d1d78179f79fdb2266d372.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/c192784886a42b775a53e38f1e8f25c0e6c5b16967d1d78179f79fdb2266d372.jpg)

![d4257a6f425cc017d50b41b037b1025d223801c6835c33dc57da73d6e08edee9.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/d4257a6f425cc017d50b41b037b1025d223801c6835c33dc57da73d6e08edee9.jpg)

![dc139958b8762625a555d223611c895ee40702aea564282c7e98b4a8ab52604e.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/dc139958b8762625a555d223611c895ee40702aea564282c7e98b4a8ab52604e.jpg)

![eb33c968021ce0636d615e46b8fd7b93fbf7b90344d2b43dc9684c6be725985c.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/eb33c968021ce0636d615e46b8fd7b93fbf7b90344d2b43dc9684c6be725985c.jpg)

![ef84e2f2e406f13be80df916825090e1707bdd3d5c17339b0e1a0333b2cdeed8.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/ef84e2f2e406f13be80df916825090e1707bdd3d5c17339b0e1a0333b2cdeed8.jpg)

![fe75ee116f87622240cc4ab011beba32c3261d19519fae1b19300bd2ddf91e55.jpg](../icml_results/638_Biases%20in%20the%20Blind%20Spot_%20Detecting%20What%20LLMs%20Fail%20to%20Mention/tables/fe75ee116f87622240cc4ab011beba32c3261d19519fae1b19300bd2ddf91e55.jpg)

<a id="paper-0630"></a>

## Chain-of-Thought Reasoning In The Wild Is Not Always Faithful


### Images

![09378109971b73e50286637adb31fbd5de5e6e4cd924d4db77d00c7c883526c1.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/09378109971b73e50286637adb31fbd5de5e6e4cd924d4db77d00c7c883526c1.jpg)

![09a1327f20b63ccad81167711b3340d9858d0115806425d663a3196e9270fd75.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/09a1327f20b63ccad81167711b3340d9858d0115806425d663a3196e9270fd75.jpg)

![1faf70dfb2202e1f9badd6c3a384d9f87b5ac8e40900811e5fee049c92e4192b.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/1faf70dfb2202e1f9badd6c3a384d9f87b5ac8e40900811e5fee049c92e4192b.jpg)

![288970b6a4af9eae65defd66fff67e0162f26b3a3b6b57f0d533d418b1a0c7ff.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/288970b6a4af9eae65defd66fff67e0162f26b3a3b6b57f0d533d418b1a0c7ff.jpg)

![2934f236d71b4081de7fdb72c03323d672f0ce980c52f1dcadd85c322003a63c.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/2934f236d71b4081de7fdb72c03323d672f0ce980c52f1dcadd85c322003a63c.jpg)

![38e3e9b23256b984a8814c252feb96abd70e444e5ea20772c9cb51ef278ac215.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/38e3e9b23256b984a8814c252feb96abd70e444e5ea20772c9cb51ef278ac215.jpg)

![5d44d30d68dcdf27dbdcf2217a31d8ef03c657704788d957d9822dadf9d6f80b.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/5d44d30d68dcdf27dbdcf2217a31d8ef03c657704788d957d9822dadf9d6f80b.jpg)

![61c18421e888122396dc427a32b31cc648d471f8ddc56a59fb617e4c9e2ddf98.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/61c18421e888122396dc427a32b31cc648d471f8ddc56a59fb617e4c9e2ddf98.jpg)

![87ae72ed9d353b6701d8d8ef98223cc12c7ae1074d4053bb795cf97820150d63.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/87ae72ed9d353b6701d8d8ef98223cc12c7ae1074d4053bb795cf97820150d63.jpg)

![bc8e0f97dbbcf14ca4662b7805a80075ee11513ec37545637e29b16661ec3834.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/bc8e0f97dbbcf14ca4662b7805a80075ee11513ec37545637e29b16661ec3834.jpg)

![c0d289da48c85d4a9e1ae04c4dc84fc8a267790b900cf61e47699dcf0ed3bb3b.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/c0d289da48c85d4a9e1ae04c4dc84fc8a267790b900cf61e47699dcf0ed3bb3b.jpg)

![c9334083b9bb5ff86f507a21ee49ec66df61fd318241e4e8e23e4bb97861dff7.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/c9334083b9bb5ff86f507a21ee49ec66df61fd318241e4e8e23e4bb97861dff7.jpg)

![cac2b727bf4fcd9d0678085a5649a9e7bad6658aa045140a661a91451e3c9e57.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/cac2b727bf4fcd9d0678085a5649a9e7bad6658aa045140a661a91451e3c9e57.jpg)

![d00d6e31283b2e61a56dc0bae0654ca1ff781b12e2b8f12128f2c1effc1eaa05.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/d00d6e31283b2e61a56dc0bae0654ca1ff781b12e2b8f12128f2c1effc1eaa05.jpg)

![dd3cb480771974ea5b11545f6ee59a8731ff8dd1de02cbc3c128bc8168e13c17.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/dd3cb480771974ea5b11545f6ee59a8731ff8dd1de02cbc3c128bc8168e13c17.jpg)

![e77a8bc41530abc941420ffa38d0e9b4b267f62a3fb594cad6ae939e286ff08c.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/e77a8bc41530abc941420ffa38d0e9b4b267f62a3fb594cad6ae939e286ff08c.jpg)

![ec9af66cc55e3164bbacd673ce559f992e0ee3d6ea01352ac7f411755da7c9fe.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/images/ec9af66cc55e3164bbacd673ce559f992e0ee3d6ea01352ac7f411755da7c9fe.jpg)

### Tables

![22c8ca9d8efbcede103cb7884f84817c051be8cb700a8d1dd8436d738456029d.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/22c8ca9d8efbcede103cb7884f84817c051be8cb700a8d1dd8436d738456029d.jpg)

![3d1ec521536e19e8e58996e79f69a685d54a28f5c1eaed7b7e22fc21b2a4e870.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/3d1ec521536e19e8e58996e79f69a685d54a28f5c1eaed7b7e22fc21b2a4e870.jpg)

![4eaf937ed734593c2468636fbc9c8db06e5a090fc7d575652a1f0f3342078dc1.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/4eaf937ed734593c2468636fbc9c8db06e5a090fc7d575652a1f0f3342078dc1.jpg)

![5c6a0f991101ecf569cfa5a24673f3fa3e2de3111ba48b0fa9b11ab1e1cc9773.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/5c6a0f991101ecf569cfa5a24673f3fa3e2de3111ba48b0fa9b11ab1e1cc9773.jpg)

![5e8c337038627f4d968fb5fc2d597b2b05d37227d3162db118759611c0b91f8b.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/5e8c337038627f4d968fb5fc2d597b2b05d37227d3162db118759611c0b91f8b.jpg)

![5ffe8837578f310f1a3ccadffdb426ceef3513ad9a1bc09a751dae3ecc0de579.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/5ffe8837578f310f1a3ccadffdb426ceef3513ad9a1bc09a751dae3ecc0de579.jpg)

![62c23aa0c6935aa6cc9942a95cac909691d88e6e67e2743238e017682cc3382a.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/62c23aa0c6935aa6cc9942a95cac909691d88e6e67e2743238e017682cc3382a.jpg)

![70f0fdabf8cc217bdcc61ce77dd18668a52767652e0cd4ebd02b926980d672f5.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/70f0fdabf8cc217bdcc61ce77dd18668a52767652e0cd4ebd02b926980d672f5.jpg)

![76fbf402a75c13c1d2a5a5545803cb55d45ff38f23aabf38cb3372a38576aeb9.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/76fbf402a75c13c1d2a5a5545803cb55d45ff38f23aabf38cb3372a38576aeb9.jpg)

![7c31dc362c4c943687b3799a72e6baefe6c93e57d827ace02cd3449ac107339b.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/7c31dc362c4c943687b3799a72e6baefe6c93e57d827ace02cd3449ac107339b.jpg)

![82bcea57277be2a607b7aa02868f54ff7de10b4ea5303bfa9aed58b8d259a4fe.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/82bcea57277be2a607b7aa02868f54ff7de10b4ea5303bfa9aed58b8d259a4fe.jpg)

![8f1e7108f763b9205a54fa0569f00993dfe5ec7d178d20fa3aca12868e2927b7.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/8f1e7108f763b9205a54fa0569f00993dfe5ec7d178d20fa3aca12868e2927b7.jpg)

![963bbbc7b27a0b756e5a2ea13f65f4b0200d24cd89784a2caca3655c8361b43e.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/963bbbc7b27a0b756e5a2ea13f65f4b0200d24cd89784a2caca3655c8361b43e.jpg)

![979c83712180e73178088d6e0fd1b302ffa37c15b878ed064af588b04f84d68f.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/979c83712180e73178088d6e0fd1b302ffa37c15b878ed064af588b04f84d68f.jpg)

![9f99a6671179b63cdc1a19e4745991baf00b361d8f9948e82db9e86916d87595.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/9f99a6671179b63cdc1a19e4745991baf00b361d8f9948e82db9e86916d87595.jpg)

![a9493fcba19236ab8d7316c53935b1c6d8fa9405d4ba209b8685c37ab2a7ed29.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/a9493fcba19236ab8d7316c53935b1c6d8fa9405d4ba209b8685c37ab2a7ed29.jpg)

![b156872ec3e009b7700466099dc1fbe29dd751eb324385c1f4f1e91832335cf9.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/b156872ec3e009b7700466099dc1fbe29dd751eb324385c1f4f1e91832335cf9.jpg)

![c3e3babb985000fee6507026d339de2850e558a5494872de1aaae770c729ce4e.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/c3e3babb985000fee6507026d339de2850e558a5494872de1aaae770c729ce4e.jpg)

![e9754296725299bf707aa9633f68dd2da6e897c6f6fc1b4d64d38f019d5b00ff.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/e9754296725299bf707aa9633f68dd2da6e897c6f6fc1b4d64d38f019d5b00ff.jpg)

![f366a77398876436604e73e1e48c5d34880ed987a6ca313037fdbc7345987b34.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/f366a77398876436604e73e1e48c5d34880ed987a6ca313037fdbc7345987b34.jpg)

![fd45fa624a257b92df7e7c1cb7c525bf9db6662a564725d1ddbe55a19eb697bc.jpg](../icml_results/639_Chain-of-Thought%20Reasoning%20In%20The%20Wild%20Is%20Not%20Always%20Faithful/tables/fd45fa624a257b92df7e7c1cb7c525bf9db6662a564725d1ddbe55a19eb697bc.jpg)

<a id="paper-0631"></a>

## LLM Self-Recognition: Steering and Retrieving Activation Signatures


### Images

![1608d43c6310661aadd6d81e838d2bec75e5c886c4a379827ed937266e56f634.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/1608d43c6310661aadd6d81e838d2bec75e5c886c4a379827ed937266e56f634.jpg)

![308996cb7ce08a3b284643a56a53d303516cf513489054da4bff06adec0e9905.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/308996cb7ce08a3b284643a56a53d303516cf513489054da4bff06adec0e9905.jpg)

![48e87d484c7996402b8d2b299c2c397b566fbdaee9976a2094fc7e54791e24cd.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/48e87d484c7996402b8d2b299c2c397b566fbdaee9976a2094fc7e54791e24cd.jpg)

![61033a22fff9d2da6eeeaff7c82c78c8aa707711c3ac118cc3ca9e4d780c475e.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/61033a22fff9d2da6eeeaff7c82c78c8aa707711c3ac118cc3ca9e4d780c475e.jpg)

![64342622d932847209f04441747fb653b43d298e48c1911de12fdae92818def6.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/64342622d932847209f04441747fb653b43d298e48c1911de12fdae92818def6.jpg)

![9b0964d35024eb67532ec50d257887025cb2ce08b9036ca039fa94fb86e3362d.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/9b0964d35024eb67532ec50d257887025cb2ce08b9036ca039fa94fb86e3362d.jpg)

![9da4659609bad1de81eb393065df9a47b6595088a111680b4f946f191e5d6f8e.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/9da4659609bad1de81eb393065df9a47b6595088a111680b4f946f191e5d6f8e.jpg)

![fd9e99cdd65c172cd7ad64d7590d2e5adca3b4df2c28962bede68cfcb66751af.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/images/fd9e99cdd65c172cd7ad64d7590d2e5adca3b4df2c28962bede68cfcb66751af.jpg)

### Tables

![018705d7842eb7697335faad1a7c1ab9da12d62c089fcac1d129232e35332c5d.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/018705d7842eb7697335faad1a7c1ab9da12d62c089fcac1d129232e35332c5d.jpg)

![30ea5c8dbfb5ec826b57c8b9fd54fe0709d215ed4447408108af82f41806540b.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/30ea5c8dbfb5ec826b57c8b9fd54fe0709d215ed4447408108af82f41806540b.jpg)

![3216060e2257b150462bb86e26d2945b362c8ce9134917b9de61dbfdec6240cf.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/3216060e2257b150462bb86e26d2945b362c8ce9134917b9de61dbfdec6240cf.jpg)

![3838d77de4cb5c0e421c35ea8a3b6f9f6d285b8fa59be2a40ddf0f7f89c003f0.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/3838d77de4cb5c0e421c35ea8a3b6f9f6d285b8fa59be2a40ddf0f7f89c003f0.jpg)

![58236fd5d4727654c95815219c7131f390076d07f70e58a550607bbb9571a5a0.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/58236fd5d4727654c95815219c7131f390076d07f70e58a550607bbb9571a5a0.jpg)

![676e07dfd22f7009cd4ea877fca38d7c45b13068a0fde11507a74ca7e380e2c7.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/676e07dfd22f7009cd4ea877fca38d7c45b13068a0fde11507a74ca7e380e2c7.jpg)

![9f9892b54d5f8fd679d95d565e0888157e13378225f4a60a669e2ec2574c5cb3.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/9f9892b54d5f8fd679d95d565e0888157e13378225f4a60a669e2ec2574c5cb3.jpg)

![a66127cc65d24f68e587db6d5197f16a964ccbf34ab91aff820ec4b64d30dfdb.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/a66127cc65d24f68e587db6d5197f16a964ccbf34ab91aff820ec4b64d30dfdb.jpg)

![c32f182394758734802a82a53170540e99bfc1ac44ee77286b1ac004d1739f1e.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/c32f182394758734802a82a53170540e99bfc1ac44ee77286b1ac004d1739f1e.jpg)

![e4444ed4947e6198055dbf7579e4878c044aff0abfcf841a564b1807d10c155f.jpg](../icml_results/640_LLM%20Self-Recognition_%20Steering%20and%20Retrieving%20Activation%20Signatures/tables/e4444ed4947e6198055dbf7579e4878c044aff0abfcf841a564b1807d10c155f.jpg)

<a id="paper-0632"></a>

## A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents


### Images

![0754ad1f60dab17b71ac412de5f45e059a6a0eb6e14b442bc3ed7904c1135904.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/0754ad1f60dab17b71ac412de5f45e059a6a0eb6e14b442bc3ed7904c1135904.jpg)

![22e8e5d0b50fc59c343efef72916544bfbd52e0600b96a640de2bcc4c0983d87.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/22e8e5d0b50fc59c343efef72916544bfbd52e0600b96a640de2bcc4c0983d87.jpg)

![2f25a3ceaa9c4901d698c82b1ee88d83f8fdbc7799b4937bd85025a9927c243d.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/2f25a3ceaa9c4901d698c82b1ee88d83f8fdbc7799b4937bd85025a9927c243d.jpg)

![2feb4854abdd11187adb72c83c70f0b34a2832419df7b5aae10590cb9a1b20a3.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/2feb4854abdd11187adb72c83c70f0b34a2832419df7b5aae10590cb9a1b20a3.jpg)

![3411d41568a5c7250d14bd3a7feec8e46f304019b98ec1af36db1dcd218f9d51.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/3411d41568a5c7250d14bd3a7feec8e46f304019b98ec1af36db1dcd218f9d51.jpg)

![3757f9bf33ae3b0a3d5e80d720a43c29ca76534971dd9f4ac8e8c2556600b33b.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/3757f9bf33ae3b0a3d5e80d720a43c29ca76534971dd9f4ac8e8c2556600b33b.jpg)

![3beecb6c94f2a8745f686c83dba560826018dd9f9b18d7cb5fb472854c1007ec.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/3beecb6c94f2a8745f686c83dba560826018dd9f9b18d7cb5fb472854c1007ec.jpg)

![4854fe75bb3e522013192d331e3cf868fc2df96411a07b5072724fcb4c30abcb.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/4854fe75bb3e522013192d331e3cf868fc2df96411a07b5072724fcb4c30abcb.jpg)

![4c4ea80a9670aaddbb72cea6c3d94628744e46cf3dd7794e89b9f089b8134a0e.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/4c4ea80a9670aaddbb72cea6c3d94628744e46cf3dd7794e89b9f089b8134a0e.jpg)

![7bd5c6f9395b7d39fe17221adabc5173de7805808b0324965fb78674178105a1.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/7bd5c6f9395b7d39fe17221adabc5173de7805808b0324965fb78674178105a1.jpg)

![86b8220551b99ef2983317c85ffaf0c7b4e354b0c3c6550ca922783da2bd7535.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/86b8220551b99ef2983317c85ffaf0c7b4e354b0c3c6550ca922783da2bd7535.jpg)

![a15225f0b3437756833e7aa8c6b0765f9252b77dcb523e987f484a8e9429eceb.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/a15225f0b3437756833e7aa8c6b0765f9252b77dcb523e987f484a8e9429eceb.jpg)

![caea8769a80c5fdf58380dbf03f9c754bd004b37cebcab1f8245c76d92d1e301.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/caea8769a80c5fdf58380dbf03f9c754bd004b37cebcab1f8245c76d92d1e301.jpg)

![d265e8f272da2289c098e8efa4780f0e0b727ef95633b8af6abf496b283e8aee.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/d265e8f272da2289c098e8efa4780f0e0b727ef95633b8af6abf496b283e8aee.jpg)

![df72138a1fd28b624f15d94c93772bc59c69efdcb4619ebc1ea52678ef73afc7.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/df72138a1fd28b624f15d94c93772bc59c69efdcb4619ebc1ea52678ef73afc7.jpg)

![fccbd992090983ed6fc631731906a53030b28c45306c66268305ebc601902b65.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/images/fccbd992090983ed6fc631731906a53030b28c45306c66268305ebc601902b65.jpg)

### Tables

![019e4646f4d96f7e558d64bd6cd48b7fc9b144338a6e3b3dbdde4628df60e350.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/019e4646f4d96f7e558d64bd6cd48b7fc9b144338a6e3b3dbdde4628df60e350.jpg)

![0b6e097711b07133ba5047308751f56e5a50f8f83704d16992eea60e9f937d07.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/0b6e097711b07133ba5047308751f56e5a50f8f83704d16992eea60e9f937d07.jpg)

![2618f35e9ccc24bb9391daeac39e01bb08de93ae9f6e5706cf6896bc1eec7f44.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/2618f35e9ccc24bb9391daeac39e01bb08de93ae9f6e5706cf6896bc1eec7f44.jpg)

![5b605f1211273b3aee54aabbc018e86a7284024400e25a4985fac17465bd2828.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/5b605f1211273b3aee54aabbc018e86a7284024400e25a4985fac17465bd2828.jpg)

![a4fd908e99a33dd0dcc3766107706f7e71d97e172f55bbbdd980b0b21febc839.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/a4fd908e99a33dd0dcc3766107706f7e71d97e172f55bbbdd980b0b21febc839.jpg)

![d2f49d491a4d5ac133017e63c7a92af0a7e5d11b0cb31c47fd79406c3eb61207.jpg](../icml_results/641_A%20Behavioural%20and%20Representational%20Evaluation%20of%20Goal-Directedness%20in%20Language%20Model%20Agents/tables/d2f49d491a4d5ac133017e63c7a92af0a7e5d11b0cb31c47fd79406c3eb61207.jpg)

<a id="paper-0633"></a>

## Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining


### Images

![078670fc6f5ac5b91ce81b98f9f25a4ec9dd5e9f9b5932b94f1715a83ebe0249.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/078670fc6f5ac5b91ce81b98f9f25a4ec9dd5e9f9b5932b94f1715a83ebe0249.jpg)

![1fdf1bff79dce5e792e77c5df861dabc1d22cd454f96ee608551adc075e3de8c.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/1fdf1bff79dce5e792e77c5df861dabc1d22cd454f96ee608551adc075e3de8c.jpg)

![3ea662c375b2496db50b2eefda9aa5e4591d98f4bc7fa3cfdae110c85f27d590.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/3ea662c375b2496db50b2eefda9aa5e4591d98f4bc7fa3cfdae110c85f27d590.jpg)

![99d1a4bcdc02610805c38ca78fecbe7e134d13ea04763cfca66992a63e9e618d.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/99d1a4bcdc02610805c38ca78fecbe7e134d13ea04763cfca66992a63e9e618d.jpg)

![bc3d498f55e99a85bd91c3f07b6942b04bd288b27f260e431651965851c34e45.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/bc3d498f55e99a85bd91c3f07b6942b04bd288b27f260e431651965851c34e45.jpg)

![d1c8542274427c992cd5664d94baa993ebc32e0d34689946c49730fc9ed3e03f.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/d1c8542274427c992cd5664d94baa993ebc32e0d34689946c49730fc9ed3e03f.jpg)

![f6bdfff3e84cf31aa234362d53c3f1e08a821050f0cbb1d545e5d7db83698d83.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/images/f6bdfff3e84cf31aa234362d53c3f1e08a821050f0cbb1d545e5d7db83698d83.jpg)

### Tables

![3022f3bd9ffc21f854fbde6390b74299f1da432a9a9d6efcc56879bf4cd55858.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/3022f3bd9ffc21f854fbde6390b74299f1da432a9a9d6efcc56879bf4cd55858.jpg)

![3e9055957f947f21f7a4fa2f9a8e62fe41b4e998174dbf99fd85e04f2dfba2a8.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/3e9055957f947f21f7a4fa2f9a8e62fe41b4e998174dbf99fd85e04f2dfba2a8.jpg)

![3fe2c1894bf901602314b162ef0b25d55a53650003fbe333b0b6a6e66ef5f303.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/3fe2c1894bf901602314b162ef0b25d55a53650003fbe333b0b6a6e66ef5f303.jpg)

![5ce8be22dc49d0b58d1ff76858620f2802eb2a90546415c8b58d67573eb45b03.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/5ce8be22dc49d0b58d1ff76858620f2802eb2a90546415c8b58d67573eb45b03.jpg)

![61b09fdbfb4ec8a839955981d35403fbb10a4d4557e4d97c81d70425a106906d.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/61b09fdbfb4ec8a839955981d35403fbb10a4d4557e4d97c81d70425a106906d.jpg)

![8019e03301048f9cb96111697715363e6667db748987ef2428f5946281015338.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/8019e03301048f9cb96111697715363e6667db748987ef2428f5946281015338.jpg)

![8bc70c81ba5dd7fee166bc82d875a217e6c7a5ba1d04010bb1ba196cd25a475d.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/8bc70c81ba5dd7fee166bc82d875a217e6c7a5ba1d04010bb1ba196cd25a475d.jpg)

![b417a47c6ae0f23ef81049240702c3f7e88454915fe3829db64040dfeebaea16.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/b417a47c6ae0f23ef81049240702c3f7e88454915fe3829db64040dfeebaea16.jpg)

![b48f701ffe37e4c01f28321cee85f24bb3754b015140a200b7efbd5f71342677.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/b48f701ffe37e4c01f28321cee85f24bb3754b015140a200b7efbd5f71342677.jpg)

![b52676a900e0c7d11ff858c761c5f0082e57388727611499dd7313b05b482192.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/b52676a900e0c7d11ff858c761c5f0082e57388727611499dd7313b05b482192.jpg)

![c2720e35ec07290885ca3000824b06e2c8f76f4d352a65677d65f864d661f305.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/c2720e35ec07290885ca3000824b06e2c8f76f4d352a65677d65f864d661f305.jpg)

![d0d31e6daaa2ec6a2a33de1e23427da5544f22adbf8610d29bc22b678f7ba446.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/d0d31e6daaa2ec6a2a33de1e23427da5544f22adbf8610d29bc22b678f7ba446.jpg)

![d6f5f2dcbd4f43a9994c17a71796be27420c503cba5f6daa341c2b5a62fcc823.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/d6f5f2dcbd4f43a9994c17a71796be27420c503cba5f6daa341c2b5a62fcc823.jpg)

![d73d8a97ffa2c97e60cdcd7d9c02319c877489c70ab408f074b2b1277e85af40.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/d73d8a97ffa2c97e60cdcd7d9c02319c877489c70ab408f074b2b1277e85af40.jpg)

![f752b91e1c477b3d31ec480886966c916c01b60922d35295ce6459e2e8ea044a.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/f752b91e1c477b3d31ec480886966c916c01b60922d35295ce6459e2e8ea044a.jpg)

![fa1363db813409943bb8535e387ad51eb36c1e422bd8a67f1c5a552e563ffa96.jpg](../icml_results/642_Learning%20the%20Neighborhood_%20Contrast-Free%20Multimodal%20Self-Supervised%20Molecular%20Graph%20Pretraining/tables/fa1363db813409943bb8535e387ad51eb36c1e422bd8a67f1c5a552e563ffa96.jpg)

<a id="paper-0634"></a>

## How does information access affect LLM monitors’ ability to detect sabotage?


### Images

![0299cff7b62bcacd67f1708181dbbecbdf5fcd1c7e14318ca1e6160e1d3beecd.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/0299cff7b62bcacd67f1708181dbbecbdf5fcd1c7e14318ca1e6160e1d3beecd.jpg)

![047c23b738376d75880b852281d4e3b041ce77221df3f472e434452a360c0652.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/047c23b738376d75880b852281d4e3b041ce77221df3f472e434452a360c0652.jpg)

![07e3873b8b939a6e5c75770dc0107a4001648da7f6ec234103711048b302245c.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/07e3873b8b939a6e5c75770dc0107a4001648da7f6ec234103711048b302245c.jpg)

![0c6422042f4490c947da85b4494bac947678b53854104764c2ea12570051d9f2.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/0c6422042f4490c947da85b4494bac947678b53854104764c2ea12570051d9f2.jpg)

![19d7dac81270b8a2548c10f6a073d2a6daec960bdc57ae436acb39607df273e0.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/19d7dac81270b8a2548c10f6a073d2a6daec960bdc57ae436acb39607df273e0.jpg)

![2a2d4b039df7cbff1487173473443ac64adc7024b313b71b2c9a29bef0414fd4.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/2a2d4b039df7cbff1487173473443ac64adc7024b313b71b2c9a29bef0414fd4.jpg)

![2c669c4098ad264a0c1bc1e4de604f593b4506554f27dc3fd8d818b306b50512.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/2c669c4098ad264a0c1bc1e4de604f593b4506554f27dc3fd8d818b306b50512.jpg)

![2dd46f555442c5991b3d2187e7c231422d46dba2d70c9344fad5bc17dbdea865.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/2dd46f555442c5991b3d2187e7c231422d46dba2d70c9344fad5bc17dbdea865.jpg)

![2ee55dcee51ca355db6fab4dfaa29f555b682973c10df6d4f364eeab8c4f99e1.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/2ee55dcee51ca355db6fab4dfaa29f555b682973c10df6d4f364eeab8c4f99e1.jpg)

![363864f634d879052381b8b4d5c2ff62217394b64c2ca27b5ad86aa79f451d5f.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/363864f634d879052381b8b4d5c2ff62217394b64c2ca27b5ad86aa79f451d5f.jpg)

![3d3bf9d7242000244986b4cd117d63a92075e03f84156304134e04929cddc70b.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/3d3bf9d7242000244986b4cd117d63a92075e03f84156304134e04929cddc70b.jpg)

![48b2a337a1e5d14afab0d60074797ab124cd2eb48889dcc443987c164a4f61e1.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/48b2a337a1e5d14afab0d60074797ab124cd2eb48889dcc443987c164a4f61e1.jpg)

![4cff65576afa686ab2ef3d4d048d2f4094319f55c20f70248399a76dd0d43586.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/4cff65576afa686ab2ef3d4d048d2f4094319f55c20f70248399a76dd0d43586.jpg)

![501b8a89743d82cde2942a1e712c97c49465d76828dc05b0d37262bb9de9b857.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/501b8a89743d82cde2942a1e712c97c49465d76828dc05b0d37262bb9de9b857.jpg)

![55da15df2866c8540405eff6f85abc95e65c546e731ad9219c35c3f5aa05d2e7.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/55da15df2866c8540405eff6f85abc95e65c546e731ad9219c35c3f5aa05d2e7.jpg)

![59240564d850493d36bf29b59ec78495ae54bbbe5af25f70e0403e31775698b8.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/59240564d850493d36bf29b59ec78495ae54bbbe5af25f70e0403e31775698b8.jpg)

![5d7c0559ed213d91f026cd643e26e3d77d90c1350d5a716e33797fd9d496a56c.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/5d7c0559ed213d91f026cd643e26e3d77d90c1350d5a716e33797fd9d496a56c.jpg)

![6f7eb7c52fb8e88c996c18f291083f47428c5dc253d6f30f8ebc90d19f7873c0.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/6f7eb7c52fb8e88c996c18f291083f47428c5dc253d6f30f8ebc90d19f7873c0.jpg)

![78f3bc37797ef2aa49dee3cd483332dce80f187549077d71818e36f36e01c44a.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/78f3bc37797ef2aa49dee3cd483332dce80f187549077d71818e36f36e01c44a.jpg)

![8dfdfeb74bd81f3c6e99fb88e403b010151e4b5d28a4ef9ea38e50acfb03f181.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/8dfdfeb74bd81f3c6e99fb88e403b010151e4b5d28a4ef9ea38e50acfb03f181.jpg)

![90d07af4fbcb1cd2a5fed51aff10817d7947fc5c6520ab31e3dde7d192d89757.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/90d07af4fbcb1cd2a5fed51aff10817d7947fc5c6520ab31e3dde7d192d89757.jpg)

![91a63354737767add1690e75353097e33d8b61fe057af9c44eb4421b4173ea0f.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/91a63354737767add1690e75353097e33d8b61fe057af9c44eb4421b4173ea0f.jpg)

![94e4f2d859242e69804e68e2fddc96233ec4f2d7dcc87c8499b3471384724479.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/94e4f2d859242e69804e68e2fddc96233ec4f2d7dcc87c8499b3471384724479.jpg)

![9c9bb1cab1f752071aa6be36a3e0837d5a676450f3140fa5c0f79bbcdd6e405f.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/9c9bb1cab1f752071aa6be36a3e0837d5a676450f3140fa5c0f79bbcdd6e405f.jpg)

![a1076cfc3b2ebde4b819e37777754a8199d89632ed3bdd09a8298af03fc6423a.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/a1076cfc3b2ebde4b819e37777754a8199d89632ed3bdd09a8298af03fc6423a.jpg)

![b33a77015dd3c6c7a1a7d58993dca2a35cce48ea85845ca360b04d06b6d0a6b0.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/b33a77015dd3c6c7a1a7d58993dca2a35cce48ea85845ca360b04d06b6d0a6b0.jpg)

![b7f75de01d3ec75a83c0eb9acbcc4a4dc92e5098b412e1e344f17c0d25009499.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/b7f75de01d3ec75a83c0eb9acbcc4a4dc92e5098b412e1e344f17c0d25009499.jpg)

![c1f65bca410a3fd5ecb0033048d3fb41d96c293535c34b48b68b6fae2fe7b992.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/c1f65bca410a3fd5ecb0033048d3fb41d96c293535c34b48b68b6fae2fe7b992.jpg)

![ccc86166cfdf1593fbdf5c15922b00526e11d852f4ab7d05b1a34abad8297b23.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/ccc86166cfdf1593fbdf5c15922b00526e11d852f4ab7d05b1a34abad8297b23.jpg)

![cfc58ab63c913d1f32f3408dabc2ea1d6a839a8708bfa170e53ba592fda75420.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/cfc58ab63c913d1f32f3408dabc2ea1d6a839a8708bfa170e53ba592fda75420.jpg)

![d01536a7669fd6186b32621b488c8da288ff372a8c0e86203a8ec36c6540b1a5.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d01536a7669fd6186b32621b488c8da288ff372a8c0e86203a8ec36c6540b1a5.jpg)

![d5d7c59d0b76e02c8871ee862d4f8b2ab3eb3eedda31115dc0de29f7b266d006.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d5d7c59d0b76e02c8871ee862d4f8b2ab3eb3eedda31115dc0de29f7b266d006.jpg)

![d6841eed515c72e033db04f10a306e4a38d6958c92a46b95076c24d62f55dec3.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d6841eed515c72e033db04f10a306e4a38d6958c92a46b95076c24d62f55dec3.jpg)

![d7d9fb7dd53a8f0a38f7afbd28f52740c17523c9e295c65cf0c7faf44e599869.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d7d9fb7dd53a8f0a38f7afbd28f52740c17523c9e295c65cf0c7faf44e599869.jpg)

![d8260328875850fb1da8cb2ef844216f2343354f140814bb72c9e104b4fa65a4.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d8260328875850fb1da8cb2ef844216f2343354f140814bb72c9e104b4fa65a4.jpg)

![d9dfb552a5ad4ed84f4bdb5e3779835afc0d07a9dd51a138b7a962b2f48282f5.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/d9dfb552a5ad4ed84f4bdb5e3779835afc0d07a9dd51a138b7a962b2f48282f5.jpg)

![e14cef7c91d38d798e1fb795867aee865019ff32649a382b7d0bf1fc24d9187b.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/e14cef7c91d38d798e1fb795867aee865019ff32649a382b7d0bf1fc24d9187b.jpg)

![fca0462afb2647e66e0e44a059fb5bde2ecbe0574c01836130873de707171739.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/fca0462afb2647e66e0e44a059fb5bde2ecbe0574c01836130873de707171739.jpg)

![fd9869273de96b5b868e17919362c0fc82973a6f92c3003f6859f9e0c949d9ef.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/images/fd9869273de96b5b868e17919362c0fc82973a6f92c3003f6859f9e0c949d9ef.jpg)

### Tables

![01fc4945ba5b7c73d5ac103b170b0e63c2f51ea95037e7996b887996cd651562.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/01fc4945ba5b7c73d5ac103b170b0e63c2f51ea95037e7996b887996cd651562.jpg)

![0c70a6ea4fcf77375803ff062d267dbbcbbd207115a5b17f1df58d8c36d02a81.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/0c70a6ea4fcf77375803ff062d267dbbcbbd207115a5b17f1df58d8c36d02a81.jpg)

![16ce678acadcecb1edeb96bdcdda903160955ebf775d39adbab0bc78d606e8b3.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/16ce678acadcecb1edeb96bdcdda903160955ebf775d39adbab0bc78d606e8b3.jpg)

![2d239dbb8fd99c866507c226c4702156c6725f6cd11f9237338f7584b119c6a4.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/2d239dbb8fd99c866507c226c4702156c6725f6cd11f9237338f7584b119c6a4.jpg)

![62c6c50917e5edc737114048ac02e8f8954d6406213ff062740d0bbf097ff5ff.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/62c6c50917e5edc737114048ac02e8f8954d6406213ff062740d0bbf097ff5ff.jpg)

![93f4dfc46b8214e10345012c929d3bd8b5c5109e903108ee30a6c4b54849bfd4.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/93f4dfc46b8214e10345012c929d3bd8b5c5109e903108ee30a6c4b54849bfd4.jpg)

![b5579784b69c627df66395542cf8367c8403bb146b43cfa2b59cbf38c9e6e9f3.jpg](../icml_results/643_How%20does%20information%20access%20affect%20LLM%20monitors%E2%80%99%20ability%20to%20detect%20sabotage_/tables/b5579784b69c627df66395542cf8367c8403bb146b43cfa2b59cbf38c9e6e9f3.jpg)

<a id="paper-0635"></a>

## High-Probability Convergence Guarantees of Decentralized SGD


### Images

![3f5048506f0b68a264a96e6f839ff1b9178640c95701c02700ecf86af2598006.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/3f5048506f0b68a264a96e6f839ff1b9178640c95701c02700ecf86af2598006.jpg)

![49cd23d0e025edb3256c3e7b5a38f1686955d458cc9df7052af368c40bb81ad8.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/49cd23d0e025edb3256c3e7b5a38f1686955d458cc9df7052af368c40bb81ad8.jpg)

![7465ba798523a44b9fc27e3ed38a660fd8bcdbb3298762b0c5186062048a7825.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/7465ba798523a44b9fc27e3ed38a660fd8bcdbb3298762b0c5186062048a7825.jpg)

![ad3ac8342df082284dfb65234f756429c8601406b19c0ae131caefcc57bb08ec.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/ad3ac8342df082284dfb65234f756429c8601406b19c0ae131caefcc57bb08ec.jpg)

![b4c71acdcf2c8caf9e25b2fe90ad281f386cabf7091ce86b2f527589946eae57.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/b4c71acdcf2c8caf9e25b2fe90ad281f386cabf7091ce86b2f527589946eae57.jpg)

![c57d6f2157040e549a9516171047769948403d6b921e8bad13980cdc412d5aae.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/images/c57d6f2157040e549a9516171047769948403d6b921e8bad13980cdc412d5aae.jpg)

### Tables

![4b4e994a65da9f333227d1e095389d8de43b03283a9b273e2687566f0efdb7b2.jpg](../icml_results/644_High-Probability%20Convergence%20Guarantees%20of%20Decentralized%20SGD/tables/4b4e994a65da9f333227d1e095389d8de43b03283a9b273e2687566f0efdb7b2.jpg)

<a id="paper-0636"></a>

## Efficient RL Training for LLMs with Experience Replay


### Images

![0fe13d615a130e7255b26baa5d5086ad629a69f245f59cd5ba0fbde3b68ea73d.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/0fe13d615a130e7255b26baa5d5086ad629a69f245f59cd5ba0fbde3b68ea73d.jpg)

![10df771437114a716ca692b39ad774b82cc89bffbf0b70deffa6b40238ee1fc2.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/10df771437114a716ca692b39ad774b82cc89bffbf0b70deffa6b40238ee1fc2.jpg)

![25c6ccef84b3906cca842164382f1c7ac0e482e637535578673c6aa23198bd71.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/25c6ccef84b3906cca842164382f1c7ac0e482e637535578673c6aa23198bd71.jpg)

![435ea6ca09855de0a00126679735165db578fe83994c605754e60b2ae592a5aa.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/435ea6ca09855de0a00126679735165db578fe83994c605754e60b2ae592a5aa.jpg)

![44eb24132db7732bac7b94d62573a19ccde586f4ae30827034e994a273032502.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/44eb24132db7732bac7b94d62573a19ccde586f4ae30827034e994a273032502.jpg)

![47fd161e590970062a9a9019f68cbb93549b8bbfa67f1548f08c1062ac97c258.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/47fd161e590970062a9a9019f68cbb93549b8bbfa67f1548f08c1062ac97c258.jpg)

![49288aa69ef111eac783ab99b1af50e05b17fc5bc17ffec0129e8ce032840cb7.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/49288aa69ef111eac783ab99b1af50e05b17fc5bc17ffec0129e8ce032840cb7.jpg)

![720a7250e7d11c7d3a1044512cdb6e03fbd91fabed8d036a538cbc2bd5af0b6d.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/720a7250e7d11c7d3a1044512cdb6e03fbd91fabed8d036a538cbc2bd5af0b6d.jpg)

![7f787928187518577d604b6fdc3ef065543b07bfcc5ecbccb02151fa7d86cc42.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/7f787928187518577d604b6fdc3ef065543b07bfcc5ecbccb02151fa7d86cc42.jpg)

![8f1e3d0ab54f2e3e1f7edfcf614dc19338d00754ac5e6a9780f5e3201893ca27.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/8f1e3d0ab54f2e3e1f7edfcf614dc19338d00754ac5e6a9780f5e3201893ca27.jpg)

![afaa18d809b4195369e7929a066c5692299d74f4d6529a6bfa0be2dc23946c9a.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/afaa18d809b4195369e7929a066c5692299d74f4d6529a6bfa0be2dc23946c9a.jpg)

![b6ed5541b457491a50659d47ec883477d3c2e56ca6fd07d2d13474633a858fb7.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/b6ed5541b457491a50659d47ec883477d3c2e56ca6fd07d2d13474633a858fb7.jpg)

![b8b4449e4f5d1448d061f17b4445480b6fb528a206c8f2dcb36794a49b0c9372.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/b8b4449e4f5d1448d061f17b4445480b6fb528a206c8f2dcb36794a49b0c9372.jpg)

![bfa71159a68071996cb77f47067b5dc35755d184c8355c33c8c9eb80083234d7.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/bfa71159a68071996cb77f47067b5dc35755d184c8355c33c8c9eb80083234d7.jpg)

![d4e480b1488d3584f4d5cb6277a5b94f955c12b1693f514129c3e81c6f57704b.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/d4e480b1488d3584f4d5cb6277a5b94f955c12b1693f514129c3e81c6f57704b.jpg)

![dd0b85e3f3520ab1411a8b7d041e9d5c8682c5b40995764535924c6f1a6ec109.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/dd0b85e3f3520ab1411a8b7d041e9d5c8682c5b40995764535924c6f1a6ec109.jpg)

![f48bb47527b0a7181bf53c27af999ea194a7dfcb229b77f0be7c0a9f64399b47.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/f48bb47527b0a7181bf53c27af999ea194a7dfcb229b77f0be7c0a9f64399b47.jpg)

![f76d46ecfa7e9799cbd42c29aa2001e8d769523e550f481455093069e5f7cda7.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/images/f76d46ecfa7e9799cbd42c29aa2001e8d769523e550f481455093069e5f7cda7.jpg)

### Tables

![18c367362033e710767289d56e4a39e5d7a9e2e2fa62d3fc3f4ff3d40b760386.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/tables/18c367362033e710767289d56e4a39e5d7a9e2e2fa62d3fc3f4ff3d40b760386.jpg)

![92ec90fac0a8e1c9728e80897800b71dedce7a4d13cb079b0dc55e9edbf3f65b.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/tables/92ec90fac0a8e1c9728e80897800b71dedce7a4d13cb079b0dc55e9edbf3f65b.jpg)

![f95e751b08a3e199b1db2ef1ad693fb5fe945b4da70a41ce646a1270c3c3f1e8.jpg](../icml_results/645_Efficient%20RL%20Training%20for%20LLMs%20with%20Experience%20Replay/tables/f95e751b08a3e199b1db2ef1ad693fb5fe945b4da70a41ce646a1270c3c3f1e8.jpg)

<a id="paper-0637"></a>

## Rank-Learner: Orthogonal Ranking of Treatment Effects


### Images

![1c5bb6c108ade180a912097fd7957e9d738b72207c0388b46aab0960cd1f610a.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/1c5bb6c108ade180a912097fd7957e9d738b72207c0388b46aab0960cd1f610a.jpg)

![24c45000c0ef1583af22851f943d2c56c7f43da46b78e02b3607524db792eb7f.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/24c45000c0ef1583af22851f943d2c56c7f43da46b78e02b3607524db792eb7f.jpg)

![44afb93a8fefe6523b278a320210dfa3ac7747f46f7f00cf17c264cabc74515b.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/44afb93a8fefe6523b278a320210dfa3ac7747f46f7f00cf17c264cabc74515b.jpg)

![5d4a0bfd138c0a2d049b9fdad788e4ddc40005cd0a3ef3803cdbc04a277159fe.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/5d4a0bfd138c0a2d049b9fdad788e4ddc40005cd0a3ef3803cdbc04a277159fe.jpg)

![663c55902d8f2af89ec4b8fefc72790fd512adcf5b3defcc26fcc7bc9fed7fb5.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/663c55902d8f2af89ec4b8fefc72790fd512adcf5b3defcc26fcc7bc9fed7fb5.jpg)

![6e628f3e9b09639c89b03c3d132a2f3828caa50e62d60a39c38290b312323341.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/6e628f3e9b09639c89b03c3d132a2f3828caa50e62d60a39c38290b312323341.jpg)

![a7b56ed575b9de4eb6abe7e6f7e305f020fc8eabb29330ad62def9af7781ec03.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/a7b56ed575b9de4eb6abe7e6f7e305f020fc8eabb29330ad62def9af7781ec03.jpg)

![d02c800d31e2ee0e6007e8b8c215f642188e633c951cc60c97980f2c53951d31.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/images/d02c800d31e2ee0e6007e8b8c215f642188e633c951cc60c97980f2c53951d31.jpg)

### Tables

![05c04b4df0d5a3db45229da0017402be5887f1bfd53343e885b909a2730e619a.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/05c04b4df0d5a3db45229da0017402be5887f1bfd53343e885b909a2730e619a.jpg)

![1f213eaa104eea3629db44e9c3a2ff3646b12d2e0566fa9c8d3f6ec0da215277.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/1f213eaa104eea3629db44e9c3a2ff3646b12d2e0566fa9c8d3f6ec0da215277.jpg)

![3510868027cea59fa9c09c50663cd3046c4c3846cb2000da7db5aa4e3833b888.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/3510868027cea59fa9c09c50663cd3046c4c3846cb2000da7db5aa4e3833b888.jpg)

![3d782f2e2559cdd38bde041cd67c6c269e7c01d6d35affc8128637e8b4187f2b.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/3d782f2e2559cdd38bde041cd67c6c269e7c01d6d35affc8128637e8b4187f2b.jpg)

![3e5f87c4d1cf21fb7522c937a68e5fd1e08c2ec044f0ef7e7f972f10ad1dedf5.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/3e5f87c4d1cf21fb7522c937a68e5fd1e08c2ec044f0ef7e7f972f10ad1dedf5.jpg)

![8013aae8f6fd3ddd00ba2a0d1b7d34795ffcc3d147b94a20c5e71a905bb64e40.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/8013aae8f6fd3ddd00ba2a0d1b7d34795ffcc3d147b94a20c5e71a905bb64e40.jpg)

![88458cbdcc2fd1b0e4c4a35b4ec619023850e80a1a8eebf6cc1d0dcb31d3eea0.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/88458cbdcc2fd1b0e4c4a35b4ec619023850e80a1a8eebf6cc1d0dcb31d3eea0.jpg)

![a81538ec065e54322143f4618ea7b1506b0c2f4dc8121d720efffe59f978b7d5.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/a81538ec065e54322143f4618ea7b1506b0c2f4dc8121d720efffe59f978b7d5.jpg)

![bbb911649d2a1f07ee3e31987e489430c3e48164b4e64a5c497fbcfcb39f2428.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/bbb911649d2a1f07ee3e31987e489430c3e48164b4e64a5c497fbcfcb39f2428.jpg)

![bee22362f32e75606c4b9fbda45ebef9e867f26ac37083e9a54474479c59b2fd.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/bee22362f32e75606c4b9fbda45ebef9e867f26ac37083e9a54474479c59b2fd.jpg)

![c4865a89c15e4eb18813aefd963d1d323a69e2c3be16e1caeee5d4fec2416553.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/c4865a89c15e4eb18813aefd963d1d323a69e2c3be16e1caeee5d4fec2416553.jpg)

![d71e0245f354cb7d7ba3094bc1fd43ed8c8240c38eb7d276b7413fcd991bfa59.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/d71e0245f354cb7d7ba3094bc1fd43ed8c8240c38eb7d276b7413fcd991bfa59.jpg)

![f421e3d898b7d99523fe95fb64fe2c09443fdecf39d12fc58e68845f2b5e20dc.jpg](../icml_results/646_Rank-Learner_%20Orthogonal%20Ranking%20of%20Treatment%20Effects/tables/f421e3d898b7d99523fe95fb64fe2c09443fdecf39d12fc58e68845f2b5e20dc.jpg)

<a id="paper-0638"></a>

## Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage


### Images

![066bfa3899c995d468dadd1a628083082aa266b565cc5ed7d2edc62c8ccd21a2.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/images/066bfa3899c995d468dadd1a628083082aa266b565cc5ed7d2edc62c8ccd21a2.jpg)

![113f8715d5a6791bad4ca4bf54c0d47c378d5f65c13162f2cef7eff517421c76.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/images/113f8715d5a6791bad4ca4bf54c0d47c378d5f65c13162f2cef7eff517421c76.jpg)

![b33a804c9ba5774faf5b0ec5ec44cbb3cc451b538e378bb9e6af5ebc20b0a1f3.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/images/b33a804c9ba5774faf5b0ec5ec44cbb3cc451b538e378bb9e6af5ebc20b0a1f3.jpg)

![bc928a2d82810125b0ea75971f3e2c082ad213455b870f6c705cf58d13a6734a.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/images/bc928a2d82810125b0ea75971f3e2c082ad213455b870f6c705cf58d13a6734a.jpg)

### Tables

![0f05aec84de1bb6b671ea56ca48d7a2eac1027297f47bf23239ec095e1eaa548.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/0f05aec84de1bb6b671ea56ca48d7a2eac1027297f47bf23239ec095e1eaa548.jpg)

![0fbea0bdf3420a616807274517a6f285b336b2b959e296df9dfe1e4010c6dffa.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/0fbea0bdf3420a616807274517a6f285b336b2b959e296df9dfe1e4010c6dffa.jpg)

![12fa9d556ade20496c7d9c1630944700de96d4fdba15e393401154f849244939.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/12fa9d556ade20496c7d9c1630944700de96d4fdba15e393401154f849244939.jpg)

![1c949002536f68bb5d3f841ab6cf0d00f1cf3847d0f35a59a8949256178bf96b.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/1c949002536f68bb5d3f841ab6cf0d00f1cf3847d0f35a59a8949256178bf96b.jpg)

![257d143eddbf0916287b8d0aea04aa0cf4019221c8a3f49a372442508240afb9.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/257d143eddbf0916287b8d0aea04aa0cf4019221c8a3f49a372442508240afb9.jpg)

![412d44f8ca5eb45b739b54c78b1682fccd36dd4340b00acb1d1acadb17882f62.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/412d44f8ca5eb45b739b54c78b1682fccd36dd4340b00acb1d1acadb17882f62.jpg)

![4458ed1e3acc8faf9691a567244312aa1f59007cebfff50cb8e660271864cbd3.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/4458ed1e3acc8faf9691a567244312aa1f59007cebfff50cb8e660271864cbd3.jpg)

![64da96dfced01a87a505489571347c5c9a10b4a6d94708e07a1174466fe0fd09.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/64da96dfced01a87a505489571347c5c9a10b4a6d94708e07a1174466fe0fd09.jpg)

![87b08a31c8c983a3cd83960ab1f674f05e588961ab55d9a7fbe94fc4ef8c5c4a.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/87b08a31c8c983a3cd83960ab1f674f05e588961ab55d9a7fbe94fc4ef8c5c4a.jpg)

![91502a8001032ed88d70050598bda698271fde53720213498445bddc383f5acd.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/91502a8001032ed88d70050598bda698271fde53720213498445bddc383f5acd.jpg)

![a47f8b757bb52b52e1c79bcf1c4f4e02591ba417e0b7961622fb1f75f476377b.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/a47f8b757bb52b52e1c79bcf1c4f4e02591ba417e0b7961622fb1f75f476377b.jpg)

![a85fcb2c728f426155d2e75e3da6a95c1da9d75649a56f5de356871f4e8e9e69.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/a85fcb2c728f426155d2e75e3da6a95c1da9d75649a56f5de356871f4e8e9e69.jpg)

![ab0788ab6c183c610241f2c3bc419c457c3e775067a1ad52d3e5680de19509f9.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/ab0788ab6c183c610241f2c3bc419c457c3e775067a1ad52d3e5680de19509f9.jpg)

![b97cb6248a498147c960dcc1a6796573717e611f9008d6f87aa3c736246e90fe.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/b97cb6248a498147c960dcc1a6796573717e611f9008d6f87aa3c736246e90fe.jpg)

![bf6aabdea2b5951ce4388dfcce184b073fadabaed58f7b21787f2c1bb035916d.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/bf6aabdea2b5951ce4388dfcce184b073fadabaed58f7b21787f2c1bb035916d.jpg)

![c36ec1540d580cb491a18818c0ac721d41e364ee856433cc06ed83f21b19d80f.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/c36ec1540d580cb491a18818c0ac721d41e364ee856433cc06ed83f21b19d80f.jpg)

![c6017213b2c8dc3aa4a61f9e12e132f86e6d2c08b0fe440dba899c2c1e20b956.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/c6017213b2c8dc3aa4a61f9e12e132f86e6d2c08b0fe440dba899c2c1e20b956.jpg)

![ce2edec13e7445fad9212de957db3789bd61ef9c307f4604e63f66e923c5736c.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/ce2edec13e7445fad9212de957db3789bd61ef9c307f4604e63f66e923c5736c.jpg)

![d3856c977d2f14d64810422c60d0d3dd435c2f652a54e76dc4c83cbf65b914a1.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/d3856c977d2f14d64810422c60d0d3dd435c2f652a54e76dc4c83cbf65b914a1.jpg)

![f0b099ff54f49d5d49ee127f12d15301c60382465dc72fbc57a60e69ff48b5a9.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/f0b099ff54f49d5d49ee127f12d15301c60382465dc72fbc57a60e69ff48b5a9.jpg)

![f8656ceb195d190a05bf8bf9ef039ee19c7b041dce2c5f323cbe726946cadb09.jpg](../icml_results/647_Stream%20RAG_%20Instant%20and%20Accurate%20Spoken%20Dialogue%20Systems%20with%20Streaming%20Tool%20Usage/tables/f8656ceb195d190a05bf8bf9ef039ee19c7b041dce2c5f323cbe726946cadb09.jpg)

<a id="paper-0639"></a>

## Minimax-Optimal Policy Regret in Partially Observable Markov Games


### Images

![16b7083a240edb2ca4c261cedf645a360cb4182711bef3d0d4dbe440da4c5ae2.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/16b7083a240edb2ca4c261cedf645a360cb4182711bef3d0d4dbe440da4c5ae2.jpg)

![47821bfbbc0573e9f390c9cf4e72b6759c5ed62ab7fbf91f5335f256579df0e4.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/47821bfbbc0573e9f390c9cf4e72b6759c5ed62ab7fbf91f5335f256579df0e4.jpg)

![6e80583142a0cfefdf0a834b6f52ac33034bd03e3fb5544fb4c9fe2339eae9d7.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/6e80583142a0cfefdf0a834b6f52ac33034bd03e3fb5544fb4c9fe2339eae9d7.jpg)

![91b0de83fdbacf496b7b7bfa204688905f651544347055dc3315cf0cb69a951e.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/91b0de83fdbacf496b7b7bfa204688905f651544347055dc3315cf0cb69a951e.jpg)

![a4deb606e3ebe384c54f30c88f70e2298cf9e1fb20811890f8544164c081d212.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/a4deb606e3ebe384c54f30c88f70e2298cf9e1fb20811890f8544164c081d212.jpg)

![f297f4dfea48df988d7d5e5edaf2a86ea28a75af461136d785f3b4663f2d3521.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/images/f297f4dfea48df988d7d5e5edaf2a86ea28a75af461136d785f3b4663f2d3521.jpg)

### Tables

![526cca02fe732a7970d43821cbe750f4f6efdeec0825e97f486bcc934709a044.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/tables/526cca02fe732a7970d43821cbe750f4f6efdeec0825e97f486bcc934709a044.jpg)

![679587b00afda62dcd4e519671102acae984ee6a05f492cf11c55b6a23799882.jpg](../icml_results/648_Minimax-Optimal%20Policy%20Regret%20in%20Partially%20Observable%20Markov%20Games/tables/679587b00afda62dcd4e519671102acae984ee6a05f492cf11c55b6a23799882.jpg)

<a id="paper-0640"></a>

## TPV: Parameter Perturbations Through the Lens of Test Prediction Variance


### Images

![1f77262febf89732ce03bd9db282049f883d61074ff7149e7f68c572fdca2299.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/1f77262febf89732ce03bd9db282049f883d61074ff7149e7f68c572fdca2299.jpg)

![25d2ab62e3812a4cb9de8938e74ad4ca7262927e2b4706e08a141c1a7265c045.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/25d2ab62e3812a4cb9de8938e74ad4ca7262927e2b4706e08a141c1a7265c045.jpg)

![2e7472e2f0860925d5b07a24b22a9e8b0da58e404093b66220e747bad7c85652.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/2e7472e2f0860925d5b07a24b22a9e8b0da58e404093b66220e747bad7c85652.jpg)

![32239177671674ef6da18fbbe11e27b088090478f4662239de26c9d67b323be5.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/32239177671674ef6da18fbbe11e27b088090478f4662239de26c9d67b323be5.jpg)

![3a20ed0d67496741d2bf8933fa504e7af0a96b24a2e107c4f8596782f3a5c79f.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/3a20ed0d67496741d2bf8933fa504e7af0a96b24a2e107c4f8596782f3a5c79f.jpg)

![44a7afcf8497e118db28e4fca00863c0f6f6ced7cd2fab5eb80ec6f830e4245f.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/44a7afcf8497e118db28e4fca00863c0f6f6ced7cd2fab5eb80ec6f830e4245f.jpg)

![4a1cd12e77c9497b483b35c09382adc46791ccb73beac94fdfa00ac6076570a9.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/4a1cd12e77c9497b483b35c09382adc46791ccb73beac94fdfa00ac6076570a9.jpg)

![5123e3234717318012e2767c9520863d257bbfc8cb071536cb944efb9a6e599c.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/5123e3234717318012e2767c9520863d257bbfc8cb071536cb944efb9a6e599c.jpg)

![51cc171b0d9f50a405f5179855b27b371ca47c9d6540d3257cba7f95105cbf32.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/51cc171b0d9f50a405f5179855b27b371ca47c9d6540d3257cba7f95105cbf32.jpg)

![6a8b3e060f98af1ca01e5aeb98fd017f84b800e140c2e3b9053c4f5fe29eb4a0.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/6a8b3e060f98af1ca01e5aeb98fd017f84b800e140c2e3b9053c4f5fe29eb4a0.jpg)

![7b9c75b4b41bb713feca1d775fad5ef1ece45c8f073ea30fa569e21d14ed518f.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/7b9c75b4b41bb713feca1d775fad5ef1ece45c8f073ea30fa569e21d14ed518f.jpg)

![7d278caa013701a0529263556786fabe49086ad44e1dee0f620ba37a63c8a50d.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/7d278caa013701a0529263556786fabe49086ad44e1dee0f620ba37a63c8a50d.jpg)

![898038f497d5b0ff3697b39f2a7ebef9ca6f8225b39a2902d02eb7ba3538a8f6.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/898038f497d5b0ff3697b39f2a7ebef9ca6f8225b39a2902d02eb7ba3538a8f6.jpg)

![8efb00f7486a887890168daff0f52991f7ba04380068b83a1bd5fd82d4047cc3.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/8efb00f7486a887890168daff0f52991f7ba04380068b83a1bd5fd82d4047cc3.jpg)

![9403c7fc11e8ce7aa18dcf674c12dd41c6bfa0459842934dd1b21bdd176ff38c.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/9403c7fc11e8ce7aa18dcf674c12dd41c6bfa0459842934dd1b21bdd176ff38c.jpg)

![9f8aa4af0327bbc1ff62d79305ed58e7da8dcd40bdff74fe4133bce8d65d311a.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/9f8aa4af0327bbc1ff62d79305ed58e7da8dcd40bdff74fe4133bce8d65d311a.jpg)

![a00a2c5fd2b141b2ebeb9f7eb14849f8c670733dfbabf5507d3925cb9ecc4df0.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/a00a2c5fd2b141b2ebeb9f7eb14849f8c670733dfbabf5507d3925cb9ecc4df0.jpg)

![b452f534f3b675e58b0f307f26e3c5eaaba9812c2cebd9ddf12db6807dc32229.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/b452f534f3b675e58b0f307f26e3c5eaaba9812c2cebd9ddf12db6807dc32229.jpg)

![bf018d653d0f39f8d4a17e9de89064107713c755fc8934397c7daf6a6026097a.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/bf018d653d0f39f8d4a17e9de89064107713c755fc8934397c7daf6a6026097a.jpg)

![c43d9592d206238884670f0afb7667e914fd6f0f83868cad375b1dfdb5ffa337.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/c43d9592d206238884670f0afb7667e914fd6f0f83868cad375b1dfdb5ffa337.jpg)

![c7630b7a0b3c72cb1fcbab64d26742236ba912b077cb3ec8dff0dbc62c870b9a.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/c7630b7a0b3c72cb1fcbab64d26742236ba912b077cb3ec8dff0dbc62c870b9a.jpg)

![c800c329bdb3b804db5db01afc2f2d12026a7669390342b690553b65767abe70.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/c800c329bdb3b804db5db01afc2f2d12026a7669390342b690553b65767abe70.jpg)

![ddda2a6a9389633a2725d1bea5e40943a85753fe5b6459b191d0760875a13958.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/ddda2a6a9389633a2725d1bea5e40943a85753fe5b6459b191d0760875a13958.jpg)

![fa54bb682b8af1351a31ccb4068c961f4f67f9797f64a4416da75dd6e91527ca.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/fa54bb682b8af1351a31ccb4068c961f4f67f9797f64a4416da75dd6e91527ca.jpg)

![fb7badd4a809323201966eab907f32a4ecd26e6c14c571ccab2451658230239a.jpg](../icml_results/649_TPV_%20Parameter%20Perturbations%20Through%20the%20Lens%20of%20Test%20Prediction%20Variance/images/fb7badd4a809323201966eab907f32a4ecd26e6c14c571ccab2451658230239a.jpg)

<a id="paper-0641"></a>

## Learning Rewrite-Invariant Reasoning with Targeted Alternation Training


### Images

![32bb908e79008b76f6428280548c71944e8c450ebc235192f4a1419d99c137e7.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/32bb908e79008b76f6428280548c71944e8c450ebc235192f4a1419d99c137e7.jpg)

![3c12e47ef6c834adc29fe25b20a8872364d0a4ded9ac424630a041d9ebc47c7e.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/3c12e47ef6c834adc29fe25b20a8872364d0a4ded9ac424630a041d9ebc47c7e.jpg)

![7ca54325eb7c2153a64443d67343a93cfe220fab1007aed2dca1c3ae146484df.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/7ca54325eb7c2153a64443d67343a93cfe220fab1007aed2dca1c3ae146484df.jpg)

![8630d5a912e5a6fae5280421304016360778513947076ff325ea65d0dc4671a8.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/8630d5a912e5a6fae5280421304016360778513947076ff325ea65d0dc4671a8.jpg)

![8a872af0a68b281d781e2a08a25ee5e0b5a267a4ec6b6d6f9db833be6196c472.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/8a872af0a68b281d781e2a08a25ee5e0b5a267a4ec6b6d6f9db833be6196c472.jpg)

![b7a1c64bf4d143f2313190ec9800c765f26211aa10e89861df0d86083b91871f.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/images/b7a1c64bf4d143f2313190ec9800c765f26211aa10e89861df0d86083b91871f.jpg)

### Tables

![1fed3b69dc69366ddb5f749efc00b6e1f2f48034c84ed6d384e171b4da7e83a0.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/1fed3b69dc69366ddb5f749efc00b6e1f2f48034c84ed6d384e171b4da7e83a0.jpg)

![43011d6ba7169bf3c4a29d191bb862c11612cc1a7e4c388228dd1ec6e8263b4a.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/43011d6ba7169bf3c4a29d191bb862c11612cc1a7e4c388228dd1ec6e8263b4a.jpg)

![7f6de0fc86e2b3efdc25f6e6dc2902f440748df770f9375bdc5f090cfac28e8b.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/7f6de0fc86e2b3efdc25f6e6dc2902f440748df770f9375bdc5f090cfac28e8b.jpg)

![8abd317726b165c836a216527261857bbfce9f83f8cdd54f191bfa60fd3c574a.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/8abd317726b165c836a216527261857bbfce9f83f8cdd54f191bfa60fd3c574a.jpg)

![94d39a3f38efd776ce0f3398005370473ce1bd095534449d4d62451e02072f0e.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/94d39a3f38efd776ce0f3398005370473ce1bd095534449d4d62451e02072f0e.jpg)

![97414c1f0ab407bc34d16cf6284e602097bfaa30279f05c9e69a9117f70fb062.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/97414c1f0ab407bc34d16cf6284e602097bfaa30279f05c9e69a9117f70fb062.jpg)

![9d7357623b09aaa57a0ab3214f8a0e338d42fde63b69fb971fac80e31ba912e9.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/9d7357623b09aaa57a0ab3214f8a0e338d42fde63b69fb971fac80e31ba912e9.jpg)

![a3bb1b17da7d39fa7141b49ac7f3b455e3d32a72f14f627d28ca367930e656b2.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/a3bb1b17da7d39fa7141b49ac7f3b455e3d32a72f14f627d28ca367930e656b2.jpg)

![c6b667ee79cf6311182009e583ba2cf5e113ccedac7cd2c460124522002a7788.jpg](../icml_results/650_Learning%20Rewrite-Invariant%20Reasoning%20with%20Targeted%20Alternation%20Training/tables/c6b667ee79cf6311182009e583ba2cf5e113ccedac7cd2c460124522002a7788.jpg)

<a id="paper-0642"></a>

## Set Diffusion: Interpolating Token Orderings between Autoregression and Diffusion for Fast and Flexible Decoding


### Images

![2e4b0077bb898092aee8858bec5cf0567b52021839efc3cacc15ac4e5374c60a.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/images/2e4b0077bb898092aee8858bec5cf0567b52021839efc3cacc15ac4e5374c60a.jpg)

![3be1cf049ea8d9a2ea36888d77b1bc925f6daa1a9fa14de85ea94520440197ab.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/images/3be1cf049ea8d9a2ea36888d77b1bc925f6daa1a9fa14de85ea94520440197ab.jpg)

![687d8f3a2f554e1de0275b5b7307a80515e6698e0c07f05e11ae48a5cdf09dcf.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/images/687d8f3a2f554e1de0275b5b7307a80515e6698e0c07f05e11ae48a5cdf09dcf.jpg)

![8d8a9690f630cb36e1c4094ec8182c32f16e343e4d7aec9e8d492dac72fffd1c.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/images/8d8a9690f630cb36e1c4094ec8182c32f16e343e4d7aec9e8d492dac72fffd1c.jpg)

![cbd7805ca5ed561c82dec55611b48eb81344df42180b1ee1b69bf083e9d2eb15.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/images/cbd7805ca5ed561c82dec55611b48eb81344df42180b1ee1b69bf083e9d2eb15.jpg)

### Tables

![07a27c075d47a9d6d478e489b616ce907d18a242e9b95cd07d58b8b75959fa23.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/07a27c075d47a9d6d478e489b616ce907d18a242e9b95cd07d58b8b75959fa23.jpg)

![099fc8854eae793d56e100b20d687f0d34b4e4a19da883b241190ca5242505ec.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/099fc8854eae793d56e100b20d687f0d34b4e4a19da883b241190ca5242505ec.jpg)

![2212b361083f1fbd3ebb39b5a5c1f439476a62aadb7dfb59fe36198dc5fbc820.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/2212b361083f1fbd3ebb39b5a5c1f439476a62aadb7dfb59fe36198dc5fbc820.jpg)

![3a1b664f62f88da8f065667317891a426559cd7b467fe7237e84181566348c9e.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/3a1b664f62f88da8f065667317891a426559cd7b467fe7237e84181566348c9e.jpg)

![5cb771ec9180c4e0d22aede67199cbe5733a83c43d9ed0626b9c136470c2bf2a.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/5cb771ec9180c4e0d22aede67199cbe5733a83c43d9ed0626b9c136470c2bf2a.jpg)

![662825824a15817268972ec6e8622a2abcf66e6543cc7eaedebf7a4ed2e71384.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/662825824a15817268972ec6e8622a2abcf66e6543cc7eaedebf7a4ed2e71384.jpg)

![7e7c350b70990cd65925f8ae37d204106a117ffc4296992a75164c9bc16af918.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/7e7c350b70990cd65925f8ae37d204106a117ffc4296992a75164c9bc16af918.jpg)

![a3613512c6a1c449c5b7e2d8185fff9512d2baa048e6c78b8aef3ee356ea17e2.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/a3613512c6a1c449c5b7e2d8185fff9512d2baa048e6c78b8aef3ee356ea17e2.jpg)

![ba478978e6c39a63c90aa2a075fda39a15bc5827cd5cbd0c816f9ea17e7245ed.jpg](../icml_results/651_Set%20Diffusion_%20Interpolating%20Token%20Orderings%20between%20Autoregression%20and%20Diffusion%20for%20Fast%20and%20Flexi/tables/ba478978e6c39a63c90aa2a075fda39a15bc5827cd5cbd0c816f9ea17e7245ed.jpg)

<a id="paper-0643"></a>

## Exploiting weight-space symmetries for approximating curvature


### Images

![212b4907bed5f89ef68d98bba1a50a2b65014647adbc3a26c93bb336c51dfc9c.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/212b4907bed5f89ef68d98bba1a50a2b65014647adbc3a26c93bb336c51dfc9c.jpg)

![29d7500793123ba35d5faa758589460756e68a58772007b9ead79ed0346cf97e.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/29d7500793123ba35d5faa758589460756e68a58772007b9ead79ed0346cf97e.jpg)

![32c1e8588eb7d6cefaa29960326c6af8de0f6d6bc3682e0190afbdae953cbd91.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/32c1e8588eb7d6cefaa29960326c6af8de0f6d6bc3682e0190afbdae953cbd91.jpg)

![3d5ccde410797ab83d469d15558d773cc3a8eae370f70908b517c96209dd2cec.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/3d5ccde410797ab83d469d15558d773cc3a8eae370f70908b517c96209dd2cec.jpg)

![85e0ef2f1b65875c10a9709b4fe26599bd274e6d1db1abd6a7119a176efd51bb.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/85e0ef2f1b65875c10a9709b4fe26599bd274e6d1db1abd6a7119a176efd51bb.jpg)

![a7f8682afe94941f98b8ceb9125c386a1540b1a2ad9ebb86af7fabe85d234f13.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/a7f8682afe94941f98b8ceb9125c386a1540b1a2ad9ebb86af7fabe85d234f13.jpg)

![b8b7dc6d89f5969bd3b1ee8ebffb8fae3445cae320a14610b9b04a70ee134dbb.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/b8b7dc6d89f5969bd3b1ee8ebffb8fae3445cae320a14610b9b04a70ee134dbb.jpg)

![ea07914d1847138edd3ce14d0c88a10c0efe0238675656a1b430fba1811c7f14.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/ea07914d1847138edd3ce14d0c88a10c0efe0238675656a1b430fba1811c7f14.jpg)

![eb659e84892af817d12162e57e51a374280e1512371d963939287b8fdf38aa6c.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/eb659e84892af817d12162e57e51a374280e1512371d963939287b8fdf38aa6c.jpg)

![fb5d59e6974aea0ca6841e991f5c539612152054405107e134cb399b84c1b0ff.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/fb5d59e6974aea0ca6841e991f5c539612152054405107e134cb399b84c1b0ff.jpg)

![ffbe8290a3233800dfc516c90c2c6dac14c6fb520563e32556aeefd985ccc84d.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/images/ffbe8290a3233800dfc516c90c2c6dac14c6fb520563e32556aeefd985ccc84d.jpg)

### Tables

![00816f861e42871a5627305dc4bdd491d907b2f61a3675e8533f7244d9d0f801.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/00816f861e42871a5627305dc4bdd491d907b2f61a3675e8533f7244d9d0f801.jpg)

![4702b0cdd80163e38905c56a2e34233e0aa36bf5da123339d8d4d76b34ed771f.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/4702b0cdd80163e38905c56a2e34233e0aa36bf5da123339d8d4d76b34ed771f.jpg)

![53f947bb7bf8d7de62863e1c1f757a202d8e7b2d0d2e7f0b401a6472413a9fd8.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/53f947bb7bf8d7de62863e1c1f757a202d8e7b2d0d2e7f0b401a6472413a9fd8.jpg)

![609581856907ca5933bb50c60e570bfa92872dae0c7455a7cd93390e7d563b7e.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/609581856907ca5933bb50c60e570bfa92872dae0c7455a7cd93390e7d563b7e.jpg)

![6d46c437330ea32d13b855ef6c470d2fef5ae1c5e56f9f5259a8be94c94e58c5.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/6d46c437330ea32d13b855ef6c470d2fef5ae1c5e56f9f5259a8be94c94e58c5.jpg)

![7f9af0362fe02b25d593e5bc2d7b8f5326a49c3f55b23bfc7c22d1caeeff79d8.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/7f9af0362fe02b25d593e5bc2d7b8f5326a49c3f55b23bfc7c22d1caeeff79d8.jpg)

![93e15a43bdd50d580f26b35bd09c978829dde266c8b51991e11945297c542e51.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/93e15a43bdd50d580f26b35bd09c978829dde266c8b51991e11945297c542e51.jpg)

![a07cdd2c8005a546ab4c5e27d182b085fb4aa238a55eeda28faf1013f581c26b.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/a07cdd2c8005a546ab4c5e27d182b085fb4aa238a55eeda28faf1013f581c26b.jpg)

![a87910ce63071b823f898c93a453d66d9d0434d47970615470de539a656f9539.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/a87910ce63071b823f898c93a453d66d9d0434d47970615470de539a656f9539.jpg)

![aaa0e7e1b9b9e6b03fad3ddb68ab1f1ad0ba830acc1b6d7f92ad13f2c78f25f5.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/aaa0e7e1b9b9e6b03fad3ddb68ab1f1ad0ba830acc1b6d7f92ad13f2c78f25f5.jpg)

![b2eb08eb146a39cd7885ce0c895bbe595eb9a1b0a2aeeb48f5e2725dcf9d4f92.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/b2eb08eb146a39cd7885ce0c895bbe595eb9a1b0a2aeeb48f5e2725dcf9d4f92.jpg)

![d12604c7def0cb6877fcf064c614c30421b782d432b16072f44a20226e7c82a0.jpg](../icml_results/652_Exploiting%20weight-space%20symmetries%20for%20approximating%20curvature/tables/d12604c7def0cb6877fcf064c614c30421b782d432b16072f44a20226e7c82a0.jpg)

<a id="paper-0644"></a>

## AutoBaxBuilder: Bootstrapping Code Security Benchmarking


### Images

![0137484536ca1a0eb8b558a71e106f551238f071294cae80f28da134df632033.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/0137484536ca1a0eb8b558a71e106f551238f071294cae80f28da134df632033.jpg)

![0344cb61a8ef916ee3eec9de47ced02a87b5fef7b121bef36de5efc9f5e12229.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/0344cb61a8ef916ee3eec9de47ced02a87b5fef7b121bef36de5efc9f5e12229.jpg)

![0400a3fc11ea6991fc41a8adf84ab61eca5f4e78ac659fcbb241ff07233ee569.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/0400a3fc11ea6991fc41a8adf84ab61eca5f4e78ac659fcbb241ff07233ee569.jpg)

![08001e080e411a6dec19bcc531d6aa171ea5100b257aa409f677a0d0ab99c3dd.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/08001e080e411a6dec19bcc531d6aa171ea5100b257aa409f677a0d0ab99c3dd.jpg)

![191705554b1bcbe8ddab24bbe59b9978c1613eb21d69a723336fa01b01bd3a63.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/191705554b1bcbe8ddab24bbe59b9978c1613eb21d69a723336fa01b01bd3a63.jpg)

![1d25197b2fe34ecbeda69af1670ec15321c6ae1f2c3530fa90359460f2d32593.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/1d25197b2fe34ecbeda69af1670ec15321c6ae1f2c3530fa90359460f2d32593.jpg)

![1de2c35d1dbbc45b390e9a3400f1c4f03075b89134b772cc896402b1153fa1e8.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/1de2c35d1dbbc45b390e9a3400f1c4f03075b89134b772cc896402b1153fa1e8.jpg)

![2218656d5dbaca971af8ca142f706223d494fb256ade260a2c5bef2ab8bef9b6.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/2218656d5dbaca971af8ca142f706223d494fb256ade260a2c5bef2ab8bef9b6.jpg)

![2620f1737bc6895380835d8b401224947d800488e50088a0c4428632eb1d8b60.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/2620f1737bc6895380835d8b401224947d800488e50088a0c4428632eb1d8b60.jpg)

![4ab72297e5a08c14522ad3c83d231d7b4045c1d2ea8295aa7d75c09ceed409e5.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/4ab72297e5a08c14522ad3c83d231d7b4045c1d2ea8295aa7d75c09ceed409e5.jpg)

![5123432cb909f9ce052a524e51ec95979d89f8eb261532ceed03113a5dee74a6.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/5123432cb909f9ce052a524e51ec95979d89f8eb261532ceed03113a5dee74a6.jpg)

![53755c8d112338095d989a97de0822f8dc497e394ebc93829238201e44fe7249.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/53755c8d112338095d989a97de0822f8dc497e394ebc93829238201e44fe7249.jpg)

![57b95e7c1fb2cbd59b0b65e64e548ce43f644e9ad9216948e0fee2b064f704e3.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/57b95e7c1fb2cbd59b0b65e64e548ce43f644e9ad9216948e0fee2b064f704e3.jpg)

![5955d7c55c8c03e43c45ff063fe453b15cfca1be8a08915c141360d9769ac530.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/5955d7c55c8c03e43c45ff063fe453b15cfca1be8a08915c141360d9769ac530.jpg)

![5c31d3da97bc34158f2bf4666d22702c65386fdf70d2fd26ba1bf98a5f59b0a1.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/5c31d3da97bc34158f2bf4666d22702c65386fdf70d2fd26ba1bf98a5f59b0a1.jpg)

![64cfa3478b8a9122ab17139e8c76d0b4ee706faf7eb62a7310580d03a32d5487.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/64cfa3478b8a9122ab17139e8c76d0b4ee706faf7eb62a7310580d03a32d5487.jpg)

![6726f5a1d19042814fad625733f4dcfbba94ac70bd92b7f82c1230594cece021.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/6726f5a1d19042814fad625733f4dcfbba94ac70bd92b7f82c1230594cece021.jpg)

![783ab5350c398052bff2d4930f107fcbb99fa78a5c535e76ef229e9bd9dadb73.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/783ab5350c398052bff2d4930f107fcbb99fa78a5c535e76ef229e9bd9dadb73.jpg)

![79bc9d0440346b80f81d6ee584e1ef0e186eece02aeb045eeb496e7177eea84f.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/79bc9d0440346b80f81d6ee584e1ef0e186eece02aeb045eeb496e7177eea84f.jpg)

![7e8110e4bb63695698ed9dfd0e1bc096b6c671dd2d0406ce424fef7339327a9a.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/7e8110e4bb63695698ed9dfd0e1bc096b6c671dd2d0406ce424fef7339327a9a.jpg)

![98f0c667efdc884ba805bd6f33f4372ce1d319d0a94d5f6f4c8e5c67d92e8171.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/98f0c667efdc884ba805bd6f33f4372ce1d319d0a94d5f6f4c8e5c67d92e8171.jpg)

![99b7ee148f8c2530b63c306f7279692fed0fbe35b066d2d95ebc40adbe6afe26.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/99b7ee148f8c2530b63c306f7279692fed0fbe35b066d2d95ebc40adbe6afe26.jpg)

![a3103236064e8534d108cc91f85a34b4929643c23a7e6dad057ec33b70c83952.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/a3103236064e8534d108cc91f85a34b4929643c23a7e6dad057ec33b70c83952.jpg)

![c3fe45bf0734d725fcda3372b46fcce3042251fe5a6e755f25f5408505175d3d.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/c3fe45bf0734d725fcda3372b46fcce3042251fe5a6e755f25f5408505175d3d.jpg)

![c41e3deb7612d4c93bffe491cee7c718205829c0052eb639861a7cef67271f85.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/c41e3deb7612d4c93bffe491cee7c718205829c0052eb639861a7cef67271f85.jpg)

![c459e8bfbd045db42de16ed866baabb88030c988c067186cfc859450b188fb59.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/c459e8bfbd045db42de16ed866baabb88030c988c067186cfc859450b188fb59.jpg)

![d4f0739fc831b99c1c185658d20f289173a61264e95e4436a6d8bbf87ee88ec8.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/d4f0739fc831b99c1c185658d20f289173a61264e95e4436a6d8bbf87ee88ec8.jpg)

![d62779a134fa32db497bfb5a860f3ca20e2938d1399a78af8d5ff00dba12a4b2.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/d62779a134fa32db497bfb5a860f3ca20e2938d1399a78af8d5ff00dba12a4b2.jpg)

![d8b3fe4316706e913671b0e1545a11e5d50714c8e5ce8ff3fc14190858938114.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/d8b3fe4316706e913671b0e1545a11e5d50714c8e5ce8ff3fc14190858938114.jpg)

![e94499cfe6a20f7f8b864308b1315d00c7bbd5a581e7d733f8a5df4f5646119e.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/e94499cfe6a20f7f8b864308b1315d00c7bbd5a581e7d733f8a5df4f5646119e.jpg)

![f8a34e1357f62e8d79302159c2a2543b618638ceffadfdbbf970f2475fc00fea.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/f8a34e1357f62e8d79302159c2a2543b618638ceffadfdbbf970f2475fc00fea.jpg)

![fe71d5f25f24f0f61031db3e50ea4513ea8ea1f55e4d006cc4f812f1b9228bd3.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/fe71d5f25f24f0f61031db3e50ea4513ea8ea1f55e4d006cc4f812f1b9228bd3.jpg)

![ffa36e18fc7ae32d56ec09b438adddcbb415a436484d92cd9bff2ed904b16486.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/images/ffa36e18fc7ae32d56ec09b438adddcbb415a436484d92cd9bff2ed904b16486.jpg)

### Tables

![2e3e995640d2b68fe45f84166f2f2ddce25017182c4868c50dbf9327a0756f59.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/2e3e995640d2b68fe45f84166f2f2ddce25017182c4868c50dbf9327a0756f59.jpg)

![3bf2c9ed8c065b8107d25b02d945eefa0fa3d3f82c59873bd719f70cbf2909c3.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/3bf2c9ed8c065b8107d25b02d945eefa0fa3d3f82c59873bd719f70cbf2909c3.jpg)

![60881a871945d07731d76bbb6c7e3d031c61de17ddef681a2a578dc778ba0cae.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/60881a871945d07731d76bbb6c7e3d031c61de17ddef681a2a578dc778ba0cae.jpg)

![7899cb02c966ae09a420e78683267da5c200d86a6a716168116729d147866f11.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/7899cb02c966ae09a420e78683267da5c200d86a6a716168116729d147866f11.jpg)

![812e0670d5b811afa101319327026dc431222cf11e670d765b4e6f1486b445dc.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/812e0670d5b811afa101319327026dc431222cf11e670d765b4e6f1486b445dc.jpg)

![9fba98a261ef920554f8452cb263e5f693b2d024f47de59f42dc1fad2c7b5a98.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/9fba98a261ef920554f8452cb263e5f693b2d024f47de59f42dc1fad2c7b5a98.jpg)

![a0a6857087acb54cb839de87a147acea990373617fa683a17e0f13436924e9e5.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/a0a6857087acb54cb839de87a147acea990373617fa683a17e0f13436924e9e5.jpg)

![a0ca3d75562edb43dc53e5b5bd73812086e57faa4679e1aeb3203b7851c4d94f.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/a0ca3d75562edb43dc53e5b5bd73812086e57faa4679e1aeb3203b7851c4d94f.jpg)

![a16454c56197b2df22de5c31c3968bc9600557ae398354ca2011d4cff991df6e.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/a16454c56197b2df22de5c31c3968bc9600557ae398354ca2011d4cff991df6e.jpg)

![c0b474c6ede951f88f8f0abcd322ae475af2e523799eab0803ea7809e3421647.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/c0b474c6ede951f88f8f0abcd322ae475af2e523799eab0803ea7809e3421647.jpg)

![e7158049c5bc7f531dbbd99e6b711134a938c29ddacf35c2ba41d0e9751d6ce0.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/e7158049c5bc7f531dbbd99e6b711134a938c29ddacf35c2ba41d0e9751d6ce0.jpg)

![eb4ca98b76b952e641d7e818034efefcf20444e8176d526b4c7fdd8faf08e468.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/eb4ca98b76b952e641d7e818034efefcf20444e8176d526b4c7fdd8faf08e468.jpg)

![f92e4898ef90dc877cc3dbc02843e2ae23df7849cfdd1846aa576406fdda5274.jpg](../icml_results/653_AutoBaxBuilder_%20Bootstrapping%20Code%20Security%20Benchmarking/tables/f92e4898ef90dc877cc3dbc02843e2ae23df7849cfdd1846aa576406fdda5274.jpg)

<a id="paper-0645"></a>

## RedDebate: Safer Responses Through Multi-Agent Red Teaming Debates


### Images

![09993f6cbd367a93de2c6aea803aba7da6aa1753f497fa059b9ea7adf0b98b73.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/09993f6cbd367a93de2c6aea803aba7da6aa1753f497fa059b9ea7adf0b98b73.jpg)

![107151dbab1e393250aeb19387407e485a546eb90afb8644a7a8757d56ab3314.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/107151dbab1e393250aeb19387407e485a546eb90afb8644a7a8757d56ab3314.jpg)

![5ba6aa214dd6a12d9ecd2806e509574075126f625f35860c953804a212939f59.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/5ba6aa214dd6a12d9ecd2806e509574075126f625f35860c953804a212939f59.jpg)

![71c7dc4264e48b7b45c5c13a2e48615fe957c50dfb823f01c996b54c2d8cd952.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/71c7dc4264e48b7b45c5c13a2e48615fe957c50dfb823f01c996b54c2d8cd952.jpg)

![77923c2ba1e0de487ef471d757d43795c3bb2fa5eaf0cacb28766d5890be03af.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/77923c2ba1e0de487ef471d757d43795c3bb2fa5eaf0cacb28766d5890be03af.jpg)

![789c92085764e19adb6d6f7a51e928ce18a883d1acf643b7f05e06319c990166.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/789c92085764e19adb6d6f7a51e928ce18a883d1acf643b7f05e06319c990166.jpg)

![ac80b36c0bf70ec6170573b39835c710a7146f5e536a72b824829818cb447b33.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/ac80b36c0bf70ec6170573b39835c710a7146f5e536a72b824829818cb447b33.jpg)

![ccaf64509807412a7dcd01d0a99ad6a899494cd21a438b1f069c61187887db8b.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/ccaf64509807412a7dcd01d0a99ad6a899494cd21a438b1f069c61187887db8b.jpg)

![d9de48ce2aa0dcb7721fba997fcff1b3d6b9ba72b2fabc06833eba371614c0b9.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/d9de48ce2aa0dcb7721fba997fcff1b3d6b9ba72b2fabc06833eba371614c0b9.jpg)

![e86e159c8b22af69376670f3d1206adf493e6f6e2917cf4f655ad96dca673a5b.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/e86e159c8b22af69376670f3d1206adf493e6f6e2917cf4f655ad96dca673a5b.jpg)

![e92ad9bfbdad9bdcc21703ad3530c51dd7c373c4d658a2026bd33dda23d0f68d.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/e92ad9bfbdad9bdcc21703ad3530c51dd7c373c4d658a2026bd33dda23d0f68d.jpg)

![fa0c8636674aa3af02905804d1db532c87c7bde0d822217719666cefe981c539.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/fa0c8636674aa3af02905804d1db532c87c7bde0d822217719666cefe981c539.jpg)

![fb6ed27ef5b37c472337c7cb8e14f2f8c3be340ac2c753059ee95743bf694872.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/fb6ed27ef5b37c472337c7cb8e14f2f8c3be340ac2c753059ee95743bf694872.jpg)

![fd5c7517f8d04e691209a30d22a3b8fc6bc66d9d226ff2a81108e13591c079e5.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/images/fd5c7517f8d04e691209a30d22a3b8fc6bc66d9d226ff2a81108e13591c079e5.jpg)

### Tables

![1db3de3fee4aa0ad40c1640f317794a8bc441f1236460cbc64fb07a2f24d0bba.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/1db3de3fee4aa0ad40c1640f317794a8bc441f1236460cbc64fb07a2f24d0bba.jpg)

![1fbfae30a3f10c3721b41df8f8cc8b199ae1eec3b01d99b021f034992d9e8930.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/1fbfae30a3f10c3721b41df8f8cc8b199ae1eec3b01d99b021f034992d9e8930.jpg)

![2304dc0f1aa7261c3e50bb4879fb6ae6f9544a3e1999c9f1edceea3e15178544.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/2304dc0f1aa7261c3e50bb4879fb6ae6f9544a3e1999c9f1edceea3e15178544.jpg)

![43d3511dbacb8055ec4e6e3cf2d58cef1c59070f5b39b2017bcf1d50275e5657.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/43d3511dbacb8055ec4e6e3cf2d58cef1c59070f5b39b2017bcf1d50275e5657.jpg)

![508d18291363f74ccebb54c46fa8192001d82ee65a8907b22a141a98caa56137.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/508d18291363f74ccebb54c46fa8192001d82ee65a8907b22a141a98caa56137.jpg)

![5dc71169db89154cbe0c188a07c106122fe0adc6e1cfc151e45c7f7919f720c6.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/5dc71169db89154cbe0c188a07c106122fe0adc6e1cfc151e45c7f7919f720c6.jpg)

![7e2dcae0a762cf6e21690fe28b6494684231c27c41f95c07e73ea2fa7f893b94.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/7e2dcae0a762cf6e21690fe28b6494684231c27c41f95c07e73ea2fa7f893b94.jpg)

![81847599c596982cb6b58e18650f2cb5f1077d6508ebd1261d0eda0cb381c741.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/81847599c596982cb6b58e18650f2cb5f1077d6508ebd1261d0eda0cb381c741.jpg)

![82456705abd9ed02a9d0c55a690efe4edc4515fd2f7353a2e9db528017ed2e80.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/82456705abd9ed02a9d0c55a690efe4edc4515fd2f7353a2e9db528017ed2e80.jpg)

![874aff93acaee24cba8527c7446b05d390a8b4a63ad57aab9ce166fb74afbb85.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/874aff93acaee24cba8527c7446b05d390a8b4a63ad57aab9ce166fb74afbb85.jpg)

![8b397e47bd4c0a5b9f4f33ac826bd1d88f21515b31d69bcd4159f2174e0c5bc0.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/8b397e47bd4c0a5b9f4f33ac826bd1d88f21515b31d69bcd4159f2174e0c5bc0.jpg)

![9096e244350720d002d441edd70fa432d7e20dbc988793df3d83e2e6baee7b8c.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/9096e244350720d002d441edd70fa432d7e20dbc988793df3d83e2e6baee7b8c.jpg)

![93d64d9df6d582968636a8d185aa60e8c937bea4fc9dfa19a2ff9825603d227f.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/93d64d9df6d582968636a8d185aa60e8c937bea4fc9dfa19a2ff9825603d227f.jpg)

![9aea332917f7089d9e9258f346954189b3df0b82fb7b5faba97e49330acf0931.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/9aea332917f7089d9e9258f346954189b3df0b82fb7b5faba97e49330acf0931.jpg)

![a1b5804551e17ff59be981be7ab157745b8a2a08848addb9005bd2f3444b8f22.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/a1b5804551e17ff59be981be7ab157745b8a2a08848addb9005bd2f3444b8f22.jpg)

![a74d3064a7e355116614f8bc30713eb2df3723b4741013fa7ae6afd3310def31.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/a74d3064a7e355116614f8bc30713eb2df3723b4741013fa7ae6afd3310def31.jpg)

![a8ce2ec9b6703b3eba2b7384552fc960b6d61d1fa7d5f16dbefb0cd7f8030fe7.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/a8ce2ec9b6703b3eba2b7384552fc960b6d61d1fa7d5f16dbefb0cd7f8030fe7.jpg)

![caec83215f866016ffe817f78a9f73f0267e1cd103b353226501110236f22e3a.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/caec83215f866016ffe817f78a9f73f0267e1cd103b353226501110236f22e3a.jpg)

![e25c102fa1ce0d1dba807891175d4a39d547295d48ed9e419be0f4e2d32c1382.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/e25c102fa1ce0d1dba807891175d4a39d547295d48ed9e419be0f4e2d32c1382.jpg)

![ee8c2a093ee4f593b6861edd876ab84a15504c312d2055f214ce3a143cf347d5.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/ee8c2a093ee4f593b6861edd876ab84a15504c312d2055f214ce3a143cf347d5.jpg)

![f027ca547c7a839eb7c370164221ef2b039e3442b55679a8d169cb4c1d8cd217.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/f027ca547c7a839eb7c370164221ef2b039e3442b55679a8d169cb4c1d8cd217.jpg)

![f3c04dbc2e236e3286735c8b99faa757dcf6921e86b4a65de40fc817158619e7.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/f3c04dbc2e236e3286735c8b99faa757dcf6921e86b4a65de40fc817158619e7.jpg)

![f8d028077f2330ab5b27e01c4abfdc5de4f7996058afdb4c12d9688759a8a7a9.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/f8d028077f2330ab5b27e01c4abfdc5de4f7996058afdb4c12d9688759a8a7a9.jpg)

![fcd7eb68645356dddf75738f42c30a3d34e360e8059491cba72cce2606eeedd9.jpg](../icml_results/654_RedDebate_%20Safer%20Responses%20Through%20Multi-Agent%20Red%20Teaming%20Debates/tables/fcd7eb68645356dddf75738f42c30a3d34e360e8059491cba72cce2606eeedd9.jpg)

<a id="paper-0646"></a>

## TRACER: Persistent Regularization for Robust Multimodal Finetuning


### Images

![0316e29765b6fe6f5df1aa1a297f037db3e1ffde25a97058689aac5da844bcea.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/0316e29765b6fe6f5df1aa1a297f037db3e1ffde25a97058689aac5da844bcea.jpg)

![03a2f257dba02c6088545883415b8541ca7c3a53e69df632bea73fa9a4053950.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/03a2f257dba02c6088545883415b8541ca7c3a53e69df632bea73fa9a4053950.jpg)

![23bc71b2c9b46f299954d1134d2af3f8dfca46cc8acb96644662f43dcd1a347c.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/23bc71b2c9b46f299954d1134d2af3f8dfca46cc8acb96644662f43dcd1a347c.jpg)

![33e1e3c5f5f918e7849de462b42b9ec488015a6db8ab10ddf6e0a0892e946c54.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/33e1e3c5f5f918e7849de462b42b9ec488015a6db8ab10ddf6e0a0892e946c54.jpg)

![4eb85544bd072f8a5f2dcedd67943880d14cbaea06ddbf11df06642243b1600a.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/4eb85544bd072f8a5f2dcedd67943880d14cbaea06ddbf11df06642243b1600a.jpg)

![a600b5e51b4504b5eeb827099a6fe9fba7765c08dc2131254126ef73bfee3140.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/a600b5e51b4504b5eeb827099a6fe9fba7765c08dc2131254126ef73bfee3140.jpg)

![d21472778e04bd99ed120ecf300f3f7209e5220121b1d132437ab18e002538db.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/d21472778e04bd99ed120ecf300f3f7209e5220121b1d132437ab18e002538db.jpg)

![e40bf7b604d2675b4b475ba37cf3e5310331ca29881a108b153654f825812886.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/images/e40bf7b604d2675b4b475ba37cf3e5310331ca29881a108b153654f825812886.jpg)

### Tables

![0440208eccb5951138ef8315bb74a091a90d1bb37d818bdde5beec94bf0c7349.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/0440208eccb5951138ef8315bb74a091a90d1bb37d818bdde5beec94bf0c7349.jpg)

![271c47c4a197081d2cdabb538afa391a1d5b5929b65e773cae59ed4fe499fbb1.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/271c47c4a197081d2cdabb538afa391a1d5b5929b65e773cae59ed4fe499fbb1.jpg)

![4b018e2884f1f2786d2a0d9093f2f8137497cc21ac38b0fee0bc6c96098b6b0e.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/4b018e2884f1f2786d2a0d9093f2f8137497cc21ac38b0fee0bc6c96098b6b0e.jpg)

![5a60dab4a2ebea8606779549536b60458a647302144b4573a984aa07c1f98f3a.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/5a60dab4a2ebea8606779549536b60458a647302144b4573a984aa07c1f98f3a.jpg)

![7257ac39bca5726211c3ec7d26b729040bac1cd62eeaa5f4b5dd5a0d3c1a0aa1.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/7257ac39bca5726211c3ec7d26b729040bac1cd62eeaa5f4b5dd5a0d3c1a0aa1.jpg)

![7d3056803f609a342ffc70c7f6ad699f78451e55a81914964a840e86ba2a6163.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/7d3056803f609a342ffc70c7f6ad699f78451e55a81914964a840e86ba2a6163.jpg)

![a56dfbe95ecb2fd44ba3a85ce412b4c6546c90170fe7b67d65d4a38df85600a5.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/a56dfbe95ecb2fd44ba3a85ce412b4c6546c90170fe7b67d65d4a38df85600a5.jpg)

![a5cbbb9bd62505826905b035c33407e422b5a0b57c3a82fd961d6ab3cc12df77.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/a5cbbb9bd62505826905b035c33407e422b5a0b57c3a82fd961d6ab3cc12df77.jpg)

![aa1c4609fc239e45f9166a65d2f88779054066236043773f38a483ed80efa297.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/aa1c4609fc239e45f9166a65d2f88779054066236043773f38a483ed80efa297.jpg)

![acef6c49edc66dfed5b01c2afb0b1df942610ae462984824ccd24cdedfb392bb.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/acef6c49edc66dfed5b01c2afb0b1df942610ae462984824ccd24cdedfb392bb.jpg)

![b2e7e882539d45254cccde6c1271c4442451b27931a4bf1aaaec88fc5043dfa5.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/b2e7e882539d45254cccde6c1271c4442451b27931a4bf1aaaec88fc5043dfa5.jpg)

![df2442e951d20b6263f537eda703803e11fb50cb9f12e33af82ecbcea4d74dca.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/df2442e951d20b6263f537eda703803e11fb50cb9f12e33af82ecbcea4d74dca.jpg)

![f7b3610ec6d824423e5110b42154ae8eaff925b1871cb164f4ddcfdb749d9182.jpg](../icml_results/655_TRACER_%20Persistent%20Regularization%20for%20Robust%20Multimodal%20Finetuning/tables/f7b3610ec6d824423e5110b42154ae8eaff925b1871cb164f4ddcfdb749d9182.jpg)

<a id="paper-0647"></a>

## Latent Reasoning in TRMs is Secretly a Policy Improvement Operator


### Images

![119f429228ce8724b29dcc0cceaa8076f1bbb347f6b99304a39b210c442b5213.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/119f429228ce8724b29dcc0cceaa8076f1bbb347f6b99304a39b210c442b5213.jpg)

![250d5dfa2fddc952e31c4db280b216724442a4f2b47ebb8e096838f89e2020d2.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/250d5dfa2fddc952e31c4db280b216724442a4f2b47ebb8e096838f89e2020d2.jpg)

![2833c1c1c6db80fb509dc042da23fc8f1fbbc597761a830e97695a92bf151961.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/2833c1c1c6db80fb509dc042da23fc8f1fbbc597761a830e97695a92bf151961.jpg)

![5919e9d7fa765ec7e2edcb66dd4fb6967b008cb876cde8d317797c0d71a78523.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/5919e9d7fa765ec7e2edcb66dd4fb6967b008cb876cde8d317797c0d71a78523.jpg)

![803dc98e01588912ae91c66a60355bc85c1aac03a17aee0a81184e33dbd3efbd.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/803dc98e01588912ae91c66a60355bc85c1aac03a17aee0a81184e33dbd3efbd.jpg)

![d399721386d62dc075e277473bbc50c142e35def8def31c335a770399f202a6f.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/d399721386d62dc075e277473bbc50c142e35def8def31c335a770399f202a6f.jpg)

![fba043199e13156e9deb265f1a9c56689a569c6fdbc2eaed62b818b67064fff1.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/images/fba043199e13156e9deb265f1a9c56689a569c6fdbc2eaed62b818b67064fff1.jpg)

### Tables

![0436e07bd1e6bbc116ae031c7cda79b37dfe23df8d0da2bf719421eaf01ff553.jpg](../icml_results/656_Latent%20Reasoning%20in%20TRMs%20is%20Secretly%20a%20Policy%20Improvement%20Operator/tables/0436e07bd1e6bbc116ae031c7cda79b37dfe23df8d0da2bf719421eaf01ff553.jpg)

<a id="paper-0648"></a>

## Zero-Shot Off-Policy Learning


### Images

![44e135a4afa558c72bcbd0c9cd3d4321ae805aceb034c5afb114f6e16269ab2b.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/44e135a4afa558c72bcbd0c9cd3d4321ae805aceb034c5afb114f6e16269ab2b.jpg)

![56bcc0c7fe6a23cef0e7a7312e5c19648a0047e2d07429083fdc4dcc50e74ca0.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/56bcc0c7fe6a23cef0e7a7312e5c19648a0047e2d07429083fdc4dcc50e74ca0.jpg)

![87df62f095b5b1dd5c62ea17ec76fd17197ab18471ed222adf3173c3264b0a8f.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/87df62f095b5b1dd5c62ea17ec76fd17197ab18471ed222adf3173c3264b0a8f.jpg)

![948aba67a9573b85eced1e2ef1b43e5190d5916cd190ec0c95a3b4ea4dbdea0e.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/948aba67a9573b85eced1e2ef1b43e5190d5916cd190ec0c95a3b4ea4dbdea0e.jpg)

![affc82c9adc15629e4a0bf3d78fc5007542454b8cc0c22a6b1a245a6c4d43d0a.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/affc82c9adc15629e4a0bf3d78fc5007542454b8cc0c22a6b1a245a6c4d43d0a.jpg)

![cbb39cfde624d7dc21bc43317bac8a91cc2d4e1a1ad00b2fbe5f8c66d53262e5.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/cbb39cfde624d7dc21bc43317bac8a91cc2d4e1a1ad00b2fbe5f8c66d53262e5.jpg)

![d64d1d8ccc6d166aeaa3b18e09008767b69e6f3751116ec698e45daf9168330b.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/d64d1d8ccc6d166aeaa3b18e09008767b69e6f3751116ec698e45daf9168330b.jpg)

![d93800d45e05c65c335a19a9795b59baf57beeff5a3aa58c80983fc51690cdc0.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/images/d93800d45e05c65c335a19a9795b59baf57beeff5a3aa58c80983fc51690cdc0.jpg)

### Tables

![0f023a79c49d4c6e428debe5db7a6200475fe23094e0483cc214a9d32274ebbb.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/0f023a79c49d4c6e428debe5db7a6200475fe23094e0483cc214a9d32274ebbb.jpg)

![793660ce788e314efba3fd9ff6529ac57df5d380c4f94d412c4e3cf41b556162.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/793660ce788e314efba3fd9ff6529ac57df5d380c4f94d412c4e3cf41b556162.jpg)

![9f0ee1fa6df55ee7d11204c1b8411a179bacb13ced77fe73e500f2d2fc233dd2.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/9f0ee1fa6df55ee7d11204c1b8411a179bacb13ced77fe73e500f2d2fc233dd2.jpg)

![b5dd997fe88769f8d80de70b70a3368cd118140bd24e0952936b424eebeca045.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/b5dd997fe88769f8d80de70b70a3368cd118140bd24e0952936b424eebeca045.jpg)

![bb9ca3ea526ec937921dd30f24bb05da672929810eeffadd854d29a850fb6fbd.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/bb9ca3ea526ec937921dd30f24bb05da672929810eeffadd854d29a850fb6fbd.jpg)

![f06184a7167cb9c79cb958f238277c89e3cd45c85b140a73a885394561da5b91.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/f06184a7167cb9c79cb958f238277c89e3cd45c85b140a73a885394561da5b91.jpg)

![f69d5f66ef9caa90bb04dca06fda51cfae47a6bbb2f635d81f1e7f9c169cd04a.jpg](../icml_results/657_Zero-Shot%20Off-Policy%20Learning/tables/f69d5f66ef9caa90bb04dca06fda51cfae47a6bbb2f635d81f1e7f9c169cd04a.jpg)

<a id="paper-0649"></a>

## How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models


### Images

![0536ed71604e87d5ce4a4f67627afbe766534c6272f317cd04f7450c1333bd4f.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/0536ed71604e87d5ce4a4f67627afbe766534c6272f317cd04f7450c1333bd4f.jpg)

![1001f0dcbe45172727eb43c3186d1ceb387fe8da18a02aab34f1819c062c95be.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/1001f0dcbe45172727eb43c3186d1ceb387fe8da18a02aab34f1819c062c95be.jpg)

![1675dcf1d189bd9d1797de587cd1337b0cd53c86c1ac3ac427dcf83d86cbfec2.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/1675dcf1d189bd9d1797de587cd1337b0cd53c86c1ac3ac427dcf83d86cbfec2.jpg)

![276c8c99adc39e9d5491420ee3743fc22c75780280523dead19f744d4dfb9c13.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/276c8c99adc39e9d5491420ee3743fc22c75780280523dead19f744d4dfb9c13.jpg)

![38d6384ab81c6e809b2dc1912a50ba9b7bb95fbd3d3ccf7f0db4ce099ae23c76.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/38d6384ab81c6e809b2dc1912a50ba9b7bb95fbd3d3ccf7f0db4ce099ae23c76.jpg)

![4c3560dbedc98cd3d5070f2819534c6671fc824dcc76b1eb02b5998988a322ad.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/4c3560dbedc98cd3d5070f2819534c6671fc824dcc76b1eb02b5998988a322ad.jpg)

![56087e1678407942f60d93b9e7b51addbe30c571eba987af58702c581b62c498.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/56087e1678407942f60d93b9e7b51addbe30c571eba987af58702c581b62c498.jpg)

![78d72eefcfbcc13e145dc2a9163655c74670c4d02b8fff5863ddbfcda1c2a2da.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/78d72eefcfbcc13e145dc2a9163655c74670c4d02b8fff5863ddbfcda1c2a2da.jpg)

![ac159b637b9ba6be792132a189e002adb4a63a168bec107850be46d54e3cb02b.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/ac159b637b9ba6be792132a189e002adb4a63a168bec107850be46d54e3cb02b.jpg)

![bda9e07e0e9a0f0b661c605c1008b3dbe5f8bc939e9b5744cb00bc2942852d14.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/bda9e07e0e9a0f0b661c605c1008b3dbe5f8bc939e9b5744cb00bc2942852d14.jpg)

![c3b9ee1077375a581292551cf600c47041014f669b2ce09dbd7a450848abcd74.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/c3b9ee1077375a581292551cf600c47041014f669b2ce09dbd7a450848abcd74.jpg)

![cd87fceb0ee602c242c6af0e25833448e10576c8b7d651cf6fc14216be629adb.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/cd87fceb0ee602c242c6af0e25833448e10576c8b7d651cf6fc14216be629adb.jpg)

![eba618374cb9d78963aee5ac91c86d5bce8d4e94bacd9a39ca8eb56c09dae80d.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/eba618374cb9d78963aee5ac91c86d5bce8d4e94bacd9a39ca8eb56c09dae80d.jpg)

![f34fba6d1003ec53d31f6c2cf99b90756209b8992f03d1d5a21a11e4c7865395.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/f34fba6d1003ec53d31f6c2cf99b90756209b8992f03d1d5a21a11e4c7865395.jpg)

![f3a7d68c6eca75a0f4ffb417ed8f574e377e5b278ff055ea2ac71f122c89e684.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/images/f3a7d68c6eca75a0f4ffb417ed8f574e377e5b278ff055ea2ac71f122c89e684.jpg)

### Tables

![5e45fb8567c14a97366f83b16de37e54b705d7c7bf0b5e71683ddef6a4d4d7ba.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/tables/5e45fb8567c14a97366f83b16de37e54b705d7c7bf0b5e71683ddef6a4d4d7ba.jpg)

![cc4bb91f74b610fd9baebade951b25ff0e3bdef15dc98458475ba8458bfa121d.jpg](../icml_results/658_How%20to%20Train%20Your%20Advisor_%20Steering%20Black-Box%20LLMs%20with%20Advisor%20Models/tables/cc4bb91f74b610fd9baebade951b25ff0e3bdef15dc98458475ba8458bfa121d.jpg)

<a id="paper-0650"></a>

## WeightCLIP: Aligning Datasets and Models for Weight Space Learning


### Images

![1be4b8eaea04aeaf1f54b56e56c205b0ae4c53c9957377e52cf77cab07a4be87.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/1be4b8eaea04aeaf1f54b56e56c205b0ae4c53c9957377e52cf77cab07a4be87.jpg)

![20fe1f12d1278d481ccc189aff07636ced660132717508408b2e1e2115aa934c.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/20fe1f12d1278d481ccc189aff07636ced660132717508408b2e1e2115aa934c.jpg)

![320830ca711000d13f0d1ed687ddc0c060aa5c4e6a8fe8867674da296b5f9466.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/320830ca711000d13f0d1ed687ddc0c060aa5c4e6a8fe8867674da296b5f9466.jpg)

![42f668b7c2a1644c60f7ea5013aa4faf4a22502a43e2b46f293cc0c6c2fca389.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/42f668b7c2a1644c60f7ea5013aa4faf4a22502a43e2b46f293cc0c6c2fca389.jpg)

![5520351b2753aec64be6a55b73bdf2683bcf1830c167a65662cf6c0db69f095d.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/5520351b2753aec64be6a55b73bdf2683bcf1830c167a65662cf6c0db69f095d.jpg)

![59333bec7f445375ed02f33764b80ead6083fd5267a874c3d8c339d7bb374814.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/59333bec7f445375ed02f33764b80ead6083fd5267a874c3d8c339d7bb374814.jpg)

![79cc737bb326294652f150a785187ddf777f16a4e4b81a03cd02c093e9c1bcfa.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/79cc737bb326294652f150a785187ddf777f16a4e4b81a03cd02c093e9c1bcfa.jpg)

![ae886f7d8dd41214493bfa772fe0595b6cda4c4732c96ef1ff47b4b94147ebbd.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/ae886f7d8dd41214493bfa772fe0595b6cda4c4732c96ef1ff47b4b94147ebbd.jpg)

![d38dd753d7d994b4be5ac9358cd81e0f424c9268531cfc2396cefadaa4495c21.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/images/d38dd753d7d994b4be5ac9358cd81e0f424c9268531cfc2396cefadaa4495c21.jpg)

### Tables

![144089e5bd4868d9d4b7cd111d7c12758b8d80b32be14a7470cbdf0948af261c.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/144089e5bd4868d9d4b7cd111d7c12758b8d80b32be14a7470cbdf0948af261c.jpg)

![1640a26ce23171b4d7ee32a9dfb24d72aa5447208a70aa4a1ce16a49c3039122.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/1640a26ce23171b4d7ee32a9dfb24d72aa5447208a70aa4a1ce16a49c3039122.jpg)

![1a76e7b13d0db5685657d15aed57aabb848e8331cd6d247f7ca73d239c5455bd.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/1a76e7b13d0db5685657d15aed57aabb848e8331cd6d247f7ca73d239c5455bd.jpg)

![25a44af82b3ba07e580ceec9b2b2b804e18f2486fd46f51bd32e6e060a40b206.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/25a44af82b3ba07e580ceec9b2b2b804e18f2486fd46f51bd32e6e060a40b206.jpg)

![2f3a2df1e8fae66271d232b9d6361d207e1ce3e48ad95fcda06a3f92cba91511.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/2f3a2df1e8fae66271d232b9d6361d207e1ce3e48ad95fcda06a3f92cba91511.jpg)

![38aa7ad34fe67e834f0d73af3cbf9bc4241447ecdc976a342aeec24faa5f544e.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/38aa7ad34fe67e834f0d73af3cbf9bc4241447ecdc976a342aeec24faa5f544e.jpg)

![3a3daea1af2f0552454b5c3a1625267382462649a39c0e087ad13c801ea83384.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/3a3daea1af2f0552454b5c3a1625267382462649a39c0e087ad13c801ea83384.jpg)

![49d07e00bdd43808583df3686c7ce2c08132de658c0648eaaa490db87ac5e10a.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/49d07e00bdd43808583df3686c7ce2c08132de658c0648eaaa490db87ac5e10a.jpg)

![4b9800483adb6c0b9f95ea1ddfe802cf2401f98bd6d23334d071ae4e3fd76977.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/4b9800483adb6c0b9f95ea1ddfe802cf2401f98bd6d23334d071ae4e3fd76977.jpg)

![5bb2da41993c1bb01557ee36655e15a1cc77de539b429c16c7c492b74108c779.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/5bb2da41993c1bb01557ee36655e15a1cc77de539b429c16c7c492b74108c779.jpg)

![6b6bbc66ac84b40021f339f3f7118bd5b9f86ab89e087664558bf398e6f7f07b.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/6b6bbc66ac84b40021f339f3f7118bd5b9f86ab89e087664558bf398e6f7f07b.jpg)

![6f25485901209c7854a95788c4b4215a37ad4af4f72884ebe61bf4a4950b1ff3.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/6f25485901209c7854a95788c4b4215a37ad4af4f72884ebe61bf4a4950b1ff3.jpg)

![841878941eb5b91c17eaa7e3db3393412892bd957dafc038682e05726b91bb33.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/841878941eb5b91c17eaa7e3db3393412892bd957dafc038682e05726b91bb33.jpg)

![9b16495bc805e1763bfc8dd6e6a390d3b3f8ef85ebd27bc529936fd6f1478a60.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/9b16495bc805e1763bfc8dd6e6a390d3b3f8ef85ebd27bc529936fd6f1478a60.jpg)

![b7c63896d87b2d81701c2d4d28276e12d78650ef5f58b1273f638de6ad03a4e3.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/b7c63896d87b2d81701c2d4d28276e12d78650ef5f58b1273f638de6ad03a4e3.jpg)

![ee5485408c6194182bf6701f868bbf8f7c0f55e8e232b881f1ae6412440f4b2e.jpg](../icml_results/659_WeightCLIP_%20Aligning%20Datasets%20and%20Models%20for%20Weight%20Space%20Learning/tables/ee5485408c6194182bf6701f868bbf8f7c0f55e8e232b881f1ae6412440f4b2e.jpg)

<a id="paper-0651"></a>

## Why DDIM Hallucinates More Than DDPM: A Theoretical Analysis of Reverse Dynamics


### Images

![0ab684200ab72b443b5192bb80d17ac77dbd61a76e2e51c1c5eae05dc32b9268.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/0ab684200ab72b443b5192bb80d17ac77dbd61a76e2e51c1c5eae05dc32b9268.jpg)

![1a277368af499e5e23ecc94303034ca994eec57122d3d1b2ef3b7fc01121416d.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/1a277368af499e5e23ecc94303034ca994eec57122d3d1b2ef3b7fc01121416d.jpg)

![4f87459f4f35b6ecf95ab30c0c40612c110572d43a38cc2686c4ddf94629ce44.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/4f87459f4f35b6ecf95ab30c0c40612c110572d43a38cc2686c4ddf94629ce44.jpg)

![5227f8d2ab5eef743b7fdcf4ca804039394a4d3c977db66c02cf1c74fb92ea4d.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/5227f8d2ab5eef743b7fdcf4ca804039394a4d3c977db66c02cf1c74fb92ea4d.jpg)

![5b9808d4a2ac0caef896e04f79aa6dc43af55b5a39a3542d0546a5a93f10f8aa.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/5b9808d4a2ac0caef896e04f79aa6dc43af55b5a39a3542d0546a5a93f10f8aa.jpg)

![6b067fe0bcd30e7df38ebbaae2c816a653c5657cef85a0614de4f4e40a6f7450.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/6b067fe0bcd30e7df38ebbaae2c816a653c5657cef85a0614de4f4e40a6f7450.jpg)

![70fb67903d76a9b4c5303b6105b3be89fce73b44b7d685382a640c58bcbf5d74.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/70fb67903d76a9b4c5303b6105b3be89fce73b44b7d685382a640c58bcbf5d74.jpg)

![73328154f72f737651373719c93ea0afbb282a02cd92ab1af30f22f0c1719586.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/73328154f72f737651373719c93ea0afbb282a02cd92ab1af30f22f0c1719586.jpg)

![843f08288e1b29b94f60290cf57c6a4c36b5cf17a3b265ce70651366efa3838c.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/843f08288e1b29b94f60290cf57c6a4c36b5cf17a3b265ce70651366efa3838c.jpg)

![8859d7dc347bb094c16b8d75c680b62617e05ca86977bc830fa14dc6922f5883.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/8859d7dc347bb094c16b8d75c680b62617e05ca86977bc830fa14dc6922f5883.jpg)

![8e735fd8f425cbc6454b39e793208c770a87d0414d0bbd77a2ad76aefda0a949.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/8e735fd8f425cbc6454b39e793208c770a87d0414d0bbd77a2ad76aefda0a949.jpg)

![991a46daa01a73bd4e49601831bd6991b5859a7869705dbeea5d8f963bd43c59.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/991a46daa01a73bd4e49601831bd6991b5859a7869705dbeea5d8f963bd43c59.jpg)

![aee9d43a0846927b45609908e2215113d574bd9b8005798773ef3604001ca184.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/aee9d43a0846927b45609908e2215113d574bd9b8005798773ef3604001ca184.jpg)

![b618f2794082361db54afafa4b6271d029f027bf530de0e8acc6483ccf833900.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/b618f2794082361db54afafa4b6271d029f027bf530de0e8acc6483ccf833900.jpg)

![b9495c4972ca4a1364b9b0dd24b9a6922476c4124b1a5cdc38f3f6f718e62b73.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/b9495c4972ca4a1364b9b0dd24b9a6922476c4124b1a5cdc38f3f6f718e62b73.jpg)

![c5dbdcf44173aff7508bb371d8184e92a4dce1c86fb490370343cb4643e45083.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/c5dbdcf44173aff7508bb371d8184e92a4dce1c86fb490370343cb4643e45083.jpg)

![dd1c64da2806026909013838a70a0d9a0cf8bd3ddfecf71378a95aebf176ca33.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/dd1c64da2806026909013838a70a0d9a0cf8bd3ddfecf71378a95aebf176ca33.jpg)

![ebe2ee86c54db4a7f99b4f020bc880d2305445a692d4b416fb3e78ab899d37d3.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/ebe2ee86c54db4a7f99b4f020bc880d2305445a692d4b416fb3e78ab899d37d3.jpg)

![fb2e835990b91b250e61d75e599e891774c94084db9349aa096e6cf2e30a498f.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/fb2e835990b91b250e61d75e599e891774c94084db9349aa096e6cf2e30a498f.jpg)

![fb5d2c39cf49683d98edcab3506735c86bd06de29b601bf52341bf385fbdccb8.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/fb5d2c39cf49683d98edcab3506735c86bd06de29b601bf52341bf385fbdccb8.jpg)

![fbfcf1aa71903ef3bfe25f4787a931cdd1cdf94586d88da21817c5f0588b5985.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/images/fbfcf1aa71903ef3bfe25f4787a931cdd1cdf94586d88da21817c5f0588b5985.jpg)

### Tables

![1d0bb2bed2470057e93ce0b217d98f8bc4c2ecc40ca06882f33a1460500159f4.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/tables/1d0bb2bed2470057e93ce0b217d98f8bc4c2ecc40ca06882f33a1460500159f4.jpg)

![eddbd7cdff2c08d9ba2277e535d2c4ae70af72978b45457ce61c4863b9acd8f3.jpg](../icml_results/660_Why%20DDIM%20Hallucinates%20More%20Than%20DDPM_%20A%20Theoretical%20Analysis%20of%20Reverse%20Dynamics/tables/eddbd7cdff2c08d9ba2277e535d2c4ae70af72978b45457ce61c4863b9acd8f3.jpg)

<a id="paper-0652"></a>

## Robust Bayes-Assisted Conformal Prediction


### Images

![1c94ba3cf25df4217f7189c60e79af9b08c7e21033c0c52fed2979f9c26a6907.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/1c94ba3cf25df4217f7189c60e79af9b08c7e21033c0c52fed2979f9c26a6907.jpg)

![3cf0de4f2806658100e11bb0e2b44fe00b72324a9babdd08e046e0aa781d5742.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/3cf0de4f2806658100e11bb0e2b44fe00b72324a9babdd08e046e0aa781d5742.jpg)

![3d10f681676e7359d093dd579d57dee0c709582529422a88e130ed91019213c1.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/3d10f681676e7359d093dd579d57dee0c709582529422a88e130ed91019213c1.jpg)

![808de4f6633a01f7146ee4735d7836dd9c21d6d5de984ebc9dc52c94209a3c72.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/808de4f6633a01f7146ee4735d7836dd9c21d6d5de984ebc9dc52c94209a3c72.jpg)

![820b1f8d4dc3f5943e34d26a29013d879532ff99327a89a0ec720fc9a6fd19e6.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/820b1f8d4dc3f5943e34d26a29013d879532ff99327a89a0ec720fc9a6fd19e6.jpg)

![b09f4e55549a3e1b174c24a841bda6f4cbc9ea1d1349aa1c288e43eb553f90b6.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/b09f4e55549a3e1b174c24a841bda6f4cbc9ea1d1349aa1c288e43eb553f90b6.jpg)

![e96e1ab65f807df3d396c6534d41fde69d6417653314fefd347003bfd59dc18b.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/e96e1ab65f807df3d396c6534d41fde69d6417653314fefd347003bfd59dc18b.jpg)

![f4aafaf986b28223cb688b82a1886d1b07034ce124f801035c5b74f595ac9a97.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/images/f4aafaf986b28223cb688b82a1886d1b07034ce124f801035c5b74f595ac9a97.jpg)

### Tables

![0d5403536031bd60adcc774e60b8231aebe8f259ff4305d4e9aad20f9fdf5789.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/0d5403536031bd60adcc774e60b8231aebe8f259ff4305d4e9aad20f9fdf5789.jpg)

![22fc84b96edcdf4bbafc4adac8079ab1bb34c0a98659af44cdd8eaf575be9758.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/22fc84b96edcdf4bbafc4adac8079ab1bb34c0a98659af44cdd8eaf575be9758.jpg)

![61483c9d13b7d4bb00c627d90d8a05cd502d9fb266678d6da514e7905ec26a14.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/61483c9d13b7d4bb00c627d90d8a05cd502d9fb266678d6da514e7905ec26a14.jpg)

![65000c5459fd13b286ef52508921443bea5545d7e0f9b67534cffe93f9c7153f.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/65000c5459fd13b286ef52508921443bea5545d7e0f9b67534cffe93f9c7153f.jpg)

![ca5b3091e3732d07591c4812e029f6878d3361d6b42d596fd1f93e4ee65adf10.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/ca5b3091e3732d07591c4812e029f6878d3361d6b42d596fd1f93e4ee65adf10.jpg)

![dd195fef4d8dcfff89dc5d6ab1812aad8abe4591ce00fd63aa1963bd687e2e41.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/dd195fef4d8dcfff89dc5d6ab1812aad8abe4591ce00fd63aa1963bd687e2e41.jpg)

![f6f691b51a0ade6b0d7080b153fbbc529e63bf068d1c6ae0360a3eefb35115bd.jpg](../icml_results/661_Robust%20Bayes-Assisted%20Conformal%20Prediction/tables/f6f691b51a0ade6b0d7080b153fbbc529e63bf068d1c6ae0360a3eefb35115bd.jpg)

<a id="paper-0653"></a>

## The Crowded Embedding Space: A Mean-Field Mechanism for Emergent Marginalization in Retrieval-Augmented Agents


### Images

![23aafe5240ca25e052021fed6311eccdbb4cb01626b789f2ff198f32d6654786.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/23aafe5240ca25e052021fed6311eccdbb4cb01626b789f2ff198f32d6654786.jpg)

![51f4b940ce0b6da5ca468c71b9e74e58d2a09c733af49b29d799e1ace6407a1c.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/51f4b940ce0b6da5ca468c71b9e74e58d2a09c733af49b29d799e1ace6407a1c.jpg)

![6d67196c932954bf38b40e9bd675145e7f080930651b408a72f05c6e0e2b17e2.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/6d67196c932954bf38b40e9bd675145e7f080930651b408a72f05c6e0e2b17e2.jpg)

![9099129745e80fd085611905898928b11948bb63a3b8bb38eb52a8d75f6b1c99.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/9099129745e80fd085611905898928b11948bb63a3b8bb38eb52a8d75f6b1c99.jpg)

![f1c6965783386352e8b309663ad190814ad7b9c82badf952154ff3dfc76900a1.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/f1c6965783386352e8b309663ad190814ad7b9c82badf952154ff3dfc76900a1.jpg)

![f589986410c3fe08e562b361d72d04b1779dbb7c164cc544f2b303bd84f9a1a5.jpg](../icml_results/662_The%20Crowded%20Embedding%20Space_%20A%20Mean-Field%20Mechanism%20for%20Emergent%20Marginalization%20in%20Retrieval-Augmen/images/f589986410c3fe08e562b361d72d04b1779dbb7c164cc544f2b303bd84f9a1a5.jpg)

<a id="paper-0654"></a>

## Fast and Near-Optimal Algorithms for Private Hypothesis Selection


### Tables

![76a2d1f6b3f0a7fc7fa25d480a5e649982ea89c020e56bad275d1ed3598e611e.jpg](../icml_results/663_Fast%20and%20Near-Optimal%20Algorithms%20for%20Private%20Hypothesis%20Selection/tables/76a2d1f6b3f0a7fc7fa25d480a5e649982ea89c020e56bad275d1ed3598e611e.jpg)

![9eeed991bf20ca40549137bf10a49a7fc632fa30e87aaf22a8c2b9abe06202ae.jpg](../icml_results/663_Fast%20and%20Near-Optimal%20Algorithms%20for%20Private%20Hypothesis%20Selection/tables/9eeed991bf20ca40549137bf10a49a7fc632fa30e87aaf22a8c2b9abe06202ae.jpg)

<a id="paper-0655"></a>

## PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs


### Images

![208e2533444ab22cd85f29788807829643fd81945caa91649c414037a3121ed8.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/208e2533444ab22cd85f29788807829643fd81945caa91649c414037a3121ed8.jpg)

![5067e11eb459ab64e95db9f896b8f75bc224e94589daf5586d89e7042d648ede.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/5067e11eb459ab64e95db9f896b8f75bc224e94589daf5586d89e7042d648ede.jpg)

![5e7475187d7e87bd0ac481840e36760587304d6a4e563964c65d1bb041b80f93.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/5e7475187d7e87bd0ac481840e36760587304d6a4e563964c65d1bb041b80f93.jpg)

![6b4eaddf56e13b0ad1f693e6407e7a131b368bba3057f584a13da5fd906b3c23.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/6b4eaddf56e13b0ad1f693e6407e7a131b368bba3057f584a13da5fd906b3c23.jpg)

![6d7fd0dde275a7b9f53ba61b532cc31ac97c297d18e0e3494352e46c31f98bfe.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/6d7fd0dde275a7b9f53ba61b532cc31ac97c297d18e0e3494352e46c31f98bfe.jpg)

![e2c882a43b5b377ec57f3f98209d63af962a0a9a50face814affdb9e041c3028.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/e2c882a43b5b377ec57f3f98209d63af962a0a9a50face814affdb9e041c3028.jpg)

![f2feaef97d5022146f0d147f268ecbd59f91d09cc9be04fd731516338a7d8028.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/f2feaef97d5022146f0d147f268ecbd59f91d09cc9be04fd731516338a7d8028.jpg)

![f85392d24540b408c316fd9ebad95119c524d66d944e80d4609a708a48b52b92.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/images/f85392d24540b408c316fd9ebad95119c524d66d944e80d4609a708a48b52b92.jpg)

### Tables

![463db07e10a6c91435df66e27840768bbe5c36695f4b2533f5b5219eb2a9eb08.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/463db07e10a6c91435df66e27840768bbe5c36695f4b2533f5b5219eb2a9eb08.jpg)

![4bd5326ff6c7ff9623e2c849d91118a0e8404a0916b949b76772760a83652998.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/4bd5326ff6c7ff9623e2c849d91118a0e8404a0916b949b76772760a83652998.jpg)

![5cf325dfddd3f02522560ef94354d2c9fdfba030082bd74a704ac6ff19a038ca.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/5cf325dfddd3f02522560ef94354d2c9fdfba030082bd74a704ac6ff19a038ca.jpg)

![75e4476831974493879e2cab9ce3c6e884ce83951c6d6e874b593cb73538304e.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/75e4476831974493879e2cab9ce3c6e884ce83951c6d6e874b593cb73538304e.jpg)

![7f13568f5fa6dd604154715942e4e97f9447dce26c5094ff9e3937904c7db7d1.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/7f13568f5fa6dd604154715942e4e97f9447dce26c5094ff9e3937904c7db7d1.jpg)

![a5e7b2ace163cfe7401ab7b4b2a5ae8a45512466f40ba175830b3e0e34d517fe.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/a5e7b2ace163cfe7401ab7b4b2a5ae8a45512466f40ba175830b3e0e34d517fe.jpg)

![a9c5f08ed99c087eca25091ec67cd57101aa9d8af2f47baebbb4e4ce118a03fd.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/a9c5f08ed99c087eca25091ec67cd57101aa9d8af2f47baebbb4e4ce118a03fd.jpg)

![cdf27a4b2510fffca83824213c885cf19838f7eb41bd248e7201b5eec501bc2a.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/cdf27a4b2510fffca83824213c885cf19838f7eb41bd248e7201b5eec501bc2a.jpg)

![e7687f90ef580c481b41969de6ff82da2b298a99f4d06655789b3df26de1d07f.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/e7687f90ef580c481b41969de6ff82da2b298a99f4d06655789b3df26de1d07f.jpg)

![e77635ea0faf17475a973471f54d4e0392e48574496269104b8431b7682678a5.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/e77635ea0faf17475a973471f54d4e0392e48574496269104b8431b7682678a5.jpg)

![f1d0bd192833713a7f91dcdbe70123c3032a2cd0b10c0f7cd57e8b08db85fbaf.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/f1d0bd192833713a7f91dcdbe70123c3032a2cd0b10c0f7cd57e8b08db85fbaf.jpg)

![f3529fcf57a183cb2cf14b56af905f72a887088c222599a2f0cf1b2b98a90a01.jpg](../icml_results/664_PGT_%20Procedurally%20Generated%20Tasks%20for%20improving%20visual%20grounding%20in%20MLLMs/tables/f3529fcf57a183cb2cf14b56af905f72a887088c222599a2f0cf1b2b98a90a01.jpg)

<a id="paper-0656"></a>

## Grounding Functional Similarity by Invariance-Aware Model Stitching


### Images

![02e5f53bd1b6a0013977b4ea1bc4004fb84f3f274e2247b441be6c17702deaa7.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/02e5f53bd1b6a0013977b4ea1bc4004fb84f3f274e2247b441be6c17702deaa7.jpg)

![177c9f067caaa72ddb6959360fdef3e7b4a15cf4803c326f9d75d21bc5f8a582.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/177c9f067caaa72ddb6959360fdef3e7b4a15cf4803c326f9d75d21bc5f8a582.jpg)

![1b88377a7ffdd40638c9233178929efcd42fe7038b7412f56000c6c0019010f1.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/1b88377a7ffdd40638c9233178929efcd42fe7038b7412f56000c6c0019010f1.jpg)

![2ce3408ea188797df06489ff04b70b566492ab49eff8c9c17cb466107859c939.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/2ce3408ea188797df06489ff04b70b566492ab49eff8c9c17cb466107859c939.jpg)

![7e329a4703586572983f4ddf8739ba18c3865c04fb621f616780b96a051fe671.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/7e329a4703586572983f4ddf8739ba18c3865c04fb621f616780b96a051fe671.jpg)

![8572513cd5ddf9baa4a35807a843b1749f4624e9db07fdd5eff6d1e3b439e025.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/8572513cd5ddf9baa4a35807a843b1749f4624e9db07fdd5eff6d1e3b439e025.jpg)

![8681bfb0b71b921def007b7ebf4c9596fda09aa7d5e35b8cc6bf0508b69855e1.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/8681bfb0b71b921def007b7ebf4c9596fda09aa7d5e35b8cc6bf0508b69855e1.jpg)

![9b6a1b6cd32b26105a600801c1549949940399b4c0f7b9dfb3b8125572cfd4d7.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/9b6a1b6cd32b26105a600801c1549949940399b4c0f7b9dfb3b8125572cfd4d7.jpg)

![b546ec5f0cc30236af7698ec3c62c83aa891474ec1a7802a5eae4f5b114a1172.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/b546ec5f0cc30236af7698ec3c62c83aa891474ec1a7802a5eae4f5b114a1172.jpg)

![b86d1146c926221fcde40bc68f69bc2f7720d47ca60d89c1799327040603e6c4.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/b86d1146c926221fcde40bc68f69bc2f7720d47ca60d89c1799327040603e6c4.jpg)

![b97d3750b03ab684129095a01a453022ad29013e30b5ce9aca4730bae47ae4bc.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/b97d3750b03ab684129095a01a453022ad29013e30b5ce9aca4730bae47ae4bc.jpg)

![b9dcda57db3a56eb343c86144ef5e0e62c0d0dc503be5611f4e2d97293e8b79b.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/b9dcda57db3a56eb343c86144ef5e0e62c0d0dc503be5611f4e2d97293e8b79b.jpg)

![c813b5a28777f4d9611a0613f41e62543d91fe6dd3d74a352eeb76b781ead0db.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/c813b5a28777f4d9611a0613f41e62543d91fe6dd3d74a352eeb76b781ead0db.jpg)

![da5ee989412051c43dc3f1ac6babf71987226f8d0bff04c61f9e6077dd79f308.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/da5ee989412051c43dc3f1ac6babf71987226f8d0bff04c61f9e6077dd79f308.jpg)

![ec5db7cb2686a8f0d1d58d84fda4ef887466bc8a7aff3a301a4c1e42b989f0f5.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/ec5db7cb2686a8f0d1d58d84fda4ef887466bc8a7aff3a301a4c1e42b989f0f5.jpg)

![ecf3ab740e513413ca280ac30d12ae423498179a11649913c518ee58e3527bc5.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/ecf3ab740e513413ca280ac30d12ae423498179a11649913c518ee58e3527bc5.jpg)

![f5d253ec771076d78d0f1e40b85d52cbcf19925ff8ce35d310c09234f0cc42c6.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/f5d253ec771076d78d0f1e40b85d52cbcf19925ff8ce35d310c09234f0cc42c6.jpg)

![fa00d81a51efa20035e33d8ee0170eac32b1c4e1891c72036df903cf6e2a5d94.jpg](../icml_results/665_Grounding%20Functional%20Similarity%20by%20Invariance-Aware%20Model%20Stitching/images/fa00d81a51efa20035e33d8ee0170eac32b1c4e1891c72036df903cf6e2a5d94.jpg)

<a id="paper-0657"></a>

## Learning Rate Annealing Improves Tuning Robustness in Stochastic Optimization


### Images

![1a9259106d74b9f3f307566093c17f382bcccc31e5499b09dd7b2f192d858db5.jpg](../icml_results/666_Learning%20Rate%20Annealing%20Improves%20Tuning%20Robustness%20in%20Stochastic%20Optimization/images/1a9259106d74b9f3f307566093c17f382bcccc31e5499b09dd7b2f192d858db5.jpg)

![79802422654a6531bb10ba872c2c6ac941af7e961810b7ab9d4e0d8c957dcdb0.jpg](../icml_results/666_Learning%20Rate%20Annealing%20Improves%20Tuning%20Robustness%20in%20Stochastic%20Optimization/images/79802422654a6531bb10ba872c2c6ac941af7e961810b7ab9d4e0d8c957dcdb0.jpg)

![d94bfd97c24b9a8f4288fbd13ffd4536c86e3bfdbc4dc4d58a62be93672e748a.jpg](../icml_results/666_Learning%20Rate%20Annealing%20Improves%20Tuning%20Robustness%20in%20Stochastic%20Optimization/images/d94bfd97c24b9a8f4288fbd13ffd4536c86e3bfdbc4dc4d58a62be93672e748a.jpg)

<a id="paper-0658"></a>

## Intrinsic Credit Assignment for Long Horizon Interaction


### Images

![1d3ad0c528a96d25d5d476b5cbd6ec006810fa7a6b5dcd8c067c420f78e7f531.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/1d3ad0c528a96d25d5d476b5cbd6ec006810fa7a6b5dcd8c067c420f78e7f531.jpg)

![2a07872ebc70651f3e80dbbc3bc569c6543568b05040db70924a8e239751b458.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/2a07872ebc70651f3e80dbbc3bc569c6543568b05040db70924a8e239751b458.jpg)

![36470bcc1fbb453e0c21be1844ded7c9ceb906c05f5656c9d0dd38ecfa28d79a.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/36470bcc1fbb453e0c21be1844ded7c9ceb906c05f5656c9d0dd38ecfa28d79a.jpg)

![40a8a45b1a57081c0d76e4494aa61efc4dc34220ef38921f86241553d18d2356.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/40a8a45b1a57081c0d76e4494aa61efc4dc34220ef38921f86241553d18d2356.jpg)

![499eef61390265275d9b994ac38858b1c3a65003d9d0d54cd3f476950a1a031a.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/499eef61390265275d9b994ac38858b1c3a65003d9d0d54cd3f476950a1a031a.jpg)

![560575e641f00952ca32ac863da36a9a1d0a23a9db62d997e98bd37ea4caff53.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/560575e641f00952ca32ac863da36a9a1d0a23a9db62d997e98bd37ea4caff53.jpg)

![74af07f165afd17a3ccff70ed6b7d92d85be7d53efb334aa81c55b5467eed060.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/74af07f165afd17a3ccff70ed6b7d92d85be7d53efb334aa81c55b5467eed060.jpg)

![7951b45127e68784859fa469c4ba53e1eb3e8c66803acece111d582423991119.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/7951b45127e68784859fa469c4ba53e1eb3e8c66803acece111d582423991119.jpg)

![9807c36d8eb10c7fa285edd5558767389be6941ba26fa049f1b5ded312ab7f87.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/9807c36d8eb10c7fa285edd5558767389be6941ba26fa049f1b5ded312ab7f87.jpg)

![98427cbd9f9ceee9fe67e707ada4c12da8afc75d6de889ce35c42d79a1342b11.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/98427cbd9f9ceee9fe67e707ada4c12da8afc75d6de889ce35c42d79a1342b11.jpg)

![9fabd6d7787e5ba551fc83c8f6312679b47528b82427e3532a495535955d5ed7.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/9fabd6d7787e5ba551fc83c8f6312679b47528b82427e3532a495535955d5ed7.jpg)

![a2ec49d260386412ab952dea70ba15212f05b39d555c5b151ddf96f112093dfb.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/a2ec49d260386412ab952dea70ba15212f05b39d555c5b151ddf96f112093dfb.jpg)

![a35efb356bc8a8ef6ae7410c107d7836a3449acb5c04b419aa77e159f9ae4d0b.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/a35efb356bc8a8ef6ae7410c107d7836a3449acb5c04b419aa77e159f9ae4d0b.jpg)

![a845f1ca4c1b020b81e6d98c2bd47bb19b819426e556bed8a8ea2811ba013009.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/a845f1ca4c1b020b81e6d98c2bd47bb19b819426e556bed8a8ea2811ba013009.jpg)

![af6a4c2ca2c28a7f364695f82f3b77b05452ff3b2b17f6aae28fa3e5e40b1c3b.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/af6a4c2ca2c28a7f364695f82f3b77b05452ff3b2b17f6aae28fa3e5e40b1c3b.jpg)

![b5dc514473b2c26ce6af64b49e33f509217c257b022694991df0703eedd1f58a.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/b5dc514473b2c26ce6af64b49e33f509217c257b022694991df0703eedd1f58a.jpg)

![dc933c7b9f90be3bbf9117422376897cf878bca7c8fef4cc3f910f03dd58ca26.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/dc933c7b9f90be3bbf9117422376897cf878bca7c8fef4cc3f910f03dd58ca26.jpg)

![e2f7465ff6426b986fd38c664e86ed5e235f8cd8dfd45d268a5d073f7a7da0f0.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/e2f7465ff6426b986fd38c664e86ed5e235f8cd8dfd45d268a5d073f7a7da0f0.jpg)

![e8de627bf879c7994d3f36f3d1b46a27b5d975ac7509078f5d810aec1264322f.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/e8de627bf879c7994d3f36f3d1b46a27b5d975ac7509078f5d810aec1264322f.jpg)

![f72fac908912c941b46968f4fd1f6617469ae41ea118816757bd3f83c42edbd0.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/images/f72fac908912c941b46968f4fd1f6617469ae41ea118816757bd3f83c42edbd0.jpg)

### Tables

![6d059abccc9fab10072de579d1ca4bb40d6dcd95d88adae7bbb9d9c3acc52465.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/6d059abccc9fab10072de579d1ca4bb40d6dcd95d88adae7bbb9d9c3acc52465.jpg)

![6de3378cffcb996b7849a8c94634284600eff8faa3d5eaa7eaa66b50da117d60.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/6de3378cffcb996b7849a8c94634284600eff8faa3d5eaa7eaa66b50da117d60.jpg)

![7ffe7c3e16d0d5b1a1a6085bd5ee03368455328b68ac7653788572d97f834b7d.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/7ffe7c3e16d0d5b1a1a6085bd5ee03368455328b68ac7653788572d97f834b7d.jpg)

![85619e3a295d2b5aa08f5672f5543ea7d904db33e819cc48bd3b5aea5b11ca84.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/85619e3a295d2b5aa08f5672f5543ea7d904db33e819cc48bd3b5aea5b11ca84.jpg)

![9740e8339fa5a1afd94b0cc6d21777c2ff5759b2c280da1250ac03dcc13bfe0d.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/9740e8339fa5a1afd94b0cc6d21777c2ff5759b2c280da1250ac03dcc13bfe0d.jpg)

![d979b6c6e3bcfb26f07dedc46c64f654a420422bd124112270c7bd8c76889d1a.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/d979b6c6e3bcfb26f07dedc46c64f654a420422bd124112270c7bd8c76889d1a.jpg)

![e7ec55b300ea4d28f962f6d24bc100f82c294718e2f12e95ce2b8bbdda3adc02.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/e7ec55b300ea4d28f962f6d24bc100f82c294718e2f12e95ce2b8bbdda3adc02.jpg)

![f58bee14cdc70a8fd7180f8e8cbd47c7518e94b08b3b26a49b65143728a704fb.jpg](../icml_results/667_Intrinsic%20Credit%20Assignment%20for%20Long%20Horizon%20Interaction/tables/f58bee14cdc70a8fd7180f8e8cbd47c7518e94b08b3b26a49b65143728a704fb.jpg)

<a id="paper-0659"></a>

## SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs


### Images

![0ab4f06e397186446362cf3789517bbfb694010fe4a4d95b9bb5fa032571edc1.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/0ab4f06e397186446362cf3789517bbfb694010fe4a4d95b9bb5fa032571edc1.jpg)

![28fedb8b0cb1f0b46bd9e77db7f1a13ca7905c4370edbc84e3f06738711dc85e.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/28fedb8b0cb1f0b46bd9e77db7f1a13ca7905c4370edbc84e3f06738711dc85e.jpg)

![2e21e4a0d8da2c27c8205fdcae0403376d232c4a087457513c1edf01ca4fb4d8.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/2e21e4a0d8da2c27c8205fdcae0403376d232c4a087457513c1edf01ca4fb4d8.jpg)

![349c40dd7b38de63a5bcb6f1e74842e14c311fd37ef610dc577d66000b011033.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/349c40dd7b38de63a5bcb6f1e74842e14c311fd37ef610dc577d66000b011033.jpg)

![429f6c4dea63a4702a4e0b80554929f288e16f1e4c531c6b6775dd6034597d64.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/429f6c4dea63a4702a4e0b80554929f288e16f1e4c531c6b6775dd6034597d64.jpg)

![44a4891dfbd487a390138e8de425e97f51d4d76c3d27752e86048eb45674377c.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/44a4891dfbd487a390138e8de425e97f51d4d76c3d27752e86048eb45674377c.jpg)

![59dda33c2d9f812b679d079f7a190ea25636ff0f86f254abaf2c0c8fd7377adf.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/59dda33c2d9f812b679d079f7a190ea25636ff0f86f254abaf2c0c8fd7377adf.jpg)

![7ffb620656dd83b08e0c3d38a4b46e06dd94a5c16b10e1f36cccb89b095bbdf3.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/7ffb620656dd83b08e0c3d38a4b46e06dd94a5c16b10e1f36cccb89b095bbdf3.jpg)

![a288dc63426db81c6d5a8b580be5ec694f76b056bd69d9036e7b21df3efa2e08.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/a288dc63426db81c6d5a8b580be5ec694f76b056bd69d9036e7b21df3efa2e08.jpg)

![d3f81c7502b89ae2c16348427c2fd51c50f281532f26d75ca21d9e26fa1a2601.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/images/d3f81c7502b89ae2c16348427c2fd51c50f281532f26d75ca21d9e26fa1a2601.jpg)

### Tables

![0825b5e02a2fd2ca1af3b52160690428c112262d9faaa9ee2bcd74eefa816cda.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/0825b5e02a2fd2ca1af3b52160690428c112262d9faaa9ee2bcd74eefa816cda.jpg)

![0f3bf48c8a565887e2453e3a085c33b1e64b2c897cb171f229f70506cc80296b.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/0f3bf48c8a565887e2453e3a085c33b1e64b2c897cb171f229f70506cc80296b.jpg)

![1b558bd42cc1763c8d522bc4e3d5a8010ed5ba021b60dc9efea024c5a4f386b5.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/1b558bd42cc1763c8d522bc4e3d5a8010ed5ba021b60dc9efea024c5a4f386b5.jpg)

![2ef33398d97d6b6dedf1d34c253b64f9218ddded4d9389814585e66e22f5c49c.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/2ef33398d97d6b6dedf1d34c253b64f9218ddded4d9389814585e66e22f5c49c.jpg)

![3b8d4772c23bdc9439ab2f37805cf0f1bf7439c0b19d752edd7c855d4c6637b1.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/3b8d4772c23bdc9439ab2f37805cf0f1bf7439c0b19d752edd7c855d4c6637b1.jpg)

![53a27867c60a6b79bfa0148ed21dd0e9e227d2c8d72e43b801620442b2c2cc30.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/53a27867c60a6b79bfa0148ed21dd0e9e227d2c8d72e43b801620442b2c2cc30.jpg)

![5c84f925d5a5843634b9cf39052711ade52b7d71dd7d21306990dca41f29a635.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/5c84f925d5a5843634b9cf39052711ade52b7d71dd7d21306990dca41f29a635.jpg)

![69e5ea25873a5299a999028da07121c2dbebfad65b05584a2c1d4bb0b79e002d.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/69e5ea25873a5299a999028da07121c2dbebfad65b05584a2c1d4bb0b79e002d.jpg)

![7031ed073654791dac75db7640eb870c4e5c51afb997904f98fc88fc0b54353f.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/7031ed073654791dac75db7640eb870c4e5c51afb997904f98fc88fc0b54353f.jpg)

![819d16f605cf227f0d47f8abcfda70860adddc3924c3fe062c96ace842091a35.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/819d16f605cf227f0d47f8abcfda70860adddc3924c3fe062c96ace842091a35.jpg)

![87c13ee9c259a03824ba3395ed8451b33d5107dcab7ae8619229f68ef8c562aa.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/87c13ee9c259a03824ba3395ed8451b33d5107dcab7ae8619229f68ef8c562aa.jpg)

![ac7dd24a9cb2d00a802d6e3efb845fec3c20a38ab4a7669c3135bf29f4150a13.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/ac7dd24a9cb2d00a802d6e3efb845fec3c20a38ab4a7669c3135bf29f4150a13.jpg)

![b0e482dcba44375f962842eb6d63ac7680614cf366a604550afb1f25edb19e4d.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/b0e482dcba44375f962842eb6d63ac7680614cf366a604550afb1f25edb19e4d.jpg)

![c90563e42a2873e55168442751c184ff1631b4a937a76d1a2889b3593a8c4586.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/c90563e42a2873e55168442751c184ff1631b4a937a76d1a2889b3593a8c4586.jpg)

![e6bcbc3523eae2e7f0718f8c9909d1d64747bb9282b888ca77e259a766908c06.jpg](../icml_results/669_SPARC_%20Separating%20Perception%20And%20Reasoning%20Circuits%20for%20Test-time%20Scaling%20of%20VLMs/tables/e6bcbc3523eae2e7f0718f8c9909d1d64747bb9282b888ca77e259a766908c06.jpg)

<a id="paper-0660"></a>

## Reducing Per-Sample Interference in Stochastic Optimization

### Images

![05fe8f6f9339f154543c99233b12657a85c96bbfd955cb9d2e6534a781fc67d4.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/05fe8f6f9339f154543c99233b12657a85c96bbfd955cb9d2e6534a781fc67d4.jpg)

![1b638b23b154202ef2754364f001117fbeab86e80c386b1195dee5182e89172a.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/1b638b23b154202ef2754364f001117fbeab86e80c386b1195dee5182e89172a.jpg)

![1e6b97978400b82e6db3367cba06d45601f8e149cddc86aecad1ba863009eccf.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/1e6b97978400b82e6db3367cba06d45601f8e149cddc86aecad1ba863009eccf.jpg)

![32bbb1dd7f103dab0f4920502fa827ee6214cf1fb3756c1972fd22557a991c26.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/32bbb1dd7f103dab0f4920502fa827ee6214cf1fb3756c1972fd22557a991c26.jpg)

![4498e7058376e3a7f132c10a916aac04f2f60a1ee12d013b531d9f5d1d70d7f0.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/4498e7058376e3a7f132c10a916aac04f2f60a1ee12d013b531d9f5d1d70d7f0.jpg)

![783949fcf6e66a6d1b306e1c1eb091417c94d4e8455fe922c38db219be8a2dcb.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/783949fcf6e66a6d1b306e1c1eb091417c94d4e8455fe922c38db219be8a2dcb.jpg)

![81703e2fe7faec10e9a66da3bc8e36cb62eb5b186efa420eb7f8f43e8d1a4e73.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/81703e2fe7faec10e9a66da3bc8e36cb62eb5b186efa420eb7f8f43e8d1a4e73.jpg)

![8b65bf821cbaf551fa0109fdc9359f22d5cf093f219e14c99474d5c4d8303d1b.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/8b65bf821cbaf551fa0109fdc9359f22d5cf093f219e14c99474d5c4d8303d1b.jpg)

![acb6085c0fbaffa31dae7c916bff628e56464c126fdd2f27468ee91d1a571780.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/acb6085c0fbaffa31dae7c916bff628e56464c126fdd2f27468ee91d1a571780.jpg)

![d8085032883ca1dbfad7f622b9a3eb5f5df74c294f70a2f4e92e7c9d40ffacf8.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/d8085032883ca1dbfad7f622b9a3eb5f5df74c294f70a2f4e92e7c9d40ffacf8.jpg)

![fd799ace91518b94c8f40d41cdf0df509bb3dd2d37174ef03f6ec341dd174e3b.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/images/fd799ace91518b94c8f40d41cdf0df509bb3dd2d37174ef03f6ec341dd174e3b.jpg)

### Tables

![3c12dca92f49c2598640f8f93d4ad387da596d1413be181e38be0006d526e5c5.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/3c12dca92f49c2598640f8f93d4ad387da596d1413be181e38be0006d526e5c5.jpg)

![52b2ddd066cd7250ffb3a6cae5bdadb04b10a664ec78c6c711e83ae3bd01af15.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/52b2ddd066cd7250ffb3a6cae5bdadb04b10a664ec78c6c711e83ae3bd01af15.jpg)

![a4bd6de71d7d6d207359c731efd1c8be9be225dd0a56f61655118cad7d12ccac.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/a4bd6de71d7d6d207359c731efd1c8be9be225dd0a56f61655118cad7d12ccac.jpg)

![b0ecfb38ba48a918d5855a22d6b820eb014c3372311407aa3b80e2f5b06cecc6.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/b0ecfb38ba48a918d5855a22d6b820eb014c3372311407aa3b80e2f5b06cecc6.jpg)

![c0d557f1a75209d2e977e1230753ab2c4ecff46196ed8b1bc2e0310b081fc134.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/c0d557f1a75209d2e977e1230753ab2c4ecff46196ed8b1bc2e0310b081fc134.jpg)

![c6c05764113858a659c382785b9a97514268923bf31b38552e8901cd697752e2.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/c6c05764113858a659c382785b9a97514268923bf31b38552e8901cd697752e2.jpg)

![e1c6919cbe47584a0f57ecfd8f25cd52a5b70c976e8b62bf144693d3b320a03c.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/e1c6919cbe47584a0f57ecfd8f25cd52a5b70c976e8b62bf144693d3b320a03c.jpg)

![e2fec2e0f5eceeb936f2c66a0662bb3ad8f1f3a082eaebb3c981ee3f21b9878d.jpg](../icml_results/670_Reducing%20Per-Sample%20Interference%20in%20Stochastic%20Optimization/tables/e2fec2e0f5eceeb936f2c66a0662bb3ad8f1f3a082eaebb3c981ee3f21b9878d.jpg)
