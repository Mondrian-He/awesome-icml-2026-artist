# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 184 of 197

## 目录 (Table of Contents)

1. [Towards Fully Parameter-Free Stochastic Optimization: Grid Search with Self-Bounding Analysis](#paper-6040)
2. [Hierarchical Abstract Tree for Cross-Document Retrieval Augmented Generation](#paper-6041)
3. [Smoothing Slot Attention Iterations and Recurrences](#paper-6042)
4. [ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning](#paper-6043)
5. [Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation](#paper-6044)
6. [SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](#paper-6045)
7. [Geometry-Aware Contrastive Learning for Few-Shot Automatic Modulation Recognition](#paper-6046)
8. [Instance-Dependent Continuous-Time Reinforcement Learning via Maximum Likelihood Estimation](#paper-6047)
9. [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](#paper-6048)
10. [Class-Conditional Distribution Balancing for Group Robust Classification](#paper-6049)
11. [Learning Tight Rejection Boundaries without Negatives for Strict One-Class Audio Deepfake Detection](#paper-6050)
12. [Non-Parametric Structural Priors for Geometry Theorem Prediction](#paper-6051)
13. [LATO: 3D Mesh Flow Matching with Structured TOpology Preserving LAtents](#paper-6052)
14. [Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models](#paper-6053)
15. [TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](#paper-6054)
16. [Advancing Analytic Class-Incremental Learning through Vision-Language Calibration](#paper-6055)
17. [Beyond Unidirectional Bias: Reciprocal Perspective Calibration in Scene Graph Generation](#paper-6056)
18. [$σ$: Sigmoid Modulation for Ultra High Resolution Diffusion](#paper-6057)
19. [Dual Quaternion SE(3) Synchronization with Recovery Guarantees](#paper-6058)
20. [ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns](#paper-6059)
21. [From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors](#paper-6060)
22. [(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](#paper-6061)
23. [From Generalist to Specialist Representation](#paper-6062)
24. [Scalable RF Simulation in Generative 4D Worlds](#paper-6063)
25. [Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing](#paper-6064)
26. [Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](#paper-6065)
27. [Proxy Compression for Language Modeling](#paper-6066)
28. [Image Restoration via Diffusion Models with Dynamic Resolution](#paper-6067)
29. [Doppler Prompting for Stable mmWave-based Human Pose Estimation](#paper-6068)
30. [Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis](#paper-6069)
31. [UniCode: Augmenting Evaluation for Code Reasoning](#paper-6070)
32. [WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](#paper-6071)
33. [LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding](#paper-6072)

---


<a id="paper-6040"></a>

## Towards Fully Parameter-Free Stochastic Optimization: Grid Search with Self-Bounding Analysis

### Images

![09a8d54d3b4caf1b3d94187cf96a5083bf931c0bac647b2b10f590f836990cb1.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/09a8d54d3b4caf1b3d94187cf96a5083bf931c0bac647b2b10f590f836990cb1.jpg)

![15ad892d4513aa2b617a80465f99c3a2f1b41200752b9521fa1a0550539a18e8.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/15ad892d4513aa2b617a80465f99c3a2f1b41200752b9521fa1a0550539a18e8.jpg)

![19cb4499d6658f57301aad44d27be29bfba96088366038a8799f59e29b81212c.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/19cb4499d6658f57301aad44d27be29bfba96088366038a8799f59e29b81212c.jpg)

![61b8385170d8052ce46132498c339c7d68ebaf6d3034b9c7ef26297c20fe0417.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/61b8385170d8052ce46132498c339c7d68ebaf6d3034b9c7ef26297c20fe0417.jpg)

![65c8e76353e3fc0b7411595c51b851effe3a0bd5cbda82cec10800c61a4b50bc.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/65c8e76353e3fc0b7411595c51b851effe3a0bd5cbda82cec10800c61a4b50bc.jpg)

![679b8cd571145d143422f4a101b825cb81ce1187fbaadc634709f6532d1c1dd9.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/679b8cd571145d143422f4a101b825cb81ce1187fbaadc634709f6532d1c1dd9.jpg)

![aa4f0cb0b3bc46a4f3e891ca0623d4e6be8ada24bd8d53f5b768d6d79c157e50.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/aa4f0cb0b3bc46a4f3e891ca0623d4e6be8ada24bd8d53f5b768d6d79c157e50.jpg)

![b337c335f0d004c2db82fc1ff242f0078a4d1c1a3b44b7daab851c62bc3cd0a8.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/b337c335f0d004c2db82fc1ff242f0078a4d1c1a3b44b7daab851c62bc3cd0a8.jpg)

![c643a1022bd6ab568edcf41b86eac5927729cc898f61ff9708c6e807138fabb2.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/c643a1022bd6ab568edcf41b86eac5927729cc898f61ff9708c6e807138fabb2.jpg)

![c9fc2fee683adc6fbda5c66836bd4a8c813c6c802e9f6a1a7e9229e226990b49.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/c9fc2fee683adc6fbda5c66836bd4a8c813c6c802e9f6a1a7e9229e226990b49.jpg)

### Tables

![153b399bc93a5dbc7deb97ab4e487f5fe924feeb72797ead2875778c1c0eca93.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/153b399bc93a5dbc7deb97ab4e487f5fe924feeb72797ead2875778c1c0eca93.jpg)

![164b9abb6e66106d5aa02dc71136f66e5a16cf2bbe27bbb96db88aa0706792fd.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/164b9abb6e66106d5aa02dc71136f66e5a16cf2bbe27bbb96db88aa0706792fd.jpg)

![1744bc4c131d85d483889e5f35a96250689847a51c85dd13d4c5272d6e47ef5a.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/1744bc4c131d85d483889e5f35a96250689847a51c85dd13d4c5272d6e47ef5a.jpg)

![2a3d6fac4bd7d9aab1b394e44306ec29d334baf6f20128c36c5f1357464f60d9.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/2a3d6fac4bd7d9aab1b394e44306ec29d334baf6f20128c36c5f1357464f60d9.jpg)

![980b1ae03eb43873bd0ae31a7cf40e77de10bace94155664d2450cedc1ed29f5.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/980b1ae03eb43873bd0ae31a7cf40e77de10bace94155664d2450cedc1ed29f5.jpg)

![b2c2b60bee9c8c5b5632c1900b3e04d8c9afa00c09db8d983decd5a2a3799a58.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/b2c2b60bee9c8c5b5632c1900b3e04d8c9afa00c09db8d983decd5a2a3799a58.jpg)

![c54a16c819e29302d067d837d7d4b16c2ed72463e1e46c8d5db094ce007a0317.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/c54a16c819e29302d067d837d7d4b16c2ed72463e1e46c8d5db094ce007a0317.jpg)

![d49d5b11df18720925288bd21eb91e07102f3f34926b0be941ede022f0d79ac5.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/d49d5b11df18720925288bd21eb91e07102f3f34926b0be941ede022f0d79ac5.jpg)

## Towards Fully Parameter-Free Stochastic Optimization: Grid Search with Self-Bounding Analysis


### Tables

![033e701e6d9d9b914272a712153d08cc1d30bebcfa9af7b41ce5fcf2a0369de1.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/033e701e6d9d9b914272a712153d08cc1d30bebcfa9af7b41ce5fcf2a0369de1.jpg)

![446212d0babb2c01f7a457a1eb689f4596429aacba086ee90476cd5191e28d40.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/446212d0babb2c01f7a457a1eb689f4596429aacba086ee90476cd5191e28d40.jpg)

![c9c049384b019297a8612010d23d89bc0d17c89a3cffee93fffb6cb5ef2a9fd2.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/c9c049384b019297a8612010d23d89bc0d17c89a3cffee93fffb6cb5ef2a9fd2.jpg)

![e05ce02282632e1cf2bdd57b2e848ebea47868faff4bfff1beca7611803f63a0.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/e05ce02282632e1cf2bdd57b2e848ebea47868faff4bfff1beca7611803f63a0.jpg)

![e7862bb924c7eddb8ae0b9d99eb9e9480849be064fd779ffe5d820ec49ea6951.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/e7862bb924c7eddb8ae0b9d99eb9e9480849be064fd779ffe5d820ec49ea6951.jpg)

![ee0b8c0d695050c26148ab435df63c19a6a370312b89e8062c9dd2622ec7fd92.jpg](../icml_results/6090_Towards%20Fully%20Parameter-Free%20Stochastic%20Optimization_%20Grid%20Search%20with%20Self-Bounding%20Analysis/tables/ee0b8c0d695050c26148ab435df63c19a6a370312b89e8062c9dd2622ec7fd92.jpg)

<a id="paper-6041"></a>

## Hierarchical Abstract Tree for Cross-Document Retrieval Augmented Generation


### Images

![23a4bd617ba38aeabc747b7ac139903974629ce086e784a92a293e13110a04de.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/23a4bd617ba38aeabc747b7ac139903974629ce086e784a92a293e13110a04de.jpg)

![443e22cb3305f06a844310248843d41ba1516fc928ef541b9bc4173c286c9e25.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/443e22cb3305f06a844310248843d41ba1516fc928ef541b9bc4173c286c9e25.jpg)

![47ac7716c411e38bd00f15bd87d6606da75a14e73b8452d1679616c208c088cc.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/47ac7716c411e38bd00f15bd87d6606da75a14e73b8452d1679616c208c088cc.jpg)

![4811f3f32b1c5fa087458cbfd2e20f350fb2585fea86e6762166a686d3ffa777.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/4811f3f32b1c5fa087458cbfd2e20f350fb2585fea86e6762166a686d3ffa777.jpg)

![737535defb38bedccb0a7ac47f8c417990862c03a434e74d120b4b1aa2170066.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/737535defb38bedccb0a7ac47f8c417990862c03a434e74d120b4b1aa2170066.jpg)

![79bc65119cc97527f801b582363c0927d027dc0cf249372afc5433f63e7af336.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/79bc65119cc97527f801b582363c0927d027dc0cf249372afc5433f63e7af336.jpg)

![915455c174c0c4e262ebecdf4f681ccfc7f17793239e282c5c86471f9add8cbc.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/915455c174c0c4e262ebecdf4f681ccfc7f17793239e282c5c86471f9add8cbc.jpg)

![987b106e2f4ffebc685352aa50f56ba93460d36341f888284b0d396b42ed1726.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/987b106e2f4ffebc685352aa50f56ba93460d36341f888284b0d396b42ed1726.jpg)

![98a3da41382bc71a598f2aa5fd570134b716feb495d0c2d24607680ada0aec15.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/98a3da41382bc71a598f2aa5fd570134b716feb495d0c2d24607680ada0aec15.jpg)

![9d7408882040002d486639cd9bca4e0f4833581a53e2838f20803185fcd4fa22.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/9d7408882040002d486639cd9bca4e0f4833581a53e2838f20803185fcd4fa22.jpg)

![c3d5969c3427feb3485ca0625e90af163bc1da6808ce7ba5914af6d75e17dcad.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/c3d5969c3427feb3485ca0625e90af163bc1da6808ce7ba5914af6d75e17dcad.jpg)

![d5df9e4ea9642cfb5e172aa39374ea7c6e56ab8f655dc75ef682362bb2ef34e5.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/d5df9e4ea9642cfb5e172aa39374ea7c6e56ab8f655dc75ef682362bb2ef34e5.jpg)

![f88b0cd70daff701eb6a885bd14bcb36ddd76e477e1fb76b5343e3fe0f70230a.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/images/f88b0cd70daff701eb6a885bd14bcb36ddd76e477e1fb76b5343e3fe0f70230a.jpg)

### Tables

![0d3be83cbfb8c86f7ab4b48b02e3a7bfb3cd828e36bff706312a154ff097837b.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/0d3be83cbfb8c86f7ab4b48b02e3a7bfb3cd828e36bff706312a154ff097837b.jpg)

![0e01e7a93809d620b9c9b9ac52c7768f4decd68822f5f001b7065671bc77c0b5.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/0e01e7a93809d620b9c9b9ac52c7768f4decd68822f5f001b7065671bc77c0b5.jpg)

![31d5a24d9b535cf65e0de880af8a1bd4e7a256f0261742da60b2ee1347260940.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/31d5a24d9b535cf65e0de880af8a1bd4e7a256f0261742da60b2ee1347260940.jpg)

![3d36b3e11cf72f390df7fe047bd7dc52b5cc00d7bfb709b85389b8862dc16f4c.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/3d36b3e11cf72f390df7fe047bd7dc52b5cc00d7bfb709b85389b8862dc16f4c.jpg)

![4452b04029fe72ad4c1814d4821ecefba7ce1d17b5722e63727333393004eb9d.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/4452b04029fe72ad4c1814d4821ecefba7ce1d17b5722e63727333393004eb9d.jpg)

![891493aee17ad9efedcfe8947ef634004aed45c33dcc9d44e41d46cca47188db.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/891493aee17ad9efedcfe8947ef634004aed45c33dcc9d44e41d46cca47188db.jpg)

![8f1fb2c82a21a5c2fa3c63a342eec85707372b900842391d4e3310f14cee0a33.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/8f1fb2c82a21a5c2fa3c63a342eec85707372b900842391d4e3310f14cee0a33.jpg)

![9368d3ec30af55d143f66f91a13b4645fe9a928e75b6110a3285381bdd012b05.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/9368d3ec30af55d143f66f91a13b4645fe9a928e75b6110a3285381bdd012b05.jpg)

![c6b9545ea768f2694f0e769351e3fdeb840b1d8857eb5075b036c642d4ce02f1.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/c6b9545ea768f2694f0e769351e3fdeb840b1d8857eb5075b036c642d4ce02f1.jpg)

![c8964526aae4e45906217ce42ec8f6985d66416151b8d785831e5dcf7d994c0c.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/c8964526aae4e45906217ce42ec8f6985d66416151b8d785831e5dcf7d994c0c.jpg)

![cdd11c7240f101839daeedb7fb33fd627c103f1b92dc7a1d5c266f056466f4d3.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/cdd11c7240f101839daeedb7fb33fd627c103f1b92dc7a1d5c266f056466f4d3.jpg)

![d211e10e25a78042950111c897c38ef90868ad9adfd7a31b3a564da8fd65ecfe.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/d211e10e25a78042950111c897c38ef90868ad9adfd7a31b3a564da8fd65ecfe.jpg)

![d78a1ddcbfb1f4814dae637609880f9e233fc88974013ef68f356eb06dd5845d.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/d78a1ddcbfb1f4814dae637609880f9e233fc88974013ef68f356eb06dd5845d.jpg)

![d92ce512a7b9173ba42c6b271ea7dc7bb8b5aff556db21ef145597d649d6c6be.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/d92ce512a7b9173ba42c6b271ea7dc7bb8b5aff556db21ef145597d649d6c6be.jpg)

![f5fa98be9aaefaf4e27a8b1d7409d186a57414f714b4d4007ab56068323f2e9f.jpg](../icml_results/6091_Hierarchical%20Abstract%20Tree%20for%20Cross-Document%20Retrieval%20Augmented%20Generation/tables/f5fa98be9aaefaf4e27a8b1d7409d186a57414f714b4d4007ab56068323f2e9f.jpg)

<a id="paper-6042"></a>

## Smoothing Slot Attention Iterations and Recurrences


### Images

![3095bf93e40da1e49b755e1380b9598b9a053f03ad923eb0d9bb316c8dcfcbea.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/images/3095bf93e40da1e49b755e1380b9598b9a053f03ad923eb0d9bb316c8dcfcbea.jpg)

![5d998f9f66cce10f392ee2c048c5a0dbb933522d696b070c28adbe29e904d8a2.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/images/5d998f9f66cce10f392ee2c048c5a0dbb933522d696b070c28adbe29e904d8a2.jpg)

![6c162f83ae3f36fd71518bcfff2845b9aaaefbc3d890047468735e940a7cd7f5.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/images/6c162f83ae3f36fd71518bcfff2845b9aaaefbc3d890047468735e940a7cd7f5.jpg)

![e6caffcf685a164b5e14d2825a9978075600620217546726ba9a400f7e001676.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/images/e6caffcf685a164b5e14d2825a9978075600620217546726ba9a400f7e001676.jpg)

### Tables

![099bbff7497e686d5e6524926d68ceac680ee4e8d60480f2bd2224da8cc81910.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/099bbff7497e686d5e6524926d68ceac680ee4e8d60480f2bd2224da8cc81910.jpg)

![1635290b20bededfb9dc63b7e4a344da674fcadffe937b4bce43c886576b1c99.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/1635290b20bededfb9dc63b7e4a344da674fcadffe937b4bce43c886576b1c99.jpg)

![2c2ff76a35c39afac253f0cca8151b5d03d76f927630c850fb789bc66efda0dd.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/2c2ff76a35c39afac253f0cca8151b5d03d76f927630c850fb789bc66efda0dd.jpg)

![8338d91a64631cdccabe1c5d5e77c0f8058a34f29902a36b4b96b7373c060c83.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/8338d91a64631cdccabe1c5d5e77c0f8058a34f29902a36b4b96b7373c060c83.jpg)

![9b7b328ae9cbb510e256e516bcd01404253d5071bf17447a57dcb4323117b113.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/9b7b328ae9cbb510e256e516bcd01404253d5071bf17447a57dcb4323117b113.jpg)

![a2dcb79550eabb0e5a1ba3ba2bfce7d400ba024aee97e749e32a0c1451281bf0.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/a2dcb79550eabb0e5a1ba3ba2bfce7d400ba024aee97e749e32a0c1451281bf0.jpg)

![d0ee8b30480637ff8575072bf6d895afb0730e698335a0e17125250fca96a8e4.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/d0ee8b30480637ff8575072bf6d895afb0730e698335a0e17125250fca96a8e4.jpg)

![f576d86c8029d1d122a808f82300c9ba58da2c4dd1cff13590a8383047bba6b1.jpg](../icml_results/6092_Smoothing%20Slot%20Attention%20Iterations%20and%20Recurrences/tables/f576d86c8029d1d122a808f82300c9ba58da2c4dd1cff13590a8383047bba6b1.jpg)

<a id="paper-6043"></a>

## ECHO: Entropy-Confidence Hybrid Optimization for Test-Time Reinforcement Learning


### Images

![05e3d34586205269a07224b7dff7c4e2c87f02c969eb90482751665a03c044ea.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/05e3d34586205269a07224b7dff7c4e2c87f02c969eb90482751665a03c044ea.jpg)

![0b9771146a00fd57a6b896a62be59ac813f0594c3e7eb8cb56b378602817f8a4.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/0b9771146a00fd57a6b896a62be59ac813f0594c3e7eb8cb56b378602817f8a4.jpg)

![5b0dab8762f3d0aa67012d832834b840224446c5718bbf4e07722fdef3910bfb.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/5b0dab8762f3d0aa67012d832834b840224446c5718bbf4e07722fdef3910bfb.jpg)

![d7be4fb284d79e5c1213d442c930ef5ffa81fcb3788d72b7c9e150b5eaef2d38.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/d7be4fb284d79e5c1213d442c930ef5ffa81fcb3788d72b7c9e150b5eaef2d38.jpg)

![db473857f1a748bf10b20a7ac3594c74010a91c512482da65a35ece8740d1309.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/db473857f1a748bf10b20a7ac3594c74010a91c512482da65a35ece8740d1309.jpg)

![dd911d17c6c7c39e5a6b861da57784e49b47f7caf2af5c8a1c3f5bad4010fc52.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/dd911d17c6c7c39e5a6b861da57784e49b47f7caf2af5c8a1c3f5bad4010fc52.jpg)

![e77c62b2c5c3c3c7e43a1d7a92dbf4d9521c03bf46fb6352fc38fb98d95d63c2.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/images/e77c62b2c5c3c3c7e43a1d7a92dbf4d9521c03bf46fb6352fc38fb98d95d63c2.jpg)

### Tables

![135e4a8c1d81be7ad05f1708cac07bfdf52726e603467927bb3711f131ed5cfa.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/135e4a8c1d81be7ad05f1708cac07bfdf52726e603467927bb3711f131ed5cfa.jpg)

![1c53972ccbf270d236d3d43e1e62e3ae1f2eb2d2f5a19ba58872fb3aa6fe7fd6.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/1c53972ccbf270d236d3d43e1e62e3ae1f2eb2d2f5a19ba58872fb3aa6fe7fd6.jpg)

![62363a97acb5526fa62625c34f8d8966d0a5c4feda36e0d557952bf23173d52f.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/62363a97acb5526fa62625c34f8d8966d0a5c4feda36e0d557952bf23173d52f.jpg)

![7ee7002265132010af275c4658a2598753ba68f2c84d84193ea2b8bf0207fae0.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/7ee7002265132010af275c4658a2598753ba68f2c84d84193ea2b8bf0207fae0.jpg)

![8cb526e11f84928c23baab3b9bc2ab9225f1b7a738275ed02fc848b009a7e1c2.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/8cb526e11f84928c23baab3b9bc2ab9225f1b7a738275ed02fc848b009a7e1c2.jpg)

![9a73ce40597cffa7e91c80fdcc0b136dfe2871aceab271bfa82ad4ac9181318b.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/9a73ce40597cffa7e91c80fdcc0b136dfe2871aceab271bfa82ad4ac9181318b.jpg)

![af70aef7d85d431a3e54b905e970bc9f15097ab2378c2b5fd639233aa6f8ea9a.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/af70aef7d85d431a3e54b905e970bc9f15097ab2378c2b5fd639233aa6f8ea9a.jpg)

![b5df8fde367f791508f5fe31cc7afd1a50a0305ad07bc86ad6a542a50768ba1e.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/b5df8fde367f791508f5fe31cc7afd1a50a0305ad07bc86ad6a542a50768ba1e.jpg)

![b68a016cffefb25b63d55739dc29523181bfbd19f7ec31da5457a8ee989a02f4.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/b68a016cffefb25b63d55739dc29523181bfbd19f7ec31da5457a8ee989a02f4.jpg)

![c4725c045b267dd309d4d86f76211837e0a87801cc30412e04ee4a4e42b958c1.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/c4725c045b267dd309d4d86f76211837e0a87801cc30412e04ee4a4e42b958c1.jpg)

![da3e6c15816c9a026fdc486ac8006dda40b0fd96eebb64faac9af8e311e1146e.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/da3e6c15816c9a026fdc486ac8006dda40b0fd96eebb64faac9af8e311e1146e.jpg)

![f0db507eace4f14becacef9609f47e550e5142c17710010446f3eeeef4ccea35.jpg](../icml_results/6093_ECHO_%20Entropy-Confidence%20Hybrid%20Optimization%20for%20Test-Time%20Reinforcement%20Learning/tables/f0db507eace4f14becacef9609f47e550e5142c17710010446f3eeeef4ccea35.jpg)

<a id="paper-6044"></a>

## Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation


### Images

![83474623b83873755ddb878b68689ae9e2b9666726597f06b795a41c27a6b7e4.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/images/83474623b83873755ddb878b68689ae9e2b9666726597f06b795a41c27a6b7e4.jpg)

![b9297c09a54dda9a74ec3277a807a9eaed74e3a995c4681f0dbe0fb68116c306.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/images/b9297c09a54dda9a74ec3277a807a9eaed74e3a995c4681f0dbe0fb68116c306.jpg)

![f36260c775de8e89e795734acfe281d8a5a3ba169253addfb4a5ec025b62e375.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/images/f36260c775de8e89e795734acfe281d8a5a3ba169253addfb4a5ec025b62e375.jpg)

### Tables

![0f74226f021505d04f0e2e963d0f77c249f39f1a2094bc1cf1f27e98e039ddc3.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/0f74226f021505d04f0e2e963d0f77c249f39f1a2094bc1cf1f27e98e039ddc3.jpg)

![1cfc645e410e336a1ed9ac859e3a13e734b9afa20d26f47a525d76595e960189.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/1cfc645e410e336a1ed9ac859e3a13e734b9afa20d26f47a525d76595e960189.jpg)

![22b3f9d6db276ba09aad88a210967238be37cd9a3cae7fedd4ed189d95a8dc5b.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/22b3f9d6db276ba09aad88a210967238be37cd9a3cae7fedd4ed189d95a8dc5b.jpg)

![2ebe86e2f95c9fcbae90a3102e767fa8a8aa556c0eaf542ff7833b7fe3057888.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/2ebe86e2f95c9fcbae90a3102e767fa8a8aa556c0eaf542ff7833b7fe3057888.jpg)

![417eeeff1071afe80575f0ba59a8da665e8e42fc0f762453b86c3f70f5b1bc52.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/417eeeff1071afe80575f0ba59a8da665e8e42fc0f762453b86c3f70f5b1bc52.jpg)

![6317eeb582aff848c5ba25efde4ffe7b75dfd43db675a71ea34e3baf0a22f936.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/6317eeb582aff848c5ba25efde4ffe7b75dfd43db675a71ea34e3baf0a22f936.jpg)

![65a5a24818a8ca3bd2fd6c0bd9682b0ffcb3c32075066f00ca940c2252ad7af4.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/65a5a24818a8ca3bd2fd6c0bd9682b0ffcb3c32075066f00ca940c2252ad7af4.jpg)

![681be4141bfda02a41f13281d5ce9a8c425b78ee34fb4b85e9e5f21f27cf2ba1.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/681be4141bfda02a41f13281d5ce9a8c425b78ee34fb4b85e9e5f21f27cf2ba1.jpg)

![c7bc8bd2bdd18312a92cc9b8063f53e9488189e205eaa9bb9db5762fa9efdf72.jpg](../icml_results/6094_Causal%20Direct%20Preference%20Optimization%20for%20Distributionally%20Robust%20Generative%20Recommendation/tables/c7bc8bd2bdd18312a92cc9b8063f53e9488189e205eaa9bb9db5762fa9efdf72.jpg)

<a id="paper-6045"></a>

## SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment


### Images

![079a029a5de4c24b5e13d0207ed6865569e3de4bb675901473f47c8b3b61be6b.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/079a029a5de4c24b5e13d0207ed6865569e3de4bb675901473f47c8b3b61be6b.jpg)

![07e5a2373b2e59157e259f0cb5395824473d81f39668177f7b7722123b5a9d92.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/07e5a2373b2e59157e259f0cb5395824473d81f39668177f7b7722123b5a9d92.jpg)

![163f66df6af8b5988a0c197dd21e5a8a39068fad590a80714bb3df5e23517b41.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/163f66df6af8b5988a0c197dd21e5a8a39068fad590a80714bb3df5e23517b41.jpg)

![17012ae9f9275b16fe24f332eb3f361df7a10703ec21cc45f743c8f6b7e55ad2.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/17012ae9f9275b16fe24f332eb3f361df7a10703ec21cc45f743c8f6b7e55ad2.jpg)

![227a11590f1bb91395b41e4175a7b162d2e001ea015005fa5023f48d0c2f3f76.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/227a11590f1bb91395b41e4175a7b162d2e001ea015005fa5023f48d0c2f3f76.jpg)

![3160819a58c448c7a251cd9d79f74cbfdd5c151e8aefeac6d67acd4c6a6d2be4.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/3160819a58c448c7a251cd9d79f74cbfdd5c151e8aefeac6d67acd4c6a6d2be4.jpg)

![37192719ddd6a9b6bc6565227199b7ab4107cbf747dbed0eea108999572dfc47.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/37192719ddd6a9b6bc6565227199b7ab4107cbf747dbed0eea108999572dfc47.jpg)

![3d88980f019468a8ca5d1695d12192a80ad251fddc1195d9a5f4574c36300867.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/3d88980f019468a8ca5d1695d12192a80ad251fddc1195d9a5f4574c36300867.jpg)

![43317c6037b230152d3fa3234acc616f85a21825e2833e390a068bdda55cf2c2.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/43317c6037b230152d3fa3234acc616f85a21825e2833e390a068bdda55cf2c2.jpg)

![446c254152b29b4b1ccd298eefd26b9cc3adb9c0d804b1c1349cc5fa698f1c08.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/446c254152b29b4b1ccd298eefd26b9cc3adb9c0d804b1c1349cc5fa698f1c08.jpg)

![47a35722462ffe10928f2ef57ac6f2f7c16591854a8a728ce307357f4f0a0232.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/47a35722462ffe10928f2ef57ac6f2f7c16591854a8a728ce307357f4f0a0232.jpg)

![5b95e6312083a68a96cb402f709d89cd4700ebafec8d39006991ef960790ba71.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/5b95e6312083a68a96cb402f709d89cd4700ebafec8d39006991ef960790ba71.jpg)

![5e0852e8b34b9569363162323f3eb70300596a0c0951ea3de7218d13668c4652.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/5e0852e8b34b9569363162323f3eb70300596a0c0951ea3de7218d13668c4652.jpg)

![6ecee767feb4b4304d6272e2fce75055e2512e32fd479435246539e099fdcfdb.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/6ecee767feb4b4304d6272e2fce75055e2512e32fd479435246539e099fdcfdb.jpg)

![76081fd411a4c827f9713d995a0f6fbafac8bdf335a99c5ffc6d557cca20d37c.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/76081fd411a4c827f9713d995a0f6fbafac8bdf335a99c5ffc6d557cca20d37c.jpg)

![7cf4216e977c67135630ccd086c4463620d7b91fb15723df7e8827f101a6eaad.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/7cf4216e977c67135630ccd086c4463620d7b91fb15723df7e8827f101a6eaad.jpg)

![7f5787fd834be1eb21381e31e33858d193a5a8d45358ca025dadadf39d5aff9a.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/7f5787fd834be1eb21381e31e33858d193a5a8d45358ca025dadadf39d5aff9a.jpg)

![9010e3a9a42b438b497c7176de8ade7ea30539a21b8af2e4ef1ce4f3d2b9bc7a.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/9010e3a9a42b438b497c7176de8ade7ea30539a21b8af2e4ef1ce4f3d2b9bc7a.jpg)

![977ca532d7a80004168c60792066ca39dc73a905b00aafed7cef172e85c8a49c.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/977ca532d7a80004168c60792066ca39dc73a905b00aafed7cef172e85c8a49c.jpg)

![98f194b8258a53c4e3136c12dca79726ed2ffeb02f637424c359bb9443226267.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/98f194b8258a53c4e3136c12dca79726ed2ffeb02f637424c359bb9443226267.jpg)

![9d0c86ee347efebfb4efb71815258e3c810e1d60a3530666a7e231fe4a6d4b84.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/9d0c86ee347efebfb4efb71815258e3c810e1d60a3530666a7e231fe4a6d4b84.jpg)

![9f4d0fd8cda388c6b5dd5108cb37aa4fa1ecea594a741ff1987e8a88cac427f0.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/9f4d0fd8cda388c6b5dd5108cb37aa4fa1ecea594a741ff1987e8a88cac427f0.jpg)

![a51d548709a98b7a3e09e26f3774a13e3d27e5fa303f946eb0994720e771880a.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/a51d548709a98b7a3e09e26f3774a13e3d27e5fa303f946eb0994720e771880a.jpg)

![a71e64709dcfbc857830c1b3b6fcf525110c6c6e30ce4d00ae2364ce8b9d223f.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/a71e64709dcfbc857830c1b3b6fcf525110c6c6e30ce4d00ae2364ce8b9d223f.jpg)

![abb09e7c1b8e965351a16904138f14a02448aa661159c08b2387a3349e5abe54.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/abb09e7c1b8e965351a16904138f14a02448aa661159c08b2387a3349e5abe54.jpg)

![ac8c085f40f8a3175d8719609473f98e6c4cb36b00375e844d0fcab4f9ef09ea.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/ac8c085f40f8a3175d8719609473f98e6c4cb36b00375e844d0fcab4f9ef09ea.jpg)

![b2377a5d23b21f03fedafb201b49e755d4daf1888bff9ba52c1790d75d0f274c.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/b2377a5d23b21f03fedafb201b49e755d4daf1888bff9ba52c1790d75d0f274c.jpg)

![c9e4f83f5d91ccb57912b899210550c6a610e8ea9a168822c252bd4690fad5eb.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/c9e4f83f5d91ccb57912b899210550c6a610e8ea9a168822c252bd4690fad5eb.jpg)

![e75a766d2ebae25032f077c2df1041664814f7f86a777e2c998760005f41f9a1.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/e75a766d2ebae25032f077c2df1041664814f7f86a777e2c998760005f41f9a1.jpg)

![f2b8eb51b5543c3a4278e7ff9fee892d6f674bd9c539dd433068374a7e60f8b4.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/f2b8eb51b5543c3a4278e7ff9fee892d6f674bd9c539dd433068374a7e60f8b4.jpg)

![f8a05aa9f51d72684af20abb4c2a322df16d1c00068d4ec413beeb4d72f90c55.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/f8a05aa9f51d72684af20abb4c2a322df16d1c00068d4ec413beeb4d72f90c55.jpg)

![f8eb07f838ad578331bc85ac18de92c0526a188837f798b0cc9f2eb5feb2fdd0.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/f8eb07f838ad578331bc85ac18de92c0526a188837f798b0cc9f2eb5feb2fdd0.jpg)

![fe3bd5144b577505d65125ca4cf2433401f9ff485ad3bd5e43293c8ef0342f62.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/images/fe3bd5144b577505d65125ca4cf2433401f9ff485ad3bd5e43293c8ef0342f62.jpg)

### Tables

![026f18617d62d60990af9ab404c7261d64c6278d58cbae9911dc67d59fd14e8b.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/026f18617d62d60990af9ab404c7261d64c6278d58cbae9911dc67d59fd14e8b.jpg)

![319ef38fcb8656a7884f8d65c0d57a0570ffd18badd794bc2691c385d0cb8550.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/319ef38fcb8656a7884f8d65c0d57a0570ffd18badd794bc2691c385d0cb8550.jpg)

![69c13494d26feb7ce6865967299b94f6d8369d6958da0bef335681fcfc095f0e.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/69c13494d26feb7ce6865967299b94f6d8369d6958da0bef335681fcfc095f0e.jpg)

![818affc7e835d3e798bc486fafc355413370c60717cc5f47c0a7e953d2d3d613.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/818affc7e835d3e798bc486fafc355413370c60717cc5f47c0a7e953d2d3d613.jpg)

![898aa98ad056ef52db05f72ce1e7976a489322b54541d424c5b1de4eae31a536.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/898aa98ad056ef52db05f72ce1e7976a489322b54541d424c5b1de4eae31a536.jpg)

![9f36f3881d71579abbfdd04ef4065149763b3f4be7189c60faacbe6cd58c1ce0.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/9f36f3881d71579abbfdd04ef4065149763b3f4be7189c60faacbe6cd58c1ce0.jpg)

![a0f7b65b1979db96833819f70f8275bdf93e69b5462f5f39d2554a50f4aa646d.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/a0f7b65b1979db96833819f70f8275bdf93e69b5462f5f39d2554a50f4aa646d.jpg)

![b3d7dc70f2eb54389839916e435387ba4987cede1f9b1bbdd9e579ecec16b5d2.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/b3d7dc70f2eb54389839916e435387ba4987cede1f9b1bbdd9e579ecec16b5d2.jpg)

![c486ab9e4284a798ba9180bcff701753a74fa1f434a584c0a5eb2f06e7fbe3ed.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/c486ab9e4284a798ba9180bcff701753a74fa1f434a584c0a5eb2f06e7fbe3ed.jpg)

![dad295df7d68edce5dfaa1da4d43b60cff0d363615f3a42a454aa4263f3f1a47.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/dad295df7d68edce5dfaa1da4d43b60cff0d363615f3a42a454aa4263f3f1a47.jpg)

![de95d8fe976b1abbd6ad5480368df634f18402287b5a5c9c23194a1ef75eeef3.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/de95d8fe976b1abbd6ad5480368df634f18402287b5a5c9c23194a1ef75eeef3.jpg)

![fa11c0aa3cdd7b3f15a48094ff4f88793482e3736cc94515464d836baa1b6f12.jpg](../icml_results/6095_SynerMedGen_%20Synergizing%20Medical%20Multimodal%20Understanding%20with%20Generation%20via%20Task%20Alignment/tables/fa11c0aa3cdd7b3f15a48094ff4f88793482e3736cc94515464d836baa1b6f12.jpg)

<a id="paper-6046"></a>

## Geometry-Aware Contrastive Learning for Few-Shot Automatic Modulation Recognition


### Images

![05395179d1e2e1e90ee8254ead50b6159373267e77808f1dabb723288578a7bb.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/05395179d1e2e1e90ee8254ead50b6159373267e77808f1dabb723288578a7bb.jpg)

![299e6817372120c4a3940e3be7fc40bd95fdcaede3a0762e9e4ee09fccbe9972.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/299e6817372120c4a3940e3be7fc40bd95fdcaede3a0762e9e4ee09fccbe9972.jpg)

![5f1943d33eed054496d14f6f68214d17182e1be4f4502b42fd3df397199cde6b.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/5f1943d33eed054496d14f6f68214d17182e1be4f4502b42fd3df397199cde6b.jpg)

![5f28957d1cba8248bb9a82b9a22902ff23507677d00f373f2a2cc596cd7eea73.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/5f28957d1cba8248bb9a82b9a22902ff23507677d00f373f2a2cc596cd7eea73.jpg)

![811a372c6fc1be3c1408938b323516f033a1cc89a2661dfc84274e7efb131b66.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/811a372c6fc1be3c1408938b323516f033a1cc89a2661dfc84274e7efb131b66.jpg)

![9fe230c8c6824f2e3798e77d8ac001e8d524e87dd9fd4dc2f9027b840fc4531e.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/9fe230c8c6824f2e3798e77d8ac001e8d524e87dd9fd4dc2f9027b840fc4531e.jpg)

![a187945eb8788220f1f7136d5071b6e3a3a3d47578b835d7d5cc79569e1f5250.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/a187945eb8788220f1f7136d5071b6e3a3a3d47578b835d7d5cc79569e1f5250.jpg)

![acb7d95945467b65dd83cc56ea61a24799c4180d5bc14e22aeb50eb73fbf7a0b.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/acb7d95945467b65dd83cc56ea61a24799c4180d5bc14e22aeb50eb73fbf7a0b.jpg)

![cfd66257a0bb0ca2f69834ce71b2cfb27a1897d9fc317d03b77413beac586d65.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/cfd66257a0bb0ca2f69834ce71b2cfb27a1897d9fc317d03b77413beac586d65.jpg)

![e1a1c1c1937392270162eab731129e914115fc32d108a58095976c2a3af6ea7c.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/images/e1a1c1c1937392270162eab731129e914115fc32d108a58095976c2a3af6ea7c.jpg)

### Tables

![1bbc593e594d25fbf1c7063bcb79a56bb20e4ab92f4ce34525db8e31ee9d0b89.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/1bbc593e594d25fbf1c7063bcb79a56bb20e4ab92f4ce34525db8e31ee9d0b89.jpg)

![258a512b31766a4f6d09830a2828a455eb2abafe2ed6fb9ff7d3289c7f22617d.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/258a512b31766a4f6d09830a2828a455eb2abafe2ed6fb9ff7d3289c7f22617d.jpg)

![2f2b6f029e2bfe24efebbfbdf2310cfbd20e2ee8b08e53c573fef805520d3687.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/2f2b6f029e2bfe24efebbfbdf2310cfbd20e2ee8b08e53c573fef805520d3687.jpg)

![7abc67f63b4ed91a3d08879835d27dc3c49d1f946809adefcb35094e5caf1384.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/7abc67f63b4ed91a3d08879835d27dc3c49d1f946809adefcb35094e5caf1384.jpg)

![b18b712604fa6cbd507f8fb39ba01cd295b262e3b8cf6f1ebc4109a1c6772159.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/b18b712604fa6cbd507f8fb39ba01cd295b262e3b8cf6f1ebc4109a1c6772159.jpg)

![e173a45796287e78570ebfb27ff2a837cf9eb71b81db9d5339305d8470a96abd.jpg](../icml_results/6096_Geometry-Aware%20Contrastive%20Learning%20for%20Few-Shot%20Automatic%20Modulation%20Recognition/tables/e173a45796287e78570ebfb27ff2a837cf9eb71b81db9d5339305d8470a96abd.jpg)

<a id="paper-6047"></a>

## Instance-Dependent Continuous-Time Reinforcement Learning via Maximum Likelihood Estimation


### Images

![3d6d9d9037267a9d945d41a7b14c7946ab79ecc35cf37770f37afc666176efa8.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/images/3d6d9d9037267a9d945d41a7b14c7946ab79ecc35cf37770f37afc666176efa8.jpg)

![7fb14d864b81a23bb926eb20998f9d2b06683007e3ec5415481ec6a14e69ad50.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/images/7fb14d864b81a23bb926eb20998f9d2b06683007e3ec5415481ec6a14e69ad50.jpg)

![ac08fc40286b31b97a6d7d389865f5a0ccbcab0fc5eccbea93d18e7fd1738e69.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/images/ac08fc40286b31b97a6d7d389865f5a0ccbcab0fc5eccbea93d18e7fd1738e69.jpg)

![b32f3737227335b901ff9cf413d5551c74abd518af718246c8b602e2d740ee72.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/images/b32f3737227335b901ff9cf413d5551c74abd518af718246c8b602e2d740ee72.jpg)

![ea25615902b38f707c737d06a710177cf3fa65d590c5ec7bff35bbd3b0c8dceb.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/images/ea25615902b38f707c737d06a710177cf3fa65d590c5ec7bff35bbd3b0c8dceb.jpg)

### Tables

![04cc114eebf68ee8e5ea88ea21323bc23a6f1b5f2a5a6aa604b6796bbc7d0278.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/tables/04cc114eebf68ee8e5ea88ea21323bc23a6f1b5f2a5a6aa604b6796bbc7d0278.jpg)

![89bc466826f9f8741788d5dfa93eb96755ba93a9231c629efe4dd666151d4bee.jpg](../icml_results/6097_Instance-Dependent%20Continuous-Time%20Reinforcement%20Learning%20via%20Maximum%20Likelihood%20Estimation/tables/89bc466826f9f8741788d5dfa93eb96755ba93a9231c629efe4dd666151d4bee.jpg)

<a id="paper-6048"></a>

## AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications


### Images

![0366caea50220b30dccd5a0fa18c1355a5399d9506570d693b9117785f424ffa.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/0366caea50220b30dccd5a0fa18c1355a5399d9506570d693b9117785f424ffa.jpg)

![1f2125689780bc12aa43dac0594737939d4693f67677f436defe621a15af9ab5.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/1f2125689780bc12aa43dac0594737939d4693f67677f436defe621a15af9ab5.jpg)

![2d67ac9c5557c6480a3df323991985c247eb0edb122bd7d6395ef1f73b689788.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/2d67ac9c5557c6480a3df323991985c247eb0edb122bd7d6395ef1f73b689788.jpg)

![3282940b588f07d0c79b350bd47f775d2aae60a015de9b3a6f9b54fd5e133bab.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/3282940b588f07d0c79b350bd47f775d2aae60a015de9b3a6f9b54fd5e133bab.jpg)

![4e34166da6f37b449ae96cec2f1db4f503e9ad84d52a5fc9c64f3ef248abec0c.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/4e34166da6f37b449ae96cec2f1db4f503e9ad84d52a5fc9c64f3ef248abec0c.jpg)

![9d1d4b5800674e6dca663317c070313d61359759e70897ad5dac23b657d8030e.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/9d1d4b5800674e6dca663317c070313d61359759e70897ad5dac23b657d8030e.jpg)

![ac4a905bb6d0868c52ac65cbff2e662540a9f3e649db534575cd57ddd31596de.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/ac4a905bb6d0868c52ac65cbff2e662540a9f3e649db534575cd57ddd31596de.jpg)

![cbe67294f1e17f2eb58ce6e5644d0ec396ded351a3c7928f4e27007f64aafa98.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/cbe67294f1e17f2eb58ce6e5644d0ec396ded351a3c7928f4e27007f64aafa98.jpg)

![cf889e091326cb95a569fca4ed7affb715c9c4d43cb421804b64f0d99de4af20.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/images/cf889e091326cb95a569fca4ed7affb715c9c4d43cb421804b64f0d99de4af20.jpg)

### Tables

![00d874b2582666395dd33b3c4603f58a56f0a7d146bacba4fc2702a0b5c34605.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/00d874b2582666395dd33b3c4603f58a56f0a7d146bacba4fc2702a0b5c34605.jpg)

![1354878ac9114e73ae87b756657f63fa527937d8deeacbbbded5a23467bc7289.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/1354878ac9114e73ae87b756657f63fa527937d8deeacbbbded5a23467bc7289.jpg)

![2839c9c914895996a37ab94e95125576780f31a513e46fd5beaae4b113094853.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/2839c9c914895996a37ab94e95125576780f31a513e46fd5beaae4b113094853.jpg)

![29e1324935e02e5228efcf20b3e8932def4cd5d2e412d6b9151385028f2940c1.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/29e1324935e02e5228efcf20b3e8932def4cd5d2e412d6b9151385028f2940c1.jpg)

![37ca839d04555c8d91692a4fbc364ea22aa1e918ca4a27545aac386d8199d74d.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/37ca839d04555c8d91692a4fbc364ea22aa1e918ca4a27545aac386d8199d74d.jpg)

![3db42548f13b1868d6427763b801578e731b848e706d5517a829744cea8186da.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/3db42548f13b1868d6427763b801578e731b848e706d5517a829744cea8186da.jpg)

![49367c50591ac09c17d9ba644bfb180dc468d762ee64e44dc2d6b16dfd170b35.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/49367c50591ac09c17d9ba644bfb180dc468d762ee64e44dc2d6b16dfd170b35.jpg)

![4a2f8ac26e60d2faafc1fe6a8ad96609b55a6f68085b693c728ac57966faad17.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/4a2f8ac26e60d2faafc1fe6a8ad96609b55a6f68085b693c728ac57966faad17.jpg)

![4b5a322b461d87fccfc5bf3dc0778a1cec8b054a33f1b6097cd49a7c37a05ff6.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/4b5a322b461d87fccfc5bf3dc0778a1cec8b054a33f1b6097cd49a7c37a05ff6.jpg)

![506ec9d4baa43eaaabc4d51ebba3d01ea36f252b1ab674ca91285b395ec7361b.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/506ec9d4baa43eaaabc4d51ebba3d01ea36f252b1ab674ca91285b395ec7361b.jpg)

![617c1872ab3df26aca1acdf6f763814dc70e758b16bb7b705ae4f95e2749ace1.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/617c1872ab3df26aca1acdf6f763814dc70e758b16bb7b705ae4f95e2749ace1.jpg)

![6754506d8073bd152602623824880a8203b14345ed010ad2e020528c44aa1f26.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/6754506d8073bd152602623824880a8203b14345ed010ad2e020528c44aa1f26.jpg)

![67df8684d27d82f579082dc5162e787a35651c7f155485b4ad9235b990d8a521.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/67df8684d27d82f579082dc5162e787a35651c7f155485b4ad9235b990d8a521.jpg)

![695fd716e58a00291b0831f853e0632ee20dba11a1eb6462fb26993372bb5198.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/695fd716e58a00291b0831f853e0632ee20dba11a1eb6462fb26993372bb5198.jpg)

![6e3d60b072800c82625c6f8f4e0e58890e96c323a58fe01ef0ed70cfcdbef712.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/6e3d60b072800c82625c6f8f4e0e58890e96c323a58fe01ef0ed70cfcdbef712.jpg)

![9262c792426f55a47befd594edb028324408b4d12bf2d8aea5fe5afb3b2950f4.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/9262c792426f55a47befd594edb028324408b4d12bf2d8aea5fe5afb3b2950f4.jpg)

![a0d418ba32053e06fcdfdaf902843817c05829bc2a5d7f46d1f59814967b778c.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/a0d418ba32053e06fcdfdaf902843817c05829bc2a5d7f46d1f59814967b778c.jpg)

![a8739403bbab7c41390918baed03f10690599e0d4e5a90d26b06b4a1b3d35a65.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/a8739403bbab7c41390918baed03f10690599e0d4e5a90d26b06b4a1b3d35a65.jpg)

![aaf8947f2327b72bf3a615283814930e33768b648f5644e2d059b4b58b7dbc30.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/aaf8947f2327b72bf3a615283814930e33768b648f5644e2d059b4b58b7dbc30.jpg)

![b0ff600fa479bfd26693d68521df17eb4c967ee81bfdadbfee32b0fa13eca482.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/b0ff600fa479bfd26693d68521df17eb4c967ee81bfdadbfee32b0fa13eca482.jpg)

![b2332b7575744b7937fbc49b34806971fe4b7015df69f9685dcaa2c433be19d9.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/b2332b7575744b7937fbc49b34806971fe4b7015df69f9685dcaa2c433be19d9.jpg)

![b7167eddebe8144089346e82ac97727cd1e3918bca6d849c5676aa62ea252376.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/b7167eddebe8144089346e82ac97727cd1e3918bca6d849c5676aa62ea252376.jpg)

![c94d027074f43b8b557ec68ac0a8ed07cc359fda52a35fd066a143cb4f289dea.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/c94d027074f43b8b557ec68ac0a8ed07cc359fda52a35fd066a143cb4f289dea.jpg)

![d440f4be90e1892dc698d079f8b8d57aa90be04c41b68af103b2a0350e2b6933.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/d440f4be90e1892dc698d079f8b8d57aa90be04c41b68af103b2a0350e2b6933.jpg)

![e2775db79bdabff1a7b60fd9327f0a655fe4be3aa21c1ae0d0ff524f803580e5.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/e2775db79bdabff1a7b60fd9327f0a655fe4be3aa21c1ae0d0ff524f803580e5.jpg)

![e3fcd97808676297a0da2e95f83dfda5163eb3003311b32767b820cfcf1d55f2.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/e3fcd97808676297a0da2e95f83dfda5163eb3003311b32767b820cfcf1d55f2.jpg)

![e63d69d6dc0047502f11bd57ff0df539927ac81730f46686399ec5c38e11da54.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/e63d69d6dc0047502f11bd57ff0df539927ac81730f46686399ec5c38e11da54.jpg)

![f7e5f08ffed8ed52278cc631c43a2bae297c1ab113a2d083a19ad11cd9611de8.jpg](../icml_results/6098_AMA-Bench_%20Evaluating%20Long-Horizon%20Memory%20for%20Agentic%20Applications/tables/f7e5f08ffed8ed52278cc631c43a2bae297c1ab113a2d083a19ad11cd9611de8.jpg)

<a id="paper-6049"></a>

## Class-Conditional Distribution Balancing for Group Robust Classification


### Images

![0f6d3151f6ae2a4cf7eaab305be0f2656641b63564f000d7c7366238911ea01d.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/0f6d3151f6ae2a4cf7eaab305be0f2656641b63564f000d7c7366238911ea01d.jpg)

![25b356cbbc0923f83cf78b8a22bf2ca727076b2a25b0cdfe197a466ec7e6b12f.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/25b356cbbc0923f83cf78b8a22bf2ca727076b2a25b0cdfe197a466ec7e6b12f.jpg)

![40042fcc059d49297b19c4e140c727787caaf5deca809f333be2c94a46052cd7.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/40042fcc059d49297b19c4e140c727787caaf5deca809f333be2c94a46052cd7.jpg)

![45d76901fc21e2c6725a02187e86b106d50666ebf24fc80f22e4f82aaa646fcf.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/45d76901fc21e2c6725a02187e86b106d50666ebf24fc80f22e4f82aaa646fcf.jpg)

![5bbf7cd1d4369215518817f275c03df9a5e9c921e8b50786090dc788eda36c83.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/5bbf7cd1d4369215518817f275c03df9a5e9c921e8b50786090dc788eda36c83.jpg)

![7fef0b7817a4b0e0ad8f4ba51e2c2bf4c4c0ed56390742259e77a11dce2e64a5.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/7fef0b7817a4b0e0ad8f4ba51e2c2bf4c4c0ed56390742259e77a11dce2e64a5.jpg)

![884f7b7906babf231f1ac44dd6ae2c43c173141047c6de07aa904bd1cce87187.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/884f7b7906babf231f1ac44dd6ae2c43c173141047c6de07aa904bd1cce87187.jpg)

![9d5fc937a29e63fbb0676294427a3bd064b40b14077bc824f4cc91116c5ace4f.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/9d5fc937a29e63fbb0676294427a3bd064b40b14077bc824f4cc91116c5ace4f.jpg)

![a05a71155fe6faa4a1331ae5613bca06aaaf0695db9e46cdfff4a9c743427d4c.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/a05a71155fe6faa4a1331ae5613bca06aaaf0695db9e46cdfff4a9c743427d4c.jpg)

![bcfe17d0cb62c11fd728745d29582a250c6a530d4fea1c6c644f969a58a5dadf.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/bcfe17d0cb62c11fd728745d29582a250c6a530d4fea1c6c644f969a58a5dadf.jpg)

![c6c6c16e313a53c6636e0fbb997205656d8f07988a49e2e28e2aaebe000c63e8.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/c6c6c16e313a53c6636e0fbb997205656d8f07988a49e2e28e2aaebe000c63e8.jpg)

![f58ea63ab2bb4dfe222ce79a0012cfdc0e1913c1d555d8f594676436f276840e.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/images/f58ea63ab2bb4dfe222ce79a0012cfdc0e1913c1d555d8f594676436f276840e.jpg)

### Tables

![08d3dd7dcc57bc914b1ff4edc29b9aa109167a5bb7d43c3b53ed855e4c50b92a.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/08d3dd7dcc57bc914b1ff4edc29b9aa109167a5bb7d43c3b53ed855e4c50b92a.jpg)

![19adeca13298499b1f432e8cfcdb13982969fa5adf95dfab25e8c42f85deaa5f.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/19adeca13298499b1f432e8cfcdb13982969fa5adf95dfab25e8c42f85deaa5f.jpg)

![230be7519737ccb90237bb63dd533d3d6886241e704f569610b4e652201e6b22.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/230be7519737ccb90237bb63dd533d3d6886241e704f569610b4e652201e6b22.jpg)

![2d8a49fd0258be14f02392035556f4ff43911b505e41f44bbff13097339f9aad.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/2d8a49fd0258be14f02392035556f4ff43911b505e41f44bbff13097339f9aad.jpg)

![3e8cc013ba05f89cec6fdf9c0d22c25f4dfd7319581fc52a338476499bd71504.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/3e8cc013ba05f89cec6fdf9c0d22c25f4dfd7319581fc52a338476499bd71504.jpg)

![49c022f0ba8ad3eb989ab1ee20e4444c9b749b245defb695db165439a21a9cb1.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/49c022f0ba8ad3eb989ab1ee20e4444c9b749b245defb695db165439a21a9cb1.jpg)

![6c1a4814fbc3a19ba988d734b50acedb1fb3d6a19cf02ef8d49e972367eff9dc.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/6c1a4814fbc3a19ba988d734b50acedb1fb3d6a19cf02ef8d49e972367eff9dc.jpg)

![8d5803e4d3438bb0cb9c9a22f929ae84254522731d5fb94662a3bb182a5e4c35.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/8d5803e4d3438bb0cb9c9a22f929ae84254522731d5fb94662a3bb182a5e4c35.jpg)

![b5909dc379523a49cccdbe6ea7e132d29f4bd6b1e3aa1016260ae5fbbf551979.jpg](../icml_results/6099_Class-Conditional%20Distribution%20Balancing%20for%20Group%20Robust%20Classification/tables/b5909dc379523a49cccdbe6ea7e132d29f4bd6b1e3aa1016260ae5fbbf551979.jpg)

<a id="paper-6050"></a>

## Learning Tight Rejection Boundaries without Negatives for Strict One-Class Audio Deepfake Detection


### Images

![87651776821117008986e91a60c1614c8f561c8e9d4158c87946b08e2e9405ac.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/images/87651776821117008986e91a60c1614c8f561c8e9d4158c87946b08e2e9405ac.jpg)

![9da468a930afbfc58eb675c07e0c3dedd9112fadcc2b37a842bfe62d2e43a980.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/images/9da468a930afbfc58eb675c07e0c3dedd9112fadcc2b37a842bfe62d2e43a980.jpg)

![d6afb7a03a38ae41c693be9659f2bc2fdcb15819b6dcb9f130c9b22e650dbedc.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/images/d6afb7a03a38ae41c693be9659f2bc2fdcb15819b6dcb9f130c9b22e650dbedc.jpg)

### Tables

![5e0349fd615472469cf1beb13127fdbf834cd7e331558da6cfd146e30215ba36.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/5e0349fd615472469cf1beb13127fdbf834cd7e331558da6cfd146e30215ba36.jpg)

![6b2c6d4e1ea1eb44debc0923ad20e29d03c6659e0c666a5b52b8bc60684b0ae6.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/6b2c6d4e1ea1eb44debc0923ad20e29d03c6659e0c666a5b52b8bc60684b0ae6.jpg)

![6e40faf6a7b3a512745b8da343ff692c1b92c81de3ed7bdef259fe0193fbdf29.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/6e40faf6a7b3a512745b8da343ff692c1b92c81de3ed7bdef259fe0193fbdf29.jpg)

![7dd2ecc27f660e1524c5d67b8d3522be68ed172dfad7427a25ffadad7d7cefe7.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/7dd2ecc27f660e1524c5d67b8d3522be68ed172dfad7427a25ffadad7d7cefe7.jpg)

![808dc2a6bee78fd54a37fd46cd1f1d181217a8985bdc57232e46476ba303435f.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/808dc2a6bee78fd54a37fd46cd1f1d181217a8985bdc57232e46476ba303435f.jpg)

![9b4f7962375d6b82e8e2d5c672a2382807706c48fcc96267e9ff66b3b6799495.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/9b4f7962375d6b82e8e2d5c672a2382807706c48fcc96267e9ff66b3b6799495.jpg)

![a11ab8b0d6296cae0a0e8f78582329a10fb7bfb5115c547711e79581fceb8388.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/a11ab8b0d6296cae0a0e8f78582329a10fb7bfb5115c547711e79581fceb8388.jpg)

![bc630d3f0383736f9f13b639774eb47e1746f611d841e64f4c021b14fd2a3bfd.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/bc630d3f0383736f9f13b639774eb47e1746f611d841e64f4c021b14fd2a3bfd.jpg)

![c32117b562d3c5c9730fc13572fbc0c2ab29629139e2cd34f28e21b65e103916.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/c32117b562d3c5c9730fc13572fbc0c2ab29629139e2cd34f28e21b65e103916.jpg)

![eaf9d9b38dacd6c427cf6ed55a253a4df4ee500da7492e6ee0390717a3f960ff.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/eaf9d9b38dacd6c427cf6ed55a253a4df4ee500da7492e6ee0390717a3f960ff.jpg)

![ebd7da81787ad95b84882f8c4f6c1abccd74c391d482feb02a0279e330ec0d23.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/ebd7da81787ad95b84882f8c4f6c1abccd74c391d482feb02a0279e330ec0d23.jpg)

![fa84cae9e1309a45713255255079025bb7cc94587fcfdeb7a90a65c573bb3157.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/fa84cae9e1309a45713255255079025bb7cc94587fcfdeb7a90a65c573bb3157.jpg)

![fa94508dbe74709a9741c0ff1f03f6b3e0b83670cc2c1a4e9e886b2d6f40daf9.jpg](../icml_results/6100_Learning%20Tight%20Rejection%20Boundaries%20without%20Negatives%20for%20Strict%20One-Class%20Audio%20Deepfake%20Detection/tables/fa94508dbe74709a9741c0ff1f03f6b3e0b83670cc2c1a4e9e886b2d6f40daf9.jpg)

<a id="paper-6051"></a>

## Non-Parametric Structural Priors for Geometry Theorem Prediction


### Images

![3df5c8f11eacb91277351ad43f79be80944dc4acd974d9050661f07cccae92f9.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/images/3df5c8f11eacb91277351ad43f79be80944dc4acd974d9050661f07cccae92f9.jpg)

![5c545a00896df17c34253b96c65d78fd00d28cc7247035b9d44ae9b8a7e1cc4e.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/images/5c545a00896df17c34253b96c65d78fd00d28cc7247035b9d44ae9b8a7e1cc4e.jpg)

![705397b87299373023a5cd6bf53707236a3dac78541c63c1e5237c500357cb47.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/images/705397b87299373023a5cd6bf53707236a3dac78541c63c1e5237c500357cb47.jpg)

![ec8b3c2c9ca00aef49ba43f96fd0a7177c8eb9e3686e25b31256f89efec583ad.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/images/ec8b3c2c9ca00aef49ba43f96fd0a7177c8eb9e3686e25b31256f89efec583ad.jpg)

### Tables

![155f6debfdeb2f21b359e339d6566d436c75ec2df9979cf0a2104d49c0bf8d0c.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/155f6debfdeb2f21b359e339d6566d436c75ec2df9979cf0a2104d49c0bf8d0c.jpg)

![1d7ef31fd905c1e5d9728bd0af6a0dea35347e0ce9a7c9224a90b14bed019119.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/1d7ef31fd905c1e5d9728bd0af6a0dea35347e0ce9a7c9224a90b14bed019119.jpg)

![295bc81ec316c96fed2d85052b52cdf248383db3e21861fe363a34c84dc504ee.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/295bc81ec316c96fed2d85052b52cdf248383db3e21861fe363a34c84dc504ee.jpg)

![2db80552a43e3d107455cc0ff6db26b2d28398b7670755a8320c3d6f33887199.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/2db80552a43e3d107455cc0ff6db26b2d28398b7670755a8320c3d6f33887199.jpg)

![37abd546dc9b7d140c2e87177f134c56489cb3716d39d1ef3253271e8f7d8575.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/37abd546dc9b7d140c2e87177f134c56489cb3716d39d1ef3253271e8f7d8575.jpg)

![739c9bf01190f2d534e9bed220f25fdbb3b7ecabc5d5e115016b80d3f171d8f6.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/739c9bf01190f2d534e9bed220f25fdbb3b7ecabc5d5e115016b80d3f171d8f6.jpg)

![82117632d166fa09a9d209f4371f941d2e73bcc2b141b8a7cc6bf668408da565.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/82117632d166fa09a9d209f4371f941d2e73bcc2b141b8a7cc6bf668408da565.jpg)

![988d084197ec321cf7f5fbe07eef23b3ba9827e6d978c52d85f4709b1e836ebb.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/988d084197ec321cf7f5fbe07eef23b3ba9827e6d978c52d85f4709b1e836ebb.jpg)

![b012fe54ada36400ade21708046a4da727c36b74f36da7ffc26920ab3daeb4b2.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/b012fe54ada36400ade21708046a4da727c36b74f36da7ffc26920ab3daeb4b2.jpg)

![ca5c662067021ed1c8cb3c293ab265bda7286faecb1e7478878c048e84b86084.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/ca5c662067021ed1c8cb3c293ab265bda7286faecb1e7478878c048e84b86084.jpg)

![db7ab0a05e55289c4a928e8f652aa089a2b7bbbf1b73c95a8daf4aa0573a54fc.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/db7ab0a05e55289c4a928e8f652aa089a2b7bbbf1b73c95a8daf4aa0573a54fc.jpg)

![e22fcab6f123b89cf4551fbda11a3ade7b6aee78f99dd2adf4fd45bce0f6a10f.jpg](../icml_results/6101_Non-Parametric%20Structural%20Priors%20for%20Geometry%20Theorem%20Prediction/tables/e22fcab6f123b89cf4551fbda11a3ade7b6aee78f99dd2adf4fd45bce0f6a10f.jpg)

<a id="paper-6052"></a>

## LATO: 3D Mesh Flow Matching with Structured TOpology Preserving LAtents


### Images

![2012c4223cd94c966defaa52936b150ae7bbec7b966aad849137c75f2be9ef48.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/2012c4223cd94c966defaa52936b150ae7bbec7b966aad849137c75f2be9ef48.jpg)

![3c355d7a4984b0fd0d5e580f90b0df34f4a8dd15587bbd01b941e4364e2879b5.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/3c355d7a4984b0fd0d5e580f90b0df34f4a8dd15587bbd01b941e4364e2879b5.jpg)

![4705fea3aeddba0570db8236c49f9435438441da463b5ed22552cd6f42509812.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/4705fea3aeddba0570db8236c49f9435438441da463b5ed22552cd6f42509812.jpg)

![4e60242eb33878df6d34c1f6d88c7866cce2a82f26b392274ffb62f7f2683154.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/4e60242eb33878df6d34c1f6d88c7866cce2a82f26b392274ffb62f7f2683154.jpg)

![58a1aed4795c75a37e526ddc496e0cd69e5e52981a5f8abf35bf472725d2b92e.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/58a1aed4795c75a37e526ddc496e0cd69e5e52981a5f8abf35bf472725d2b92e.jpg)

![59074c3fb96bbb235787f9e589cad3a747206b1337d91ef82eb45fa02cba004f.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/59074c3fb96bbb235787f9e589cad3a747206b1337d91ef82eb45fa02cba004f.jpg)

![59bcb0af79349289a42ed4ee191764d65d0ee2d05f1e2d5ccf64aab023536852.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/59bcb0af79349289a42ed4ee191764d65d0ee2d05f1e2d5ccf64aab023536852.jpg)

![5cabea6cc60881742e350b7ebf532ff40267d16f166a97d10d2a55a61e183965.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/5cabea6cc60881742e350b7ebf532ff40267d16f166a97d10d2a55a61e183965.jpg)

![74b8fe1f8390954d95759dd4420ec53a7c5a155113c9930fe74f5dbd6672413e.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/74b8fe1f8390954d95759dd4420ec53a7c5a155113c9930fe74f5dbd6672413e.jpg)

![75fe5dacd7b32b67bfa24e0b201be56dba947b6ad388967b252e76eda3092758.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/75fe5dacd7b32b67bfa24e0b201be56dba947b6ad388967b252e76eda3092758.jpg)

![a1547569473e01cfbaea6f475c571ff7b53bc761efa7762542cfb8291c340c90.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/a1547569473e01cfbaea6f475c571ff7b53bc761efa7762542cfb8291c340c90.jpg)

![afb8e7dd9e8428e0dea5dee0f03e7dd65670bceb62f0ee38a2a747accebf2a34.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/afb8e7dd9e8428e0dea5dee0f03e7dd65670bceb62f0ee38a2a747accebf2a34.jpg)

![b0bee7d5eee4516f56bc3e6f5e640f9a8f16e0ca8aad8e145e36ad732e53c624.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/b0bee7d5eee4516f56bc3e6f5e640f9a8f16e0ca8aad8e145e36ad732e53c624.jpg)

![b56aac180a1f8b6865bcbe6653a6e1b39356ec532a662f4514fa99fda7b922a2.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/b56aac180a1f8b6865bcbe6653a6e1b39356ec532a662f4514fa99fda7b922a2.jpg)

![b6342645307be5c029d42c1c7c9a891bee8282483a3cdd056e402e6c7af8e129.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/b6342645307be5c029d42c1c7c9a891bee8282483a3cdd056e402e6c7af8e129.jpg)

![db4a3654d50df83fd9a7f7b24cdffd02f1b8061895e96dd08811ae766bda0377.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/images/db4a3654d50df83fd9a7f7b24cdffd02f1b8061895e96dd08811ae766bda0377.jpg)

### Tables

![0c005aa9a55867d81d56ea2d84f8c6bca21c85f634876baf0904c7d03edb5b1f.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/tables/0c005aa9a55867d81d56ea2d84f8c6bca21c85f634876baf0904c7d03edb5b1f.jpg)

![5735060b716ea7019390b28d995fba6c660548c3eea163f1eabc6acf4276bc14.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/tables/5735060b716ea7019390b28d995fba6c660548c3eea163f1eabc6acf4276bc14.jpg)

![811d2d826a7385605d8cab6d42831a91d29853f30e49df04287be12c6345b26a.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/tables/811d2d826a7385605d8cab6d42831a91d29853f30e49df04287be12c6345b26a.jpg)

![c290b1b54bbc9132c7f4e314e1924d8de9b1f6bd6f190842af12ebd2b1285b37.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/tables/c290b1b54bbc9132c7f4e314e1924d8de9b1f6bd6f190842af12ebd2b1285b37.jpg)

![e1c34c07333aa2226f4147cc47f033e11e88ab422daec132ecd9ae104df98f50.jpg](../icml_results/6102_LATO_%203D%20Mesh%20Flow%20Matching%20with%20Structured%20TOpology%20Preserving%20LAtents/tables/e1c34c07333aa2226f4147cc47f033e11e88ab422daec132ecd9ae104df98f50.jpg)

<a id="paper-6053"></a>

## Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models


### Images

![16b6ad35c7f48f7639edc0e87b3597214f040f2234263f5e13335a893fa5a136.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/16b6ad35c7f48f7639edc0e87b3597214f040f2234263f5e13335a893fa5a136.jpg)

![21111e13df1649f7e5512f8a015fef6befd67dadb4aa280cc02ed17129107db4.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/21111e13df1649f7e5512f8a015fef6befd67dadb4aa280cc02ed17129107db4.jpg)

![52b5d2a4f6ef6fb4c46aa36573945dcc80cebf9038829e99a3d4b8fc07eeb87d.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/52b5d2a4f6ef6fb4c46aa36573945dcc80cebf9038829e99a3d4b8fc07eeb87d.jpg)

![937b66431a59a3789df48f1f6ad92e21769c33cf1a1fcda4bf8b3ad7f96b33fa.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/937b66431a59a3789df48f1f6ad92e21769c33cf1a1fcda4bf8b3ad7f96b33fa.jpg)

![b356f5278d2b835dd66b06a270a32ac477daf92aa3f69f333a4b5f89d0b1f44a.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/b356f5278d2b835dd66b06a270a32ac477daf92aa3f69f333a4b5f89d0b1f44a.jpg)

![c59acadc7a1e39c218172f658b7504cb89f2f7ede37dc7edbb76176ffb79b39e.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/c59acadc7a1e39c218172f658b7504cb89f2f7ede37dc7edbb76176ffb79b39e.jpg)

![d1e896de8af62c57e7fada48ed3aef778caeff1d6b248de912816b191a91f557.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/d1e896de8af62c57e7fada48ed3aef778caeff1d6b248de912816b191a91f557.jpg)

![ec93245626b4a42c0e7a8ac18608559e30975cf63b252da3b9e9283641361c52.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/images/ec93245626b4a42c0e7a8ac18608559e30975cf63b252da3b9e9283641361c52.jpg)

### Tables

![230639ec05ecc7c09e5d57beb3aa61dc6b7c51c25150dd3c2437d1fad5b3b756.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/230639ec05ecc7c09e5d57beb3aa61dc6b7c51c25150dd3c2437d1fad5b3b756.jpg)

![2b540d17fff597ad5100a5827cee172b1d254ce6ff28295bf7e95853c192cbe5.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/2b540d17fff597ad5100a5827cee172b1d254ce6ff28295bf7e95853c192cbe5.jpg)

![72cd4abaeeed28bd36cfd8f7b1eb65c6867f2ac87d09d9acac6b068e8db38c04.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/72cd4abaeeed28bd36cfd8f7b1eb65c6867f2ac87d09d9acac6b068e8db38c04.jpg)

![b77fe54d7cc11aa6d47e1f0bdda2aec38f0c314113f1b32a1cfedc6ead1e68d3.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/b77fe54d7cc11aa6d47e1f0bdda2aec38f0c314113f1b32a1cfedc6ead1e68d3.jpg)

![f8b6e42db65776ef3000e08fb5225e572c4306e6bb39031b0d04cf4f013ffba3.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/f8b6e42db65776ef3000e08fb5225e572c4306e6bb39031b0d04cf4f013ffba3.jpg)

![f8e24b5e9ac2cd4a297aaf8a1b9726db00deec7cbfb0b30a763a2b488aa64bf1.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/f8e24b5e9ac2cd4a297aaf8a1b9726db00deec7cbfb0b30a763a2b488aa64bf1.jpg)

![f98ed2a864c83c7796503d0c6917a0146cf2940a7b111c4f6176b742145bf9a9.jpg](../icml_results/6103_Mean%20Flow%20Distillation_%20Robust%20and%20Stable%20Distillation%20for%20Flow%20Matching%20Models/tables/f98ed2a864c83c7796503d0c6917a0146cf2940a7b111c4f6176b742145bf9a9.jpg)

<a id="paper-6054"></a>

## TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation


### Images

![011fcd9ebbe7f885b6fa396ea1cfc5db7e2bbf44205a0d329d396fd48a7e5406.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/011fcd9ebbe7f885b6fa396ea1cfc5db7e2bbf44205a0d329d396fd48a7e5406.jpg)

![0997f5980baf24fdfa2f32b515d0b1663b40e3834e5f53e5d97ddfe81176e4a6.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/0997f5980baf24fdfa2f32b515d0b1663b40e3834e5f53e5d97ddfe81176e4a6.jpg)

![247f6ecfe9b88352dd0dddc94948e67b58f5533af1edb61d7121773dff12df88.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/247f6ecfe9b88352dd0dddc94948e67b58f5533af1edb61d7121773dff12df88.jpg)

![3e368329450efe229f442026c762799e2fb94caa0bae62e63d95e7e642911247.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/3e368329450efe229f442026c762799e2fb94caa0bae62e63d95e7e642911247.jpg)

![557a9b2e7ed9a302853a928ed0128e5cb41db9d69639e076306950e64a9bbf2f.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/557a9b2e7ed9a302853a928ed0128e5cb41db9d69639e076306950e64a9bbf2f.jpg)

![91bcb1b4df6d4cde7dd4278a309657e3151347f0be27c0d48587eafe211c8d55.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/91bcb1b4df6d4cde7dd4278a309657e3151347f0be27c0d48587eafe211c8d55.jpg)

![bdb3a8985d887ea75b280dba67a51b2cab5b6e4a729d658fd0793375c201ffaa.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/bdb3a8985d887ea75b280dba67a51b2cab5b6e4a729d658fd0793375c201ffaa.jpg)

![cd706a5ce45bb69f7e78b5d5b5fc7835ad37790fbc0cc6905f9dedd907d7621b.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/cd706a5ce45bb69f7e78b5d5b5fc7835ad37790fbc0cc6905f9dedd907d7621b.jpg)

![f2d92d8a3ba8577a0baf4b0d284b1ca0a002220e0bcb603da4f82228ddf0743e.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/images/f2d92d8a3ba8577a0baf4b0d284b1ca0a002220e0bcb603da4f82228ddf0743e.jpg)

### Tables

![105825df20abb811a6c6aa80f3b6a043e9b8063aed711f060535d7d7bc0e36e4.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/105825df20abb811a6c6aa80f3b6a043e9b8063aed711f060535d7d7bc0e36e4.jpg)

![19c87a408aa049d8473a922bd04b6aa6893d0a02f21a8e5103678f1ab8aeaf4b.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/19c87a408aa049d8473a922bd04b6aa6893d0a02f21a8e5103678f1ab8aeaf4b.jpg)

![2998dd946f36d5dcb61c8b03e73ff2aab60667ff7cf808f63f0f94a28bbe8ddd.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/2998dd946f36d5dcb61c8b03e73ff2aab60667ff7cf808f63f0f94a28bbe8ddd.jpg)

![44835dca8d9f85dcde86a576f7ef4ab4cc25be879b43f0636c9cc7970f31e4a0.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/44835dca8d9f85dcde86a576f7ef4ab4cc25be879b43f0636c9cc7970f31e4a0.jpg)

![4725bd55726204db45aae988517c60264e30b4cb900b302ad7c96b3aba263c0c.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/4725bd55726204db45aae988517c60264e30b4cb900b302ad7c96b3aba263c0c.jpg)

![d90bfec19a1a81e92894030378887ef3f37c18e8fa58420cab110abe43764a63.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/d90bfec19a1a81e92894030378887ef3f37c18e8fa58420cab110abe43764a63.jpg)

![e4c15ffa97b9b164a999c795aaa7e86ebd3578947cc5df6a09b7c384fabb4791.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/e4c15ffa97b9b164a999c795aaa7e86ebd3578947cc5df6a09b7c384fabb4791.jpg)

![e6491c3e1453384e63cc8a7afed08bc6594b295aae667656051b2a6d796a921a.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/e6491c3e1453384e63cc8a7afed08bc6594b295aae667656051b2a6d796a921a.jpg)

![f2f09fef11cb3a624b27b13430cd938553912c93c38fc182f19fe69cbdce3620.jpg](../icml_results/6104_TapSampling_%20Inference-Time%20Sampling%20with%20a%20Task-Progress-Understanding%20Verifier%20for%20Robotic%20Manipul/tables/f2f09fef11cb3a624b27b13430cd938553912c93c38fc182f19fe69cbdce3620.jpg)

<a id="paper-6055"></a>

## Advancing Analytic Class-Incremental Learning through Vision-Language Calibration


### Images

![234133582c115b6afd942b1af8866563fb6bf17934e4240712c09a605eecf988.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/234133582c115b6afd942b1af8866563fb6bf17934e4240712c09a605eecf988.jpg)

![30fec3aed64c4de65154a6b935613b32034294c9540f989701c5f19005d16250.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/30fec3aed64c4de65154a6b935613b32034294c9540f989701c5f19005d16250.jpg)

![35a878d95b06e6b5f0081b6c0c227239785dc6bc01d41ddd05731430ce21f8d1.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/35a878d95b06e6b5f0081b6c0c227239785dc6bc01d41ddd05731430ce21f8d1.jpg)

![36616fdd4c63f126ff010a996bafb716c2770adcc2f6d499d447185c8b6ec475.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/36616fdd4c63f126ff010a996bafb716c2770adcc2f6d499d447185c8b6ec475.jpg)

![7086f2d23d0d7d044a27075592e071d1c3579727bdea86a3cb9b4d9785ab1468.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/7086f2d23d0d7d044a27075592e071d1c3579727bdea86a3cb9b4d9785ab1468.jpg)

![7b3faddd9967d9e15c1bd0cc97b49faf5426041231bcc1fe1dc228beb304a2b8.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/7b3faddd9967d9e15c1bd0cc97b49faf5426041231bcc1fe1dc228beb304a2b8.jpg)

![8845ab56c4ef25cb99b4e2e2bf64a452ea27b87c69dd6c9a692b9e1c7cc9c2bc.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/8845ab56c4ef25cb99b4e2e2bf64a452ea27b87c69dd6c9a692b9e1c7cc9c2bc.jpg)

![a2824e5f5a76f0af4e2954c64fbd4f35563f4e8c81819d563b7b25201d788301.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/a2824e5f5a76f0af4e2954c64fbd4f35563f4e8c81819d563b7b25201d788301.jpg)

![db5f6930515453adeda3e722a3d9145d8da0b2d6c2d16d66d03843a5de953294.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/db5f6930515453adeda3e722a3d9145d8da0b2d6c2d16d66d03843a5de953294.jpg)

![eb6d6f8ea3337c800adbb343500a84b190bce46fff9460f602b2bc6e3da41889.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/eb6d6f8ea3337c800adbb343500a84b190bce46fff9460f602b2bc6e3da41889.jpg)

![f90fab2a469d29440a1e34e635c013565936532f172cb0543afd96bc84163948.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/images/f90fab2a469d29440a1e34e635c013565936532f172cb0543afd96bc84163948.jpg)

### Tables

![2c596bb2b94d7c2087a91b8440b47f1edb872210fe82914dedcfc09ac0515c90.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/2c596bb2b94d7c2087a91b8440b47f1edb872210fe82914dedcfc09ac0515c90.jpg)

![436cdc7c048181724f6f358efa3bc5a93701b4efbe012f239b9ff2e6c5906fc8.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/436cdc7c048181724f6f358efa3bc5a93701b4efbe012f239b9ff2e6c5906fc8.jpg)

![495224523d56ee53cd9ae7814c0a929e02aa6bb7a181d995e37d3f69f4ebb344.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/495224523d56ee53cd9ae7814c0a929e02aa6bb7a181d995e37d3f69f4ebb344.jpg)

![547511ac318856cd44c256fc74015fe0a3696d96746fd844e5541e9693b3cb79.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/547511ac318856cd44c256fc74015fe0a3696d96746fd844e5541e9693b3cb79.jpg)

![69fa69fdc77d44c9f352c735a4b0cd7f022f94fefa12277222352757a996e7c7.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/69fa69fdc77d44c9f352c735a4b0cd7f022f94fefa12277222352757a996e7c7.jpg)

![6fbc60b49f03692d5fb3748368e37cfb6dee069ed17007e4619bca5404164d66.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/6fbc60b49f03692d5fb3748368e37cfb6dee069ed17007e4619bca5404164d66.jpg)

![9ccc96a9a915e06be854c19824241f3eea2298172424e0bd23b9e8e6bc287730.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/9ccc96a9a915e06be854c19824241f3eea2298172424e0bd23b9e8e6bc287730.jpg)

![a4bc2bc08b3d4f1caaba24db6cd33f1bfa0ae7f8cc380b310e299b43264e762d.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/a4bc2bc08b3d4f1caaba24db6cd33f1bfa0ae7f8cc380b310e299b43264e762d.jpg)

![e9af462b3cc26f49d483e8c84eb795cfb972a74baf9211e4152dd67c203c95c8.jpg](../icml_results/6105_Advancing%20Analytic%20Class-Incremental%20Learning%20through%20Vision-Language%20Calibration/tables/e9af462b3cc26f49d483e8c84eb795cfb972a74baf9211e4152dd67c203c95c8.jpg)

<a id="paper-6056"></a>

## Beyond Unidirectional Bias: Reciprocal Perspective Calibration in Scene Graph Generation


### Images

![29b37f4463f97ba9f63126bb1387695d3ced2adbc657de353951b18df57ff392.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/images/29b37f4463f97ba9f63126bb1387695d3ced2adbc657de353951b18df57ff392.jpg)

![7dd014085f9aabd4d65d6687ce29ba8eb421c2ecddee24beed31d2f1a1b599c2.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/images/7dd014085f9aabd4d65d6687ce29ba8eb421c2ecddee24beed31d2f1a1b599c2.jpg)

![b098d9f89f35cc827c5b193a420e811d68942fc158ad19febd54dee6e6da399f.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/images/b098d9f89f35cc827c5b193a420e811d68942fc158ad19febd54dee6e6da399f.jpg)

### Tables

![04de63906b422b23dee50a85084bf3369171e18fc9d6afcbe25611e68fb47949.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/04de63906b422b23dee50a85084bf3369171e18fc9d6afcbe25611e68fb47949.jpg)

![22a2c84f8a7115209f3c13a377c1445c841a18ffc3e960cac3dd121399d0dfc9.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/22a2c84f8a7115209f3c13a377c1445c841a18ffc3e960cac3dd121399d0dfc9.jpg)

![2ef83a547f03a28ac0d53f259880e08f43215d84d1445f1b35e3bdb624b53a52.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/2ef83a547f03a28ac0d53f259880e08f43215d84d1445f1b35e3bdb624b53a52.jpg)

![3ca4b69bab0cb9e660a010e0fe4b9afc8d835e2a7fe643b29b2ee903b667e1b0.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/3ca4b69bab0cb9e660a010e0fe4b9afc8d835e2a7fe643b29b2ee903b667e1b0.jpg)

![4e751bdea34774f7d56aaed3709ab2e74b5ef231ae5250ebf5cf6d2e9844938a.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/4e751bdea34774f7d56aaed3709ab2e74b5ef231ae5250ebf5cf6d2e9844938a.jpg)

![64640d4ae35faaed0a9bb2fae9c7a03bda0df1ea81f806751935e37104dbaab5.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/64640d4ae35faaed0a9bb2fae9c7a03bda0df1ea81f806751935e37104dbaab5.jpg)

![78d7b70ab20daf86a09f0c8c3bb7878900befc0839c6c701ec4aae32ab5adc4d.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/78d7b70ab20daf86a09f0c8c3bb7878900befc0839c6c701ec4aae32ab5adc4d.jpg)

![8e3ad4f11f55b44040caedac16e50ed9e361b2bb7dee743f7bb4894f01eb2560.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/8e3ad4f11f55b44040caedac16e50ed9e361b2bb7dee743f7bb4894f01eb2560.jpg)

![96ee93f538f373e63422ad334ff9a98dc7facf24fac5856a02f9b86082bb4fae.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/96ee93f538f373e63422ad334ff9a98dc7facf24fac5856a02f9b86082bb4fae.jpg)

![a1e619d89344a1f7e51974b51ff527b85173cc96b7beb6dd59de87b72c459e27.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/a1e619d89344a1f7e51974b51ff527b85173cc96b7beb6dd59de87b72c459e27.jpg)

![a7dbe2d184e582def5ed11b717c225cecab0e8feed08edcd576852c22fe49082.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/a7dbe2d184e582def5ed11b717c225cecab0e8feed08edcd576852c22fe49082.jpg)

![f9279a9e4bb69422b422f8b039cee5b0a91e52e8d15d39daffac05990f846b47.jpg](../icml_results/6106_Beyond%20Unidirectional%20Bias_%20Reciprocal%20Perspective%20Calibration%20in%20Scene%20Graph%20Generation/tables/f9279a9e4bb69422b422f8b039cee5b0a91e52e8d15d39daffac05990f846b47.jpg)

<a id="paper-6057"></a>

## $σ$: Sigmoid Modulation for Ultra High Resolution Diffusion


### Images

![4055a8953fe6d1a5f01988fbb985536c1025dbce973e8e342b06ba965989a1b2.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/4055a8953fe6d1a5f01988fbb985536c1025dbce973e8e342b06ba965989a1b2.jpg)

![4ae1d502775d49a89c260e935dbd286620c6ed0bc4460043e53d06fda20c967b.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/4ae1d502775d49a89c260e935dbd286620c6ed0bc4460043e53d06fda20c967b.jpg)

![52b4b5b3be1391b5d31baa6a9119426ece92f888fc1467f0867f703ea8ed9679.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/52b4b5b3be1391b5d31baa6a9119426ece92f888fc1467f0867f703ea8ed9679.jpg)

![5bcd646d6cf9cb6850909274b39d0eb0c86a14651613af20c71cd7627bc0cd4a.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/5bcd646d6cf9cb6850909274b39d0eb0c86a14651613af20c71cd7627bc0cd4a.jpg)

![91436bc49f70ba3ddc47852cdd421505d23ccf483dc8fe5b01a4b328e2402c23.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/91436bc49f70ba3ddc47852cdd421505d23ccf483dc8fe5b01a4b328e2402c23.jpg)

![b27ae8552ad822507e4d8e28cef23cc53c09e7e9bb570097db818fdd2db8ecf6.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/b27ae8552ad822507e4d8e28cef23cc53c09e7e9bb570097db818fdd2db8ecf6.jpg)

![cce984ff61f6f47cdf84a8dd2c7a6e3a61b62a0f5ce46e7023f1ee778250d49c.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/cce984ff61f6f47cdf84a8dd2c7a6e3a61b62a0f5ce46e7023f1ee778250d49c.jpg)

![f44ac2a8053bb76b858af80453da0ae17b6ed92ff17260f4bc8d0ece27f44729.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/images/f44ac2a8053bb76b858af80453da0ae17b6ed92ff17260f4bc8d0ece27f44729.jpg)

### Tables

![37d8eee9857d78aac5f4ca5dacebff6f4d20a5723a2425d8c4ce9c6cd4dc34cb.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/37d8eee9857d78aac5f4ca5dacebff6f4d20a5723a2425d8c4ce9c6cd4dc34cb.jpg)

![3cb468cd66a9f8ea0d3e55ca2ee7d3595453db14894ef95107ed7193df7462b1.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/3cb468cd66a9f8ea0d3e55ca2ee7d3595453db14894ef95107ed7193df7462b1.jpg)

![46f309d8d9c706293ab3c8cdedef6ad7bef05b92fb8cb10d36bd5f573c4a854d.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/46f309d8d9c706293ab3c8cdedef6ad7bef05b92fb8cb10d36bd5f573c4a854d.jpg)

![846159af47ae51c74a5ac44e21ec24c9667833e2996183ac75461331d803d73d.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/846159af47ae51c74a5ac44e21ec24c9667833e2996183ac75461331d803d73d.jpg)

![85d4890506aa4bfd9707fef9ee645208040a2aab89243ed4a88a7309a0b60d08.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/85d4890506aa4bfd9707fef9ee645208040a2aab89243ed4a88a7309a0b60d08.jpg)

![c09822a277921091562eb1c26f1b5b3610eb788b610f3c23f0b769015737f36e.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/c09822a277921091562eb1c26f1b5b3610eb788b610f3c23f0b769015737f36e.jpg)

![f8cca17fa46e0fab5dff937c50721e4a25701e69e5267afd121561d4a42d82d4.jpg](../icml_results/6107_%24%CF%83%24_%20Sigmoid%20Modulation%20for%20Ultra%20High%20Resolution%20Diffusion/tables/f8cca17fa46e0fab5dff937c50721e4a25701e69e5267afd121561d4a42d82d4.jpg)

<a id="paper-6058"></a>

## Dual Quaternion SE(3) Synchronization with Recovery Guarantees


### Images

![2f54d24a8374d75d5b217b4668867e54297bf38b519d34bad6eb5dc26afbb628.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/2f54d24a8374d75d5b217b4668867e54297bf38b519d34bad6eb5dc26afbb628.jpg)

![385f809d319e757e74c06992c312ea444d80859e3e4143ee376e19f9edb00801.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/385f809d319e757e74c06992c312ea444d80859e3e4143ee376e19f9edb00801.jpg)

![4b36260f2ad815c13eb766312a159aa302f3a496624eac1cbe45758ba220cf79.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/4b36260f2ad815c13eb766312a159aa302f3a496624eac1cbe45758ba220cf79.jpg)

![542ad29e9fb5e7e2e67278613d4c2560f386ff0f2dcc2ad57c3aec21cca5881c.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/542ad29e9fb5e7e2e67278613d4c2560f386ff0f2dcc2ad57c3aec21cca5881c.jpg)

![5a5914db79052bacf3cc4bf17d1bece357c71daab07c2a9273e161186d35136c.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/5a5914db79052bacf3cc4bf17d1bece357c71daab07c2a9273e161186d35136c.jpg)

![83bce974f4e4f59c7f7b927debd0debb527735d92a57dc30cc18bc59043b4411.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/83bce974f4e4f59c7f7b927debd0debb527735d92a57dc30cc18bc59043b4411.jpg)

![e761f98ef731a37c470ca51655b84f788edf630fa21ae9ee056fe95779366c8d.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/e761f98ef731a37c470ca51655b84f788edf630fa21ae9ee056fe95779366c8d.jpg)

![e7663fb8a52a405a1d445eaa222af8d0ba8091bf619ed56f74c100db25a474b9.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/images/e7663fb8a52a405a1d445eaa222af8d0ba8091bf619ed56f74c100db25a474b9.jpg)

### Tables

![114363f48073cc9f25229e4a83e4447fa1cd2b99ef12269e19399b34b926fe2a.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/114363f48073cc9f25229e4a83e4447fa1cd2b99ef12269e19399b34b926fe2a.jpg)

![6f306cd742e56d0aa48970d180977794d2f284526c3bfdad552c96d52f3f57a2.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/6f306cd742e56d0aa48970d180977794d2f284526c3bfdad552c96d52f3f57a2.jpg)

![89750e81259686c8967b9a069d545b64be2417ee80cdc111dbb154945c55a571.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/89750e81259686c8967b9a069d545b64be2417ee80cdc111dbb154945c55a571.jpg)

![9195a74949eaa8413a878acd2ea0bab838de46aa03704e93ebd122f6e724065a.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/9195a74949eaa8413a878acd2ea0bab838de46aa03704e93ebd122f6e724065a.jpg)

![b529df7773e92567167ec0ee4d8087bca32fc85607efb078a40a11c4232ad810.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/b529df7773e92567167ec0ee4d8087bca32fc85607efb078a40a11c4232ad810.jpg)

![e847bfa4589b14595160af10571f951f10339e887640cc4bdaea0ca1b14d58c4.jpg](../icml_results/6108_Dual%20Quaternion%20SE%283%29%20Synchronization%20with%20Recovery%20Guarantees/tables/e847bfa4589b14595160af10571f951f10339e887640cc4bdaea0ca1b14d58c4.jpg)

<a id="paper-6059"></a>

## ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns


### Images

![124d42f665df041a716939c3da3feb418ebd73b201b343dffce0b2c5c73c350e.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/124d42f665df041a716939c3da3feb418ebd73b201b343dffce0b2c5c73c350e.jpg)

![40a65b0cac3369dc116098b3426e206d60f897c80803a8c14984180b74744749.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/40a65b0cac3369dc116098b3426e206d60f897c80803a8c14984180b74744749.jpg)

![4ce7701ce72042e9cfea79477f79ea48a01ab3be250696cf6c56943f417dba8d.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/4ce7701ce72042e9cfea79477f79ea48a01ab3be250696cf6c56943f417dba8d.jpg)

![6997341947374b62a9aabce6b29d8da75f2aea335356c5f3d3c8b0020a3876ce.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/6997341947374b62a9aabce6b29d8da75f2aea335356c5f3d3c8b0020a3876ce.jpg)

![9abd968004628ebba24342afb27297f8297e3cdeaccfce1ffceffb3f3ae984e8.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/9abd968004628ebba24342afb27297f8297e3cdeaccfce1ffceffb3f3ae984e8.jpg)

![b09ccb1f0d3ba37d8f0fc79a71f9ab8a8a1c907415a4e2d4c34c2b5171ffea5a.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/b09ccb1f0d3ba37d8f0fc79a71f9ab8a8a1c907415a4e2d4c34c2b5171ffea5a.jpg)

![b5a7d7da2e63b2857e6e81525a5930a3019a768e81f59cd68762a57b5c456798.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/b5a7d7da2e63b2857e6e81525a5930a3019a768e81f59cd68762a57b5c456798.jpg)

![ed277a1835227469df760b40e54cacfdaf2bb229bb5dcbf05c5828bda6720aa0.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/images/ed277a1835227469df760b40e54cacfdaf2bb229bb5dcbf05c5828bda6720aa0.jpg)

### Tables

![1450daac8910cd5b93cfbee328f45b865b211900617be2283d8b80c66d3961b0.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/1450daac8910cd5b93cfbee328f45b865b211900617be2283d8b80c66d3961b0.jpg)

![1c8589deb9ae80d98a7ec092aab4d33055bdd56086fdbf47ed97b226d42ab004.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/1c8589deb9ae80d98a7ec092aab4d33055bdd56086fdbf47ed97b226d42ab004.jpg)

![32f9004f22cc465a9aa3e68d17f0f6797b31ee715d1c01a37d7a39fd2e40b5ea.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/32f9004f22cc465a9aa3e68d17f0f6797b31ee715d1c01a37d7a39fd2e40b5ea.jpg)

![36ce6a14f30a0fd3ac61c192206f463c0fd4bed695cb9356a17bc0359dfb4226.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/36ce6a14f30a0fd3ac61c192206f463c0fd4bed695cb9356a17bc0359dfb4226.jpg)

![4e5a2fcf5bb65f7e3568c7c387f85e6626bed68414fe6be040cbb9ebe2cac91c.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/4e5a2fcf5bb65f7e3568c7c387f85e6626bed68414fe6be040cbb9ebe2cac91c.jpg)

![69fbc6cf42705cf4572fc0d332a55ae27b211434c5490f11919c3bd5fb077c37.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/69fbc6cf42705cf4572fc0d332a55ae27b211434c5490f11919c3bd5fb077c37.jpg)

![7b507210ec14d67d84ec487e470cbbf093d665a94d8bed81cf3b8ea76c43378b.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/7b507210ec14d67d84ec487e470cbbf093d665a94d8bed81cf3b8ea76c43378b.jpg)

![9863c9d83f89a5bb010e285ffa14cffbb6d01fbc2f5559f9742058b5ea2940d9.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/9863c9d83f89a5bb010e285ffa14cffbb6d01fbc2f5559f9742058b5ea2940d9.jpg)

![e644ff6c74807667553c00c675d7667b9a6b06632bfd26566c6e637186f18642.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/e644ff6c74807667553c00c675d7667b9a6b06632bfd26566c6e637186f18642.jpg)

![e8077b174e6cea76478a46300c1cb0109cf22a3dc56af42c6a659616e7f9c44f.jpg](../icml_results/6109_ExpertWeaver_%20Unlocking%20the%20Inherent%20MoE%20in%20Dense%20LLMs%20with%20GLU%20Activation%20Patterns/tables/e8077b174e6cea76478a46300c1cb0109cf22a3dc56af42c6a659616e7f9c44f.jpg)

<a id="paper-6060"></a>

## From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors


### Images

![54042dac17b3d6de882104bd7433d5350e95e75c3fc891e8b768b14984a6f367.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/54042dac17b3d6de882104bd7433d5350e95e75c3fc891e8b768b14984a6f367.jpg)

![582e4c444b18bf2cbc0ed90879bbbb28c7d9119bc60f5f9cc921e82a14df51ef.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/582e4c444b18bf2cbc0ed90879bbbb28c7d9119bc60f5f9cc921e82a14df51ef.jpg)

![721daa05093847e06b8b16a7c5a03971d0c2a5f018aaf902bdec40bd82047191.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/721daa05093847e06b8b16a7c5a03971d0c2a5f018aaf902bdec40bd82047191.jpg)

![94eb759d63c37713c7370dc3aba0fd67caf6b5103bce0e323936c4e61cf8abab.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/94eb759d63c37713c7370dc3aba0fd67caf6b5103bce0e323936c4e61cf8abab.jpg)

![a661a869ab744c0d9420926a02d75c2e184060f0df658a3fa9c9793e42820572.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/a661a869ab744c0d9420926a02d75c2e184060f0df658a3fa9c9793e42820572.jpg)

![c0af3779499f6acd179efba69693caf7af790d744cc5c1b39789a888b06ca9d2.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/c0af3779499f6acd179efba69693caf7af790d744cc5c1b39789a888b06ca9d2.jpg)

![dd84affeb750d27b6ef353bb1fbe1b26925a81c6f61d08ba86364740d92710ce.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/dd84affeb750d27b6ef353bb1fbe1b26925a81c6f61d08ba86364740d92710ce.jpg)

![eeebefbe723543aca115f638571d912670f58062886e83b5fe80b207d6819a5d.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/eeebefbe723543aca115f638571d912670f58062886e83b5fe80b207d6819a5d.jpg)

![f6b1d0691c6cfcf813bca834bc0276f891ac0a3b37a873506b9553cded4f9420.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/f6b1d0691c6cfcf813bca834bc0276f891ac0a3b37a873506b9553cded4f9420.jpg)

![fe730477585c4e2ea97664a75f3ddc458a3a5fa5a317e2ed1c56b5f5c231a45e.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/images/fe730477585c4e2ea97664a75f3ddc458a3a5fa5a317e2ed1c56b5f5c231a45e.jpg)

### Tables

![00988eb9ca26c29bb22e1a8cf8e965a2316aee972864db0aa194fa2e70a695ae.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/00988eb9ca26c29bb22e1a8cf8e965a2316aee972864db0aa194fa2e70a695ae.jpg)

![0dc3ced4cf14b8833566886abf8d1f2114ed4dbe2fdbf9ce15b2e33a338aba11.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/0dc3ced4cf14b8833566886abf8d1f2114ed4dbe2fdbf9ce15b2e33a338aba11.jpg)

![1eea6bf8622bb2a275d549fc0ab35973b4b060336203a5f204f4592c43c5a8d3.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/1eea6bf8622bb2a275d549fc0ab35973b4b060336203a5f204f4592c43c5a8d3.jpg)

![329d9913c31cba8cdc8cd089f9c072e58829a02c1861350295c7f127c14d8731.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/329d9913c31cba8cdc8cd089f9c072e58829a02c1861350295c7f127c14d8731.jpg)

![55a35b0aa3c3d5a607e7e37d528a664e744cfd353f3b1b2012d29ecd907a9f94.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/55a35b0aa3c3d5a607e7e37d528a664e744cfd353f3b1b2012d29ecd907a9f94.jpg)

![74dc158b276abb0763ebf7c523992fa7768e7e6ea784a6789a4a062905dad4fe.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/74dc158b276abb0763ebf7c523992fa7768e7e6ea784a6789a4a062905dad4fe.jpg)

![77808857c4d8126e09d000b0e3a929a5c0d18e9f0c13e6b4ad914af109372c80.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/77808857c4d8126e09d000b0e3a929a5c0d18e9f0c13e6b4ad914af109372c80.jpg)

![86c93daca72c6ea711a3104edd4b0d2162db2e6e5cd9cbcc23e6492b09c9bc15.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/86c93daca72c6ea711a3104edd4b0d2162db2e6e5cd9cbcc23e6492b09c9bc15.jpg)

![9903b9040e9eda6cbdbc36e9499669d94c48685610ce4b7b82381faf25673ce0.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/9903b9040e9eda6cbdbc36e9499669d94c48685610ce4b7b82381faf25673ce0.jpg)

![dc28e9c7f8ba1ab71c0b6dfbd99c3a7c4a21cde69e288c4892c3edcc0b825893.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/dc28e9c7f8ba1ab71c0b6dfbd99c3a7c4a21cde69e288c4892c3edcc0b825893.jpg)

![e26abdc67b622ed8162fab9eeb8c002e03daa1737437b342ccee4b31a68cf4b1.jpg](../icml_results/6110_From%20Statics%20to%20Dynamics_%20Physics-Aware%20Image%20Editing%20with%20Latent%20Transition%20Priors/tables/e26abdc67b622ed8162fab9eeb8c002e03daa1737437b342ccee4b31a68cf4b1.jpg)

<a id="paper-6061"></a>

## (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models


### Images

![16fccd97838428d77b957cfde14e4434f996f0aa35c70cf693ab1801320a39ea.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/16fccd97838428d77b957cfde14e4434f996f0aa35c70cf693ab1801320a39ea.jpg)

![2e2687b5f8c9a21938d3258c25122d0a75a9bab6afb5528d58c0e5254a4abba8.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/2e2687b5f8c9a21938d3258c25122d0a75a9bab6afb5528d58c0e5254a4abba8.jpg)

![2f5897ab695abac4e402a44e1379a1a7138c610057153079177e7ccc966a276c.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/2f5897ab695abac4e402a44e1379a1a7138c610057153079177e7ccc966a276c.jpg)

![2f61ff0771127ef530640351601d24cd1d971ad5c91645899863a056f6a23173.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/2f61ff0771127ef530640351601d24cd1d971ad5c91645899863a056f6a23173.jpg)

![4ad46b8ad160f545943af7b0838ba0a54548e782546bfa71b896f8f65375c6e3.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/4ad46b8ad160f545943af7b0838ba0a54548e782546bfa71b896f8f65375c6e3.jpg)

![4f1922da225db2f43984230cde569298e43a6c1b99dfe6478c4ee35bd3ce950f.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/4f1922da225db2f43984230cde569298e43a6c1b99dfe6478c4ee35bd3ce950f.jpg)

![6124ad68e2c410dbb45315cc3c42df92a0084ef3b89263ee1bead7f0d53837cc.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/6124ad68e2c410dbb45315cc3c42df92a0084ef3b89263ee1bead7f0d53837cc.jpg)

![63eac17a957c47bee98b3427751c1816bc7a94882397afcde1c639e1fa0b9fdd.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/63eac17a957c47bee98b3427751c1816bc7a94882397afcde1c639e1fa0b9fdd.jpg)

![66ebad9b26119998c104654257a52c4d06ece1426e98c1c96cdcab417812b546.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/66ebad9b26119998c104654257a52c4d06ece1426e98c1c96cdcab417812b546.jpg)

![9269d2c9d79130d57c302df33e953cb7e6e1fdb03c68ea74366f5dee899557a3.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/9269d2c9d79130d57c302df33e953cb7e6e1fdb03c68ea74366f5dee899557a3.jpg)

![9f4b77651a603ced021bf765ada947e8f554fe820f3c171b36ef7fcc530e84db.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/9f4b77651a603ced021bf765ada947e8f554fe820f3c171b36ef7fcc530e84db.jpg)

![c79fe97394bd131320450e287b2f9e865f7d61a57a5f90c1d8eecc51ca73c7a6.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/c79fe97394bd131320450e287b2f9e865f7d61a57a5f90c1d8eecc51ca73c7a6.jpg)

![f3bdff51aa72cd6791eaf14683c530b5a4bdee572de1333dd6250a79a3482e81.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/f3bdff51aa72cd6791eaf14683c530b5a4bdee572de1333dd6250a79a3482e81.jpg)

![fc3b4776abd7cbeb634e81f8547ccaa2740da417ed9aebc79f8fda81704d4a84.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/images/fc3b4776abd7cbeb634e81f8547ccaa2740da417ed9aebc79f8fda81704d4a84.jpg)

### Tables

![05c25d952d161691543cd67629e414111b99446b3897a7941d1d4be2bd3aa82c.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/05c25d952d161691543cd67629e414111b99446b3897a7941d1d4be2bd3aa82c.jpg)

![0c1145bae49bca7b611413cc867230cf4ca0842483b3ab338944947a681e6340.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/0c1145bae49bca7b611413cc867230cf4ca0842483b3ab338944947a681e6340.jpg)

![2093341649c8c6518e63810ae49db1f60f128183b0006aef886411379012eea1.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/2093341649c8c6518e63810ae49db1f60f128183b0006aef886411379012eea1.jpg)

![2f144e5fbf4ada1cd62e52652b88b14cc3737afe4045986d6ac82c308e22f679.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/2f144e5fbf4ada1cd62e52652b88b14cc3737afe4045986d6ac82c308e22f679.jpg)

![36bea345b756cf687a5532e5f70da08d226e7cd0b1945568d8331f35e63c477b.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/36bea345b756cf687a5532e5f70da08d226e7cd0b1945568d8331f35e63c477b.jpg)

![3a2eb4dc6e10d26f047ceb35894335bec466f4b657e4f7492ae83ed36698d9a1.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/3a2eb4dc6e10d26f047ceb35894335bec466f4b657e4f7492ae83ed36698d9a1.jpg)

![481a6a3ea2f596b35ffca3c7378b9ca44ffdc24d87b2a62f05a02692ddf203a3.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/481a6a3ea2f596b35ffca3c7378b9ca44ffdc24d87b2a62f05a02692ddf203a3.jpg)

![4b1fc90bb6cbd8955480a4c9f4a433978c3d6af514b6d63e761c7432af7508c1.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/4b1fc90bb6cbd8955480a4c9f4a433978c3d6af514b6d63e761c7432af7508c1.jpg)

![570903202392880eb29e87e826ca1591aa037a2b3bf08fdc1a070c5bdb69495e.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/570903202392880eb29e87e826ca1591aa037a2b3bf08fdc1a070c5bdb69495e.jpg)

![64263e93911fc2f9709e98577bd1e7e5baccefd915bf06531f5dda5c4b8e3fc3.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/64263e93911fc2f9709e98577bd1e7e5baccefd915bf06531f5dda5c4b8e3fc3.jpg)

![6e51ed4e943a120d047242f83278eea501f37fcaee905b35b3a131431e8f9b98.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/6e51ed4e943a120d047242f83278eea501f37fcaee905b35b3a131431e8f9b98.jpg)

![75c2179f0dc02a3f57427e94460c1a6b2e794aedb755f33ee085d25d41fd19b5.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/75c2179f0dc02a3f57427e94460c1a6b2e794aedb755f33ee085d25d41fd19b5.jpg)

![7e48586bbcb597b9ca710185f765cdcd9abb72c94369456edc1a66f4074d2675.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/7e48586bbcb597b9ca710185f765cdcd9abb72c94369456edc1a66f4074d2675.jpg)

![8fa14971867e60ef799813dfd5ae73b1f19a15594e9037b2f95940cc63a02895.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/8fa14971867e60ef799813dfd5ae73b1f19a15594e9037b2f95940cc63a02895.jpg)

![935b9c607c7f91924c2310aa0bd57b0168f6aab2bd87d40a4e98d4bbdee4e3a6.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/935b9c607c7f91924c2310aa0bd57b0168f6aab2bd87d40a4e98d4bbdee4e3a6.jpg)

![b172d83672400cec6c99b78f84284ba934b09cc20078c5f61c1a129e5e60f169.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/b172d83672400cec6c99b78f84284ba934b09cc20078c5f61c1a129e5e60f169.jpg)

![de6d292ba709165e96a016809122ac515eb8e14ea342173ef91c9a1f9c1b656f.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/de6d292ba709165e96a016809122ac515eb8e14ea342173ef91c9a1f9c1b656f.jpg)

![de89966c07e60144e9a32be1529e7609812ccbf34befbd1115c6bdc82d8d141f.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/de89966c07e60144e9a32be1529e7609812ccbf34befbd1115c6bdc82d8d141f.jpg)

![e844033d993a92480020bd01a3cfa62aeb402ebe207bf2a772fa13ab04abe448.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/e844033d993a92480020bd01a3cfa62aeb402ebe207bf2a772fa13ab04abe448.jpg)

![f5afdd21972c2edabbcf3a977e687e17b3106ca57933840e83d625a3202c8b91.jpg](../icml_results/6111_%28Sparse%29%20Attention%20to%20the%20Details_%20Preserving%20Spectral%20Fidelity%20in%20ML-based%20Weather%20Forecasting%20Mode/tables/f5afdd21972c2edabbcf3a977e687e17b3106ca57933840e83d625a3202c8b91.jpg)

<a id="paper-6062"></a>

## From Generalist to Specialist Representation


### Images

![0a7244ec4276df98d39017c79c82c675cde603985857c148203d55e3553f6f8e.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/0a7244ec4276df98d39017c79c82c675cde603985857c148203d55e3553f6f8e.jpg)

![343188c13de4a199a216bc9877ca208e8bf8bfe131fc9996043bbd8bd7c4d64e.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/343188c13de4a199a216bc9877ca208e8bf8bfe131fc9996043bbd8bd7c4d64e.jpg)

![460b17b7acf612d1d592ce6ad63836cbf536a05a3898a1d43975ff01c0e26376.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/460b17b7acf612d1d592ce6ad63836cbf536a05a3898a1d43975ff01c0e26376.jpg)

![6e137912deadd8ef3819fc6c92d2e68a3082414df3c227b2d6723a8c4eb8af30.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/6e137912deadd8ef3819fc6c92d2e68a3082414df3c227b2d6723a8c4eb8af30.jpg)

![9c74a439c24198a4090ead8d7610d45a85603cf799e58a4f8ec6f71584f2ba40.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/9c74a439c24198a4090ead8d7610d45a85603cf799e58a4f8ec6f71584f2ba40.jpg)

![db3a6eba3fbd58e306b0fa6f57dec1366162faf763f2fcfa547b55f50584d5e0.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/db3a6eba3fbd58e306b0fa6f57dec1366162faf763f2fcfa547b55f50584d5e0.jpg)

![effbde9b117434159de134d74293e492acd82fb59ac6940ecb08cf2704039e17.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/effbde9b117434159de134d74293e492acd82fb59ac6940ecb08cf2704039e17.jpg)

![fbf680c28794fc35f02f72280b924f17fd3dec03012fa8205a2c42e0fe110ce5.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/images/fbf680c28794fc35f02f72280b924f17fd3dec03012fa8205a2c42e0fe110ce5.jpg)

### Tables

![9299958c70d2c97e777654bf8339b2c2a62dbd32d2a8ae131e5e9a5c7e0bc8a4.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/tables/9299958c70d2c97e777654bf8339b2c2a62dbd32d2a8ae131e5e9a5c7e0bc8a4.jpg)

![ad8af603a85718d9ee96cc20c108becc75d1a8e0a98b8ad084ed01d3c95a9b84.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/tables/ad8af603a85718d9ee96cc20c108becc75d1a8e0a98b8ad084ed01d3c95a9b84.jpg)

![dadad6b01e311be5926afb2ff5e10af9128b7ff69128ac9c9fd4b608d47bfef4.jpg](../icml_results/6112_From%20Generalist%20to%20Specialist%20Representation/tables/dadad6b01e311be5926afb2ff5e10af9128b7ff69128ac9c9fd4b608d47bfef4.jpg)

<a id="paper-6063"></a>

## Scalable RF Simulation in Generative 4D Worlds


### Images

![062b4ec54c993e9d2b382f66eab7c4b1b9827a877b04c4969d9bb3172b47a91b.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/062b4ec54c993e9d2b382f66eab7c4b1b9827a877b04c4969d9bb3172b47a91b.jpg)

![095cc57783a980a0860df66674409c621d61fed69d9ecb3d0a4516c61ac2848f.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/095cc57783a980a0860df66674409c621d61fed69d9ecb3d0a4516c61ac2848f.jpg)

![10a3dd9b52e4c386b7f428cd7cbfb07a1a10985d6a6f2d0b94aad63ce846f277.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/10a3dd9b52e4c386b7f428cd7cbfb07a1a10985d6a6f2d0b94aad63ce846f277.jpg)

![1421d853a9baef5f9c98008f83b46b921572139e1fefb142fef0ce7ef12691d2.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/1421d853a9baef5f9c98008f83b46b921572139e1fefb142fef0ce7ef12691d2.jpg)

![187dacc7b7379d728dfcd52435dbc35262e5958b6b3376ec5859235521b1149d.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/187dacc7b7379d728dfcd52435dbc35262e5958b6b3376ec5859235521b1149d.jpg)

![3f0fcf31512ddf521688a73ecc727aa5ba8a0c492558842811bff8f42197fe7e.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/3f0fcf31512ddf521688a73ecc727aa5ba8a0c492558842811bff8f42197fe7e.jpg)

![451497584291b3fb15089fd527e86606a0451b33c80aaa32939af79b1260673a.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/451497584291b3fb15089fd527e86606a0451b33c80aaa32939af79b1260673a.jpg)

![479f6b0aea906b2938d45a55e39259d0930869b12028026780abc7451d21beca.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/479f6b0aea906b2938d45a55e39259d0930869b12028026780abc7451d21beca.jpg)

![577601d4b5e958a08f40595c05c23ab13ccea708ea2ceb660d90990cd0a529a5.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/577601d4b5e958a08f40595c05c23ab13ccea708ea2ceb660d90990cd0a529a5.jpg)

![6cb945f32a6b1f96cb3e79b5d8f5d0b24da3f5921b43fcdff9d53b0b50e7033b.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/6cb945f32a6b1f96cb3e79b5d8f5d0b24da3f5921b43fcdff9d53b0b50e7033b.jpg)

![751a73b1b5a55a3de157488822d134447a02bfebb08f04d6660cbb9a58938762.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/751a73b1b5a55a3de157488822d134447a02bfebb08f04d6660cbb9a58938762.jpg)

![7ce9c0083b79bd56d343a8eb0f1a038e019ed53927a14905b0be059f63da3dda.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/7ce9c0083b79bd56d343a8eb0f1a038e019ed53927a14905b0be059f63da3dda.jpg)

![7ed953084385320b10cf1dce6a80dde5f2bc88a166bb833722e5445efab00eb7.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/7ed953084385320b10cf1dce6a80dde5f2bc88a166bb833722e5445efab00eb7.jpg)

![95349d7888182d9b0de5dccad8a701c71bd6f72a4c89f4ed812afec98af8c227.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/95349d7888182d9b0de5dccad8a701c71bd6f72a4c89f4ed812afec98af8c227.jpg)

![987d8b9aa15c038866ba51a78b04980b964e12c7949c9027a162fc60a42c4378.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/987d8b9aa15c038866ba51a78b04980b964e12c7949c9027a162fc60a42c4378.jpg)

![9b958ba4de7e49e7b519d3cccda0e743c377b1c1586f86d035d73cf83e0ac71a.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/9b958ba4de7e49e7b519d3cccda0e743c377b1c1586f86d035d73cf83e0ac71a.jpg)

![9e9f1332e6d718bea153b7376ea84149fcdcfa561d1ef148859e25304d27aec3.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/9e9f1332e6d718bea153b7376ea84149fcdcfa561d1ef148859e25304d27aec3.jpg)

![a57a81a47e42d3b4e8302b513d73641d6eb33264c921d17909397598b4eb4b7f.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/a57a81a47e42d3b4e8302b513d73641d6eb33264c921d17909397598b4eb4b7f.jpg)

![b0aef3eef20207f1d6ffd3188ab7840943c35a6680b124f7e84ea37d0f413017.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/b0aef3eef20207f1d6ffd3188ab7840943c35a6680b124f7e84ea37d0f413017.jpg)

![b2ac3b2c1ac9f3db4d97f005e55348d431d6a11cfbb1afe92d55133bccfdd0b1.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/b2ac3b2c1ac9f3db4d97f005e55348d431d6a11cfbb1afe92d55133bccfdd0b1.jpg)

![b8c58fe201ee06561ac0ef19dcc3962826d1ce22f20dad59847ebe298995fe6a.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/b8c58fe201ee06561ac0ef19dcc3962826d1ce22f20dad59847ebe298995fe6a.jpg)

![c33e16c7ddabce4b2cb4a80ef07efd84624951cad08ce14e49a600b65db6c4ea.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/c33e16c7ddabce4b2cb4a80ef07efd84624951cad08ce14e49a600b65db6c4ea.jpg)

![c3d746d7aff3b1496d13729075f0004a2b809519a2e772ac5b9a085789ce9bdd.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/c3d746d7aff3b1496d13729075f0004a2b809519a2e772ac5b9a085789ce9bdd.jpg)

![ca767a3abca5884f302dcf62ddaffd34e8ceb0166991e009219efb1ff7c50424.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/ca767a3abca5884f302dcf62ddaffd34e8ceb0166991e009219efb1ff7c50424.jpg)

![cd6dc8035c739694c05ac02f080186f29f604f3e3845ee9b5851ed2d8bbfb837.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/cd6dc8035c739694c05ac02f080186f29f604f3e3845ee9b5851ed2d8bbfb837.jpg)

![d1876079954861083381a4503dabe77e053bc704cd9da60e273bcdbcfffbebc8.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/d1876079954861083381a4503dabe77e053bc704cd9da60e273bcdbcfffbebc8.jpg)

![d49f16a676b5fdca5560c685a63e13ad1a2004c6b18c432e08191be0da7ffa95.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/d49f16a676b5fdca5560c685a63e13ad1a2004c6b18c432e08191be0da7ffa95.jpg)

![d55dbc24c1ca23ac7abd6649ce8d251afd97a98903cd3b4fbcd7deb437f76faa.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/d55dbc24c1ca23ac7abd6649ce8d251afd97a98903cd3b4fbcd7deb437f76faa.jpg)

![d5963c09d49870553b7a6d3f96ec9f79e163d705abc13c042f4adb2f54e1568d.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/d5963c09d49870553b7a6d3f96ec9f79e163d705abc13c042f4adb2f54e1568d.jpg)

![dec0f5336c3bdccd7f3679985d08b2e9487fde94d18151fc8609b9b7fb79abd3.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/dec0f5336c3bdccd7f3679985d08b2e9487fde94d18151fc8609b9b7fb79abd3.jpg)

![e487bd93d83a620ded524bec8e2d83ca66008f2e37796d6071369aa7a12ebc95.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/e487bd93d83a620ded524bec8e2d83ca66008f2e37796d6071369aa7a12ebc95.jpg)

![e9d2c3ff2b316aa54d627de377842ed4aae8a36ced581d0d7d8a4cb29cf5105e.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/e9d2c3ff2b316aa54d627de377842ed4aae8a36ced581d0d7d8a4cb29cf5105e.jpg)

![edcd90214dea87aa51a0296f5979ba6898a4a33992d474fd83ab0623bb16b167.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/edcd90214dea87aa51a0296f5979ba6898a4a33992d474fd83ab0623bb16b167.jpg)

![f40cef99fe2fc05d2aa1cdcda93bdf4fff2c33bff7b7de6ee89a5cf2dd4f55b8.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/f40cef99fe2fc05d2aa1cdcda93bdf4fff2c33bff7b7de6ee89a5cf2dd4f55b8.jpg)

![f8811a060caf216ab6182385ab77845eaeedfe94c7e5825aca28970ec4a39df8.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/f8811a060caf216ab6182385ab77845eaeedfe94c7e5825aca28970ec4a39df8.jpg)

![ff006846a8dd66bfe2e26be319939c40409388fa602d294f3924dc172fa5915a.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/images/ff006846a8dd66bfe2e26be319939c40409388fa602d294f3924dc172fa5915a.jpg)

### Tables

![055d2a8af9c3720b5547850be8d13545305338985b923c740159483b059e86bb.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/055d2a8af9c3720b5547850be8d13545305338985b923c740159483b059e86bb.jpg)

![113b4aa9216acf9551e5301e21bd768ba9616c04b236e124fa8ce69deb64af61.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/113b4aa9216acf9551e5301e21bd768ba9616c04b236e124fa8ce69deb64af61.jpg)

![2b812a1f0a4450d041460f03fa816fdb174fd76b755248f5c1527d67cb162c10.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/2b812a1f0a4450d041460f03fa816fdb174fd76b755248f5c1527d67cb162c10.jpg)

![32651d8a5ee7de32ec879fc9e5eb290e1f0287385b2551685f2f90ec9f1f18db.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/32651d8a5ee7de32ec879fc9e5eb290e1f0287385b2551685f2f90ec9f1f18db.jpg)

![3592db52ea21d98aaf97d63136098b1841284dca628677aba7566a471416a86f.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/3592db52ea21d98aaf97d63136098b1841284dca628677aba7566a471416a86f.jpg)

![7651d952b5c42b6251491784ccc37c03ffff09a0a239e27378ec8da3a47ec29c.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/7651d952b5c42b6251491784ccc37c03ffff09a0a239e27378ec8da3a47ec29c.jpg)

![7ad5d67a5097d27349856d2e7b9d835d2aa2f07ff6cfc71728768f0a23a0f235.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/7ad5d67a5097d27349856d2e7b9d835d2aa2f07ff6cfc71728768f0a23a0f235.jpg)

![7b9c2961ad7f0ff688b2b27cdc7b2b76eda4ee34a2a74a3d2665d491e48503e5.jpg](../icml_results/6113_Scalable%20RF%20Simulation%20in%20Generative%204D%20Worlds/tables/7b9c2961ad7f0ff688b2b27cdc7b2b76eda4ee34a2a74a3d2665d491e48503e5.jpg)

<a id="paper-6064"></a>

## Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing


### Images

![22f7d4c6a3f4d150e5ff8d3893b8badc03a9e8aa9bef49a807ebf300bb95fe5a.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/22f7d4c6a3f4d150e5ff8d3893b8badc03a9e8aa9bef49a807ebf300bb95fe5a.jpg)

![2da13f2d5ab1c08ded8430eb35b1712b6f8a45541e0d3796940e002a22739635.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/2da13f2d5ab1c08ded8430eb35b1712b6f8a45541e0d3796940e002a22739635.jpg)

![3d767a666b1acd3e7ce0151616e5b96cc51b04520c47e4d7b2b9748b39d91f22.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/3d767a666b1acd3e7ce0151616e5b96cc51b04520c47e4d7b2b9748b39d91f22.jpg)

![64aca18a0f00b964596f9f88e0436c66bf672b2493bd2d3e8b87b4335029c4d8.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/64aca18a0f00b964596f9f88e0436c66bf672b2493bd2d3e8b87b4335029c4d8.jpg)

![90c2d26f50bc27d88b807e2bc0e6db737c12af5ab9e1b1867c802f420acd9128.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/90c2d26f50bc27d88b807e2bc0e6db737c12af5ab9e1b1867c802f420acd9128.jpg)

![9c08c55fb5299ba17bcee15122446218ca7f3c9ca86a945b9bed63b64d8abc9c.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/9c08c55fb5299ba17bcee15122446218ca7f3c9ca86a945b9bed63b64d8abc9c.jpg)

![cca029c535ef2df6b67daf09b1a8c43c1d5810732a51ad752b90ed82a4e078ad.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/images/cca029c535ef2df6b67daf09b1a8c43c1d5810732a51ad752b90ed82a4e078ad.jpg)

### Tables

![62171d518fda0b54ad629ed7da9b89d0b788b551ff15bbf7ba0547fb45332f59.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/tables/62171d518fda0b54ad629ed7da9b89d0b788b551ff15bbf7ba0547fb45332f59.jpg)

![82c958fd657f612cecab964d78fc7e21c921fb70ab5b601efcb67e758c1b890f.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/tables/82c958fd657f612cecab964d78fc7e21c921fb70ab5b601efcb67e758c1b890f.jpg)

![a1fe7558cd8860634f9a6a2f88f8008d74acc020020dd47dc075637a25ebb4f0.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/tables/a1fe7558cd8860634f9a6a2f88f8008d74acc020020dd47dc075637a25ebb4f0.jpg)

![ca3fa89b94de8df015324f5304fd7dff295774f25126f65c9079404bcdc902d8.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/tables/ca3fa89b94de8df015324f5304fd7dff295774f25126f65c9079404bcdc902d8.jpg)

![d5535372f74d22303c5b3ff01b277e66371ce6f78987068821e0d7a38dc8ffe9.jpg](../icml_results/6114_Parallel-Probe_%20Towards%20Efficient%20Parallel%20Thinking%20via%202D%20Probing/tables/d5535372f74d22303c5b3ff01b277e66371ce6f78987068821e0d7a38dc8ffe9.jpg)

<a id="paper-6065"></a>

## Bilevel Optimization over Saddle Points of Zero-Sum Markov Games


### Images

![2351cb35f9c2bb9f5927e880407bd310b65659165f241bf2962ed3026c50957c.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/2351cb35f9c2bb9f5927e880407bd310b65659165f241bf2962ed3026c50957c.jpg)

![3042ab4520ca3b9de60239bfc0eb232b295a603bffb289ad92363ace39c23be7.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/3042ab4520ca3b9de60239bfc0eb232b295a603bffb289ad92363ace39c23be7.jpg)

![3641d68fe02441a1080d32872bc509228056e88f63cd214e30751e9abd6a250d.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/3641d68fe02441a1080d32872bc509228056e88f63cd214e30751e9abd6a250d.jpg)

![6b9390e3b6d8765e3c4cd5697918969af6d2ed045c5c6a764bf77f2f415e3866.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/6b9390e3b6d8765e3c4cd5697918969af6d2ed045c5c6a764bf77f2f415e3866.jpg)

![8546f77098fcc321b32f4225ad4c3c82c5eda6f7d9ea0a80911b9ba11ef0eebd.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/8546f77098fcc321b32f4225ad4c3c82c5eda6f7d9ea0a80911b9ba11ef0eebd.jpg)

![adc08c02978f3da0c8ac5cbd936f97e5acd6a1501d0d33f8c4012cfc7760b509.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/adc08c02978f3da0c8ac5cbd936f97e5acd6a1501d0d33f8c4012cfc7760b509.jpg)

![b847407003f50dedf362a044ff2c08ccdb9a969d4e9c27cd09adda2a11f04299.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/images/b847407003f50dedf362a044ff2c08ccdb9a969d4e9c27cd09adda2a11f04299.jpg)

### Tables

![0ffac94012ba6ebcfb7da93294c773573b3a179491c31f93471777d147853fba.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/tables/0ffac94012ba6ebcfb7da93294c773573b3a179491c31f93471777d147853fba.jpg)

![ad39e5f41e1a1eaa25bb239d158d07686188544841ca8806cfb4c0f22576ced9.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/tables/ad39e5f41e1a1eaa25bb239d158d07686188544841ca8806cfb4c0f22576ced9.jpg)

![b1d046b10928fc653270fc0482b8e58b83b783d8df7490e368b35a3572dea065.jpg](../icml_results/6115_Bilevel%20Optimization%20over%20Saddle%20Points%20of%20Zero-Sum%20Markov%20Games/tables/b1d046b10928fc653270fc0482b8e58b83b783d8df7490e368b35a3572dea065.jpg)

<a id="paper-6066"></a>

## Proxy Compression for Language Modeling


### Images

![10a047b92aa686bc4e5f7779dadecd20d1e07867add3a4e3bbc3dc4023702c88.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/10a047b92aa686bc4e5f7779dadecd20d1e07867add3a4e3bbc3dc4023702c88.jpg)

![160fd97458523e11215210cb9d494388cf3db2889ce9a70a726c40ab429f9c80.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/160fd97458523e11215210cb9d494388cf3db2889ce9a70a726c40ab429f9c80.jpg)

![27862da2876502eeac287f40ff2ee4263ff90f60b88233a0d54d8e5636084f1f.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/27862da2876502eeac287f40ff2ee4263ff90f60b88233a0d54d8e5636084f1f.jpg)

![46a17aa00ceeb9629f2215835db5e9333e7a982397333228454f7f16adec4161.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/46a17aa00ceeb9629f2215835db5e9333e7a982397333228454f7f16adec4161.jpg)

![46c0abdc4bedcb1e181dbf5d41b37c741839ec1c128d08ae2be0be33387c0162.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/46c0abdc4bedcb1e181dbf5d41b37c741839ec1c128d08ae2be0be33387c0162.jpg)

![54d944678496a6810c2a0b9e5a2a8e57b0d7dab1bd10131561be3d489d8917ca.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/54d944678496a6810c2a0b9e5a2a8e57b0d7dab1bd10131561be3d489d8917ca.jpg)

![5801a7545cbb982fc11a00553e5f6dbd8e6d47b541a1973907d7ad51c4cc2503.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/5801a7545cbb982fc11a00553e5f6dbd8e6d47b541a1973907d7ad51c4cc2503.jpg)

![58444a877d8a2c5588084f5bffdcf59fc104f6d77afcfe116825ad72e497f5c9.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/58444a877d8a2c5588084f5bffdcf59fc104f6d77afcfe116825ad72e497f5c9.jpg)

![61de4dd51f484bc981a76b549bd916365a3a2453efda449b0a438f1c61dabd54.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/61de4dd51f484bc981a76b549bd916365a3a2453efda449b0a438f1c61dabd54.jpg)

![6a240df695567b00d2ae6ab71a38bd2d2c7f5536d3ff4a2c4d8741e62b726279.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/6a240df695567b00d2ae6ab71a38bd2d2c7f5536d3ff4a2c4d8741e62b726279.jpg)

![770dd971ced0fff67d5f7ca6b58ce6854da9297c6bac69d7f1755f3a638d5bc9.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/770dd971ced0fff67d5f7ca6b58ce6854da9297c6bac69d7f1755f3a638d5bc9.jpg)

![7f1e4b9da8f7a8fb93be2068cb589ec09901cf9ec9945fe73643ae60772505ed.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/7f1e4b9da8f7a8fb93be2068cb589ec09901cf9ec9945fe73643ae60772505ed.jpg)

![8fd071fd8a922c9bc03e31daf96b99aa04514c86cbcc8ac02a9d8453e302682f.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/8fd071fd8a922c9bc03e31daf96b99aa04514c86cbcc8ac02a9d8453e302682f.jpg)

![95c8717f896c8212cb5d4628c637c2a81511ff636cbc97a910641db455e0f49f.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/95c8717f896c8212cb5d4628c637c2a81511ff636cbc97a910641db455e0f49f.jpg)

![9e6cce2ba97cc0b90210f8e0c40d1e5d4a1c20a5c5cf3a0d642d8385f8835aec.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/9e6cce2ba97cc0b90210f8e0c40d1e5d4a1c20a5c5cf3a0d642d8385f8835aec.jpg)

![b57652b74c8dc625b0137b9c155a78f48e00a315ec976d1a8391bfb18c6160ab.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/b57652b74c8dc625b0137b9c155a78f48e00a315ec976d1a8391bfb18c6160ab.jpg)

![bfd3b496c8c0fc8f4380b0b32d67b532c10bc063e3f6ab0d9c5477de65ff7573.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/bfd3b496c8c0fc8f4380b0b32d67b532c10bc063e3f6ab0d9c5477de65ff7573.jpg)

![e0f723fc369327877c76da9e6300eed40a59e21c92754e566a4c21fe38c2d201.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/e0f723fc369327877c76da9e6300eed40a59e21c92754e566a4c21fe38c2d201.jpg)

![e162b27838b0e2a40f4b7a5d5f9009c7d835b59520018fb5ddb9b29bec40a700.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/e162b27838b0e2a40f4b7a5d5f9009c7d835b59520018fb5ddb9b29bec40a700.jpg)

![e61f16393a16b050bc5f164a655729da954d2e6b51ef502cc701607ff76609bb.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/e61f16393a16b050bc5f164a655729da954d2e6b51ef502cc701607ff76609bb.jpg)

![ffee47b4799c35dec87afd1cbdf96ee19b477af18cd3b0a4bbe723b2df795b03.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/images/ffee47b4799c35dec87afd1cbdf96ee19b477af18cd3b0a4bbe723b2df795b03.jpg)

### Tables

![00c965554caa4f299d0577096c3458cb5a010ad7f4a70539ed4fac657083463d.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/00c965554caa4f299d0577096c3458cb5a010ad7f4a70539ed4fac657083463d.jpg)

![1118893ffa3e77bec6a087f597ac2cae517901f99ceeef8bff151be7af7df265.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/1118893ffa3e77bec6a087f597ac2cae517901f99ceeef8bff151be7af7df265.jpg)

![1c022b796b1fb3317501262c6a6a660bbaf2087593d83a1762f2eb3e59245654.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/1c022b796b1fb3317501262c6a6a660bbaf2087593d83a1762f2eb3e59245654.jpg)

![3b66c8bae284550d3dedadf1c21bb31496a06e7bdbfef6b01c3a11e1504efa9e.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/3b66c8bae284550d3dedadf1c21bb31496a06e7bdbfef6b01c3a11e1504efa9e.jpg)

![3b92d6d0dc900c40c38924c28f2fb298a869aa93bea8305347c68fec294b1ff2.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/3b92d6d0dc900c40c38924c28f2fb298a869aa93bea8305347c68fec294b1ff2.jpg)

![488a7dd106c8468bb2f4700b581a511d5066a574fe91cbf87b8475b540402196.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/488a7dd106c8468bb2f4700b581a511d5066a574fe91cbf87b8475b540402196.jpg)

![5aed474c6058085ecf551348a90bee9fe0ce043bfa980f5571db7da4d99541e3.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/5aed474c6058085ecf551348a90bee9fe0ce043bfa980f5571db7da4d99541e3.jpg)

![654fb78729553e2f1596cb9e165c3a8ca804d8a81380e3b7e3546a884a455b6d.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/654fb78729553e2f1596cb9e165c3a8ca804d8a81380e3b7e3546a884a455b6d.jpg)

![791d047f17537ce8bafabac4d7706fc29b4d32533ea6d4f68d921b5282ab3013.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/791d047f17537ce8bafabac4d7706fc29b4d32533ea6d4f68d921b5282ab3013.jpg)

![9935508ba8607a85330b31912ed0f4b2b66807918829d5bc15fb0b24b0cf7a2c.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/9935508ba8607a85330b31912ed0f4b2b66807918829d5bc15fb0b24b0cf7a2c.jpg)

![a5eaaa004d8d00190fb6010bd56cfd7e5d2ef7883ccbbf671479c6836b25df8f.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/a5eaaa004d8d00190fb6010bd56cfd7e5d2ef7883ccbbf671479c6836b25df8f.jpg)

![aa632178c378f30bcc35d2965632b958a6e20840bc808ba324bb41612932bc1a.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/aa632178c378f30bcc35d2965632b958a6e20840bc808ba324bb41612932bc1a.jpg)

![af11f2c2cc55dc876e9eee343fd6a974fdfbdbcd3f8900345c5cc6421cb9c749.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/af11f2c2cc55dc876e9eee343fd6a974fdfbdbcd3f8900345c5cc6421cb9c749.jpg)

![b56e5035c87fdcef9b1f6f4e387eac016a1078f18c12ccb4cde54fa0a6c3f11a.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/b56e5035c87fdcef9b1f6f4e387eac016a1078f18c12ccb4cde54fa0a6c3f11a.jpg)

![e25efa5eca7b88b26d583123475c6b4876fa6d763512a5d11ed3695385596c7c.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/e25efa5eca7b88b26d583123475c6b4876fa6d763512a5d11ed3695385596c7c.jpg)

![fb337bf5e2a27c51cf819a002b9a316d2a998b32643168f2efbf45429b0b3475.jpg](../icml_results/6116_Proxy%20Compression%20for%20Language%20Modeling/tables/fb337bf5e2a27c51cf819a002b9a316d2a998b32643168f2efbf45429b0b3475.jpg)

<a id="paper-6067"></a>

## Image Restoration via Diffusion Models with Dynamic Resolution


### Images

![1f2b1c0cb2ea6236166006b18f99cb5a63f496deef8ef65dfc8e8b739564066a.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/1f2b1c0cb2ea6236166006b18f99cb5a63f496deef8ef65dfc8e8b739564066a.jpg)

![2136e65af3e5f3dfa6c9b801d33baeea290ba028def19f9f3b6c3710989d12d3.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/2136e65af3e5f3dfa6c9b801d33baeea290ba028def19f9f3b6c3710989d12d3.jpg)

![51b871bb79290c0d8288d49a58e7059b994e22463dd2358e77e59f4f4899410a.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/51b871bb79290c0d8288d49a58e7059b994e22463dd2358e77e59f4f4899410a.jpg)

![754fa0b4e539ac4b5524758153d37ed51e9758859059e8a5ad90eb569451d7c6.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/754fa0b4e539ac4b5524758153d37ed51e9758859059e8a5ad90eb569451d7c6.jpg)

![9537352f71a15a4864fb520610cbbb05bb9c4d57615beff4ea9015eaa97924ce.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/9537352f71a15a4864fb520610cbbb05bb9c4d57615beff4ea9015eaa97924ce.jpg)

![aff79f0e706f285999f67bdf604edb20c22fbbc9f852104ea92a7579d74e12e9.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/aff79f0e706f285999f67bdf604edb20c22fbbc9f852104ea92a7579d74e12e9.jpg)

![b09995de6ea0a29ab65966c9a4948cdd0824295608a902033031823e8d25dbad.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/b09995de6ea0a29ab65966c9a4948cdd0824295608a902033031823e8d25dbad.jpg)

![d6467cfbfc1e5e84446ba1a9220e3351c0935551e87e287d716c89e8d3098571.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/d6467cfbfc1e5e84446ba1a9220e3351c0935551e87e287d716c89e8d3098571.jpg)

![de8628487aa3899a5659deb4cb8c714dfa873e8018e68ac570213b6269ce5b97.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/images/de8628487aa3899a5659deb4cb8c714dfa873e8018e68ac570213b6269ce5b97.jpg)

### Tables

![17488a2267a12a6326867bec7bfb284b029404b1aebc98c084b422f58ff6ea01.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/17488a2267a12a6326867bec7bfb284b029404b1aebc98c084b422f58ff6ea01.jpg)

![2371ecf99b9b3f605f01e1b719422d6c895ef2958142e92c7894d8ebf2ab094b.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/2371ecf99b9b3f605f01e1b719422d6c895ef2958142e92c7894d8ebf2ab094b.jpg)

![3a8131515a9889c68a71f011728a8b8490f7e119f45db7b219f7462f01ea2dd7.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/3a8131515a9889c68a71f011728a8b8490f7e119f45db7b219f7462f01ea2dd7.jpg)

![3fb9a3fc624f7f14a36b182b686e7fcde3823052ae21440635441f64af0b4546.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/3fb9a3fc624f7f14a36b182b686e7fcde3823052ae21440635441f64af0b4546.jpg)

![436346d454f401bff2ee3960f83253474ca46ce68c74a6ef18a60202ba3fe0e7.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/436346d454f401bff2ee3960f83253474ca46ce68c74a6ef18a60202ba3fe0e7.jpg)

![85e29957a3c0901e34caddf77d52d4290f5ff3cf12c953282cde941a3e3794af.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/85e29957a3c0901e34caddf77d52d4290f5ff3cf12c953282cde941a3e3794af.jpg)

![989140d81e3631f3bf9656a5a60333904c8862660b087f842c6c3612fd74c967.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/989140d81e3631f3bf9656a5a60333904c8862660b087f842c6c3612fd74c967.jpg)

![bf238c35271978386d9ef8990d30a3f3c72cc9011a7fe426d7ee3fb0871e29e2.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/bf238c35271978386d9ef8990d30a3f3c72cc9011a7fe426d7ee3fb0871e29e2.jpg)

![cac4141e9a1a1daa0d9a7accf116c1af625056f9340318a4f02799fd74f8f357.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/cac4141e9a1a1daa0d9a7accf116c1af625056f9340318a4f02799fd74f8f357.jpg)

![d4b79c465b4fc0b04a0ddc78c1392da307973bbeee92cb0902ec16836b6c8c9f.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/d4b79c465b4fc0b04a0ddc78c1392da307973bbeee92cb0902ec16836b6c8c9f.jpg)

![e021170c183a393d620262e0099b73d4b9bfb4b1923f968bbc6bf0b8c4cf5faf.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/e021170c183a393d620262e0099b73d4b9bfb4b1923f968bbc6bf0b8c4cf5faf.jpg)

![ed3d5f42656182bd568758967453d33f9a354166eb9e963aa1d0ef482aa274ad.jpg](../icml_results/6117_Image%20Restoration%20via%20Diffusion%20Models%20with%20Dynamic%20Resolution/tables/ed3d5f42656182bd568758967453d33f9a354166eb9e963aa1d0ef482aa274ad.jpg)

<a id="paper-6068"></a>

## Doppler Prompting for Stable mmWave-based Human Pose Estimation


### Images

![76715d5f6b7441590772280d6c63b5c6f6a00a2bc2cdfcc6b2f1a19118649d96.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/76715d5f6b7441590772280d6c63b5c6f6a00a2bc2cdfcc6b2f1a19118649d96.jpg)

![7ef55b98632d3af133da23397a6a6b442b65cc24352cbe2daf7aa4fe426de7ac.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/7ef55b98632d3af133da23397a6a6b442b65cc24352cbe2daf7aa4fe426de7ac.jpg)

![b26d57d9a03c057ad4c7891e08e716f9613889c0712d0bb655d8239eb6e57bb8.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/b26d57d9a03c057ad4c7891e08e716f9613889c0712d0bb655d8239eb6e57bb8.jpg)

![c2fbf9123e155c6b8c1f53ebccac780e0c7c37ccc47b87a05457a558454f8a58.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/c2fbf9123e155c6b8c1f53ebccac780e0c7c37ccc47b87a05457a558454f8a58.jpg)

![ecafda8a7e95fdcc348993e2956457a085e4098be89a2607dcca40e4dad1dfb6.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/ecafda8a7e95fdcc348993e2956457a085e4098be89a2607dcca40e4dad1dfb6.jpg)

![fc2120622012fce105be7558d3fbc2962d668641c387b74341e137b42f63e0cf.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/images/fc2120622012fce105be7558d3fbc2962d668641c387b74341e137b42f63e0cf.jpg)

### Tables

![08213be412e7a3b69fe5452c2a8a49d7e93fc3930a032671ba79aeb8711a03cc.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/08213be412e7a3b69fe5452c2a8a49d7e93fc3930a032671ba79aeb8711a03cc.jpg)

![1c521d6de4383ac5400a1312fe98ea2431b323797c4a4c2454bcabe210913eae.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/1c521d6de4383ac5400a1312fe98ea2431b323797c4a4c2454bcabe210913eae.jpg)

![42f728c6e3233dddc359ea3da77ca5cfb5ad3376f74d8f84411a677aba772578.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/42f728c6e3233dddc359ea3da77ca5cfb5ad3376f74d8f84411a677aba772578.jpg)

![47c042cc42d07054955e50f6b2977c8ff57f3878a571e1c8ddb98ab5a392d307.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/47c042cc42d07054955e50f6b2977c8ff57f3878a571e1c8ddb98ab5a392d307.jpg)

![4d9408dacef671d0ee468622675cb892089be02b17a1880ffed6a5f1f5f4cbc7.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/4d9408dacef671d0ee468622675cb892089be02b17a1880ffed6a5f1f5f4cbc7.jpg)

![4eb60d4e26c61355d2676b4faf94540f79c342ff620eb4614eeec50dcc5b2dff.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/4eb60d4e26c61355d2676b4faf94540f79c342ff620eb4614eeec50dcc5b2dff.jpg)

![70d7abb97190c66d77be60b53d8c67001ee1742ccb226201ae8674dc2c8c0816.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/70d7abb97190c66d77be60b53d8c67001ee1742ccb226201ae8674dc2c8c0816.jpg)

![7bd503458fa4decdad801899b42b11179448b991b1ee656a1ffc911ebc1cfd17.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/7bd503458fa4decdad801899b42b11179448b991b1ee656a1ffc911ebc1cfd17.jpg)

![8077dbbccaa67a94fce507f8e07c7814aa8fffb2a9576aaee407b54d242331de.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/8077dbbccaa67a94fce507f8e07c7814aa8fffb2a9576aaee407b54d242331de.jpg)

![ac64e253f568582929e75db095ef2b1044c515a5f04a0758d6b6f666b9f41950.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/ac64e253f568582929e75db095ef2b1044c515a5f04a0758d6b6f666b9f41950.jpg)

![b6070995c84ca31d5c4dd275f06997e69cfdfcc35461a99499734392331d8788.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/b6070995c84ca31d5c4dd275f06997e69cfdfcc35461a99499734392331d8788.jpg)

![c13e515ee0245bae5de9ef5dbbea2eb5eda67a253bf7f10361ca1598644e3a02.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/c13e515ee0245bae5de9ef5dbbea2eb5eda67a253bf7f10361ca1598644e3a02.jpg)

![cc2d4a31cbb969659079fe6d09ea68d00ba733f3fbf4f352a38bd98900e5efde.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/cc2d4a31cbb969659079fe6d09ea68d00ba733f3fbf4f352a38bd98900e5efde.jpg)

![f0a42f0287d9176081ad6797109c9e72da3b670efa92c2bc943942477f7b0f6a.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/f0a42f0287d9176081ad6797109c9e72da3b670efa92c2bc943942477f7b0f6a.jpg)

![fc31c6fba96484f0b43c8b94ddc5df9afe33588a69b469fb530e325675bb1fb4.jpg](../icml_results/6118_Doppler%20Prompting%20for%20Stable%20mmWave-based%20Human%20Pose%20Estimation/tables/fc31c6fba96484f0b43c8b94ddc5df9afe33588a69b469fb530e325675bb1fb4.jpg)

<a id="paper-6069"></a>

## Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis


### Images

![136bf4ed83a3cdf313bcb38aef7a33389e63d4a265f712e4d86403082aacfce7.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/136bf4ed83a3cdf313bcb38aef7a33389e63d4a265f712e4d86403082aacfce7.jpg)

![41ea31485d96ebb3834d7bcc4279b5f6275ef7caa511c9d65f419c1bbc999c65.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/41ea31485d96ebb3834d7bcc4279b5f6275ef7caa511c9d65f419c1bbc999c65.jpg)

![4ca0f5422fbceecbad45a38c487bde3fc259db939b73c5dd278645e4c7076d7c.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/4ca0f5422fbceecbad45a38c487bde3fc259db939b73c5dd278645e4c7076d7c.jpg)

![68f20d892dfb5f2a76d85ca1a1a07d03ef67457a36e7d9ef902e0e440ddaee46.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/68f20d892dfb5f2a76d85ca1a1a07d03ef67457a36e7d9ef902e0e440ddaee46.jpg)

![6f742a0ce5959d536cf6b9cd0db33fe3d05f7ea33e1aedf7f490f5b292ae05c8.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/6f742a0ce5959d536cf6b9cd0db33fe3d05f7ea33e1aedf7f490f5b292ae05c8.jpg)

![73bf1224a4fb6f18d2689c1ac045d8dfbe7edb45abe24f91a6fe1523fda1c4f5.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/73bf1224a4fb6f18d2689c1ac045d8dfbe7edb45abe24f91a6fe1523fda1c4f5.jpg)

![7e3cdf6d843d8f48254db19535fc8ca8f77ced9a640e0149613bfd59398a76e7.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/7e3cdf6d843d8f48254db19535fc8ca8f77ced9a640e0149613bfd59398a76e7.jpg)

![9107caa861af34dd76c344238625285a1adf10ed1f887138a0d75632a0ed4ed4.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/9107caa861af34dd76c344238625285a1adf10ed1f887138a0d75632a0ed4ed4.jpg)

![bb7ccea2ba389e663bcc2c6f79e30c8c8ad66bbac28d11755a5a5d5c501538ea.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/bb7ccea2ba389e663bcc2c6f79e30c8c8ad66bbac28d11755a5a5d5c501538ea.jpg)

![c334d08032706571c5c183109e9d2704b4430c1fd8d9b76945114a7ef94e2013.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/c334d08032706571c5c183109e9d2704b4430c1fd8d9b76945114a7ef94e2013.jpg)

![c6ba4a874ad16bf829a2a83781eb73a40f067f5cffba62784c4e599ba5dcb58c.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/images/c6ba4a874ad16bf829a2a83781eb73a40f067f5cffba62784c4e599ba5dcb58c.jpg)

### Tables

![04036565bf7ff9adec8713779b211d7925aedfeef53c9aca7e7fee336f41f9cd.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/04036565bf7ff9adec8713779b211d7925aedfeef53c9aca7e7fee336f41f9cd.jpg)

![1d357d979d5f349fc9c45c1a30f969fd0cec894d62958d227bea162daa64ffec.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/1d357d979d5f349fc9c45c1a30f969fd0cec894d62958d227bea162daa64ffec.jpg)

![2595bff5c812290aab7080589af24c3ca43883ff3d2511e4460f3fb2163d1163.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/2595bff5c812290aab7080589af24c3ca43883ff3d2511e4460f3fb2163d1163.jpg)

![2b13d9912a09be2e6953397410d8b73dbb00c42fa09036f46a21e1759047a2a9.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/2b13d9912a09be2e6953397410d8b73dbb00c42fa09036f46a21e1759047a2a9.jpg)

![8ca8b6b97ffe64ef39ba3ec7986cc5ba9f5ee315cba1e8ea63a3c02c17b4247c.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/8ca8b6b97ffe64ef39ba3ec7986cc5ba9f5ee315cba1e8ea63a3c02c17b4247c.jpg)

![a8bc1169cdc163c1e30b972fe0eba2a12f63dae27dcd6634307390e5a46df6df.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/a8bc1169cdc163c1e30b972fe0eba2a12f63dae27dcd6634307390e5a46df6df.jpg)

![cff9b83ee463eb8689bead303aef5fa00f718ad7a40d088600e8c784e4b1cbc4.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/cff9b83ee463eb8689bead303aef5fa00f718ad7a40d088600e8c784e4b1cbc4.jpg)

![fd34b1c91e47e19b2abcfc235216847a684a43225ebeffe0bdbb7b1c277a6d03.jpg](../icml_results/6119_Towards%20Cold-Start%20Drafting%20and%20Continual%20Refining_%20A%20Value-Driven%20Memory%20Approach%20with%20Application%20/tables/fd34b1c91e47e19b2abcfc235216847a684a43225ebeffe0bdbb7b1c277a6d03.jpg)

<a id="paper-6070"></a>

## UniCode: Augmenting Evaluation for Code Reasoning


### Images

![002a14ccf536d90f6b98bd763a15c2fafc9ab6be2c4867fa1d86afced9180f8c.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/002a14ccf536d90f6b98bd763a15c2fafc9ab6be2c4867fa1d86afced9180f8c.jpg)

![1ca9a5c43f9bccd4a38eb61179d8235e64fa979fa78086b6e7c167979c862604.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/1ca9a5c43f9bccd4a38eb61179d8235e64fa979fa78086b6e7c167979c862604.jpg)

![216d949e99dcb6ae535f0a6d82fb8ac2bde65b7a94146c2eb94fbc7f1769c6d3.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/216d949e99dcb6ae535f0a6d82fb8ac2bde65b7a94146c2eb94fbc7f1769c6d3.jpg)

![2427e1b980bea7c2267e1e4a7e5d599663f2fec94e3f5ebb039f07b4cecd2e5d.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/2427e1b980bea7c2267e1e4a7e5d599663f2fec94e3f5ebb039f07b4cecd2e5d.jpg)

![276ead335cdffcceb2a6f3637bf895762f2095a65a5037ae94fb214959e3950c.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/276ead335cdffcceb2a6f3637bf895762f2095a65a5037ae94fb214959e3950c.jpg)

![419271db84068f598c8713c16fa01d5601e26e4ed6af19366a2e1de0d0e6a3a3.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/419271db84068f598c8713c16fa01d5601e26e4ed6af19366a2e1de0d0e6a3a3.jpg)

![7176fab1b8725365913a6361c2ad980b973661f1821fc0b4df5609282fb83260.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/7176fab1b8725365913a6361c2ad980b973661f1821fc0b4df5609282fb83260.jpg)

![81b191214493688e91838b16f3d372a87689601b2e77369d2f109f9e778a8287.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/81b191214493688e91838b16f3d372a87689601b2e77369d2f109f9e778a8287.jpg)

![88d0e3def806d82bfc78f93303e6be1499ee0aafec38af9c14d5bdb02336eeab.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/88d0e3def806d82bfc78f93303e6be1499ee0aafec38af9c14d5bdb02336eeab.jpg)

![a3d3318c139ed786a8cd67515abb729c96803a6d2a551ce8ac6977e41f92b0a8.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/a3d3318c139ed786a8cd67515abb729c96803a6d2a551ce8ac6977e41f92b0a8.jpg)

![bdf1ecd8684b830d309b777082687682a86ae6a3e07d1baf4890033888e130c5.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/bdf1ecd8684b830d309b777082687682a86ae6a3e07d1baf4890033888e130c5.jpg)

![bee33e1f7c4305931b0b96fc5d4cc4671f45dfeeb9e4690e18e44e825918fa08.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/bee33e1f7c4305931b0b96fc5d4cc4671f45dfeeb9e4690e18e44e825918fa08.jpg)

![d3c5391af1adb89912f62a8a9e7b043e22b779f21d3b315672334eef1782d129.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/d3c5391af1adb89912f62a8a9e7b043e22b779f21d3b315672334eef1782d129.jpg)

![f3f224d686f77d14b28f893bdb363c276c6055b6655c79731793c65c368bdec0.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/f3f224d686f77d14b28f893bdb363c276c6055b6655c79731793c65c368bdec0.jpg)

![f7eccc8f4018c05572aa631db68815175e58f171132922d249004c0fad02f9ad.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/f7eccc8f4018c05572aa631db68815175e58f171132922d249004c0fad02f9ad.jpg)

![fb847f5393bda2eeee0c7781b91694b0862894697e66ebdc84d57c07f90bcdf9.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/images/fb847f5393bda2eeee0c7781b91694b0862894697e66ebdc84d57c07f90bcdf9.jpg)

### Tables

![033c1a76179e3583640deb18a2acfd14b7130eb70830bac61d115c3f0fdd51e5.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/033c1a76179e3583640deb18a2acfd14b7130eb70830bac61d115c3f0fdd51e5.jpg)

![2f56edc771e9deeda3b32c26ebf24afcc99675922f0f2cec250eecb853eb7df3.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/2f56edc771e9deeda3b32c26ebf24afcc99675922f0f2cec250eecb853eb7df3.jpg)

![57a0888fd5ee0db7e252dfe2015a5fd55225745528148b55ed5c963792b05264.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/57a0888fd5ee0db7e252dfe2015a5fd55225745528148b55ed5c963792b05264.jpg)

![a6533a853d2f9e79ec17602ddfe0d81595047a4178531408a245972946bf014e.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/a6533a853d2f9e79ec17602ddfe0d81595047a4178531408a245972946bf014e.jpg)

![cd6f4959128f17f28ab98c4baa2b35c238e3c3fc383295a0c9ee7b295bb76a7b.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/cd6f4959128f17f28ab98c4baa2b35c238e3c3fc383295a0c9ee7b295bb76a7b.jpg)

![d0cd4a05b4913617dc889f6b1959cc338215249a733f5ab78248371289ab188f.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/d0cd4a05b4913617dc889f6b1959cc338215249a733f5ab78248371289ab188f.jpg)

![e20d513dc221adcc68ca2ab4e5b26c7243848c2ff0507b0109c1fa9dd8bec4cd.jpg](../icml_results/6120_UniCode_%20Augmenting%20Evaluation%20for%20Code%20Reasoning/tables/e20d513dc221adcc68ca2ab4e5b26c7243848c2ff0507b0109c1fa9dd8bec4cd.jpg)

<a id="paper-6071"></a>

## WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation


### Images

![00f89416da2251a347ae455cbfa18976bb6fdbcaa0347c6923f233e684220d13.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/00f89416da2251a347ae455cbfa18976bb6fdbcaa0347c6923f233e684220d13.jpg)

![16da08a2da8b9834d8b5da9a94a8a3905a47f94e2067246d3ae1691119a75e57.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/16da08a2da8b9834d8b5da9a94a8a3905a47f94e2067246d3ae1691119a75e57.jpg)

![197b8344c34d20b542d908e2ad8b83053118179b6ae74f97137beb8a645c2ead.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/197b8344c34d20b542d908e2ad8b83053118179b6ae74f97137beb8a645c2ead.jpg)

![25b4cf36cf8292262f9ff2769777f273e4b38ac5215435839568b73c488aeabd.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/25b4cf36cf8292262f9ff2769777f273e4b38ac5215435839568b73c488aeabd.jpg)

![3baebd5f1fd5d37c599d29cd880b22c74b9a023d2ec92afe74db1f8dc76d11df.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/3baebd5f1fd5d37c599d29cd880b22c74b9a023d2ec92afe74db1f8dc76d11df.jpg)

![4cd96b149b6d94bc04e919c261f0e9b2c1aa88bf54bab8c835984296f229e869.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/4cd96b149b6d94bc04e919c261f0e9b2c1aa88bf54bab8c835984296f229e869.jpg)

![5fa30c6d2bb52b359e17806332b6557a8348032164c4eab3b9dd7831c0e86839.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/5fa30c6d2bb52b359e17806332b6557a8348032164c4eab3b9dd7831c0e86839.jpg)

![68742d9687b9e9ff18331b55b8320bfff2a2d6b37e13e009a67912c9042409d1.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/68742d9687b9e9ff18331b55b8320bfff2a2d6b37e13e009a67912c9042409d1.jpg)

![8c292e98c29060494e217416c41f7a3d8e7cfa609aff4b50fd8c1cf14576b553.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/8c292e98c29060494e217416c41f7a3d8e7cfa609aff4b50fd8c1cf14576b553.jpg)

![9ce91fdf1620a167bf9c2ed7cde110a431ec3b41f655be36049ccff87b231bd9.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/9ce91fdf1620a167bf9c2ed7cde110a431ec3b41f655be36049ccff87b231bd9.jpg)

![a21e7237110a8d710ea3dc65bd39b0639af7f1966d7731ee3bf83db320ce9ece.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/a21e7237110a8d710ea3dc65bd39b0639af7f1966d7731ee3bf83db320ce9ece.jpg)

![c3724b1492e84bbaa7349c0c14f5570dcdbb92409624844243219d9040d45d18.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/c3724b1492e84bbaa7349c0c14f5570dcdbb92409624844243219d9040d45d18.jpg)

![c9e8e689b224319a9b92cee61e4f290f8a0671d270b21d736c9b98b756c7615f.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/c9e8e689b224319a9b92cee61e4f290f8a0671d270b21d736c9b98b756c7615f.jpg)

![e66cc8268a81c8d219e500fbbaa96d70cb54625f58668f3e19c031f4333921d6.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/e66cc8268a81c8d219e500fbbaa96d70cb54625f58668f3e19c031f4333921d6.jpg)

![fb03f57a6ae5dd83106d7a465cbffa273970496a77444b556db2e1f88aa0b83b.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/images/fb03f57a6ae5dd83106d7a465cbffa273970496a77444b556db2e1f88aa0b83b.jpg)

### Tables

![04c48df7e2fd27927fe4fa8c539b06b36bd9c4f76f78912f231febf57784092d.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/04c48df7e2fd27927fe4fa8c539b06b36bd9c4f76f78912f231febf57784092d.jpg)

![0c04f2208c7b6f0ef5d6684b2014e9eb15bddfb6460f85582b8968cc1406681c.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/0c04f2208c7b6f0ef5d6684b2014e9eb15bddfb6460f85582b8968cc1406681c.jpg)

![12b4236ff69ea340bbbe558287d51494026a1e78ebb9e4e288a60b2675b99f08.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/12b4236ff69ea340bbbe558287d51494026a1e78ebb9e4e288a60b2675b99f08.jpg)

![1bffb18eed5f8a2fa65572d3b1844699856e7f66de5ca4932c6c5d6bd7aa05fc.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/1bffb18eed5f8a2fa65572d3b1844699856e7f66de5ca4932c6c5d6bd7aa05fc.jpg)

![25f693c2f71a30fdeb44e1f9bcbce6a8a2115b4f94ea354d658a1db6aff2b619.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/25f693c2f71a30fdeb44e1f9bcbce6a8a2115b4f94ea354d658a1db6aff2b619.jpg)

![6b8a26b9b9b81f351967b9647bed64dfd11835322948a0f9ce74c5445e49cb82.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/6b8a26b9b9b81f351967b9647bed64dfd11835322948a0f9ce74c5445e49cb82.jpg)

![9bff1de6a40702546875e73a0960109043f5c7e0bc212dc42e266c38f04d1a7d.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/9bff1de6a40702546875e73a0960109043f5c7e0bc212dc42e266c38f04d1a7d.jpg)

![b2f3bbadb76440f67e608738acf3bf6c94c953bec3d90120b4a18cb82af48ac4.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/b2f3bbadb76440f67e608738acf3bf6c94c953bec3d90120b4a18cb82af48ac4.jpg)

![bdbe6aa515e0e2a80a20a7c3f8864758097ae2dbd5b70c87b03e0faa441989e5.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/bdbe6aa515e0e2a80a20a7c3f8864758097ae2dbd5b70c87b03e0faa441989e5.jpg)

![ca61e8f1cf38546a9cdedb3d47101666fb07f6c72306b6b99ef7ee22ef4fb69e.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/ca61e8f1cf38546a9cdedb3d47101666fb07f6c72306b6b99ef7ee22ef4fb69e.jpg)

![e1add8954073f2b0ee784f00076e94871304bbbb0b45422de3359cdf760526ea.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/e1add8954073f2b0ee784f00076e94871304bbbb0b45422de3359cdf760526ea.jpg)

![e7c67a25ec57ea4d3eae9c64d7fdf3d26720afb07c71b66ecd0c8ef276ac0562.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/e7c67a25ec57ea4d3eae9c64d7fdf3d26720afb07c71b66ecd0c8ef276ac0562.jpg)

![f478c1770566e6cf4429b2deb66fbd7346c9c684e410c24f5a13741dfa325b76.jpg](../icml_results/6121_WeatherSyn_%20An%20Instruction%20Tuning%20MLLM%20For%20Weather%20Forecasting%20Report%20Generation/tables/f478c1770566e6cf4429b2deb66fbd7346c9c684e410c24f5a13741dfa325b76.jpg)

<a id="paper-6072"></a>

## LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding

### Images

![3f85faa1faa2f292af6d976a700ca6b6918d205d10f78441f2941d56d7a0c243.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/3f85faa1faa2f292af6d976a700ca6b6918d205d10f78441f2941d56d7a0c243.jpg)

![732ca629fa7bacba567870e2f7cc2619404b6e8b43fe9fdab6ca35c327115bd2.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/732ca629fa7bacba567870e2f7cc2619404b6e8b43fe9fdab6ca35c327115bd2.jpg)

![735e7ee0063fa9cb9e92d5ca11461067057bf115ea893b298c82ebab1c0ab27f.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/735e7ee0063fa9cb9e92d5ca11461067057bf115ea893b298c82ebab1c0ab27f.jpg)

![8a491673aa02a2fce912cda73c21a0c7e5d7aab931caa14f0eafa72868196a05.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/8a491673aa02a2fce912cda73c21a0c7e5d7aab931caa14f0eafa72868196a05.jpg)

![91ff5de6da9fb61db5ab733840936555624291c9c38934f41a10bd7f253dc237.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/91ff5de6da9fb61db5ab733840936555624291c9c38934f41a10bd7f253dc237.jpg)

![bd7947166d45da59626c531cde0b0a554cd2389b0fc1615b5f060623a33a3a25.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/bd7947166d45da59626c531cde0b0a554cd2389b0fc1615b5f060623a33a3a25.jpg)

![c2d24b7114f7184b10887d7155853f508316bf027be2998f7614187a9bfd50c7.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/c2d24b7114f7184b10887d7155853f508316bf027be2998f7614187a9bfd50c7.jpg)

![f6f1318a362219e937173c1597631a09bf6d4771c697c230a0d00ee85ba20162.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/images/f6f1318a362219e937173c1597631a09bf6d4771c697c230a0d00ee85ba20162.jpg)

### Tables

![0720e953c2f9b6854372c6ca6b44aa2ab9069ef221357ce63d6664da273e1c93.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/0720e953c2f9b6854372c6ca6b44aa2ab9069ef221357ce63d6664da273e1c93.jpg)

![5eb4a34cc7314a50087ee1f4d465f37eba38023a4ff0318ecfe7fd6a01e78263.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/5eb4a34cc7314a50087ee1f4d465f37eba38023a4ff0318ecfe7fd6a01e78263.jpg)

![93ff50d6912e13e4fd7c5c4ecdabb061045e35ce1c72cbcd78a8f2e67518d091.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/93ff50d6912e13e4fd7c5c4ecdabb061045e35ce1c72cbcd78a8f2e67518d091.jpg)

![9ec85499189e789571dd96c42971e2135512893134b26186e7248318fc1d718b.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/9ec85499189e789571dd96c42971e2135512893134b26186e7248318fc1d718b.jpg)

![a36fd8a76626cb4c982baf182e4419cd0b79a81cfd239eefda408fe579789022.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/a36fd8a76626cb4c982baf182e4419cd0b79a81cfd239eefda408fe579789022.jpg)

![a5d33931e2f62ef398df758b07562153fb3034d46f16cd5fff521739a4610a86.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/a5d33931e2f62ef398df758b07562153fb3034d46f16cd5fff521739a4610a86.jpg)

![b1d38b59ecd83cfb5cba880fe286a85825df98678c07db513c015e0e4b0d1813.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/b1d38b59ecd83cfb5cba880fe286a85825df98678c07db513c015e0e4b0d1813.jpg)

![b4a71047157ca985d57d904bab6c78fbb3a27f48af35433aad15c6c646277bed.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/b4a71047157ca985d57d904bab6c78fbb3a27f48af35433aad15c6c646277bed.jpg)

![befdb969ac96731725413cd1dc32cfd8daed250a9623b5f9c2471f3fcf806155.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/befdb969ac96731725413cd1dc32cfd8daed250a9623b5f9c2471f3fcf806155.jpg)

![da0e38a2e172d8883877d7a7793e091203ab030bdd8adbd15138763321988496.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/da0e38a2e172d8883877d7a7793e091203ab030bdd8adbd15138763321988496.jpg)

![e6dfb1e93eb3260970c25d0e4cacf2ee1321bc2d6dd09fd3867c5f6273a914d6.jpg](../icml_results/6122_LEAP_%20Zone-Aware%20MCTS%20for%20LLM%20Self-Speculative%20Decoding/tables/e6dfb1e93eb3260970c25d0e4cacf2ee1321bc2d6dd09fd3867c5f6273a914d6.jpg)
