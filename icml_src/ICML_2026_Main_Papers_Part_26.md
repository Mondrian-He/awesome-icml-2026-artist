# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 26 of 197

## 目录 (Table of Contents)

1. [Scaling depth capacity via zero/one-layer model expansion](#paper-0826)
2. [Stochastic Neural Ray Tracing for Radio Frequency Channel Modeling](#paper-0827)
3. [Grounding Multi-Hop Reasoning in Structural Causal Models via Group Relative Policy Optimization](#paper-0828)
4. [Efficient and Uncertainty-Aware Diffusion Framework for Offline-to-Online Reinforcement Learning](#paper-0829)
5. [HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel](#paper-0830)
6. [Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability](#paper-0831)
7. [Hierarchical Reinforcement Learning for Sparse-Reward Search in Commutative Algebra](#paper-0832)
8. [Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](#paper-0833)
9. [HEXST: Hexagonal Shifted-Window Transformer for Spatial Transcriptomics Gene Expression Prediction](#paper-0834)
10. [Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](#paper-0835)
11. [Robust Human-AI Complementarity under Uncertainty](#paper-0836)
12. [Deriving Neural Scaling Laws from the Statistics of Natural Language](#paper-0837)
13. [AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments](#paper-0838)
14. [From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](#paper-0839)
15. [Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization](#paper-0840)
16. [ElicitR: Unlocking Latent Reasoning in Dense Retrievers via Generative Regularization](#paper-0841)
17. [Privacy-Aware Data Integration for Enhanced Quantile Inference under Heterogeneity](#paper-0842)
18. [CONTEXTOR: Contextualized High-order Contrastive Learning](#paper-0843)
19. [Improving Classifier-Free Guidance of Flow Matching via Manifold Projection](#paper-0844)
20. [Mode Seeking meets Mean Seeking for Fast Long Video Generation](#paper-0845)
21. [Mining Tensor/Neuron-Level Sparsity to Maximize Mixture-of-Experts Potential in Post-Training and Inference](#paper-0846)
22. [Escaping the Verifier: Learning to Reason via Demonstrations](#paper-0847)
23. [Column Thresholding for Sparse Spiked Wigner Models: Improved Signal Strength Requirements](#paper-0848)
24. [FUSE: Full-spectrum Unlearnable Examples via Spectral Equalization](#paper-0849)
25. [TIMI: Training-Free Image-to-3D Multi-Instance Generation with Spatial Fidelity](#paper-0850)
26. [The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence](#paper-0851)
27. [Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality](#paper-0852)
28. [Richer Bayesian Last Layers with Subsampled NTK Features](#paper-0853)
29. [One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models](#paper-0854)
30. [Probably Approximately Correct Labels](#paper-0855)
31. [Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers](#paper-0856)
32. [Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?](#paper-0857)
33. [On the Robustness of Langevin Dynamics to Score Function Error](#paper-0858)

---


<a id="paper-0826"></a>

## Scaling depth capacity via zero/one-layer model expansion

### Images

![07910c4e6754d40d836c7c811c9f9c131fa50c620c3ce5372070aeffe6ba56e0.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/07910c4e6754d40d836c7c811c9f9c131fa50c620c3ce5372070aeffe6ba56e0.jpg)

![07ddd4b42c8bd9320c63f13c670f85c198cb3e763f1d0172c5392d46e96ee8c1.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/07ddd4b42c8bd9320c63f13c670f85c198cb3e763f1d0172c5392d46e96ee8c1.jpg)

![1a93a9d5829a0ebd9bb77364e027ab94a0cfdb305d84e8e5ce3331c194613907.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/1a93a9d5829a0ebd9bb77364e027ab94a0cfdb305d84e8e5ce3331c194613907.jpg)

![325fa815230c018f97f853d2b6f8a7afeeafb13988d0de9d8c518fcaaf64a57e.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/325fa815230c018f97f853d2b6f8a7afeeafb13988d0de9d8c518fcaaf64a57e.jpg)

![448c9ec876931ddee1b39ebcf7383631eba094df0af231c4cffa856aeec21b45.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/448c9ec876931ddee1b39ebcf7383631eba094df0af231c4cffa856aeec21b45.jpg)

![53e18af1f1fc0f18049fcd43e7099e46e27cd0d000056ad7eb93053a7a063c66.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/53e18af1f1fc0f18049fcd43e7099e46e27cd0d000056ad7eb93053a7a063c66.jpg)

![712fa091f993b2404eabd1706ede400b4cd3af4885a5ec364b1f3f1d6f451f52.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/712fa091f993b2404eabd1706ede400b4cd3af4885a5ec364b1f3f1d6f451f52.jpg)

![92ba370d137d65bf4a7eee96f456bab53c95033b64c8d6c2a944ceca3b3cb4b6.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/92ba370d137d65bf4a7eee96f456bab53c95033b64c8d6c2a944ceca3b3cb4b6.jpg)

![9498bbf2141add8d3dd3fd8b8cf2112e446699759de4a069ab017d693508752b.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/9498bbf2141add8d3dd3fd8b8cf2112e446699759de4a069ab017d693508752b.jpg)

![9a5cacd059d0330c32599524bd101ea0e4dc1ad1732996964ef15c11332077c6.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/9a5cacd059d0330c32599524bd101ea0e4dc1ad1732996964ef15c11332077c6.jpg)

![bb08aeb11d308d7ac884e19be1fb3c7aa13c16defb90fbee21e59b16019eec4c.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/bb08aeb11d308d7ac884e19be1fb3c7aa13c16defb90fbee21e59b16019eec4c.jpg)

![c2c31c614e00f31f024606942b2d74c9cba9a82740adf5217b36697544dca322.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/c2c31c614e00f31f024606942b2d74c9cba9a82740adf5217b36697544dca322.jpg)

![ca4ab5644f71c7c0379b3d49dbabbc8d3074c101087c8c3595e2a27609d3370e.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/ca4ab5644f71c7c0379b3d49dbabbc8d3074c101087c8c3595e2a27609d3370e.jpg)

![ccbaf12fb37467368acdc945fe3247c1b9813851e5c1a4e72033fc85525528df.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/ccbaf12fb37467368acdc945fe3247c1b9813851e5c1a4e72033fc85525528df.jpg)

![f95156e2c695f8431388a98e167683f3a9e9832b9c6bd72f17a665f2bc6b1879.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/images/f95156e2c695f8431388a98e167683f3a9e9832b9c6bd72f17a665f2bc6b1879.jpg)

### Tables

![21c595554833cf6614ea9241ce799f482bf22a51399c73df63433e9a511ca634.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/21c595554833cf6614ea9241ce799f482bf22a51399c73df63433e9a511ca634.jpg)

![2729d4bf89f18d254434702db53c2685459be1af96a5f48e24293c125738ef82.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/2729d4bf89f18d254434702db53c2685459be1af96a5f48e24293c125738ef82.jpg)

![6358bd3a8b2a4951a70c5528ae125aac0d0c661ea0022dcb0a9c69e3efd16eeb.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/6358bd3a8b2a4951a70c5528ae125aac0d0c661ea0022dcb0a9c69e3efd16eeb.jpg)

![7a0752c08b823f1ac6b88cf7976c307d7e4d5f6607ca63c06294c27d4f2262f2.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/7a0752c08b823f1ac6b88cf7976c307d7e4d5f6607ca63c06294c27d4f2262f2.jpg)

![8d1b1bd0a12fde17560713c50469666a85968a2e523a4b75efb50da084b1488a.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/8d1b1bd0a12fde17560713c50469666a85968a2e523a4b75efb50da084b1488a.jpg)

![e0e0c460bcfb40266f2b0eb24a887cd4fe1e480ee83195b062b6c189ca0fde46.jpg](../icml_results/840_Provable%20Sample%20Efficiency%20of%20Curriculum%20Post-Training%20for%20Transformer%20Reasoning/tables/e0e0c460bcfb40266f2b0eb24a887cd4fe1e480ee83195b062b6c189ca0fde46.jpg)

## Scaling depth capacity via zero/one-layer model expansion


### Images

![06d85f3986e792093d307dc3f189d49b9ed80bb8fa246b15d6d5cf000c026572.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/06d85f3986e792093d307dc3f189d49b9ed80bb8fa246b15d6d5cf000c026572.jpg)

![0720d4654ce4d2c64b5562aa9c5e346a59a7de1817188d580f074ed804b2df91.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/0720d4654ce4d2c64b5562aa9c5e346a59a7de1817188d580f074ed804b2df91.jpg)

![2d5c5ddf3db98abf03f693bfe5f4d63ddb2dec6208663227bfd33b77fe55e3bd.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/2d5c5ddf3db98abf03f693bfe5f4d63ddb2dec6208663227bfd33b77fe55e3bd.jpg)

![2de5290710a142e223e470eed723131478a8e07cdbc42e6843b76f9d1005887e.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/2de5290710a142e223e470eed723131478a8e07cdbc42e6843b76f9d1005887e.jpg)

![46c8f7ac5d38b77f6d8381e85cb5492c017d79d5254637ac556bac9fc2a3e4f5.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/46c8f7ac5d38b77f6d8381e85cb5492c017d79d5254637ac556bac9fc2a3e4f5.jpg)

![4a3f5d3730607315b4b00fece4f1f16b4cdc6fa0655167e8a1e4fdd19c5bd4dd.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/4a3f5d3730607315b4b00fece4f1f16b4cdc6fa0655167e8a1e4fdd19c5bd4dd.jpg)

![4f346bd10880ed15e1334ff141f14c4670e8362ea8629f4c3f3d29247918473c.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/4f346bd10880ed15e1334ff141f14c4670e8362ea8629f4c3f3d29247918473c.jpg)

![6025c8021989f3cc1611f4f86113f5dedd66eb83595f60a56ff9d986d033c1e1.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/6025c8021989f3cc1611f4f86113f5dedd66eb83595f60a56ff9d986d033c1e1.jpg)

![6c8a44a775f0d84830184b3876081d597f38108b84901d0799c92998763b113c.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/6c8a44a775f0d84830184b3876081d597f38108b84901d0799c92998763b113c.jpg)

![a2de4ba4eb44622861ad29df81df90ed9a769fa9e693f0be9868c504393441fd.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/a2de4ba4eb44622861ad29df81df90ed9a769fa9e693f0be9868c504393441fd.jpg)

![b51d7ba654eaeb02c2c9c0152b915eede46460797002b4a4f6547d0326980a4e.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/b51d7ba654eaeb02c2c9c0152b915eede46460797002b4a4f6547d0326980a4e.jpg)

![c15bcd226cfd098612bf134c63f944f1d837ac6105945f1a2565facc8aec66b1.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/c15bcd226cfd098612bf134c63f944f1d837ac6105945f1a2565facc8aec66b1.jpg)

![c558a97ac8faf60ef9192f76a16d79290820ccaeb95783e7064786414adb5552.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/c558a97ac8faf60ef9192f76a16d79290820ccaeb95783e7064786414adb5552.jpg)

![c5dd06ed7ea9bd5406e82874d492a0f2b98a9fb98b9fc4125b938baaf8802ba2.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/c5dd06ed7ea9bd5406e82874d492a0f2b98a9fb98b9fc4125b938baaf8802ba2.jpg)

![cb9355e26df0283aad1e3f389aa854a524d3b6cfeabd0d245d68c72320a14fd6.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/cb9355e26df0283aad1e3f389aa854a524d3b6cfeabd0d245d68c72320a14fd6.jpg)

![cd21a9be261782278699cb2c5782bed1e2ab0701931af9199f20a4825dfd61a2.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/cd21a9be261782278699cb2c5782bed1e2ab0701931af9199f20a4825dfd61a2.jpg)

![e86e3ebe37c1707bad5771d188600f46b8a70003cc1a573ef0eb37176efad5b9.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/e86e3ebe37c1707bad5771d188600f46b8a70003cc1a573ef0eb37176efad5b9.jpg)

![ec2adc903c809b7e5eb233aa6962779b5eebd03f46a658164ea7c1072ab4d0e1.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/ec2adc903c809b7e5eb233aa6962779b5eebd03f46a658164ea7c1072ab4d0e1.jpg)

![f718c1d589fd19d4b10c383b96425a817ed5de047d036fbe6c52862328698155.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/f718c1d589fd19d4b10c383b96425a817ed5de047d036fbe6c52862328698155.jpg)

![faa51b69409666b885c726eff38e011fd96070a95a12a18a2cbaa4396a2c108f.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/faa51b69409666b885c726eff38e011fd96070a95a12a18a2cbaa4396a2c108f.jpg)

![fc29d597ca7b35735b6d7c0d07ed58561b211103e04901dd45946d29a838c3e3.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/fc29d597ca7b35735b6d7c0d07ed58561b211103e04901dd45946d29a838c3e3.jpg)

![fea33ed9ec07dbce4ff8bf8989c3f45f462970dd10ac3755813e5a41024028d9.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/images/fea33ed9ec07dbce4ff8bf8989c3f45f462970dd10ac3755813e5a41024028d9.jpg)

### Tables

![3f566a623bf2e84fd47ee82ac70e682a847cbe5e2152387b9a335fec4f6e7d99.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/tables/3f566a623bf2e84fd47ee82ac70e682a847cbe5e2152387b9a335fec4f6e7d99.jpg)

![dc1de2509c7db0ab917b70738e0cb1883ffb2fa9d390a11e0fa47090a25733b5.jpg](../icml_results/841_Scaling%20depth%20capacity%20via%20zero_one-layer%20model%20expansion/tables/dc1de2509c7db0ab917b70738e0cb1883ffb2fa9d390a11e0fa47090a25733b5.jpg)

<a id="paper-0827"></a>

## Stochastic Neural Ray Tracing for Radio Frequency Channel Modeling


### Images

![00065ccc1d6d0c7090b29c91700bd5e8fa4abfa4b7d67a0db21497ef5097776e.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/00065ccc1d6d0c7090b29c91700bd5e8fa4abfa4b7d67a0db21497ef5097776e.jpg)

![141d1c5e3fd3a6f4773f300e09331daf2e9cf9c103c4dd1c65c56cdae0e592c1.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/141d1c5e3fd3a6f4773f300e09331daf2e9cf9c103c4dd1c65c56cdae0e592c1.jpg)

![7f5a0c369e38f27b7788e7232935f8259483baa8f7bc82a3fae0da9816d438ab.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/7f5a0c369e38f27b7788e7232935f8259483baa8f7bc82a3fae0da9816d438ab.jpg)

![c4b6ed0b85e1119943ee8c45f6fa5daf9d27f0b62401d5ce6170dd1925562c83.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/c4b6ed0b85e1119943ee8c45f6fa5daf9d27f0b62401d5ce6170dd1925562c83.jpg)

![e1c5fb5d3638ed3a05be67e23cbaed810bf734c053ac894ef1b87cd1d5f661f5.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/e1c5fb5d3638ed3a05be67e23cbaed810bf734c053ac894ef1b87cd1d5f661f5.jpg)

![f3bba90fea2478040a40b08ee0808d96e6d12c02da237c2d00853a19a2acabac.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/images/f3bba90fea2478040a40b08ee0808d96e6d12c02da237c2d00853a19a2acabac.jpg)

### Tables

![38e79c339a5bad26eaece389397415b4c86dc1d4c07aeaf390dcd60fa4e0dab5.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/tables/38e79c339a5bad26eaece389397415b4c86dc1d4c07aeaf390dcd60fa4e0dab5.jpg)

![47987db6da473009c85030b1125560d42298d8bfd4dfd00ee9d6e3a1438d6e3b.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/tables/47987db6da473009c85030b1125560d42298d8bfd4dfd00ee9d6e3a1438d6e3b.jpg)

![b8377391d357f03b26e5c35a4a213007532958bd967e89ca56d9efffa0f0f9fe.jpg](../icml_results/842_Stochastic%20Neural%20Ray%20Tracing%20for%20Radio%20Frequency%20Channel%20Modeling/tables/b8377391d357f03b26e5c35a4a213007532958bd967e89ca56d9efffa0f0f9fe.jpg)

<a id="paper-0828"></a>

## Grounding Multi-Hop Reasoning in Structural Causal Models via Group Relative Policy Optimization


### Images

![1f25a0bd195dca3162b7dd8eaca3ebd58e5cf61781000dec98b5a30a2e1cc295.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/1f25a0bd195dca3162b7dd8eaca3ebd58e5cf61781000dec98b5a30a2e1cc295.jpg)

![252514bfa15c20e5f9cc70febd5dd267767d4782cd9b150aedc5640f2f1225ae.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/252514bfa15c20e5f9cc70febd5dd267767d4782cd9b150aedc5640f2f1225ae.jpg)

![3a5d6fea6551310e60b89261f803799c31fbb581e5a141791cd0961bdb4c7cc6.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/3a5d6fea6551310e60b89261f803799c31fbb581e5a141791cd0961bdb4c7cc6.jpg)

![5dd4b9938c1a6672c0663d28bf795947b0dd7ef2ba43865b99902ee60b5da435.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/5dd4b9938c1a6672c0663d28bf795947b0dd7ef2ba43865b99902ee60b5da435.jpg)

![601bcb0a6440c3fbc18514bc8bbf4003b48b215fa1cbc48353f7ea58378c8150.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/601bcb0a6440c3fbc18514bc8bbf4003b48b215fa1cbc48353f7ea58378c8150.jpg)

![7e60b4ee37b035b25ad08b9c0186b0a49d77f639ecac60fb481c3eb9172a6b77.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/7e60b4ee37b035b25ad08b9c0186b0a49d77f639ecac60fb481c3eb9172a6b77.jpg)

![87af2d2ab671b8f6bee2a6508823242b97cedc4cf31e7c300b3ac58ab5af7344.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/87af2d2ab671b8f6bee2a6508823242b97cedc4cf31e7c300b3ac58ab5af7344.jpg)

![8e53ccae6a28d9b686c34d3e76d00877ec4691ad6c11f68e78ee2147fa5df387.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/8e53ccae6a28d9b686c34d3e76d00877ec4691ad6c11f68e78ee2147fa5df387.jpg)

![afdc58dee2229f0d2bb4a2ed2e295a49eeb574794827900b036c565186d1d14f.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/images/afdc58dee2229f0d2bb4a2ed2e295a49eeb574794827900b036c565186d1d14f.jpg)

### Tables

![136628b5b49b40f93dedc971873bb824b018db6c92ae7c4b4b82078ece0a030e.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/tables/136628b5b49b40f93dedc971873bb824b018db6c92ae7c4b4b82078ece0a030e.jpg)

![297684123b1dcd44d29de842149a67721727a3a407dffa020acfa59c4df177ba.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/tables/297684123b1dcd44d29de842149a67721727a3a407dffa020acfa59c4df177ba.jpg)

![2daa304ee87b1a1c7c3e1358e67e540a31306f142e59e742bb0b67091f0a9360.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/tables/2daa304ee87b1a1c7c3e1358e67e540a31306f142e59e742bb0b67091f0a9360.jpg)

![651079238e3f95f9aa2026b40782f07effad88c36a953b1ab38a07b00acabdbf.jpg](../icml_results/843_Grounding%20Multi-Hop%20Reasoning%20in%20Structural%20Causal%20Models%20via%20Group%20Relative%20Policy%20Optimization/tables/651079238e3f95f9aa2026b40782f07effad88c36a953b1ab38a07b00acabdbf.jpg)

<a id="paper-0829"></a>

## Efficient and Uncertainty-Aware Diffusion Framework for Offline-to-Online Reinforcement Learning


### Images

![03d22141c8faa9dfa7e0b298f2c87d7c7f1d7baa33408e3f17b87eacc8eb3108.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/03d22141c8faa9dfa7e0b298f2c87d7c7f1d7baa33408e3f17b87eacc8eb3108.jpg)

![3c88d86b095c197d8285a718171bf3cba96e42601d5e13ef2adfd2f2b34fbcf0.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/3c88d86b095c197d8285a718171bf3cba96e42601d5e13ef2adfd2f2b34fbcf0.jpg)

![5a8e21ba5370ebfff358d91581ba9dc4722230e4d7524db6d937f8f7ae80c492.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/5a8e21ba5370ebfff358d91581ba9dc4722230e4d7524db6d937f8f7ae80c492.jpg)

![5fc000d3e1e00fdf9ecb3294915d5622dc5beabea949c34d3e274d662d6b4499.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/5fc000d3e1e00fdf9ecb3294915d5622dc5beabea949c34d3e274d662d6b4499.jpg)

![63aae73604a7c1f5e8cd9b84039ca08cebd3cbcd2a2660bff5a708c8b41c57bb.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/63aae73604a7c1f5e8cd9b84039ca08cebd3cbcd2a2660bff5a708c8b41c57bb.jpg)

![7519b3ba8c2227b16ceb7546f8f01df709a5d48e9f9e82dfb4b429d2f22707ac.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/7519b3ba8c2227b16ceb7546f8f01df709a5d48e9f9e82dfb4b429d2f22707ac.jpg)

![cc5fc625b131bbd3fa90d258978f3f5eef9bc6909c9c36c3c6d9d74c1afc7f19.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/cc5fc625b131bbd3fa90d258978f3f5eef9bc6909c9c36c3c6d9d74c1afc7f19.jpg)

![d0170c1420a28eeb5a6c00bdf5b77f82922e51a2a2ed5c8263d907da48a7893d.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/d0170c1420a28eeb5a6c00bdf5b77f82922e51a2a2ed5c8263d907da48a7893d.jpg)

![e3de89de60f51334d6bf49c20b2f00852815025544b44f2aebc7804f9498fef0.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/images/e3de89de60f51334d6bf49c20b2f00852815025544b44f2aebc7804f9498fef0.jpg)

### Tables

![1af928f3124f8c34401646f2c648f3c08b925790a4eaeaf1001f9c5fed2e5f50.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/1af928f3124f8c34401646f2c648f3c08b925790a4eaeaf1001f9c5fed2e5f50.jpg)

![46d4c5d88cb9f8f39088a68ad66870a52394fe0e17cf8d105b9ed12dd565ffc3.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/46d4c5d88cb9f8f39088a68ad66870a52394fe0e17cf8d105b9ed12dd565ffc3.jpg)

![8b33b3aae4db6916bef6d974391d1a641e7c460e73ea76dd9a29406cd1928039.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/8b33b3aae4db6916bef6d974391d1a641e7c460e73ea76dd9a29406cd1928039.jpg)

![9b009885adb2dc007bd3238fd9522530fe55cc307945272ec4355f4fc5989847.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/9b009885adb2dc007bd3238fd9522530fe55cc307945272ec4355f4fc5989847.jpg)

![b87990fd2cbf1d56f10649a93994350ad79de8f167f5316e468ef497dac87cbf.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/b87990fd2cbf1d56f10649a93994350ad79de8f167f5316e468ef497dac87cbf.jpg)

![f0a354a892c04adce992ecef76f1302ab4d3bdb5ea8360ac137b69684d808b0b.jpg](../icml_results/844_Efficient%20and%20Uncertainty-Aware%20Diffusion%20Framework%20for%20Offline-to-Online%20Reinforcement%20Learning/tables/f0a354a892c04adce992ecef76f1302ab4d3bdb5ea8360ac137b69684d808b0b.jpg)

<a id="paper-0830"></a>

## HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel


### Images

![5255d905d54225b1406ed5dd8668392844fd22b1464bdb2e26b400c5492a6e3e.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/5255d905d54225b1406ed5dd8668392844fd22b1464bdb2e26b400c5492a6e3e.jpg)

![7e4d5ddf51300a44138f528c96bfdb5448f39dfc0807e28635b01b09d1d4fd13.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/7e4d5ddf51300a44138f528c96bfdb5448f39dfc0807e28635b01b09d1d4fd13.jpg)

![915660c0f9eac41505566eb50294446142d2acaa8bf32be777778184c1944b3a.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/915660c0f9eac41505566eb50294446142d2acaa8bf32be777778184c1944b3a.jpg)

![9ec4514c54bcdff25b8226428d26d0952eab9f891d8c4c19f9922bb8d13e7cc6.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/9ec4514c54bcdff25b8226428d26d0952eab9f891d8c4c19f9922bb8d13e7cc6.jpg)

![a659d4f3ea5769d998d8337dbe89d927205ad9cfbebb855b2c2175b47df83d95.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/a659d4f3ea5769d998d8337dbe89d927205ad9cfbebb855b2c2175b47df83d95.jpg)

![a968ca007a1a89803eaed52dfb354559971bb1b5ab544a5ee0c43bd3310275ec.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/a968ca007a1a89803eaed52dfb354559971bb1b5ab544a5ee0c43bd3310275ec.jpg)

![f287a6ce3efe4c5fa6ac2e22e541071943faa7e12c2b526d94093b121f35eea3.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/f287a6ce3efe4c5fa6ac2e22e541071943faa7e12c2b526d94093b121f35eea3.jpg)

![fc67f9065aa6cefdd7295e08d20eb9738bfaf8c20eaf61e8f61e23e6d2c59fc2.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/images/fc67f9065aa6cefdd7295e08d20eb9738bfaf8c20eaf61e8f61e23e6d2c59fc2.jpg)

### Tables

![0a1d0175257beab36adb2392efa70e682bd432d29078c9cdeebc08210543932d.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/0a1d0175257beab36adb2392efa70e682bd432d29078c9cdeebc08210543932d.jpg)

![0bc6fdb660539e125cbfcace83fbc89d84cb4a7fbf1d8a2815aceede51758dcd.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/0bc6fdb660539e125cbfcace83fbc89d84cb4a7fbf1d8a2815aceede51758dcd.jpg)

![0e2bd11a839734b46bad25304af40966556959348b288ee939ce03d4d60ae6d4.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/0e2bd11a839734b46bad25304af40966556959348b288ee939ce03d4d60ae6d4.jpg)

![0fa6c2db77ecdcbde430ad745b2ec7f2a2ece5d2f6bde7814e869c472ac3a75a.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/0fa6c2db77ecdcbde430ad745b2ec7f2a2ece5d2f6bde7814e869c472ac3a75a.jpg)

![17175a49ef8fd7e2c167bc7a4033327ef9a55d35e8e9b8043083a158198aa968.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/17175a49ef8fd7e2c167bc7a4033327ef9a55d35e8e9b8043083a158198aa968.jpg)

![1f3b556b7b19c3ddb5b9f235a3165c3ff8d7521e891863fe1e31df02bb3f44ef.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/1f3b556b7b19c3ddb5b9f235a3165c3ff8d7521e891863fe1e31df02bb3f44ef.jpg)

![2c4db55f8b32ab32649b3178f771cc386ca8a6bdab4ce29bb0454ca3c398cbc4.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/2c4db55f8b32ab32649b3178f771cc386ca8a6bdab4ce29bb0454ca3c398cbc4.jpg)

![2d1f7ad55dc71d2ea0825d926649924ae279937a8ce1b04eb3896c3b4c3e91f1.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/2d1f7ad55dc71d2ea0825d926649924ae279937a8ce1b04eb3896c3b4c3e91f1.jpg)

![3917ff3165cbe24d5ef470b674cdd3ad4ba5e6f637ef410c6ca150a2c5501d98.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/3917ff3165cbe24d5ef470b674cdd3ad4ba5e6f637ef410c6ca150a2c5501d98.jpg)

![3a2d8c56f0fe4d52833b7066c7ab97a7196df8f9311c6a944f6bd35c01073b01.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/3a2d8c56f0fe4d52833b7066c7ab97a7196df8f9311c6a944f6bd35c01073b01.jpg)

![48f37d5750f1fe5676a6c97c6feae065423ab2b9bc172e343a2d75766e10ef6f.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/48f37d5750f1fe5676a6c97c6feae065423ab2b9bc172e343a2d75766e10ef6f.jpg)

![49a3ec0b6a1fb72cf3a8767de0ed0d5b85a2a91920afc8d015bd73a51ab7fbd0.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/49a3ec0b6a1fb72cf3a8767de0ed0d5b85a2a91920afc8d015bd73a51ab7fbd0.jpg)

![4e41148a128c60af5324f79e20d0078681c370add0244156b5fdf3d4240bd3cf.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/4e41148a128c60af5324f79e20d0078681c370add0244156b5fdf3d4240bd3cf.jpg)

![64016090c5a5592d570ff3be3150c579a2250a2c3a8d032fce9651696214871d.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/64016090c5a5592d570ff3be3150c579a2250a2c3a8d032fce9651696214871d.jpg)

![697ca3e3e29cefaea9c73346d7b3ff3ca44263464e5bab6ef10e84827db66df5.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/697ca3e3e29cefaea9c73346d7b3ff3ca44263464e5bab6ef10e84827db66df5.jpg)

![698367bbdebbadccd16b9c8e9cd519e129c63b5198ca65a87a32771c040a6335.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/698367bbdebbadccd16b9c8e9cd519e129c63b5198ca65a87a32771c040a6335.jpg)

![6a9220c2353d1e621deeee6cd248ae1d147c42ec38425042f7abc4bd4eb2f405.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/6a9220c2353d1e621deeee6cd248ae1d147c42ec38425042f7abc4bd4eb2f405.jpg)

![7fccf20e529c42ce075d2d8f5b6f31f4bb034bc4a9d1f4e3181fc4bbf9a5faaf.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/7fccf20e529c42ce075d2d8f5b6f31f4bb034bc4a9d1f4e3181fc4bbf9a5faaf.jpg)

![824bf47aaf4fa344032cc0951ca1ab5f0270fba49affb226f8cf4ad4b573a521.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/824bf47aaf4fa344032cc0951ca1ab5f0270fba49affb226f8cf4ad4b573a521.jpg)

![82a730130ad785b60736f55b642183ff7c14bdc98bbde3960cf9106de7021373.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/82a730130ad785b60736f55b642183ff7c14bdc98bbde3960cf9106de7021373.jpg)

![8ba32559ebbc7daa9a15009341b27914eec8cd0768c871145186c0eca2ad01c5.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/8ba32559ebbc7daa9a15009341b27914eec8cd0768c871145186c0eca2ad01c5.jpg)

![a7bba05fd87dc2e9edd410826777b8a9f1c160d15b7c39f09df82dbe230e17dc.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/a7bba05fd87dc2e9edd410826777b8a9f1c160d15b7c39f09df82dbe230e17dc.jpg)

![a9b1ba57b9240c4aa9c2abc8173357130f2005f8c02b88ff2871c16de304bc45.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/a9b1ba57b9240c4aa9c2abc8173357130f2005f8c02b88ff2871c16de304bc45.jpg)

![b0abf17c2d97792093f8fbf2959c49f934fa3137b46004b450f501989490a700.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/b0abf17c2d97792093f8fbf2959c49f934fa3137b46004b450f501989490a700.jpg)

![b4938cb3ff2b01139d7b4df3df95ae0e45089b50ea99a456bd6b0ed6dbe966da.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/b4938cb3ff2b01139d7b4df3df95ae0e45089b50ea99a456bd6b0ed6dbe966da.jpg)

![c002d5796e9a9ac4ead178284283c63a3cb69ed23febc06f6fa156c90c04d169.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/c002d5796e9a9ac4ead178284283c63a3cb69ed23febc06f6fa156c90c04d169.jpg)

![c0671907b00d59ba2186868d36767ee8904b4dd30b0fa52ebdb06edce9a4daa1.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/c0671907b00d59ba2186868d36767ee8904b4dd30b0fa52ebdb06edce9a4daa1.jpg)

![c51bbd22b01fb9040402bdd32781b4a8362d2d72ddc0a360af78d8b6caebf1eb.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/c51bbd22b01fb9040402bdd32781b4a8362d2d72ddc0a360af78d8b6caebf1eb.jpg)

![cfa86e2bc358ae98da26e1158577c33de2eb306d130c29784056f79ba6414473.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/cfa86e2bc358ae98da26e1158577c33de2eb306d130c29784056f79ba6414473.jpg)

![d040795c651d7964156ad04082ece9f05bde905f540e661df75e579c9b006b2c.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/d040795c651d7964156ad04082ece9f05bde905f540e661df75e579c9b006b2c.jpg)

![d90510a9ea6773011d0e95311b50fec93b6fb365606b6ca408fc5ce64d84f223.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/d90510a9ea6773011d0e95311b50fec93b6fb365606b6ca408fc5ce64d84f223.jpg)

![e6a0652a0b42b909c563da7b82df13e4c3e126a544f8b3da341864c2e1a32aca.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/e6a0652a0b42b909c563da7b82df13e4c3e126a544f8b3da341864c2e1a32aca.jpg)

![e88dc5e95d17c0ff76de53e607f39707e21890396974c15285b83453a4032211.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/e88dc5e95d17c0ff76de53e607f39707e21890396974c15285b83453a4032211.jpg)

![ed54d6199117291d4674819a742478d1ea133052a2336114977c05fe53efd55e.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/ed54d6199117291d4674819a742478d1ea133052a2336114977c05fe53efd55e.jpg)

![eecfda9ad0ab825178a5714ba4da55cc768d24ee8e64df6701e9f54201fad1ba.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/eecfda9ad0ab825178a5714ba4da55cc768d24ee8e64df6701e9f54201fad1ba.jpg)

![efc018feab8c590b465c3cecf4f56481c7ac5edc933f929c7a6fc92e2d473c9e.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/efc018feab8c590b465c3cecf4f56481c7ac5edc933f929c7a6fc92e2d473c9e.jpg)

![fd08c530fcc9ce63a777f8061c67106a5724f02995b0bfdacbf07a248d003725.jpg](../icml_results/845_HiMAP-Travel_%20Hierarchical%20Multi-Agent%20Planning%20for%20Long-Horizon%20Constrained%20Travel/tables/fd08c530fcc9ce63a777f8061c67106a5724f02995b0bfdacbf07a248d003725.jpg)

<a id="paper-0831"></a>

## Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability


### Images

![0df09003f87ba973904fd8310d1626a10034fbaddc3c620a2daa5fffc1cf8318.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/0df09003f87ba973904fd8310d1626a10034fbaddc3c620a2daa5fffc1cf8318.jpg)

![139716c4c6920f37af012acf6b6cb13eacd598e8488fc901a77ea6c4609f0304.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/139716c4c6920f37af012acf6b6cb13eacd598e8488fc901a77ea6c4609f0304.jpg)

![1e81000a4a527d01aa4d7b33aa313cb7772187209affdea7c71ef0687be8ab6d.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/1e81000a4a527d01aa4d7b33aa313cb7772187209affdea7c71ef0687be8ab6d.jpg)

![29ef581e3991ce7d1748acd65403a5314d55f0df83d8edbde4e8ad17f2833745.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/29ef581e3991ce7d1748acd65403a5314d55f0df83d8edbde4e8ad17f2833745.jpg)

![2c8a9e9191299cb672bfa0188d1cf6a7695fce07cb750b8b81bb68e5163b4b35.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/2c8a9e9191299cb672bfa0188d1cf6a7695fce07cb750b8b81bb68e5163b4b35.jpg)

![4c4b4e041bc77c929beadec1d6d932ae70181f49fd9e35a5e1de5f0d4c5a85d6.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/4c4b4e041bc77c929beadec1d6d932ae70181f49fd9e35a5e1de5f0d4c5a85d6.jpg)

![6280040d485d92be5d8fe60aa3e79e530efe6d83101198491e6d82cc0a58fe67.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/6280040d485d92be5d8fe60aa3e79e530efe6d83101198491e6d82cc0a58fe67.jpg)

![6fea042eea9045c49dedd44b08525654875c384023f880fb6a9bb5ef3d51262c.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/6fea042eea9045c49dedd44b08525654875c384023f880fb6a9bb5ef3d51262c.jpg)

![7e85e1d6b3c47ba6198170d6625430d4623d503efaa1576d72bcb6715d43bb0c.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/7e85e1d6b3c47ba6198170d6625430d4623d503efaa1576d72bcb6715d43bb0c.jpg)

![8392615389c93b3736ba7fda1ad1753c06efcb872ab289d3a20e0b6309932419.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/8392615389c93b3736ba7fda1ad1753c06efcb872ab289d3a20e0b6309932419.jpg)

![9035fd46f7459898793187e1a4af4fbf9149c428de3db14f9d77d8dba74c6ee0.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/9035fd46f7459898793187e1a4af4fbf9149c428de3db14f9d77d8dba74c6ee0.jpg)

![cb9eca98c9c6669833ba7c956df15a40849590d7b15b65335c3e2241f7e2eb1e.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/cb9eca98c9c6669833ba7c956df15a40849590d7b15b65335c3e2241f7e2eb1e.jpg)

![cdb9406c7540d7f883e7cf814525f980b4bff1e0f57b1144443b6c73de8de567.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/cdb9406c7540d7f883e7cf814525f980b4bff1e0f57b1144443b6c73de8de567.jpg)

![ce72c85927af798f4bb4ff065bbb64726f30d543ee752b1101929f3b1d8aa842.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/ce72c85927af798f4bb4ff065bbb64726f30d543ee752b1101929f3b1d8aa842.jpg)

![d349c2841a5d0134f6f11792e05ff8046774ae54fdd09a0257434a826b318a24.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/d349c2841a5d0134f6f11792e05ff8046774ae54fdd09a0257434a826b318a24.jpg)

![d3bfb8452cc56969b9135c4b6d52a025ce328b57d1bd8051a3ccf2e1d2ab65d1.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/d3bfb8452cc56969b9135c4b6d52a025ce328b57d1bd8051a3ccf2e1d2ab65d1.jpg)

![daa5c0289f7dd6b02d0e3e449c4e601456aa3a082d17b2da52b34102a40823b4.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/daa5c0289f7dd6b02d0e3e449c4e601456aa3a082d17b2da52b34102a40823b4.jpg)

![df95834a4f9359df85fb45bdf8081be43ba75cb1d69d5b5c1fd607c4804d7259.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/df95834a4f9359df85fb45bdf8081be43ba75cb1d69d5b5c1fd607c4804d7259.jpg)

![e43d9234a1c1aa26358b5ee5454c06ffbdf43bbda9e1a0f679b689e3df694969.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/e43d9234a1c1aa26358b5ee5454c06ffbdf43bbda9e1a0f679b689e3df694969.jpg)

![e6e590435ec54f7fcbd49f078d7ed29a0080319580847069d6d12dfd89275906.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/e6e590435ec54f7fcbd49f078d7ed29a0080319580847069d6d12dfd89275906.jpg)

![ee2cd456015a659e10578a66661ad4b5b4f69fff8e0545784864a28c1b2d0872.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/images/ee2cd456015a659e10578a66661ad4b5b4f69fff8e0545784864a28c1b2d0872.jpg)

### Tables

![6157cf1fc623c747de10a9f9365d88964c3a237ec96b0e229526273972a31c7f.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/tables/6157cf1fc623c747de10a9f9365d88964c3a237ec96b0e229526273972a31c7f.jpg)

![7b37ef9c06a1b65d76a17afc0a066ad9b87ed9d74d117370a1ffa09ddb737799.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/tables/7b37ef9c06a1b65d76a17afc0a066ad9b87ed9d74d117370a1ffa09ddb737799.jpg)

![f9448ad50f392b61bd4d68cc727dcb67dd22ebbfbab7e9370620d2c16c96cf40.jpg](../icml_results/846_Beyond%20Structural%20Symmetries_%20Linear%20Mode%20Connectivity%20via%20Neuron%20Identifiability/tables/f9448ad50f392b61bd4d68cc727dcb67dd22ebbfbab7e9370620d2c16c96cf40.jpg)

<a id="paper-0832"></a>

## Hierarchical Reinforcement Learning for Sparse-Reward Search in Commutative Algebra


### Images

![04570ddf5aea66e8312d5e5b03c086b146b69b627c4d7ec1d2ea7414cfb04c62.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/04570ddf5aea66e8312d5e5b03c086b146b69b627c4d7ec1d2ea7414cfb04c62.jpg)

![1731f7cbe3afa1ff69f18fa7bbb4b781d9ca62fa3c107b49ddf3e79cdb43f597.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/1731f7cbe3afa1ff69f18fa7bbb4b781d9ca62fa3c107b49ddf3e79cdb43f597.jpg)

![1f7c545ea4ac298474f32a70474f67026b8d750ac1813f058c3f8f29ad7f8efa.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/1f7c545ea4ac298474f32a70474f67026b8d750ac1813f058c3f8f29ad7f8efa.jpg)

![2d41ff3a91a53d5c65d36688ee82e3fc6a5b4aa6bc33b007d5d72d582974754c.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/2d41ff3a91a53d5c65d36688ee82e3fc6a5b4aa6bc33b007d5d72d582974754c.jpg)

![51668bdd409bde360fbb4697dbd627815f927978cda395556ef15349bf84dabf.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/51668bdd409bde360fbb4697dbd627815f927978cda395556ef15349bf84dabf.jpg)

![52de53f013ae2edc42e42868d7741cb9a7a32c20c3ff4daa31b9bb3186d089f6.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/52de53f013ae2edc42e42868d7741cb9a7a32c20c3ff4daa31b9bb3186d089f6.jpg)

![579ceff6c355b8ba3201c6ad67055161dbd45cbb330745d3371222ae5680129d.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/579ceff6c355b8ba3201c6ad67055161dbd45cbb330745d3371222ae5680129d.jpg)

![57af80b72c073b6189417d5a3f4dcf209216adedca9cef7520b8acb537428e21.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/57af80b72c073b6189417d5a3f4dcf209216adedca9cef7520b8acb537428e21.jpg)

![7c0c317caab87509489b45f7c094c51be088a80d55ca07efe33fabc5446c1319.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/7c0c317caab87509489b45f7c094c51be088a80d55ca07efe33fabc5446c1319.jpg)

![813ff2d5ddf1d34a7b64332ad721a6c3d805dcaa137c1e07b20afeb32b3a77d8.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/813ff2d5ddf1d34a7b64332ad721a6c3d805dcaa137c1e07b20afeb32b3a77d8.jpg)

![942f7a72db3d85fbd9589571ea475345a4244187dc930d41aafb25e9efe26ef0.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/942f7a72db3d85fbd9589571ea475345a4244187dc930d41aafb25e9efe26ef0.jpg)

![a35b663a0aa86f2eda86267732f8f6188e1874ed73953c0b512e0be1862be5f9.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/a35b663a0aa86f2eda86267732f8f6188e1874ed73953c0b512e0be1862be5f9.jpg)

![bce22f8b49e581099edf0f5e36ae48d5b64704e1767507bfc448da9a49d8aef7.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/bce22f8b49e581099edf0f5e36ae48d5b64704e1767507bfc448da9a49d8aef7.jpg)

![d6b62c774e215c340072db4b8b0d06e2a740da1a3c4fcd1feb54b6956efb835b.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/d6b62c774e215c340072db4b8b0d06e2a740da1a3c4fcd1feb54b6956efb835b.jpg)

![dfdefac912132b854f658e34e4effd0735f90ac06df3543f2af047ffc4e5e49f.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/dfdefac912132b854f658e34e4effd0735f90ac06df3543f2af047ffc4e5e49f.jpg)

![f3de5799e5e8ca2c551f8d788679e084a3c2923dafd138dcac237d1f45f7d688.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/images/f3de5799e5e8ca2c551f8d788679e084a3c2923dafd138dcac237d1f45f7d688.jpg)

### Tables

![32c4772bba64df2a7a88f23ba2c34a0737d778975db81073c3c3e7f76a66c09c.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/tables/32c4772bba64df2a7a88f23ba2c34a0737d778975db81073c3c3e7f76a66c09c.jpg)

![458815ee75e894a1e6bb7dda527ebda4c6005d40c4d7d420c7fabeb6a3df647c.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/tables/458815ee75e894a1e6bb7dda527ebda4c6005d40c4d7d420c7fabeb6a3df647c.jpg)

![a0d9e66b8b672448ac67ddf058c059db89f79500bf719eec7a166ca1de84bf61.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/tables/a0d9e66b8b672448ac67ddf058c059db89f79500bf719eec7a166ca1de84bf61.jpg)

![f699f2cfbc7aac18493db5bd9e1d3f01d635178f2e368329fbb6a409ecead65f.jpg](../icml_results/848_Hierarchical%20Reinforcement%20Learning%20for%20Sparse-Reward%20Search%20in%20Commutative%20Algebra/tables/f699f2cfbc7aac18493db5bd9e1d3f01d635178f2e368329fbb6a409ecead65f.jpg)

<a id="paper-0833"></a>

## Seeing to Generalize: How Visual Data Corrects Binding Shortcuts


### Images

![01f8e16df7187c2de6ed5687a660a797506869b8c1aac7cdc68f7113b7b97381.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/01f8e16df7187c2de6ed5687a660a797506869b8c1aac7cdc68f7113b7b97381.jpg)

![0503539686dc1851684596012c3a9c38406aef3d22ace78266fadfac001b2115.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/0503539686dc1851684596012c3a9c38406aef3d22ace78266fadfac001b2115.jpg)

![09229e2bdf9053c4c2a9f141064263e53358ffa445b27411f8e16624a156e918.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/09229e2bdf9053c4c2a9f141064263e53358ffa445b27411f8e16624a156e918.jpg)

![150b6ddbd3738b1114e57212875012306ce4e63229bee82d072e91ba39abd607.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/150b6ddbd3738b1114e57212875012306ce4e63229bee82d072e91ba39abd607.jpg)

![15a995a9f8b5167a786fa5a613e2ea743e58291caabf0b74b4f947d2f3b4ba87.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/15a995a9f8b5167a786fa5a613e2ea743e58291caabf0b74b4f947d2f3b4ba87.jpg)

![2f1a89782772b4c11abaf3679a806cb1ac9ed552fb1c447d10769a9e57ccfa72.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/2f1a89782772b4c11abaf3679a806cb1ac9ed552fb1c447d10769a9e57ccfa72.jpg)

![4da60cd17bc2f0b46e226175aa5f3fa6a5ba4968e96c2b5f81db4db3652462c7.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/4da60cd17bc2f0b46e226175aa5f3fa6a5ba4968e96c2b5f81db4db3652462c7.jpg)

![5202ce7634f95ef88a2a864fc66c96bff5b7bd5b8434a4c5f10fbdcaa8b04074.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/5202ce7634f95ef88a2a864fc66c96bff5b7bd5b8434a4c5f10fbdcaa8b04074.jpg)

![569056f95af66455d04fbff5d353b6f0741610acd3b86bf06dc3543fc3304c80.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/569056f95af66455d04fbff5d353b6f0741610acd3b86bf06dc3543fc3304c80.jpg)

![6e2576cc5b632922fc5409dd2748a1c5d6abc6cd276c4880958b54be07ccb2c5.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/6e2576cc5b632922fc5409dd2748a1c5d6abc6cd276c4880958b54be07ccb2c5.jpg)

![7e8e0e9d04d620fa03a713460f32ebba519569f7dc7e5e7d109ec09a74cadadd.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/7e8e0e9d04d620fa03a713460f32ebba519569f7dc7e5e7d109ec09a74cadadd.jpg)

![c399063d9904dd83cc1d6a9d0d5c630e0f212d2fc700f89444294b6c73478a52.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/c399063d9904dd83cc1d6a9d0d5c630e0f212d2fc700f89444294b6c73478a52.jpg)

![c54664c1a77cc47265f6848fd6c438d32a3c90775fe34f0f3702997912534790.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/c54664c1a77cc47265f6848fd6c438d32a3c90775fe34f0f3702997912534790.jpg)

![c57bd89af431a14783cb9ef31b71b2639773b31c014c7a7ad607bb43e996b14b.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/c57bd89af431a14783cb9ef31b71b2639773b31c014c7a7ad607bb43e996b14b.jpg)

![c8a36819675c7ad46a21eca63d1706117ae780147a1d77ea5324c1588dd9080f.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/c8a36819675c7ad46a21eca63d1706117ae780147a1d77ea5324c1588dd9080f.jpg)

![e682e3ca29278e1b526ba9731ad18231ffd4507df636f7f11e933cf53375683c.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/e682e3ca29278e1b526ba9731ad18231ffd4507df636f7f11e933cf53375683c.jpg)

![f1ac9ebd4c620d38345b8296d0f6a3cf7a61a8e2b81874bd7e058a7db68b426e.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/images/f1ac9ebd4c620d38345b8296d0f6a3cf7a61a8e2b81874bd7e058a7db68b426e.jpg)

### Tables

![157ac1ce773963aabfb1c03530d73ba8782b72ce18eeffece29915c711b27733.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/tables/157ac1ce773963aabfb1c03530d73ba8782b72ce18eeffece29915c711b27733.jpg)

![448e9bad7bfb0549de5c8fe16d5c0ea989ed0392b2299564c8d2789df136f09b.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/tables/448e9bad7bfb0549de5c8fe16d5c0ea989ed0392b2299564c8d2789df136f09b.jpg)

![8098d81882fc916c441323bd2ca92c2f34350a847b39519acfb39e62255721a2.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/tables/8098d81882fc916c441323bd2ca92c2f34350a847b39519acfb39e62255721a2.jpg)

![b2e77adf338d81d8fe196a3d2a535835cbbee103d20d1fbd5e13d7c86e1633b7.jpg](../icml_results/849_Seeing%20to%20Generalize_%20How%20Visual%20Data%20Corrects%20Binding%20Shortcuts/tables/b2e77adf338d81d8fe196a3d2a535835cbbee103d20d1fbd5e13d7c86e1633b7.jpg)

<a id="paper-0834"></a>

## HEXST: Hexagonal Shifted-Window Transformer for Spatial Transcriptomics Gene Expression Prediction


### Images

![12a195ffa869fc16bec13a3448eed58a57d557b84e3f7556ccc3a33fee592b25.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/12a195ffa869fc16bec13a3448eed58a57d557b84e3f7556ccc3a33fee592b25.jpg)

![69739b081b67d111e62b8eaaf8604c8431863aeec3573461c44d493aedff0cee.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/69739b081b67d111e62b8eaaf8604c8431863aeec3573461c44d493aedff0cee.jpg)

![9f837e871a84f446d64698fe141b233b1314cf2a3f87c62bdabccb2ffaff2056.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/9f837e871a84f446d64698fe141b233b1314cf2a3f87c62bdabccb2ffaff2056.jpg)

![a20967f52fc6db36994c8ce1086d670e04f324d4f6decfebd2e5a5b9a0b3e26d.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/a20967f52fc6db36994c8ce1086d670e04f324d4f6decfebd2e5a5b9a0b3e26d.jpg)

![ac698dde5403ea8ad75821a2577b10f46effc720c1f085ef10771ed4aed1b1cd.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/ac698dde5403ea8ad75821a2577b10f46effc720c1f085ef10771ed4aed1b1cd.jpg)

![bee1eacd188e0f9c7eae4d5397685c7a561b37f369471a08f00bef07a6d75604.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/bee1eacd188e0f9c7eae4d5397685c7a561b37f369471a08f00bef07a6d75604.jpg)

![c0aa1b90c61f6787af84976e8b78f2fcadc76a380735c0744142b711b295dada.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/c0aa1b90c61f6787af84976e8b78f2fcadc76a380735c0744142b711b295dada.jpg)

![db8254ecc20715f386b104ce7e6d8b437d6d9cdc79b664a2d2a722a7530c06af.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/images/db8254ecc20715f386b104ce7e6d8b437d6d9cdc79b664a2d2a722a7530c06af.jpg)

### Tables

![0105f608c4b9df424afec092758aba840b46c93fd121460454ce9df87b84f52e.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/0105f608c4b9df424afec092758aba840b46c93fd121460454ce9df87b84f52e.jpg)

![03671392c891e0139662051f9d122ca647763536e8e609fa498cfa5de567f3ee.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/03671392c891e0139662051f9d122ca647763536e8e609fa498cfa5de567f3ee.jpg)

![0a08d0992207491750fa3a4b18eff49084de94035cb5026105e711a9a78975f0.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/0a08d0992207491750fa3a4b18eff49084de94035cb5026105e711a9a78975f0.jpg)

![0d65d2fdb0746c97ff8117ad8c84d1f5ac3976335beee5c820b46e6bcfae2d1f.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/0d65d2fdb0746c97ff8117ad8c84d1f5ac3976335beee5c820b46e6bcfae2d1f.jpg)

![2a91292248383c79a88b537bd6a96df9c20cd81f727e77a8667e8789fc4e1ad5.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/2a91292248383c79a88b537bd6a96df9c20cd81f727e77a8667e8789fc4e1ad5.jpg)

![2afbf53b2d7ecd81308e3985774825b62760380de4cb451068384703148cbbf7.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/2afbf53b2d7ecd81308e3985774825b62760380de4cb451068384703148cbbf7.jpg)

![2cae2702d5a8942f2014017f0b542ee1f6e14c027bcc9fddc9114ad88bcc1c68.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/2cae2702d5a8942f2014017f0b542ee1f6e14c027bcc9fddc9114ad88bcc1c68.jpg)

![2ec319ebdedb91cf8bb4d809877411c2c07d7d79624065a2fde915375aefa278.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/2ec319ebdedb91cf8bb4d809877411c2c07d7d79624065a2fde915375aefa278.jpg)

![38cef3edc42f389edd888d5d587ddff8c64755b153ff3800f181c109a4c1e8e3.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/38cef3edc42f389edd888d5d587ddff8c64755b153ff3800f181c109a4c1e8e3.jpg)

![6835fd86b216f2fb5f13439574a665edaac3c5b80c1667ddf383cfda94eb2d63.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/6835fd86b216f2fb5f13439574a665edaac3c5b80c1667ddf383cfda94eb2d63.jpg)

![6a406129b1d50f97702b4a520b7d33774dbf2aa17382fd64fd7035043d25de9b.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/6a406129b1d50f97702b4a520b7d33774dbf2aa17382fd64fd7035043d25de9b.jpg)

![75851e9c8c1924368122f260e469daf5e6598ef9cb2f5ecaac9c2e392ee8a6f5.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/75851e9c8c1924368122f260e469daf5e6598ef9cb2f5ecaac9c2e392ee8a6f5.jpg)

![950d7124f1f5adaf5a9cdd3b7ddbf0559ca79fcd3386fe45fa8f82946573c13a.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/950d7124f1f5adaf5a9cdd3b7ddbf0559ca79fcd3386fe45fa8f82946573c13a.jpg)

![9f7faebb2287ba9a5ccf728cf586b591c42a21a207c8d5c042bbc837418a727a.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/9f7faebb2287ba9a5ccf728cf586b591c42a21a207c8d5c042bbc837418a727a.jpg)

![b0b5e5e734c0ef04c765784e28b820b17609498f5811e7e309fef92456b8b274.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/b0b5e5e734c0ef04c765784e28b820b17609498f5811e7e309fef92456b8b274.jpg)

![d7e2ae8ecbd4f482cd16bc99407eb5ea9d00316401943dc3de223e3033409876.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/d7e2ae8ecbd4f482cd16bc99407eb5ea9d00316401943dc3de223e3033409876.jpg)

![d9fe5ecd1f8fee7e22782065a7d789ac254256c6894ce5c9dbfd7207f1a37065.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/d9fe5ecd1f8fee7e22782065a7d789ac254256c6894ce5c9dbfd7207f1a37065.jpg)

![dd9d5d244b33694244499eae26f0e1cb35fd57993f004ee848ed88f74abf56be.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/dd9d5d244b33694244499eae26f0e1cb35fd57993f004ee848ed88f74abf56be.jpg)

![e0693a9e4c497e537cdf7c7dbdcd07c0d1e1d4b4a5743d956e1a47adab7a76c4.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/e0693a9e4c497e537cdf7c7dbdcd07c0d1e1d4b4a5743d956e1a47adab7a76c4.jpg)

![e7ac4c14f9a980a34724da3dc3695bc02fb4b41a62ef430f33362ef1800cb81b.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/e7ac4c14f9a980a34724da3dc3695bc02fb4b41a62ef430f33362ef1800cb81b.jpg)

![f11fdfa2999f03566589d5cebd4e331e8f6380cca352f91b82004b660e9d04ac.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/f11fdfa2999f03566589d5cebd4e331e8f6380cca352f91b82004b660e9d04ac.jpg)

![f233ba414d75d57c89e094c2a332f3db8c322fc63009e0a3d808ad002d8dbf09.jpg](../icml_results/850_HEXST_%20Hexagonal%20Shifted-Window%20Transformer%20for%20Spatial%20Transcriptomics%20Gene%20Expression%20Prediction/tables/f233ba414d75d57c89e094c2a332f3db8c322fc63009e0a3d808ad002d8dbf09.jpg)

<a id="paper-0835"></a>

## Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models


### Images

![65e277c921e0a32dbeab2d5dd659def9116b22d72cb5275db29a9cf48d3b1689.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/65e277c921e0a32dbeab2d5dd659def9116b22d72cb5275db29a9cf48d3b1689.jpg)

![71cef0b3be45f50acd41077871dff3c80a07eee3c1c70a7c657f12ac46db565b.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/71cef0b3be45f50acd41077871dff3c80a07eee3c1c70a7c657f12ac46db565b.jpg)

![958712733e67127e945e1d9909be3449e1343705885c664ddbc2b6f3547b8672.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/958712733e67127e945e1d9909be3449e1343705885c664ddbc2b6f3547b8672.jpg)

![a102b3ad17e02e21ba65ca177e58b0bc447108dae5205b561df49d4354d46183.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/a102b3ad17e02e21ba65ca177e58b0bc447108dae5205b561df49d4354d46183.jpg)

![a7539c666a7b7f4201b40e40fd5d18a65284fdee9e44e909b34cf5252d2d6d3b.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/a7539c666a7b7f4201b40e40fd5d18a65284fdee9e44e909b34cf5252d2d6d3b.jpg)

![a786a23702d68643bfbc93523018c281e01b849fad4876d25c9cd925cb1228c7.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/a786a23702d68643bfbc93523018c281e01b849fad4876d25c9cd925cb1228c7.jpg)

![ba989298f3a48d842a8906fee5a4e9830f8e0a1784d296cea77e72d7a9b59a9e.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/images/ba989298f3a48d842a8906fee5a4e9830f8e0a1784d296cea77e72d7a9b59a9e.jpg)

### Tables

![023d7326e322f2ee0fc4cbf05713f0ffd5ed9977d2110089b33154134fdc3bd2.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/023d7326e322f2ee0fc4cbf05713f0ffd5ed9977d2110089b33154134fdc3bd2.jpg)

![0a76ddc701b6293288a3d0d45a88a8d76f9b7d0ec49b8b9d2aa9fb6a7cb9e626.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/0a76ddc701b6293288a3d0d45a88a8d76f9b7d0ec49b8b9d2aa9fb6a7cb9e626.jpg)

![142a56f48145d8824f92cfaef537df763786e5bc45bb7346a98e4c1163e9b447.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/142a56f48145d8824f92cfaef537df763786e5bc45bb7346a98e4c1163e9b447.jpg)

![1f4695b35f5de1266b0143fd7ffbd16dd138369e69144e8a7d8fccc2ee58806c.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/1f4695b35f5de1266b0143fd7ffbd16dd138369e69144e8a7d8fccc2ee58806c.jpg)

![2b9a7c5a902a083f35fabe49c140b8d34232ce0aa21cdb121f114a753d94c798.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/2b9a7c5a902a083f35fabe49c140b8d34232ce0aa21cdb121f114a753d94c798.jpg)

![363c962a2642ff2d046fc7fadca14447d4e81a016f547bc7353f1ccf2dae1b64.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/363c962a2642ff2d046fc7fadca14447d4e81a016f547bc7353f1ccf2dae1b64.jpg)

![434f7b303554b380c8d5e84c4d1ce222d6c3d4678e104f1187eca54764ffaacf.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/434f7b303554b380c8d5e84c4d1ce222d6c3d4678e104f1187eca54764ffaacf.jpg)

![602ae5376b269bf0482b60244fc98c96b62ebe5681b03640ee87bc6d7696fca1.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/602ae5376b269bf0482b60244fc98c96b62ebe5681b03640ee87bc6d7696fca1.jpg)

![dc2f6c02523a31c5196ee7f17d7541b24dd78d690ff03962758e048aa0b91ea0.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/dc2f6c02523a31c5196ee7f17d7541b24dd78d690ff03962758e048aa0b91ea0.jpg)

![de7c54008c7dd31bafc7bf775da285e1701d936d1d14a94a6d9d85da9be1db6f.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/de7c54008c7dd31bafc7bf775da285e1701d936d1d14a94a6d9d85da9be1db6f.jpg)

![e585d4757ea00cc4af93d59a32d540b5c6ed3e1c4b0ad5ab522f46705ffa3e28.jpg](../icml_results/851_Bounded%20Hyperbolic%20Tangent_%20A%20Stable%20and%20Efficient%20Alternative%20to%20Pre-Layer%20Normalization%20in%20Large%20L/tables/e585d4757ea00cc4af93d59a32d540b5c6ed3e1c4b0ad5ab522f46705ffa3e28.jpg)

<a id="paper-0836"></a>

## Robust Human-AI Complementarity under Uncertainty


### Images

![1f67363347eb3b1b327e6ef83a7a565fb97155c4c54bdcfb004b391fbd5d2810.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/1f67363347eb3b1b327e6ef83a7a565fb97155c4c54bdcfb004b391fbd5d2810.jpg)

![4f702cc171a9a81b1ab75b31594818299f5ea9acf5ee5641676a6a259acd6bcb.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/4f702cc171a9a81b1ab75b31594818299f5ea9acf5ee5641676a6a259acd6bcb.jpg)

![691d622fcf7511adbcd20c4c569a2a120e9c22746cd60741b8e3a4dfd81b37ec.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/691d622fcf7511adbcd20c4c569a2a120e9c22746cd60741b8e3a4dfd81b37ec.jpg)

![6d9b9e3940e30abb992a56a03a56c207a838d05b2088233d4b88562067cc2fd7.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/6d9b9e3940e30abb992a56a03a56c207a838d05b2088233d4b88562067cc2fd7.jpg)

![984755882a189575afae80ffd49ded811272b7892c012bff998e49b4bb61170c.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/984755882a189575afae80ffd49ded811272b7892c012bff998e49b4bb61170c.jpg)

![b3261992abe8c26aeebbd454a7e8a4f31da66eed5836d2b3db22d32b5a46ab2d.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/b3261992abe8c26aeebbd454a7e8a4f31da66eed5836d2b3db22d32b5a46ab2d.jpg)

![f6aa995a0177d948fd92923f84b7c04334d8b7f98d781eb14d84882af9f0e66d.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/images/f6aa995a0177d948fd92923f84b7c04334d8b7f98d781eb14d84882af9f0e66d.jpg)

### Tables

![282202148101e7e03c504b878a50bc5f83adbc26fbae5919e2cf91871b1877c5.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/tables/282202148101e7e03c504b878a50bc5f83adbc26fbae5919e2cf91871b1877c5.jpg)

![91373d70a0864ee97286bc3491601692b7b491f56c33433caba5712967b68eac.jpg](../icml_results/852_Robust%20Human-AI%20Complementarity%20under%20Uncertainty/tables/91373d70a0864ee97286bc3491601692b7b491f56c33433caba5712967b68eac.jpg)

<a id="paper-0837"></a>

## Deriving Neural Scaling Laws from the Statistics of Natural Language


### Images

![024f7fc9dbef1a5294edb4d26e08b159b71dc04dc08e1cf6ab31622a88ffea39.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/024f7fc9dbef1a5294edb4d26e08b159b71dc04dc08e1cf6ab31622a88ffea39.jpg)

![1bde2437fa7efebb68822477da94bc774f7a45d8be423da6f77282b4fb144bd0.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/1bde2437fa7efebb68822477da94bc774f7a45d8be423da6f77282b4fb144bd0.jpg)

![20f3e83106587b2a9896e848dbda65828965757d6aa2068ef14e9ed03638a976.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/20f3e83106587b2a9896e848dbda65828965757d6aa2068ef14e9ed03638a976.jpg)

![23e0e2c154e29b27279b2b0df127047b27de497d71d4bfbdd68ee3ddc9f5df30.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/23e0e2c154e29b27279b2b0df127047b27de497d71d4bfbdd68ee3ddc9f5df30.jpg)

![633ce000ee7ad7e27574970d1bcb98e52537af12c6f33de1286327ddc7d9e137.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/633ce000ee7ad7e27574970d1bcb98e52537af12c6f33de1286327ddc7d9e137.jpg)

![6d3478c62081a47f4636f7a7f1b980b590010b427f48201690fbe5005034fd25.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/6d3478c62081a47f4636f7a7f1b980b590010b427f48201690fbe5005034fd25.jpg)

![7db10b29d5faab1732b7ef69354aee26a2c128911aeb9b769d711d2560c9ac01.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/7db10b29d5faab1732b7ef69354aee26a2c128911aeb9b769d711d2560c9ac01.jpg)

![abd7cf27b63b1f10c5d78e02fcf1fd77a10888d5f04cbec974f30cfec29c54d1.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/abd7cf27b63b1f10c5d78e02fcf1fd77a10888d5f04cbec974f30cfec29c54d1.jpg)

![ae98739d0f5af79d13ecd0b36fa186892009648962bce7e8dbb5d5daf4b8d180.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/ae98739d0f5af79d13ecd0b36fa186892009648962bce7e8dbb5d5daf4b8d180.jpg)

![aeb752cfffdc48cdab3afde22e58c4721f4d198754ec3c8c6f1b94051782109b.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/aeb752cfffdc48cdab3afde22e58c4721f4d198754ec3c8c6f1b94051782109b.jpg)

![b1baa5728cad3c9ca9718ea60a05d40975be3ed5b17e054e0424642c9cac861d.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/b1baa5728cad3c9ca9718ea60a05d40975be3ed5b17e054e0424642c9cac861d.jpg)

![b5ebb99d287e7712be732d2c80fef15e6b4c1b99ea9e6cf14480458bbde5af76.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/b5ebb99d287e7712be732d2c80fef15e6b4c1b99ea9e6cf14480458bbde5af76.jpg)

![b875c91728764c9966e9ddc64a8e874f4e90149dc4c0d132481fd4cf81a8967a.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/b875c91728764c9966e9ddc64a8e874f4e90149dc4c0d132481fd4cf81a8967a.jpg)

![c90f75cc52839ddc86589079ac9e2c448bdb1b13f781d64c65806bcee6f0bbe4.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/c90f75cc52839ddc86589079ac9e2c448bdb1b13f781d64c65806bcee6f0bbe4.jpg)

![d95d7eb5bfff0f57dacc5d6fdfc5989dae5e7075c5914f2a294b19a0a9403fe7.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/d95d7eb5bfff0f57dacc5d6fdfc5989dae5e7075c5914f2a294b19a0a9403fe7.jpg)

![ea850639a0caa3f45f69d32f1e9f63861953963bf2053b207c4aca50db7de482.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/ea850639a0caa3f45f69d32f1e9f63861953963bf2053b207c4aca50db7de482.jpg)

![eaa74015f7928d39aad9cccf8dd10901fc87dddb11b6a41bf70f704e6ab7afd8.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/images/eaa74015f7928d39aad9cccf8dd10901fc87dddb11b6a41bf70f704e6ab7afd8.jpg)

### Tables

![582b05e03b28c9941c9835fc9c12f1f54daddc39990a4bb50588798afa3ce91b.jpg](../icml_results/853_Deriving%20Neural%20Scaling%20Laws%20from%20the%20Statistics%20of%20Natural%20Language/tables/582b05e03b28c9941c9835fc9c12f1f54daddc39990a4bb50588798afa3ce91b.jpg)

<a id="paper-0838"></a>

## AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments


### Images

![1a4efc3fa70497ff3bbca92da1ac55e3ae200574f00850ea0f1b7e5c1577786b.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/1a4efc3fa70497ff3bbca92da1ac55e3ae200574f00850ea0f1b7e5c1577786b.jpg)

![23cc73c99091c2b0c6f0500bd284309f24e070e16a5f1cfd2af0ea101c288a47.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/23cc73c99091c2b0c6f0500bd284309f24e070e16a5f1cfd2af0ea101c288a47.jpg)

![3ce44a076b3c8c05d634bd84805a7f214197253f84a836fc3d721a7d5a15ca3d.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/3ce44a076b3c8c05d634bd84805a7f214197253f84a836fc3d721a7d5a15ca3d.jpg)

![40827d5aa2febf4e13ca25f2a69cff2e204e3143b54c2a64df50b3c110e6243b.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/40827d5aa2febf4e13ca25f2a69cff2e204e3143b54c2a64df50b3c110e6243b.jpg)

![52fa8ea7b8308b8e502ff80e6758362aabfd1ceeadb58ad7962e8eabf4ea9e7a.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/52fa8ea7b8308b8e502ff80e6758362aabfd1ceeadb58ad7962e8eabf4ea9e7a.jpg)

![8030e7a71bd4fb29d762ee96c071e112022c24514eea87308af06a5cf3be1b57.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/8030e7a71bd4fb29d762ee96c071e112022c24514eea87308af06a5cf3be1b57.jpg)

![d4dfc997b281f1a1b0ca0324b031bef4ea1b10558f4596b00d87172664f692b0.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/images/d4dfc997b281f1a1b0ca0324b031bef4ea1b10558f4596b00d87172664f692b0.jpg)

### Tables

![0e2f8bbf8ce4f73e8565669225c0a05ba8c893678fb690841e8314b827af447a.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/0e2f8bbf8ce4f73e8565669225c0a05ba8c893678fb690841e8314b827af447a.jpg)

![2fe51f73378e3dd7e5a68c48f47e23319064389e81d97c60b0fa39efc90eb605.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/2fe51f73378e3dd7e5a68c48f47e23319064389e81d97c60b0fa39efc90eb605.jpg)

![339a415f9bdd1a8e2823ebfbe939f60830de886cc6cef2d9fb1bbfcda89d59a4.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/339a415f9bdd1a8e2823ebfbe939f60830de886cc6cef2d9fb1bbfcda89d59a4.jpg)

![375f56b338ed4a46897eceb3491e4316e774bac38ae989b940534e863db631e8.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/375f56b338ed4a46897eceb3491e4316e774bac38ae989b940534e863db631e8.jpg)

![465321edd392a5ac5ea00cf82309a4bc773dd408af0ca27a73f8c900d3140d0e.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/465321edd392a5ac5ea00cf82309a4bc773dd408af0ca27a73f8c900d3140d0e.jpg)

![57bac53c11c46a40d87b2a33414851716e34dc2bae2dad58276940a2f4478a1f.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/57bac53c11c46a40d87b2a33414851716e34dc2bae2dad58276940a2f4478a1f.jpg)

![5dd2b888ba77db23b14449528c4b17e57df7543ac03d98b5aeeb1b30c2e62eb7.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/5dd2b888ba77db23b14449528c4b17e57df7543ac03d98b5aeeb1b30c2e62eb7.jpg)

![6546b37d1dffc53611cbf1ac6a5cf857fe082af5f0b7c63fc2999037e21a32f3.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/6546b37d1dffc53611cbf1ac6a5cf857fe082af5f0b7c63fc2999037e21a32f3.jpg)

![81380b4b2d70b7557d52176fd7083abffd6ca15cc361ddb6cd77b7aaaec70c02.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/81380b4b2d70b7557d52176fd7083abffd6ca15cc361ddb6cd77b7aaaec70c02.jpg)

![841b9bd8a96f4530ff47088f15eb2cd718cb621d5228c74fa79211a797c3e2f1.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/841b9bd8a96f4530ff47088f15eb2cd718cb621d5228c74fa79211a797c3e2f1.jpg)

![8d6529ba49ca803c7bc4726106c19b9f214d9c2b9d3a9617d1bf7caccefddc55.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/8d6529ba49ca803c7bc4726106c19b9f214d9c2b9d3a9617d1bf7caccefddc55.jpg)

![9034304505289821fa68dd37c832d818c2433648b83b7a99f803e3c7c9c1116a.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/9034304505289821fa68dd37c832d818c2433648b83b7a99f803e3c7c9c1116a.jpg)

![c587512abe9ef73fe4eeb320a4705ea2922d04e3736458fd86095918106abd45.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/c587512abe9ef73fe4eeb320a4705ea2922d04e3736458fd86095918106abd45.jpg)

![de32689b61f8097f0817841c8edb43ba1b8877854ca6ad9cd6326b88296c4a1e.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/de32689b61f8097f0817841c8edb43ba1b8877854ca6ad9cd6326b88296c4a1e.jpg)

![f5e9bcf5bd65f302c1c692782613f9f6d5e1388a7d6f6f78857b86ab9864d51b.jpg](../icml_results/854_AdaMeZO_%20Adam-style%20Zeroth-Order%20Optimizer%20for%20LLM%20Fine-tuning%20Without%20Maintaining%20the%20Moments/tables/f5e9bcf5bd65f302c1c692782613f9f6d5e1388a7d6f6f78857b86ab9864d51b.jpg)

<a id="paper-0839"></a>

## From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory


### Images

![02cb3ebb8b66b808a3dcbbd7fda2a59f3521ddc2d3621d7327202617f29eb5c1.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/02cb3ebb8b66b808a3dcbbd7fda2a59f3521ddc2d3621d7327202617f29eb5c1.jpg)

![1903c5e9f50ebf80fe5cbfc407611e6cfb01050da76fbf18fca2821320837cfe.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/1903c5e9f50ebf80fe5cbfc407611e6cfb01050da76fbf18fca2821320837cfe.jpg)

![39fc25da7684122efe462d05dacde28be35fff37d141464e2f10564ba0cdb5a2.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/39fc25da7684122efe462d05dacde28be35fff37d141464e2f10564ba0cdb5a2.jpg)

![54811e57a074525cdfce29b4ea7f341baef965a00c723b3a5c72089ac60d1e9f.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/54811e57a074525cdfce29b4ea7f341baef965a00c723b3a5c72089ac60d1e9f.jpg)

![54a31fe0da3bd03e3645554f4387650b4dc8823f506ed9ce135faea6945eee0e.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/54a31fe0da3bd03e3645554f4387650b4dc8823f506ed9ce135faea6945eee0e.jpg)

![72468cc3455c60b3e6d72a081ae1926b0c87c39c93b03eea54910dca2de64cce.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/72468cc3455c60b3e6d72a081ae1926b0c87c39c93b03eea54910dca2de64cce.jpg)

![837c0ebdceb55ae5132505a955902bf76f06dda8ec603aefa0b2527d1397ef73.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/837c0ebdceb55ae5132505a955902bf76f06dda8ec603aefa0b2527d1397ef73.jpg)

![ee9f3c13cf97d08de42effbaff7c9b2f546984625f6a1462360f0785ae0f6219.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/images/ee9f3c13cf97d08de42effbaff7c9b2f546984625f6a1462360f0785ae0f6219.jpg)

### Tables

![24424d4cc4f90ffbdcf11dabaef5f442540f3491b42d305c5eb5f80806df78d8.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/24424d4cc4f90ffbdcf11dabaef5f442540f3491b42d305c5eb5f80806df78d8.jpg)

![3a8be972c09a720ef458974ea09198e92d6033b322231d4e04a8437eb6048eaf.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/3a8be972c09a720ef458974ea09198e92d6033b322231d4e04a8437eb6048eaf.jpg)

![69542db0a1dec1010be042fcb385b59c55adde0a81871045e8f466de66b780a9.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/69542db0a1dec1010be042fcb385b59c55adde0a81871045e8f466de66b780a9.jpg)

![91e0e8a4b96a68c3c1b4718f7790fb9fbed6e5db02c8b58c4bd478794f4c6627.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/91e0e8a4b96a68c3c1b4718f7790fb9fbed6e5db02c8b58c4bd478794f4c6627.jpg)

![9c72302d65a5415c8b835efec39bf9f887d76aba291ef6dba3c1a39130dcb4bd.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/9c72302d65a5415c8b835efec39bf9f887d76aba291ef6dba3c1a39130dcb4bd.jpg)

![a71c3999c25bb234f4ef2198fffd3cbc8cdce20d52aceb3e0edd6cd90d862167.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/a71c3999c25bb234f4ef2198fffd3cbc8cdce20d52aceb3e0edd6cd90d862167.jpg)

![ba0bdab97e1bb269d97907d5aa3cd120af562efa9297150d8c41d3facf3506be.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/ba0bdab97e1bb269d97907d5aa3cd120af562efa9297150d8c41d3facf3506be.jpg)

![dada36dad2367c7e0a303012d07a7893586152b436c80f39989b9c7a3cd8dd7e.jpg](../icml_results/855_From%20Player%20to%20Master_%20Enhancing%20Test-Time%20Learning%20of%20LLM%20Agents%20via%20Reinforcement%20Learning%20over%20Me/tables/dada36dad2367c7e0a303012d07a7893586152b436c80f39989b9c7a3cd8dd7e.jpg)

<a id="paper-0840"></a>

## Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization


### Images

![0013ba1475d8054d6febd8b5276cfccd36733551f5f09efc4955f0e4a378b011.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/images/0013ba1475d8054d6febd8b5276cfccd36733551f5f09efc4955f0e4a378b011.jpg)

![133aeeeb363528a1155ec1c55d58c3dc5b048f7e8e7a1c8d653024a6e26775b2.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/images/133aeeeb363528a1155ec1c55d58c3dc5b048f7e8e7a1c8d653024a6e26775b2.jpg)

![7ea7d00210227af0a5c0c78b6444b60d1c7bd37c6dfa43c7512677110ced53c1.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/images/7ea7d00210227af0a5c0c78b6444b60d1c7bd37c6dfa43c7512677110ced53c1.jpg)

### Tables

![1aba07f0ed4e0d155d67151d858176fe48e3533618f31e31034b6270256c1d9f.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/1aba07f0ed4e0d155d67151d858176fe48e3533618f31e31034b6270256c1d9f.jpg)

![2f667b1ccd0eb3c31cee42a3eedec23aef1fa2932131dcb16c829b515eca8ead.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/2f667b1ccd0eb3c31cee42a3eedec23aef1fa2932131dcb16c829b515eca8ead.jpg)

![599e69b74a1d2af04b9daf8bdd1037f74ce4a10a33bd0c3c5509f508e1cbe09d.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/599e69b74a1d2af04b9daf8bdd1037f74ce4a10a33bd0c3c5509f508e1cbe09d.jpg)

![7bbc433f173e8d63a47e76b62c6d6b1d2847ba1a576235deaefd23e7c9c8bc2f.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/7bbc433f173e8d63a47e76b62c6d6b1d2847ba1a576235deaefd23e7c9c8bc2f.jpg)

![a9651aebb83b7361a51bb21c4b72566d55a673cfe7342bdb7bd96fd1d737005b.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/a9651aebb83b7361a51bb21c4b72566d55a673cfe7342bdb7bd96fd1d737005b.jpg)

![e045a46268c50567685fe189cfe367f9e321048375833742ad87f32aea85a392.jpg](../icml_results/856_Beyond%20Euclidean%20Clipping_%20Overcoming%20Exploration%20Collapse%20in%20LLM%20RL%20via%20Riemannian%20Isometric%20Policy/tables/e045a46268c50567685fe189cfe367f9e321048375833742ad87f32aea85a392.jpg)

<a id="paper-0841"></a>

## ElicitR: Unlocking Latent Reasoning in Dense Retrievers via Generative Regularization


### Images

![21facd2c916abf46a8d8f4df49de8fd5f73dc4ebf7fabf52adc93420c922a417.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/images/21facd2c916abf46a8d8f4df49de8fd5f73dc4ebf7fabf52adc93420c922a417.jpg)

![ef8d81a4067427454c06fc4bf7157a9ce269057055ab276981091534d780c9a5.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/images/ef8d81a4067427454c06fc4bf7157a9ce269057055ab276981091534d780c9a5.jpg)

![f193e56274464baf7a140cad01ab6234b82a324f407edc672524c378621b4021.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/images/f193e56274464baf7a140cad01ab6234b82a324f407edc672524c378621b4021.jpg)

### Tables

![03b55f5ab7c299f51f047718a4d858f398d92db3c7bfda776f77e703d13230c6.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/03b55f5ab7c299f51f047718a4d858f398d92db3c7bfda776f77e703d13230c6.jpg)

![08fcc7d5830f5c3ea8098b9911f7f28f6418dc32a1daa64658bdaa52a8d3602a.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/08fcc7d5830f5c3ea8098b9911f7f28f6418dc32a1daa64658bdaa52a8d3602a.jpg)

![12a3d1c0686fce26c45921d6b0f1c1c276be49e2e2dfc650706cefa5d2c0dd3c.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/12a3d1c0686fce26c45921d6b0f1c1c276be49e2e2dfc650706cefa5d2c0dd3c.jpg)

![1e7846fae376d153a5aa925e86fc7dc1d4c7e56239205d2d76ac63c4411e5056.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/1e7846fae376d153a5aa925e86fc7dc1d4c7e56239205d2d76ac63c4411e5056.jpg)

![2d1f42c41b0cfebd570071a7588d56a73ff43dff8f6596a104f741a2d009888d.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/2d1f42c41b0cfebd570071a7588d56a73ff43dff8f6596a104f741a2d009888d.jpg)

![35625c45f281ba29e333e49fd28eb858891eb74da6f35d12c78efb240779e92f.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/35625c45f281ba29e333e49fd28eb858891eb74da6f35d12c78efb240779e92f.jpg)

![4e14209c31b45077e8eac5ccd4bd241913d99d1fa75945ff7d0df0cfcc5da262.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/4e14209c31b45077e8eac5ccd4bd241913d99d1fa75945ff7d0df0cfcc5da262.jpg)

![51a25ee3f732c990d1619dc0e0a6d2d6108ccd0a990743311a587c72416e3ed2.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/51a25ee3f732c990d1619dc0e0a6d2d6108ccd0a990743311a587c72416e3ed2.jpg)

![5aa4e693288f4ace21419b823531046baed04ce6ba1d18a0465c07a1df78f26b.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/5aa4e693288f4ace21419b823531046baed04ce6ba1d18a0465c07a1df78f26b.jpg)

![7cac9e1202010e5f00fd1dcd8c13cca4fb70e223e333087229097d9dc4ec4417.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/7cac9e1202010e5f00fd1dcd8c13cca4fb70e223e333087229097d9dc4ec4417.jpg)

![84531ff00c4ee91c7bbe39bf4768eea59c842fb3fd4d28905893c8167885ff86.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/84531ff00c4ee91c7bbe39bf4768eea59c842fb3fd4d28905893c8167885ff86.jpg)

![89b9cb23210e8e9857e7b2cb03adb0058927a0b0b55b5ac6d0be2a01730a60af.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/89b9cb23210e8e9857e7b2cb03adb0058927a0b0b55b5ac6d0be2a01730a60af.jpg)

![c10c3659c20ea9d5578a6a900d4b99c6cbd6a414eca1a739b49afb7a8bb300c3.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/c10c3659c20ea9d5578a6a900d4b99c6cbd6a414eca1a739b49afb7a8bb300c3.jpg)

![c37167c80c9278404c604c03726a452a09bac3117102023ff6fe04bd53f26cb1.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/c37167c80c9278404c604c03726a452a09bac3117102023ff6fe04bd53f26cb1.jpg)

![cd9da59d9acf2e9e28857c131cc1c03652f9a11a09d8e23151e99f0d2b765b79.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/cd9da59d9acf2e9e28857c131cc1c03652f9a11a09d8e23151e99f0d2b765b79.jpg)

![d31c703466d0f96a228818ed135c28350bcac18740c59de77cd1afc331ec4019.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/d31c703466d0f96a228818ed135c28350bcac18740c59de77cd1afc331ec4019.jpg)

![d44a9360b159484e7cf2f9c7a60475b81d144d2cbd5dea9fa3ae31f21e2d5521.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/d44a9360b159484e7cf2f9c7a60475b81d144d2cbd5dea9fa3ae31f21e2d5521.jpg)

![f1ed5f3f9ed2728d868213170442abae4d17ed55153265cf4a1df98a2e511a90.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/f1ed5f3f9ed2728d868213170442abae4d17ed55153265cf4a1df98a2e511a90.jpg)

![fa8bd155aa1477b6de2b730ded5da1c6deabbfc1e8c906b327cacd9bb3530a2b.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/fa8bd155aa1477b6de2b730ded5da1c6deabbfc1e8c906b327cacd9bb3530a2b.jpg)

![fe322772a8e3b6cca0efeb54c59ff36d80c5aa6087b7bb3a24db386adc83bdf5.jpg](../icml_results/857_ElicitR_%20Unlocking%20Latent%20Reasoning%20in%20Dense%20Retrievers%20via%20Generative%20Regularization/tables/fe322772a8e3b6cca0efeb54c59ff36d80c5aa6087b7bb3a24db386adc83bdf5.jpg)

<a id="paper-0842"></a>

## Privacy-Aware Data Integration for Enhanced Quantile Inference under Heterogeneity


### Images

![1e413efa566f5b1cd482f4f94d01523b47bc751324340e5cac4e0fad20e5ef68.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/1e413efa566f5b1cd482f4f94d01523b47bc751324340e5cac4e0fad20e5ef68.jpg)

![23bb6137cb87828c1a8af743b15b282699230d87b96a79f0b7fa1ae73c2dc655.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/23bb6137cb87828c1a8af743b15b282699230d87b96a79f0b7fa1ae73c2dc655.jpg)

![38c33a5b8cda6b743620466ae89c3c79ce50e31a3f60cab4475d2d2ac17f6711.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/38c33a5b8cda6b743620466ae89c3c79ce50e31a3f60cab4475d2d2ac17f6711.jpg)

![64ef4a71be156942e33b18750799840a4099bd24c7bb304d87dbd7d2ba070dfb.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/64ef4a71be156942e33b18750799840a4099bd24c7bb304d87dbd7d2ba070dfb.jpg)

![6e3b14d9bbe45de6eaf5621553df1c8b37ce4877a9ddbd31ea55a28d74e01ea2.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/6e3b14d9bbe45de6eaf5621553df1c8b37ce4877a9ddbd31ea55a28d74e01ea2.jpg)

![77f3551612ab9fae4e2a8d7778f46d4fb2a72a3290e6bdb83515d88a46bd4aec.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/77f3551612ab9fae4e2a8d7778f46d4fb2a72a3290e6bdb83515d88a46bd4aec.jpg)

![77ff34fa86afdccd54127842c941e8b95ad5c1ce58be892df42923d05e631cc9.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/77ff34fa86afdccd54127842c941e8b95ad5c1ce58be892df42923d05e631cc9.jpg)

![79f2421d5fe5929d67a386e554b18a2e971bcda9ee274bfbbc9edd8ee9b800af.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/79f2421d5fe5929d67a386e554b18a2e971bcda9ee274bfbbc9edd8ee9b800af.jpg)

![7ea78127443c02daca2ca28389d8a433da2cb0e87919a986f79092cc210eaa4a.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/7ea78127443c02daca2ca28389d8a433da2cb0e87919a986f79092cc210eaa4a.jpg)

![8605079baea63f67f8cf78319cba1f1667cdc7d3aef34f40f8dcb37124d425cb.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/8605079baea63f67f8cf78319cba1f1667cdc7d3aef34f40f8dcb37124d425cb.jpg)

![9429d1c37778d1ae2cc3f186e28de30bdbc8c59787b0f788ff083cc4418608d4.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/9429d1c37778d1ae2cc3f186e28de30bdbc8c59787b0f788ff083cc4418608d4.jpg)

![a89b3820fd3a4a2c9146d2a7d96222a8900ae211bd9e864802216d2fdfea8248.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/a89b3820fd3a4a2c9146d2a7d96222a8900ae211bd9e864802216d2fdfea8248.jpg)

![cc85b98f95d3db8929e6a6883500610e32e729ea9fbc8de36bf369e13d0fc39f.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/cc85b98f95d3db8929e6a6883500610e32e729ea9fbc8de36bf369e13d0fc39f.jpg)

![da6a667397408328251280bdb14b32604f434e6fd838a5ebd6a94d10c1c4f3e4.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/da6a667397408328251280bdb14b32604f434e6fd838a5ebd6a94d10c1c4f3e4.jpg)

![fb0b609b5744a60b49d7e5beebdb764dc110a041ef43b6ad5684d9c548dd9fbf.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/fb0b609b5744a60b49d7e5beebdb764dc110a041ef43b6ad5684d9c548dd9fbf.jpg)

![fbc441176acd91f8b2a34e7f0e2ba78085867e3fedca7737d7a519d64f9ac6cf.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/images/fbc441176acd91f8b2a34e7f0e2ba78085867e3fedca7737d7a519d64f9ac6cf.jpg)

### Tables

![46ae106e422fe93919de078e11391ac2f8c5be2b057d7c199e5c4658ca840145.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/tables/46ae106e422fe93919de078e11391ac2f8c5be2b057d7c199e5c4658ca840145.jpg)

![a6ade8566b898e649479cd41dfb1ff4fb6e2bc4b3f50b134fbdd15af5908b316.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/tables/a6ade8566b898e649479cd41dfb1ff4fb6e2bc4b3f50b134fbdd15af5908b316.jpg)

![ba6fc520df65083e410754536c7f7c97ea689e3ca270fb58f2047d804dc12701.jpg](../icml_results/858_Privacy-Aware%20Data%20Integration%20for%20Enhanced%20Quantile%20Inference%20under%20Heterogeneity/tables/ba6fc520df65083e410754536c7f7c97ea689e3ca270fb58f2047d804dc12701.jpg)

<a id="paper-0843"></a>

## CONTEXTOR: Contextualized High-order Contrastive Learning


### Images

![087c0f754e55172a1e12d4feacfd709b458e358eb152decf0a68265062d5657e.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/087c0f754e55172a1e12d4feacfd709b458e358eb152decf0a68265062d5657e.jpg)

![2595a057872f4ac10f1d83f1231107d973034723687e6e3687db62316f697dcc.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/2595a057872f4ac10f1d83f1231107d973034723687e6e3687db62316f697dcc.jpg)

![32a94c8811c43666ab869dcfeba6f1b0663b44fb9d4eb14a57bc1643bdeb4e47.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/32a94c8811c43666ab869dcfeba6f1b0663b44fb9d4eb14a57bc1643bdeb4e47.jpg)

![ada6ff7b795fe3ee324cfc0f5a00f148af9acca7dd40ef18d31ac50857353ae5.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/ada6ff7b795fe3ee324cfc0f5a00f148af9acca7dd40ef18d31ac50857353ae5.jpg)

![ae6779c5e4b23d4f9b76ec7c83f2a763be7edb9275a2183a81c6e85586b0398a.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/ae6779c5e4b23d4f9b76ec7c83f2a763be7edb9275a2183a81c6e85586b0398a.jpg)

![ae7a98dee1ac55077869dc2df0a65e10b24e30451773774b089983657957eadf.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/ae7a98dee1ac55077869dc2df0a65e10b24e30451773774b089983657957eadf.jpg)

![c1688b5b12d6a88d0b056ab8ff24d272d04a7d9650995b311edc7b6b9d0eccde.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/c1688b5b12d6a88d0b056ab8ff24d272d04a7d9650995b311edc7b6b9d0eccde.jpg)

![d143a55fed401d922d52965a2b886026ee308cc465cfc55bac97b0edb06094e0.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/d143a55fed401d922d52965a2b886026ee308cc465cfc55bac97b0edb06094e0.jpg)

![ebab7c7e174d83a025570504dd7e46140a4d823f252ec57c04fb08f0f2388e49.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/images/ebab7c7e174d83a025570504dd7e46140a4d823f252ec57c04fb08f0f2388e49.jpg)

### Tables

![75429feadfdb4770dc293adeb8f51a8640fec6f971963fbcea5005dea433a565.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/75429feadfdb4770dc293adeb8f51a8640fec6f971963fbcea5005dea433a565.jpg)

![8c64871304b97890bf841baabf2604472a298f288f1caaaeb0128b5f70a6fa11.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/8c64871304b97890bf841baabf2604472a298f288f1caaaeb0128b5f70a6fa11.jpg)

![8ee9a18457a7b62baa1674d27156a96eaea5b5044cf20c2352ed5f0ad164a82d.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/8ee9a18457a7b62baa1674d27156a96eaea5b5044cf20c2352ed5f0ad164a82d.jpg)

![9ea4b62586d457d76c1c80ea657a091523245547767f2eb3479d6533371e0527.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/9ea4b62586d457d76c1c80ea657a091523245547767f2eb3479d6533371e0527.jpg)

![9fc3516b2b22965a1e12d93dffcc66137c0ce6ec1fb764624db7e24272b08824.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/9fc3516b2b22965a1e12d93dffcc66137c0ce6ec1fb764624db7e24272b08824.jpg)

![bbbed5871f9fceacd79c3589bd74e221381c45d23f7f1f16f5eb074de5994938.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/bbbed5871f9fceacd79c3589bd74e221381c45d23f7f1f16f5eb074de5994938.jpg)

![bbda9388204e78f7c7e32906192188342ad3aa395f95342e3a6d5acba213ba62.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/bbda9388204e78f7c7e32906192188342ad3aa395f95342e3a6d5acba213ba62.jpg)

![c2c0338398c2423202f189f936470f8802052b37fa9fd6a70adaa285de7838d6.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/c2c0338398c2423202f189f936470f8802052b37fa9fd6a70adaa285de7838d6.jpg)

![e0b9dc55060a4c62d9aafd9cd6a174043c268371b66705ced7d7b0ba88a9f535.jpg](../icml_results/859_CONTEXTOR_%20Contextualized%20High-order%20Contrastive%20Learning/tables/e0b9dc55060a4c62d9aafd9cd6a174043c268371b66705ced7d7b0ba88a9f535.jpg)

<a id="paper-0844"></a>

## Improving Classifier-Free Guidance of Flow Matching via Manifold Projection


### Images

![0201aff70f1b89012b377099df9ccd88c694579bd005d2484064884ea1eb6a4b.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/0201aff70f1b89012b377099df9ccd88c694579bd005d2484064884ea1eb6a4b.jpg)

![0d94995255a3338f72234f9b72e5a92ef32e766dec1186d2dcbe1d4669e25eca.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/0d94995255a3338f72234f9b72e5a92ef32e766dec1186d2dcbe1d4669e25eca.jpg)

![1e0aefd89983a4a81df766ff1037db87d9cfcc481c0971b291c9d0e6472f50bc.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/1e0aefd89983a4a81df766ff1037db87d9cfcc481c0971b291c9d0e6472f50bc.jpg)

![2d6b4b1ae7ae34e0b35c6a01a46212a2060080f536ba6b4b8bcefa3c03fc3e12.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/2d6b4b1ae7ae34e0b35c6a01a46212a2060080f536ba6b4b8bcefa3c03fc3e12.jpg)

![2f2e2964cde885530a42476958fb0ed13add5bbc6c33c5564058b44053575ab0.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/2f2e2964cde885530a42476958fb0ed13add5bbc6c33c5564058b44053575ab0.jpg)

![311b8e1277d3853b326b02146e8f0e51434a3d4d0640dea810abb9150fa09439.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/311b8e1277d3853b326b02146e8f0e51434a3d4d0640dea810abb9150fa09439.jpg)

![319d8e5e34c667d60162d4b9c23fc8e37114ba7ce5abb6116a22c0ccad575cda.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/319d8e5e34c667d60162d4b9c23fc8e37114ba7ce5abb6116a22c0ccad575cda.jpg)

![500573658de8d9634d449dc3faeeeb169b03ef807d2f4332a10ee1b77a0f8863.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/500573658de8d9634d449dc3faeeeb169b03ef807d2f4332a10ee1b77a0f8863.jpg)

![540faace9ad3d17d234f3ba8a82fa0dba84fdd29a0f2323f0349bf10e153f600.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/540faace9ad3d17d234f3ba8a82fa0dba84fdd29a0f2323f0349bf10e153f600.jpg)

![58d353b84f9bea32791e530a26375cf29fc0a8dc7cee38955d04520a88c3ac57.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/58d353b84f9bea32791e530a26375cf29fc0a8dc7cee38955d04520a88c3ac57.jpg)

![85acb7c739f15e8f087f5bcf99d4eb6d9138e9782328368fda6e04a92c4afdb6.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/85acb7c739f15e8f087f5bcf99d4eb6d9138e9782328368fda6e04a92c4afdb6.jpg)

![a30dbfc29f30458ed62aad5fdb68ea0f67baf8ff4faabbfbe2026ab9fb7c3f50.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/a30dbfc29f30458ed62aad5fdb68ea0f67baf8ff4faabbfbe2026ab9fb7c3f50.jpg)

![a627f18ca92ae676199c67852ba937b8a8f1771ebc6849186ebc3a3c943480aa.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/a627f18ca92ae676199c67852ba937b8a8f1771ebc6849186ebc3a3c943480aa.jpg)

![c0b0ab3a0e5b74a1ef615b182d272643a98f153cc4fb23def40435e09701d97b.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/c0b0ab3a0e5b74a1ef615b182d272643a98f153cc4fb23def40435e09701d97b.jpg)

![c7c8d4bd4a82c6275b6fe4141f899fbc2bcb01cc004b0cee8d8bce6ed60a9088.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/c7c8d4bd4a82c6275b6fe4141f899fbc2bcb01cc004b0cee8d8bce6ed60a9088.jpg)

![ca66f39274821c43f91eb7cf8f2d9a90bcb40477541eb71d398814b3833bf733.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/ca66f39274821c43f91eb7cf8f2d9a90bcb40477541eb71d398814b3833bf733.jpg)

![d981b6cb0d9bf77ded66e9f46a60dbd89678a8a9e906cfdc49299b8bac558c06.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/d981b6cb0d9bf77ded66e9f46a60dbd89678a8a9e906cfdc49299b8bac558c06.jpg)

![f12836305dce0749f4eb80fc1cfda6f9fc729d04bad1abba7b7de3451c7b08a6.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/images/f12836305dce0749f4eb80fc1cfda6f9fc729d04bad1abba7b7de3451c7b08a6.jpg)

### Tables

![0476465e59ce54240f3139c2a18026b357f39e96dee7cc716f503934e0e9a475.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/0476465e59ce54240f3139c2a18026b357f39e96dee7cc716f503934e0e9a475.jpg)

![16c1633f7c24eee8bcd81c6ae8e8142ed3c730963f7c4c7a76369c706660f9c0.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/16c1633f7c24eee8bcd81c6ae8e8142ed3c730963f7c4c7a76369c706660f9c0.jpg)

![27f690c434fb4f0e4cc6cb3164398dab219fcaf6bdde85a7010135c3774d7803.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/27f690c434fb4f0e4cc6cb3164398dab219fcaf6bdde85a7010135c3774d7803.jpg)

![3c329c1b4920cee00711e9f969bf713d5ea79d4ddabd222f24cb6d3d4fc04eee.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/3c329c1b4920cee00711e9f969bf713d5ea79d4ddabd222f24cb6d3d4fc04eee.jpg)

![45a219999d5822fc617d0d36267d833ae37cd238939bafd833f4d4e9440f5a28.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/45a219999d5822fc617d0d36267d833ae37cd238939bafd833f4d4e9440f5a28.jpg)

![def5cbfb1e404ef4e57ec5bcc5da7ed71b53e0ea4415edbc12646f27d1a55a38.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/def5cbfb1e404ef4e57ec5bcc5da7ed71b53e0ea4415edbc12646f27d1a55a38.jpg)

![e0822e9b58c6872610ac8362f2f83fdfd21142655e687bf5a76007dbdae5d4a2.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/e0822e9b58c6872610ac8362f2f83fdfd21142655e687bf5a76007dbdae5d4a2.jpg)

![e4592a26e80412709845a1bfe9853cf4b3c8c8b4a4bb678423fba447563d780c.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/e4592a26e80412709845a1bfe9853cf4b3c8c8b4a4bb678423fba447563d780c.jpg)

![e78d44ec4d68e7efb2419142a388d5527c2f891f42c862d15f7e4de4b92fce5f.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/e78d44ec4d68e7efb2419142a388d5527c2f891f42c862d15f7e4de4b92fce5f.jpg)

![f146b78b3b71faa88955642df3b2eff96f0df70b70a98e8949bf06b605754d69.jpg](../icml_results/860_Improving%20Classifier-Free%20Guidance%20of%20Flow%20Matching%20via%20Manifold%20Projection/tables/f146b78b3b71faa88955642df3b2eff96f0df70b70a98e8949bf06b605754d69.jpg)

<a id="paper-0845"></a>

## Mode Seeking meets Mean Seeking for Fast Long Video Generation


### Images

![3aab03a9cea9128ef21a10170a3c205a82e54474a3a08802735da7f1928b1a56.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/images/3aab03a9cea9128ef21a10170a3c205a82e54474a3a08802735da7f1928b1a56.jpg)

![48ce49c04059c5438dc1ee1f21f209939d5681c1197227f7cbcccc3b2f9bbd93.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/images/48ce49c04059c5438dc1ee1f21f209939d5681c1197227f7cbcccc3b2f9bbd93.jpg)

![56328a678ece191a4fddedc093e24b06b2d87c28c64eba2d5dd40a268260d7f7.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/images/56328a678ece191a4fddedc093e24b06b2d87c28c64eba2d5dd40a268260d7f7.jpg)

![aa4b0927cfc0de3e48c5c26467f9aec2ff3654b6959a3a896e908e8c589c2281.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/images/aa4b0927cfc0de3e48c5c26467f9aec2ff3654b6959a3a896e908e8c589c2281.jpg)

![c0c00121a7755d87506da9a4f04fcc8591f78e8e90c120468c667bcd3ce6ee02.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/images/c0c00121a7755d87506da9a4f04fcc8591f78e8e90c120468c667bcd3ce6ee02.jpg)

### Tables

![5755501d07afb5bc95dcfc45e3d73f7add12bf55492a003f8d214d8fe40c6f6b.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/tables/5755501d07afb5bc95dcfc45e3d73f7add12bf55492a003f8d214d8fe40c6f6b.jpg)

![917bfef8517b291ade7924ee4b0e0c25596f362371a830ee970297cd770d6356.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/tables/917bfef8517b291ade7924ee4b0e0c25596f362371a830ee970297cd770d6356.jpg)

![e286997d113563032ad23b042aa85da1f3cdf28dffb2da1abb5b7de0df467533.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/tables/e286997d113563032ad23b042aa85da1f3cdf28dffb2da1abb5b7de0df467533.jpg)

![ec720f33402be6d31f178a7391ab0f6be410ea72523217ca50e66b91601cae68.jpg](../icml_results/861_Mode%20Seeking%20meets%20Mean%20Seeking%20for%20Fast%20Long%20Video%20Generation/tables/ec720f33402be6d31f178a7391ab0f6be410ea72523217ca50e66b91601cae68.jpg)

<a id="paper-0846"></a>

## Mining Tensor/Neuron-Level Sparsity to Maximize Mixture-of-Experts Potential in Post-Training and Inference


### Images

![3030cf05a71ff78d3aa4b98a2999a7d9be25678442a2dca8e0bfd166db62994f.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/3030cf05a71ff78d3aa4b98a2999a7d9be25678442a2dca8e0bfd166db62994f.jpg)

![38292cbd5a10fe2cae1513aa295553bd307dde8639260509956652ad2e60b1d7.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/38292cbd5a10fe2cae1513aa295553bd307dde8639260509956652ad2e60b1d7.jpg)

![619f47701b3c64dbb18ff93fef96a212889df3aaf4560d58c7660125b9250d8b.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/619f47701b3c64dbb18ff93fef96a212889df3aaf4560d58c7660125b9250d8b.jpg)

![649563e30814c629a1b1ad6c611694596d860fefe1c6168b46df9c364476dd68.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/649563e30814c629a1b1ad6c611694596d860fefe1c6168b46df9c364476dd68.jpg)

![6949564812165da0680d0d1c331b30dedeaf3916742ba81b1da11fede0c62310.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/6949564812165da0680d0d1c331b30dedeaf3916742ba81b1da11fede0c62310.jpg)

![90ee1576262c5b71ca0978fc195beaeb691db076ea6e54af5cadea1ca44d4f0a.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/90ee1576262c5b71ca0978fc195beaeb691db076ea6e54af5cadea1ca44d4f0a.jpg)

![9eddeed513a19575c65d0dfb328478c103234b95774cc7177fe6de5bcc63f68c.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/9eddeed513a19575c65d0dfb328478c103234b95774cc7177fe6de5bcc63f68c.jpg)

![a5ba76141cbc1290c7287fb7046b2efa6d8eee0d3415a841a0f6ec9bf509a594.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/a5ba76141cbc1290c7287fb7046b2efa6d8eee0d3415a841a0f6ec9bf509a594.jpg)

![aa898c3508d8d10f594de5441088b735256897e6d08326555130f674f88b9fa0.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/aa898c3508d8d10f594de5441088b735256897e6d08326555130f674f88b9fa0.jpg)

![b31f79e8f7ee9f6c4ae4202e387dee2f3024e2afed350d97ca5f312afc846587.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/b31f79e8f7ee9f6c4ae4202e387dee2f3024e2afed350d97ca5f312afc846587.jpg)

![ca8939721fa5704c7403c5914e0d0ef160991c573709f228f522698e981c0ac4.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/ca8939721fa5704c7403c5914e0d0ef160991c573709f228f522698e981c0ac4.jpg)

![e57aa8546301d2a34de86ae026cb42b55479764728386f84adf219e69c0f761e.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/e57aa8546301d2a34de86ae026cb42b55479764728386f84adf219e69c0f761e.jpg)

![f6e99217db3de1700d62f77ee9005c90e8f988e43b959497dce034633a0bbfe0.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/images/f6e99217db3de1700d62f77ee9005c90e8f988e43b959497dce034633a0bbfe0.jpg)

### Tables

![0280bc47be2eb874eb27ec76aaee8a0c2ebf19b7a15f21a813d283a2baafe89b.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/tables/0280bc47be2eb874eb27ec76aaee8a0c2ebf19b7a15f21a813d283a2baafe89b.jpg)

![8915a6e82f6d4a8722540fdbbbfa1b546f5c94e5912cbd9603d13df4df1a35da.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/tables/8915a6e82f6d4a8722540fdbbbfa1b546f5c94e5912cbd9603d13df4df1a35da.jpg)

![c6103df9cef17812b18ca772c2ab685f7fd56715391ee374d468fb99e311c664.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/tables/c6103df9cef17812b18ca772c2ab685f7fd56715391ee374d468fb99e311c664.jpg)

![e3228b0889bf9f8bb80a5e17dd51c2f93754cc8685f3789986d84caed3a3eea1.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/tables/e3228b0889bf9f8bb80a5e17dd51c2f93754cc8685f3789986d84caed3a3eea1.jpg)

![f31e9bfaf44a62f1827b1e272c4ac72ff2387cfb57dbabdcbc1ae5281e56bb26.jpg](../icml_results/862_Mining%20Tensor_Neuron-Level%20Sparsity%20to%20Maximize%20Mixture-of-Experts%20Potential%20in%20Post-Training%20and%20In/tables/f31e9bfaf44a62f1827b1e272c4ac72ff2387cfb57dbabdcbc1ae5281e56bb26.jpg)

<a id="paper-0847"></a>

## Escaping the Verifier: Learning to Reason via Demonstrations


### Images

![00c41c2d003b7e760999dd164608cdb93eea350a0615783eae9c4e995e2cf304.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/00c41c2d003b7e760999dd164608cdb93eea350a0615783eae9c4e995e2cf304.jpg)

![08d8052aac3d7dcee8ff070c78fbe2bf76e4846b450e35d03bf4aad2f5804c05.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/08d8052aac3d7dcee8ff070c78fbe2bf76e4846b450e35d03bf4aad2f5804c05.jpg)

![101e496f4b5190f364098c5bb46a9c9123e927bcf5fcd6f35f13715e46c32d11.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/101e496f4b5190f364098c5bb46a9c9123e927bcf5fcd6f35f13715e46c32d11.jpg)

![280c01ca8e8fdfb500a0e9a267ec5b04d5f681ac3009f80ef206d6262fe62449.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/280c01ca8e8fdfb500a0e9a267ec5b04d5f681ac3009f80ef206d6262fe62449.jpg)

![3578897b39070cf1721a8e5d7c489b992530afbae518acfb0737f59a4494aa40.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/3578897b39070cf1721a8e5d7c489b992530afbae518acfb0737f59a4494aa40.jpg)

![3ce33debbc3130a372ff1524b91dee4369d49ac9bb9df21d8f66f55047ebf5b3.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/3ce33debbc3130a372ff1524b91dee4369d49ac9bb9df21d8f66f55047ebf5b3.jpg)

![45ca49847a2d5147bf3c32821364def970518967ab4ae90e6a735dda8b87c881.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/45ca49847a2d5147bf3c32821364def970518967ab4ae90e6a735dda8b87c881.jpg)

![4787255b7ce3e91742d4c15e6bfc04cc59a3aee97a2121029aa304d3763f904d.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/4787255b7ce3e91742d4c15e6bfc04cc59a3aee97a2121029aa304d3763f904d.jpg)

![6bc88f3f82c8194670ecc18797f54177a663c7e9a25fb53831ec86785fba748f.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/6bc88f3f82c8194670ecc18797f54177a663c7e9a25fb53831ec86785fba748f.jpg)

![705e219a3d8c3441cf3c4b2c691fdc5784121085a069da3f1647cfcd3d171581.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/705e219a3d8c3441cf3c4b2c691fdc5784121085a069da3f1647cfcd3d171581.jpg)

![74a13255cb431bb3cede6d070006abb7b9e596f7b7cea89f3f53d2df8d3737e8.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/74a13255cb431bb3cede6d070006abb7b9e596f7b7cea89f3f53d2df8d3737e8.jpg)

![7f513f37b20d5ec326945d37b53df3a5fda51e925b69a4e008bcbaaf55e96833.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/7f513f37b20d5ec326945d37b53df3a5fda51e925b69a4e008bcbaaf55e96833.jpg)

![822af058d0d75b731ac0bd86bd3bcdd66ea3c02f024b843139016dade62b995d.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/822af058d0d75b731ac0bd86bd3bcdd66ea3c02f024b843139016dade62b995d.jpg)

![87a4c4bd3aa0e3352bf580717903c2c52d3b39c40130838851e19f85e6f4269d.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/87a4c4bd3aa0e3352bf580717903c2c52d3b39c40130838851e19f85e6f4269d.jpg)

![917aaf3b861fde7a0abb510a5da41718b0fd93bbc2d4442175f4df646481ff87.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/917aaf3b861fde7a0abb510a5da41718b0fd93bbc2d4442175f4df646481ff87.jpg)

![9f9a5458d44c8d3259ae1bba1ad183a9d48161ac889f742ce2b768d0c242f531.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/9f9a5458d44c8d3259ae1bba1ad183a9d48161ac889f742ce2b768d0c242f531.jpg)

![a892ba650ac3a22fe0d49d93eecf1f67dccf80f84f88e5a001dc2e5ba1d83f73.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/a892ba650ac3a22fe0d49d93eecf1f67dccf80f84f88e5a001dc2e5ba1d83f73.jpg)

![bd344265ba3aed3545c4fcd29035f4f1531fc15b653b195ce87e222632dd35fb.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/bd344265ba3aed3545c4fcd29035f4f1531fc15b653b195ce87e222632dd35fb.jpg)

![ed7b49e8023fb3f1e0dc8d2b39ef3e9864a0c6c81a1ddcdf5aa62cfba92db2d5.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/images/ed7b49e8023fb3f1e0dc8d2b39ef3e9864a0c6c81a1ddcdf5aa62cfba92db2d5.jpg)

### Tables

![014fd2555536132a0c1e1da0a4a04e5b548bbdabb71f252338b9b6c13e953143.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/014fd2555536132a0c1e1da0a4a04e5b548bbdabb71f252338b9b6c13e953143.jpg)

![098195a26fe8367fb82e55514f441061509a7369b28759cfa50a4431acb389ae.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/098195a26fe8367fb82e55514f441061509a7369b28759cfa50a4431acb389ae.jpg)

![21d0380dcf4c4b38cc5f376561405aa264dd2f807af0f14dfabf608db191b7f3.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/21d0380dcf4c4b38cc5f376561405aa264dd2f807af0f14dfabf608db191b7f3.jpg)

![442739269f7c4ce873520770aece5f1210c9fccf9f421bb1484b34170ca388f2.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/442739269f7c4ce873520770aece5f1210c9fccf9f421bb1484b34170ca388f2.jpg)

![464acb368f4b6751f5c7450e4121895b40cc914551126ba94b0400b3200ada24.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/464acb368f4b6751f5c7450e4121895b40cc914551126ba94b0400b3200ada24.jpg)

![60bdda40338ab8b7daa2651bc4d3c4b78cda4c7f30d616d29ffc49da8c587bd0.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/60bdda40338ab8b7daa2651bc4d3c4b78cda4c7f30d616d29ffc49da8c587bd0.jpg)

![80e3d23fd8cc3bf6c392e2da0f8fe20c0e1f5fa7290fcd331644bf2514a54410.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/80e3d23fd8cc3bf6c392e2da0f8fe20c0e1f5fa7290fcd331644bf2514a54410.jpg)

![894698e24cdf32a30b2a137ed5fdce457e70d16cc4cb0b20c43949d5379a309b.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/894698e24cdf32a30b2a137ed5fdce457e70d16cc4cb0b20c43949d5379a309b.jpg)

![8a159a41a6b0fa8f6429235937585a37dee654919b25a87a98d0566d066966a0.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/8a159a41a6b0fa8f6429235937585a37dee654919b25a87a98d0566d066966a0.jpg)

![8a69faf2b40780e3efa3f02dbf608f663edb87dee865e2a6b7141b17dd2b54ba.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/8a69faf2b40780e3efa3f02dbf608f663edb87dee865e2a6b7141b17dd2b54ba.jpg)

![90f173097cfa456b477fe1f24b6881d1e34126043177ba1845906c50ecb605f4.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/90f173097cfa456b477fe1f24b6881d1e34126043177ba1845906c50ecb605f4.jpg)

![93306fd7ea3e3a653f7ea5ed3d03860ae3a897332d3a7b8517f362d72bc54ec8.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/93306fd7ea3e3a653f7ea5ed3d03860ae3a897332d3a7b8517f362d72bc54ec8.jpg)

![a4c0405744ffc5d7f26d19d67faf8b655bcd4ce4089d067d96006b0bad435ba0.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/a4c0405744ffc5d7f26d19d67faf8b655bcd4ce4089d067d96006b0bad435ba0.jpg)

![cf584a7a48f1789ac9a76a9973a5c64e450aacf37176f852ba0f5378aafbca8e.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/cf584a7a48f1789ac9a76a9973a5c64e450aacf37176f852ba0f5378aafbca8e.jpg)

![cf8ebf29c1c43a58ab74f55fcba4d6492da87065d435f16fab19c97396617af1.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/cf8ebf29c1c43a58ab74f55fcba4d6492da87065d435f16fab19c97396617af1.jpg)

![d4b0be9d838c4e23d741ae3d9a59c1c3e16bc1ab5972ba2aa0138640333cf628.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/d4b0be9d838c4e23d741ae3d9a59c1c3e16bc1ab5972ba2aa0138640333cf628.jpg)

![dda8c7cde84406d1032b8c779d4782d85b090db82747bac22b4a2aa221e77feb.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/dda8c7cde84406d1032b8c779d4782d85b090db82747bac22b4a2aa221e77feb.jpg)

![f14148aa36a334be2c4a87a05bfbf89031cf330b6437a97aa9832f4f9cdd393f.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/f14148aa36a334be2c4a87a05bfbf89031cf330b6437a97aa9832f4f9cdd393f.jpg)

![fc283c98a6a3840e562cefbe1dfdd273ca0248fa8f03e83612615daf8e98b45b.jpg](../icml_results/863_Escaping%20the%20Verifier_%20Learning%20to%20Reason%20via%20Demonstrations/tables/fc283c98a6a3840e562cefbe1dfdd273ca0248fa8f03e83612615daf8e98b45b.jpg)

<a id="paper-0848"></a>

## Column Thresholding for Sparse Spiked Wigner Models: Improved Signal Strength Requirements


### Images

![0c18a8ca695598cbab3f629fb052d4bcf363c3424ab2f21f1048e07036e0bb18.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/0c18a8ca695598cbab3f629fb052d4bcf363c3424ab2f21f1048e07036e0bb18.jpg)

![2a3477534eca8b07efd54a137989d9cdc13e36674510c1cae99a889e537b082d.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/2a3477534eca8b07efd54a137989d9cdc13e36674510c1cae99a889e537b082d.jpg)

![7e81f17c35cd198802c5abe8069b1fb0f2ace663e20ae2918db531d508df213b.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/7e81f17c35cd198802c5abe8069b1fb0f2ace663e20ae2918db531d508df213b.jpg)

![9ddf1193ba717f56bdfe7aceb2f0e75307e9a096c26d21bb66487413103e02a1.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/9ddf1193ba717f56bdfe7aceb2f0e75307e9a096c26d21bb66487413103e02a1.jpg)

![b2b8f8e4788c3e9096d462c46274efa7fbef761a681ec7b29e57de1e3edd1c03.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/b2b8f8e4788c3e9096d462c46274efa7fbef761a681ec7b29e57de1e3edd1c03.jpg)

![bc2422cf12f6f66537862c20a35d503e7c00dcb72c852902193f3fe87d7c8b6c.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/bc2422cf12f6f66537862c20a35d503e7c00dcb72c852902193f3fe87d7c8b6c.jpg)

![c706114895a080ead8ac431ebaa20aab30bd969256c17ea0e217e75210747823.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/c706114895a080ead8ac431ebaa20aab30bd969256c17ea0e217e75210747823.jpg)

![ce3cdef28ef2d01a3a2c4955daf22afcef89cfa1344bbe375c5be8dc3d1bef4d.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/ce3cdef28ef2d01a3a2c4955daf22afcef89cfa1344bbe375c5be8dc3d1bef4d.jpg)

![cfd24ecb765c2356d3f8fe97e3af99ef993b02190762d3fc078abe5398960e48.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/cfd24ecb765c2356d3f8fe97e3af99ef993b02190762d3fc078abe5398960e48.jpg)

![dea4791b84c67b02b2eb2c619a5f6e2d4e63bed782bf087c8334dcfe8c79a7f3.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/dea4791b84c67b02b2eb2c619a5f6e2d4e63bed782bf087c8334dcfe8c79a7f3.jpg)

![eee7a9099c8cfa492d80be5e9f2c751b46c87ca216c841daa5942daeca034e15.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/images/eee7a9099c8cfa492d80be5e9f2c751b46c87ca216c841daa5942daeca034e15.jpg)

### Tables

![7e8dfff8989fab8233a359b306ac1580ee39e3ab8162593a0a39198a234e663f.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/tables/7e8dfff8989fab8233a359b306ac1580ee39e3ab8162593a0a39198a234e663f.jpg)

![90fad29ae593f5110883c434144a1d8a0b46811a9b12d35200b7da18b98d1ae6.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/tables/90fad29ae593f5110883c434144a1d8a0b46811a9b12d35200b7da18b98d1ae6.jpg)

![a3bd65a2144e060d211d62298033352d0cb823d1f422e40060fef71d89dd8535.jpg](../icml_results/864_Column%20Thresholding%20for%20Sparse%20Spiked%20Wigner%20Models_%20Improved%20Signal%20Strength%20Requirements/tables/a3bd65a2144e060d211d62298033352d0cb823d1f422e40060fef71d89dd8535.jpg)

<a id="paper-0849"></a>

## FUSE: Full-spectrum Unlearnable Examples via Spectral Equalization


### Images

![1dce7905c4b1a4dd6ea90d9357b8cb3444ac4c81be5db15c9cdea76fd9bc9d3f.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/1dce7905c4b1a4dd6ea90d9357b8cb3444ac4c81be5db15c9cdea76fd9bc9d3f.jpg)

![4297d1d286f394358f6424e75ec8ff5aa82ac476fad95c4ab924494d65126d87.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/4297d1d286f394358f6424e75ec8ff5aa82ac476fad95c4ab924494d65126d87.jpg)

![4906f5d4d14a7431041a97b7bc84b2e3e3ac05b34eb758f00acb58fecf7d09bd.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/4906f5d4d14a7431041a97b7bc84b2e3e3ac05b34eb758f00acb58fecf7d09bd.jpg)

![64a95f34547022977bf306908d554265e4e5606239d78eee8c3c1269314c8dc0.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/64a95f34547022977bf306908d554265e4e5606239d78eee8c3c1269314c8dc0.jpg)

![923d82a206cbac2162674a632571b646539b46f072ca7d258aead843cac34049.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/923d82a206cbac2162674a632571b646539b46f072ca7d258aead843cac34049.jpg)

![96400e612f550a8f0a43b1f38b08216e36824eda3317f65d5de3476c547d3726.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/96400e612f550a8f0a43b1f38b08216e36824eda3317f65d5de3476c547d3726.jpg)

![9859b9c2c0cdd2c4c474e6b0d2101508d1bd2dcf1ef943d2805397164c4e2555.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/9859b9c2c0cdd2c4c474e6b0d2101508d1bd2dcf1ef943d2805397164c4e2555.jpg)

![9beb39cf7435a440139d08689e47075bb8221227cc43b68d608aca4b0844a5ae.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/9beb39cf7435a440139d08689e47075bb8221227cc43b68d608aca4b0844a5ae.jpg)

![afcce6c4982941939b20a5bac0783f671ad0bdbd25e9a9eb54459fac1063a803.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/afcce6c4982941939b20a5bac0783f671ad0bdbd25e9a9eb54459fac1063a803.jpg)

![bcf8612e94c484303df339f8fdf31b8eb93c20c98d06492d4dc460582617e171.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/bcf8612e94c484303df339f8fdf31b8eb93c20c98d06492d4dc460582617e171.jpg)

![c87a96e568b8d96f7eb13894e4da8cf5df59238bcf5917fe86e56633da2c56f4.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/c87a96e568b8d96f7eb13894e4da8cf5df59238bcf5917fe86e56633da2c56f4.jpg)

![f505ede4ee6f1f2ebec5b5a04b90f5ca8d6c2354a0f5260944898e73ea956cc9.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/images/f505ede4ee6f1f2ebec5b5a04b90f5ca8d6c2354a0f5260944898e73ea956cc9.jpg)

### Tables

![01a0eeb47e9f19c8984bb70c67fee33a77dfb90e52c7883b89b97ab693c0420d.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/01a0eeb47e9f19c8984bb70c67fee33a77dfb90e52c7883b89b97ab693c0420d.jpg)

![0450e5ba44f6558d4597549b56e22a125a9599b1aadcff18dd9ff23a6293ff72.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/0450e5ba44f6558d4597549b56e22a125a9599b1aadcff18dd9ff23a6293ff72.jpg)

![0d5d5c0f9c0c8bceb65b576bdf29fecd79f51141623a847e8608e2652610d07b.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/0d5d5c0f9c0c8bceb65b576bdf29fecd79f51141623a847e8608e2652610d07b.jpg)

![12051c648b42a217323e3256fad5074efd18d988afbd41c65e2ea33a8bda4b4c.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/12051c648b42a217323e3256fad5074efd18d988afbd41c65e2ea33a8bda4b4c.jpg)

![24502cac7f5a3146699091a6f89ee854ea64a9f8c16318eee8ec6bd2e61e0db9.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/24502cac7f5a3146699091a6f89ee854ea64a9f8c16318eee8ec6bd2e61e0db9.jpg)

![330cbc89cdd5b8a439e18386f7196731197b720cad9b8481a0eec562e0aeddda.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/330cbc89cdd5b8a439e18386f7196731197b720cad9b8481a0eec562e0aeddda.jpg)

![3b077a6101d822ca35ce6bbed8113c76002ad745c4da27ffa8ccd70ee49029bf.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/3b077a6101d822ca35ce6bbed8113c76002ad745c4da27ffa8ccd70ee49029bf.jpg)

![3b721dcdb370ab37c7a3dff4bfa23ad33dff3718fad211b0e0bb492b481505aa.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/3b721dcdb370ab37c7a3dff4bfa23ad33dff3718fad211b0e0bb492b481505aa.jpg)

![400f4fced95509af988538bb9459629b5cb24c8f3e532a52d56180ed12a59a7c.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/400f4fced95509af988538bb9459629b5cb24c8f3e532a52d56180ed12a59a7c.jpg)

![433be5b41c2c7d42ab7422fc5fd419cb894bc5ae58e74a589ea01f9f1dc02f82.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/433be5b41c2c7d42ab7422fc5fd419cb894bc5ae58e74a589ea01f9f1dc02f82.jpg)

![4749867fd90385026717c8395ffba1382fc1c9fea027ad2ffed81792ebc1f08d.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/4749867fd90385026717c8395ffba1382fc1c9fea027ad2ffed81792ebc1f08d.jpg)

![47a6846ca63a8ad8a03e2dc81138a2b74a10b6981c3436dd26091f5a221cefc5.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/47a6846ca63a8ad8a03e2dc81138a2b74a10b6981c3436dd26091f5a221cefc5.jpg)

![65e674faabf5c7e8e6e04f41c123640e23176bac3156db88d85aa4e7f4c373a0.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/65e674faabf5c7e8e6e04f41c123640e23176bac3156db88d85aa4e7f4c373a0.jpg)

![6ac4f641681b1116bfe4b4a272c5145dfc1d840e3494ae7b8ed9ef9a42e8d0c2.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/6ac4f641681b1116bfe4b4a272c5145dfc1d840e3494ae7b8ed9ef9a42e8d0c2.jpg)

![7155e15b9565673920c43cd388c217ca27cd9cb7feddd04b1e61f0f965fed6fc.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/7155e15b9565673920c43cd388c217ca27cd9cb7feddd04b1e61f0f965fed6fc.jpg)

![745013a432dff129df0f8a733deeedd697d63749509e771290da7044f26d34cb.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/745013a432dff129df0f8a733deeedd697d63749509e771290da7044f26d34cb.jpg)

![85b96880ead7de9dbdf6e8ae271eb3164992821d72c3834f8c9b00b016562625.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/85b96880ead7de9dbdf6e8ae271eb3164992821d72c3834f8c9b00b016562625.jpg)

![8636db9e5e8e77ef2cb6fbb1eccf00f183cf8efb52466c6c12cb5247c6c5e3da.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/8636db9e5e8e77ef2cb6fbb1eccf00f183cf8efb52466c6c12cb5247c6c5e3da.jpg)

![8d92c34bd92f779e0880d4a4cd1db5a9984e6534cb5f70927cdef4df68cfb6f0.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/8d92c34bd92f779e0880d4a4cd1db5a9984e6534cb5f70927cdef4df68cfb6f0.jpg)

![a5b69ce527cbaa9775e96945b78fa9898bb7d2ae3cf0e700bcbb9a4340692a15.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/a5b69ce527cbaa9775e96945b78fa9898bb7d2ae3cf0e700bcbb9a4340692a15.jpg)

![a79ba66b8de7ef9d837a64262a96605325f929c8dc9f7fc99ecb3fae411ce4ae.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/a79ba66b8de7ef9d837a64262a96605325f929c8dc9f7fc99ecb3fae411ce4ae.jpg)

![afa70be3c39adc0e6ebdf820dfedac284fbc483bd39bd5abe390ed39837e9281.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/afa70be3c39adc0e6ebdf820dfedac284fbc483bd39bd5abe390ed39837e9281.jpg)

![bde54c78bc356aa74f6272caf9f96d103f59ced312f54d96526952a86f971b9a.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/bde54c78bc356aa74f6272caf9f96d103f59ced312f54d96526952a86f971b9a.jpg)

![c33310a617f5deb392ad85315fec159a1b36edbc6c0b078e328dd268d44f5d9f.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/c33310a617f5deb392ad85315fec159a1b36edbc6c0b078e328dd268d44f5d9f.jpg)

![c88d2173ae65ae73b3a8c4a29205606e74da43b98a0e6eb9d3a1a78b83583c15.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/c88d2173ae65ae73b3a8c4a29205606e74da43b98a0e6eb9d3a1a78b83583c15.jpg)

![ca00d896b36f1324636458a8f31c91e0832cbdc3282043efa282b69ee26923f2.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/ca00d896b36f1324636458a8f31c91e0832cbdc3282043efa282b69ee26923f2.jpg)

![cc271aa7d4ea210fcae65f2bd7a2ff85dfa9fdc582e418e1254f6d68579d20a5.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/cc271aa7d4ea210fcae65f2bd7a2ff85dfa9fdc582e418e1254f6d68579d20a5.jpg)

![d2e3b708db91a64272605f7a4382eef7bcfcc7bc5150a7d606e77abb9b455f8f.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/d2e3b708db91a64272605f7a4382eef7bcfcc7bc5150a7d606e77abb9b455f8f.jpg)

![db239d01f413f407e31f9676cfa012b68760c0e6330f46222208accd4e6f90fa.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/db239d01f413f407e31f9676cfa012b68760c0e6330f46222208accd4e6f90fa.jpg)

![eae4fc29da1b916a9d981c89bf4c1ebda1200d6ff1257b52ee591fb587df0e4f.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/eae4fc29da1b916a9d981c89bf4c1ebda1200d6ff1257b52ee591fb587df0e4f.jpg)

![fa660c6e9094914efe0a1b0ab29e7d659f11817ccfdbf2abc8e25b8ade95c50c.jpg](../icml_results/865_FUSE_%20Full-spectrum%20Unlearnable%20Examples%20via%20Spectral%20Equalization/tables/fa660c6e9094914efe0a1b0ab29e7d659f11817ccfdbf2abc8e25b8ade95c50c.jpg)

<a id="paper-0850"></a>

## TIMI: Training-Free Image-to-3D Multi-Instance Generation with Spatial Fidelity


### Images

![0e519a3c50b3db0d447e6a9ce16ceb7bcf3f4d7b47b47f7fc0ef27b80b758f4b.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/0e519a3c50b3db0d447e6a9ce16ceb7bcf3f4d7b47b47f7fc0ef27b80b758f4b.jpg)

![1d9ff6f65b6d047e5fdb75c8d4743f047ab6d38db778f09b8100cd9c5ad99f56.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/1d9ff6f65b6d047e5fdb75c8d4743f047ab6d38db778f09b8100cd9c5ad99f56.jpg)

![275cfdec99d8dc4db65f2a787f26aaadf1adc660a2de17b68a2ee921e49a0d25.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/275cfdec99d8dc4db65f2a787f26aaadf1adc660a2de17b68a2ee921e49a0d25.jpg)

![318fdb3affd444ad65906a288b470faf3de9534bf023617b319fe5a9acb40d32.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/318fdb3affd444ad65906a288b470faf3de9534bf023617b319fe5a9acb40d32.jpg)

![3d3ec116fc164aa8ba79d06cc9b8976cbc217e859d86c558432d370234451914.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/3d3ec116fc164aa8ba79d06cc9b8976cbc217e859d86c558432d370234451914.jpg)

![427011b8491424e961ec78dffe04a18e2e533bea9ba2e14b0e7ba05cab6b5e46.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/427011b8491424e961ec78dffe04a18e2e533bea9ba2e14b0e7ba05cab6b5e46.jpg)

![81cff8536d856816954a95fbbe586d5117e66df95ae9f00b4039c858b52b1251.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/81cff8536d856816954a95fbbe586d5117e66df95ae9f00b4039c858b52b1251.jpg)

![842cd297b4a74412295f523eb3d5b35c6125df507a7c8bd183c73438b1b33199.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/842cd297b4a74412295f523eb3d5b35c6125df507a7c8bd183c73438b1b33199.jpg)

![c79de5da6575a570cd394c87a0b2ede0ff752004b0ec1b19a4f2f5d1a6f24b15.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/c79de5da6575a570cd394c87a0b2ede0ff752004b0ec1b19a4f2f5d1a6f24b15.jpg)

![f7e9a7055658884e04414b93ec14c441f721e61638f72f8be00d05a9c7be36d9.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/images/f7e9a7055658884e04414b93ec14c441f721e61638f72f8be00d05a9c7be36d9.jpg)

### Tables

![054b82411ca5e9f69751f5b384f4c311f1a908af56b1c8be3fb7ee9b5073b2be.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/054b82411ca5e9f69751f5b384f4c311f1a908af56b1c8be3fb7ee9b5073b2be.jpg)

![14e04b880d05a3c5f1cad0f9e9ec60d30590f457ff5f54ee4905d6e137435ef3.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/14e04b880d05a3c5f1cad0f9e9ec60d30590f457ff5f54ee4905d6e137435ef3.jpg)

![351626a3b97e789518ac601cc960b822ca5d20c967862bbdd829f6db8f68e9f6.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/351626a3b97e789518ac601cc960b822ca5d20c967862bbdd829f6db8f68e9f6.jpg)

![38cec9b1c2198ae8e37e9ccd46281ecd63da496a32ddca4bcffa2416cf4d79d0.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/38cec9b1c2198ae8e37e9ccd46281ecd63da496a32ddca4bcffa2416cf4d79d0.jpg)

![7100bcfaae2801dd04e16a621611e296b7e690044e0fd82fcfa73394ba36095d.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/7100bcfaae2801dd04e16a621611e296b7e690044e0fd82fcfa73394ba36095d.jpg)

![828b7844c698dbf365b02556ce392650cd500b9eb2de79cb54cc8cade24e1fc3.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/828b7844c698dbf365b02556ce392650cd500b9eb2de79cb54cc8cade24e1fc3.jpg)

![92baee0f40e67835200e018ac19497d21753d344652b109312eab5456b15f0f2.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/92baee0f40e67835200e018ac19497d21753d344652b109312eab5456b15f0f2.jpg)

![f73636fe4b100ed73412e518e56822436086b372296faf5348b8c82b6a65bec7.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/f73636fe4b100ed73412e518e56822436086b372296faf5348b8c82b6a65bec7.jpg)

![f869ebcb16c38ee25543b5950aecea2ce3a863f773b24c918f7eae697d1ac198.jpg](../icml_results/866_TIMI_%20Training-Free%20Image-to-3D%20Multi-Instance%20Generation%20with%20Spatial%20Fidelity/tables/f869ebcb16c38ee25543b5950aecea2ce3a863f773b24c918f7eae697d1ac198.jpg)

<a id="paper-0851"></a>

## The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence


### Images

![5870134b4c1a49d8158a801f8297c1084054e94f351fcaf69fa08b1afe6a12ad.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/5870134b4c1a49d8158a801f8297c1084054e94f351fcaf69fa08b1afe6a12ad.jpg)

![5b04b1e179fbbdba6971aa9388683dae69d00e7f18a8d419fded25fad1a83567.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/5b04b1e179fbbdba6971aa9388683dae69d00e7f18a8d419fded25fad1a83567.jpg)

![6de99e25faf0d046d7d4eca66db85af9e111207aedc444b26ab6a47fdced95ca.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/6de99e25faf0d046d7d4eca66db85af9e111207aedc444b26ab6a47fdced95ca.jpg)

![97f7a3b0e173f25e849daa6ee0de87494a8713c05f9b46671f796f6b4e9f2567.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/97f7a3b0e173f25e849daa6ee0de87494a8713c05f9b46671f796f6b4e9f2567.jpg)

![9c2bef23ff3f0a28aaf5ecf163d70ba03b2bc51ae0565fdb6d17466b158a2620.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/9c2bef23ff3f0a28aaf5ecf163d70ba03b2bc51ae0565fdb6d17466b158a2620.jpg)

![b030915b521413596370b3d117f1a1bd6b97576ee6fcf45866af6b6657df6b8e.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/b030915b521413596370b3d117f1a1bd6b97576ee6fcf45866af6b6657df6b8e.jpg)

![b5ded52fc7585df532446c2793316e96827ae9c4fcde6d13aa1c87de8e3fbeb4.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/b5ded52fc7585df532446c2793316e96827ae9c4fcde6d13aa1c87de8e3fbeb4.jpg)

![bb8cadee42e4a594588b9115e8a6ce926c6a01f672025097d4ac22aa835a89f1.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/bb8cadee42e4a594588b9115e8a6ce926c6a01f672025097d4ac22aa835a89f1.jpg)

![bd13693f7244d50ad56ea4c9266a03a7642750dfe7a2e53c584c163a28738fe4.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/bd13693f7244d50ad56ea4c9266a03a7642750dfe7a2e53c584c163a28738fe4.jpg)

![c7302757c1209f2b88d43268be755c5d21cb5e59c55cb0968821c6f29419353e.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/c7302757c1209f2b88d43268be755c5d21cb5e59c55cb0968821c6f29419353e.jpg)

![cc9780fbfcca7f0b2dbdf7a1dc0db29dc0be56f89b509d9f9fbe9fa00a063519.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/cc9780fbfcca7f0b2dbdf7a1dc0db29dc0be56f89b509d9f9fbe9fa00a063519.jpg)

![d6e6a401d15f113e07a38ab2e19d88686d223965d97a5e08f022cb37cd923658.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/images/d6e6a401d15f113e07a38ab2e19d88686d223965d97a5e08f022cb37cd923658.jpg)

### Tables

![17888facba8bada43041f90ba13e07724370c7dea55d1ac0e3e04cfe2b83004d.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/tables/17888facba8bada43041f90ba13e07724370c7dea55d1ac0e3e04cfe2b83004d.jpg)

![8877345674eb6bdd620c8e015328b2cd5ac28c0ff9644456859746b8bdd0cd01.jpg](../icml_results/867_The%20Geometric%20Mechanics%20of%20Contrastive%20Representation%20Learning_%20Alignment%20Potentials%2C%20Entropic%20Dispe/tables/8877345674eb6bdd620c8e015328b2cd5ac28c0ff9644456859746b8bdd0cd01.jpg)

<a id="paper-0852"></a>

## Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality


### Images

![11fa7064edb0642da447306dc307fbbab590c6afb3b08a4174e3892ab9970559.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/11fa7064edb0642da447306dc307fbbab590c6afb3b08a4174e3892ab9970559.jpg)

![1e9aa7aba706efe501258e61176990cd76bbc38f8cf7741c6f51ce037bb967bc.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/1e9aa7aba706efe501258e61176990cd76bbc38f8cf7741c6f51ce037bb967bc.jpg)

![2051882efcc6c7a3e19bab68d9ddd548c7dde2205e73f3fd362b9c5a0f7e39d3.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/2051882efcc6c7a3e19bab68d9ddd548c7dde2205e73f3fd362b9c5a0f7e39d3.jpg)

![22d765883817a6f4fa803976ccfb2930cb2b82cc95967b98d3c84a88427aa3d1.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/22d765883817a6f4fa803976ccfb2930cb2b82cc95967b98d3c84a88427aa3d1.jpg)

![24bfc0181f4620b4008fc6662a06ce3c75acd5510d977deeb6a8974a40b39d50.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/24bfc0181f4620b4008fc6662a06ce3c75acd5510d977deeb6a8974a40b39d50.jpg)

![25666e4657456e77b2dad9a0fca886d9d48cdd0a6e988ccc1ab3b22c7cb700cf.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/25666e4657456e77b2dad9a0fca886d9d48cdd0a6e988ccc1ab3b22c7cb700cf.jpg)

![3010869d0a805156e5b4497b8407638ca8ab3dd34485c29970884dd0f22885bf.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/3010869d0a805156e5b4497b8407638ca8ab3dd34485c29970884dd0f22885bf.jpg)

![3386b25dbcc34747cd4d4a1434e9504cd90c1790752c5776c9bf3b0bfca48809.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/3386b25dbcc34747cd4d4a1434e9504cd90c1790752c5776c9bf3b0bfca48809.jpg)

![3a93475a2f37dc12c8ec2bb03b87160661013eb99a4e3b6c054fe0b87f1167a3.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/3a93475a2f37dc12c8ec2bb03b87160661013eb99a4e3b6c054fe0b87f1167a3.jpg)

![4a677a959f178fe7488870c9999dd9529ae9a5428745fb6e03b36689a0e7ac15.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/4a677a959f178fe7488870c9999dd9529ae9a5428745fb6e03b36689a0e7ac15.jpg)

![699065accc36d39d7d8cffa5fda65ef95931a4d8ce1534efaf49ddccadd71242.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/699065accc36d39d7d8cffa5fda65ef95931a4d8ce1534efaf49ddccadd71242.jpg)

![6a1718762384c2225d414efac2675769985ac7cdeec30f2e4aa290b1e2565a95.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/6a1718762384c2225d414efac2675769985ac7cdeec30f2e4aa290b1e2565a95.jpg)

![9bd90005b51a67e007d3c47e4ac6ce954f7e5b41d213daed3a3c7d1378cf467a.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/9bd90005b51a67e007d3c47e4ac6ce954f7e5b41d213daed3a3c7d1378cf467a.jpg)

![a55ea91829d3a4a274acc5c957f3e50bdfa7338ebec69d288a771631a5d68875.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/a55ea91829d3a4a274acc5c957f3e50bdfa7338ebec69d288a771631a5d68875.jpg)

![b08e34423617f5baed8627ee7c5b132adf72e058d8b1e878b205e2279bf0695e.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/b08e34423617f5baed8627ee7c5b132adf72e058d8b1e878b205e2279bf0695e.jpg)

![b7d2890fa3261b1cce64c4943456ddbfc2bd65d0aad5dd6f54799364fb242eb9.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/b7d2890fa3261b1cce64c4943456ddbfc2bd65d0aad5dd6f54799364fb242eb9.jpg)

![ba44605097f65f41b20d70275d5b06e17e4a184a4e88cae8f67eda92c7a15c8d.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/ba44605097f65f41b20d70275d5b06e17e4a184a4e88cae8f67eda92c7a15c8d.jpg)

![c646509917f16d077740662a21c0df60c3a12675b8003982064b06ac665a2c65.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/c646509917f16d077740662a21c0df60c3a12675b8003982064b06ac665a2c65.jpg)

![fdc9cc7f65d46454cadfe6177242afccfaf5159be8cdc0074c46d3c93fc310a7.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/images/fdc9cc7f65d46454cadfe6177242afccfaf5159be8cdc0074c46d3c93fc310a7.jpg)

### Tables

![15ce3e3356b0e87bb897432169e1dc3808f671b47a6345aba811100994f8d8b9.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/tables/15ce3e3356b0e87bb897432169e1dc3808f671b47a6345aba811100994f8d8b9.jpg)

![314e3cab20b963b62cecc60408f60e4a8b9625f92b0bd393422d6cb37c58b273.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/tables/314e3cab20b963b62cecc60408f60e4a8b9625f92b0bd393422d6cb37c58b273.jpg)

![56d6f8f9cec4acfc9b8a86f67d54ceaa2f4dee6332ee069a14980e313106d949.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/tables/56d6f8f9cec4acfc9b8a86f67d54ceaa2f4dee6332ee069a14980e313106d949.jpg)

![81c68b64018e4db9a197d3e58f4284f27da8bfbb48fbe23fb41d11806a82cb48.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/tables/81c68b64018e4db9a197d3e58f4284f27da8bfbb48fbe23fb41d11806a82cb48.jpg)

![900fd22c6e3d67700102c99c65851f931165c8b1a67726390d035364dd7b09de.jpg](../icml_results/869_Empty%20Shelves%20or%20Lost%20Keys_%20Recall%20Is%20the%20Bottleneck%20for%20Parametric%20Factuality/tables/900fd22c6e3d67700102c99c65851f931165c8b1a67726390d035364dd7b09de.jpg)

<a id="paper-0853"></a>

## Richer Bayesian Last Layers with Subsampled NTK Features


### Images

![0a115e9c8c5e3d07df69f00ca4d0026576c010f5d58c2be3af3ed462cbada9a7.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/images/0a115e9c8c5e3d07df69f00ca4d0026576c010f5d58c2be3af3ed462cbada9a7.jpg)

![1485efc2ae7f4c21d50c5012a14558a3c5ae3b67af9fc3fe1852f9522404648c.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/images/1485efc2ae7f4c21d50c5012a14558a3c5ae3b67af9fc3fe1852f9522404648c.jpg)

![3122065874536397799e797e19ea4e02c00ff3fb39bce49a445535c2673a58d9.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/images/3122065874536397799e797e19ea4e02c00ff3fb39bce49a445535c2673a58d9.jpg)

![467c873140848e7b6d4a55bcd931479d2bf9e5e0b975bc50f7bf6a4c697498b8.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/images/467c873140848e7b6d4a55bcd931479d2bf9e5e0b975bc50f7bf6a4c697498b8.jpg)

### Tables

![21654f4821d47127c2da492fcb54247125e71ccc125957482c93dbda8a35dea4.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/21654f4821d47127c2da492fcb54247125e71ccc125957482c93dbda8a35dea4.jpg)

![a1538ab8f7f9c9db66c96ddbaeaf4ac62796b5e89c6ed4bdb2f6c6e736e8a775.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/a1538ab8f7f9c9db66c96ddbaeaf4ac62796b5e89c6ed4bdb2f6c6e736e8a775.jpg)

![b0ceed0ff2912bad6f1a604ebfe91b78ec94389d2510d7ad764a436d93a040c1.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/b0ceed0ff2912bad6f1a604ebfe91b78ec94389d2510d7ad764a436d93a040c1.jpg)

![db33c8a6308a95395533d1a66e9531eb86c45543dcb908a7c3799be37967669d.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/db33c8a6308a95395533d1a66e9531eb86c45543dcb908a7c3799be37967669d.jpg)

![df6b9584ff2140527b4560dd49af41fe055173b52a8c0e0b11999cb0e19fd8a2.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/df6b9584ff2140527b4560dd49af41fe055173b52a8c0e0b11999cb0e19fd8a2.jpg)

![eeb91dbbeb5218ba28d779112364b808c06839a974a0ae00a22fe5de54f5541a.jpg](../icml_results/870_Richer%20Bayesian%20Last%20Layers%20with%20Subsampled%20NTK%20Features/tables/eeb91dbbeb5218ba28d779112364b808c06839a974a0ae00a22fe5de54f5541a.jpg)

<a id="paper-0854"></a>

## One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models


### Images

![00e397a74da0ff84f16309a0fa36edee7d592f70caad429b6b1831fc46546277.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/00e397a74da0ff84f16309a0fa36edee7d592f70caad429b6b1831fc46546277.jpg)

![13993b1a77c31988c20dc4bd78da6ce613749ed8a5b52118c42f8b29e5b33502.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/13993b1a77c31988c20dc4bd78da6ce613749ed8a5b52118c42f8b29e5b33502.jpg)

![1aff4e61b1e0f5605e51c94a7ff11837ede05a7a693169161e0883740e4811be.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/1aff4e61b1e0f5605e51c94a7ff11837ede05a7a693169161e0883740e4811be.jpg)

![2092f380f02a440626096efce26a4ca2e4fa4d0d3413d1fe07e021cbb19df96b.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/2092f380f02a440626096efce26a4ca2e4fa4d0d3413d1fe07e021cbb19df96b.jpg)

![3537c4be1aff95b8e2eb87125e7e88702d56166703b161b12aaf47d7b714abac.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/3537c4be1aff95b8e2eb87125e7e88702d56166703b161b12aaf47d7b714abac.jpg)

![48d26a7d6255f7caf491f72ef488a0ee3dbde3d21e0d90e1f7f2b771ad9bf6fd.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/48d26a7d6255f7caf491f72ef488a0ee3dbde3d21e0d90e1f7f2b771ad9bf6fd.jpg)

![64ed77b5d13d144ae0ea141db429caeeb8531964c191fcbb152fab89083de08a.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/64ed77b5d13d144ae0ea141db429caeeb8531964c191fcbb152fab89083de08a.jpg)

![6720973c123c8287d08dc458368d3ce07b14816e117789f89794848a5403e1d4.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/6720973c123c8287d08dc458368d3ce07b14816e117789f89794848a5403e1d4.jpg)

![70005c743adac584ce5a913e2336827de59cc7884d25f1c3ac20d654d17b23dd.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/70005c743adac584ce5a913e2336827de59cc7884d25f1c3ac20d654d17b23dd.jpg)

![7fb7fd4fad2a3882212b4ab7153a1a1a825f094a5fb6fe0fcd32c2fe142afe39.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/7fb7fd4fad2a3882212b4ab7153a1a1a825f094a5fb6fe0fcd32c2fe142afe39.jpg)

![c4ed15537e37be4196c4efec8cbb97a2d1979066482a9a5b8d209746c384b8a3.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/c4ed15537e37be4196c4efec8cbb97a2d1979066482a9a5b8d209746c384b8a3.jpg)

![e96da1cfa84ae91c3b7e1a808f3aa28b6d2e84eca7ec899cabdf842b3082c650.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/images/e96da1cfa84ae91c3b7e1a808f3aa28b6d2e84eca7ec899cabdf842b3082c650.jpg)

### Tables

![180f1b1e346a38bcec50af07381e1278e6138c73fe48f4c76fc1efc952dbd666.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/180f1b1e346a38bcec50af07381e1278e6138c73fe48f4c76fc1efc952dbd666.jpg)

![9557ee85860aab65ef9b636dbf1a03a83a8613ea5cc759856a8e3b58de9485b2.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/9557ee85860aab65ef9b636dbf1a03a83a8613ea5cc759856a8e3b58de9485b2.jpg)

![c41e41ca6b524e9d6f954fce304f9ccd0d4ba5ff50512eb108ece697cee5d604.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/c41e41ca6b524e9d6f954fce304f9ccd0d4ba5ff50512eb108ece697cee5d604.jpg)

![c5fbc46eab367b546e50a20a9a7c9021c001cb6b7d98c2c52fc9d18c043cf529.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/c5fbc46eab367b546e50a20a9a7c9021c001cb6b7d98c2c52fc9d18c043cf529.jpg)

![ca87b14e2623f3a18e3ec2fa5384bfa0a71d42c455a5bbf4f0ab133db36aae93.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/ca87b14e2623f3a18e3ec2fa5384bfa0a71d42c455a5bbf4f0ab133db36aae93.jpg)

![dd7814a04347f7813bcebc2f788f9adfe85785ca53f175bfb9f2e5ef46412d59.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/dd7814a04347f7813bcebc2f788f9adfe85785ca53f175bfb9f2e5ef46412d59.jpg)

![de876510034baee01a286c2c8add3338680e64335d8ca74bc7b93fca7e72c31b.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/de876510034baee01a286c2c8add3338680e64335d8ca74bc7b93fca7e72c31b.jpg)

![e124a004afafc41a1f0d8450c4a462927c57a129fd7af232bd697eb68d4344c9.jpg](../icml_results/871_One%20Step%20Forward%20and%20K%20Steps%20Back_%20Better%20Reasoning%20with%20Denoising%20Recursion%20Models/tables/e124a004afafc41a1f0d8450c4a462927c57a129fd7af232bd697eb68d4344c9.jpg)

<a id="paper-0855"></a>

## Probably Approximately Correct Labels


### Images

![00dfe788c6ce28f271698163806ae8b1ba9c77d1eb294fbf9154ae8ad235d123.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/00dfe788c6ce28f271698163806ae8b1ba9c77d1eb294fbf9154ae8ad235d123.jpg)

![0eb59623989c1755a023c73acc9374a4d4fe3fed4aa33e0cba60be11af5d3249.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/0eb59623989c1755a023c73acc9374a4d4fe3fed4aa33e0cba60be11af5d3249.jpg)

![1d210825189cd15e0cdb9f0c5dc827721c43c20b1fec9c3623e776b6379fddeb.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/1d210825189cd15e0cdb9f0c5dc827721c43c20b1fec9c3623e776b6379fddeb.jpg)

![1dece85d913d0b142cfbb2f129bbbf813a1bfa04bb24db8f14e89adaa0ce2838.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/1dece85d913d0b142cfbb2f129bbbf813a1bfa04bb24db8f14e89adaa0ce2838.jpg)

![5967aff0702dea2b348f4a809c10cfaa6efea7a76051fde4ff509d0805d6a5ae.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/5967aff0702dea2b348f4a809c10cfaa6efea7a76051fde4ff509d0805d6a5ae.jpg)

![5ee7f69897fdd01b2f79b72871fcc5c56ff83f071f4be2ba856e3eb525dbda4c.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/5ee7f69897fdd01b2f79b72871fcc5c56ff83f071f4be2ba856e3eb525dbda4c.jpg)

![7780f7ffaff8f151cac8b9f287f8037d4210ae57e326bcf9645ed03e2f180f9b.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/7780f7ffaff8f151cac8b9f287f8037d4210ae57e326bcf9645ed03e2f180f9b.jpg)

![ecf4aeb8d03ccde06bdb3bded47dcf301bdb31f8e34cd437d53ba9b0b2103b30.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/images/ecf4aeb8d03ccde06bdb3bded47dcf301bdb31f8e34cd437d53ba9b0b2103b30.jpg)

### Tables

![01c4849ff98034a122e5654ed066df3f3aae65b9950595a40b4ceff733278841.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/01c4849ff98034a122e5654ed066df3f3aae65b9950595a40b4ceff733278841.jpg)

![10855d2d30813716665746af7590b82f3b7d39d6a1d4bd0d8e31b871a3f9d08b.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/10855d2d30813716665746af7590b82f3b7d39d6a1d4bd0d8e31b871a3f9d08b.jpg)

![4c968b8ca26fa6f59467666dc23bee7843ef088de72f954a70868818d91c5f1e.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/4c968b8ca26fa6f59467666dc23bee7843ef088de72f954a70868818d91c5f1e.jpg)

![618edee2f81eb05a0ac0a450074bb5cb6685c72264d54d017bc4ba36dfcf1f23.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/618edee2f81eb05a0ac0a450074bb5cb6685c72264d54d017bc4ba36dfcf1f23.jpg)

![99c7dd94a1da661c4ded888da5804c6e4e03900c13d397e0bfde280b50219294.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/99c7dd94a1da661c4ded888da5804c6e4e03900c13d397e0bfde280b50219294.jpg)

![d9a01877595759fa42eb6e405fa582afbc876fda15d8b732b7a0f25a04912af7.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/d9a01877595759fa42eb6e405fa582afbc876fda15d8b732b7a0f25a04912af7.jpg)

![f3b36058a89698673d87e00e78aa025ebcfa348a781c4bc224aa27b5358f7a4d.jpg](../icml_results/872_Probably%20Approximately%20Correct%20Labels/tables/f3b36058a89698673d87e00e78aa025ebcfa348a781c4bc224aa27b5358f7a4d.jpg)

<a id="paper-0856"></a>

## Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers


### Images

![0b2676d541dd864ddb7c0b448cccde398db8a4ed1e65f2d39006dc15805b186b.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/0b2676d541dd864ddb7c0b448cccde398db8a4ed1e65f2d39006dc15805b186b.jpg)

![146f397ffca40b92194ca1c350098707cb93de1f6307ca6c5cb92270b02a16a6.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/146f397ffca40b92194ca1c350098707cb93de1f6307ca6c5cb92270b02a16a6.jpg)

![1d786143c3fafebdb137194a6758a0333963093cafbcfa248f4eb82a103ca679.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/1d786143c3fafebdb137194a6758a0333963093cafbcfa248f4eb82a103ca679.jpg)

![2930131bf0e0fbe8127dee2195e7103f19f6c62c825871ef5f5a6abd429a214d.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/2930131bf0e0fbe8127dee2195e7103f19f6c62c825871ef5f5a6abd429a214d.jpg)

![51da0f7546275dd019b84ece906c7a75874f07fd3696a25ccec911ae6a98304c.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/51da0f7546275dd019b84ece906c7a75874f07fd3696a25ccec911ae6a98304c.jpg)

![5229e2446cdfab55bf4082a7cf4d3c2b9993e5d26af526940abdfd0c69abcf9f.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/5229e2446cdfab55bf4082a7cf4d3c2b9993e5d26af526940abdfd0c69abcf9f.jpg)

![6e1a7d7f9ddbc07b73cec6deecc1c374170f040d97af148dc97734fe81e4c812.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/6e1a7d7f9ddbc07b73cec6deecc1c374170f040d97af148dc97734fe81e4c812.jpg)

![78d16e6740651eb9d7151a7cca9c9107060d7cd94e48fa8e9aa16a1c59fa4f43.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/78d16e6740651eb9d7151a7cca9c9107060d7cd94e48fa8e9aa16a1c59fa4f43.jpg)

![7eee6a1a17df649cf5f9cb056b418f73a63ac6ca00c4452301a51266b74d9955.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/7eee6a1a17df649cf5f9cb056b418f73a63ac6ca00c4452301a51266b74d9955.jpg)

![86f18d2362fda27e839da066e99d468987799e8c2e910c8d371798c655060053.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/86f18d2362fda27e839da066e99d468987799e8c2e910c8d371798c655060053.jpg)

![be8dfcced30f3ee1e150f222706a71d095c8b421a7bc68d1a3d6d7a9e9b5d632.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/be8dfcced30f3ee1e150f222706a71d095c8b421a7bc68d1a3d6d7a9e9b5d632.jpg)

![c39f03e74ef0c28959853db30650d8cbdf8ea054810a17ffbf57f43fd4c0a413.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/c39f03e74ef0c28959853db30650d8cbdf8ea054810a17ffbf57f43fd4c0a413.jpg)

![c7e3e1a5447363b5e2d353a846aa1041e9782c0f7ba1ebffca956e84e289644e.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/images/c7e3e1a5447363b5e2d353a846aa1041e9782c0f7ba1ebffca956e84e289644e.jpg)

### Tables

![0286e27bcdb7813788f34055155db47cb7a946605ea836452dd9045a2175f630.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/0286e27bcdb7813788f34055155db47cb7a946605ea836452dd9045a2175f630.jpg)

![0e62f925bcaecf82ec8900037d495518fc4d0da633286aba659141ba10c041e3.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/0e62f925bcaecf82ec8900037d495518fc4d0da633286aba659141ba10c041e3.jpg)

![29afd9e10b828afbb0e273e85bfaf535b8cdae9b70ee2930b7f1e1b69de83a32.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/29afd9e10b828afbb0e273e85bfaf535b8cdae9b70ee2930b7f1e1b69de83a32.jpg)

![34fad2ed143a9f0ff904e8d574b41ee86266d05a93b71ebd964d20b3bf60e385.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/34fad2ed143a9f0ff904e8d574b41ee86266d05a93b71ebd964d20b3bf60e385.jpg)

![3936ea4638c08912abe37976a3495ee54785c9f581bc2d75db502c74f173df90.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/3936ea4638c08912abe37976a3495ee54785c9f581bc2d75db502c74f173df90.jpg)

![42f189f3b16152ffd53df44520cabb4ba4274ff244d8d36e693aaea6058ef7ed.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/42f189f3b16152ffd53df44520cabb4ba4274ff244d8d36e693aaea6058ef7ed.jpg)

![43c6b49225675f1d0955d66702a6cde59a1e53764a829caba5aaa0f9ebe1d4f6.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/43c6b49225675f1d0955d66702a6cde59a1e53764a829caba5aaa0f9ebe1d4f6.jpg)

![5107d544f95c8458aef390bbf4bb231fac1db7a659c5dd0d5d690147a53db6d5.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/5107d544f95c8458aef390bbf4bb231fac1db7a659c5dd0d5d690147a53db6d5.jpg)

![56b6cf2263ca8b851ffdf79bf457e95fd9d5eb2cd9d270c72e2b2b14d9a938d7.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/56b6cf2263ca8b851ffdf79bf457e95fd9d5eb2cd9d270c72e2b2b14d9a938d7.jpg)

![56e4e1b8ebb9f5e14e4024611aecc322d4e816bcd55f7d8a6b066abb0915e410.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/56e4e1b8ebb9f5e14e4024611aecc322d4e816bcd55f7d8a6b066abb0915e410.jpg)

![58a407bc0e237a280dd221398be7a87249d74cd4c4befe68987084c43cc6381f.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/58a407bc0e237a280dd221398be7a87249d74cd4c4befe68987084c43cc6381f.jpg)

![58d0bfa89685dd3b0e5da581940be3b5a8486897ee463121f803e2c3e406ca4e.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/58d0bfa89685dd3b0e5da581940be3b5a8486897ee463121f803e2c3e406ca4e.jpg)

![6407da112485c529dab6dad7e9a8ff60d43779740681b988b416ffb345e55c57.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/6407da112485c529dab6dad7e9a8ff60d43779740681b988b416ffb345e55c57.jpg)

![70a586ece0cb398d08da0b3be17cd4fb7b07c91a391eebb9afb233d5cd82589c.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/70a586ece0cb398d08da0b3be17cd4fb7b07c91a391eebb9afb233d5cd82589c.jpg)

![753f3551155e8c0d9668ae630f760a27c86e51afcce683572757e675cb952418.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/753f3551155e8c0d9668ae630f760a27c86e51afcce683572757e675cb952418.jpg)

![83ed43f7085b5fcbec5583bd91cc4cbed1b5e0b60bce7136f44c8150b5a2bbe7.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/83ed43f7085b5fcbec5583bd91cc4cbed1b5e0b60bce7136f44c8150b5a2bbe7.jpg)

![857c708e58a45be14d8b5b19571bed5585bcdc30ce42304698da0f44d47a1f38.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/857c708e58a45be14d8b5b19571bed5585bcdc30ce42304698da0f44d47a1f38.jpg)

![922b258df1d996179fae32681649c5dd172aee17beefc57a3114901f5ed4097e.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/922b258df1d996179fae32681649c5dd172aee17beefc57a3114901f5ed4097e.jpg)

![a6eeb2ce48b3b95f222d2c7b23107eee7643e3cbfae1f149c09668025a5022f9.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/a6eeb2ce48b3b95f222d2c7b23107eee7643e3cbfae1f149c09668025a5022f9.jpg)

![b23c3adca58c6caa02271cfd53ec96582fd46cb22ed362de58ceee4f3dc71ab4.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/b23c3adca58c6caa02271cfd53ec96582fd46cb22ed362de58ceee4f3dc71ab4.jpg)

![c56e4ed3b7d20cf8d07df9a25924236dde394b275bb5a4aa677decdbda2c752f.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/c56e4ed3b7d20cf8d07df9a25924236dde394b275bb5a4aa677decdbda2c752f.jpg)

![db93c5ab980fe0fbecb1eb019589de51ddb258c06e56d3c54c0d98ba8fb46a62.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/db93c5ab980fe0fbecb1eb019589de51ddb258c06e56d3c54c0d98ba8fb46a62.jpg)

![dc3c5642296fdd428744fdf45ba4dbbbc7e21f2e83855198b62b6b0e8d37d5a7.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/dc3c5642296fdd428744fdf45ba4dbbbc7e21f2e83855198b62b6b0e8d37d5a7.jpg)

![e8101d83bfe7fb149293ac5e1ce4babd59caa815d58adeb76c8cbc4540025d49.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/e8101d83bfe7fb149293ac5e1ce4babd59caa815d58adeb76c8cbc4540025d49.jpg)

![e8e386150190146f844c95c79e1907269aff22374b56b49d136a700669272190.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/e8e386150190146f844c95c79e1907269aff22374b56b49d136a700669272190.jpg)

![e9e1f8bcce211f3bc0ea75a1ffa1b2ec9098f888939fd095b617f364c572780e.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/e9e1f8bcce211f3bc0ea75a1ffa1b2ec9098f888939fd095b617f364c572780e.jpg)

![ec4cb7d7fbf44790bc689c7f5af43edec99ac06323db9417fc8486c0a37f5d0d.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/ec4cb7d7fbf44790bc689c7f5af43edec99ac06323db9417fc8486c0a37f5d0d.jpg)

![efd9060c0a14049138768dbb267ae4f3ff52ce45f73b423129004bbe997a13a4.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/efd9060c0a14049138768dbb267ae4f3ff52ce45f73b423129004bbe997a13a4.jpg)

![f96ee395e83384c5bfb6f15c79f8d5ab489a059034b6260a45f16cb581736bbb.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/f96ee395e83384c5bfb6f15c79f8d5ab489a059034b6260a45f16cb581736bbb.jpg)

![ff612d552dd632e6b049e9144a2ae9d75a31cb3fdacad67409219dc3bf58d11d.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/ff612d552dd632e6b049e9144a2ae9d75a31cb3fdacad67409219dc3bf58d11d.jpg)

![ff70a2fe213f1c0cd1123401cae6b9a1629f93183e2d20b552081f4568bce6d4.jpg](../icml_results/873_Spatial%20Priors%20via%20Space%20Filling%20Curves%20for%20Small%20and%20Limited%20Data%20Vision%20Transformers/tables/ff70a2fe213f1c0cd1123401cae6b9a1629f93183e2d20b552081f4568bce6d4.jpg)

<a id="paper-0857"></a>

## Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?


### Images

![6d9a4fc76c93fe3f2c322fd29f4cdd64ab083440409aa0d5a5fd574c0ca08b8a.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/images/6d9a4fc76c93fe3f2c322fd29f4cdd64ab083440409aa0d5a5fd574c0ca08b8a.jpg)

![fb6e911f179f150e13be4e848743107753ee8c5a6d5ce98c9e46e906df1cbf02.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/images/fb6e911f179f150e13be4e848743107753ee8c5a6d5ce98c9e46e906df1cbf02.jpg)

### Tables

![014d067c27e26456df36920e5c5d34ee7a258797a67f9622bc15cd88f2f8ef21.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/014d067c27e26456df36920e5c5d34ee7a258797a67f9622bc15cd88f2f8ef21.jpg)

![052504ae6cbd74fda6e953b58a6eb7a9008196191707a28aad74c44847fb77ff.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/052504ae6cbd74fda6e953b58a6eb7a9008196191707a28aad74c44847fb77ff.jpg)

![07452367ccaaf4b9b972802e01573cf5650d5dff5ed59d4de9d30135ef878243.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/07452367ccaaf4b9b972802e01573cf5650d5dff5ed59d4de9d30135ef878243.jpg)

![1ba7b768f3e48fbb4b2f8f2aa0b5dad07689010487190f92fd90c74de92584f2.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/1ba7b768f3e48fbb4b2f8f2aa0b5dad07689010487190f92fd90c74de92584f2.jpg)

![436acc54520f6b0b3f4bb709e1c81c14c65175db624c263ec82b418c7dc8f862.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/436acc54520f6b0b3f4bb709e1c81c14c65175db624c263ec82b418c7dc8f862.jpg)

![47f7ea656062d9a767c34112b5e331b0e7349e6c8fefc408a444ab3688fb9c4f.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/47f7ea656062d9a767c34112b5e331b0e7349e6c8fefc408a444ab3688fb9c4f.jpg)

![57745c0633f63ab7dea5bd39afbaa3c37aee88d458e5938e775821011e59efe3.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/57745c0633f63ab7dea5bd39afbaa3c37aee88d458e5938e775821011e59efe3.jpg)

![5cb8e48afd02a6c6504b792684bd2db993fbbf3742ea81e3cfff821cc8f146cf.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/5cb8e48afd02a6c6504b792684bd2db993fbbf3742ea81e3cfff821cc8f146cf.jpg)

![7374fa6db8fc0804808ff58f3d65778d4811ca4f4681bf0afc892dcef60b6d22.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/7374fa6db8fc0804808ff58f3d65778d4811ca4f4681bf0afc892dcef60b6d22.jpg)

![7ca6636f3960c02bcfc1eb951213bf455ae1eaa97222fa2cf8e4dec89441f739.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/7ca6636f3960c02bcfc1eb951213bf455ae1eaa97222fa2cf8e4dec89441f739.jpg)

![91083e59523a65a6777894e35d1c2afebf8ba5243b1a560cd5d8b39d4c154dde.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/91083e59523a65a6777894e35d1c2afebf8ba5243b1a560cd5d8b39d4c154dde.jpg)

![9144976ffd27cdf8a4606d8f37e1c4f43d1edb54a7e21b1a78f7fbb83d5acbe6.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/9144976ffd27cdf8a4606d8f37e1c4f43d1edb54a7e21b1a78f7fbb83d5acbe6.jpg)

![9f9684cca8f0ccb3cf09124e33f5b84fa82ac16fd50b571352dac958ff47fc80.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/9f9684cca8f0ccb3cf09124e33f5b84fa82ac16fd50b571352dac958ff47fc80.jpg)

![c64066f9fbe3be4c08f543b2046166d33c7ecb8bce07b9caa73dcdb5c39f2c1d.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/c64066f9fbe3be4c08f543b2046166d33c7ecb8bce07b9caa73dcdb5c39f2c1d.jpg)

![cfb0ab527c385fa7b9fa52710e80af64d1c0cd896c96c0051e5434ecdddc29f1.jpg](../icml_results/874_Can%20Computational%20Reducibility%20Lead%20to%20Transferable%20Models%20for%20Graph%20Combinatorial%20Optimization_/tables/cfb0ab527c385fa7b9fa52710e80af64d1c0cd896c96c0051e5434ecdddc29f1.jpg)

<a id="paper-0858"></a>

## On the Robustness of Langevin Dynamics to Score Function Error

### Images

![890025811ef70c9b43babd3c543767f7230ceb6147d97906fd80e1f2811bbba5.jpg](../icml_results/875_On%20the%20Robustness%20of%20Langevin%20Dynamics%20to%20Score%20Function%20Error/images/890025811ef70c9b43babd3c543767f7230ceb6147d97906fd80e1f2811bbba5.jpg)

![a7c4f6a5b399ab3561a63d3f05abe400ad00f9063994f326d12e60c30490f383.jpg](../icml_results/875_On%20the%20Robustness%20of%20Langevin%20Dynamics%20to%20Score%20Function%20Error/images/a7c4f6a5b399ab3561a63d3f05abe400ad00f9063994f326d12e60c30490f383.jpg)

![b2a9036093f7708652cce8875d29d0584a38e35a23daab1fc0eeb70cf137726b.jpg](../icml_results/875_On%20the%20Robustness%20of%20Langevin%20Dynamics%20to%20Score%20Function%20Error/images/b2a9036093f7708652cce8875d29d0584a38e35a23daab1fc0eeb70cf137726b.jpg)

![fa5bf2e64f23cd85abb030fbea5b1c1cab36683ba6356f11be758db94c1b1a9e.jpg](../icml_results/875_On%20the%20Robustness%20of%20Langevin%20Dynamics%20to%20Score%20Function%20Error/images/fa5bf2e64f23cd85abb030fbea5b1c1cab36683ba6356f11be758db94c1b1a9e.jpg)

### Tables

![4b768fca499a76d0d85d300b600584d6c8c98e01946817a3f99d628a860be77c.jpg](../icml_results/875_On%20the%20Robustness%20of%20Langevin%20Dynamics%20to%20Score%20Function%20Error/tables/4b768fca499a76d0d85d300b600584d6c8c98e01946817a3f99d628a860be77c.jpg)
