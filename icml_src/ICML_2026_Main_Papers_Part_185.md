# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 185 of 197

## 目录 (Table of Contents)

1. [Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding](#paper-6073)
2. [HilbertA: Hilbert-Curve–Aligned Sparse Attention for 2D Structured Data](#paper-6074)
3. [Overcoming the Modality Gap in Context-Aided Forecasting](#paper-6075)
4. [Mosaic: Unlocking Over 30$\times$ Context Length for Diffusion LLMs Inference via Global Memory Planning and Dynamic Peak Taming](#paper-6076)
5. [Solving Imperfect-Recall Games via Sum-of-Squares Optimization](#paper-6077)
6. [SPADA: A Verifiable Test-Driven Agent for Controllable Parametric CAD Assembly Generation](#paper-6078)
7. [Spatio-Temporal LLM: Reasoning about Environments and Actions](#paper-6079)
8. [Unsupervised Process-Aware Coreset Selection for In-Context Learning](#paper-6080)
9. [NeuroCLUS: A Foundation Model with Functional Clustering for Intracranial Neural Decoding](#paper-6081)
10. [LoCoT2V-Bench: Benchmarking Long-Form and Complex Text-to-Video Generation](#paper-6082)
11. [Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs](#paper-6083)
12. [Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](#paper-6084)
13. [iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](#paper-6085)
14. [CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving](#paper-6086)
15. [MDGMIX: Boundary-Aware Subgraph Mixing for Multi-Domain Graph Pre-Training](#paper-6087)
16. [RGGT: A Generative-Prior-Guided Transformer for Unified Rigid and Non-Rigid Point Cloud Registration](#paper-6088)
17. [Plasticity Activation via Polar Operator: A Plug-in Method for Balancing Stability and Plasticity](#paper-6089)
18. [Temporal-Aware Reasoning Optimization for Video Temporal Grounding](#paper-6090)
19. [Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](#paper-6091)
20. [Rethinking Gating Mechanism in Sparse MoE: Handling Arbitrary Modality Inputs with Confidence-Guided Gate](#paper-6092)
21. [Theoretical Investigation on Inductive Bias of Isolation Forest](#paper-6093)
22. [Context-Aware Reasoner: Enhancing Contextual Reasoning in Multimodal Large Language Models](#paper-6094)
23. [DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs](#paper-6095)
24. [Topology-Preserving Neural Operator Learning via Hodge Decomposition](#paper-6096)
25. [From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](#paper-6097)
26. [SubspacePath Pruner: Inference-time Pruning via Probe-based Representation–Parameter Coupling](#paper-6098)
27. [XDomainBench: Diagnosing Reasoning Collapse in High-Dimensional Scientific Knowledge Composition](#paper-6099)
28. [Learning 3D-Gaussian Simulators from RGB Videos](#paper-6100)
29. [PGD-NO: A Neural Operator with Precomputed Geometry Decomposition for 3D Million-Scale Physics Simulations](#paper-6101)
30. [3D MeanFlow: One-Step Point Cloud Completion and Generation via Average-Velocity Transport](#paper-6102)
31. [Better, Faster: Harnessing Self-Improvement in Large Reasoning Models](#paper-6103)
32. [LightAVSeg: Lightweight Audio-Visual Segmentation](#paper-6104)
33. [Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](#paper-6105)

---


<a id="paper-6073"></a>

## Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding

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

## Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding


### Images

![0f3c7dcbe197d79bf676b877d262a095df68a2b00e5a71da2446ffa82140d057.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/0f3c7dcbe197d79bf676b877d262a095df68a2b00e5a71da2446ffa82140d057.jpg)

![51b8c9a84d5993fe8b85c6930dc257fa8edaaf7d4ff452fbefa4d5f8b15dcded.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/51b8c9a84d5993fe8b85c6930dc257fa8edaaf7d4ff452fbefa4d5f8b15dcded.jpg)

![55b163071b2cbdbc410a474010f7962a3b83a41ce480e2fe12529f0d92cad80c.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/55b163071b2cbdbc410a474010f7962a3b83a41ce480e2fe12529f0d92cad80c.jpg)

![5f8102c4ad3a2778c85c91251fb0e4396da5bedf287059373ba6fd1c8b05f7af.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/5f8102c4ad3a2778c85c91251fb0e4396da5bedf287059373ba6fd1c8b05f7af.jpg)

![80fb7cf468bc547eeaec6867434bb40638c345d73841f4e4fd9e11259e8611f9.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/80fb7cf468bc547eeaec6867434bb40638c345d73841f4e4fd9e11259e8611f9.jpg)

![b401f6a0eba6c2b0b377b4258407115afa8a57ed87b5dd524fb6c2c29e32a769.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/b401f6a0eba6c2b0b377b4258407115afa8a57ed87b5dd524fb6c2c29e32a769.jpg)

![c1d64b32a0a7781e6904089d762390856e065a94b84fbfd48bb77be066049d7d.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/c1d64b32a0a7781e6904089d762390856e065a94b84fbfd48bb77be066049d7d.jpg)

![cf002fb09d8eab51e52e670906af073f5b27a74b282313faea851eed779f7cee.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/cf002fb09d8eab51e52e670906af073f5b27a74b282313faea851eed779f7cee.jpg)

![d5012c1c60201deee1f5920b8e21b0ead4eb6e3d711476e2c2930a6d2743bdd1.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/d5012c1c60201deee1f5920b8e21b0ead4eb6e3d711476e2c2930a6d2743bdd1.jpg)

![de6262b651d78698d575f23a805064d3c17817f5ef261820ab1ca5aff0f0021b.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/de6262b651d78698d575f23a805064d3c17817f5ef261820ab1ca5aff0f0021b.jpg)

![efb79e2f394c3961ef0e69a5bd863d27a9e5b2a92b8ec2a9eaf648bb34eb87c3.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/efb79e2f394c3961ef0e69a5bd863d27a9e5b2a92b8ec2a9eaf648bb34eb87c3.jpg)

![fe496470d6799f8dc7d86afce229b01bfff4cfe80f7c64c68c4b2c825c99c3a1.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/images/fe496470d6799f8dc7d86afce229b01bfff4cfe80f7c64c68c4b2c825c99c3a1.jpg)

### Tables

![1f878ff09c4eee8abb77dca90301a2cc1d0442693cbe8e457a77c889d6411bba.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/1f878ff09c4eee8abb77dca90301a2cc1d0442693cbe8e457a77c889d6411bba.jpg)

![49c203e84b9e58d4ee194d1f1d4f694e23da8536f3c26c996ae2c9e3f78784c3.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/49c203e84b9e58d4ee194d1f1d4f694e23da8536f3c26c996ae2c9e3f78784c3.jpg)

![538956e213f008a88cba24ac998b48bcc995b305a19e61f36e89a29b16ade0d8.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/538956e213f008a88cba24ac998b48bcc995b305a19e61f36e89a29b16ade0d8.jpg)

![6a85834c3b04525e58c63ccace8ea79c3103607374759a5a0af1ea2688441c11.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/6a85834c3b04525e58c63ccace8ea79c3103607374759a5a0af1ea2688441c11.jpg)

![79e3010affe2b49bec7c2d2fb18105da1caec89622c84769c18bc3d233be9401.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/79e3010affe2b49bec7c2d2fb18105da1caec89622c84769c18bc3d233be9401.jpg)

![9f9eb7d497decf0650fe834d896167ba9bfb841688f0fab65a7d079574ec93a8.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/9f9eb7d497decf0650fe834d896167ba9bfb841688f0fab65a7d079574ec93a8.jpg)

![a5efb8f73a412dd85cbcc07a0af6dbe5274e1931d9f35a8effeea5630536efa4.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/a5efb8f73a412dd85cbcc07a0af6dbe5274e1931d9f35a8effeea5630536efa4.jpg)

![db5a74f8483ddf38767c8dfc7f6b57b40efc54e15acc91afa2474c4de05488df.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/db5a74f8483ddf38767c8dfc7f6b57b40efc54e15acc91afa2474c4de05488df.jpg)

![dfb71858a767497352f29d4f9182faa7825388748dbb9e7eb1a6e1a9f365d432.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/dfb71858a767497352f29d4f9182faa7825388748dbb9e7eb1a6e1a9f365d432.jpg)

![fddce64a87a852ec059cb01a4841bf99c064dd8eb87ee97c4eeadff5d900ff4f.jpg](../icml_results/6123_Foresee-to-Ground_%20From%20Predictive%20Temporal%20Perception%20to%20Evidence-Driven%20Reasoning%20for%20Video%20Tempor/tables/fddce64a87a852ec059cb01a4841bf99c064dd8eb87ee97c4eeadff5d900ff4f.jpg)

<a id="paper-6074"></a>

## HilbertA: Hilbert-Curve–Aligned Sparse Attention for 2D Structured Data


### Images

![00f994c4ac562e856eb4b32f4af34e252b5a5e84348e2b3e1cdc36ab91901c41.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/00f994c4ac562e856eb4b32f4af34e252b5a5e84348e2b3e1cdc36ab91901c41.jpg)

![13945ba355db684f7df15b48a7850b30f29b116262cbfe7d11df11d2869abbfd.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/13945ba355db684f7df15b48a7850b30f29b116262cbfe7d11df11d2869abbfd.jpg)

![2b4e6cddb5828d8210addf9097b0a33794d2445d1c788ab2106768ac14ae7550.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/2b4e6cddb5828d8210addf9097b0a33794d2445d1c788ab2106768ac14ae7550.jpg)

![2bde1f1c9f36a7744d23a0bc9173234b255744da47b12a4186b0218fdccec0d2.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/2bde1f1c9f36a7744d23a0bc9173234b255744da47b12a4186b0218fdccec0d2.jpg)

![4ca5696700175a73f2c4a3479695f5c841a39416097ddb6ed99c8e89e564db01.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/4ca5696700175a73f2c4a3479695f5c841a39416097ddb6ed99c8e89e564db01.jpg)

![71187effdb75ea6d256b39d539d8cad651437234ffd2f3bede87eb54965a1e69.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/71187effdb75ea6d256b39d539d8cad651437234ffd2f3bede87eb54965a1e69.jpg)

![71b25f89a5a7e1be103f68c449fcc10d1190c5a55e5436cf44710140df1f72ec.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/71b25f89a5a7e1be103f68c449fcc10d1190c5a55e5436cf44710140df1f72ec.jpg)

![72c9f9020a38854b0412c791f84914788349e0e4c974796052558548dd52ce13.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/72c9f9020a38854b0412c791f84914788349e0e4c974796052558548dd52ce13.jpg)

![826eeb66a64ace48f5b6939bfd2533890150d72ff201efbb745efbf93afcb42c.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/826eeb66a64ace48f5b6939bfd2533890150d72ff201efbb745efbf93afcb42c.jpg)

![9edd21c1329086b1bc114e81fbb7602c9c6b2903fe51e82f06adba4376a0a726.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/9edd21c1329086b1bc114e81fbb7602c9c6b2903fe51e82f06adba4376a0a726.jpg)

![a9ea62d4aa7003abf7b12a04aef9e3645fb166563500b31af281c1009abe869e.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/a9ea62d4aa7003abf7b12a04aef9e3645fb166563500b31af281c1009abe869e.jpg)

![b5c781b4db51d60cc86276cd1f5445812cbcbf52709da49ba4f1fb836b6626fe.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/b5c781b4db51d60cc86276cd1f5445812cbcbf52709da49ba4f1fb836b6626fe.jpg)

![c493c543a72f37213668646cff2d0f61151df4a0d43abfaacc21f703a8628674.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/c493c543a72f37213668646cff2d0f61151df4a0d43abfaacc21f703a8628674.jpg)

![cfc73122d4dbd8aa7f18079ab51b4c64aae180a2fedcb53b056f83ba16d53a4a.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/cfc73122d4dbd8aa7f18079ab51b4c64aae180a2fedcb53b056f83ba16d53a4a.jpg)

![ec9f2f071ab373ee19220d90b51a7be62c41f9b6d4f7d49cc6bf46bc7c9f1328.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/ec9f2f071ab373ee19220d90b51a7be62c41f9b6d4f7d49cc6bf46bc7c9f1328.jpg)

![f50b38e103bfb4a8b4e3c431af216e9f48566aa06679160741a5cf3abecd86d4.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/images/f50b38e103bfb4a8b4e3c431af216e9f48566aa06679160741a5cf3abecd86d4.jpg)

### Tables

![07f9e6fd3fad01624057e17f3699ae81852b065ac0acf7f6edbe09f7d7202f4d.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/07f9e6fd3fad01624057e17f3699ae81852b065ac0acf7f6edbe09f7d7202f4d.jpg)

![162096283e83d8f1bd239d375db5cc3f884561209a4a479b4c117c915b9610f7.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/162096283e83d8f1bd239d375db5cc3f884561209a4a479b4c117c915b9610f7.jpg)

![2e34e4d89ed800a7598a7f2e69f260927e35cfa6263b747a46047e7795527d77.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/2e34e4d89ed800a7598a7f2e69f260927e35cfa6263b747a46047e7795527d77.jpg)

![2ec6b10da8f38abf8e03f38e3b06c792fe4bf867fa97ad58cbfbe35694b419c9.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/2ec6b10da8f38abf8e03f38e3b06c792fe4bf867fa97ad58cbfbe35694b419c9.jpg)

![a34acf9c6d293b9288f9e8f1bb404f70d91efc1a9850635b9f98c0278167588c.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/a34acf9c6d293b9288f9e8f1bb404f70d91efc1a9850635b9f98c0278167588c.jpg)

![a4824e38bbd49e520634b3e3fa2e9bfc3c575e9d1cd9e537aeb333e7aa9d755a.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/a4824e38bbd49e520634b3e3fa2e9bfc3c575e9d1cd9e537aeb333e7aa9d755a.jpg)

![b92384158014b7c820a33e014fa26adb5fe0e940467d675b284c62bf6ddcac23.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/b92384158014b7c820a33e014fa26adb5fe0e940467d675b284c62bf6ddcac23.jpg)

![be09424730ff2d35e7cebfa99d504002715a618801b434d5c533b133161a7af7.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/be09424730ff2d35e7cebfa99d504002715a618801b434d5c533b133161a7af7.jpg)

![c73dd1fc9c970ddc7492838e682f949efe902f55740d721083614b323427d768.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/c73dd1fc9c970ddc7492838e682f949efe902f55740d721083614b323427d768.jpg)

![e90b1786a9f1e3108dbb8f912b6ac2042e7241ad9ef1ad7a26f95efd6a7f0748.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/e90b1786a9f1e3108dbb8f912b6ac2042e7241ad9ef1ad7a26f95efd6a7f0748.jpg)

![ffb7f3886bbc634a8b0588427cc792e8e55d98fc47527c9a1095fdbb0ed51f26.jpg](../icml_results/6124_HilbertA_%20Hilbert-Curve%E2%80%93Aligned%20Sparse%20Attention%20for%202D%20Structured%20Data/tables/ffb7f3886bbc634a8b0588427cc792e8e55d98fc47527c9a1095fdbb0ed51f26.jpg)

<a id="paper-6075"></a>

## Overcoming the Modality Gap in Context-Aided Forecasting


### Images

![03fce0beda8dc458d8445367e43de20545a6d7d94ce3e8b6101e95ea08699bf6.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/03fce0beda8dc458d8445367e43de20545a6d7d94ce3e8b6101e95ea08699bf6.jpg)

![0832ff82810ec7c470c0e1e380140208098f5fdb2beddf51ecadaae61a436d04.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/0832ff82810ec7c470c0e1e380140208098f5fdb2beddf51ecadaae61a436d04.jpg)

![09112d5fa6c5e52ba5942fbedc277fed8c2729ed5ea14832b297e72278d84135.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/09112d5fa6c5e52ba5942fbedc277fed8c2729ed5ea14832b297e72278d84135.jpg)

![0b29dacffa7e9c805860a1067482aaaf1e9fb121414eff3ccbc6b07a6dd529f8.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/0b29dacffa7e9c805860a1067482aaaf1e9fb121414eff3ccbc6b07a6dd529f8.jpg)

![0ddbfbf73cecb85ebe9f9005dbbd58c1c8af9173dede744e6f871ce4e2a0b713.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/0ddbfbf73cecb85ebe9f9005dbbd58c1c8af9173dede744e6f871ce4e2a0b713.jpg)

![123bc5af4399ffdcc2569ebd3e520b82dda47a673fb648d609688616e2cea5c4.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/123bc5af4399ffdcc2569ebd3e520b82dda47a673fb648d609688616e2cea5c4.jpg)

![241b099df6a03ca20eba5409fbb1768a2264c6f338b4917d19ea607fd888d844.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/241b099df6a03ca20eba5409fbb1768a2264c6f338b4917d19ea607fd888d844.jpg)

![26e74e70723545a2511c4e5420e5bb35f5b1be3df1a1188d11f92ede40edcb0d.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/26e74e70723545a2511c4e5420e5bb35f5b1be3df1a1188d11f92ede40edcb0d.jpg)

![29e5515c6c86cd04e0c929616d010047fd52afb876280e34bffc5e6ed798cbcc.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/29e5515c6c86cd04e0c929616d010047fd52afb876280e34bffc5e6ed798cbcc.jpg)

![2ef03160b28ba6ba5271a1f4ae8a557923f77edcabbb0eccf8d3a3ccfa8c86c8.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/2ef03160b28ba6ba5271a1f4ae8a557923f77edcabbb0eccf8d3a3ccfa8c86c8.jpg)

![371064c057b8426d8f25f0777be2cb13dc5480e69a67918ee92d6c9891b41215.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/371064c057b8426d8f25f0777be2cb13dc5480e69a67918ee92d6c9891b41215.jpg)

![3cc64d4371b012045ca8f450da6787e664c0a7a84a0f68b4ba83d339564c840a.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/3cc64d4371b012045ca8f450da6787e664c0a7a84a0f68b4ba83d339564c840a.jpg)

![3e0d2661a82ca7c1498a7a6d336150e86f3fcb82db924f4d3527a62db26a6b09.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/3e0d2661a82ca7c1498a7a6d336150e86f3fcb82db924f4d3527a62db26a6b09.jpg)

![3e42fe74971a97242690ad8e470ea09faa15bdfb0ad2b5cca4c9c0318b77d8dc.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/3e42fe74971a97242690ad8e470ea09faa15bdfb0ad2b5cca4c9c0318b77d8dc.jpg)

![45e05a598d4dcddd7777ac9c02b03d46627b7c911db94cd13129e2a8da644d5e.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/45e05a598d4dcddd7777ac9c02b03d46627b7c911db94cd13129e2a8da644d5e.jpg)

![469e7d0874fa65213606a426aea1e27430e6c285824d7a6ec07c99c7599b32d2.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/469e7d0874fa65213606a426aea1e27430e6c285824d7a6ec07c99c7599b32d2.jpg)

![48d1ee341c31510072266315b5885bd05d4d348f7079b77b3235e9598cc48143.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/48d1ee341c31510072266315b5885bd05d4d348f7079b77b3235e9598cc48143.jpg)

![52e4bab0b96e3624ff36746fb46d6ab49201605a4bb92ed139b7d35ead8163c1.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/52e4bab0b96e3624ff36746fb46d6ab49201605a4bb92ed139b7d35ead8163c1.jpg)

![59b38073977b0e7dc1ea263c5ec7a6255e8cbf351fd91776ae7e26b2111b3794.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/59b38073977b0e7dc1ea263c5ec7a6255e8cbf351fd91776ae7e26b2111b3794.jpg)

![618704b13c254baed3bc23b5670908b472e55b46f471d5e5ef2cb14ae1b37e30.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/618704b13c254baed3bc23b5670908b472e55b46f471d5e5ef2cb14ae1b37e30.jpg)

![6406f7f66f42daf90b6bf9868ae498252c8ccf538e94e99b016058bbc642e827.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/6406f7f66f42daf90b6bf9868ae498252c8ccf538e94e99b016058bbc642e827.jpg)

![643640bbdec1e1e1fd52e4389805833f9574477ddd20d7336d4d0006dace31c7.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/643640bbdec1e1e1fd52e4389805833f9574477ddd20d7336d4d0006dace31c7.jpg)

![64f2dc09e4c06d567129249e078d79462412e076ffce70fc7ea727e4c7cfa6ee.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/64f2dc09e4c06d567129249e078d79462412e076ffce70fc7ea727e4c7cfa6ee.jpg)

![68a5573b0f378377a294eb4c6c2a1d8531a3c52f87ed56a2e6e94fc4341c3fde.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/68a5573b0f378377a294eb4c6c2a1d8531a3c52f87ed56a2e6e94fc4341c3fde.jpg)

![816874e2853184bec87515380bb1ec8d371d431865036b9728efae02c7ac868f.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/816874e2853184bec87515380bb1ec8d371d431865036b9728efae02c7ac868f.jpg)

![83ba270a48f46c55bb56449e3681e92951678c77d8a344fdb424c51465f623c5.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/83ba270a48f46c55bb56449e3681e92951678c77d8a344fdb424c51465f623c5.jpg)

![867329d8a871ea9f49360937edad51fb600b7f5b9e3f5887495ce1b6ad5ec72a.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/867329d8a871ea9f49360937edad51fb600b7f5b9e3f5887495ce1b6ad5ec72a.jpg)

![8785db50191c10aae3b11a7ab2d3b622249054311cc7345d3714ba9c179575b4.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/8785db50191c10aae3b11a7ab2d3b622249054311cc7345d3714ba9c179575b4.jpg)

![8d1e9cceb6740bcc172fdfce5e6d588973416ba1e418410eb3827f0fa4188758.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/8d1e9cceb6740bcc172fdfce5e6d588973416ba1e418410eb3827f0fa4188758.jpg)

![9347611fce61d04a96f7ba5dda6993c3361f92b6d881c31aa35a7fa906ffb5b3.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/9347611fce61d04a96f7ba5dda6993c3361f92b6d881c31aa35a7fa906ffb5b3.jpg)

![937eda575a3dd30c9e3f04370966dd7641147068d60401fdf0dc2fa88aa5d894.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/937eda575a3dd30c9e3f04370966dd7641147068d60401fdf0dc2fa88aa5d894.jpg)

![97d13b794947e816f36572a4ef9134e9db032fc80ca8da6846f3bb0242970a94.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/97d13b794947e816f36572a4ef9134e9db032fc80ca8da6846f3bb0242970a94.jpg)

![9b54e9dfb6662360d54b7aac2cd8cf8cbb1782aa3a09a755f86e3a4d1cbb7f48.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/9b54e9dfb6662360d54b7aac2cd8cf8cbb1782aa3a09a755f86e3a4d1cbb7f48.jpg)

![9d9439f657a04f27b8eb571e4cad0c2274dee1e696ed3e1b7878b3ca85eae955.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/9d9439f657a04f27b8eb571e4cad0c2274dee1e696ed3e1b7878b3ca85eae955.jpg)

![9faa38e97b035a2ce61bf0bb3af46135a0072da040db137e93c690d609355590.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/9faa38e97b035a2ce61bf0bb3af46135a0072da040db137e93c690d609355590.jpg)

![a0fca140601c8f0e67eb985b3addd024a7e7d7afe581bd0fd290f1734b1647c2.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/a0fca140601c8f0e67eb985b3addd024a7e7d7afe581bd0fd290f1734b1647c2.jpg)

![abaf318bd4cc8e4017cc0e8c919106466a9249d4adf21b1a771da39e9ae1d836.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/abaf318bd4cc8e4017cc0e8c919106466a9249d4adf21b1a771da39e9ae1d836.jpg)

![abe1b7d2a73d18a3f61b0a7c129c6faefa183ab165fd634fc5970683bd2f2513.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/abe1b7d2a73d18a3f61b0a7c129c6faefa183ab165fd634fc5970683bd2f2513.jpg)

![ae3b4666683161f349b0c922c5a9ea6dff6b9094f6c5a1bd3f60799f8456f661.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/ae3b4666683161f349b0c922c5a9ea6dff6b9094f6c5a1bd3f60799f8456f661.jpg)

![b26fac98c58260595464837c1853e4850a38b4954eec1525a142153c21c7f010.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/b26fac98c58260595464837c1853e4850a38b4954eec1525a142153c21c7f010.jpg)

![b76205a267e11842162c3b561c4dfb3cb1da8994e2a27194ee067a8403b78bf8.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/b76205a267e11842162c3b561c4dfb3cb1da8994e2a27194ee067a8403b78bf8.jpg)

![b8dee9cabedf672733452f09c44371dbac9c665412d2666f90c86f50e9073ade.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/b8dee9cabedf672733452f09c44371dbac9c665412d2666f90c86f50e9073ade.jpg)

![be5651b669f090a37b929e18d32419b0b3a52d1b8be071e72f40107c4520fb3e.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/be5651b669f090a37b929e18d32419b0b3a52d1b8be071e72f40107c4520fb3e.jpg)

![ce9185b38857737ebd37ffcaf6fdd43a0fbe994b18866862c179264db5bfdac2.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/ce9185b38857737ebd37ffcaf6fdd43a0fbe994b18866862c179264db5bfdac2.jpg)

![d43c88fcb31399a2a9583a31a3863d0a4adf90c9ad87cfe4d1c93fdc79911dbe.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/d43c88fcb31399a2a9583a31a3863d0a4adf90c9ad87cfe4d1c93fdc79911dbe.jpg)

![def8845db237ecf1a6789c4cd42ab935623745fbaacee9c8c84c534d6fc1e274.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/def8845db237ecf1a6789c4cd42ab935623745fbaacee9c8c84c534d6fc1e274.jpg)

![e354efe6f2569063c4c9d3c8a28119c9d65e254aeabb7256a29a61f0b18720ca.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/e354efe6f2569063c4c9d3c8a28119c9d65e254aeabb7256a29a61f0b18720ca.jpg)

![ee80a828312b5745803326d299dc9259080cb7a2cc9967a5b95ce0885afcf862.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/ee80a828312b5745803326d299dc9259080cb7a2cc9967a5b95ce0885afcf862.jpg)

![f1b242628dec1f597eac885bca867ded5ee0aafb736d9d18bcd16e3b122c6ac1.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/f1b242628dec1f597eac885bca867ded5ee0aafb736d9d18bcd16e3b122c6ac1.jpg)

![f38094da13fbccd5d4c6cf36391c393df040665039aafbf5be6cb673d028fe52.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/f38094da13fbccd5d4c6cf36391c393df040665039aafbf5be6cb673d028fe52.jpg)

![f81085042d23db1640c18c8e3e11e4fd7b29b9a21da4cfd9c2c7d694fac7429b.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/f81085042d23db1640c18c8e3e11e4fd7b29b9a21da4cfd9c2c7d694fac7429b.jpg)

![fac3ed49a1bcceec737998dfcfdac1999cddf8147bf2a7b81b666931d6c6497d.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/fac3ed49a1bcceec737998dfcfdac1999cddf8147bf2a7b81b666931d6c6497d.jpg)

![fac58698702684e9b61e665659b9a8800ce199ec272e2e39cfd8047505ed0603.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/fac58698702684e9b61e665659b9a8800ce199ec272e2e39cfd8047505ed0603.jpg)

![fe14c37ca2ba9fa6cc20daf7888445e46226c3f2f32df08ea3273b746c3ae43b.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/images/fe14c37ca2ba9fa6cc20daf7888445e46226c3f2f32df08ea3273b746c3ae43b.jpg)

### Tables

![081573c20a6740706ca49aef26923f25bc2ff7f1bfa79518ae269a03bafcf191.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/081573c20a6740706ca49aef26923f25bc2ff7f1bfa79518ae269a03bafcf191.jpg)

![12d73ea0d3bfc270e56604a0f1ef1584f8c4bc54b3c304ae2097f951cae383ae.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/12d73ea0d3bfc270e56604a0f1ef1584f8c4bc54b3c304ae2097f951cae383ae.jpg)

![4030c878a7b24dc94482ab4f703607b00bab709dd9f233f0e0c155aa9d4ce05e.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/4030c878a7b24dc94482ab4f703607b00bab709dd9f233f0e0c155aa9d4ce05e.jpg)

![41a794dbc103a76cd49e29ea56dcb879b71d78b689e79269767b0aba581dc531.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/41a794dbc103a76cd49e29ea56dcb879b71d78b689e79269767b0aba581dc531.jpg)

![5475e319e034c4510944a9532d6a425c371ce4fa3e985a12cf94f392d4d4e4cf.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/5475e319e034c4510944a9532d6a425c371ce4fa3e985a12cf94f392d4d4e4cf.jpg)

![55bc79f8035e1317bf594ed2f67c4c133881bed61c11de9bbce14d10c8208c3d.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/55bc79f8035e1317bf594ed2f67c4c133881bed61c11de9bbce14d10c8208c3d.jpg)

![6e9923e38448ecfdce91e4ee62d3c95c47b0beeb4e84cc3d3a91f5a4fc418ae2.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/6e9923e38448ecfdce91e4ee62d3c95c47b0beeb4e84cc3d3a91f5a4fc418ae2.jpg)

![b2623ea48820d8e693d686d4edaebbd23e2d76ec4ac94d12c8364159ef6ae688.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/b2623ea48820d8e693d686d4edaebbd23e2d76ec4ac94d12c8364159ef6ae688.jpg)

![c9a0dbc5e6056357891735fbb5a5d4a47cf3ba65df33977a4d24ced932826937.jpg](../icml_results/6125_Overcoming%20the%20Modality%20Gap%20in%20Context-Aided%20Forecasting/tables/c9a0dbc5e6056357891735fbb5a5d4a47cf3ba65df33977a4d24ced932826937.jpg)

<a id="paper-6076"></a>

## Mosaic: Unlocking Over 30$\times$ Context Length for Diffusion LLMs Inference via Global Memory Planning and Dynamic Peak Taming


### Images

![09e449af5dcfcaf2ae03aa9137f3f4bb9309c77ab77526b0a9d2aaad13128a90.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/09e449af5dcfcaf2ae03aa9137f3f4bb9309c77ab77526b0a9d2aaad13128a90.jpg)

![3b51997802cede2287e55225b24868c6b5178a2a443a16278b393e16d4d87e29.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/3b51997802cede2287e55225b24868c6b5178a2a443a16278b393e16d4d87e29.jpg)

![4750399bc0b338329d46d5b083aa1e362ca0e700da3c581cdfe2967226796b8b.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/4750399bc0b338329d46d5b083aa1e362ca0e700da3c581cdfe2967226796b8b.jpg)

![759926032955da0279c0ae738347b7ad6af75de9dbe29dd9fc0c021168b8791b.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/759926032955da0279c0ae738347b7ad6af75de9dbe29dd9fc0c021168b8791b.jpg)

![799d7bbfacce75919fa080f055bd4b2d7557fd21dfe01e033c971999b0926be9.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/799d7bbfacce75919fa080f055bd4b2d7557fd21dfe01e033c971999b0926be9.jpg)

![8a70c8be7f654072dbcea1f9761cf75c464c67a76cd72ba789f24e839389bcc8.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/8a70c8be7f654072dbcea1f9761cf75c464c67a76cd72ba789f24e839389bcc8.jpg)

![92d6898aded798711fc745bce0f4992c41616bbb1cd1c4788ffd5bb129b98cb8.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/92d6898aded798711fc745bce0f4992c41616bbb1cd1c4788ffd5bb129b98cb8.jpg)

![9a7cf81b1d713f869c325ab8bc921c831f0644a3e0c5b0182919ce095cd9ceff.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/9a7cf81b1d713f869c325ab8bc921c831f0644a3e0c5b0182919ce095cd9ceff.jpg)

![9dd58139c3b6b4003415f1114020fce8a0252589d3137d87e49e777c9c79bf93.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/9dd58139c3b6b4003415f1114020fce8a0252589d3137d87e49e777c9c79bf93.jpg)

![a5a6ba9a934c4fb971198973998637fa9d017bcb12ba2189bf45e9e90037a360.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/a5a6ba9a934c4fb971198973998637fa9d017bcb12ba2189bf45e9e90037a360.jpg)

![b11211f98a8a4eab08c48f961cdcaaa27d3d55e54dd382952aa903a410e99e43.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/b11211f98a8a4eab08c48f961cdcaaa27d3d55e54dd382952aa903a410e99e43.jpg)

![c280719278bcd91f35f6acc79a6756700c2f2a42baeebb2109065525a515f03a.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/c280719278bcd91f35f6acc79a6756700c2f2a42baeebb2109065525a515f03a.jpg)

![c2dcdfb8cc251868541379adea08c21b6418e095cd01491df1dbb3f96e505da4.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/c2dcdfb8cc251868541379adea08c21b6418e095cd01491df1dbb3f96e505da4.jpg)

![c8f8d34903d1d491e950ca88771ed79fd88b8eaa97e22a6d89d192f91fcb7ea0.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/c8f8d34903d1d491e950ca88771ed79fd88b8eaa97e22a6d89d192f91fcb7ea0.jpg)

![e353714d088575e54b98ddc042b9855ac0b6fe5c258900401af4e348c0bb359b.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/e353714d088575e54b98ddc042b9855ac0b6fe5c258900401af4e348c0bb359b.jpg)

![f925f82aa6650abf1fd82e84c6d2b9e6e105ad9936b24337fe30d7984e9f8d48.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/f925f82aa6650abf1fd82e84c6d2b9e6e105ad9936b24337fe30d7984e9f8d48.jpg)

![ffca4f7e8c4c906d397645243f39255faaf8b96d12b2baa4dc28739d2e03c445.jpg](../icml_results/6126_Mosaic_%20Unlocking%20Over%2030%24_times%24%20Context%20Length%20for%20Diffusion%20LLMs%20Inference%20via%20Global%20Memory%20Plan/images/ffca4f7e8c4c906d397645243f39255faaf8b96d12b2baa4dc28739d2e03c445.jpg)

<a id="paper-6077"></a>

## Solving Imperfect-Recall Games via Sum-of-Squares Optimization


### Images

![1d3b4583903b189bd619d9a2c2b4da569d4e4cc48cd3ec66a6f7975888d85829.jpg](../icml_results/6127_Solving%20Imperfect-Recall%20Games%20via%20Sum-of-Squares%20Optimization/images/1d3b4583903b189bd619d9a2c2b4da569d4e4cc48cd3ec66a6f7975888d85829.jpg)

![d6128741f6af968c60f02bdf1990dd0089810fe6a8482c33b9ffd27b775dea73.jpg](../icml_results/6127_Solving%20Imperfect-Recall%20Games%20via%20Sum-of-Squares%20Optimization/images/d6128741f6af968c60f02bdf1990dd0089810fe6a8482c33b9ffd27b775dea73.jpg)

### Tables

![956b4f801ff05c4dda4ec75cf0ab3d4037efa8cbd721206f61d8d23fa6f3a316.jpg](../icml_results/6127_Solving%20Imperfect-Recall%20Games%20via%20Sum-of-Squares%20Optimization/tables/956b4f801ff05c4dda4ec75cf0ab3d4037efa8cbd721206f61d8d23fa6f3a316.jpg)

![ca11a8a977268655b9e0d9e6aa68483909bec4ca036f3d58f4d9d7362efd9d8f.jpg](../icml_results/6127_Solving%20Imperfect-Recall%20Games%20via%20Sum-of-Squares%20Optimization/tables/ca11a8a977268655b9e0d9e6aa68483909bec4ca036f3d58f4d9d7362efd9d8f.jpg)

![fce798674800b5f60fd142e8a631415226e2fe87f67307d270e063659a6b59bb.jpg](../icml_results/6127_Solving%20Imperfect-Recall%20Games%20via%20Sum-of-Squares%20Optimization/tables/fce798674800b5f60fd142e8a631415226e2fe87f67307d270e063659a6b59bb.jpg)

<a id="paper-6078"></a>

## SPADA: A Verifiable Test-Driven Agent for Controllable Parametric CAD Assembly Generation


### Images

![20a9363071a03d7aac889bbddcd4c1f03998c9d7bc68092f93aa2c8a82f94e8b.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/20a9363071a03d7aac889bbddcd4c1f03998c9d7bc68092f93aa2c8a82f94e8b.jpg)

![45643b42a3c8e5664e685bd8f2e1258fc4a53ec274c8b3ea9e11f87efcc59697.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/45643b42a3c8e5664e685bd8f2e1258fc4a53ec274c8b3ea9e11f87efcc59697.jpg)

![4f7b8379e1a6c13e66bdddaa7611d7fa29cbba745c405948944f24362304e604.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/4f7b8379e1a6c13e66bdddaa7611d7fa29cbba745c405948944f24362304e604.jpg)

![5daee2c2769a9c66e31e1692e320f53bcf002f10b1428b51be8796967f020f28.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/5daee2c2769a9c66e31e1692e320f53bcf002f10b1428b51be8796967f020f28.jpg)

![7bfcd3f9367c45d00837c83b3710d9e0c143940787bb325d77ec99f7a90b4f3e.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/7bfcd3f9367c45d00837c83b3710d9e0c143940787bb325d77ec99f7a90b4f3e.jpg)

![7cf5619d5afdcfb23dfd194410a8790ce33174ecf1abb3d8920f0ad71d268a65.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/7cf5619d5afdcfb23dfd194410a8790ce33174ecf1abb3d8920f0ad71d268a65.jpg)

![8162db91abf25b794aa82f42d79a25c79b345c4a64bb37b2a536155e86d164f3.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/8162db91abf25b794aa82f42d79a25c79b345c4a64bb37b2a536155e86d164f3.jpg)

![9a30e9ffdead8d1aa2730fbf9d67f2302e33e4371b1c2daad9c40bc6a78aff60.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/9a30e9ffdead8d1aa2730fbf9d67f2302e33e4371b1c2daad9c40bc6a78aff60.jpg)

![a465133a2ad3a429d8c0058f8b2e864b3bc82ed83bc1d0ca54dda94591df6f3a.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/a465133a2ad3a429d8c0058f8b2e864b3bc82ed83bc1d0ca54dda94591df6f3a.jpg)

![cafff163a5a01d8bcd60feb7c659cf6d4a4956c97aa2e81f587b18ed48eb635e.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/cafff163a5a01d8bcd60feb7c659cf6d4a4956c97aa2e81f587b18ed48eb635e.jpg)

![e94d2a70399188377d0970d5618efaa3ac12526e7041d8632b19f62ab837ada6.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/images/e94d2a70399188377d0970d5618efaa3ac12526e7041d8632b19f62ab837ada6.jpg)

### Tables

![05a66f71e7e197aa0bcbc3cbb27e77aee7533351e70390b6f14e19180d5dedb5.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/05a66f71e7e197aa0bcbc3cbb27e77aee7533351e70390b6f14e19180d5dedb5.jpg)

![132c47241e2322c49f1ac63212d97340ee2bd52fb4acd9ff2295fabd90802205.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/132c47241e2322c49f1ac63212d97340ee2bd52fb4acd9ff2295fabd90802205.jpg)

![19fd1dc37f81e26f3014ac4a849101bb859a7d078b10f8822371c459b2c6026e.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/19fd1dc37f81e26f3014ac4a849101bb859a7d078b10f8822371c459b2c6026e.jpg)

![35d4644baeec19d60088dd86799d63fb79e829414e3d85b274c508af371b92dc.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/35d4644baeec19d60088dd86799d63fb79e829414e3d85b274c508af371b92dc.jpg)

![6101dd58ecfec2bb4e879b5359917e2a7219601e57bfaccf5515bd91bfbdacaf.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/6101dd58ecfec2bb4e879b5359917e2a7219601e57bfaccf5515bd91bfbdacaf.jpg)

![84ab369324fc399c78531c5058b6137af1f45f270e0aa37dc9980461ae656c97.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/84ab369324fc399c78531c5058b6137af1f45f270e0aa37dc9980461ae656c97.jpg)

![8df86a0df89057165e483ff1cb5594a68723fc292e85898183711531f5804515.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/8df86a0df89057165e483ff1cb5594a68723fc292e85898183711531f5804515.jpg)

![90a1b5ce557715a23c97da85cf18e4ef65342d86e63c4e95a43b78d7298dc962.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/90a1b5ce557715a23c97da85cf18e4ef65342d86e63c4e95a43b78d7298dc962.jpg)

![978ce1cf78ed472208927a856cc24976e8681be3d95199782176c5300207ff81.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/978ce1cf78ed472208927a856cc24976e8681be3d95199782176c5300207ff81.jpg)

![c40711627c96883b6c6f2bba08f184c49e68c1550571e2f711f632629f4ac748.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/c40711627c96883b6c6f2bba08f184c49e68c1550571e2f711f632629f4ac748.jpg)

![ce5936c4e9d859aeea64fb5d1f319e8bdcbc421f07e7d84a8fcbcbcf9c598e86.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/ce5936c4e9d859aeea64fb5d1f319e8bdcbc421f07e7d84a8fcbcbcf9c598e86.jpg)

![e02f5add30e056b1c1f6ed9c591fb6fa7874fcaaced4e656adf7836568abfde0.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/e02f5add30e056b1c1f6ed9c591fb6fa7874fcaaced4e656adf7836568abfde0.jpg)

![e06770c611d309f6c1c2fcf46d322883796e1a259912e48007a1f626df41a1b5.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/e06770c611d309f6c1c2fcf46d322883796e1a259912e48007a1f626df41a1b5.jpg)

![f13c9ff3436b5ca2b0d6a8bc42087c319ee951035bdee3d56a89960ba4d4af90.jpg](../icml_results/6128_SPADA_%20A%20Verifiable%20Test-Driven%20Agent%20for%20Controllable%20Parametric%20CAD%20Assembly%20Generation/tables/f13c9ff3436b5ca2b0d6a8bc42087c319ee951035bdee3d56a89960ba4d4af90.jpg)

<a id="paper-6079"></a>

## Spatio-Temporal LLM: Reasoning about Environments and Actions


### Images

![1f721821fc4dc5b4f39ca1fc9c8cb04a53f5ae51f615deea68513d5ebd7b14ca.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/1f721821fc4dc5b4f39ca1fc9c8cb04a53f5ae51f615deea68513d5ebd7b14ca.jpg)

![212bfe63ab0ec31a0235ae287f3a4220a8792a277acc233c38d4f424efd01472.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/212bfe63ab0ec31a0235ae287f3a4220a8792a277acc233c38d4f424efd01472.jpg)

![29b7b6f448a7c6d7bfee1b7c93333a0033645eda2ad9522c1afc0e99fcbe7572.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/29b7b6f448a7c6d7bfee1b7c93333a0033645eda2ad9522c1afc0e99fcbe7572.jpg)

![554214f1ba9a4770da0c39ae00883fbb5eeebc13ac5847d3e5153e90ac0be6d8.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/554214f1ba9a4770da0c39ae00883fbb5eeebc13ac5847d3e5153e90ac0be6d8.jpg)

![577b6a7994a2f4e7aded9b070f979fe3613b75aabf3e917c4781324322213d5f.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/577b6a7994a2f4e7aded9b070f979fe3613b75aabf3e917c4781324322213d5f.jpg)

![6d8c5ec232b2e40df13a27fc63cd9b151d9e1a9316b915dca15d82b66f4bd16b.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/6d8c5ec232b2e40df13a27fc63cd9b151d9e1a9316b915dca15d82b66f4bd16b.jpg)

![7e2ca69733738468da3e529970fc3900a965d27977c45bc98b6a48664589d517.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/7e2ca69733738468da3e529970fc3900a965d27977c45bc98b6a48664589d517.jpg)

![9a8d4cebf69cf84c8a9354464247506b3d567aad910c919e268c96c9e339f91e.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/9a8d4cebf69cf84c8a9354464247506b3d567aad910c919e268c96c9e339f91e.jpg)

![bae5ef8b8d218943c3db6072434fab73e83625a1ef73fdd02320ab3f454f5653.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/bae5ef8b8d218943c3db6072434fab73e83625a1ef73fdd02320ab3f454f5653.jpg)

![c325ba4f44320b6f61f591c6e7c3e11b2b2bec7f2adcf8da463fae10155bfdd0.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/c325ba4f44320b6f61f591c6e7c3e11b2b2bec7f2adcf8da463fae10155bfdd0.jpg)

![c94ceb5941db8f0bc0bcbdb2034f6682d9cb8b0701c2ba809510ca79325b98c2.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/images/c94ceb5941db8f0bc0bcbdb2034f6682d9cb8b0701c2ba809510ca79325b98c2.jpg)

### Tables

![013144b2b03b0f607ab1760e4e3bdde595dd011dc4e1d48cf8871c83bf1ea1dc.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/013144b2b03b0f607ab1760e4e3bdde595dd011dc4e1d48cf8871c83bf1ea1dc.jpg)

![10b93007ce73b916af22db8c7be1df70bf7a02356c80d2e30babf06ed43c1e9f.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/10b93007ce73b916af22db8c7be1df70bf7a02356c80d2e30babf06ed43c1e9f.jpg)

![11e5962546f881f72003c084fbff567f323c2355b2651aaff3dda11480aa3a6c.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/11e5962546f881f72003c084fbff567f323c2355b2651aaff3dda11480aa3a6c.jpg)

![167cc39fb9a98e52741dc0b10f0347d10405bc86785206dda1e38a6eb8b24d36.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/167cc39fb9a98e52741dc0b10f0347d10405bc86785206dda1e38a6eb8b24d36.jpg)

![1714a18891a7ef30de26c21f56e09ffa1460a46f42e7a824869dd42357ec5777.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/1714a18891a7ef30de26c21f56e09ffa1460a46f42e7a824869dd42357ec5777.jpg)

![503c25b32553e3abc44b189b4af26a2fc2aeaaa7c2eb8f024ad26084d207d70a.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/503c25b32553e3abc44b189b4af26a2fc2aeaaa7c2eb8f024ad26084d207d70a.jpg)

![88f6a25b4bac926f53fe0189f73d0d8cb1bfc693eb1106c1bdc68e0d76e8c9be.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/88f6a25b4bac926f53fe0189f73d0d8cb1bfc693eb1106c1bdc68e0d76e8c9be.jpg)

![920a73f480592b1f274289373932ab1dc1039906222d3a0ce3f26db058777430.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/920a73f480592b1f274289373932ab1dc1039906222d3a0ce3f26db058777430.jpg)

![9aeea3fb560f6afdd6f12b7e0abc2de8fea6c0740e801768a735dc0787388961.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/9aeea3fb560f6afdd6f12b7e0abc2de8fea6c0740e801768a735dc0787388961.jpg)

![a4324d26100eefdd745ddcbb52663435fc3743553d83b0acfd47d4b7d15568a3.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/a4324d26100eefdd745ddcbb52663435fc3743553d83b0acfd47d4b7d15568a3.jpg)

![c1171e214d99436e4f5b90db483fc442499e1b230a561e89ba4c3efe9a5993c5.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/c1171e214d99436e4f5b90db483fc442499e1b230a561e89ba4c3efe9a5993c5.jpg)

![c5af28409ce13e2d7d5576579bdd56ae94754a05e2ca43ad787cd73e86122022.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/c5af28409ce13e2d7d5576579bdd56ae94754a05e2ca43ad787cd73e86122022.jpg)

![d266cb4750a75abb513b00b81916f2e73d22e1c51244a4c35182af0dab46e847.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/d266cb4750a75abb513b00b81916f2e73d22e1c51244a4c35182af0dab46e847.jpg)

![d6ac7783c54dba3a6aacc99c5c253eda2492c6e10f98f9bf8087608633c05262.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/d6ac7783c54dba3a6aacc99c5c253eda2492c6e10f98f9bf8087608633c05262.jpg)

![dbabeee981bb98e1a80db128c630ad15bb076c5734014abea8c25fe22f3d4d14.jpg](../icml_results/6129_Spatio-Temporal%20LLM_%20Reasoning%20about%20Environments%20and%20Actions/tables/dbabeee981bb98e1a80db128c630ad15bb076c5734014abea8c25fe22f3d4d14.jpg)

<a id="paper-6080"></a>

## Unsupervised Process-Aware Coreset Selection for In-Context Learning


### Images

![23aabfd7fd6044775991b06f3563c1b58f7428e8d4e17839028aa55386231353.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/23aabfd7fd6044775991b06f3563c1b58f7428e8d4e17839028aa55386231353.jpg)

![47e6e61c6000ea4b4015bc89f3585e8af303a2c531f774053b21fc52c0b4331f.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/47e6e61c6000ea4b4015bc89f3585e8af303a2c531f774053b21fc52c0b4331f.jpg)

![64e0710e5a841ea6d084e740b94ce22be6571911c3e7f38338b11a371fdd4ade.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/64e0710e5a841ea6d084e740b94ce22be6571911c3e7f38338b11a371fdd4ade.jpg)

![9040cb8007de341c5ac7f69acce8ff679f93d3aee6c062ef2f47d34656088827.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/9040cb8007de341c5ac7f69acce8ff679f93d3aee6c062ef2f47d34656088827.jpg)

![b117b7d569322ae9dcb91b54920dbde3e3eaf8ce29d3effabd70e6851acfddfd.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/b117b7d569322ae9dcb91b54920dbde3e3eaf8ce29d3effabd70e6851acfddfd.jpg)

![b638417006b3b504fbe0269764b988c46677106bfa45679639d20a0a46c90a8b.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/images/b638417006b3b504fbe0269764b988c46677106bfa45679639d20a0a46c90a8b.jpg)

### Tables

![4efd0441a96131c1e6e7ea2a9e32b09aa990f2fa1f9acbd1208593715a286d5b.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/4efd0441a96131c1e6e7ea2a9e32b09aa990f2fa1f9acbd1208593715a286d5b.jpg)

![682c557e1bf21ba0b04a7bffdc559346f1eaea70591a82f5b0eec5b0ca666917.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/682c557e1bf21ba0b04a7bffdc559346f1eaea70591a82f5b0eec5b0ca666917.jpg)

![6f7795566a67666a99b189eeecae90f248984ecfebf841f61e31fa40ff158322.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/6f7795566a67666a99b189eeecae90f248984ecfebf841f61e31fa40ff158322.jpg)

![70c6fd61313a4021d33afef1300f714bfdac3d2096803df32fb8bc86dea86a06.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/70c6fd61313a4021d33afef1300f714bfdac3d2096803df32fb8bc86dea86a06.jpg)

![7fb1715a43f52b29d13e3e1f99a1f06bacdad004e7f7cb3d927152450035a4b4.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/7fb1715a43f52b29d13e3e1f99a1f06bacdad004e7f7cb3d927152450035a4b4.jpg)

![a2e1027a93b511e32b0f93fa0e6cac2aad65c537e54600921b4091714d428e16.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/a2e1027a93b511e32b0f93fa0e6cac2aad65c537e54600921b4091714d428e16.jpg)

![a4c0b1e5e58d8611c6c0f4cd97cdd46d32d776b6a90585190bba78007d52eafc.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/a4c0b1e5e58d8611c6c0f4cd97cdd46d32d776b6a90585190bba78007d52eafc.jpg)

![acc4eeb4760e0f7fe272f37a7458be1e9007fa5dd01615e648d7cb573f25c8fd.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/acc4eeb4760e0f7fe272f37a7458be1e9007fa5dd01615e648d7cb573f25c8fd.jpg)

![c51cbbfc5866d601e4b6b5c5ebdda6c275b12bd59a5ce08fd321480cb41d3873.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/c51cbbfc5866d601e4b6b5c5ebdda6c275b12bd59a5ce08fd321480cb41d3873.jpg)

![e9440e943408f871fad2033d267c51a24f2631c528cc807ba8f8dc79c99214fe.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/e9440e943408f871fad2033d267c51a24f2631c528cc807ba8f8dc79c99214fe.jpg)

![eb692d70c23ce13711c85f7395e866fd6273b2a679d770e5e953985d16895ef8.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/eb692d70c23ce13711c85f7395e866fd6273b2a679d770e5e953985d16895ef8.jpg)

![eed3cd363429ae2955830c8f78bfdfb669123f182389d767dd5eb2f96d9d0d5c.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/eed3cd363429ae2955830c8f78bfdfb669123f182389d767dd5eb2f96d9d0d5c.jpg)

![f9df45fadd8acf3d5b44fb54130b91a4de65da4e4f7d8ae7daf8b704e53bd376.jpg](../icml_results/6130_Unsupervised%20Process-Aware%20Coreset%20Selection%20for%20In-Context%20Learning/tables/f9df45fadd8acf3d5b44fb54130b91a4de65da4e4f7d8ae7daf8b704e53bd376.jpg)

<a id="paper-6081"></a>

## NeuroCLUS: A Foundation Model with Functional Clustering for Intracranial Neural Decoding


### Images

![179732b19ea6799043bb303cab4c8d4985d1b5132196e8c1a6cf29dbf5bf85ec.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/179732b19ea6799043bb303cab4c8d4985d1b5132196e8c1a6cf29dbf5bf85ec.jpg)

![179b76efcba6f9003af4572dd0309ac3df5b67348bfde085cc9ee8864f0dbb56.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/179b76efcba6f9003af4572dd0309ac3df5b67348bfde085cc9ee8864f0dbb56.jpg)

![2e08752568513a5af68d4914b9d4311eff26c0173b47dcc02f19d3e87a2812a8.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/2e08752568513a5af68d4914b9d4311eff26c0173b47dcc02f19d3e87a2812a8.jpg)

![45a2e87ab05d9e7117a6a3cf3a083dc95ef2e9bfcbf09f885ef508456f9be6fe.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/45a2e87ab05d9e7117a6a3cf3a083dc95ef2e9bfcbf09f885ef508456f9be6fe.jpg)

![6c894d70faabac9cc9f2cd2232dfc48c8b074cf5cd3707466f80b4d51de79fb7.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/6c894d70faabac9cc9f2cd2232dfc48c8b074cf5cd3707466f80b4d51de79fb7.jpg)

![9a2bf7e63bda0147a4ea4cd525a55c4c29f5a46383f6f1d439f74c1515107c24.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/9a2bf7e63bda0147a4ea4cd525a55c4c29f5a46383f6f1d439f74c1515107c24.jpg)

![b2a765ae43f879114c71857c32cc6d273fbfcaf858a49b828776d07c7615d2ce.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/images/b2a765ae43f879114c71857c32cc6d273fbfcaf858a49b828776d07c7615d2ce.jpg)

### Tables

![01d744cc8a3b42e7933390cc5bfdbb27b6b80d9741981751856dba43d4c95498.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/01d744cc8a3b42e7933390cc5bfdbb27b6b80d9741981751856dba43d4c95498.jpg)

![09e2dbd0ac6a2358101f195e1b1b106fd71d48321367c6c73dcc3f2d9d420dcb.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/09e2dbd0ac6a2358101f195e1b1b106fd71d48321367c6c73dcc3f2d9d420dcb.jpg)

![31a3345734cd6f9ec0e5ea6ae596d30a1fb55eb516cdafae93ff57163cb81497.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/31a3345734cd6f9ec0e5ea6ae596d30a1fb55eb516cdafae93ff57163cb81497.jpg)

![52755d48977f207b84de9b387bd5ac389ceda3a813cb12e7799095a55f51f177.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/52755d48977f207b84de9b387bd5ac389ceda3a813cb12e7799095a55f51f177.jpg)

![5d774c6c9e4167f206dcbdd467033c610d9a46d2219ebae7bb55d050762ced76.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/5d774c6c9e4167f206dcbdd467033c610d9a46d2219ebae7bb55d050762ced76.jpg)

![74de95ea7b0ba96c8d6e7e924d94327356598ace36ed1ba520bb5cd7574638ca.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/74de95ea7b0ba96c8d6e7e924d94327356598ace36ed1ba520bb5cd7574638ca.jpg)

![889a5efb9d6d5652af2d8c8b1192ecf28236c9a36cfdd991abf0c0f81f7902ce.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/889a5efb9d6d5652af2d8c8b1192ecf28236c9a36cfdd991abf0c0f81f7902ce.jpg)

![e38bff503fa8175a23ab9462e71cd621b5e89ddf50d4d7852a54e30b5527122d.jpg](../icml_results/6131_NeuroCLUS_%20A%20Foundation%20Model%20with%20Functional%20Clustering%20for%20Intracranial%20Neural%20Decoding/tables/e38bff503fa8175a23ab9462e71cd621b5e89ddf50d4d7852a54e30b5527122d.jpg)

<a id="paper-6082"></a>

## LoCoT2V-Bench: Benchmarking Long-Form and Complex Text-to-Video Generation


### Images

![0cb4fedccb87b8d49e3937aab10987a7d0eb0b0d36e846803d16f34793843dc5.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/0cb4fedccb87b8d49e3937aab10987a7d0eb0b0d36e846803d16f34793843dc5.jpg)

![171c9e497e927fdff4be8dc1ff561317e9c59d1b0322a2ae0987ab9a208d46d3.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/171c9e497e927fdff4be8dc1ff561317e9c59d1b0322a2ae0987ab9a208d46d3.jpg)

![2ebd789355c2ef502b0d237f30f23ae86348dc7ea85094796e3b0be603f07b8e.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/2ebd789355c2ef502b0d237f30f23ae86348dc7ea85094796e3b0be603f07b8e.jpg)

![4040f12ef32994d9434ceafb19bf24bad3134fd6b54a496726cf03c65f598a62.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/4040f12ef32994d9434ceafb19bf24bad3134fd6b54a496726cf03c65f598a62.jpg)

![4bc7e96008bc5dbed5ee568cd5a377aa9498f4833f9afa7fd26a1ff7ee766891.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/4bc7e96008bc5dbed5ee568cd5a377aa9498f4833f9afa7fd26a1ff7ee766891.jpg)

![4c19bd5275325cd9af3e1f34fe27e612b9fa224a1de5036832fbca637ceee174.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/4c19bd5275325cd9af3e1f34fe27e612b9fa224a1de5036832fbca637ceee174.jpg)

![5f6356405e23444e6e2dc4123288803df48b935043a7554c0f0960fdd5672573.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/5f6356405e23444e6e2dc4123288803df48b935043a7554c0f0960fdd5672573.jpg)

![66671035df700647f14a59f095c3db12881ac0b99ff6d5054db1b2e3285db2fe.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/66671035df700647f14a59f095c3db12881ac0b99ff6d5054db1b2e3285db2fe.jpg)

![7903904208ab0f3b6369bcd5dc6cb2ab8509f1ed11bbfd9c51f19d8d085e4f8f.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/7903904208ab0f3b6369bcd5dc6cb2ab8509f1ed11bbfd9c51f19d8d085e4f8f.jpg)

![8b949c5265f735fd912ee0efe4ccb0e9967adf9ddce0c32bb28d545a62463240.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/8b949c5265f735fd912ee0efe4ccb0e9967adf9ddce0c32bb28d545a62463240.jpg)

![9249e40a9ffe2f2b5e9a86c9f7ee9ccfff7c87dfa70f0ee51c33d27a8b6b2a55.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/9249e40a9ffe2f2b5e9a86c9f7ee9ccfff7c87dfa70f0ee51c33d27a8b6b2a55.jpg)

![aa861a182d6fcfac24302fdf254b328257864de1c27fd8df975940ff614879c3.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/aa861a182d6fcfac24302fdf254b328257864de1c27fd8df975940ff614879c3.jpg)

![b8078f949829d3f3dd2f50df8727fd4c28182e28e996cd1ceef8964589a19327.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/b8078f949829d3f3dd2f50df8727fd4c28182e28e996cd1ceef8964589a19327.jpg)

![bcd30c46f4be05bdc97901c8649197194dacf42fff5eff847f50029758032689.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/bcd30c46f4be05bdc97901c8649197194dacf42fff5eff847f50029758032689.jpg)

![be910a931a50be847e10ad8b663e7b3c0ed292b029a79ed549122276863e273b.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/be910a931a50be847e10ad8b663e7b3c0ed292b029a79ed549122276863e273b.jpg)

![c7f25822f6cbd1ac6ee3e50a4f31be0f31223e222938385a30694a5536b516f9.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/c7f25822f6cbd1ac6ee3e50a4f31be0f31223e222938385a30694a5536b516f9.jpg)

![d0032d1074be7508e387e7050d80a28835178639be15fbcb31b4a9b6e6722ca7.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/d0032d1074be7508e387e7050d80a28835178639be15fbcb31b4a9b6e6722ca7.jpg)

![e66f0342b54866e1db6a892595a6629349dc799ec7b534a40be5ec7a2c44cdad.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/images/e66f0342b54866e1db6a892595a6629349dc799ec7b534a40be5ec7a2c44cdad.jpg)

### Tables

![35364752a93cb43cf327c1ee75591e80bb7b537ae10f230dbd4c966c115c376f.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/35364752a93cb43cf327c1ee75591e80bb7b537ae10f230dbd4c966c115c376f.jpg)

![413c2c7af6ac5584262e487b1d8cc8f12f6ad5e27c9afd4957dc8a51cbf75ce3.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/413c2c7af6ac5584262e487b1d8cc8f12f6ad5e27c9afd4957dc8a51cbf75ce3.jpg)

![447008d4eef29fe974ab4342aff1255fa38984f1808bdd11c2f9a18963be8caa.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/447008d4eef29fe974ab4342aff1255fa38984f1808bdd11c2f9a18963be8caa.jpg)

![579af5d19ba8fa22302d83615367e203c0850054be8defec83e0ca92b1d6c7b2.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/579af5d19ba8fa22302d83615367e203c0850054be8defec83e0ca92b1d6c7b2.jpg)

![60789b934dde2591770b0b88d22c707a29fea032f43a0d98365b8bc6abe0533d.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/60789b934dde2591770b0b88d22c707a29fea032f43a0d98365b8bc6abe0533d.jpg)

![60d4780226f7c3941c501502e1984068e613842dd57e5ae294490f4bf13f5dad.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/60d4780226f7c3941c501502e1984068e613842dd57e5ae294490f4bf13f5dad.jpg)

![85bec1e44de3e1a6ad12e4d2ed5e9e22829ffbf823ef8cd1c2ccc8f5aaca5c5c.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/85bec1e44de3e1a6ad12e4d2ed5e9e22829ffbf823ef8cd1c2ccc8f5aaca5c5c.jpg)

![cd2e0648d4d5dcd5511c8f986b6a146ca31dd3ec1052eac411cd41d4eb34c4c5.jpg](../icml_results/6132_LoCoT2V-Bench_%20Benchmarking%20Long-Form%20and%20Complex%20Text-to-Video%20Generation/tables/cd2e0648d4d5dcd5511c8f986b6a146ca31dd3ec1052eac411cd41d4eb34c4c5.jpg)

<a id="paper-6083"></a>

## Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs


### Images

![05d15194d58228c71c47b4ead27323fe2cc5c946acce103070d118f4824af114.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/05d15194d58228c71c47b4ead27323fe2cc5c946acce103070d118f4824af114.jpg)

![712bcac67af18db7c8f953544387814fc1f8bdf4b896953d00c4460d0fba2baa.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/712bcac67af18db7c8f953544387814fc1f8bdf4b896953d00c4460d0fba2baa.jpg)

![933ee73527ed81d77b9f68d70538bcf4725f7842adf597c273003528cd9f6f36.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/933ee73527ed81d77b9f68d70538bcf4725f7842adf597c273003528cd9f6f36.jpg)

![a1e0698625f4b9c46396f48d38338a2be4ec3a188b2a631369f33c35e7d5cfed.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/a1e0698625f4b9c46396f48d38338a2be4ec3a188b2a631369f33c35e7d5cfed.jpg)

![cbcfaac5098a6abb693e3d9ff56f704d19cb8f33f055d693fefedd8baf03a2b2.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/cbcfaac5098a6abb693e3d9ff56f704d19cb8f33f055d693fefedd8baf03a2b2.jpg)

![e435e73b9a1cca5d736dafc9a13d777f479e271aad0f11f642685ddb6d67cfee.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/images/e435e73b9a1cca5d736dafc9a13d777f479e271aad0f11f642685ddb6d67cfee.jpg)

### Tables

![0df1db5747c05b958ced9bac611f409dfdd37ede36f48bb4e1189d30b9c1f01a.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/0df1db5747c05b958ced9bac611f409dfdd37ede36f48bb4e1189d30b9c1f01a.jpg)

![15bad15cfcf9793d484a2d8631ce783793eb7fc999ab2d290afcb3463d847966.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/15bad15cfcf9793d484a2d8631ce783793eb7fc999ab2d290afcb3463d847966.jpg)

![1a46c1030801cc77f29d6aeb9adfe57921c7d0f5461d01373b46a417d0729541.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/1a46c1030801cc77f29d6aeb9adfe57921c7d0f5461d01373b46a417d0729541.jpg)

![1a730a816ff0c3ce1ada33741fb5555b1c4074c6386edb72d445d0f9436e275c.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/1a730a816ff0c3ce1ada33741fb5555b1c4074c6386edb72d445d0f9436e275c.jpg)

![2cc01db77138513c4bd57f51e302b13992fd388b3c17c8ff42783281285f035a.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/2cc01db77138513c4bd57f51e302b13992fd388b3c17c8ff42783281285f035a.jpg)

![5a4e9ec0e988fa444a6b9cba7943464cdd77f03ada5238fb6e98dd866e729bda.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/5a4e9ec0e988fa444a6b9cba7943464cdd77f03ada5238fb6e98dd866e729bda.jpg)

![7012af7e3a1b3c0af72c483a7584676186c5379223593cf9e1e96c8e24b662e4.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/7012af7e3a1b3c0af72c483a7584676186c5379223593cf9e1e96c8e24b662e4.jpg)

![73ffe83cdc1f60f18a3376a4ba6a8a71dbf0fbdafd17f0fe14c3d5c77fb062fe.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/73ffe83cdc1f60f18a3376a4ba6a8a71dbf0fbdafd17f0fe14c3d5c77fb062fe.jpg)

![761c2c794adc32ce146185b438ce8f9b10e5b20eb7424c8ef1db8ba32469b20c.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/761c2c794adc32ce146185b438ce8f9b10e5b20eb7424c8ef1db8ba32469b20c.jpg)

![83b2366e582e7399992409b968095f517dce10679fc53ee1b81a48643f48827b.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/83b2366e582e7399992409b968095f517dce10679fc53ee1b81a48643f48827b.jpg)

![90a91857c443dd31c36bc1ad369a16cbab7ff84f7d8f2ed1375924f4912384a9.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/90a91857c443dd31c36bc1ad369a16cbab7ff84f7d8f2ed1375924f4912384a9.jpg)

![99d854903c75f0cfe8d0dd992b8da013ea84e8d3bfdc8629f9527087d0e1b6ef.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/99d854903c75f0cfe8d0dd992b8da013ea84e8d3bfdc8629f9527087d0e1b6ef.jpg)

![a0439fb069171eb4702d6cfe5668a12c4067c44e281d7bd0adde394ff11ec27f.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/a0439fb069171eb4702d6cfe5668a12c4067c44e281d7bd0adde394ff11ec27f.jpg)

![a789258d7d883a46612256ba4ffa3d7f0dde9fefd7e0dc99bfb0bfb24a8a9647.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/a789258d7d883a46612256ba4ffa3d7f0dde9fefd7e0dc99bfb0bfb24a8a9647.jpg)

![c5935934d5a1dd34686702fa1a9fba63d00045c2a0eaf33288031a91b6a5ce4e.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/c5935934d5a1dd34686702fa1a9fba63d00045c2a0eaf33288031a91b6a5ce4e.jpg)

![c8913c9adc2ef21525aff3cdec08d6185edde39bbbba82e97f1e6fffbf44278a.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/c8913c9adc2ef21525aff3cdec08d6185edde39bbbba82e97f1e6fffbf44278a.jpg)

![c8c8d9f61de7ae94a8aff1a2566c163a3eaf1d5f8d77de559123cff7302f41ac.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/c8c8d9f61de7ae94a8aff1a2566c163a3eaf1d5f8d77de559123cff7302f41ac.jpg)

![ce84ebd0a9c849cd025c840a7bf6b9683b63eb49b56c43053ab7e1e9e6f9e07a.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/ce84ebd0a9c849cd025c840a7bf6b9683b63eb49b56c43053ab7e1e9e6f9e07a.jpg)

![dcefa7c324ed553c111f13c0210b2212dc40a55568c4821e6457a1e7df5f573d.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/dcefa7c324ed553c111f13c0210b2212dc40a55568c4821e6457a1e7df5f573d.jpg)

![ec12d1c4af211bde41379cc6ce5086c1e8cf1464fb95d0022fe82ca4cccd22b3.jpg](../icml_results/6133_Just%20Ask_%20Curious%20Code%20Agents%20Reveal%20System%20Prompts%20in%20Frontier%20LLMs/tables/ec12d1c4af211bde41379cc6ce5086c1e8cf1464fb95d0022fe82ca4cccd22b3.jpg)

<a id="paper-6084"></a>

## Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization


### Images

![101fdbaf19593d8ad203d3e58f2584d115a5fcc515e0b0650701343cd2d6c901.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/101fdbaf19593d8ad203d3e58f2584d115a5fcc515e0b0650701343cd2d6c901.jpg)

![3002e583c1df1ff21b828821089ebddfd4af91eb91a244b55d9d40819668af8d.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/3002e583c1df1ff21b828821089ebddfd4af91eb91a244b55d9d40819668af8d.jpg)

![53047404dbcb887b5869a3e1a12d59e11d37ba9ade0b0014d4655076832a619e.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/53047404dbcb887b5869a3e1a12d59e11d37ba9ade0b0014d4655076832a619e.jpg)

![6dad82e42f5c646cd979f856d11dd940a9a393d9c479893a3798d6d8b0eb87d2.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/6dad82e42f5c646cd979f856d11dd940a9a393d9c479893a3798d6d8b0eb87d2.jpg)

![c772ee4c33cc361a765c6b9ee47b92b1956ff2f61c4d68c0049a3a6701ae51cd.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/c772ee4c33cc361a765c6b9ee47b92b1956ff2f61c4d68c0049a3a6701ae51cd.jpg)

![cdca5f64bb1b213909da121350d73f2d1dc4bea91e6b93b6595925a2f4c15e19.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/cdca5f64bb1b213909da121350d73f2d1dc4bea91e6b93b6595925a2f4c15e19.jpg)

![da48c96565dd646fe18050f0bda7c7791a3a7c4ea0fbc142d834e544ba308277.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/da48c96565dd646fe18050f0bda7c7791a3a7c4ea0fbc142d834e544ba308277.jpg)

![f187704a1f2d750ca1e5f9a188a7b68321000d41df40cf14625fc2f22ad35295.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/images/f187704a1f2d750ca1e5f9a188a7b68321000d41df40cf14625fc2f22ad35295.jpg)

### Tables

![1a3dcdc3326a7f71964230d478994bf11bd0df552121b501c8cf79d1cfb20044.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/1a3dcdc3326a7f71964230d478994bf11bd0df552121b501c8cf79d1cfb20044.jpg)

![206ffc37a61ee6278da14fe7ddfac169ff82ff87b14eb357172ebb61fa0a9e91.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/206ffc37a61ee6278da14fe7ddfac169ff82ff87b14eb357172ebb61fa0a9e91.jpg)

![2470d46873700dd8d1b6f165a18c399664e74be9d070dff985d0c94a826323e7.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/2470d46873700dd8d1b6f165a18c399664e74be9d070dff985d0c94a826323e7.jpg)

![2b52583f4b0797649df559b5d1d7e79dcd5388ac7ef40e8bfc3eed6e2369f32f.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/2b52583f4b0797649df559b5d1d7e79dcd5388ac7ef40e8bfc3eed6e2369f32f.jpg)

![2cfc7f2e14e25e468d43cfb66e8a1d4cadb3361050fb26281530a2ca01ea14d1.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/2cfc7f2e14e25e468d43cfb66e8a1d4cadb3361050fb26281530a2ca01ea14d1.jpg)

![2dcd7e96d04189f8caaea3fa423fe02ff605e492ed0da19df0b0a8bd620613c7.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/2dcd7e96d04189f8caaea3fa423fe02ff605e492ed0da19df0b0a8bd620613c7.jpg)

![38a131863dce5f050d818d741ef2d5db421ba6832ebcf88ba3fad024e85d5d2e.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/38a131863dce5f050d818d741ef2d5db421ba6832ebcf88ba3fad024e85d5d2e.jpg)

![4f8d8a7823d28e41f93a4927752dbcd97cff41a1e609c8a0823c3540822a80b7.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/4f8d8a7823d28e41f93a4927752dbcd97cff41a1e609c8a0823c3540822a80b7.jpg)

![60b9044107b703bcd0c0eba47432ef2c7c2e498d0580db4fafac1a86d625a83e.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/60b9044107b703bcd0c0eba47432ef2c7c2e498d0580db4fafac1a86d625a83e.jpg)

![683d5a3388b17454418586817a140edae15b217051ee5ce557a693eb46876032.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/683d5a3388b17454418586817a140edae15b217051ee5ce557a693eb46876032.jpg)

![88c5b6ef20a79daac06faa18de22b5f4bfc992fb3a3fdd6845f1bbb5b146d265.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/88c5b6ef20a79daac06faa18de22b5f4bfc992fb3a3fdd6845f1bbb5b146d265.jpg)

![999945fc42451564e741a4a058a63aa112be1cd9d93e00ee277fd9b232d4a452.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/999945fc42451564e741a4a058a63aa112be1cd9d93e00ee277fd9b232d4a452.jpg)

![a4e2ddb096c930948a1f2e800d9c3e129cc7bb69111dbbb476763ab151a99e70.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/a4e2ddb096c930948a1f2e800d9c3e129cc7bb69111dbbb476763ab151a99e70.jpg)

![aac03f12fdeaa26b0c248ce30bc2200ced43a2ab9df94c1bc51ad30b95da2644.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/aac03f12fdeaa26b0c248ce30bc2200ced43a2ab9df94c1bc51ad30b95da2644.jpg)

![abb7c55d0fa10f59e8607d1ffe595886ee5043830153c3aab6d4f1bb11dd1bc7.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/abb7c55d0fa10f59e8607d1ffe595886ee5043830153c3aab6d4f1bb11dd1bc7.jpg)

![c1c3088326f192948a50425f74e26b50bf4fa8ba4901ff1dfdadeb8906d27346.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/c1c3088326f192948a50425f74e26b50bf4fa8ba4901ff1dfdadeb8906d27346.jpg)

![e781db35110f72bb5a7a0ea8be1233f88bc3838a57647c39699c7a1663eeab00.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/e781db35110f72bb5a7a0ea8be1233f88bc3838a57647c39699c7a1663eeab00.jpg)

![e7d4bfa6a1b359b5d93a6a2c159007e35577fed6261b617b408ee6258b1ffbd9.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/e7d4bfa6a1b359b5d93a6a2c159007e35577fed6261b617b408ee6258b1ffbd9.jpg)

![f57b73d80760f20d2144857a306981a78fe7090df62f7f404849cc18fc4d2aec.jpg](../icml_results/6134_Neural%20QAOA%24%5E2%24_%20Differentiable%20Joint%20Graph%20Partitioning%20and%20Parameter%20Initialization%20for%20Quantum%20Co/tables/f57b73d80760f20d2144857a306981a78fe7090df62f7f404849cc18fc4d2aec.jpg)

<a id="paper-6085"></a>

## iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance


### Images

![3a82236189ec03c1118a21795f8707f71bb309333820494a692ae815099b9c81.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/3a82236189ec03c1118a21795f8707f71bb309333820494a692ae815099b9c81.jpg)

![5a3f413e25685349b0e69babe27e9d767d52d9555a07ca83eea253fa8c1e4a92.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/5a3f413e25685349b0e69babe27e9d767d52d9555a07ca83eea253fa8c1e4a92.jpg)

![94eb75a745a7e1bf219b5a210712a57cace52f1c05c3c7d6150f89056e399a66.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/94eb75a745a7e1bf219b5a210712a57cace52f1c05c3c7d6150f89056e399a66.jpg)

![da4d37a82707cc6087070153859385582d5380905706dca1c9c743c8a138bebb.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/da4d37a82707cc6087070153859385582d5380905706dca1c9c743c8a138bebb.jpg)

![f27c5d3fe775ff9f30a729175841c49551412fea1f48c00192ae124d30b02b04.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/f27c5d3fe775ff9f30a729175841c49551412fea1f48c00192ae124d30b02b04.jpg)

![fd17d09ed0de70ca2e7e3bcc9643653c0c137cad3eb537c6050b11301d3a459b.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/fd17d09ed0de70ca2e7e3bcc9643653c0c137cad3eb537c6050b11301d3a459b.jpg)

![fed641f8c6288f5ee940133d3dc52ef1d18d7ba294cdbe2d71fd07df9add9879.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/fed641f8c6288f5ee940133d3dc52ef1d18d7ba294cdbe2d71fd07df9add9879.jpg)

![ff3cfaf74c5a68bb1d8c63297ca35e7949e7dd0fded7712da32d7446fb599d3c.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/images/ff3cfaf74c5a68bb1d8c63297ca35e7949e7dd0fded7712da32d7446fb599d3c.jpg)

### Tables

![002f057de34961b10423028aba6c28e89ca122b552623f3b9a69bd9fec846aec.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/002f057de34961b10423028aba6c28e89ca122b552623f3b9a69bd9fec846aec.jpg)

![0300c7a4667234be549d3d240fcd1862539c64bfb0ec3dc6f61d860e29083b56.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/0300c7a4667234be549d3d240fcd1862539c64bfb0ec3dc6f61d860e29083b56.jpg)

![03c31b945d9c19c87f43c5c8cd02172cb4f150a8865fc17a6ccbffc62ef75d77.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/03c31b945d9c19c87f43c5c8cd02172cb4f150a8865fc17a6ccbffc62ef75d77.jpg)

![526e289521dfe12a1a36542a800bbcd22ef16694f0e9525930071fd7e24ad30b.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/526e289521dfe12a1a36542a800bbcd22ef16694f0e9525930071fd7e24ad30b.jpg)

![54205b07b7231fb8226d7ff52c166db4483adabb6873dbe47b40f6c564a3f76b.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/54205b07b7231fb8226d7ff52c166db4483adabb6873dbe47b40f6c564a3f76b.jpg)

![78fa6e4b700fee3204f3c5fc385ee2a2b238aee18c2c550012880ce152903039.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/78fa6e4b700fee3204f3c5fc385ee2a2b238aee18c2c550012880ce152903039.jpg)

![9c2f1b398379f2d413b910f5abc9a37033a1a9577d7044eebf6b7367b98809d5.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/9c2f1b398379f2d413b910f5abc9a37033a1a9577d7044eebf6b7367b98809d5.jpg)

![c8072b4c1d62040c6191dd285cfcfd072f1b31991d12938927e97794437af581.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/c8072b4c1d62040c6191dd285cfcfd072f1b31991d12938927e97794437af581.jpg)

![d7269e90ad10aa2d674e5b801de479a9de8ddcb69b508a194fa4800700e8ee2f.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/d7269e90ad10aa2d674e5b801de479a9de8ddcb69b508a194fa4800700e8ee2f.jpg)

![ff4fad7d2c05941293f6dc9aeee5e34c799cf337aa6c831ec9b37027d19de480.jpg](../icml_results/6135_iTryOn_%20Mastering%20Interactive%20Video%20Virtual%20Try-On%20with%20Spatial-Semantic%20Guidance/tables/ff4fad7d2c05941293f6dc9aeee5e34c799cf337aa6c831ec9b37027d19de480.jpg)

<a id="paper-6086"></a>

## CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving


### Images

![2e9a1ea2ce3abba6a09b47b37b34cdb93bce591e99e1fa23cb7d60310fef17d7.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/2e9a1ea2ce3abba6a09b47b37b34cdb93bce591e99e1fa23cb7d60310fef17d7.jpg)

![490c6b9d5bd378a5cebf0961d712a540a4460c8eab5f4a07a57dfc0d8609cefc.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/490c6b9d5bd378a5cebf0961d712a540a4460c8eab5f4a07a57dfc0d8609cefc.jpg)

![4defeed0a78b22b6ada3e9b18cfed9cd532040f885da523e92e0724f12503d0c.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/4defeed0a78b22b6ada3e9b18cfed9cd532040f885da523e92e0724f12503d0c.jpg)

![518d5e188fae7bdba98672cac668b8f1ae77767752a6b81bd1962da7876b8dd4.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/518d5e188fae7bdba98672cac668b8f1ae77767752a6b81bd1962da7876b8dd4.jpg)

![585e42c675dd50b6756c83405a995d22a83090f0028d7e820bdd977bf66d9869.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/585e42c675dd50b6756c83405a995d22a83090f0028d7e820bdd977bf66d9869.jpg)

![58651d795f003739f4805ea5dff33e56606b096851a5396c686a6fbd35d1bde9.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/58651d795f003739f4805ea5dff33e56606b096851a5396c686a6fbd35d1bde9.jpg)

![5f7c94e11d3eb61e7ab62688dc95341e921c3b6bcd1082c722abdf7a4d78371d.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/5f7c94e11d3eb61e7ab62688dc95341e921c3b6bcd1082c722abdf7a4d78371d.jpg)

![73d722a4ffad7471da4208e18b2b815f06df9af3508356ba9ad434f1022ec642.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/73d722a4ffad7471da4208e18b2b815f06df9af3508356ba9ad434f1022ec642.jpg)

![a3f65ddf4e0d8017c0d8634a2639f46c098e060088a87dea602920c5bf25bed2.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/a3f65ddf4e0d8017c0d8634a2639f46c098e060088a87dea602920c5bf25bed2.jpg)

![a6148df56b4b6bdc76fcb96deb4cbee7e89eeed9fbee70e10acdb4af631147c9.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/a6148df56b4b6bdc76fcb96deb4cbee7e89eeed9fbee70e10acdb4af631147c9.jpg)

![ac1367beaf6cdb55d64fab025936a2e02c477c1e4282ef5ed82269d2c7ef5760.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/ac1367beaf6cdb55d64fab025936a2e02c477c1e4282ef5ed82269d2c7ef5760.jpg)

![b2a85803d05ec34dbde1001624a1a1ae7e901ce84e2ab15071957db451d3dd31.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/b2a85803d05ec34dbde1001624a1a1ae7e901ce84e2ab15071957db451d3dd31.jpg)

![cb52ee12e8e2be122fac22c8f65bb8e74fc1143431603707b1da03ea106dc437.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/cb52ee12e8e2be122fac22c8f65bb8e74fc1143431603707b1da03ea106dc437.jpg)

![d0061c392a535b31cc81bee2f667d4395525141aaa73d117ed0ae963dbcac4ae.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/d0061c392a535b31cc81bee2f667d4395525141aaa73d117ed0ae963dbcac4ae.jpg)

![df90eb9a41d2320c42b1e40166110db8d712b804f6f1b782665e5f52f6947226.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/df90eb9a41d2320c42b1e40166110db8d712b804f6f1b782665e5f52f6947226.jpg)

![ec959478d1af8e422e777c5445cc58e268ffe1b0f34190b8eda8c3acc12c5c54.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/ec959478d1af8e422e777c5445cc58e268ffe1b0f34190b8eda8c3acc12c5c54.jpg)

![f2f0e29450211030d7c94de426fab2f769691fce5b9f3a8f25cc2a59e7a86a58.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/f2f0e29450211030d7c94de426fab2f769691fce5b9f3a8f25cc2a59e7a86a58.jpg)

![f6ad63c4bd812c230d9a63c6bc96381e7d196f3208c09cc16af20ae5a3d03b8c.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/f6ad63c4bd812c230d9a63c6bc96381e7d196f3208c09cc16af20ae5a3d03b8c.jpg)

![fb3a670d7f66d34a0c666e7b01b6cc182177dc641a0146e862b11aefb18277b0.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/images/fb3a670d7f66d34a0c666e7b01b6cc182177dc641a0146e862b11aefb18277b0.jpg)

### Tables

![04b27aace0295b1eee2a35ff0d79b1ce4ab862a5fce6043f5996baee5a1abf50.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/04b27aace0295b1eee2a35ff0d79b1ce4ab862a5fce6043f5996baee5a1abf50.jpg)

![0688a86f1ed01836cc2696fd914ffcb6d2f7781f837e221ed2407e840f99e6dc.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/0688a86f1ed01836cc2696fd914ffcb6d2f7781f837e221ed2407e840f99e6dc.jpg)

![3d5d25f175aa9ddc8c9db9fe91da1fc17e97f4b7ac61c601b49a4caeaa5c5b38.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/3d5d25f175aa9ddc8c9db9fe91da1fc17e97f4b7ac61c601b49a4caeaa5c5b38.jpg)

![618fa4bb856fcec1dc0722bf354d3296e497f4446173b2292d98edd42fc9b457.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/618fa4bb856fcec1dc0722bf354d3296e497f4446173b2292d98edd42fc9b457.jpg)

![7ef947e06639caa4df880cd4fd88752fc6cb885cd900d423decd10192500498a.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/7ef947e06639caa4df880cd4fd88752fc6cb885cd900d423decd10192500498a.jpg)

![9f2f15b52a51254755c5d785e29935db23bdaffdca4351d9f84a205115239640.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/9f2f15b52a51254755c5d785e29935db23bdaffdca4351d9f84a205115239640.jpg)

![a8b464c0ba8d48b2129f038cd3597f19c8c002dd1caf2351f0e2813c27658d1f.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/a8b464c0ba8d48b2129f038cd3597f19c8c002dd1caf2351f0e2813c27658d1f.jpg)

![a90717e331e4bee51dc06a9190a0c77dbc7d639fce1f4a9264c22fd933ab9810.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/a90717e331e4bee51dc06a9190a0c77dbc7d639fce1f4a9264c22fd933ab9810.jpg)

![b1939552e1736e95647cbf4cf5d9f93bf0d207fe2fd2e87a2595cc4f837d90a4.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/b1939552e1736e95647cbf4cf5d9f93bf0d207fe2fd2e87a2595cc4f837d90a4.jpg)

![e7bfcbaa90509c39c05bf2e1b044dbffd1ab4bea24e8bd4e99373a334809e63e.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/e7bfcbaa90509c39c05bf2e1b044dbffd1ab4bea24e8bd4e99373a334809e63e.jpg)

![fcf9f4bc03a2a3156e24da71129e6bf047f4d705a157af91e0e7f158b3ee58af.jpg](../icml_results/6136_CoIRL-AD_%20Collaborative-Competitive%20Imitation-Reinforcement%20Learning%20in%20Latent%20World%20Models%20for%20Auto/tables/fcf9f4bc03a2a3156e24da71129e6bf047f4d705a157af91e0e7f158b3ee58af.jpg)

<a id="paper-6087"></a>

## MDGMIX: Boundary-Aware Subgraph Mixing for Multi-Domain Graph Pre-Training


### Images

![0b80ac62aa04fd181c3dfc9a416c1012edac8186bbb1d81d47a5a41f3f38d869.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/0b80ac62aa04fd181c3dfc9a416c1012edac8186bbb1d81d47a5a41f3f38d869.jpg)

![0d3db43b4e0bd5fcae503ece64db1b2372ba8bf1645327558064f7db335acdb8.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/0d3db43b4e0bd5fcae503ece64db1b2372ba8bf1645327558064f7db335acdb8.jpg)

![41acb8fca87aee6e944b3ee9b83716a010be5d2c7db89f8e37dc0da907f02421.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/41acb8fca87aee6e944b3ee9b83716a010be5d2c7db89f8e37dc0da907f02421.jpg)

![527208d91081525d2ab37eaa90b993ad2cbe8438ef13f68a02c3b2307d6725a1.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/527208d91081525d2ab37eaa90b993ad2cbe8438ef13f68a02c3b2307d6725a1.jpg)

![6b3bb7e678987c158faaede8299f57ce3ce29f75452168543ba1a9a3651f3a6f.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/6b3bb7e678987c158faaede8299f57ce3ce29f75452168543ba1a9a3651f3a6f.jpg)

![6ed5bae9ef8a7fda2e16df9fdc9f502e0874756d193f73ebc16b5a04b6ee6bf4.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/6ed5bae9ef8a7fda2e16df9fdc9f502e0874756d193f73ebc16b5a04b6ee6bf4.jpg)

![87d14f9e3625091c97cde687c1af28c21d6f8a3a257db561adeb43c896f06999.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/87d14f9e3625091c97cde687c1af28c21d6f8a3a257db561adeb43c896f06999.jpg)

![9402a1a5f31cdab4148ad7fb2bfa7bc921393c6e844897ef91bb452f67829e46.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/9402a1a5f31cdab4148ad7fb2bfa7bc921393c6e844897ef91bb452f67829e46.jpg)

![d090b1d713a080f80bd64721d3fb471218738ba9a03916d6532867922f9c6f1f.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/images/d090b1d713a080f80bd64721d3fb471218738ba9a03916d6532867922f9c6f1f.jpg)

### Tables

![00ff94fa8c17fedad8a00a7ef7f0ef170c879591f489efb36e5633380806450a.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/00ff94fa8c17fedad8a00a7ef7f0ef170c879591f489efb36e5633380806450a.jpg)

![0cc97d330dc0d972fec843722c1244b707f7975b36b1fcaa2bbd0695cef146cf.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/0cc97d330dc0d972fec843722c1244b707f7975b36b1fcaa2bbd0695cef146cf.jpg)

![25eccbfeeac7accbe08b142afacf5120f04975f07d05fb1656ca28c7f3bf5710.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/25eccbfeeac7accbe08b142afacf5120f04975f07d05fb1656ca28c7f3bf5710.jpg)

![458f7056c81e56fa41fb375f011ed0d8e614c4b01de89b7718e845a5c09ca8d5.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/458f7056c81e56fa41fb375f011ed0d8e614c4b01de89b7718e845a5c09ca8d5.jpg)

![50dec99e2eaceb184b2ce98869be29da24b3e25809a5a23c5d30f8d6e208fc59.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/50dec99e2eaceb184b2ce98869be29da24b3e25809a5a23c5d30f8d6e208fc59.jpg)

![61dc3066f499fa2aacf247851148f696727284fc8aff0bc0cf1890209280ce9f.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/61dc3066f499fa2aacf247851148f696727284fc8aff0bc0cf1890209280ce9f.jpg)

![68e7862617f64fead760363359a25dd8e3e1045bfb5888ca0f7ebc3783056afb.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/68e7862617f64fead760363359a25dd8e3e1045bfb5888ca0f7ebc3783056afb.jpg)

![72d38d9cb91c57b9395469707719dd9caa126e1734df102968edd84f5859d6e9.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/72d38d9cb91c57b9395469707719dd9caa126e1734df102968edd84f5859d6e9.jpg)

![815694901d0009115377680eee60b181a7690f7f0158e4f6652de33d7f5dcd3b.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/815694901d0009115377680eee60b181a7690f7f0158e4f6652de33d7f5dcd3b.jpg)

![c98f34e7d66fd710f4485dc70dc29ca4181e31d92007782f66dddf00fbf30d78.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/c98f34e7d66fd710f4485dc70dc29ca4181e31d92007782f66dddf00fbf30d78.jpg)

![cb55a26b3f5baa11f39c17f929208be58c21508c31716c6a950724a2390fa447.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/cb55a26b3f5baa11f39c17f929208be58c21508c31716c6a950724a2390fa447.jpg)

![daa541920391ff9928ef421ad9f777960a337413b97a41712979380c142334a1.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/daa541920391ff9928ef421ad9f777960a337413b97a41712979380c142334a1.jpg)

![e8a866bb4eb1df7c8ea2ba89025b43c3f9b4e0af5b98a48c7182dabbbd78dfc2.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/e8a866bb4eb1df7c8ea2ba89025b43c3f9b4e0af5b98a48c7182dabbbd78dfc2.jpg)

![e8bc68ee3ada832309438d895fd2187db28cae3940a2ce1e59d2ca19d6634a3f.jpg](../icml_results/6137_MDGMIX_%20Boundary-Aware%20Subgraph%20Mixing%20for%20Multi-Domain%20Graph%20Pre-Training/tables/e8bc68ee3ada832309438d895fd2187db28cae3940a2ce1e59d2ca19d6634a3f.jpg)

<a id="paper-6088"></a>

## RGGT: A Generative-Prior-Guided Transformer for Unified Rigid and Non-Rigid Point Cloud Registration


### Images

![11e3336c47a654d7b800f4b44bb9439f8528d0b05572dc821b0ae614c0e62bc8.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/11e3336c47a654d7b800f4b44bb9439f8528d0b05572dc821b0ae614c0e62bc8.jpg)

![25f588e62f276a187c91257fa4eb7268b030fcb93487fe96033ceaa9d2070d3a.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/25f588e62f276a187c91257fa4eb7268b030fcb93487fe96033ceaa9d2070d3a.jpg)

![343a9728d72139d3f2c660daeb5277582ba027aa6cdc8be74ea9faabd1bca9db.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/343a9728d72139d3f2c660daeb5277582ba027aa6cdc8be74ea9faabd1bca9db.jpg)

![a9e5ec5bc1d707d844b926d25c5c6dc0f65a5c9240fe2cf2f7d4661cc909aa89.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/a9e5ec5bc1d707d844b926d25c5c6dc0f65a5c9240fe2cf2f7d4661cc909aa89.jpg)

![bff77bc55349ab7bf41059f05ebe51aae573b1735780716f6239a4274f551ea4.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/bff77bc55349ab7bf41059f05ebe51aae573b1735780716f6239a4274f551ea4.jpg)

![c22c241fd24127c7c05644b54d3b4cd182c1dc535b4290d607611d34c7514bb9.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/c22c241fd24127c7c05644b54d3b4cd182c1dc535b4290d607611d34c7514bb9.jpg)

![d6b284b179c4f9a4aac2fb4a229ae514a6b880f19a77e3694d87040a1d61fe44.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/d6b284b179c4f9a4aac2fb4a229ae514a6b880f19a77e3694d87040a1d61fe44.jpg)

![f91042bdaa3d10c8447e5002b5c6984bf8d93d20e98756b5eaa36ec0fd384ed1.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/images/f91042bdaa3d10c8447e5002b5c6984bf8d93d20e98756b5eaa36ec0fd384ed1.jpg)

### Tables

![019657a8290e86ddad90a5dca17487080673b833f9f3d7216a9db16954e7e199.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/019657a8290e86ddad90a5dca17487080673b833f9f3d7216a9db16954e7e199.jpg)

![1dcc18c0532eef507cad76ac711cf04e2db78080937560443b5c91cbca931116.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/1dcc18c0532eef507cad76ac711cf04e2db78080937560443b5c91cbca931116.jpg)

![41c96e3c4d61a397278beca4df4b2e89a2a2005ddb35204878995a0cad83ed9a.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/41c96e3c4d61a397278beca4df4b2e89a2a2005ddb35204878995a0cad83ed9a.jpg)

![4aad811c51c87824947202502a6324449399459e9cfb22f1ec9cdd4dd4538c26.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/4aad811c51c87824947202502a6324449399459e9cfb22f1ec9cdd4dd4538c26.jpg)

![72da18cd5ea2ae77ecef23695447740e07d9ba3d082004c0e7debc0bc90ba18f.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/72da18cd5ea2ae77ecef23695447740e07d9ba3d082004c0e7debc0bc90ba18f.jpg)

![81d53df3911376ec30a5d2dd8097daa1cffe568742cbade6beec88227624e16d.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/81d53df3911376ec30a5d2dd8097daa1cffe568742cbade6beec88227624e16d.jpg)

![82a96eb73a367851774dfe4a062e9bc11922e331897effb7ee7fae0a507907f6.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/82a96eb73a367851774dfe4a062e9bc11922e331897effb7ee7fae0a507907f6.jpg)

![a926026021362f28605fcc7978ad6c375f364f8032cb73aa82138ff223c38751.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/a926026021362f28605fcc7978ad6c375f364f8032cb73aa82138ff223c38751.jpg)

![b4ecaf32f3e387745db0ac31e87276a84f281bb25f8771e84934a83756b45c66.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/b4ecaf32f3e387745db0ac31e87276a84f281bb25f8771e84934a83756b45c66.jpg)

![ed6da6602f3de50a714aa4bf9a097cc32487ac523f2e25a553d8ef22ae48df8f.jpg](../icml_results/6138_RGGT_%20A%20Generative-Prior-Guided%20Transformer%20for%20Unified%20Rigid%20and%20Non-Rigid%20Point%20Cloud%20Registration/tables/ed6da6602f3de50a714aa4bf9a097cc32487ac523f2e25a553d8ef22ae48df8f.jpg)

<a id="paper-6089"></a>

## Plasticity Activation via Polar Operator: A Plug-in Method for Balancing Stability and Plasticity


### Images

![042c388eb160e6ad829d2d4e2d774327c59a4ea6271562300c99f4b743fde12c.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/042c388eb160e6ad829d2d4e2d774327c59a4ea6271562300c99f4b743fde12c.jpg)

![059b2c12e6a656acce2aa0c603ec7fe1e858147d081f6933bfd411c72d194134.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/059b2c12e6a656acce2aa0c603ec7fe1e858147d081f6933bfd411c72d194134.jpg)

![177967f7a22e6fb11f97bc3c4a4189594a7b9af168a8519a43e0a34b725465ee.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/177967f7a22e6fb11f97bc3c4a4189594a7b9af168a8519a43e0a34b725465ee.jpg)

![2a85815b8d89ffcb7812712b09d10f924abab3812478f7e8ad50b871e22369c3.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/2a85815b8d89ffcb7812712b09d10f924abab3812478f7e8ad50b871e22369c3.jpg)

![3132e704bbfd33d4046f7c16dbf4b9df890e2e59d1e21aa97efe7e4f2fd00db4.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/3132e704bbfd33d4046f7c16dbf4b9df890e2e59d1e21aa97efe7e4f2fd00db4.jpg)

![38dbeb575123b5094cf5ccd6fa19596578828ba5b3a9a1f8015eb75cb1750e65.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/38dbeb575123b5094cf5ccd6fa19596578828ba5b3a9a1f8015eb75cb1750e65.jpg)

![6d74cf006a9ab54d2efe4c70e5adfef606ba29f991e4732158f6ee8929b86955.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/6d74cf006a9ab54d2efe4c70e5adfef606ba29f991e4732158f6ee8929b86955.jpg)

![89933c2e05bf2fff23e72c2c4cedf9e84f42e81c5310855ef25440de84e04d7c.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/89933c2e05bf2fff23e72c2c4cedf9e84f42e81c5310855ef25440de84e04d7c.jpg)

![9af412f73c0658c501d6623d06894f18d5ed6a7a93ab485535e1c0a231ce7d08.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/9af412f73c0658c501d6623d06894f18d5ed6a7a93ab485535e1c0a231ce7d08.jpg)

![d44012db1bbc86fd62a6d3f413d6aa2374844c86daf9221c36f237c6072bfaf1.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/d44012db1bbc86fd62a6d3f413d6aa2374844c86daf9221c36f237c6072bfaf1.jpg)

![dfb2f9f1b622b14753718450447d1a132ef2b9d5575f79eb2772ad7f164f525c.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/dfb2f9f1b622b14753718450447d1a132ef2b9d5575f79eb2772ad7f164f525c.jpg)

![f26e5508fc47019afbffd72b94f6392fadb11d7d494031a0e8d3d4c8775ef748.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/f26e5508fc47019afbffd72b94f6392fadb11d7d494031a0e8d3d4c8775ef748.jpg)

![f3ebd3ccb696b11314db65692818096246272df94be9716f7f345f9a80b617d4.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/f3ebd3ccb696b11314db65692818096246272df94be9716f7f345f9a80b617d4.jpg)

![f76e3d82388fed6703f542382cdaa21ea514e97d7d5146b0b94b14da7ea39725.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/images/f76e3d82388fed6703f542382cdaa21ea514e97d7d5146b0b94b14da7ea39725.jpg)

### Tables

![2e53c3b788a82b4a8483ad52a81c295062e977624086e9f263d969eeceffa7e6.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/2e53c3b788a82b4a8483ad52a81c295062e977624086e9f263d969eeceffa7e6.jpg)

![4ef8f051f8d487dc43bcf36db6544dc3a3fa3b06e852bde2b53ff23408420c0e.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/4ef8f051f8d487dc43bcf36db6544dc3a3fa3b06e852bde2b53ff23408420c0e.jpg)

![4f7dec43c6ad3765c7c96b31cf7b608b8e57f2476d2b77d4068363f2e0765f91.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/4f7dec43c6ad3765c7c96b31cf7b608b8e57f2476d2b77d4068363f2e0765f91.jpg)

![63b71a6c5d0a92441bec25f80e8c24cb83cc4134788b1eeda77b55f3255a5609.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/63b71a6c5d0a92441bec25f80e8c24cb83cc4134788b1eeda77b55f3255a5609.jpg)

![9404e3d0471a6d60548843bf745089de1942c210abb7311ff07646457fb4fed6.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/9404e3d0471a6d60548843bf745089de1942c210abb7311ff07646457fb4fed6.jpg)

![96f21ba44920f025496bf4fbb20ffcc6ad78555578d6b82dda19b3062eb40d46.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/96f21ba44920f025496bf4fbb20ffcc6ad78555578d6b82dda19b3062eb40d46.jpg)

![98efa09c27d750cd4e4735ba6c8705f7b86e30f4817205c8c8c3084cbae43650.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/98efa09c27d750cd4e4735ba6c8705f7b86e30f4817205c8c8c3084cbae43650.jpg)

![b0c6f92073f8ea736ed53cbdf97b7f8cc43ac44f83e20971c1b4880e84cae233.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/b0c6f92073f8ea736ed53cbdf97b7f8cc43ac44f83e20971c1b4880e84cae233.jpg)

![c0463b7e2e2882850f13db84dc992fd40b4ae27072bbc2da4ef2e0268dc66c60.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/c0463b7e2e2882850f13db84dc992fd40b4ae27072bbc2da4ef2e0268dc66c60.jpg)

![ee39fab8622149b013d2aaf600d4cabb33549a1ea8fa6ce3f2ae59bddcb8b1cd.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/ee39fab8622149b013d2aaf600d4cabb33549a1ea8fa6ce3f2ae59bddcb8b1cd.jpg)

![f1b8eccceabd993930412ae25cebca0f40726b67cab921d1cc5ba2a23f7d3b88.jpg](../icml_results/6139_Plasticity%20Activation%20via%20Polar%20Operator_%20A%20Plug-in%20Method%20for%20Balancing%20Stability%20and%20Plasticity/tables/f1b8eccceabd993930412ae25cebca0f40726b67cab921d1cc5ba2a23f7d3b88.jpg)

<a id="paper-6090"></a>

## Temporal-Aware Reasoning Optimization for Video Temporal Grounding


### Images

![005a0075f5399f08ddfa3127e21d1734033ee15824d4a19785cb5305cc9e2c02.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/005a0075f5399f08ddfa3127e21d1734033ee15824d4a19785cb5305cc9e2c02.jpg)

![006b1634c3a22b9b7e08bc286ca2d460e56888fc6109867f2f4d0543d46af83e.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/006b1634c3a22b9b7e08bc286ca2d460e56888fc6109867f2f4d0543d46af83e.jpg)

![1dbe0a27a545ef8035e51f8d37403a36481422eb502f629cc6b3415cb4169517.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/1dbe0a27a545ef8035e51f8d37403a36481422eb502f629cc6b3415cb4169517.jpg)

![370e1159b224e94dddd63dbf54e8efd3691094f80319b5fee689b2788c342f92.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/370e1159b224e94dddd63dbf54e8efd3691094f80319b5fee689b2788c342f92.jpg)

![5deab3807b183afb1707e588a9935a467227f8b97510c1530226b3e5bc4427c4.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/5deab3807b183afb1707e588a9935a467227f8b97510c1530226b3e5bc4427c4.jpg)

![641e17149f7328c8e80a0ea4ba0d3c8b7cfd0e0aaa1dcd9b3dc066602d669da3.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/641e17149f7328c8e80a0ea4ba0d3c8b7cfd0e0aaa1dcd9b3dc066602d669da3.jpg)

![73000be87d2b3571429038a4c3fde4aeb114a7189f81937ea11b2b57cd10adcf.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/73000be87d2b3571429038a4c3fde4aeb114a7189f81937ea11b2b57cd10adcf.jpg)

![79815f78628f57fd79929c9f92f0fc21f37df4905170abf4c3a6a32fdf11fb8e.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/79815f78628f57fd79929c9f92f0fc21f37df4905170abf4c3a6a32fdf11fb8e.jpg)

![823a598fd1c6b5365b451589b60a70b68bb2c9d587b41660c1020ce9e4b178bb.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/823a598fd1c6b5365b451589b60a70b68bb2c9d587b41660c1020ce9e4b178bb.jpg)

![83d77a21653b4cf4b0de96f7b8eefec1a3ce46f4291332b4c8c69223756cbb69.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/83d77a21653b4cf4b0de96f7b8eefec1a3ce46f4291332b4c8c69223756cbb69.jpg)

![958d7d16f25517c843f5f0768d78f5a7345cc3eebcba5675e8628e62d20ca906.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/958d7d16f25517c843f5f0768d78f5a7345cc3eebcba5675e8628e62d20ca906.jpg)

![9cf10ed8454c7b555ae5c308abf57f6602783dcc9c37fd0c33460e540e193418.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/9cf10ed8454c7b555ae5c308abf57f6602783dcc9c37fd0c33460e540e193418.jpg)

![c9284e4f7be38762cff39f366aa67f2b9a6e75a1b982fe747444eac3eab2898f.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/c9284e4f7be38762cff39f366aa67f2b9a6e75a1b982fe747444eac3eab2898f.jpg)

![d811ec172c907fa89f4fe32551374ecaf735f24e496c327fa9fae88ccac15644.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/d811ec172c907fa89f4fe32551374ecaf735f24e496c327fa9fae88ccac15644.jpg)

![d9c52f753e64c3d70a67e9263fab562bdd7305e27072015b54ab67031c52c102.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/d9c52f753e64c3d70a67e9263fab562bdd7305e27072015b54ab67031c52c102.jpg)

![f13633ec51cfc81b2851b907dbe7a545fbef459298b855cd5318cfeddd51d8d4.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/f13633ec51cfc81b2851b907dbe7a545fbef459298b855cd5318cfeddd51d8d4.jpg)

![f335fe2611e0dbed8b11bf1fb46c1bf81defbfb85db7bbb995a6a2a26f360de6.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/images/f335fe2611e0dbed8b11bf1fb46c1bf81defbfb85db7bbb995a6a2a26f360de6.jpg)

### Tables

![2993e8a34939fecf946f1dc4b10740ac0059d17bf63844ad2d07c9e7e37cc72f.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/2993e8a34939fecf946f1dc4b10740ac0059d17bf63844ad2d07c9e7e37cc72f.jpg)

![29d49a14671161e47c8c09972e5e7a83856338a83406dcac7837cdc9e0fc8e07.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/29d49a14671161e47c8c09972e5e7a83856338a83406dcac7837cdc9e0fc8e07.jpg)

![62b9ae86ff8e64e3afb6b307235020a5521cba882d1f955dc111639f51b6d0bb.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/62b9ae86ff8e64e3afb6b307235020a5521cba882d1f955dc111639f51b6d0bb.jpg)

![6b1ccbbb9a1359bb1c6a647c69a4415902db89b0439ae39d007f8a29d3f61e6d.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/6b1ccbbb9a1359bb1c6a647c69a4415902db89b0439ae39d007f8a29d3f61e6d.jpg)

![74846d5f20bb9b3fd381331f4b1c77af30bc9b6253c10148941f4c50853310ed.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/74846d5f20bb9b3fd381331f4b1c77af30bc9b6253c10148941f4c50853310ed.jpg)

![7687809878f1433cee49d16a978d55e2422e2c02dda2256b0a562dbc10e7f028.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/7687809878f1433cee49d16a978d55e2422e2c02dda2256b0a562dbc10e7f028.jpg)

![8b711750cafa877ce793201445a4e76544461790ebeb7667496b287b4344ef0a.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/8b711750cafa877ce793201445a4e76544461790ebeb7667496b287b4344ef0a.jpg)

![92f544d00894b5619eb51fce646c64743bb51aac8d1113f4e0d3bc652298cde6.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/92f544d00894b5619eb51fce646c64743bb51aac8d1113f4e0d3bc652298cde6.jpg)

![9ad122a3cac29a0ad8ff2c7b89048b64a1a89c24589278826868c38888cbcfc3.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/9ad122a3cac29a0ad8ff2c7b89048b64a1a89c24589278826868c38888cbcfc3.jpg)

![efa570b329dffa28f50a50258e16272cce0053b9e69581f770d82f3854f2511c.jpg](../icml_results/6140_Temporal-Aware%20Reasoning%20Optimization%20for%20Video%20Temporal%20Grounding/tables/efa570b329dffa28f50a50258e16272cce0053b9e69581f770d82f3854f2511c.jpg)

<a id="paper-6091"></a>

## Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions


### Images

![049330c9f784bb436279bae98ce5812709ca18479ab600ad16591d3b0086f57c.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/049330c9f784bb436279bae98ce5812709ca18479ab600ad16591d3b0086f57c.jpg)

![0f808ef8c4de92bf6cca2cf863931e2a01ec6e31a762dae27c59f752f64ee352.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/0f808ef8c4de92bf6cca2cf863931e2a01ec6e31a762dae27c59f752f64ee352.jpg)

![333aae4d1839ad64de1a9da69b86210da0d3c1db552c34ca4bc00fb213433255.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/333aae4d1839ad64de1a9da69b86210da0d3c1db552c34ca4bc00fb213433255.jpg)

![38cf145253f0db29e959140f8ddaebed9108834e4feda1e7328ee6bd78d827d2.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/38cf145253f0db29e959140f8ddaebed9108834e4feda1e7328ee6bd78d827d2.jpg)

![530ab3c333d4d2208e3d73f5e2f02f71cfd3125c1afbb102a273c9fa54c16189.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/530ab3c333d4d2208e3d73f5e2f02f71cfd3125c1afbb102a273c9fa54c16189.jpg)

![85058905b6ed56178ee514a0e2750ab9310bc4935ae4d0c7a27ffe75aed6d3ab.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/85058905b6ed56178ee514a0e2750ab9310bc4935ae4d0c7a27ffe75aed6d3ab.jpg)

![89f4ea60e7cc8df1c7a94c27286c238977d2e808345883ec78c36a12b66aeaf6.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/89f4ea60e7cc8df1c7a94c27286c238977d2e808345883ec78c36a12b66aeaf6.jpg)

![b1807cbdd13ca70cd635643b254dd22a66ac4dc01f5a425f6d480fa69eea69b2.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/b1807cbdd13ca70cd635643b254dd22a66ac4dc01f5a425f6d480fa69eea69b2.jpg)

![b840463244f5bd5f1135a420792cca03f15bba7020f3d8086f354b89add030fc.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/b840463244f5bd5f1135a420792cca03f15bba7020f3d8086f354b89add030fc.jpg)

![d194111221d089a4e33a6a11a4c5dee3ed3f83dc61d3583b11213d85a60ed909.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/images/d194111221d089a4e33a6a11a4c5dee3ed3f83dc61d3583b11213d85a60ed909.jpg)

### Tables

![3581a306f7df8f693f089399f74078ee9b9641c0003c2e40419c23c5bd1090df.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/3581a306f7df8f693f089399f74078ee9b9641c0003c2e40419c23c5bd1090df.jpg)

![4e00893e1337fd5f169bf0b79d04d32c8a6d838d9775ce21358b19d42e1496ee.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/4e00893e1337fd5f169bf0b79d04d32c8a6d838d9775ce21358b19d42e1496ee.jpg)

![4e54f5a99771fddd39e9f14314239461d950d165798dd3f6bab8dd730c5a25e4.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/4e54f5a99771fddd39e9f14314239461d950d165798dd3f6bab8dd730c5a25e4.jpg)

![6aee0ffd9035db16a8f68a95aea711cd85464f1ba5262e2b8979a2e83c5b6ff9.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/6aee0ffd9035db16a8f68a95aea711cd85464f1ba5262e2b8979a2e83c5b6ff9.jpg)

![8fac3fb2df87795aea4c69cfdcddf86cc74245564805f9c26e9c980b961051b2.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/8fac3fb2df87795aea4c69cfdcddf86cc74245564805f9c26e9c980b961051b2.jpg)

![9b2fae1b2945bc12b8d6796aea72baebfc0eca52cd6e0982d99fc4f94aea83db.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/9b2fae1b2945bc12b8d6796aea72baebfc0eca52cd6e0982d99fc4f94aea83db.jpg)

![9b6119a0d33742f09f1625b65faf8b0ff4934f06e6585f296421e18c19125686.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/9b6119a0d33742f09f1625b65faf8b0ff4934f06e6585f296421e18c19125686.jpg)

![9c7f0e5d5562a00b5d2dee0c885867a7c9716179500ddb515efab9b59072c731.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/9c7f0e5d5562a00b5d2dee0c885867a7c9716179500ddb515efab9b59072c731.jpg)

![a88e6237765d5abaf16e3964c4721376f338fcdad8ac7220857af8480c529330.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/a88e6237765d5abaf16e3964c4721376f338fcdad8ac7220857af8480c529330.jpg)

![c68049adc52dc8152ff3fc3c3b63f201d347f5ef6e0d27e0d26ef2ec6a81d7d5.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/c68049adc52dc8152ff3fc3c3b63f201d347f5ef6e0d27e0d26ef2ec6a81d7d5.jpg)

![cb4ca4159e765981397b42811903aee81fe464f427f4d2c80e798b8ed9c83d2e.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/cb4ca4159e765981397b42811903aee81fe464f427f4d2c80e798b8ed9c83d2e.jpg)

![f81015e7f17ab4e139db4305c1a53ee582dbec3b60892257549a00ee6f0cb8b7.jpg](../icml_results/6141_Scheduling%20LLM%20Inference%20with%20Uncertainty-Aware%20Output%20Length%20Predictions/tables/f81015e7f17ab4e139db4305c1a53ee582dbec3b60892257549a00ee6f0cb8b7.jpg)

<a id="paper-6092"></a>

## Rethinking Gating Mechanism in Sparse MoE: Handling Arbitrary Modality Inputs with Confidence-Guided Gate


### Images

![0f41bf2157a240ddaddae5ca45550b7642506a668eb9239e95eeb2c87032b080.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/0f41bf2157a240ddaddae5ca45550b7642506a668eb9239e95eeb2c87032b080.jpg)

![197a2feab838a457854e85645edbfccc3800b7c0f0cc05cb781241674ac2ca56.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/197a2feab838a457854e85645edbfccc3800b7c0f0cc05cb781241674ac2ca56.jpg)

![299980f1fb821f9a082a7e730092e533bd758c1db74f36abaae69d84193a6540.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/299980f1fb821f9a082a7e730092e533bd758c1db74f36abaae69d84193a6540.jpg)

![31a167d257e3f011921499110af6bf8bc59c16a2dfc9229908131c938af7412a.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/31a167d257e3f011921499110af6bf8bc59c16a2dfc9229908131c938af7412a.jpg)

![57786b9607f69ab35cf984897765914905da2c1d30d896eddf7899cb80317cc0.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/57786b9607f69ab35cf984897765914905da2c1d30d896eddf7899cb80317cc0.jpg)

![63c6588f65e20f0b658d1a78ba40fc08109e681d03c1e48f97790ccbbfd71b56.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/63c6588f65e20f0b658d1a78ba40fc08109e681d03c1e48f97790ccbbfd71b56.jpg)

![892074ee674810e130e25e1cc2f1d3a719c7124aba0a775248cbf35df3be3eb1.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/892074ee674810e130e25e1cc2f1d3a719c7124aba0a775248cbf35df3be3eb1.jpg)

![b33029ce2afd21a6cdf6e740cc27e9758fd2dcdde2bc148b67aff1721938eecc.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/b33029ce2afd21a6cdf6e740cc27e9758fd2dcdde2bc148b67aff1721938eecc.jpg)

![b49751427317d1ef5a4afaaefdc302d64ac94c9ba36f71a13999d9ff6e0e953c.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/b49751427317d1ef5a4afaaefdc302d64ac94c9ba36f71a13999d9ff6e0e953c.jpg)

![b5448af45a80dd456a76153c421d4ad8fc7bfb34e75066a5d49495eaa8246cdb.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/b5448af45a80dd456a76153c421d4ad8fc7bfb34e75066a5d49495eaa8246cdb.jpg)

![bab264c7aaf74ed0e285934fa4d32c10d4dd54260614e5c3228b5925d2ea5280.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/bab264c7aaf74ed0e285934fa4d32c10d4dd54260614e5c3228b5925d2ea5280.jpg)

![cfba8689984e3291f4a35785d83818003f149df042f8fcd1f42891871adc7361.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/cfba8689984e3291f4a35785d83818003f149df042f8fcd1f42891871adc7361.jpg)

![db1cc8914476d119b590d52317fe9aadb33cbf90ca4514f9969536316ba2492d.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/db1cc8914476d119b590d52317fe9aadb33cbf90ca4514f9969536316ba2492d.jpg)

![dbd5b995b651b4182fcff71f1b0a1b962f3702d78d22cd8cddea3f4249f1a614.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/dbd5b995b651b4182fcff71f1b0a1b962f3702d78d22cd8cddea3f4249f1a614.jpg)

![fdae91b1a1ec35eaea63b913bc81ffd3d43b61c79ba05624fb5ebf20ebda2d39.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/fdae91b1a1ec35eaea63b913bc81ffd3d43b61c79ba05624fb5ebf20ebda2d39.jpg)

![ff41ce39de30e46494171fb92167fcba4b75c63b76078d68d11dcf7743433caa.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/images/ff41ce39de30e46494171fb92167fcba4b75c63b76078d68d11dcf7743433caa.jpg)

### Tables

![04d44abe8ea5f87e5c5e6c16e58d2811aadfb645589a9a534afb81305e7cf142.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/04d44abe8ea5f87e5c5e6c16e58d2811aadfb645589a9a534afb81305e7cf142.jpg)

![2082f29bc37e06bab8a9dc9fd19d5def9a859bd99081940a4878f558b29e1441.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/2082f29bc37e06bab8a9dc9fd19d5def9a859bd99081940a4878f558b29e1441.jpg)

![4413421cd2e4d0dd9f2adc84382bf83f15b7efa9f2d2c32b2e4a183ffe33352a.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/4413421cd2e4d0dd9f2adc84382bf83f15b7efa9f2d2c32b2e4a183ffe33352a.jpg)

![474d259392eac1b57e23c367877db727c72efc7b3c5d4abe67beb4c815f35049.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/474d259392eac1b57e23c367877db727c72efc7b3c5d4abe67beb4c815f35049.jpg)

![6e3a8e15d6c908005f28dcea5899ef1f5e7f392ce46dca98d1aa562edcafd15e.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/6e3a8e15d6c908005f28dcea5899ef1f5e7f392ce46dca98d1aa562edcafd15e.jpg)

![8500738da4c861d0c20bd4c1bd3be82e663d50a71e43d6293a6cb49f0288b05b.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/8500738da4c861d0c20bd4c1bd3be82e663d50a71e43d6293a6cb49f0288b05b.jpg)

![a1159775a85702e2583b29651abb57ef1ed5b0913799031ccbcdf5cf2e59380b.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/a1159775a85702e2583b29651abb57ef1ed5b0913799031ccbcdf5cf2e59380b.jpg)

![b42dcd8ba12e0632274f177fc65ebaf3c6f42f948aedf7f8b3aff46974691677.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/b42dcd8ba12e0632274f177fc65ebaf3c6f42f948aedf7f8b3aff46974691677.jpg)

![b6cce9c9450169ae3835847052ec44983af84b4db0b141f638d731e267696036.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/b6cce9c9450169ae3835847052ec44983af84b4db0b141f638d731e267696036.jpg)

![b9577e079d3e0b772e4c33105eed3b1c94e7b104c003081f2d3ca8e0bf847cbf.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/b9577e079d3e0b772e4c33105eed3b1c94e7b104c003081f2d3ca8e0bf847cbf.jpg)

![cb44310449ccb55e949b2ed20d4d4ba95f3e4e823948e7f6d6bc5fcf4ef4943b.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/cb44310449ccb55e949b2ed20d4d4ba95f3e4e823948e7f6d6bc5fcf4ef4943b.jpg)

![d1fd69b4e0a1c221a3b4383bfbfd1c799abbd20e4142558a441b408c8495e9bf.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/d1fd69b4e0a1c221a3b4383bfbfd1c799abbd20e4142558a441b408c8495e9bf.jpg)

![d41b7c516b051ad1f018ee1578395e0ca44ac307cd666cee9e50a8e0e5aab5e7.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/d41b7c516b051ad1f018ee1578395e0ca44ac307cd666cee9e50a8e0e5aab5e7.jpg)

![d60c40ba3f002461abdb5bb75b64b22f12e94a9e71eb5ad905f1a86554b6e0f9.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/d60c40ba3f002461abdb5bb75b64b22f12e94a9e71eb5ad905f1a86554b6e0f9.jpg)

![dc61b63913302b19922e281fadec08032cc2b468e8f23deaec1f76e4bd4f5f40.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/dc61b63913302b19922e281fadec08032cc2b468e8f23deaec1f76e4bd4f5f40.jpg)

![f1769556e7090a6aaed21b7dca1d6f36308b03da1110853f36b84558539c836d.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/f1769556e7090a6aaed21b7dca1d6f36308b03da1110853f36b84558539c836d.jpg)

![f9a3f1a1d1be05fb918e953bb37d79b3d1dface583c6934c103ac2ee833944ad.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/f9a3f1a1d1be05fb918e953bb37d79b3d1dface583c6934c103ac2ee833944ad.jpg)

![fd65dd83786b5db0313853049f70b649f7219d55c402f9713b8046fb11f55489.jpg](../icml_results/6142_Rethinking%20Gating%20Mechanism%20in%20Sparse%20MoE_%20Handling%20Arbitrary%20Modality%20Inputs%20with%20Confidence-Guided/tables/fd65dd83786b5db0313853049f70b649f7219d55c402f9713b8046fb11f55489.jpg)

<a id="paper-6093"></a>

## Theoretical Investigation on Inductive Bias of Isolation Forest


### Images

![1c197138e15149f07d94d1b996f03622d6ab94171f64cd394faf0581595ea5ce.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/1c197138e15149f07d94d1b996f03622d6ab94171f64cd394faf0581595ea5ce.jpg)

![273564bec9f97a105cad0457dda86f2037876e174b47e441278b252e1d24b95a.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/273564bec9f97a105cad0457dda86f2037876e174b47e441278b252e1d24b95a.jpg)

![3ba80c9de78c009d90e85eafd484ab0348019d10d9c6c7350b5d7418d89e5906.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/3ba80c9de78c009d90e85eafd484ab0348019d10d9c6c7350b5d7418d89e5906.jpg)

![449cf729ff32f4502b78ba036e6c49525de35ff3f7d164c9daff98adf2c323fc.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/449cf729ff32f4502b78ba036e6c49525de35ff3f7d164c9daff98adf2c323fc.jpg)

![5d0a3cf2cbb4fcc66d7ec3fbfdcf25365d617e8ba23e3f9ff75237539ca8e328.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/5d0a3cf2cbb4fcc66d7ec3fbfdcf25365d617e8ba23e3f9ff75237539ca8e328.jpg)

![f048e60d98f7bfb42be43653ee4d4116b601400c846f0f023b0a2e5d8949925b.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/f048e60d98f7bfb42be43653ee4d4116b601400c846f0f023b0a2e5d8949925b.jpg)

![f98aabe79001f9caca355e41fcd2923c048b73b2068c4f3d7cc0ed9059486c1d.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/images/f98aabe79001f9caca355e41fcd2923c048b73b2068c4f3d7cc0ed9059486c1d.jpg)

### Tables

![6e27cbb4d89e18a805d631ea7d42d5b69d04f833fba2a3de10c9c580ab2b2c5e.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/tables/6e27cbb4d89e18a805d631ea7d42d5b69d04f833fba2a3de10c9c580ab2b2c5e.jpg)

![986e377930ef7faa44015a977bb8e96f003be32a2da632de6f95ae368cdecfe0.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/tables/986e377930ef7faa44015a977bb8e96f003be32a2da632de6f95ae368cdecfe0.jpg)

![e70857ec14167b77cfd5cfca525980849b6e7c8b32e13979a86b7b4cbdd6966b.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/tables/e70857ec14167b77cfd5cfca525980849b6e7c8b32e13979a86b7b4cbdd6966b.jpg)

![faabdcebc26fa57aa095d089b3b6e30e57d30db122b18f0407c69ee9240e487a.jpg](../icml_results/6143_Theoretical%20Investigation%20on%20Inductive%20Bias%20of%20Isolation%20Forest/tables/faabdcebc26fa57aa095d089b3b6e30e57d30db122b18f0407c69ee9240e487a.jpg)

<a id="paper-6094"></a>

## Context-Aware Reasoner: Enhancing Contextual Reasoning in Multimodal Large Language Models


### Images

![0795d8f4d4d4c235ce145d31edc0358115c7b95e5ebdefe67e421634dad7133b.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/0795d8f4d4d4c235ce145d31edc0358115c7b95e5ebdefe67e421634dad7133b.jpg)

![1592f002dd96603ca98d6c755832b58c7a022656c9571428d255771f368e849e.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/1592f002dd96603ca98d6c755832b58c7a022656c9571428d255771f368e849e.jpg)

![1f226a567630b01f13d50b0b41eb164ab59aa3827efb3fa9f20e6f5741bfd7c4.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/1f226a567630b01f13d50b0b41eb164ab59aa3827efb3fa9f20e6f5741bfd7c4.jpg)

![469f8dc7d1516a18a8765ceb97774e376ac65b92c80091009ef1f3ca9556be48.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/469f8dc7d1516a18a8765ceb97774e376ac65b92c80091009ef1f3ca9556be48.jpg)

![6096061c41d50bf9d5e13b83eccc1cd789fad56649a2e3a2277a217f65deaae8.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/6096061c41d50bf9d5e13b83eccc1cd789fad56649a2e3a2277a217f65deaae8.jpg)

![631684d5ac1645596b239ad327c550643d92dd36841792db949624a4b648b6d3.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/631684d5ac1645596b239ad327c550643d92dd36841792db949624a4b648b6d3.jpg)

![814a4ca0b8da11e5a235630082dd6157b897c18387d7b715c203daa430d528f9.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/814a4ca0b8da11e5a235630082dd6157b897c18387d7b715c203daa430d528f9.jpg)

![989d4f233c6efba5ddfc0b9da9f1389164e8718fc9060e407c06e88a20014c40.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/989d4f233c6efba5ddfc0b9da9f1389164e8718fc9060e407c06e88a20014c40.jpg)

![b28bc65d204b87503e8c2450e4bc72ac8dec6ea68d0a35a49eecb8ba6f025f92.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/b28bc65d204b87503e8c2450e4bc72ac8dec6ea68d0a35a49eecb8ba6f025f92.jpg)

![b5a84ce2e4276a2183af3b8a8624b005174ce839c0f7314ec6fe95585ad51a4d.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/b5a84ce2e4276a2183af3b8a8624b005174ce839c0f7314ec6fe95585ad51a4d.jpg)

![c18b79fd46f770414a5b4a410042ae078f5c7ecac3a231eb2eedb94b3b50d67f.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/c18b79fd46f770414a5b4a410042ae078f5c7ecac3a231eb2eedb94b3b50d67f.jpg)

![d757c51a892d3eb2117a131901863f50e76fa1efe7344b80224fbef8a1c1360e.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/d757c51a892d3eb2117a131901863f50e76fa1efe7344b80224fbef8a1c1360e.jpg)

![e7f60e097a90cd9762db1afd3c742dc9de13b82f3f3ad4285e3271181a927fe9.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/e7f60e097a90cd9762db1afd3c742dc9de13b82f3f3ad4285e3271181a927fe9.jpg)

![efe2b793fc28523f70f2f08bd1a0e64e99708b2962c2431c1fb3cc7804e589ea.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/efe2b793fc28523f70f2f08bd1a0e64e99708b2962c2431c1fb3cc7804e589ea.jpg)

![f17d86ca32e4a032344661b76d135f5851207399e73f5e1e79df4cef8c1de0a9.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/f17d86ca32e4a032344661b76d135f5851207399e73f5e1e79df4cef8c1de0a9.jpg)

![f39eb03ff1509de5b9c39eb9ec7cac024230278057bef82238a20380f1e287c7.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/f39eb03ff1509de5b9c39eb9ec7cac024230278057bef82238a20380f1e287c7.jpg)

![fc1f55f4199b707b6524e1f1f2eb35c46d609d6fb8ebcb4b096d5ccf4a1aa8ad.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/images/fc1f55f4199b707b6524e1f1f2eb35c46d609d6fb8ebcb4b096d5ccf4a1aa8ad.jpg)

### Tables

![0280be6072336f293766d078c064fc7a91778d6c263e7f81951086381b487102.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/0280be6072336f293766d078c064fc7a91778d6c263e7f81951086381b487102.jpg)

![07dea6dc5c7745a4f088df42a16192c2261e60bc91fea8b5a94738a84291fddf.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/07dea6dc5c7745a4f088df42a16192c2261e60bc91fea8b5a94738a84291fddf.jpg)

![2463c1b221c45fb24e0a1120321e4ff87c202e1ff3c7778a06c3f88cf5982af1.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/2463c1b221c45fb24e0a1120321e4ff87c202e1ff3c7778a06c3f88cf5982af1.jpg)

![cd46efce17053ab6d216f11daf2eb2f32e2d10f80464940ead80983768775c6c.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/cd46efce17053ab6d216f11daf2eb2f32e2d10f80464940ead80983768775c6c.jpg)

![dbcd983add6ec13e27762922536994d7ca05b1e363d23f1190841defa9713296.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/dbcd983add6ec13e27762922536994d7ca05b1e363d23f1190841defa9713296.jpg)

![e952d98a7d85025ba6b12f8c72188b7a244f930f4f32d5471ad22b81889562bb.jpg](../icml_results/6144_Context-Aware%20Reasoner_%20Enhancing%20Contextual%20Reasoning%20in%20Multimodal%20Large%20Language%20Models/tables/e952d98a7d85025ba6b12f8c72188b7a244f930f4f32d5471ad22b81889562bb.jpg)

<a id="paper-6095"></a>

## DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs


### Images

![00a0277a77a44d8805bcb15f22a5cbff7a7a69b0d9bbb48520e58fa2f4109baa.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/00a0277a77a44d8805bcb15f22a5cbff7a7a69b0d9bbb48520e58fa2f4109baa.jpg)

![296338015ae564e7bcec931cef359bf14a5f5d479f127dd8e82c7f9b49d07ba2.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/296338015ae564e7bcec931cef359bf14a5f5d479f127dd8e82c7f9b49d07ba2.jpg)

![2da7621d89c329e476bbdd1c2234c70b7b0164259aed7ba710ab0cc016098da1.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/2da7621d89c329e476bbdd1c2234c70b7b0164259aed7ba710ab0cc016098da1.jpg)

![2f9ed69bc6a31f21f230995e66ea84a9447e25fd5a15d7a9028987696e0c16ea.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/2f9ed69bc6a31f21f230995e66ea84a9447e25fd5a15d7a9028987696e0c16ea.jpg)

![3b601891f451b55290923b2ace848917cc910f85afa3c55833cc810b95e03172.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/3b601891f451b55290923b2ace848917cc910f85afa3c55833cc810b95e03172.jpg)

![42503aeb07b30bc7bc43d2801743895f0f9efa7cf6f6bb4c6374951e42983192.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/42503aeb07b30bc7bc43d2801743895f0f9efa7cf6f6bb4c6374951e42983192.jpg)

![48d469ef26b70a9811a067c4a02d670029c8d812d965b73f0465eb0e68af299e.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/48d469ef26b70a9811a067c4a02d670029c8d812d965b73f0465eb0e68af299e.jpg)

![4dce90de9719f87507418de3d13241f253a6a21976720aacbbfa08973cad0e93.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/4dce90de9719f87507418de3d13241f253a6a21976720aacbbfa08973cad0e93.jpg)

![519f9c9cf125a2dc615c245ec9dc9aaf8bf65c75bfc345de50953e5f0e59722f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/519f9c9cf125a2dc615c245ec9dc9aaf8bf65c75bfc345de50953e5f0e59722f.jpg)

![525a5569f566491e571d71fa9f68ea1cf40c9b284e09b6eab83d2b09505970c3.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/525a5569f566491e571d71fa9f68ea1cf40c9b284e09b6eab83d2b09505970c3.jpg)

![66dddba1cfc88eb012551e2dc1db5625e7223e0fca594925abe19656f04be566.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/66dddba1cfc88eb012551e2dc1db5625e7223e0fca594925abe19656f04be566.jpg)

![678ebe2572b82e03192c7ff72c9f51c88502be9da17fe58efb0c7b3a730c6fdd.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/678ebe2572b82e03192c7ff72c9f51c88502be9da17fe58efb0c7b3a730c6fdd.jpg)

![67fd7b6e39f8c7167a4b3fc5903b2c8c7d698be1c5c0a7e063ff555524e4e2bd.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/67fd7b6e39f8c7167a4b3fc5903b2c8c7d698be1c5c0a7e063ff555524e4e2bd.jpg)

![7f36c5eeb42151edb69a33c57ec958cb4399d74c315d1020a3bd82caf215743f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/7f36c5eeb42151edb69a33c57ec958cb4399d74c315d1020a3bd82caf215743f.jpg)

![8073a0b9581211735ce5bb435860f8fbb781a51adae19aadd562f1ce5524434e.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/8073a0b9581211735ce5bb435860f8fbb781a51adae19aadd562f1ce5524434e.jpg)

![839368ab4cc61c9dbc631bc3a8ccbc27bd666be8e2486b0f278f047c9db40c45.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/839368ab4cc61c9dbc631bc3a8ccbc27bd666be8e2486b0f278f047c9db40c45.jpg)

![917219bdaa758adef2d3eb043d56b667041b6dacb044071e9cb5098ef50dad34.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/917219bdaa758adef2d3eb043d56b667041b6dacb044071e9cb5098ef50dad34.jpg)

![98591cb2f04bb1b785aebc20fadf20c1a504576701f2bc849d682eb5cda7baff.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/98591cb2f04bb1b785aebc20fadf20c1a504576701f2bc849d682eb5cda7baff.jpg)

![9e843ba41a92bfb31a3c838c9c112e3a539391bcfe267308a605660a06eb093f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/9e843ba41a92bfb31a3c838c9c112e3a539391bcfe267308a605660a06eb093f.jpg)

![a35b6a082b5aaa325591de053dbb6a4ebd8e7b0cacfd5baf5bdf9bc826963c2e.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/a35b6a082b5aaa325591de053dbb6a4ebd8e7b0cacfd5baf5bdf9bc826963c2e.jpg)

![a422e3d4922d6618af336063b31974bb497a6c8b2d3da05af9dabe9d1b95f87c.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/a422e3d4922d6618af336063b31974bb497a6c8b2d3da05af9dabe9d1b95f87c.jpg)

![a8aca81b69e73def7c96f0c916f03a54b7aae80f1f085dc909e661df807662b3.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/a8aca81b69e73def7c96f0c916f03a54b7aae80f1f085dc909e661df807662b3.jpg)

![b39b5a41420d86c9530718be21774933f02956ba29ef71fec0a2ee0dfed837a1.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/b39b5a41420d86c9530718be21774933f02956ba29ef71fec0a2ee0dfed837a1.jpg)

![b5e208ea744b29a6c62ee8e5521d437ebc8a2580a18f8cc3b274acd332401fd7.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/b5e208ea744b29a6c62ee8e5521d437ebc8a2580a18f8cc3b274acd332401fd7.jpg)

![bf588f02b0270fdf8b1caf7db037b97c2567474350d23b83bf9e21e734f1a52f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/bf588f02b0270fdf8b1caf7db037b97c2567474350d23b83bf9e21e734f1a52f.jpg)

![c03b19a47d9a2f0dc1c51d8cc309f2be303e30073c5ddcfba766b148c0419e02.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/c03b19a47d9a2f0dc1c51d8cc309f2be303e30073c5ddcfba766b148c0419e02.jpg)

![c515af57b96c798722433cedb8c45bd1b7d68e0709f250d9410a0150873e2532.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/c515af57b96c798722433cedb8c45bd1b7d68e0709f250d9410a0150873e2532.jpg)

![c63fab742154a069cdd5e3caafedbeb432895ebe993d9bf561224917b00d98b1.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/c63fab742154a069cdd5e3caafedbeb432895ebe993d9bf561224917b00d98b1.jpg)

![ce0f13fe3f560b704d9acc185a04cb33ac4bc38eb3bef626166144c67b1a3b05.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/ce0f13fe3f560b704d9acc185a04cb33ac4bc38eb3bef626166144c67b1a3b05.jpg)

![d264142fe1ab68b9a5373e3bfad6790031b8472197f7dac26a7cc965e5eae807.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/d264142fe1ab68b9a5373e3bfad6790031b8472197f7dac26a7cc965e5eae807.jpg)

![d882831823799790aca13732e70b62f22495e41478461e9f629c343467d3bfe2.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/d882831823799790aca13732e70b62f22495e41478461e9f629c343467d3bfe2.jpg)

![de8846e567e4bc2463068d601f69a4c4f9df0fef6ca59fc509abecc2c6c81a79.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/de8846e567e4bc2463068d601f69a4c4f9df0fef6ca59fc509abecc2c6c81a79.jpg)

![eaf720bd11d20f6d03cd2690db5a69f4271f294036526fd6fd89d4027b623510.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/eaf720bd11d20f6d03cd2690db5a69f4271f294036526fd6fd89d4027b623510.jpg)

![ee78fd73471367f5d20f28419e1f0c8d80de02b881b3590901bb6bab22b07740.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/images/ee78fd73471367f5d20f28419e1f0c8d80de02b881b3590901bb6bab22b07740.jpg)

### Tables

![0280e7db1de0ecb1ca9e3660359353b985529f1c5b335ace77597490d31ac37e.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/0280e7db1de0ecb1ca9e3660359353b985529f1c5b335ace77597490d31ac37e.jpg)

![03aff605e67670c7b6e551b9862482d54baca08710b5639ac4c26f4f2a8d476f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/03aff605e67670c7b6e551b9862482d54baca08710b5639ac4c26f4f2a8d476f.jpg)

![0f9b0c5acdd934daf758a4f150452d850caf84f1f556d4951b7f6f249badd11c.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/0f9b0c5acdd934daf758a4f150452d850caf84f1f556d4951b7f6f249badd11c.jpg)

![1df64bd6dabcdd781b0cc04d0ae51129a950f42458517b35ca13efd145dd0731.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/1df64bd6dabcdd781b0cc04d0ae51129a950f42458517b35ca13efd145dd0731.jpg)

![247b471c981b5a9db5d9d83e258795fe4dcaf4b4a602310fa42fc997ef1df6ee.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/247b471c981b5a9db5d9d83e258795fe4dcaf4b4a602310fa42fc997ef1df6ee.jpg)

![26433bf3d75fc8355286ca805425bcc176ea78c37209719f89f2b814be5f91dc.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/26433bf3d75fc8355286ca805425bcc176ea78c37209719f89f2b814be5f91dc.jpg)

![2d726b428dd4feb5d64c2f6f36108d9f071930d7bda260ca7a2cee879197a994.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/2d726b428dd4feb5d64c2f6f36108d9f071930d7bda260ca7a2cee879197a994.jpg)

![383b3c47ceb85b4ba567cbf2cfe999d45fd60059acf9558916a51d2e24758915.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/383b3c47ceb85b4ba567cbf2cfe999d45fd60059acf9558916a51d2e24758915.jpg)

![5fd3b03230b8860e63d523da0b16e0771ca3e1f5245776859d301463c2eff64b.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/5fd3b03230b8860e63d523da0b16e0771ca3e1f5245776859d301463c2eff64b.jpg)

![6078d62694824226e11d6b40bc0687d2eca1b024e3d584ffb9e82a9637d7b719.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/6078d62694824226e11d6b40bc0687d2eca1b024e3d584ffb9e82a9637d7b719.jpg)

![70de78d9a9cec34460ce179d675d5374585ad42a057cb71c22dbca8b5164c80e.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/70de78d9a9cec34460ce179d675d5374585ad42a057cb71c22dbca8b5164c80e.jpg)

![7b9cd2abe8b9dc574bc18fa506aa29ab31799bbd5768d9494aa6559a3f8988ad.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/7b9cd2abe8b9dc574bc18fa506aa29ab31799bbd5768d9494aa6559a3f8988ad.jpg)

![8e92c975d69b5cb64060192c5909a42e36859aca2986f3b81aefef2c6a864120.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/8e92c975d69b5cb64060192c5909a42e36859aca2986f3b81aefef2c6a864120.jpg)

![9c0a09240ac123df9a0ef27155d4520a216fcf7cedfa429d475e7eaee9b0debb.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/9c0a09240ac123df9a0ef27155d4520a216fcf7cedfa429d475e7eaee9b0debb.jpg)

![a2c7f868dd34edbf9397e2fcdb05cf088907cb8593986dbf9f80fee735a89f70.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/a2c7f868dd34edbf9397e2fcdb05cf088907cb8593986dbf9f80fee735a89f70.jpg)

![b4b988868bd24429dc7fb571fe057480d0139cde28d88a925df536e72235124a.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/b4b988868bd24429dc7fb571fe057480d0139cde28d88a925df536e72235124a.jpg)

![b7d3a34ee16e4e23d5adf35cae886277d88526cb3768e81cf282ba5110f70910.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/b7d3a34ee16e4e23d5adf35cae886277d88526cb3768e81cf282ba5110f70910.jpg)

![c8cee1f381205a4a41757ca85b3b3e1a0a38ebb73d8e3445950cf11fa3f9709f.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/c8cee1f381205a4a41757ca85b3b3e1a0a38ebb73d8e3445950cf11fa3f9709f.jpg)

![d0ecc7db5b9717a2b2442ef1f04c4ac7a4846d752ea53c471e7729fcbd4d69ff.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/d0ecc7db5b9717a2b2442ef1f04c4ac7a4846d752ea53c471e7729fcbd4d69ff.jpg)

![d403b83ac97c3ede0d516e02a8ffa88d3cdc0c83863a484a033851d5e41efdec.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/d403b83ac97c3ede0d516e02a8ffa88d3cdc0c83863a484a033851d5e41efdec.jpg)

![f6107ca0cc09f4c6a1e7f6e627d22ac2c145dacdbe1549ea3dfb73a6ae8d9e03.jpg](../icml_results/6145_DITRON_%20Distributed%20Multi-level%20Tiling%20Compiler%20for%20Parallel%20Tensor%20Programs/tables/f6107ca0cc09f4c6a1e7f6e627d22ac2c145dacdbe1549ea3dfb73a6ae8d9e03.jpg)

<a id="paper-6096"></a>

## Topology-Preserving Neural Operator Learning via Hodge Decomposition


### Images

![0017fa6fc794bf5bfc1d8bfda7acbb24d7f38bee288655280b30e2a113fb3325.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/0017fa6fc794bf5bfc1d8bfda7acbb24d7f38bee288655280b30e2a113fb3325.jpg)

![0d789967801f8aa56ccdca268cca0545a123e9532e31cd362f8a9bfee8018319.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/0d789967801f8aa56ccdca268cca0545a123e9532e31cd362f8a9bfee8018319.jpg)

![186e9cead760064e13a9208868043096a672ad8d93733430034f78d2e09efff1.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/186e9cead760064e13a9208868043096a672ad8d93733430034f78d2e09efff1.jpg)

![4c418a63842fbd25537222cfe282b9d65789599e1e2f3c5e57cc91945e680209.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/4c418a63842fbd25537222cfe282b9d65789599e1e2f3c5e57cc91945e680209.jpg)

![4fdbf11d950745509e0679b36cecd92f266520abbde68c51b3dfbcabd21f46b5.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/4fdbf11d950745509e0679b36cecd92f266520abbde68c51b3dfbcabd21f46b5.jpg)

![7ba61cb23db50a96bb2ebe82e01568f66bda4c3ffc341ff4b149bb9a5925068e.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/7ba61cb23db50a96bb2ebe82e01568f66bda4c3ffc341ff4b149bb9a5925068e.jpg)

![8ef85a4a00a348121bbcc06d01b625d40d2a1dc1c16966963af22b782c0697cd.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/8ef85a4a00a348121bbcc06d01b625d40d2a1dc1c16966963af22b782c0697cd.jpg)

![934b2ee0b5b50bee8586356f01451458c54031217967d18f9533a5aa574bfb16.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/934b2ee0b5b50bee8586356f01451458c54031217967d18f9533a5aa574bfb16.jpg)

![9c0e5a9b0183be2075a445f5396e691a7c06449a46b61f1b8a8798b07767855d.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/9c0e5a9b0183be2075a445f5396e691a7c06449a46b61f1b8a8798b07767855d.jpg)

![9e400af8a55df6939fad373cc280f4f9b38905fa46552e422a3faa826a16979d.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/9e400af8a55df6939fad373cc280f4f9b38905fa46552e422a3faa826a16979d.jpg)

![ae495e6214c37f0115c022c0f3891e4ac1942f5dc9ab7b652eb80c36ccf8dd1e.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/ae495e6214c37f0115c022c0f3891e4ac1942f5dc9ab7b652eb80c36ccf8dd1e.jpg)

![ba6190859653b2fee82bf6c699c7774adeb4f97d6ed97c02111a267fb48ccc1a.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/ba6190859653b2fee82bf6c699c7774adeb4f97d6ed97c02111a267fb48ccc1a.jpg)

![bf1c1952d7bcbf020336c93953e956c67f16b9345be68d4d9a44aa46df57ba38.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/bf1c1952d7bcbf020336c93953e956c67f16b9345be68d4d9a44aa46df57ba38.jpg)

![d377a8d9b36bb2fdf254c42b08b172dec9dd3d02ecc23c47c6510c4211c7c561.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/d377a8d9b36bb2fdf254c42b08b172dec9dd3d02ecc23c47c6510c4211c7c561.jpg)

![e4c226eb71a3d6d01fc29eb1d2a1ee7ddf6eec8c0c7613a1bdab02e8100b0e30.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/e4c226eb71a3d6d01fc29eb1d2a1ee7ddf6eec8c0c7613a1bdab02e8100b0e30.jpg)

![eaf4a839766913487c02e76b8363b204bc4d749b9dbf3c38f60473ab8bb881d4.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/eaf4a839766913487c02e76b8363b204bc4d749b9dbf3c38f60473ab8bb881d4.jpg)

![f27405617b32bad75c75c763866ffc3d7b46dde69a44b2027abad39a8f6e310c.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/f27405617b32bad75c75c763866ffc3d7b46dde69a44b2027abad39a8f6e310c.jpg)

![f535f19d8920902fceb6aa7471a7abbd874afeeb475b96f3dda8ad62415ff7f7.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/f535f19d8920902fceb6aa7471a7abbd874afeeb475b96f3dda8ad62415ff7f7.jpg)

![f973c699e1dd729d34f63afded4d77fb850554a30881603cb6475ff0f4b85d4f.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/images/f973c699e1dd729d34f63afded4d77fb850554a30881603cb6475ff0f4b85d4f.jpg)

### Tables

![0705bb96f947ec0ba07ed19515d62e9528f3a8fab031d583ff66b9ffe68788d3.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/0705bb96f947ec0ba07ed19515d62e9528f3a8fab031d583ff66b9ffe68788d3.jpg)

![0dc9c8e42364c90db4ae6c1fae33e1cc0029ea07d4f81084c4db167d15c41882.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/0dc9c8e42364c90db4ae6c1fae33e1cc0029ea07d4f81084c4db167d15c41882.jpg)

![31752e9fa56d9121b6fcae4210ece648cd0f06bc2888fed2c519055bcdd6d4b5.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/31752e9fa56d9121b6fcae4210ece648cd0f06bc2888fed2c519055bcdd6d4b5.jpg)

![418fc639847a8e6a6a85c7d6fbc84f46169fd212f6001dd536b624f2e65ec491.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/418fc639847a8e6a6a85c7d6fbc84f46169fd212f6001dd536b624f2e65ec491.jpg)

![87100bf67867fe061409267500876d37d4ddcc003c0c8cd729f9518ec1a92601.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/87100bf67867fe061409267500876d37d4ddcc003c0c8cd729f9518ec1a92601.jpg)

![a53b48245697c86799211fd3bc54835494781add81f35448103f0813e10a328f.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/a53b48245697c86799211fd3bc54835494781add81f35448103f0813e10a328f.jpg)

![d8123252095344514065ea1ed5c1056e0bff2c923b826b5e023f7d3fd2b4a6f7.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/d8123252095344514065ea1ed5c1056e0bff2c923b826b5e023f7d3fd2b4a6f7.jpg)

![fa3c34288f3efd6d4f2a8962bdc6cafee8103432311060082ecdd11d8455a4eb.jpg](../icml_results/6146_Topology-Preserving%20Neural%20Operator%20Learning%20via%20Hodge%20Decomposition/tables/fa3c34288f3efd6d4f2a8962bdc6cafee8103432311060082ecdd11d8455a4eb.jpg)

<a id="paper-6097"></a>

## From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity


### Images

![25128462ba4502dbd68683711720d4caf113335047f6961104bb8aaf00c79afc.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/images/25128462ba4502dbd68683711720d4caf113335047f6961104bb8aaf00c79afc.jpg)

![5e19b39d2e9d074b509dce761977df71346fbde99ca544bdffc8c5b1c09e6d7d.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/images/5e19b39d2e9d074b509dce761977df71346fbde99ca544bdffc8c5b1c09e6d7d.jpg)

![8a2ff6a2740766bf09c37cbc658550c0bf66eebea5e440388435ac38730d59ad.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/images/8a2ff6a2740766bf09c37cbc658550c0bf66eebea5e440388435ac38730d59ad.jpg)

![a6d256c3f7a57dc2ec8670e032ae275845d0829a69324df6309562d98174ebdc.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/images/a6d256c3f7a57dc2ec8670e032ae275845d0829a69324df6309562d98174ebdc.jpg)

![d3bc20954dea0fcd2e9f0be0c6f67f74797cdcc8fd4ef86065dd7bb8e46203f0.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/images/d3bc20954dea0fcd2e9f0be0c6f67f74797cdcc8fd4ef86065dd7bb8e46203f0.jpg)

### Tables

![3805011d756650a5a674a36bda390e7d08ba28e267bf7b097436a4c6c81e8974.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/3805011d756650a5a674a36bda390e7d08ba28e267bf7b097436a4c6c81e8974.jpg)

![3c55830a2a6e965efa16555d1c805d78a31e45e0a09cd4c8fe266e54a29bea79.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/3c55830a2a6e965efa16555d1c805d78a31e45e0a09cd4c8fe266e54a29bea79.jpg)

![46c5021739d3475c7878f85813310e93093a02f059db7fd876711e6f17d247f8.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/46c5021739d3475c7878f85813310e93093a02f059db7fd876711e6f17d247f8.jpg)

![5bb2b40cb1a2376464253c4c26daa49628236156914ca68a3c9300a07ceeafbe.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/5bb2b40cb1a2376464253c4c26daa49628236156914ca68a3c9300a07ceeafbe.jpg)

![5d0d0df77a29ff15e77075936d0bb45b6cbbce0352e7694ef763544e5bce5275.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/5d0d0df77a29ff15e77075936d0bb45b6cbbce0352e7694ef763544e5bce5275.jpg)

![69aa43fc5c3dd63aa5aa94fbdb138560734910c96f1da9a1e0c5ed12557cca69.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/69aa43fc5c3dd63aa5aa94fbdb138560734910c96f1da9a1e0c5ed12557cca69.jpg)

![742a1bc19fd08d3b565147bcdd4b6f02110c11fe80c9a65c73afe572635426a0.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/742a1bc19fd08d3b565147bcdd4b6f02110c11fe80c9a65c73afe572635426a0.jpg)

![76141c38247b2f068302bda7d5e6355bd9c22fec66af0fae07ef2ba0b7e1f8a0.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/76141c38247b2f068302bda7d5e6355bd9c22fec66af0fae07ef2ba0b7e1f8a0.jpg)

![79d023c1204d6e214e57c95232df6146b91e690417ec34b9f4f5d51da3680454.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/79d023c1204d6e214e57c95232df6146b91e690417ec34b9f4f5d51da3680454.jpg)

![7d662e18deb27b0266aa7c85489c16536024d001dc002144f51bb400f12a77cd.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/7d662e18deb27b0266aa7c85489c16536024d001dc002144f51bb400f12a77cd.jpg)

![e256c0dfa12c36cddccd55edddb64beddcfb4ccc68e7f4574644b4f6b667b804.jpg](../icml_results/6147_From%20Flat%20Facts%20to%20Sharp%20Hallucinations_%20Detecting%20Stubborn%20Errors%20via%20Gradient%20Sensitivity/tables/e256c0dfa12c36cddccd55edddb64beddcfb4ccc68e7f4574644b4f6b667b804.jpg)

<a id="paper-6098"></a>

## SubspacePath Pruner: Inference-time Pruning via Probe-based Representation–Parameter Coupling


### Images

![19b709870e5fd4f13e0f47268a8832932692bb176652065336f9b8c8131284aa.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/19b709870e5fd4f13e0f47268a8832932692bb176652065336f9b8c8131284aa.jpg)

![23e73ec5b6bf6abd68b14fe03f478e408995f290dfea146c8e20ea2acb9d84fb.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/23e73ec5b6bf6abd68b14fe03f478e408995f290dfea146c8e20ea2acb9d84fb.jpg)

![23e875493b6330b221a540e274fc353d585900823e8d782bfcd47ee4ed973a0f.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/23e875493b6330b221a540e274fc353d585900823e8d782bfcd47ee4ed973a0f.jpg)

![4546b37a314bbbd474a1283a2ced94740996c3bb3687bac40a8c35ae2500f238.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/4546b37a314bbbd474a1283a2ced94740996c3bb3687bac40a8c35ae2500f238.jpg)

![4bb47075a7bb422715d8b9124d5c4b57da28c0ea5201cf2a932e00be60d7a888.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/4bb47075a7bb422715d8b9124d5c4b57da28c0ea5201cf2a932e00be60d7a888.jpg)

![4c7c5d82f5f1955654d83a31af62544de8e53ca7347ae102d2adfcd289c07745.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/4c7c5d82f5f1955654d83a31af62544de8e53ca7347ae102d2adfcd289c07745.jpg)

![512d62964789843f3402336a632cd05c7bacf9e654e917b04eb0be4203bfb373.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/512d62964789843f3402336a632cd05c7bacf9e654e917b04eb0be4203bfb373.jpg)

![55cce0c8d26f4e27433982f9625d6538e165981048354f98e7b45a39a8c9fc98.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/55cce0c8d26f4e27433982f9625d6538e165981048354f98e7b45a39a8c9fc98.jpg)

![56bc69a6625db3cd2add8e5ebb2f3a4e094a2dcd0beae2b986b3cb7a14bfaaef.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/56bc69a6625db3cd2add8e5ebb2f3a4e094a2dcd0beae2b986b3cb7a14bfaaef.jpg)

![5d7767902357750affdbcce1983c53de7509e930d41f962c3834c5084fa4668a.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/5d7767902357750affdbcce1983c53de7509e930d41f962c3834c5084fa4668a.jpg)

![5edd4573a5793d5c1456e4bc9f03c955cfce4d101952f7e83a501c4d2575f47a.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/5edd4573a5793d5c1456e4bc9f03c955cfce4d101952f7e83a501c4d2575f47a.jpg)

![739ff6ffe5a6ded5a18c5b89db9ec72c4881f9a214778e27bbbf8aa991308722.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/739ff6ffe5a6ded5a18c5b89db9ec72c4881f9a214778e27bbbf8aa991308722.jpg)

![75edd959a1c393f46e2424791742ab1e00518a1ac95209e4c312c51c874dc666.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/75edd959a1c393f46e2424791742ab1e00518a1ac95209e4c312c51c874dc666.jpg)

![81c759f6aff197af472881662136929b1163b77f2f5ebada30cd3ef0f9667288.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/81c759f6aff197af472881662136929b1163b77f2f5ebada30cd3ef0f9667288.jpg)

![8da708bd10efd1e3d6908474674a143ab9c220c60571d782a8309ac38faea691.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/8da708bd10efd1e3d6908474674a143ab9c220c60571d782a8309ac38faea691.jpg)

![90bbe1df39e802af90eb4ec1c2b7cdd3215f79d50aa5d03f9abb2b38e8c2ee8e.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/90bbe1df39e802af90eb4ec1c2b7cdd3215f79d50aa5d03f9abb2b38e8c2ee8e.jpg)

![9dcdd253d98d8ae589b0f77f9807d56ba2f657cc834563754876b87695c2ba58.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/9dcdd253d98d8ae589b0f77f9807d56ba2f657cc834563754876b87695c2ba58.jpg)

![ac5b17cb6369bc7113b39a15ba6d76286475a21f6108e70022a175b1bf30e732.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/ac5b17cb6369bc7113b39a15ba6d76286475a21f6108e70022a175b1bf30e732.jpg)

![bd53eb883aa1bfbe8600297f827d245e950b87d509cd4b00034b561497cbcf60.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/bd53eb883aa1bfbe8600297f827d245e950b87d509cd4b00034b561497cbcf60.jpg)

![c41161f49600ff4c60aa88f0cbc8c7f014673c79f74219970ec5270128f77502.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/c41161f49600ff4c60aa88f0cbc8c7f014673c79f74219970ec5270128f77502.jpg)

![c52f2f174ec5a408fdbb63656802ced2643b15cca5321ae714eacc25009064f4.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/c52f2f174ec5a408fdbb63656802ced2643b15cca5321ae714eacc25009064f4.jpg)

![c9893cc62ba1750cd4e139e43564079ac82d55d1afca670a40880a891340a030.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/c9893cc62ba1750cd4e139e43564079ac82d55d1afca670a40880a891340a030.jpg)

![d321516e227a9a916832cf51c30854879b6ac379173deb614a39f9d6c9359199.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/d321516e227a9a916832cf51c30854879b6ac379173deb614a39f9d6c9359199.jpg)

![d77a60a77216a56427889cbf747c2ff411a4af9ceb06e2b7d6d82a53d9357151.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/d77a60a77216a56427889cbf747c2ff411a4af9ceb06e2b7d6d82a53d9357151.jpg)

![e1a4beca337a74bc0e9bd8d079030caaf7939d505f883ff31ddd48af09620466.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/e1a4beca337a74bc0e9bd8d079030caaf7939d505f883ff31ddd48af09620466.jpg)

![eebd4cf6d3de27ce5ebd3422f99ede77b70004b2b13cf25479953c3922c12e5e.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/images/eebd4cf6d3de27ce5ebd3422f99ede77b70004b2b13cf25479953c3922c12e5e.jpg)

### Tables

![34e34af17bbf3cf177665f73c1e08eb090ad01d776245ee07385921463d7c379.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/34e34af17bbf3cf177665f73c1e08eb090ad01d776245ee07385921463d7c379.jpg)

![433e20967aa599112a4cb73ab5bf582549d81e6afcf8f479d0705fa9c4e1f7a2.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/433e20967aa599112a4cb73ab5bf582549d81e6afcf8f479d0705fa9c4e1f7a2.jpg)

![51dc729783ec475e4448a4300d572168f436b2a64f952f1360d21bcd532ecad4.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/51dc729783ec475e4448a4300d572168f436b2a64f952f1360d21bcd532ecad4.jpg)

![55d89f355c8e1419a9e6d3b479ca5c6fccc7cceba1d76fe9ddd2e63b8484378e.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/55d89f355c8e1419a9e6d3b479ca5c6fccc7cceba1d76fe9ddd2e63b8484378e.jpg)

![5ef26fb8d47ca80f0f66a50647cab78454dd61fc16f81bbfce3a82d2fec68848.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/5ef26fb8d47ca80f0f66a50647cab78454dd61fc16f81bbfce3a82d2fec68848.jpg)

![7b4d26d5d4457e96748693dbc38e76c3adb39f4d9397297735a41b0a429ecb8b.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/7b4d26d5d4457e96748693dbc38e76c3adb39f4d9397297735a41b0a429ecb8b.jpg)

![a7a6abf31abe5c9695ae77a150678916f2f436ad243f9c1ba7921851668ed320.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/a7a6abf31abe5c9695ae77a150678916f2f436ad243f9c1ba7921851668ed320.jpg)

![adfd9eb7e81fee0a96e255cfab3de62d48c6e80a78e767e75bbef5d25c168237.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/adfd9eb7e81fee0a96e255cfab3de62d48c6e80a78e767e75bbef5d25c168237.jpg)

![af49d87e4c92678fa56421e7da98cd47a8f58a10c697d61689514863da6561fe.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/af49d87e4c92678fa56421e7da98cd47a8f58a10c697d61689514863da6561fe.jpg)

![b70fab394fcd406664a85bb09eb2edbde80058b171c59c9767f6088de023952e.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/b70fab394fcd406664a85bb09eb2edbde80058b171c59c9767f6088de023952e.jpg)

![cab50c352a538a7cbf2172360cd17562306445e984609b26efa2fa09203ed7ea.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/cab50c352a538a7cbf2172360cd17562306445e984609b26efa2fa09203ed7ea.jpg)

![d17a058ba193b521abe7b83fb2b52b74382d60912126969cf89f326d5cd70e36.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/d17a058ba193b521abe7b83fb2b52b74382d60912126969cf89f326d5cd70e36.jpg)

![e38541af6759d649845a17bc0119a82cc9c1c85911eb173e05daa3efd998d9d0.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/e38541af6759d649845a17bc0119a82cc9c1c85911eb173e05daa3efd998d9d0.jpg)

![e3c0b7299ef09a4b4150a86db132156451f553fe47a16a2ebc4cafd2e0246545.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/e3c0b7299ef09a4b4150a86db132156451f553fe47a16a2ebc4cafd2e0246545.jpg)

![e4e5fddc08dc1ab63d4aa072379db6f9d6f832988c319e6f60d1511bab46fe23.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/e4e5fddc08dc1ab63d4aa072379db6f9d6f832988c319e6f60d1511bab46fe23.jpg)

![eaca0705f76d21f73da782279608ed13240438c4f1e3a6fe0f84b388556ee841.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/eaca0705f76d21f73da782279608ed13240438c4f1e3a6fe0f84b388556ee841.jpg)

![ef1ec1b82ce06d71beae6ba919b4c43d55e11dae3e0301c85018091fc63d9c37.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/ef1ec1b82ce06d71beae6ba919b4c43d55e11dae3e0301c85018091fc63d9c37.jpg)

![f25705d790f009b055f1e4ea1217c5b3ce9301c9541a56937ce395c7be4cd3ad.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/f25705d790f009b055f1e4ea1217c5b3ce9301c9541a56937ce395c7be4cd3ad.jpg)

![fa1b17026353f2bdc8aec70db0d87197ba660e25bdc378bb829fc86c7b1e5b3d.jpg](../icml_results/6148_SubspacePath%20Pruner_%20Inference-time%20Pruning%20via%20Probe-based%20Representation%E2%80%93Parameter%20Coupling/tables/fa1b17026353f2bdc8aec70db0d87197ba660e25bdc378bb829fc86c7b1e5b3d.jpg)

<a id="paper-6099"></a>

## XDomainBench: Diagnosing Reasoning Collapse in High-Dimensional Scientific Knowledge Composition


### Images

![3da81c338455de8a264c739e9191f10d0713ad80f4a6654826dcaac7c99de6a4.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/3da81c338455de8a264c739e9191f10d0713ad80f4a6654826dcaac7c99de6a4.jpg)

![5ce2ee1c677df8b58039ba186650deaeea8bb91cff1eae20830c1c4f90c09f16.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/5ce2ee1c677df8b58039ba186650deaeea8bb91cff1eae20830c1c4f90c09f16.jpg)

![5e3591e3ad677163b7609d6f4f395974c17878a36b4a114b4970194e2046cd11.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/5e3591e3ad677163b7609d6f4f395974c17878a36b4a114b4970194e2046cd11.jpg)

![75b0f17d2afdb6433a760e40c851d27ee7f71c4443535f495f4211c8a82776e4.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/75b0f17d2afdb6433a760e40c851d27ee7f71c4443535f495f4211c8a82776e4.jpg)

![7a100f672f7c3180a73ab2cf371e125074230d765af1b70d723ac73624e41eb7.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/7a100f672f7c3180a73ab2cf371e125074230d765af1b70d723ac73624e41eb7.jpg)

![7edf36ced2bab89199f450321667c0e71e86dc692aeb973db63d412c35affcc2.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/7edf36ced2bab89199f450321667c0e71e86dc692aeb973db63d412c35affcc2.jpg)

![b3aef82c058f2e413c0c67c839d70a1e66581cf548c66da16aff8682091561f6.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/b3aef82c058f2e413c0c67c839d70a1e66581cf548c66da16aff8682091561f6.jpg)

![bd00cfa092a81224114abb4390fac5662919e1630ebd1630d7c681e01e97bc65.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/bd00cfa092a81224114abb4390fac5662919e1630ebd1630d7c681e01e97bc65.jpg)

![c1ab8c84d97b4c2ffb7b9e1b728c6f08f01c7b53740ff6c4e5182d2af0cf84c7.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/c1ab8c84d97b4c2ffb7b9e1b728c6f08f01c7b53740ff6c4e5182d2af0cf84c7.jpg)

![e7c67645cc2d3430d32baf1e1e36b49a092112f1f0cd52ba8820566cd628e165.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/e7c67645cc2d3430d32baf1e1e36b49a092112f1f0cd52ba8820566cd628e165.jpg)

![f6f6d7b8ec0d533934d119e1583185bd59fe44ad1f892ccb67b6c9d40544a8bb.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/images/f6f6d7b8ec0d533934d119e1583185bd59fe44ad1f892ccb67b6c9d40544a8bb.jpg)

### Tables

![03d64ddbceb3e47982164303a70dbc8af5f6946dc492327d784641b7aa1c7e72.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/03d64ddbceb3e47982164303a70dbc8af5f6946dc492327d784641b7aa1c7e72.jpg)

![16f495651b650f7ee64f2946334f0626702418b721d68d79478b1debc0d33ce7.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/16f495651b650f7ee64f2946334f0626702418b721d68d79478b1debc0d33ce7.jpg)

![26316ad6bccba41615416a91a76bd8ee8e91432c01d81ddf7a01ebb2158cdf1d.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/26316ad6bccba41615416a91a76bd8ee8e91432c01d81ddf7a01ebb2158cdf1d.jpg)

![30773ab8dfd2b4508662eed8269547652680597af55e159b8f109edcd01a9e75.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/30773ab8dfd2b4508662eed8269547652680597af55e159b8f109edcd01a9e75.jpg)

![33fff7db2396abd1dbbb0c20acf299f6e3eb563cdd67b08267e0e86d3193eb6c.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/33fff7db2396abd1dbbb0c20acf299f6e3eb563cdd67b08267e0e86d3193eb6c.jpg)

![3753544aff0b5dbc2602bd61ac531c52acb0a876784e8607a5c17a6679c5626b.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/3753544aff0b5dbc2602bd61ac531c52acb0a876784e8607a5c17a6679c5626b.jpg)

![3b29b94e4a10c16fed46fba9b6b32825fdb7663dc844fb91d5f1003220d3c245.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/3b29b94e4a10c16fed46fba9b6b32825fdb7663dc844fb91d5f1003220d3c245.jpg)

![3c17a721a979246bda7d761f3af49c56bca2e227d4dab0610bd05c8f34a66d5f.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/3c17a721a979246bda7d761f3af49c56bca2e227d4dab0610bd05c8f34a66d5f.jpg)

![44206117e1ce45ecfd937f862d3af369f173d20b5e822a6c744d2542454f6dc6.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/44206117e1ce45ecfd937f862d3af369f173d20b5e822a6c744d2542454f6dc6.jpg)

![46d2e5c9dd0a5147d5fee9531fd59ebb91bdb4f3c8c0af440335afa510eb8a68.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/46d2e5c9dd0a5147d5fee9531fd59ebb91bdb4f3c8c0af440335afa510eb8a68.jpg)

![4b7cd973d2c83b708030b73d919bc4e3f7b339d727fffd29b54520b653d027e2.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/4b7cd973d2c83b708030b73d919bc4e3f7b339d727fffd29b54520b653d027e2.jpg)

![64dafd986cff0c9d8433064db9fa2de3f333dcd5c319d600544fccd892e4070d.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/64dafd986cff0c9d8433064db9fa2de3f333dcd5c319d600544fccd892e4070d.jpg)

![69c8eee697a3bad21ba621530434bcebb6f6c574286fdceee046e2995606383f.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/69c8eee697a3bad21ba621530434bcebb6f6c574286fdceee046e2995606383f.jpg)

![6b2889a650f742018fa607b0db3e2aecf7df06eceb7d3c98ba1ae1c5d39418b2.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/6b2889a650f742018fa607b0db3e2aecf7df06eceb7d3c98ba1ae1c5d39418b2.jpg)

![79f9a0b1152fb30bee51f7cc2f92e11635134b4d6a92f6c3677776c3f1695800.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/79f9a0b1152fb30bee51f7cc2f92e11635134b4d6a92f6c3677776c3f1695800.jpg)

![8c018349cd6f92442bcb2a799c0b1598c800ca7462551ecc65b53fbec252f6f9.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/8c018349cd6f92442bcb2a799c0b1598c800ca7462551ecc65b53fbec252f6f9.jpg)

![92140b92963812138e91349ba4e73dc158448a58f934093ae1b2982fa822a15c.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/92140b92963812138e91349ba4e73dc158448a58f934093ae1b2982fa822a15c.jpg)

![95bda9361a48a60806ea5565dbfa33af35685ac738065dabb460c3b5682a194e.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/95bda9361a48a60806ea5565dbfa33af35685ac738065dabb460c3b5682a194e.jpg)

![97a3792819438e3ac5cc7bb659c11a12386408e1b0430586588b4239349be9af.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/97a3792819438e3ac5cc7bb659c11a12386408e1b0430586588b4239349be9af.jpg)

![b69c59aab697bb420874c723c12c62b856aec714213103ecee481ebb5bce8ad4.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/b69c59aab697bb420874c723c12c62b856aec714213103ecee481ebb5bce8ad4.jpg)

![b895941bceb3ff629c47fc7918cd8538583f8a2f3fe9a28199c13c1bfc03ab07.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/b895941bceb3ff629c47fc7918cd8538583f8a2f3fe9a28199c13c1bfc03ab07.jpg)

![be3422e75030bb4e6ab199b8ec466a050c3d541d0cd57da5458aa7018e2e5caf.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/be3422e75030bb4e6ab199b8ec466a050c3d541d0cd57da5458aa7018e2e5caf.jpg)

![ce772c872dad560e316c7cf9174dba1521bb61567616c9688bb83cee53173a8d.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/ce772c872dad560e316c7cf9174dba1521bb61567616c9688bb83cee53173a8d.jpg)

![dac0be04ff415c070a1418886e04998bac208554467d24f4938b6c72b33b933b.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/dac0be04ff415c070a1418886e04998bac208554467d24f4938b6c72b33b933b.jpg)

![ef6a2e19d89b50de0e959f87ad7757ff8f61cce6c88c5a590d33bf4a047f9af7.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/ef6a2e19d89b50de0e959f87ad7757ff8f61cce6c88c5a590d33bf4a047f9af7.jpg)

![f7b6c2f2eb6366bfd95a92fc631d6348f5a75b2ddc7fbbbced29a52272597949.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/f7b6c2f2eb6366bfd95a92fc631d6348f5a75b2ddc7fbbbced29a52272597949.jpg)

![fe77daf46d7a6b15f68e9be0536c3601d53571fac15d78ffbb45b9c1d7298a8b.jpg](../icml_results/6149_XDomainBench_%20Diagnosing%20Reasoning%20Collapse%20in%20High-Dimensional%20Scientific%20Knowledge%20Composition/tables/fe77daf46d7a6b15f68e9be0536c3601d53571fac15d78ffbb45b9c1d7298a8b.jpg)

<a id="paper-6100"></a>

## Learning 3D-Gaussian Simulators from RGB Videos


### Images

![075aea83259ef400a05b480fc04d02ad6229d3600ad7b82028004dace504483f.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/075aea83259ef400a05b480fc04d02ad6229d3600ad7b82028004dace504483f.jpg)

![08941fb5888d6d42fa086557202f60be8cf1d6260696357d8ededb4bded400e3.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/08941fb5888d6d42fa086557202f60be8cf1d6260696357d8ededb4bded400e3.jpg)

![08f13aa5583cab3e3e5a9b3bbb0c039080d9354e23023b16c5dc80c0537f14b1.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/08f13aa5583cab3e3e5a9b3bbb0c039080d9354e23023b16c5dc80c0537f14b1.jpg)

![112500f278021acfd55603198de7567b553b06ee38c79d03ad7b298ef49b9d54.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/112500f278021acfd55603198de7567b553b06ee38c79d03ad7b298ef49b9d54.jpg)

![1374d750729758a52acc2ac8d451809a73ce3290b1039e5dfa861d9595efcd9e.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/1374d750729758a52acc2ac8d451809a73ce3290b1039e5dfa861d9595efcd9e.jpg)

![16bb45e0891f64b99087e5adeb1a21c34b5823d11205fc8f3ca87e2c26dc0f55.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/16bb45e0891f64b99087e5adeb1a21c34b5823d11205fc8f3ca87e2c26dc0f55.jpg)

![1b9bc3276df9aa0e7515f06108de1fd117c52b7d6c231f0c991c5113932c6b15.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/1b9bc3276df9aa0e7515f06108de1fd117c52b7d6c231f0c991c5113932c6b15.jpg)

![283e61f074b045e3f92a9c0a3fc6c4763e405154e863aef2f3777d425c36d3a2.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/283e61f074b045e3f92a9c0a3fc6c4763e405154e863aef2f3777d425c36d3a2.jpg)

![2af7aad06746068bf4d82e3553e9665a37a6390341932f1f30745df712d6948d.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/2af7aad06746068bf4d82e3553e9665a37a6390341932f1f30745df712d6948d.jpg)

![347569dc2affb9ee52cb9e43aacc7b6c9529a901f59a71d92614ca9541e4c182.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/347569dc2affb9ee52cb9e43aacc7b6c9529a901f59a71d92614ca9541e4c182.jpg)

![368335900321bbc1790730e3310f05fcc6a05ad5f00fdcceb9977d992fbd77a2.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/368335900321bbc1790730e3310f05fcc6a05ad5f00fdcceb9977d992fbd77a2.jpg)

![39c7a8a412e5c0256c0b92d13970907fdb9f601b5822a65b444c499be6c00db2.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/39c7a8a412e5c0256c0b92d13970907fdb9f601b5822a65b444c499be6c00db2.jpg)

![3acba8c8cbb7ed8acfcbf0f5c75295e56c8e39978ad4cea41314c4d137ead2db.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/3acba8c8cbb7ed8acfcbf0f5c75295e56c8e39978ad4cea41314c4d137ead2db.jpg)

![45f089795c8e993fae39e2fae7d9c6236c087ffb9c27258bdd23d83911b71a85.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/45f089795c8e993fae39e2fae7d9c6236c087ffb9c27258bdd23d83911b71a85.jpg)

![46b20906735d05f08ff336a45b2e2533bc3129a63f09528d0f898fbb4d949bdf.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/46b20906735d05f08ff336a45b2e2533bc3129a63f09528d0f898fbb4d949bdf.jpg)

![51db4a46eff0a6b1a9ba2b04483cc1487aa7bef3c8a853c3c6fe653558c28e8d.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/51db4a46eff0a6b1a9ba2b04483cc1487aa7bef3c8a853c3c6fe653558c28e8d.jpg)

![66017de43943b5c094350b6cc8d99fa43eeab54a405449cb8fdb596dc1339f17.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/66017de43943b5c094350b6cc8d99fa43eeab54a405449cb8fdb596dc1339f17.jpg)

![7ebffdb531b7929184c11caad28baf4f6d57103bd420ffd68b4cb56903c291f0.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/7ebffdb531b7929184c11caad28baf4f6d57103bd420ffd68b4cb56903c291f0.jpg)

![8ff1ee21244c36d31dc4a74abf6fc307e0651aeb977e5382e8aefdacaf115068.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/8ff1ee21244c36d31dc4a74abf6fc307e0651aeb977e5382e8aefdacaf115068.jpg)

![938b125cc68769f5ceda0f5447d2c1c122c2b577907730f03c03d3f2ad4af0b8.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/938b125cc68769f5ceda0f5447d2c1c122c2b577907730f03c03d3f2ad4af0b8.jpg)

![97e9d164e25ff10aca38a30f4f2551ed5cf3e6e4e1f86cb370c4bc9ca7399f67.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/97e9d164e25ff10aca38a30f4f2551ed5cf3e6e4e1f86cb370c4bc9ca7399f67.jpg)

![acc84d6b9f6a292ffc77f18739ec6128ef44657927bd85f820bdf2eadd05f70f.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/acc84d6b9f6a292ffc77f18739ec6128ef44657927bd85f820bdf2eadd05f70f.jpg)

![af2ebfe179afc9b962eb68dc7c7e69083163096db2c99d79bd371255c619ed57.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/af2ebfe179afc9b962eb68dc7c7e69083163096db2c99d79bd371255c619ed57.jpg)

![b5d891464020c2206910dc03260bb0fec8616f1be6cd00a133a9e48d28095ec1.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/b5d891464020c2206910dc03260bb0fec8616f1be6cd00a133a9e48d28095ec1.jpg)

![b8cae71e88d0b8e191f4ca8ef8470422f601e6db881dc9e82b9804a92037e173.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/b8cae71e88d0b8e191f4ca8ef8470422f601e6db881dc9e82b9804a92037e173.jpg)

![b9f808fdce823e410e448274b9f8daf75fec5735a055bfa9958833f4bc44a6dc.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/b9f808fdce823e410e448274b9f8daf75fec5735a055bfa9958833f4bc44a6dc.jpg)

![bdec0941fbbec84d8521872e1a5fdcd1741968c40d2540426c81bea18124d454.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/bdec0941fbbec84d8521872e1a5fdcd1741968c40d2540426c81bea18124d454.jpg)

![c9f197a03f418acd6b72bb7c361fa5594cbc5ffe1f3fd9f76bb97b2b54dba556.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/c9f197a03f418acd6b72bb7c361fa5594cbc5ffe1f3fd9f76bb97b2b54dba556.jpg)

![d1e3f843c531660186aeda63355ed96c2dc425928756db49e95df6e7a15d972b.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/d1e3f843c531660186aeda63355ed96c2dc425928756db49e95df6e7a15d972b.jpg)

![dcf413ac57540c27012bed4fb681f609564a8b7da5b67f072237d850e905641b.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/dcf413ac57540c27012bed4fb681f609564a8b7da5b67f072237d850e905641b.jpg)

![f11d75c94bf64efd0a2da10ef79b2cd1866512d077f3c0e99b4505d7950bfbca.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/f11d75c94bf64efd0a2da10ef79b2cd1866512d077f3c0e99b4505d7950bfbca.jpg)

![f478eef97a5210343cf5434c858b185135adc482a99c0bd9c09537f98c8e0e7b.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/images/f478eef97a5210343cf5434c858b185135adc482a99c0bd9c09537f98c8e0e7b.jpg)

### Tables

![5511870ca2dc4f5f9b5c1efa9e46e556804ebab2a555cefbe369f72b9f720e5d.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/5511870ca2dc4f5f9b5c1efa9e46e556804ebab2a555cefbe369f72b9f720e5d.jpg)

![5992d09b0361cb86612813325dda2fd5e289d674aef3736faa2c5701f33613e5.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/5992d09b0361cb86612813325dda2fd5e289d674aef3736faa2c5701f33613e5.jpg)

![87b3e915adbf005584ee05e021a524e81e44a131a16adb851b58e0ad94ded820.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/87b3e915adbf005584ee05e021a524e81e44a131a16adb851b58e0ad94ded820.jpg)

![94cf2017986e650ac1d8678d436101be93ece76fe93243925189443eb2f05f47.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/94cf2017986e650ac1d8678d436101be93ece76fe93243925189443eb2f05f47.jpg)

![94f82263fd4507964cebf60cddd14fe808369392ea359f10ab4be8927c6f39d1.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/94f82263fd4507964cebf60cddd14fe808369392ea359f10ab4be8927c6f39d1.jpg)

![c1d593722929ad7a011105d4e48fadbb7ddb843b66ba363da9c6bb2ba2d7392c.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/c1d593722929ad7a011105d4e48fadbb7ddb843b66ba363da9c6bb2ba2d7392c.jpg)

![cea13a8225c66a68f563854519d2aa80561dd8e3fc33abdf80afc367f961bf7b.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/cea13a8225c66a68f563854519d2aa80561dd8e3fc33abdf80afc367f961bf7b.jpg)

![d30c94f24bfca393b22d411ead5b4ee206e8c772d372d504f87634ad76b84eaf.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/d30c94f24bfca393b22d411ead5b4ee206e8c772d372d504f87634ad76b84eaf.jpg)

![e170a4f93f57765d435bb79046357ef39f16d6cb66ebaa06053a81c34f186d5b.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/e170a4f93f57765d435bb79046357ef39f16d6cb66ebaa06053a81c34f186d5b.jpg)

![ec6515589e94fe38a5d882c2fe7b0307f263e232a6ac4a0ce19c80257fdb6288.jpg](../icml_results/6150_Learning%203D-Gaussian%20Simulators%20from%20RGB%20Videos/tables/ec6515589e94fe38a5d882c2fe7b0307f263e232a6ac4a0ce19c80257fdb6288.jpg)

<a id="paper-6101"></a>

## PGD-NO: A Neural Operator with Precomputed Geometry Decomposition for 3D Million-Scale Physics Simulations


### Images

![060060f21af6fa699998df6ccffdef10d98a5feccfdae187047cc5f466052930.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/060060f21af6fa699998df6ccffdef10d98a5feccfdae187047cc5f466052930.jpg)

![0a4de96e8827072aedcbd8a641305749c66a206831f29c77a6493d483a988e01.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/0a4de96e8827072aedcbd8a641305749c66a206831f29c77a6493d483a988e01.jpg)

![15e9c1035a6bdb10f55a5a3710f99e44eee748b67640d198c9025107db8dc2fc.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/15e9c1035a6bdb10f55a5a3710f99e44eee748b67640d198c9025107db8dc2fc.jpg)

![22a3ffd93f5ee21c9b03a67b045d7ba74acaf6d7c4b084645a2730792793062f.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/22a3ffd93f5ee21c9b03a67b045d7ba74acaf6d7c4b084645a2730792793062f.jpg)

![2d5f82bf8d064e780d6bb0f42fd2f831e92891663d9a51a221b0385a12468911.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/2d5f82bf8d064e780d6bb0f42fd2f831e92891663d9a51a221b0385a12468911.jpg)

![54030671f93c882576e81ebdc5cee2d83db7eb9847bcd2260d4245901d3cc035.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/54030671f93c882576e81ebdc5cee2d83db7eb9847bcd2260d4245901d3cc035.jpg)

![8dd049efcdc053f16e374ba83a31f5fda6da319eb7260e8b8a9c82fae1bce629.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/8dd049efcdc053f16e374ba83a31f5fda6da319eb7260e8b8a9c82fae1bce629.jpg)

![a91856c7af5aa734f4b89af4ddd70b378430e778306a552ec9ca15a75676d49f.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/a91856c7af5aa734f4b89af4ddd70b378430e778306a552ec9ca15a75676d49f.jpg)

![be2b840b1ff25eb9804c23edd63e207566c14eabd3b948e7677e085380058055.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/be2b840b1ff25eb9804c23edd63e207566c14eabd3b948e7677e085380058055.jpg)

![d61708adb6143c96b586e3986cfe1414496427e7771c85e5fad4a55d476407e7.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/images/d61708adb6143c96b586e3986cfe1414496427e7771c85e5fad4a55d476407e7.jpg)

### Tables

![3287a6dbed9da4470aa37311c9c9bd96e9a21188ddf534c60d9eb91c31642548.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/3287a6dbed9da4470aa37311c9c9bd96e9a21188ddf534c60d9eb91c31642548.jpg)

![6674f403157ce4276ac5127581bf21f5962eb4f211248135ae954bffd0af8f50.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/6674f403157ce4276ac5127581bf21f5962eb4f211248135ae954bffd0af8f50.jpg)

![8c088de16edcca1fcf503087d80817cdca019717abf42099ad8f3a2d6460abe5.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/8c088de16edcca1fcf503087d80817cdca019717abf42099ad8f3a2d6460abe5.jpg)

![bd658d38b496a6a23cc822aac39dc0af0c50d3aa85034241e21084b6ed10c013.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/bd658d38b496a6a23cc822aac39dc0af0c50d3aa85034241e21084b6ed10c013.jpg)

![bd6a65a602500886b636781861e9cb089c7298e03a8fbbe9ade6e6daa673f2da.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/bd6a65a602500886b636781861e9cb089c7298e03a8fbbe9ade6e6daa673f2da.jpg)

![cfa8d00f15268797b93ae67111ed338db226c59bd14d51c8f4142acc0993321c.jpg](../icml_results/6151_PGD-NO_%20A%20Neural%20Operator%20with%20Precomputed%20Geometry%20Decomposition%20for%203D%20Million-Scale%20Physics%20Simul/tables/cfa8d00f15268797b93ae67111ed338db226c59bd14d51c8f4142acc0993321c.jpg)

<a id="paper-6102"></a>

## 3D MeanFlow: One-Step Point Cloud Completion and Generation via Average-Velocity Transport


### Images

![06ebf29da83073009ab1c1480e6b931c350a1095d007cbcce2aae84baf049fc2.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/06ebf29da83073009ab1c1480e6b931c350a1095d007cbcce2aae84baf049fc2.jpg)

![3865843cfea9fc51d8d08b0170f748194961afa79427007698d00b66bf36ac61.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/3865843cfea9fc51d8d08b0170f748194961afa79427007698d00b66bf36ac61.jpg)

![427c8502c539c7b702a85f7ffe54b6f9bd601830cd2f6e43cac1c103e4289aac.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/427c8502c539c7b702a85f7ffe54b6f9bd601830cd2f6e43cac1c103e4289aac.jpg)

![68b5dac5d5892b35fb5530a34cb53916f87fa2a6ff234cc009a634b97f998963.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/68b5dac5d5892b35fb5530a34cb53916f87fa2a6ff234cc009a634b97f998963.jpg)

![6e862fae89f4135eb4ff7692ed07dbc9d89771fe7dc73d2758c4b14b7c9d4008.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/6e862fae89f4135eb4ff7692ed07dbc9d89771fe7dc73d2758c4b14b7c9d4008.jpg)

![77b4d8a87b919004d350c4ef11aac4156f72bd05aef918fa8cd84e02eaa1a997.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/77b4d8a87b919004d350c4ef11aac4156f72bd05aef918fa8cd84e02eaa1a997.jpg)

![83ebede297a791705ba08a0ad0f8f23e9c6f428e876636e2dd4bf87f4e75ed7f.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/83ebede297a791705ba08a0ad0f8f23e9c6f428e876636e2dd4bf87f4e75ed7f.jpg)

![8aa2ee4fe3ec0d52a8a7b01427ea27b604cefa953f4da4b9724e233491b93274.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/8aa2ee4fe3ec0d52a8a7b01427ea27b604cefa953f4da4b9724e233491b93274.jpg)

![d210a227b05b3365cc185bb6f96e19e373cdcc6159037ce52406cb8cf217e5a9.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/d210a227b05b3365cc185bb6f96e19e373cdcc6159037ce52406cb8cf217e5a9.jpg)

![d59fc02b3e42c7c693c3102c4dced462d2f44a609e03bcd2e74c8c0ea55a5e9a.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/d59fc02b3e42c7c693c3102c4dced462d2f44a609e03bcd2e74c8c0ea55a5e9a.jpg)

![eb0e27d39a95e82846c3ce63c978862fed723e5d32e72a39135289e51c2a13e9.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/eb0e27d39a95e82846c3ce63c978862fed723e5d32e72a39135289e51c2a13e9.jpg)

![f52dd4685339a34ee2621e58f418dc7db9ce744fec5944e847c425557c7f3f90.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/images/f52dd4685339a34ee2621e58f418dc7db9ce744fec5944e847c425557c7f3f90.jpg)

### Tables

![0573e3564018263426d11ab251b7a391902def0e8484fb51967076d7d38badfe.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/0573e3564018263426d11ab251b7a391902def0e8484fb51967076d7d38badfe.jpg)

![102c4687ff920aed814d4d03c6291be8750ce06c824d26519abb68a3e26b2fe2.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/102c4687ff920aed814d4d03c6291be8750ce06c824d26519abb68a3e26b2fe2.jpg)

![1b5acdff764b112019d6f8d30dbe40cf72fc02099424d61bea69aabefda116e0.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/1b5acdff764b112019d6f8d30dbe40cf72fc02099424d61bea69aabefda116e0.jpg)

![2383fe117df70733bb380bebc640b52b7e19f4c5adadebe4ade938dcaef3bc5f.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/2383fe117df70733bb380bebc640b52b7e19f4c5adadebe4ade938dcaef3bc5f.jpg)

![2ac730bd5ac7a5777eaf2c0606b40f624e7fdc40693968c4adaa2e3b3243df43.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/2ac730bd5ac7a5777eaf2c0606b40f624e7fdc40693968c4adaa2e3b3243df43.jpg)

![2c6f0e686ffa5be1f148b0840d7b36333cae53d7751a6e991f80a331a2371d82.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/2c6f0e686ffa5be1f148b0840d7b36333cae53d7751a6e991f80a331a2371d82.jpg)

![4db101d2b1f7015538bac2ab4f515adcfca14e74d5a1a43a3eda455327396745.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/4db101d2b1f7015538bac2ab4f515adcfca14e74d5a1a43a3eda455327396745.jpg)

![585890297dbfeaa41b8665447858422dfdf8e3a665e51068f097519533098be5.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/585890297dbfeaa41b8665447858422dfdf8e3a665e51068f097519533098be5.jpg)

![5d5dc98a75cd7e0869aacf83db0e70de65873b0734cdcfd2ac766943bcb57670.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/5d5dc98a75cd7e0869aacf83db0e70de65873b0734cdcfd2ac766943bcb57670.jpg)

![63fdccb7bd84ce90f9c8cf8444d2d56891dd8929f6d673dc2486fb4b39f3f260.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/63fdccb7bd84ce90f9c8cf8444d2d56891dd8929f6d673dc2486fb4b39f3f260.jpg)

![7a238be1bd0c22f23fa42700921db8081b821e39830e5630bb6c9832d1c4ff4e.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/7a238be1bd0c22f23fa42700921db8081b821e39830e5630bb6c9832d1c4ff4e.jpg)

![9005520a3d2a64ba7579ff8194ef8e24b8a5743d8f59e77f2416092a4ad6df4d.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/9005520a3d2a64ba7579ff8194ef8e24b8a5743d8f59e77f2416092a4ad6df4d.jpg)

![95cc2dbfa63e3b0bee711b4d40fbdf139004544d031204297b25b2f859974be4.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/95cc2dbfa63e3b0bee711b4d40fbdf139004544d031204297b25b2f859974be4.jpg)

![a6ed9164e0dad9f0e8bc16d7e775ba2c13e0091adb47302f48c0970b7d4956ba.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/a6ed9164e0dad9f0e8bc16d7e775ba2c13e0091adb47302f48c0970b7d4956ba.jpg)

![b6db7de1596bcbdc5e4530535ee04c3da2087aa4aaedbf0e09d32f51035d65ad.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/b6db7de1596bcbdc5e4530535ee04c3da2087aa4aaedbf0e09d32f51035d65ad.jpg)

![c365f6db8ed6422fb4651467c459d6938a077b17cf32ee8607d4b689b3fc2cdd.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/c365f6db8ed6422fb4651467c459d6938a077b17cf32ee8607d4b689b3fc2cdd.jpg)

![dc4c8ee991ce27fcb3a259b5852cc3f3a496f25921a04adcf9ffa4821359c98a.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/dc4c8ee991ce27fcb3a259b5852cc3f3a496f25921a04adcf9ffa4821359c98a.jpg)

![de28820339e404e7cd8c5847655e948b358f70eaf79d3a43e3ddd7cb70e3587e.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/de28820339e404e7cd8c5847655e948b358f70eaf79d3a43e3ddd7cb70e3587e.jpg)

![e09fc3f5c3eb0b7d1b6574e6abe5f585482dddbb27771df076d8e5c9461caf0d.jpg](../icml_results/6152_3D%20MeanFlow_%20One-Step%20Point%20Cloud%20Completion%20and%20Generation%20via%20Average-Velocity%20Transport/tables/e09fc3f5c3eb0b7d1b6574e6abe5f585482dddbb27771df076d8e5c9461caf0d.jpg)

<a id="paper-6103"></a>

## Better, Faster: Harnessing Self-Improvement in Large Reasoning Models


### Images

![0fd53db6d0cd90ee6aea0a9ec8c4ac41530892b916872ed409b8f8fa356d98b7.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/0fd53db6d0cd90ee6aea0a9ec8c4ac41530892b916872ed409b8f8fa356d98b7.jpg)

![1e5a906f2b958569ac5b296ab41114313e0ad6e585055bf5bc9ede69f73e7294.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/1e5a906f2b958569ac5b296ab41114313e0ad6e585055bf5bc9ede69f73e7294.jpg)

![307702f84e682aacd7541381cb1d96077f2063f0fa463d5c8b8c0c5b7c0c9ace.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/307702f84e682aacd7541381cb1d96077f2063f0fa463d5c8b8c0c5b7c0c9ace.jpg)

![61a41e2a081af5b293ddd4ab21c546c61d2ba4c649acc8d358d017d8fcfd8148.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/61a41e2a081af5b293ddd4ab21c546c61d2ba4c649acc8d358d017d8fcfd8148.jpg)

![8509e38eff7dee4add207f76a92784974300075c2cc63e2d69bf78eb5a34bf67.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/8509e38eff7dee4add207f76a92784974300075c2cc63e2d69bf78eb5a34bf67.jpg)

![8d820b5fb20171277244747a516d63718ae094f3afa670c8d9882182cdaecd8b.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/8d820b5fb20171277244747a516d63718ae094f3afa670c8d9882182cdaecd8b.jpg)

![d3ef9b983254efd641dad37be18515d44434093b8c80fa4faf88a666973db210.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/d3ef9b983254efd641dad37be18515d44434093b8c80fa4faf88a666973db210.jpg)

![d47245862804b54b050cc1714c83c47a853bceab2020959966c439d27c14d0d5.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/d47245862804b54b050cc1714c83c47a853bceab2020959966c439d27c14d0d5.jpg)

![f8fa14206a723dd01c137b1d8ef89397b6939f6d5c40eb6397a45e273a5f07a7.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/images/f8fa14206a723dd01c137b1d8ef89397b6939f6d5c40eb6397a45e273a5f07a7.jpg)

### Tables

![0cc91b7367f1085ca0223165af2e9b4eb52e7f30edbf66768d2bbb2a9861d552.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/0cc91b7367f1085ca0223165af2e9b4eb52e7f30edbf66768d2bbb2a9861d552.jpg)

![108f7c8c5a523d942aade0e006dfd00b43dc06d16500a883dec6307703ab06c8.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/108f7c8c5a523d942aade0e006dfd00b43dc06d16500a883dec6307703ab06c8.jpg)

![13d7eddbbd3e545c975e101d98681f7fd9b47ca5e8f3433c9facf1cd4368d57c.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/13d7eddbbd3e545c975e101d98681f7fd9b47ca5e8f3433c9facf1cd4368d57c.jpg)

![24a5a5aa19074e470da6140b9ecc84b4cfa338969d4d8cac452bed72cf1fb597.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/24a5a5aa19074e470da6140b9ecc84b4cfa338969d4d8cac452bed72cf1fb597.jpg)

![4a7e4d81c4fe9bdc41926d51714e62019fd86dc4bfae13ead309b2d2fe8d2d12.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/4a7e4d81c4fe9bdc41926d51714e62019fd86dc4bfae13ead309b2d2fe8d2d12.jpg)

![a4a5fb34fab4ef5e58d268609de1d4e2c5d1f099b58d44074e219dce6e3aeb97.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/a4a5fb34fab4ef5e58d268609de1d4e2c5d1f099b58d44074e219dce6e3aeb97.jpg)

![aaa9121c8699b44e7a49bac1c260f823842a2ad44beb3df92c81148ae4b4fa6a.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/aaa9121c8699b44e7a49bac1c260f823842a2ad44beb3df92c81148ae4b4fa6a.jpg)

![adf06b6850671f3d740478cc7f8daf2e6e004d64b585d6bb094a8a811c134f7d.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/adf06b6850671f3d740478cc7f8daf2e6e004d64b585d6bb094a8a811c134f7d.jpg)

![b0cddf0e2c0245ca99b64f3b01dfedcc76e8aaa729320c4ffebfc653a261771f.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/b0cddf0e2c0245ca99b64f3b01dfedcc76e8aaa729320c4ffebfc653a261771f.jpg)

![b8cfd809539d3c47237d8ab69141a05d2f289593d9710179833f8e2c80e6a010.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/b8cfd809539d3c47237d8ab69141a05d2f289593d9710179833f8e2c80e6a010.jpg)

![c48155646c6f8979fce34dc7a7fd8a77255b6345d2ed04624745feecf2b2a3d4.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/c48155646c6f8979fce34dc7a7fd8a77255b6345d2ed04624745feecf2b2a3d4.jpg)

![c53cc05fd4c89ec0a2efa2cd6191095569b292419c6bebef0a83eee5fb92a030.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/c53cc05fd4c89ec0a2efa2cd6191095569b292419c6bebef0a83eee5fb92a030.jpg)

![d619e17d3ec766574926b86ca021a302341c249196b118f7537877c6efb9e826.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/d619e17d3ec766574926b86ca021a302341c249196b118f7537877c6efb9e826.jpg)

![db7d3acc817a0a3a8ac7fc1dbbcffcda11a4448cfbceb53a165ad86ae403c0f6.jpg](../icml_results/6153_Better%2C%20Faster_%20Harnessing%20Self-Improvement%20in%20Large%20Reasoning%20Models/tables/db7d3acc817a0a3a8ac7fc1dbbcffcda11a4448cfbceb53a165ad86ae403c0f6.jpg)

<a id="paper-6104"></a>

## LightAVSeg: Lightweight Audio-Visual Segmentation


### Images

![0c096cd8c91ea9991563f7877d12ea3fe37e9e9c7cdbebb713e45f9e2ce7de80.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/0c096cd8c91ea9991563f7877d12ea3fe37e9e9c7cdbebb713e45f9e2ce7de80.jpg)

![10dcc768c840ef962072582f343eb221010a17d07b58271c444ac7baa3f2331a.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/10dcc768c840ef962072582f343eb221010a17d07b58271c444ac7baa3f2331a.jpg)

![7f6412ba42ad3e46501fa3c8703ffc22cdab4e2452a68d67b32bceb047b579f2.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/7f6412ba42ad3e46501fa3c8703ffc22cdab4e2452a68d67b32bceb047b579f2.jpg)

![a3894490040675febfe0495bdc4bfd21b368160e27e97fc896b61323b6e4618c.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/a3894490040675febfe0495bdc4bfd21b368160e27e97fc896b61323b6e4618c.jpg)

![cd490bccf973bce09e7633d7503efbf9b19cded4c0d4a2743cb9652296ecb36a.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/cd490bccf973bce09e7633d7503efbf9b19cded4c0d4a2743cb9652296ecb36a.jpg)

![e5a0a061407768d1c659ebd21c16f47c6eaa8249b3e94d960e44f3f9651640a8.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/e5a0a061407768d1c659ebd21c16f47c6eaa8249b3e94d960e44f3f9651640a8.jpg)

![f42d7922d5e195b94af128b7450ed81510244eaf067b71a6440c526b25ac23b2.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/f42d7922d5e195b94af128b7450ed81510244eaf067b71a6440c526b25ac23b2.jpg)

![fb92bd23270639965aa85f795be8a69432820de8dd12e3928cd88e4b9b18ce2c.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/images/fb92bd23270639965aa85f795be8a69432820de8dd12e3928cd88e4b9b18ce2c.jpg)

### Tables

![08252d7f789776954195d889fc00064e667ecb8db953d5a0496e129771d68fcb.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/08252d7f789776954195d889fc00064e667ecb8db953d5a0496e129771d68fcb.jpg)

![22c23386227e9709e30c5ba69f1ba48fa8c29edf8fa8445c72db77adb206e058.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/22c23386227e9709e30c5ba69f1ba48fa8c29edf8fa8445c72db77adb206e058.jpg)

![240fb459a39d1ee1c535e8813e3a2b37fa6215733520bc31f0dce9d3897f9624.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/240fb459a39d1ee1c535e8813e3a2b37fa6215733520bc31f0dce9d3897f9624.jpg)

![4d76c34e2d26758acd3592e40efcedbf652570379bc633933f07225f6b3fbe12.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/4d76c34e2d26758acd3592e40efcedbf652570379bc633933f07225f6b3fbe12.jpg)

![581b6060ece8be88e25b478acd4603b9b456ab7ffd0a8f65f5de309ff5488bfd.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/581b6060ece8be88e25b478acd4603b9b456ab7ffd0a8f65f5de309ff5488bfd.jpg)

![7a6fca7b199c289f4f8cdf2f5ec12562a8d358f08ab053c999c2781085f9538d.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/7a6fca7b199c289f4f8cdf2f5ec12562a8d358f08ab053c999c2781085f9538d.jpg)

![a05bab1ec521ee052408943f8d46d326d1ea1362b158cc818651cb66ce6630d8.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/a05bab1ec521ee052408943f8d46d326d1ea1362b158cc818651cb66ce6630d8.jpg)

![c9c334e672d750b954e6dae1d22d8eeec294c4db01a6bc961a06671b99258218.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/c9c334e672d750b954e6dae1d22d8eeec294c4db01a6bc961a06671b99258218.jpg)

![cb9c706742d687fb88e93e8de60f67a99bc8f9e77615b259a21771697d9552f7.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/cb9c706742d687fb88e93e8de60f67a99bc8f9e77615b259a21771697d9552f7.jpg)

![dacb8ccaadf0391d8a51eb83732fdcf9dc4fd241769ff01f16c39cfda354a2ec.jpg](../icml_results/6154_LightAVSeg_%20Lightweight%20Audio-Visual%20Segmentation/tables/dacb8ccaadf0391d8a51eb83732fdcf9dc4fd241769ff01f16c39cfda354a2ec.jpg)

<a id="paper-6105"></a>

## Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training

### Images

![079ce7f54c29e440433ed3e41b06ac070417df446859ddfee5fe7830263f546a.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/079ce7f54c29e440433ed3e41b06ac070417df446859ddfee5fe7830263f546a.jpg)

![095cd12af125cd436cba68ec25b5abbf7d28ff8c4384f75525ed7210f5273c03.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/095cd12af125cd436cba68ec25b5abbf7d28ff8c4384f75525ed7210f5273c03.jpg)

![237a05e454b4b52e8b98c78a292effbd70eb777cba5f59c6c4ce1371e575f44c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/237a05e454b4b52e8b98c78a292effbd70eb777cba5f59c6c4ce1371e575f44c.jpg)

![2621baaca78e41ea5bee1ba5899b6af8112db7f331b8466490a7abb14abce836.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/2621baaca78e41ea5bee1ba5899b6af8112db7f331b8466490a7abb14abce836.jpg)

![344c4c158ff1fe1377a1cdb1c07ef8fa06d980ea5560aec0c4a200f89bc0948c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/344c4c158ff1fe1377a1cdb1c07ef8fa06d980ea5560aec0c4a200f89bc0948c.jpg)

![34cfe3fc9f265a619c5848acb30fc5f878be1836bfb8685f66d56694049ca461.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/34cfe3fc9f265a619c5848acb30fc5f878be1836bfb8685f66d56694049ca461.jpg)

![3818a7ff6f9a44226f28cc62edf346a37aac84b570f4fe5d866c3fc3daa1b02d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/3818a7ff6f9a44226f28cc62edf346a37aac84b570f4fe5d866c3fc3daa1b02d.jpg)

![3abbd7ea7f1e32ca62d4550da59eb9447fa80eb5465b4f43f363374844177808.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/3abbd7ea7f1e32ca62d4550da59eb9447fa80eb5465b4f43f363374844177808.jpg)

![47318fdee13924079033b9387e800df3307cc6020e8c67df8e3c32a48ee8552b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/47318fdee13924079033b9387e800df3307cc6020e8c67df8e3c32a48ee8552b.jpg)

![523ab46bb20bee039762942d69a8cb8ca91e7dd86e7568484470617139c62987.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/523ab46bb20bee039762942d69a8cb8ca91e7dd86e7568484470617139c62987.jpg)

![54685fa476ea9ab11bcb41a5eccf8f5401633d1bbf779dbe164c03dc8c59ba58.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/54685fa476ea9ab11bcb41a5eccf8f5401633d1bbf779dbe164c03dc8c59ba58.jpg)

![55e8fb82e648d1b13e65f5c2af568a0acc9d60232c85141cbbfaa399e4a9fe79.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/55e8fb82e648d1b13e65f5c2af568a0acc9d60232c85141cbbfaa399e4a9fe79.jpg)

![58ff2a8ee4fba57a478cd5b89a3e1f2c3eedc1adfdc6449c558089cf6ef3e0d0.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/58ff2a8ee4fba57a478cd5b89a3e1f2c3eedc1adfdc6449c558089cf6ef3e0d0.jpg)

![5f8b84f128dec83dadc871c0287c2fdaa8ff65e96793f24705d5edf81eacf6dd.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/5f8b84f128dec83dadc871c0287c2fdaa8ff65e96793f24705d5edf81eacf6dd.jpg)

![684b9306c3f570bb01228a1a1db017af352fb8dee97f9011ac16d529973c602b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/684b9306c3f570bb01228a1a1db017af352fb8dee97f9011ac16d529973c602b.jpg)

![75c983c0a67891f94760e9b7eb05921eea078e5a9b0b2ac98cdffd9461fc23ff.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/75c983c0a67891f94760e9b7eb05921eea078e5a9b0b2ac98cdffd9461fc23ff.jpg)

![78a0cbcf0d80f8df3499e7272846ff6cf768539f9589ef3c5691c4da51f424d5.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/78a0cbcf0d80f8df3499e7272846ff6cf768539f9589ef3c5691c4da51f424d5.jpg)

![7caa81cdf2ff10abccd996904969ccc04b2cf2b01bb1467f23f3f2b515163190.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/7caa81cdf2ff10abccd996904969ccc04b2cf2b01bb1467f23f3f2b515163190.jpg)

![80bb31e876114c109c591ac245ae81893d974595396f858c01624e6987dd49be.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/80bb31e876114c109c591ac245ae81893d974595396f858c01624e6987dd49be.jpg)

![83b89ed4cded861b27562cc8595a1d6bf3ddc52e74f1f82c8a5451d3cdc8da37.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/83b89ed4cded861b27562cc8595a1d6bf3ddc52e74f1f82c8a5451d3cdc8da37.jpg)

![8e92e7da2fd7334710263802bfb3b9f8e1cda1c475d460c8d6667ed020a044c8.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/8e92e7da2fd7334710263802bfb3b9f8e1cda1c475d460c8d6667ed020a044c8.jpg)

![a263fd56cccc389486926eefbcb7a68901488cc23f564caeda75cd43ddb7c808.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/a263fd56cccc389486926eefbcb7a68901488cc23f564caeda75cd43ddb7c808.jpg)

![a7b5cbc1b8ad7b65e945bced8ea5d0093da43c4eb6276019b3995fd2aaf35b93.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/a7b5cbc1b8ad7b65e945bced8ea5d0093da43c4eb6276019b3995fd2aaf35b93.jpg)

![aa09821fe0cdddfed2e25a36ec82facd6b9a949d53816fa10647a2077a62751d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/aa09821fe0cdddfed2e25a36ec82facd6b9a949d53816fa10647a2077a62751d.jpg)

![af47f103b3f1c37124d675c5cb68aa83ef5556a016aa9bafab0290e7de4c0972.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/af47f103b3f1c37124d675c5cb68aa83ef5556a016aa9bafab0290e7de4c0972.jpg)

![b2a0c380b0e7190393ccc41f001b4556c0d34c98d3cd2caef4f0f3238f9af5d6.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/b2a0c380b0e7190393ccc41f001b4556c0d34c98d3cd2caef4f0f3238f9af5d6.jpg)

![bcdcc8f7dd1b4f180c2a1de4a975b3522629c98e8884cecb8d158367c6c4429c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/bcdcc8f7dd1b4f180c2a1de4a975b3522629c98e8884cecb8d158367c6c4429c.jpg)

![d7895f71b310b01934b319dc3852c2d0bd1853eb6bb58563ef435f0771fd04a7.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/d7895f71b310b01934b319dc3852c2d0bd1853eb6bb58563ef435f0771fd04a7.jpg)

![d8fe96acfc886c2b82fff19cbc9255e5a021d841b8379bfac5f78ae19204f6cf.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/d8fe96acfc886c2b82fff19cbc9255e5a021d841b8379bfac5f78ae19204f6cf.jpg)

![dc7556a41b33e0f9bb86295c6b9ff3d25e0755257a5b4fec2f9f58ce432aaeae.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/dc7556a41b33e0f9bb86295c6b9ff3d25e0755257a5b4fec2f9f58ce432aaeae.jpg)

![e53622ecd0dcd032b23d431235df2fc415af754e451cd6658ce0a29d534a35bc.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/e53622ecd0dcd032b23d431235df2fc415af754e451cd6658ce0a29d534a35bc.jpg)

![e700aceaec773830fc50ee870d36784223b62ca9c0861ef9bb0ec164ea45678c.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/e700aceaec773830fc50ee870d36784223b62ca9c0861ef9bb0ec164ea45678c.jpg)

![ee225cbe65dbadfaa8038ea26ca0b5a49d5a505902580456c0b6a6399e08394b.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/ee225cbe65dbadfaa8038ea26ca0b5a49d5a505902580456c0b6a6399e08394b.jpg)

![f43c8251db15abb7cd6614d086f9ce48f15cbd55712d7dfcd636e5d836ac392d.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/images/f43c8251db15abb7cd6614d086f9ce48f15cbd55712d7dfcd636e5d836ac392d.jpg)

### Tables

![012e087a0fd48c893926a3af9c1bccd341d1639ed9e903aa224e812c12332b01.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/012e087a0fd48c893926a3af9c1bccd341d1639ed9e903aa224e812c12332b01.jpg)

![1c8bba6d1a1f402204d1dd83788c0199bcfa7e15b4d4d5a82ec6835be02fd4c4.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/1c8bba6d1a1f402204d1dd83788c0199bcfa7e15b4d4d5a82ec6835be02fd4c4.jpg)

![28bd524492879fe767534f93024a9669b670972c135828c3ecb326aa01b78888.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/28bd524492879fe767534f93024a9669b670972c135828c3ecb326aa01b78888.jpg)

![3288a204fad0c9167a84ca766f3d531d92940254e7d6e45340272983cabac6b6.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/3288a204fad0c9167a84ca766f3d531d92940254e7d6e45340272983cabac6b6.jpg)

![8ace9761dcb23662098f544b51ece8578fbc932933a10dc98402bf45a41f8a48.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/8ace9761dcb23662098f544b51ece8578fbc932933a10dc98402bf45a41f8a48.jpg)

![94e20fc72ca2b0f12e684ef632c18f406a8bfc3fd9e987bebaa72e96e35a4971.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/94e20fc72ca2b0f12e684ef632c18f406a8bfc3fd9e987bebaa72e96e35a4971.jpg)

![db02b7f79cd2b63c82e9a50ba15ed294f92314e0f5b179261c4965a59dcb90b9.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/db02b7f79cd2b63c82e9a50ba15ed294f92314e0f5b179261c4965a59dcb90b9.jpg)

![f1fe4962914bae0389bdce335d1ba3aa02fa473b84cdb89467a1094654fe4bb0.jpg](../icml_results/6155_Learn%20to%20Think_%20Improving%20Multimodal%20Reasoning%20through%20Vision-Aware%20Self-Improvement%20Training/tables/f1fe4962914bae0389bdce335d1ba3aa02fa473b84cdb89467a1094654fe4bb0.jpg)
