# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 78 of 197

## 目录 (Table of Contents)

1. [Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](#paper-2542)
2. [Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation](#paper-2543)
3. [Search Space Synthesis for Parametric Functions](#paper-2544)
4. [Refined Analysis of Entropy-Regularized Actor-Critic](#paper-2545)
5. [Quantifying the noise sensitivity of the Wasserstein metric for images](#paper-2546)
6. [Graph Alignment for Benchmarking Graph Neural Networks and Learning Positional Encodings](#paper-2547)
7. [UniRRM: Unified Reasoning Reward Models Across Languages and Evaluation Paradigms](#paper-2548)
8. [DecoderTCR: Compositional Pretraining and Entropy-Guided Decoding for TCR-pMHC Interactions](#paper-2549)
9. [Predictive variational inference: Learn the predictively optimal posterior distribution](#paper-2550)
10. [MoCL: Metabolic Optimization for Curvature-Aware Continual Learning](#paper-2551)
11. [Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](#paper-2552)
12. [LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](#paper-2553)
13. [Adversarial Attack and Defense for Denoising Diffusion Sampling](#paper-2554)
14. [Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training](#paper-2555)
15. [Random Process Flow Matching: Generative Implicit Representations of Multivariate Random Fields](#paper-2556)
16. [Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](#paper-2557)
17. [Population-Aware Imitation Learning in Mean-field Games with Common Noise](#paper-2558)
18. [ConTSG-Bench: A Unified Benchmark for Conditional Time Series Generation](#paper-2559)
19. [Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy](#paper-2560)
20. [FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates](#paper-2561)
21. [Scaling Vision Transformers for Functional MRI with Flat Maps](#paper-2562)
22. [AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning](#paper-2563)
23. [Functional Decomposition and Shapley Interactions for Interpreting Survival Models](#paper-2564)
24. [OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data](#paper-2565)
25. [Nested birth-death processes are competitive with neural networks as time-dependent models of protein evolution](#paper-2566)
26. [Learning High-Dimensional Parity Functions with Product Networks using Gradient Descent](#paper-2567)
27. [Translation Heads: Disentangling meaning from language in LLM-based machine translation](#paper-2568)
28. [RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing](#paper-2569)
29. [Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections](#paper-2570)
30. [MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations](#paper-2571)
31. [MAFE: Enabling Equitable Algorithm Design in Multi-Agent Multi-Stage Decision-Making Systems](#paper-2572)
32. [Trust-Region Diffusion Policies for Massively Parallel On-Policy RL](#paper-2573)
33. [HieRD: Hierarchical Relational Distillation for Vision-Language Embedding Models](#paper-2574)

---


<a id="paper-2542"></a>

## Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed

### Images

![195fc88634996c17942800d1d27ff7ba7863c89f5a43f9ef23f5f4134b285105.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/195fc88634996c17942800d1d27ff7ba7863c89f5a43f9ef23f5f4134b285105.jpg)

![19ea77ff2e2b1a1c0dcac0dd16bffaae293e83fe6d9ad0fe8145452f20043887.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/19ea77ff2e2b1a1c0dcac0dd16bffaae293e83fe6d9ad0fe8145452f20043887.jpg)

![1bc279c7eeacb8b0e9d1a0e2d7f314e4aec9e2ec3a91c2f55e7c34a416ab0f05.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/1bc279c7eeacb8b0e9d1a0e2d7f314e4aec9e2ec3a91c2f55e7c34a416ab0f05.jpg)

![3d16b4a267f8c222bd55e54414164b7bcdd29d9cd1fc9e377ea640e7796eeb84.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/3d16b4a267f8c222bd55e54414164b7bcdd29d9cd1fc9e377ea640e7796eeb84.jpg)

![48adc70978ccbe4d542c3cc6544f8d2f999f5ef42c66f7a53bf983aab9b58d7e.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/48adc70978ccbe4d542c3cc6544f8d2f999f5ef42c66f7a53bf983aab9b58d7e.jpg)

![630958bc4e9cdb282cf64098d6f18b8f59c4e86054ec5be622a62c267ad03461.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/630958bc4e9cdb282cf64098d6f18b8f59c4e86054ec5be622a62c267ad03461.jpg)

![913321571a14cc67f2816ab52d11b67f3116854bbf3c9d839699d4b9f5bcf2e8.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/913321571a14cc67f2816ab52d11b67f3116854bbf3c9d839699d4b9f5bcf2e8.jpg)

![fdee141d981b26ed3fa6b72a82791a6e9f95377d59e3709afa675c3a47b2b0e2.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/images/fdee141d981b26ed3fa6b72a82791a6e9f95377d59e3709afa675c3a47b2b0e2.jpg)

### Tables

![6c6f48b91178b03199f98c8dd5c7e065b376a875603104a707721ae3bd833ab2.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/6c6f48b91178b03199f98c8dd5c7e065b376a875603104a707721ae3bd833ab2.jpg)

![786b2ddfe0e56dd566743231e9111fbfe80de35b295721f72d4e146358e61d77.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/786b2ddfe0e56dd566743231e9111fbfe80de35b295721f72d4e146358e61d77.jpg)

![8eef28a48d621f1579a63af6a7a0275417945bb4150c8b29fabbd0f296d8c3f5.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/8eef28a48d621f1579a63af6a7a0275417945bb4150c8b29fabbd0f296d8c3f5.jpg)

![b01cdabf869500c706dfac7cda6ae322abc08a909bf13d9007a927df02244040.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/b01cdabf869500c706dfac7cda6ae322abc08a909bf13d9007a927df02244040.jpg)

![b803edcc04cd1de2b33b1949177e4a0d33e650a7e51cd1b7c6db46de2768f542.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/b803edcc04cd1de2b33b1949177e4a0d33e650a7e51cd1b7c6db46de2768f542.jpg)

![b979478ef74f954e9641f8bd2db7acaf189f631b7ebc320d2ca73741252f17da.jpg](../icml_results/2573_Function-Valued%20Causal%20Influence%20in%20Nonlinear%20Time%20Series/tables/b979478ef74f954e9641f8bd2db7acaf189f631b7ebc320d2ca73741252f17da.jpg)

## Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed


### Images

![2db12e619914a7d1aaf8dc3b11d762803bc5197950d6a20efeec38cc21352aa4.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/2db12e619914a7d1aaf8dc3b11d762803bc5197950d6a20efeec38cc21352aa4.jpg)

![35a5bc04e9aa6204b7197f2b7dddf808cdc2a67dd618ef1921539ce2fe90659b.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/35a5bc04e9aa6204b7197f2b7dddf808cdc2a67dd618ef1921539ce2fe90659b.jpg)

![7ccf9d5bbe3a59ef4c9aad525dc1f8031aa4dfc1e8b3f5a7fd32de81474f7d89.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/7ccf9d5bbe3a59ef4c9aad525dc1f8031aa4dfc1e8b3f5a7fd32de81474f7d89.jpg)

![899773581784108e7c05e6f89de4183b0fd69a5a9597a7e48896551285da5e39.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/899773581784108e7c05e6f89de4183b0fd69a5a9597a7e48896551285da5e39.jpg)

![9565f5878a35fac4f132fabc3a0d13b68cf6560b6da4d553ab5c50a50ba72175.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/9565f5878a35fac4f132fabc3a0d13b68cf6560b6da4d553ab5c50a50ba72175.jpg)

![a0440f7e4e3026392e2ebe448db5f34c9f6e8fe9cb21f8f3da4cfceb04961469.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/a0440f7e4e3026392e2ebe448db5f34c9f6e8fe9cb21f8f3da4cfceb04961469.jpg)

![aa344f5f098d8b9980eb083daf8ddf3cbba272563a71472e3a3e6110c9b78e34.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/aa344f5f098d8b9980eb083daf8ddf3cbba272563a71472e3a3e6110c9b78e34.jpg)

![cc808fd780754a3c1a8b7bfe209aeb671a697027f2b5de98b5f1f2a17a481db0.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/cc808fd780754a3c1a8b7bfe209aeb671a697027f2b5de98b5f1f2a17a481db0.jpg)

![d0fdeeb42cd54f46323be127213bcce0971a6cc9e81ae1bc0cb1281ef709888d.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/d0fdeeb42cd54f46323be127213bcce0971a6cc9e81ae1bc0cb1281ef709888d.jpg)

![e0339550ca37ac3756f326e54a23cc4a0cfff3e9e66cc1649bf6c07a7795a5a7.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/e0339550ca37ac3756f326e54a23cc4a0cfff3e9e66cc1649bf6c07a7795a5a7.jpg)

![e1fbcb1bec4b70e8bd6a824639ba57516a7c570b2d97f40f801c1a727b74d09a.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/e1fbcb1bec4b70e8bd6a824639ba57516a7c570b2d97f40f801c1a727b74d09a.jpg)

![f38d7bd0fd003fddf170338676bee31c55127eff35caae152957c26583923482.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/f38d7bd0fd003fddf170338676bee31c55127eff35caae152957c26583923482.jpg)

![f6764280ec2ae9d24d29ec4695357063e6d4b3c69e04326f179d1d61de590042.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/f6764280ec2ae9d24d29ec4695357063e6d4b3c69e04326f179d1d61de590042.jpg)

![fd6e3cf66a13cf2b3db8bb87ff0b02dd136269b1ba798d4a4bac9531d81dba25.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/fd6e3cf66a13cf2b3db8bb87ff0b02dd136269b1ba798d4a4bac9531d81dba25.jpg)

![fe13505d4ff1413f3ab9b649bf9a745703dd1f2d1cd14b66781022c9767f38d1.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/images/fe13505d4ff1413f3ab9b649bf9a745703dd1f2d1cd14b66781022c9767f38d1.jpg)

### Tables

![0415fba1d1c144928aa1a6aeb34d19f3eeac1ef040da205efb7d0b2ab45f925c.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/0415fba1d1c144928aa1a6aeb34d19f3eeac1ef040da205efb7d0b2ab45f925c.jpg)

![06f73a1093c108c305b573175afbc05aa554dcf1064f21c96165f582690f19d5.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/06f73a1093c108c305b573175afbc05aa554dcf1064f21c96165f582690f19d5.jpg)

![0cc8e49d06bae23d27cf2f3644d4be1be443e99e31b2f1352fe33b11c3979df8.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/0cc8e49d06bae23d27cf2f3644d4be1be443e99e31b2f1352fe33b11c3979df8.jpg)

![2e0d4886b0c4bb5bbcc362f6b8c2c1a60b03d9778db8351da450f73b745c3852.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/2e0d4886b0c4bb5bbcc362f6b8c2c1a60b03d9778db8351da450f73b745c3852.jpg)

![31744e36849734f19cd4dc53b6b7a1698ff5aa72af9a2df9d1a1483c04585ea7.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/31744e36849734f19cd4dc53b6b7a1698ff5aa72af9a2df9d1a1483c04585ea7.jpg)

![34b4b60023219461ccf1cfd01b51885330219857e2b8e27c30188aa2ecb029d3.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/34b4b60023219461ccf1cfd01b51885330219857e2b8e27c30188aa2ecb029d3.jpg)

![6b3121d1f468af71f18ace681dcd23c5bd208b9c044f662172051b2d98db355a.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/6b3121d1f468af71f18ace681dcd23c5bd208b9c044f662172051b2d98db355a.jpg)

![7adcbb7b6b33a6151e68310e01cbba9837ba8c34f79543099b87a3aed8c135d7.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/7adcbb7b6b33a6151e68310e01cbba9837ba8c34f79543099b87a3aed8c135d7.jpg)

![8f5c0c2e3241974793586b4882d33f6e10e223fbc78c46c099d4db2a2d4b023e.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/8f5c0c2e3241974793586b4882d33f6e10e223fbc78c46c099d4db2a2d4b023e.jpg)

![a0d9e9396f9d5fd32b3e7bd3eb9e157f24721e0a923b7f0204af3fb8b053049c.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/a0d9e9396f9d5fd32b3e7bd3eb9e157f24721e0a923b7f0204af3fb8b053049c.jpg)

![aa160a0f10c989922a1bc753bfebe218072c69813894a2b74abcea22b578724a.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/aa160a0f10c989922a1bc753bfebe218072c69813894a2b74abcea22b578724a.jpg)

![ada230ae531b49c403c35d6b025efc43eb3be8b02815377f472d07b596022ac8.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/ada230ae531b49c403c35d6b025efc43eb3be8b02815377f472d07b596022ac8.jpg)

![b312c1e1935c0d3d0342913af0c3cd292f23611c2123fa1444e5679228ec06f2.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/b312c1e1935c0d3d0342913af0c3cd292f23611c2123fa1444e5679228ec06f2.jpg)

![b9ace8f1a4c6346df32e0e1da5dc4b162405422f9dff2018f46c816442e794d9.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/b9ace8f1a4c6346df32e0e1da5dc4b162405422f9dff2018f46c816442e794d9.jpg)

![beda60eef255d29adbdd9d01450c372d52f5af4e856d69272f31ee0be8eaf5eb.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/beda60eef255d29adbdd9d01450c372d52f5af4e856d69272f31ee0be8eaf5eb.jpg)

![c3d67ce3ef6211dbfbd62b5ba18553f92f52a910c49b5e0592bd648ff18c56f1.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/c3d67ce3ef6211dbfbd62b5ba18553f92f52a910c49b5e0592bd648ff18c56f1.jpg)

![cb8ec98efe54c90a934aa89fa5be8e3137ac151cde9d30c3196d02ce5d8ecf24.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/cb8ec98efe54c90a934aa89fa5be8e3137ac151cde9d30c3196d02ce5d8ecf24.jpg)

![d385f91c818b5ccece1a9604c86e5e85f099600d530cc6024a6178058c165620.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/d385f91c818b5ccece1a9604c86e5e85f099600d530cc6024a6178058c165620.jpg)

![d515a75849919b27b5acb4f894784a176d57c6bc53dd593bcec504baf994b7be.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/d515a75849919b27b5acb4f894784a176d57c6bc53dd593bcec504baf994b7be.jpg)

![e7971ee4e0e6bff002079b8ef649a5c60bb4f157b4db4343b3dd30e90d8004fe.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/e7971ee4e0e6bff002079b8ef649a5c60bb4f157b4db4343b3dd30e90d8004fe.jpg)

![f24400ede27b1e0701b053cafea3762d08bbfbd1d94c9e4209fe05fa00145945.jpg](../icml_results/2574_Safety%20Generalization%20Under%20Distribution%20Shift%20in%20Safe%20Reinforcement%20Learning_%20A%20Diabetes%20Testbed/tables/f24400ede27b1e0701b053cafea3762d08bbfbd1d94c9e4209fe05fa00145945.jpg)

<a id="paper-2543"></a>

## Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation


### Images

![1871372a4eb70f2806531fc72920d21c1c7892423cc8dd2af78873a4d236edc5.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/1871372a4eb70f2806531fc72920d21c1c7892423cc8dd2af78873a4d236edc5.jpg)

![1a0dbe3cb7140958a5b9a2083d9df368f283aa58e55475f396fbb96891e6971d.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/1a0dbe3cb7140958a5b9a2083d9df368f283aa58e55475f396fbb96891e6971d.jpg)

![1a466b41dc546fd11ac667939c5569d845a5333dc4214b0ec7769ad43a1e4d9e.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/1a466b41dc546fd11ac667939c5569d845a5333dc4214b0ec7769ad43a1e4d9e.jpg)

![1f6159e7ae915abb4b01c873aea1de9ed3eaf5c2eb512e018e537aa80a989ff5.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/1f6159e7ae915abb4b01c873aea1de9ed3eaf5c2eb512e018e537aa80a989ff5.jpg)

![22e6f798bfc4e9ae2944fb25198edcf460c3052feaf29547d940daad72ee2bbb.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/22e6f798bfc4e9ae2944fb25198edcf460c3052feaf29547d940daad72ee2bbb.jpg)

![32ec2b8e26d0f115532ae24eab8812167ac27de954c805c480909504a34c51e3.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/32ec2b8e26d0f115532ae24eab8812167ac27de954c805c480909504a34c51e3.jpg)

![3a74104cd9f017efd176ca8c42e16265136934c4f02009feda92f8dba3586947.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/3a74104cd9f017efd176ca8c42e16265136934c4f02009feda92f8dba3586947.jpg)

![8ce222558268c99b5f0117bfd386b89fdb9e392e68424cc5d18c7386e15e4d78.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/8ce222558268c99b5f0117bfd386b89fdb9e392e68424cc5d18c7386e15e4d78.jpg)

![96dd5e368771618b6512a1d213f9e1d1158b39eca0f8f02b50eb25ff0c603d0b.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/96dd5e368771618b6512a1d213f9e1d1158b39eca0f8f02b50eb25ff0c603d0b.jpg)

![a1c849aba01febef038fb225b8ed8ff442983eefb5c48590ffce0222baf8abc8.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/a1c849aba01febef038fb225b8ed8ff442983eefb5c48590ffce0222baf8abc8.jpg)

![a766653f55e0cc88e1385ad455c4a98c82f96c480efff1bc487e0b8a442f78fe.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/a766653f55e0cc88e1385ad455c4a98c82f96c480efff1bc487e0b8a442f78fe.jpg)

![ad6a61799cc7d75b4e3a1ef1ffa6fc05a751a9ceabe829ca84edc3b5e280d27b.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/ad6a61799cc7d75b4e3a1ef1ffa6fc05a751a9ceabe829ca84edc3b5e280d27b.jpg)

![e8707370bbd7a99de2f49abf7dbaf5cf000a2df2fa951317f46fb3fc995efd84.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/images/e8707370bbd7a99de2f49abf7dbaf5cf000a2df2fa951317f46fb3fc995efd84.jpg)

### Tables

![0c564410169678c1327bf282e9034a3b9729a85c53370bb7df73a404e51b661f.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/0c564410169678c1327bf282e9034a3b9729a85c53370bb7df73a404e51b661f.jpg)

![141cc3887a96f00ab6c2be7ae228f99a36af02e9a29280ceb16f3d02371e00d9.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/141cc3887a96f00ab6c2be7ae228f99a36af02e9a29280ceb16f3d02371e00d9.jpg)

![16b2051b6be79280d033521c525d03a90f895f919bcac90ec90e84bf4a5ffdc2.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/16b2051b6be79280d033521c525d03a90f895f919bcac90ec90e84bf4a5ffdc2.jpg)

![29a6077449f13ec9aff94b760af9ccbdabdd726fd66f1d869da1fe653862f7ce.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/29a6077449f13ec9aff94b760af9ccbdabdd726fd66f1d869da1fe653862f7ce.jpg)

![4c77d9831ebe9f36ca4374c22f27fe4dfc55de0ccb881b3a2d6d902fc66acb5c.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/4c77d9831ebe9f36ca4374c22f27fe4dfc55de0ccb881b3a2d6d902fc66acb5c.jpg)

![6761cdcade98724f120a25817fc573bd3852e2c02a2c692ccc4676058585dde1.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/6761cdcade98724f120a25817fc573bd3852e2c02a2c692ccc4676058585dde1.jpg)

![858846da2bfb74b06cc14ea8c0fe07157006f8f50c7de31487a23aac828ce348.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/858846da2bfb74b06cc14ea8c0fe07157006f8f50c7de31487a23aac828ce348.jpg)

![e2f5cf0787ca9f01dac10049603be7a5f21e964703509304e923f211b6e774b9.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/e2f5cf0787ca9f01dac10049603be7a5f21e964703509304e923f211b6e774b9.jpg)

![e7607dfbb83336cee8d53e36ebddb6ad4ecf97c516d75b481a413c37cce270b0.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/e7607dfbb83336cee8d53e36ebddb6ad4ecf97c516d75b481a413c37cce270b0.jpg)

![eae4d47c8c9f6e95a8fbe1a39c2d002e332a446e25b3f97899c36e8c860308ef.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/eae4d47c8c9f6e95a8fbe1a39c2d002e332a446e25b3f97899c36e8c860308ef.jpg)

![ed567bf09f4122e4c25e18353496ddb8c044cb9c323b3da1ed527c59f8327beb.jpg](../icml_results/2575_Breaking%20the%20Lock-in_%20Diversifying%20Text-to-Image%20Generation%20via%20Representation%20Modulation/tables/ed567bf09f4122e4c25e18353496ddb8c044cb9c323b3da1ed527c59f8327beb.jpg)

<a id="paper-2544"></a>

## Search Space Synthesis for Parametric Functions


### Images

![02df99920cbafbf78ad8916f88803a68f0d9d9fcf49cd5c08d2bd5092f11c044.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/02df99920cbafbf78ad8916f88803a68f0d9d9fcf49cd5c08d2bd5092f11c044.jpg)

![0b4218d1f68265da72c55383e2bc95ae0357c96e167342d56ae1dd00a481ca77.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/0b4218d1f68265da72c55383e2bc95ae0357c96e167342d56ae1dd00a481ca77.jpg)

![0c34e7b61a37731bae4eeab9808efc87e7a7af915d6a3c05d38b35dba970c9bf.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/0c34e7b61a37731bae4eeab9808efc87e7a7af915d6a3c05d38b35dba970c9bf.jpg)

![1f50cc4341c9592491b5347b6c133ccaa0ef394d5dc05cd9d6e5d4a96983f0a6.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/1f50cc4341c9592491b5347b6c133ccaa0ef394d5dc05cd9d6e5d4a96983f0a6.jpg)

![2a3c04df9b174f70608c3d895b53ff8d71c8303a58113590d135491f32573747.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/2a3c04df9b174f70608c3d895b53ff8d71c8303a58113590d135491f32573747.jpg)

![2f3ebd4a26bdfe407e7f1d12c9f4239032456308f6ca95f6725694e70e183531.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/2f3ebd4a26bdfe407e7f1d12c9f4239032456308f6ca95f6725694e70e183531.jpg)

![5ca022f0c1d3ed5c7f8a5109649d4d3914732257b7be9d47b376a61ade0d7332.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/5ca022f0c1d3ed5c7f8a5109649d4d3914732257b7be9d47b376a61ade0d7332.jpg)

![61a3aa9b750bad8507267ca14c3a29b04aa7752632053b1f2c413c3ef1984821.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/61a3aa9b750bad8507267ca14c3a29b04aa7752632053b1f2c413c3ef1984821.jpg)

![816349fb46736864aff63d174ac412acc4eeb583a28ea854f4b413b3ec6b1248.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/816349fb46736864aff63d174ac412acc4eeb583a28ea854f4b413b3ec6b1248.jpg)

![87a1f611ac38371207dadabee66af1c6992562cdb83c68363fed27859bcc9811.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/87a1f611ac38371207dadabee66af1c6992562cdb83c68363fed27859bcc9811.jpg)

![9ee8397641a9e8dd71c0d1829f466f6eb06fe2fa5ed3f6a04d4882da0fc1f975.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/9ee8397641a9e8dd71c0d1829f466f6eb06fe2fa5ed3f6a04d4882da0fc1f975.jpg)

![a66addf43b5ad04823ee3e805384d75e3253327b21bf03929292863c8e6832eb.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/a66addf43b5ad04823ee3e805384d75e3253327b21bf03929292863c8e6832eb.jpg)

![fbbaf56d679839e78eaf528366d10a22ac94385f2afe8579fc49dd1e3f71e2db.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/fbbaf56d679839e78eaf528366d10a22ac94385f2afe8579fc49dd1e3f71e2db.jpg)

![fea4b23413e11f853b753a3be632718781910db275c4d3b1855f49ecb1e2afb8.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/images/fea4b23413e11f853b753a3be632718781910db275c4d3b1855f49ecb1e2afb8.jpg)

### Tables

![8933c4247318f25533884c5bac27f30ecf0b2d5be1dbc4220f18230a862f838f.jpg](../icml_results/2576_Search%20Space%20Synthesis%20for%20Parametric%20Functions/tables/8933c4247318f25533884c5bac27f30ecf0b2d5be1dbc4220f18230a862f838f.jpg)

<a id="paper-2545"></a>

## Refined Analysis of Entropy-Regularized Actor-Critic


### Images

![2d349e9b8bf2148f42d8a7d285eeffecdc8218960a63361533f1a0dbf2353c30.jpg](../icml_results/2577_Refined%20Analysis%20of%20Entropy-Regularized%20Actor-Critic/images/2d349e9b8bf2148f42d8a7d285eeffecdc8218960a63361533f1a0dbf2353c30.jpg)

### Tables

![40aeadbc015684178af881fbe55b51cd50ca338c3295f599a137dec5ed8afca2.jpg](../icml_results/2577_Refined%20Analysis%20of%20Entropy-Regularized%20Actor-Critic/tables/40aeadbc015684178af881fbe55b51cd50ca338c3295f599a137dec5ed8afca2.jpg)

<a id="paper-2546"></a>

## Quantifying the noise sensitivity of the Wasserstein metric for images


### Images

![00bf36f2ef3e17dd7eae464e09dad52b2290738823257c2f3414fb59fe9199d5.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/00bf36f2ef3e17dd7eae464e09dad52b2290738823257c2f3414fb59fe9199d5.jpg)

![08c25d5cb9da4855e98e1701b42e68fcf4384624f2ebbc454ebe92eb12ae65bd.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/08c25d5cb9da4855e98e1701b42e68fcf4384624f2ebbc454ebe92eb12ae65bd.jpg)

![48a0fd75899f8aee4ace3f5e75a01a8d3a7bf1b3226068d7b04ff0e30ed771c6.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/48a0fd75899f8aee4ace3f5e75a01a8d3a7bf1b3226068d7b04ff0e30ed771c6.jpg)

![9bb206f8f24744b662115165719a2a2b9921eea6ed7056b95348145cbcd0d25e.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/9bb206f8f24744b662115165719a2a2b9921eea6ed7056b95348145cbcd0d25e.jpg)

![a4528f1fdfce41a98dac6ea50039654adc62711a457aed28800aa48fe08b4b54.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/a4528f1fdfce41a98dac6ea50039654adc62711a457aed28800aa48fe08b4b54.jpg)

![b41b57ba91fc67d2ef0a0c252a755cdac6ac668cd5d00a8b3333cf7c6df7f4ae.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/b41b57ba91fc67d2ef0a0c252a755cdac6ac668cd5d00a8b3333cf7c6df7f4ae.jpg)

![e27a9e163eb3883962136bc2cd37bc1f989bd95cb2b50ee2a1a5bf1e945e2599.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/e27a9e163eb3883962136bc2cd37bc1f989bd95cb2b50ee2a1a5bf1e945e2599.jpg)

![eea2dfcae53f67ac15705e6273b71fdef8f1ba36556866aeb6df18ce65b4dc88.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/eea2dfcae53f67ac15705e6273b71fdef8f1ba36556866aeb6df18ce65b4dc88.jpg)

![fe80290fde4c18222b9fc55e6c2e98ff2c34611da63897496b7b7dceeee3e6d2.jpg](../icml_results/2578_Quantifying%20the%20noise%20sensitivity%20of%20the%20Wasserstein%20metric%20for%20images/images/fe80290fde4c18222b9fc55e6c2e98ff2c34611da63897496b7b7dceeee3e6d2.jpg)

<a id="paper-2547"></a>

## Graph Alignment for Benchmarking Graph Neural Networks and Learning Positional Encodings


### Images

![1d03dd434fea6b0096ec2b9fed678789da5450a85b21b19209d88b2deaae4b80.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/1d03dd434fea6b0096ec2b9fed678789da5450a85b21b19209d88b2deaae4b80.jpg)

![50451125fa299bf84b52d567d09c10ade60c5c1aaac224baf7fa29dcb963d349.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/50451125fa299bf84b52d567d09c10ade60c5c1aaac224baf7fa29dcb963d349.jpg)

![a540f73cb0d117746ec99ce19550bf44d7e5df7746198ca643fc260fba7ad78a.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/a540f73cb0d117746ec99ce19550bf44d7e5df7746198ca643fc260fba7ad78a.jpg)

![bda95791aedc2a31701320fd4231a1df1540e4726d831eb08ff2210a6c723105.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/bda95791aedc2a31701320fd4231a1df1540e4726d831eb08ff2210a6c723105.jpg)

![cfc90ca7f911589667c60a3be22915b8ac0aba3bbd5fd9c3e7d57d1c87c64650.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/cfc90ca7f911589667c60a3be22915b8ac0aba3bbd5fd9c3e7d57d1c87c64650.jpg)

![d6223dc397d661c5d80d12aaa3a5d60499459b767479091babda8ded4bda3e20.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/d6223dc397d661c5d80d12aaa3a5d60499459b767479091babda8ded4bda3e20.jpg)

![d741bede38df4966ab7b240879109e914d7138cfe53df301404915f7b4f117f5.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/d741bede38df4966ab7b240879109e914d7138cfe53df301404915f7b4f117f5.jpg)

![da3af46d9ce3289fcc64825f737819ad9a153b8d206c1a16dd2e78f8a5b441b1.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/images/da3af46d9ce3289fcc64825f737819ad9a153b8d206c1a16dd2e78f8a5b441b1.jpg)

### Tables

![0688f8dc9ad4ab0007501c4dd7121f6b44fadc9384be3c1936685b181e0bf60e.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/0688f8dc9ad4ab0007501c4dd7121f6b44fadc9384be3c1936685b181e0bf60e.jpg)

![1d423782326f8b8db149c5be7497205534813c50c7d592868c973051663db72c.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/1d423782326f8b8db149c5be7497205534813c50c7d592868c973051663db72c.jpg)

![1ee83d904b81f724b8f155127fb78e3c76a16174fdcd185ce6e59eff08ee7f41.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/1ee83d904b81f724b8f155127fb78e3c76a16174fdcd185ce6e59eff08ee7f41.jpg)

![2154d27d8b21109b77f58bfd65e6e27f85723983b14d5db88aa90c38e09aae20.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/2154d27d8b21109b77f58bfd65e6e27f85723983b14d5db88aa90c38e09aae20.jpg)

![32d7174b12ae5e7b5dc9eb738e93f3577615fc90cd40febfba300bb79db33e53.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/32d7174b12ae5e7b5dc9eb738e93f3577615fc90cd40febfba300bb79db33e53.jpg)

![4b160529fef189b42c96eda7a032d5bf8dfb4c733f3e80bcd8eb961efcab5cf1.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/4b160529fef189b42c96eda7a032d5bf8dfb4c733f3e80bcd8eb961efcab5cf1.jpg)

![5a94cad36b37195a908c7b6a1e319a599c87193c11cd31109e3043d3a332a99d.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/5a94cad36b37195a908c7b6a1e319a599c87193c11cd31109e3043d3a332a99d.jpg)

![65cf88e8a6b80d8b4cebcf2f7370140f5e29a007c999c7efc67d55b3b28394ac.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/65cf88e8a6b80d8b4cebcf2f7370140f5e29a007c999c7efc67d55b3b28394ac.jpg)

![73ba8f2cced10402aaec3a5a7377a1d9e11e7495c5ddf7813cb09cb7f0aec77e.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/73ba8f2cced10402aaec3a5a7377a1d9e11e7495c5ddf7813cb09cb7f0aec77e.jpg)

![7eff479a1543f4780438b94a0d4d0067ed89742aa89ee6e949d31b2cfb0ad894.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/7eff479a1543f4780438b94a0d4d0067ed89742aa89ee6e949d31b2cfb0ad894.jpg)

![80e1155ad6bfa728e52a3fe79b01c17adbb2cb169695d5dbf96e1b399d5e26e0.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/80e1155ad6bfa728e52a3fe79b01c17adbb2cb169695d5dbf96e1b399d5e26e0.jpg)

![942c5fe5a2c7921a76d01f8697b6663bacb3572733121fc698b341bf72111ab7.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/942c5fe5a2c7921a76d01f8697b6663bacb3572733121fc698b341bf72111ab7.jpg)

![9c7bc3bd2f966169fd55e092bd748c76836e3456c2259d32f5f9ffa5649c64f4.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/9c7bc3bd2f966169fd55e092bd748c76836e3456c2259d32f5f9ffa5649c64f4.jpg)

![bb1448f699f0cc06ecd41a1b66d4581e0f1e8c02521d065ebd00679a3d24d226.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/bb1448f699f0cc06ecd41a1b66d4581e0f1e8c02521d065ebd00679a3d24d226.jpg)

![c0c9b547f8e9e1f06c4459863411f9f46a8c1732b115965f9cb21fa175131790.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/c0c9b547f8e9e1f06c4459863411f9f46a8c1732b115965f9cb21fa175131790.jpg)

![d079a28c3228297cb76bca04d08699ba77b851b551615d879b14517e572da7bb.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/d079a28c3228297cb76bca04d08699ba77b851b551615d879b14517e572da7bb.jpg)

![f15bd968e5fb2893d3d6c0f6ac11318fbe8b8ae897cb05be6f45889f83b4e85c.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/f15bd968e5fb2893d3d6c0f6ac11318fbe8b8ae897cb05be6f45889f83b4e85c.jpg)

![f9d47b14557eb760fe8dc756eb7d272caeb932d33f3670b7d1633b55f81a80a4.jpg](../icml_results/2579_Graph%20Alignment%20for%20Benchmarking%20Graph%20Neural%20Networks%20and%20Learning%20Positional%20Encodings/tables/f9d47b14557eb760fe8dc756eb7d272caeb932d33f3670b7d1633b55f81a80a4.jpg)

<a id="paper-2548"></a>

## UniRRM: Unified Reasoning Reward Models Across Languages and Evaluation Paradigms


### Images

![22ed081cb3b4b38119e1bc3bd575008ad1af38f9139ba69a44e2087151fac63a.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/22ed081cb3b4b38119e1bc3bd575008ad1af38f9139ba69a44e2087151fac63a.jpg)

![236bbda4f71ebadf84c34fe7266dacdd9840cf02e5a7fac0645ffeb3a5c72636.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/236bbda4f71ebadf84c34fe7266dacdd9840cf02e5a7fac0645ffeb3a5c72636.jpg)

![2d20dfee82c0f6ee137a2e09cd2a1d3e7bf808e1e0a2d8636f17923c537be3db.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/2d20dfee82c0f6ee137a2e09cd2a1d3e7bf808e1e0a2d8636f17923c537be3db.jpg)

![56f556525c43128ba03e5b6172c0d5dfebdefc0c7e8af73dbf460e93a7fc5b40.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/56f556525c43128ba03e5b6172c0d5dfebdefc0c7e8af73dbf460e93a7fc5b40.jpg)

![9135e9b29eec8e370d37aac22ac1829b3ce52ed6a97dae35be73297978d93e60.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/9135e9b29eec8e370d37aac22ac1829b3ce52ed6a97dae35be73297978d93e60.jpg)

![95bc9016aefaadd01ade4ead17b9d7fa8a859476e92d1d8c71499b24681b3ba3.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/95bc9016aefaadd01ade4ead17b9d7fa8a859476e92d1d8c71499b24681b3ba3.jpg)

![ac69b9d41ca9ed4f709c552d7f078ad37ce20cb16240683430f85f70ea021c10.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/ac69b9d41ca9ed4f709c552d7f078ad37ce20cb16240683430f85f70ea021c10.jpg)

![b1f653ace6fd801d2719d9497dce3c86bbe2cc3c978834a5a3aa1b1991284b66.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/images/b1f653ace6fd801d2719d9497dce3c86bbe2cc3c978834a5a3aa1b1991284b66.jpg)

### Tables

![18653ccc7f3a35e4558f19d183b2b790afcff45d2e064fcc9bf7c103427a2485.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/18653ccc7f3a35e4558f19d183b2b790afcff45d2e064fcc9bf7c103427a2485.jpg)

![1997e047d4f439a138683daf981759ff0d29c514faeb6a2abc7383a227fc4613.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/1997e047d4f439a138683daf981759ff0d29c514faeb6a2abc7383a227fc4613.jpg)

![30260526ce807811a7500d305380c548661d00070ec52df5c237886e562a9fa5.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/30260526ce807811a7500d305380c548661d00070ec52df5c237886e562a9fa5.jpg)

![322e166c1c8898c6cac825c62eb08493961a334b3fbc8adf2951815dce7ff604.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/322e166c1c8898c6cac825c62eb08493961a334b3fbc8adf2951815dce7ff604.jpg)

![32e3f9fc34bcee5388aea7b0f499efd8a7edc5e8ed1f92a725d99653e00ba4fa.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/32e3f9fc34bcee5388aea7b0f499efd8a7edc5e8ed1f92a725d99653e00ba4fa.jpg)

![48c9bc86043dc4b2f5ecc148a412c661f48c53389791d5fc3037e3ab968eb4a7.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/48c9bc86043dc4b2f5ecc148a412c661f48c53389791d5fc3037e3ab968eb4a7.jpg)

![56703758da20acb988b21e3880f0ae9d40cabd362807099901c1ebb528d4b53b.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/56703758da20acb988b21e3880f0ae9d40cabd362807099901c1ebb528d4b53b.jpg)

![6388a1d0995c92744a6b7bc323e44c5490dfa4d339bb4a42df4869d959e8ebdb.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/6388a1d0995c92744a6b7bc323e44c5490dfa4d339bb4a42df4869d959e8ebdb.jpg)

![9853f188fd2043f8f7795746c665587d012d907a3b8949cd5f6c6cf5b2a25aff.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/9853f188fd2043f8f7795746c665587d012d907a3b8949cd5f6c6cf5b2a25aff.jpg)

![cbca464a1ac78320e4dd4f946539b46998395e32697c9958fd4e9fd23c759b41.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/cbca464a1ac78320e4dd4f946539b46998395e32697c9958fd4e9fd23c759b41.jpg)

![e10a83e7b9e2c7b9d56550aa57838314a314d71a94cd395a5f66771aa3a00147.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/e10a83e7b9e2c7b9d56550aa57838314a314d71a94cd395a5f66771aa3a00147.jpg)

![f60ee4a32eb24eee44f190fc608e64ac2036acf25ca2ae60959ba129c8931fbf.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/f60ee4a32eb24eee44f190fc608e64ac2036acf25ca2ae60959ba129c8931fbf.jpg)

![f8583daffb0822a838aacc310ef18d2566d2dedacc31b8a08265c148d8a7f334.jpg](../icml_results/2580_UniRRM_%20Unified%20Reasoning%20Reward%20Models%20Across%20Languages%20and%20Evaluation%20Paradigms/tables/f8583daffb0822a838aacc310ef18d2566d2dedacc31b8a08265c148d8a7f334.jpg)

<a id="paper-2549"></a>

## DecoderTCR: Compositional Pretraining and Entropy-Guided Decoding for TCR-pMHC Interactions


### Images

![03cbec8755c0b7e97ed4238d6da4fc619a15af8670c1bc7596444e8d96da23bb.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/03cbec8755c0b7e97ed4238d6da4fc619a15af8670c1bc7596444e8d96da23bb.jpg)

![06a43cc073b25421c1961c3473b448f6220a43a5583e7bc533091159bf9cd86c.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/06a43cc073b25421c1961c3473b448f6220a43a5583e7bc533091159bf9cd86c.jpg)

![19e10c7cbc39748c883337b34126a151a796297426f7cfc78ceef98829e842df.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/19e10c7cbc39748c883337b34126a151a796297426f7cfc78ceef98829e842df.jpg)

![33b061d839eb495b1faa06bc237b79b67b62df5f16d8c1289aa75a5a209e4876.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/33b061d839eb495b1faa06bc237b79b67b62df5f16d8c1289aa75a5a209e4876.jpg)

![410fb96dfc5525dc27131aa848721c5b892c62746151436d860aaee8e5994a69.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/410fb96dfc5525dc27131aa848721c5b892c62746151436d860aaee8e5994a69.jpg)

![83ba98f79cfa3be089c8da0c583ea08f14be91769b711014607e27e1d07d7e41.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/83ba98f79cfa3be089c8da0c583ea08f14be91769b711014607e27e1d07d7e41.jpg)

![a3e50b233d1b37907cb7e144257b726463d81e2141402dec7b095487f359ff20.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/a3e50b233d1b37907cb7e144257b726463d81e2141402dec7b095487f359ff20.jpg)

![e74f2ac60e0844e3f63e0387e0dc0a883df5eaf70d2061345ea7ee5c271c8a48.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/images/e74f2ac60e0844e3f63e0387e0dc0a883df5eaf70d2061345ea7ee5c271c8a48.jpg)

### Tables

![03fea9c918913288f5252152b8ee874e4c2c492c386d59b26d7819e49e642bba.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/03fea9c918913288f5252152b8ee874e4c2c492c386d59b26d7819e49e642bba.jpg)

![1786eb756ebdfdf3b2bdbec469bfd9c4322ebb313a3fe15a5b4aaaa53114e20d.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/1786eb756ebdfdf3b2bdbec469bfd9c4322ebb313a3fe15a5b4aaaa53114e20d.jpg)

![2420b4c6126ca1fd58d04a15fe569f01d1c384c5c51788ca45181362d1344718.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/2420b4c6126ca1fd58d04a15fe569f01d1c384c5c51788ca45181362d1344718.jpg)

![2baecf4a6456df7401b6496a70404bc8ea664267d2ccad9818313c023a5549bb.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/2baecf4a6456df7401b6496a70404bc8ea664267d2ccad9818313c023a5549bb.jpg)

![3d3a054df2b54765cdaa299b60fe238247e5527f133860a1380c3179d2872818.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/3d3a054df2b54765cdaa299b60fe238247e5527f133860a1380c3179d2872818.jpg)

![5fe15cb4170634d4d090e6d6d5bdf321c1635ee87cd8beb4cbf023387b820bf7.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/5fe15cb4170634d4d090e6d6d5bdf321c1635ee87cd8beb4cbf023387b820bf7.jpg)

![7d75c6229ebe9becd37d53ae7a35d7953d9612e95ade01349c6a22c4d0a3536c.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/7d75c6229ebe9becd37d53ae7a35d7953d9612e95ade01349c6a22c4d0a3536c.jpg)

![8dd7181c3aaf5e80dd9a2b9802f7da3d5060effc5552de92edca7c018cfdcd1e.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/8dd7181c3aaf5e80dd9a2b9802f7da3d5060effc5552de92edca7c018cfdcd1e.jpg)

![91ce18c233953171db73df81cee6537f6ff35e14134313dea008d14329bfb890.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/91ce18c233953171db73df81cee6537f6ff35e14134313dea008d14329bfb890.jpg)

![9f1920ecc612ff96044f8d1d4819af8f3df8280923a38631d451b676bcfe2c96.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/9f1920ecc612ff96044f8d1d4819af8f3df8280923a38631d451b676bcfe2c96.jpg)

![9f7e25a8abeab2168bbe380052e119cba709656325fe9fed3125d49063376aba.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/9f7e25a8abeab2168bbe380052e119cba709656325fe9fed3125d49063376aba.jpg)

![a6ad5c4e3c0496c9a699986c976801ecb4e4cf1114bbfb65ab102fd77129db7b.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/a6ad5c4e3c0496c9a699986c976801ecb4e4cf1114bbfb65ab102fd77129db7b.jpg)

![b30a2093391c8fc39da90193ef0810edee335344753c39956d8deaf475b730b1.jpg](../icml_results/2581_DecoderTCR_%20Compositional%20Pretraining%20and%20Entropy-Guided%20Decoding%20for%20TCR-pMHC%20Interactions/tables/b30a2093391c8fc39da90193ef0810edee335344753c39956d8deaf475b730b1.jpg)

<a id="paper-2550"></a>

## Predictive variational inference: Learn the predictively optimal posterior distribution


### Images

![179cf26a98d3b551f00dce7ffaf4c1ceea53685ab2acc35e6e02f94167057f5b.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/179cf26a98d3b551f00dce7ffaf4c1ceea53685ab2acc35e6e02f94167057f5b.jpg)

![4806f18b48f807427cec19e41bfa4007c3acadf61be90fa509163697db05dc59.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/4806f18b48f807427cec19e41bfa4007c3acadf61be90fa509163697db05dc59.jpg)

![4a5ce048628c7d64ec3fe0536d28597aeef0d1cc9f26658c19149d7ffc334764.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/4a5ce048628c7d64ec3fe0536d28597aeef0d1cc9f26658c19149d7ffc334764.jpg)

![5593bec8bce027deee02fc8cc9f5c95a2464daa3a60c167ff6a6cd716e5fbd46.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/5593bec8bce027deee02fc8cc9f5c95a2464daa3a60c167ff6a6cd716e5fbd46.jpg)

![56c6453aedbd8d99a4b303aba0bc0b11f807d63b2d3c26dd6f988a9200bf571d.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/56c6453aedbd8d99a4b303aba0bc0b11f807d63b2d3c26dd6f988a9200bf571d.jpg)

![6ff982f29e47908b37ceb4efb42ef7300454989df97f51c308fdac1b5b35a9b2.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/6ff982f29e47908b37ceb4efb42ef7300454989df97f51c308fdac1b5b35a9b2.jpg)

![82d2f0dd38dac97040c2d725b6c043c7e58ac2a5f885498b1ae7b7860dcb7770.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/82d2f0dd38dac97040c2d725b6c043c7e58ac2a5f885498b1ae7b7860dcb7770.jpg)

![94e7b550f08c6228fdb6a63d8e795b0c4ba36b2f42d54d533d185743cd0a4c7a.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/94e7b550f08c6228fdb6a63d8e795b0c4ba36b2f42d54d533d185743cd0a4c7a.jpg)

![9749a58d5542aac2d37c729b8c8265c0802c7f53e0d5c1cbf6ef25d65f5c4360.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/9749a58d5542aac2d37c729b8c8265c0802c7f53e0d5c1cbf6ef25d65f5c4360.jpg)

![b80434358366d6f2f6e215d6ebe761468c044ae05bf500e6bd5ebc588abb1b99.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/b80434358366d6f2f6e215d6ebe761468c044ae05bf500e6bd5ebc588abb1b99.jpg)

![d6532c5c00ca2abd78280b802ce0ec4c6b1646d1972c30f1ccaf2a71ffc4ee7e.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/d6532c5c00ca2abd78280b802ce0ec4c6b1646d1972c30f1ccaf2a71ffc4ee7e.jpg)

![e167bf20244841db6eea0f3844a4fa2c285883b230289ca476c5566153de7476.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/e167bf20244841db6eea0f3844a4fa2c285883b230289ca476c5566153de7476.jpg)

![fe24dd4e38788158d17203404d6e7c86217afa7d0bf21a3afd9c15bb7ede786d.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/images/fe24dd4e38788158d17203404d6e7c86217afa7d0bf21a3afd9c15bb7ede786d.jpg)

### Tables

![0b004737cfd0d9343e46ac8023ebeb1c03e6e60db757369c18c73ed49a751ae1.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/0b004737cfd0d9343e46ac8023ebeb1c03e6e60db757369c18c73ed49a751ae1.jpg)

![0f0a390874cf3d6f5f3ffba3e59a78c22435e0f3c079678364b9fa4e06a6f0fc.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/0f0a390874cf3d6f5f3ffba3e59a78c22435e0f3c079678364b9fa4e06a6f0fc.jpg)

![632f37e677638d577d209e9d92517eb62114e026903b585a9ad7e3a80044a5e0.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/632f37e677638d577d209e9d92517eb62114e026903b585a9ad7e3a80044a5e0.jpg)

![6a8417829c3681af4a2a9b74fcc9a539457412bc83fef4326f70b7ae46a961a0.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/6a8417829c3681af4a2a9b74fcc9a539457412bc83fef4326f70b7ae46a961a0.jpg)

![8ca67744adc879ac920b808d1077b2d37910a808b7ceb27564b5dbe54b391c82.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/8ca67744adc879ac920b808d1077b2d37910a808b7ceb27564b5dbe54b391c82.jpg)

![9a3dfd798f9ad182e5783e2f29eff9e08f69cabd5442375fffcb0e285e8d5177.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/9a3dfd798f9ad182e5783e2f29eff9e08f69cabd5442375fffcb0e285e8d5177.jpg)

![a883bc1284532526f2452253c6c60150fde80d9bf35361830b6862a3eb745da5.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/a883bc1284532526f2452253c6c60150fde80d9bf35361830b6862a3eb745da5.jpg)

![f9aaacbc768dbf2bb71038983cea6db936fdfd358e22966f90813db21649cfa7.jpg](../icml_results/2582_Predictive%20variational%20inference_%20Learn%20the%20predictively%20optimal%20posterior%20distribution/tables/f9aaacbc768dbf2bb71038983cea6db936fdfd358e22966f90813db21649cfa7.jpg)

<a id="paper-2551"></a>

## MoCL: Metabolic Optimization for Curvature-Aware Continual Learning


### Images

![1636e0ca72d93cfd86b43383a393ef68e2af599590465cfce5772841860eb3bd.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/1636e0ca72d93cfd86b43383a393ef68e2af599590465cfce5772841860eb3bd.jpg)

![3d1298331b84cf113830f2095f5d29b99f61e723b19bd1d87a680b33618dba27.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/3d1298331b84cf113830f2095f5d29b99f61e723b19bd1d87a680b33618dba27.jpg)

![6d4cfb93dd8ae3789aede00cee030cef7b762cf3179db8838b51c90d9208e1a5.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/6d4cfb93dd8ae3789aede00cee030cef7b762cf3179db8838b51c90d9208e1a5.jpg)

![8c7a732c5fcae2ec1a569d2d6fbc501627b5ee97750b8c4f9250a8d2e54a88f2.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/8c7a732c5fcae2ec1a569d2d6fbc501627b5ee97750b8c4f9250a8d2e54a88f2.jpg)

![9d38f221878a8016dd1270935aad7c806b8cbfb34b1ccdaeb06889a393311c0a.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/9d38f221878a8016dd1270935aad7c806b8cbfb34b1ccdaeb06889a393311c0a.jpg)

![f18405675b2550b63a06f75da028f640b34f0c5870db92b4ef09aae701940882.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/images/f18405675b2550b63a06f75da028f640b34f0c5870db92b4ef09aae701940882.jpg)

### Tables

![0793569359f702f14c70109d2d6c5c3f19ea60757abd025509a0956d960a16c6.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/0793569359f702f14c70109d2d6c5c3f19ea60757abd025509a0956d960a16c6.jpg)

![2357196d51164784111e84087d4865f00dee676188dd19174b4e9901ef0aba1a.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/2357196d51164784111e84087d4865f00dee676188dd19174b4e9901ef0aba1a.jpg)

![26db4565d64ecba4906bc9706a770eec1061312529b015a18cafe91626404dcd.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/26db4565d64ecba4906bc9706a770eec1061312529b015a18cafe91626404dcd.jpg)

![2ad9939d4d9076bbaa494601b09e9d23edb4ddda6ecdaaf70ceb327f83e83bbe.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/2ad9939d4d9076bbaa494601b09e9d23edb4ddda6ecdaaf70ceb327f83e83bbe.jpg)

![2e11b27fb91da7656ca8e7635dc8025781263d956ccd4d059ad10f891f6cf00f.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/2e11b27fb91da7656ca8e7635dc8025781263d956ccd4d059ad10f891f6cf00f.jpg)

![4ad0437a81bbd21ef3fcb669960b4e3c07097bb273d3e6ac8f740c544e13605a.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/4ad0437a81bbd21ef3fcb669960b4e3c07097bb273d3e6ac8f740c544e13605a.jpg)

![70e68aa40156eb69627d1dbe31d734747ded7828769ee27d7990759c76b5aa46.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/70e68aa40156eb69627d1dbe31d734747ded7828769ee27d7990759c76b5aa46.jpg)

![73484b0df96f41d1e522906926155a91a39bc73a4d262d8205cdf5765d58a651.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/73484b0df96f41d1e522906926155a91a39bc73a4d262d8205cdf5765d58a651.jpg)

![f0fd19b5960cf7e5d99f0dad8aecd373d569213282a692149527a96ff1e2f18c.jpg](../icml_results/2583_MoCL_%20Metabolic%20Optimization%20for%20Curvature-Aware%20Continual%20Learning/tables/f0fd19b5960cf7e5d99f0dad8aecd373d569213282a692149527a96ff1e2f18c.jpg)

<a id="paper-2552"></a>

## Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models


### Images

![005bc1e83d5ebb6a6dfe39c7a2969759fed976c6d344b34259ca48d51033a0ca.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/005bc1e83d5ebb6a6dfe39c7a2969759fed976c6d344b34259ca48d51033a0ca.jpg)

![03aba61fbf216cb1c182e279e502e1cbe72de6c00427c2bba2f3132967bb0a88.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/03aba61fbf216cb1c182e279e502e1cbe72de6c00427c2bba2f3132967bb0a88.jpg)

![0836db1faa57c9cebfcf2da19c8e867ca2c99d3ed99da3a3cc58a17f18427dc3.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/0836db1faa57c9cebfcf2da19c8e867ca2c99d3ed99da3a3cc58a17f18427dc3.jpg)

![1133c22b61474ae84cd4f3a4a880473a89c4bb7a636e6040c4189a58873c5c7f.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/1133c22b61474ae84cd4f3a4a880473a89c4bb7a636e6040c4189a58873c5c7f.jpg)

![19643a3c23d267fa459292c83fb7593de58b0b3cd44cb74c46ed682e651b38db.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/19643a3c23d267fa459292c83fb7593de58b0b3cd44cb74c46ed682e651b38db.jpg)

![20908b34ea8e94888911ef90b914b3bdbad812397a71ff68bf29acf7b376cc4b.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/20908b34ea8e94888911ef90b914b3bdbad812397a71ff68bf29acf7b376cc4b.jpg)

![22cae13d3a16f69ff79888d63f71bf79fbdb00d2fe3813bbd6b724427f93bc7e.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/22cae13d3a16f69ff79888d63f71bf79fbdb00d2fe3813bbd6b724427f93bc7e.jpg)

![2ecc52776e4ea4806d6902d7cb7321ab9a536c0518ab40e7e804cd64eea6efd8.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/2ecc52776e4ea4806d6902d7cb7321ab9a536c0518ab40e7e804cd64eea6efd8.jpg)

![32616e6801f51ebce994a3a22b73781888562682294ae55ac5bbc2943b250094.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/32616e6801f51ebce994a3a22b73781888562682294ae55ac5bbc2943b250094.jpg)

![48a93fb564f8e9427ccae750d680a8a99606600a048323457b681ceb3fd5b672.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/48a93fb564f8e9427ccae750d680a8a99606600a048323457b681ceb3fd5b672.jpg)

![70c602b6b761528a8b14a7c04f9f6e17997df76620189f16645b85538b136247.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/70c602b6b761528a8b14a7c04f9f6e17997df76620189f16645b85538b136247.jpg)

![7b19ef30133906ad4ebd8942f59b2fb5af3fbcc043a6000f6f012c130cdd9521.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/7b19ef30133906ad4ebd8942f59b2fb5af3fbcc043a6000f6f012c130cdd9521.jpg)

![92cbd3594c3290c46374acd5e6a29293e4357a8ca2da5ee559bc846a8ad349d3.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/92cbd3594c3290c46374acd5e6a29293e4357a8ca2da5ee559bc846a8ad349d3.jpg)

![ba1eef86a3d74258e1e56ce22cc74635aa250bd80f125c48a0fdc7810e574275.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/ba1eef86a3d74258e1e56ce22cc74635aa250bd80f125c48a0fdc7810e574275.jpg)

![ce9c8b8f89fd7473c72463ba050d33faff47a167edc9d1c82fb03d7c6ca51288.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/ce9c8b8f89fd7473c72463ba050d33faff47a167edc9d1c82fb03d7c6ca51288.jpg)

![d82950d26ca471f59ba6cfaa4b9b01d3385bfe144b6e4bf93f2722e3780dece4.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/d82950d26ca471f59ba6cfaa4b9b01d3385bfe144b6e4bf93f2722e3780dece4.jpg)

![d9a462a176992d4544ea946e1ab808e2632b1373dd9377f7d1e1c27b4d5dc9fc.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/d9a462a176992d4544ea946e1ab808e2632b1373dd9377f7d1e1c27b4d5dc9fc.jpg)

![e2d812692f708465b7886624cf7e0eb78bc6138a1dab808ee1fa107b29e37f26.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/e2d812692f708465b7886624cf7e0eb78bc6138a1dab808ee1fa107b29e37f26.jpg)

![e559b937f30881b15f2d4c02221a798388d46a1b4b242190c1c930d127b4ae0a.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/e559b937f30881b15f2d4c02221a798388d46a1b4b242190c1c930d127b4ae0a.jpg)

![e82ec98fab9370fc6f6ba1d0d6ffb4a86f2031b8d2e558e524f51beaea275741.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/e82ec98fab9370fc6f6ba1d0d6ffb4a86f2031b8d2e558e524f51beaea275741.jpg)

![ef646ec83938f1a8570d14b184ca93c431803cfcf211da272ce4a35302947015.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/ef646ec83938f1a8570d14b184ca93c431803cfcf211da272ce4a35302947015.jpg)

![efdf89f7067da729d0d41f71e54bee415c013650c63fba9a1f826e1510574348.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/images/efdf89f7067da729d0d41f71e54bee415c013650c63fba9a1f826e1510574348.jpg)

### Tables

![20e110bc9c82b245fbc06eec074af2af5fcfd644aeccbe63c5403d9b9a30d291.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/20e110bc9c82b245fbc06eec074af2af5fcfd644aeccbe63c5403d9b9a30d291.jpg)

![3220e4854002c1775dcf9ec4faa309313d4ce5c33250d7a113bcb90ceccd9a83.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/3220e4854002c1775dcf9ec4faa309313d4ce5c33250d7a113bcb90ceccd9a83.jpg)

![4759bec8864869d2cdf9e568855588de0c7123aa73d109a7632a6d98cbb33e8d.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/4759bec8864869d2cdf9e568855588de0c7123aa73d109a7632a6d98cbb33e8d.jpg)

![54ba772d6ec8d61265cf2c771604ec3a21c88d741ac85217109409b5a388ff7b.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/54ba772d6ec8d61265cf2c771604ec3a21c88d741ac85217109409b5a388ff7b.jpg)

![6392130e57866226dc033977a1dad6dc8047347500bb135357f94b5c53475f85.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/6392130e57866226dc033977a1dad6dc8047347500bb135357f94b5c53475f85.jpg)

![6394d569801b178c1d7e0b9dfc7692d08a3dfb1732fa42d2e71eeef421cf8d04.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/6394d569801b178c1d7e0b9dfc7692d08a3dfb1732fa42d2e71eeef421cf8d04.jpg)

![65b377c385cb0a2124480dc81c492640ba60150aa5fc9e5c6faaf7f13465e403.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/65b377c385cb0a2124480dc81c492640ba60150aa5fc9e5c6faaf7f13465e403.jpg)

![6ac3cbd158859494a5a5a5de08290743eee054d3aa8da76173123055b8927511.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/6ac3cbd158859494a5a5a5de08290743eee054d3aa8da76173123055b8927511.jpg)

![7ca5dc61be2248b666fed5262382c22844d571df3287e9d00c8ef6a7af691c8e.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/7ca5dc61be2248b666fed5262382c22844d571df3287e9d00c8ef6a7af691c8e.jpg)

![94dfb8dc6e5f7035ba27b5d5257a53f178c0e1b376c58ac7d07750fab39ea532.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/94dfb8dc6e5f7035ba27b5d5257a53f178c0e1b376c58ac7d07750fab39ea532.jpg)

![9abb62c8884a3e6d7d99fb22acecc629046c271bbfaea998ea01f7db6d25d27e.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/9abb62c8884a3e6d7d99fb22acecc629046c271bbfaea998ea01f7db6d25d27e.jpg)

![bb073b7418ec62739e4a27658eba1d3bd543a2f24f2c0df05b6ab9820ec06de3.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/bb073b7418ec62739e4a27658eba1d3bd543a2f24f2c0df05b6ab9820ec06de3.jpg)

![e2242d017272cf90ed740fd9474355d6c12775db8e9e76c6b01bee794a9600d1.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/e2242d017272cf90ed740fd9474355d6c12775db8e9e76c6b01bee794a9600d1.jpg)

![e65a5b1f1bd80e8ba5b30ee54947c5ee340bc21736b31b2b8cdff5909e0e7682.jpg](../icml_results/2584_Instruction%20Lens%20Score_%20Your%20Instruction%20Contributes%20a%20Powerful%20Object%20Hallucination%20Detector%20for%20Mu/tables/e65a5b1f1bd80e8ba5b30ee54947c5ee340bc21736b31b2b8cdff5909e0e7682.jpg)

<a id="paper-2553"></a>

## LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching


### Images

![23c58171624ab1792c4ce9c59ca93fd0a4a8e05b0258d06c7faf0aaaeafc4994.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/23c58171624ab1792c4ce9c59ca93fd0a4a8e05b0258d06c7faf0aaaeafc4994.jpg)

![26b5b1d94a8179fcb13628c8d7774585f1d7898a65ea74ced96323615b46a216.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/26b5b1d94a8179fcb13628c8d7774585f1d7898a65ea74ced96323615b46a216.jpg)

![497a5f9812277825f245712552f9532417189af305f2fd161a0561e6429d997c.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/497a5f9812277825f245712552f9532417189af305f2fd161a0561e6429d997c.jpg)

![53b0fe974187f8fd95e4d3f690f63d5e2582ae4beaa94aba1fed2982dce975e6.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/53b0fe974187f8fd95e4d3f690f63d5e2582ae4beaa94aba1fed2982dce975e6.jpg)

![75f58c9733e3ddc7cb92cb3bc3ed7aa18b7505073886530b040ec6798c7833f0.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/75f58c9733e3ddc7cb92cb3bc3ed7aa18b7505073886530b040ec6798c7833f0.jpg)

![8c407c6c4543e3bcc837b9ebded3bb6b309d0701d0c3d75a4a477f0960a455e5.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/8c407c6c4543e3bcc837b9ebded3bb6b309d0701d0c3d75a4a477f0960a455e5.jpg)

![94928789d178ad0899575ad5309a5eaa4ab0e2f1c25ddb2d70d7894712f21a42.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/94928789d178ad0899575ad5309a5eaa4ab0e2f1c25ddb2d70d7894712f21a42.jpg)

![99e6b3c99b37ba61eef75ea4706da898597d4be1477c4661dcb31e8fbd4fe138.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/99e6b3c99b37ba61eef75ea4706da898597d4be1477c4661dcb31e8fbd4fe138.jpg)

![a1719e50073da2377d1fbf479de0099d63ccd7101e6ad6cd7379d9111253686c.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/a1719e50073da2377d1fbf479de0099d63ccd7101e6ad6cd7379d9111253686c.jpg)

![a1ba16d033b81c7a6b5e067ae805853ca4b9ca1ca1a85900dbecdbd9dccc4481.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/a1ba16d033b81c7a6b5e067ae805853ca4b9ca1ca1a85900dbecdbd9dccc4481.jpg)

![be93686c21979b46a14784b40445d9700e6852841c9ffe72daad8f98ff587e2e.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/be93686c21979b46a14784b40445d9700e6852841c9ffe72daad8f98ff587e2e.jpg)

![bf9ea9171f75bd77156e16b9d97eea2abdbe2092ae03e910a89cbbd81d5151ce.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/bf9ea9171f75bd77156e16b9d97eea2abdbe2092ae03e910a89cbbd81d5151ce.jpg)

![d4b14a3df568f857478bb1baaed9652c4a5dd40add107e6870f1ed12c31a2a0f.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/d4b14a3df568f857478bb1baaed9652c4a5dd40add107e6870f1ed12c31a2a0f.jpg)

![dcfca9f046401b04182557568218abb2c8e0d327a4a8abb4ef210bc4ab90a86c.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/dcfca9f046401b04182557568218abb2c8e0d327a4a8abb4ef210bc4ab90a86c.jpg)

![eaf48995a2b5e9be3d1ad62f785971f7a1ed1955c0a333a9eaeb8719f538893e.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/images/eaf48995a2b5e9be3d1ad62f785971f7a1ed1955c0a333a9eaeb8719f538893e.jpg)

### Tables

![073da5af8576aa90b07238a1b3156aa1e7583b3e48f16e7fb1ef9ee697415e82.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/tables/073da5af8576aa90b07238a1b3156aa1e7583b3e48f16e7fb1ef9ee697415e82.jpg)

![10b55601e894f2a59079159d9b5a548143d40a64843770744105684e73799a95.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/tables/10b55601e894f2a59079159d9b5a548143d40a64843770744105684e73799a95.jpg)

![87d7d118597da0d7c895001b7a3607543027f539c8d68de7e6f4d2c187be6492.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/tables/87d7d118597da0d7c895001b7a3607543027f539c8d68de7e6f4d2c187be6492.jpg)

![b9c2e25ac7f6890f917c00d3e321b1ca95ee48f138a95b460d084502d977d2d6.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/tables/b9c2e25ac7f6890f917c00d3e321b1ca95ee48f138a95b460d084502d977d2d6.jpg)

![de3c689a2bf4c7737384bce92a8c17109783293b5bb4874689cd33d53f3cbf0f.jpg](../icml_results/2585_LithoGRPO_%20Fast%20Inverse%20Lithography%20via%20GRPO%20Reinforced%20Flow%20Matching/tables/de3c689a2bf4c7737384bce92a8c17109783293b5bb4874689cd33d53f3cbf0f.jpg)

<a id="paper-2554"></a>

## Adversarial Attack and Defense for Denoising Diffusion Sampling


### Images

![07e50c4031bff103b7f09792aedee5de310392cbda142c0afb445a39b9bb7254.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/07e50c4031bff103b7f09792aedee5de310392cbda142c0afb445a39b9bb7254.jpg)

![12e303def3db8003f4950f1f5b589ec2ac2bf03b9bda7839a4a0494622b9a62f.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/12e303def3db8003f4950f1f5b589ec2ac2bf03b9bda7839a4a0494622b9a62f.jpg)

![14c449e16c1fd1031d5b98ff15da3c01dfa4d66296e23c39469239d3b9ffcf7e.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/14c449e16c1fd1031d5b98ff15da3c01dfa4d66296e23c39469239d3b9ffcf7e.jpg)

![1ae93b87f0ef9400f33ec2cf4498574615de58654b9a25791445a25c6db9404d.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/1ae93b87f0ef9400f33ec2cf4498574615de58654b9a25791445a25c6db9404d.jpg)

![2356513b5e06c54e66572c6ec76123a3c1dba79ee36b062c30eaea204f050e88.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/2356513b5e06c54e66572c6ec76123a3c1dba79ee36b062c30eaea204f050e88.jpg)

![2684173b75f915131340538d18176182e7662e835e656c3407dde5bb2e207f30.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/2684173b75f915131340538d18176182e7662e835e656c3407dde5bb2e207f30.jpg)

![35c7ea742a3389922bfff91053f6e02a67740522e29f381dc2d74a34140bac79.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/35c7ea742a3389922bfff91053f6e02a67740522e29f381dc2d74a34140bac79.jpg)

![3d067c6bfbbed6f4993f411c70285c286553856fc4c6eb0e2c63b69d192ca55f.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/3d067c6bfbbed6f4993f411c70285c286553856fc4c6eb0e2c63b69d192ca55f.jpg)

![40a84466c429f4012d838f15090e26f4375278d7f4057bc3828975390ed21ad4.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/40a84466c429f4012d838f15090e26f4375278d7f4057bc3828975390ed21ad4.jpg)

![46ce241d562d4407d9804b309c3d8f19a134bba24e82cdcb74d386337d0e5885.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/46ce241d562d4407d9804b309c3d8f19a134bba24e82cdcb74d386337d0e5885.jpg)

![47dd73c7fedf10c4f04401f71aeacbf49d8acda95bd9a256c97a8948400424cc.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/47dd73c7fedf10c4f04401f71aeacbf49d8acda95bd9a256c97a8948400424cc.jpg)

![4caba3a7cc18650404e3bf49ee16bf1e335ead4cdb52ba4ae1da757deb6ae243.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/4caba3a7cc18650404e3bf49ee16bf1e335ead4cdb52ba4ae1da757deb6ae243.jpg)

![4eb04c4db141c7bfd3fa322648829583f26809c6fbcdb32ac5308a485bf40a65.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/4eb04c4db141c7bfd3fa322648829583f26809c6fbcdb32ac5308a485bf40a65.jpg)

![524c19c24cd3a1feb26d0735597c78f9e9d1043b3d0eef14a9978ab7d81f0de8.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/524c19c24cd3a1feb26d0735597c78f9e9d1043b3d0eef14a9978ab7d81f0de8.jpg)

![5a0a5735991918139dcad1b0acfd6bf7c6a63985551f0711426f8016742e050e.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/5a0a5735991918139dcad1b0acfd6bf7c6a63985551f0711426f8016742e050e.jpg)

![6196452c7fa151c9b1078d43a35d4181d79fe13d89791b72380f4fda91ddda72.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/6196452c7fa151c9b1078d43a35d4181d79fe13d89791b72380f4fda91ddda72.jpg)

![66329562d018d059202f8d2ac4857a15b0fa30e2ae0eddb689b32605e648f07c.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/66329562d018d059202f8d2ac4857a15b0fa30e2ae0eddb689b32605e648f07c.jpg)

![6b4ab514a30a8479d5e964935a38afd48f91217fb0c57291be31a33a62d3df29.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/6b4ab514a30a8479d5e964935a38afd48f91217fb0c57291be31a33a62d3df29.jpg)

![6fd757dab82197894698db3b201da41e06757fdd87190f83f4129dc6d2b559ad.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/6fd757dab82197894698db3b201da41e06757fdd87190f83f4129dc6d2b559ad.jpg)

![729a99fa8475d5d881979463b71d693a6582e87ee94cc4d940b84289b9b0d6d0.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/729a99fa8475d5d881979463b71d693a6582e87ee94cc4d940b84289b9b0d6d0.jpg)

![7d2f181034c40aec1c2e6703adfe8105e2cdfcff5326e9cb975c8d95af39b8ea.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/7d2f181034c40aec1c2e6703adfe8105e2cdfcff5326e9cb975c8d95af39b8ea.jpg)

![829aaadd6101eb975c570f68cf6b9712947e3c5de542505e7be3648688346ee0.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/829aaadd6101eb975c570f68cf6b9712947e3c5de542505e7be3648688346ee0.jpg)

![875a92bc50adf0f74e7408c60b792208523a3a08599653196d88043b7c76cf82.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/875a92bc50adf0f74e7408c60b792208523a3a08599653196d88043b7c76cf82.jpg)

![877457668c0aa66f5707a9dc976e5007fb346e3ed071a3ebc3a762bf01690b18.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/877457668c0aa66f5707a9dc976e5007fb346e3ed071a3ebc3a762bf01690b18.jpg)

![90660670a9214f80a12860350e6a1bdbd3aae8f8a8158fc2a1d11d264b05b6c4.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/90660670a9214f80a12860350e6a1bdbd3aae8f8a8158fc2a1d11d264b05b6c4.jpg)

![98759d6e1cd5e6bb24df3ffe02dcbf0129dac1fbf25191ebf3061229d28fd0c5.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/98759d6e1cd5e6bb24df3ffe02dcbf0129dac1fbf25191ebf3061229d28fd0c5.jpg)

![a363c93906aa618e65e09d42f4a8174b6ea61e640482af7e8f80efc3e62784b3.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/a363c93906aa618e65e09d42f4a8174b6ea61e640482af7e8f80efc3e62784b3.jpg)

![a9978a94a87a961ec400b426153a1ccbf97c428a693e4520f507b43519257d0f.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/a9978a94a87a961ec400b426153a1ccbf97c428a693e4520f507b43519257d0f.jpg)

![b29eb98a172b20402e923100c3365d6409005f7d42bad80398cfb85efd8b4a9f.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/b29eb98a172b20402e923100c3365d6409005f7d42bad80398cfb85efd8b4a9f.jpg)

![b81eb470fec2e827e5a22505044e9dcdb23ee209ccbb5372071704277d28eb87.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/b81eb470fec2e827e5a22505044e9dcdb23ee209ccbb5372071704277d28eb87.jpg)

![bb5d3075ebc7c13fbf23f05564e0f06568329d894947cd47af27f89f88c30600.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/bb5d3075ebc7c13fbf23f05564e0f06568329d894947cd47af27f89f88c30600.jpg)

![c8fd6ca1a9b0d2b4acd9118b2759d435b8829ce15ae1af51b6a8ef55e895572e.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/c8fd6ca1a9b0d2b4acd9118b2759d435b8829ce15ae1af51b6a8ef55e895572e.jpg)

![d45128956372b714bab00398982252eeef1eabb6b41f44cc068cd0f734bf0a96.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/d45128956372b714bab00398982252eeef1eabb6b41f44cc068cd0f734bf0a96.jpg)

![d79884618e4d68d6243064161f125d80130b8bbfdea10db5d2cf6f5e51e51c28.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/d79884618e4d68d6243064161f125d80130b8bbfdea10db5d2cf6f5e51e51c28.jpg)

![f3224d2713dd3ee0c7f7ce117b03134e77cb5f70010ae055a3b8f3a4f85d7a01.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/images/f3224d2713dd3ee0c7f7ce117b03134e77cb5f70010ae055a3b8f3a4f85d7a01.jpg)

### Tables

![29fe1d52217e76c749c406b8ee8e43908fb800cf09b2ea2aa6117b0e2c884f4c.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/29fe1d52217e76c749c406b8ee8e43908fb800cf09b2ea2aa6117b0e2c884f4c.jpg)

![75e579cddcef8c741846ce0616f8b7f33f2867ba2cb8cdc2b3a5d88347f4838d.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/75e579cddcef8c741846ce0616f8b7f33f2867ba2cb8cdc2b3a5d88347f4838d.jpg)

![9c39be7ace4ac571708b1482f08a2c6e8603e8778afa78f4f31a051d6ab682b2.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/9c39be7ace4ac571708b1482f08a2c6e8603e8778afa78f4f31a051d6ab682b2.jpg)

![9f9cf92ffde6bceb614e49306a474854497e7afdefb915a6f70615350f327ef9.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/9f9cf92ffde6bceb614e49306a474854497e7afdefb915a6f70615350f327ef9.jpg)

![cab6147d7ab389c1d3d899f38501bd3f1ba4471483da6a82df0017f81dd3cf02.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/cab6147d7ab389c1d3d899f38501bd3f1ba4471483da6a82df0017f81dd3cf02.jpg)

![d0160d05b46464cc91c9859284b52df0630d3e0dccf8c273ddfe0586b47bd937.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/d0160d05b46464cc91c9859284b52df0630d3e0dccf8c273ddfe0586b47bd937.jpg)

![d73c12a20ef45f409ee8f4ab78b21fbdaaaf3d62dba809ecf19460d813c7461c.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/d73c12a20ef45f409ee8f4ab78b21fbdaaaf3d62dba809ecf19460d813c7461c.jpg)

![d77ed8ce0f8123414b2e98b8ceeebdfda8d204577bf2a59ec775526d670ccdad.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/d77ed8ce0f8123414b2e98b8ceeebdfda8d204577bf2a59ec775526d670ccdad.jpg)

![e726781b2c4b2b00164c6822bbb8d64116013d550aca9b1f051953187fe2f471.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/e726781b2c4b2b00164c6822bbb8d64116013d550aca9b1f051953187fe2f471.jpg)

![e98949a605dde5b4d9a40703f55fdc1da53006f6c7557b059b23a8acad38e2ab.jpg](../icml_results/2586_Adversarial%20Attack%20and%20Defense%20for%20Denoising%20Diffusion%20Sampling/tables/e98949a605dde5b4d9a40703f55fdc1da53006f6c7557b059b23a8acad38e2ab.jpg)

<a id="paper-2555"></a>

## Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training


### Images

![06891fbd705a3ebdf2ba3561e813fbac002cc86aa8903e540fd0d6ec89ac2ef6.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/images/06891fbd705a3ebdf2ba3561e813fbac002cc86aa8903e540fd0d6ec89ac2ef6.jpg)

![0e95d34b8dd45e456f06003b500326af577aee1e3b3c58182665922db113e784.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/images/0e95d34b8dd45e456f06003b500326af577aee1e3b3c58182665922db113e784.jpg)

![cbee2dd522b6bd2a1eb92380201e88b01b056f60907f20a87263f0da1b96bf2b.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/images/cbee2dd522b6bd2a1eb92380201e88b01b056f60907f20a87263f0da1b96bf2b.jpg)

![f0043fdb5f7fea87acf659407f2072e28fae69c4a6a49691731457a24017f39e.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/images/f0043fdb5f7fea87acf659407f2072e28fae69c4a6a49691731457a24017f39e.jpg)

### Tables

![1b6dd0b33226d8f2d4be609422865739c40107227b9bb679d0e925ad1d81870a.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/1b6dd0b33226d8f2d4be609422865739c40107227b9bb679d0e925ad1d81870a.jpg)

![216d7a8a1de85b708b8a2098c8784082918aaa62da3aa60218f2aa910394cc3e.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/216d7a8a1de85b708b8a2098c8784082918aaa62da3aa60218f2aa910394cc3e.jpg)

![27cfb2c2e42cc0108130a9fc334cdb277126a99088a270603e4c31c5b42c43a1.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/27cfb2c2e42cc0108130a9fc334cdb277126a99088a270603e4c31c5b42c43a1.jpg)

![5572823aed4f34d632b3f53bc882d2eb0cba33f67d1e1dd172c9eefaded62475.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/5572823aed4f34d632b3f53bc882d2eb0cba33f67d1e1dd172c9eefaded62475.jpg)

![5a8e3156d543305575c50d1e6e814909d9ea26996c5a5687f0c62780b56fc64a.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/5a8e3156d543305575c50d1e6e814909d9ea26996c5a5687f0c62780b56fc64a.jpg)

![5d282af34750c6bfecaf0763b255ab0839d4295f6af6767e2e5d09dec0b1ff6f.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/5d282af34750c6bfecaf0763b255ab0839d4295f6af6767e2e5d09dec0b1ff6f.jpg)

![a4894eab94ae8151bca77d261a53ab90099c7304c29fed3025a1c03a62c2fc30.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/a4894eab94ae8151bca77d261a53ab90099c7304c29fed3025a1c03a62c2fc30.jpg)

![b4d305d686082857bee16bc69b2d947227f325ef7fa0c1118bf76e12c8f1f950.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/b4d305d686082857bee16bc69b2d947227f325ef7fa0c1118bf76e12c8f1f950.jpg)

![b74bc6d3b60a24633e7cabe544f43a819727513b2b87440b74453c9c4327f856.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/b74bc6d3b60a24633e7cabe544f43a819727513b2b87440b74453c9c4327f856.jpg)

![bb08c50be6f8164695ba78262e9e2e1e6f4f7a8f9783859aaaf5859b1d4be041.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/bb08c50be6f8164695ba78262e9e2e1e6f4f7a8f9783859aaaf5859b1d4be041.jpg)

![d832987f38e94525736dc67385b914d59e6267ac7a187d83c0c4472e25ae6f67.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/d832987f38e94525736dc67385b914d59e6267ac7a187d83c0c4472e25ae6f67.jpg)

![d8350e7e285273c44a93f53579d09df974b7a1c7f32538ebc850050fefca4d42.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/d8350e7e285273c44a93f53579d09df974b7a1c7f32538ebc850050fefca4d42.jpg)

![e720223deb2661bcf1dbcb7c3996c5530cef9fdf10ac20edc2d179ed80952135.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/e720223deb2661bcf1dbcb7c3996c5530cef9fdf10ac20edc2d179ed80952135.jpg)

![f5a96dc6725686dfea569f743fd772393334c352d7d2117558c31ea7ea99e51f.jpg](../icml_results/2587_Reinforcement%20Fine-Tuning%20Naturally%20Mitigates%20Forgetting%20in%20Continual%20Post-Training/tables/f5a96dc6725686dfea569f743fd772393334c352d7d2117558c31ea7ea99e51f.jpg)

<a id="paper-2556"></a>

## Random Process Flow Matching: Generative Implicit Representations of Multivariate Random Fields


### Images

![06301500a2ee22195f432582f38633817a35d445e49cdcbec326269adba5bb0e.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/06301500a2ee22195f432582f38633817a35d445e49cdcbec326269adba5bb0e.jpg)

![07193ba0f907ba6f880b16cf2ed17208a138ed35e6b75d87b9209dfd9a1a8b65.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/07193ba0f907ba6f880b16cf2ed17208a138ed35e6b75d87b9209dfd9a1a8b65.jpg)

![16f0e532e35e4696f4a2c1c710080300a7889dd0eb30f3349e09865bbe2cb84a.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/16f0e532e35e4696f4a2c1c710080300a7889dd0eb30f3349e09865bbe2cb84a.jpg)

![20fff28da1eae5ac384c1a7d213318ce4ca24920b895477b1d565e4258c61b67.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/20fff28da1eae5ac384c1a7d213318ce4ca24920b895477b1d565e4258c61b67.jpg)

![23ff42360819da82d45472a3dde6fb588288ebc7d7251bf1b61ea411ee79c39d.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/23ff42360819da82d45472a3dde6fb588288ebc7d7251bf1b61ea411ee79c39d.jpg)

![2f90f22e7ee891170339cac945a128b35099b308dd1c7d13306e9f297491fc88.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/2f90f22e7ee891170339cac945a128b35099b308dd1c7d13306e9f297491fc88.jpg)

![32caeb52cfc1048f8d1fa2384f6c0dd9ff47eb1b3852a54040257ea5a1e825a1.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/32caeb52cfc1048f8d1fa2384f6c0dd9ff47eb1b3852a54040257ea5a1e825a1.jpg)

![4e63e72f7833df25857a71131f2394ba717c52cf553106f8286501d843d58ce2.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/4e63e72f7833df25857a71131f2394ba717c52cf553106f8286501d843d58ce2.jpg)

![7d6675cf35b7c916f8c2f0bf34f5862bbab0c30b5e89c78ba458fc8a1f9a914b.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/7d6675cf35b7c916f8c2f0bf34f5862bbab0c30b5e89c78ba458fc8a1f9a914b.jpg)

![7ebd3e1022f47ddaa5b86495240b5e886c8200f74ca9fe46c97df25840032e67.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/7ebd3e1022f47ddaa5b86495240b5e886c8200f74ca9fe46c97df25840032e67.jpg)

![915110f957afbce497c5cfbfe07e24dc241996dcc1386444e44bddbffd3e66af.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/915110f957afbce497c5cfbfe07e24dc241996dcc1386444e44bddbffd3e66af.jpg)

![cc9d30bf8e53ae131af22c81eb064bfbf756ba80df19f0a2972304c7bcd8419d.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/cc9d30bf8e53ae131af22c81eb064bfbf756ba80df19f0a2972304c7bcd8419d.jpg)

![d4e117f24e426ecc12b4b4f6d44b33f7c31547a11306ec5df5acde9827145f2b.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/d4e117f24e426ecc12b4b4f6d44b33f7c31547a11306ec5df5acde9827145f2b.jpg)

![d77598a64bc94b3359abcb10451c9ef80c2cfa1c6c3d380b16ebba557839be70.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/d77598a64bc94b3359abcb10451c9ef80c2cfa1c6c3d380b16ebba557839be70.jpg)

![ddad95055ed26cbee8cf6ef1298e067c5a7476ba2e48245b453be93efc634088.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/ddad95055ed26cbee8cf6ef1298e067c5a7476ba2e48245b453be93efc634088.jpg)

![e5257cfbd7f4da801dd8cd18b39d36c605f6544d6aad50d9161637adfa2551f2.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/e5257cfbd7f4da801dd8cd18b39d36c605f6544d6aad50d9161637adfa2551f2.jpg)

![e97987fb208335643dd8a56dcd0e83be27120c003f7975ee182101f84e08c7cf.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/e97987fb208335643dd8a56dcd0e83be27120c003f7975ee182101f84e08c7cf.jpg)

![f05658b4955a18713960cdd1c8e91e70ced2877c867614fa09952dc8c42b51fb.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/f05658b4955a18713960cdd1c8e91e70ced2877c867614fa09952dc8c42b51fb.jpg)

![f41c8f2bc378af8f29d790127271baad56e9669df0e734e403c7844d07a1d16d.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/images/f41c8f2bc378af8f29d790127271baad56e9669df0e734e403c7844d07a1d16d.jpg)

### Tables

![062eb0e6c4b91a7fa1c42e39a9073f055c16118d0c094f4d7b38dd931c0d4760.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/062eb0e6c4b91a7fa1c42e39a9073f055c16118d0c094f4d7b38dd931c0d4760.jpg)

![405a86b51a21703adfd95cc243f15c341c001de116aff49d7a92c76e6cd99f17.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/405a86b51a21703adfd95cc243f15c341c001de116aff49d7a92c76e6cd99f17.jpg)

![642588cf0eb52235fe015c2c9704349551531465bdf2a188d45913bb7a47da38.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/642588cf0eb52235fe015c2c9704349551531465bdf2a188d45913bb7a47da38.jpg)

![7ad8a0de078a095e0f51ee12d3ad0c0dc5938cf3f1f6090062a586f9d681635c.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/7ad8a0de078a095e0f51ee12d3ad0c0dc5938cf3f1f6090062a586f9d681635c.jpg)

![9ae2c925fb0b61ba27046f275373c6f5194f7d0a39343b12913be16393dcf2d0.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/9ae2c925fb0b61ba27046f275373c6f5194f7d0a39343b12913be16393dcf2d0.jpg)

![c7461753f7008c688609cc48e1b5dbb6a43a50947c18f60e288855741fa864da.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/c7461753f7008c688609cc48e1b5dbb6a43a50947c18f60e288855741fa864da.jpg)

![d1a5a72318debb1a3eed3f1570a7ed73277f8d1b9191349bbca75c55483e6e3f.jpg](../icml_results/2588_Random%20Process%20Flow%20Matching_%20Generative%20Implicit%20Representations%20of%20Multivariate%20Random%20Fields/tables/d1a5a72318debb1a3eed3f1570a7ed73277f8d1b9191349bbca75c55483e6e3f.jpg)

<a id="paper-2557"></a>

## Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines


### Images

![00fabd11b642302d15abf322e233d27517d8bd02181243b7ffc0e45d51e06d61.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/00fabd11b642302d15abf322e233d27517d8bd02181243b7ffc0e45d51e06d61.jpg)

![31934f08b87ae59b354b31927a4bc2e320c14dbda878588d386c67422f8f7227.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/31934f08b87ae59b354b31927a4bc2e320c14dbda878588d386c67422f8f7227.jpg)

![4446733f8b00c334efb159ebd6ee0062053997b100bc35bca6c574283d38f5ca.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/4446733f8b00c334efb159ebd6ee0062053997b100bc35bca6c574283d38f5ca.jpg)

![a7df6c6fb0aae0d7fa2f534c75892e63d8b890afc6ce0c032b56353e042c6bc3.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/a7df6c6fb0aae0d7fa2f534c75892e63d8b890afc6ce0c032b56353e042c6bc3.jpg)

![ad213e336a99400537d4823b7b1e015bdb1ed433b2503803636346d220d4a6c2.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/ad213e336a99400537d4823b7b1e015bdb1ed433b2503803636346d220d4a6c2.jpg)

![be3f14f2b60c8dfee5037b6e121de5bee5489e1df7c66fb0977cba0959c05331.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/images/be3f14f2b60c8dfee5037b6e121de5bee5489e1df7c66fb0977cba0959c05331.jpg)

### Tables

![01fc7a6b169e9898e627fdf8692b20ddb37a9c15374c4d2d56f8e1a02a39affa.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/tables/01fc7a6b169e9898e627fdf8692b20ddb37a9c15374c4d2d56f8e1a02a39affa.jpg)

![03c201f873e39dd133b71718cd579d222e5dd119122e68035e3a07f66e383969.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/tables/03c201f873e39dd133b71718cd579d222e5dd119122e68035e3a07f66e383969.jpg)

![0709e4c9316478c1eb1acc927169408b72a7ce2cb18d9c1c54df7dc3831125df.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/tables/0709e4c9316478c1eb1acc927169408b72a7ce2cb18d9c1c54df7dc3831125df.jpg)

![bf63a07e76ad8fc8943e671e3f883169ecc6ba837e13b0c7bc24d13aa2fe6cca.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/tables/bf63a07e76ad8fc8943e671e3f883169ecc6ba837e13b0c7bc24d13aa2fe6cca.jpg)

![dd18ea3df07c5d6f3d78cfcf43d562e20ba998e34e6ab45232a186d154d155b3.jpg](../icml_results/2589_Towards%20Resource-Efficient%20LLMs_%20End-to-End%20Energy%20Accounting%20of%20Distillation%20Pipelines/tables/dd18ea3df07c5d6f3d78cfcf43d562e20ba998e34e6ab45232a186d154d155b3.jpg)

<a id="paper-2558"></a>

## Population-Aware Imitation Learning in Mean-field Games with Common Noise


### Images

![107409d5cab1547474669a2a50ab38e468abc0129f8553afcf4877450c8eb14b.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/107409d5cab1547474669a2a50ab38e468abc0129f8553afcf4877450c8eb14b.jpg)

![151dedc78114428578445fec25f6b0b6f30a56311493a165c888806e769e01b8.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/151dedc78114428578445fec25f6b0b6f30a56311493a165c888806e769e01b8.jpg)

![1c8dbf08513973f214b6f962cd85cd7bfce366bead13496892cb6776aa5f43c7.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/1c8dbf08513973f214b6f962cd85cd7bfce366bead13496892cb6776aa5f43c7.jpg)

![22a2608871dc55f50034ca949d24d2f76aa5e48a82d209b1c408fdb44b73429e.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/22a2608871dc55f50034ca949d24d2f76aa5e48a82d209b1c408fdb44b73429e.jpg)

![4cbcfcd9729a7e6e831e1d058e9511834f28304d7da75eb4c6726f6db8fcdf65.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/4cbcfcd9729a7e6e831e1d058e9511834f28304d7da75eb4c6726f6db8fcdf65.jpg)

![69c18bf4509592438f7851255cb33cb4d5941063c390a54b544fda9a1d78bbc9.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/69c18bf4509592438f7851255cb33cb4d5941063c390a54b544fda9a1d78bbc9.jpg)

![736779e80f439639e27de25f219e903d91632c8a37e4cb770ef657037b8693e2.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/736779e80f439639e27de25f219e903d91632c8a37e4cb770ef657037b8693e2.jpg)

![79123da79f28c469cdb37766f6cf47314bcb278aadf3b5c8cb4165351d9a7590.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/79123da79f28c469cdb37766f6cf47314bcb278aadf3b5c8cb4165351d9a7590.jpg)

![7e22d5a84e3968fdfb335ef64c707419f6b569098067e1e16ef96873aa951448.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/7e22d5a84e3968fdfb335ef64c707419f6b569098067e1e16ef96873aa951448.jpg)

![80757f5c39e2970d94129e1994f79e347d377c8d7e178426c3996add40baa82e.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/80757f5c39e2970d94129e1994f79e347d377c8d7e178426c3996add40baa82e.jpg)

![8dc1bd403f3e7671c71f4db51d0117de733231bf85b7b044b70dea6315c5b5a0.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/8dc1bd403f3e7671c71f4db51d0117de733231bf85b7b044b70dea6315c5b5a0.jpg)

![8e625eb918df192401403f4744043507e170203e5ec91a92ebf8e8b6e5e81f44.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/8e625eb918df192401403f4744043507e170203e5ec91a92ebf8e8b6e5e81f44.jpg)

![921a9463ded4eb3bee50f39975b6b8797572ad44355293aa1ba0e215b62df7d6.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/921a9463ded4eb3bee50f39975b6b8797572ad44355293aa1ba0e215b62df7d6.jpg)

![caeb5a84f4151b95e0aa075b60a6c21304f3560532e72a26dca677225570a1b8.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/caeb5a84f4151b95e0aa075b60a6c21304f3560532e72a26dca677225570a1b8.jpg)

![d36d79ed74acd15684e6366571bca5652a29cb6d9861a9861d4435f39601cde1.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/d36d79ed74acd15684e6366571bca5652a29cb6d9861a9861d4435f39601cde1.jpg)

![d5aeb3d31862a2b110e8ac28277d0cca8f1ae56a302428977c6b577672b7eb4b.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/d5aeb3d31862a2b110e8ac28277d0cca8f1ae56a302428977c6b577672b7eb4b.jpg)

![db38e3895da0e773ca24c900adb7114f26e75cb952a97e3b2ecc1bb47f4b31bb.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/db38e3895da0e773ca24c900adb7114f26e75cb952a97e3b2ecc1bb47f4b31bb.jpg)

![df74634a54103fdf65dc978c5a70917e2d46d3a6c522c1abf7066cdb5c50287c.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/df74634a54103fdf65dc978c5a70917e2d46d3a6c522c1abf7066cdb5c50287c.jpg)

![dfbc11acf22a4836c5564486d3383ac1fdc2bb9bdc267e7c097a9807856851d5.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/dfbc11acf22a4836c5564486d3383ac1fdc2bb9bdc267e7c097a9807856851d5.jpg)

![e16bf3757641136e6fd29852763015d7c3da30e1d1353cceda03dba6c314625c.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/e16bf3757641136e6fd29852763015d7c3da30e1d1353cceda03dba6c314625c.jpg)

![f63493e5594fc669464d41f578f3a94d5d35ee5e94fb72bcc0c07cebb44b2f1c.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/images/f63493e5594fc669464d41f578f3a94d5d35ee5e94fb72bcc0c07cebb44b2f1c.jpg)

### Tables

![2ebb6313246db6bddf0be417f4f86b9655209e8e5773e72b001fc97dbf98f9d3.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/tables/2ebb6313246db6bddf0be417f4f86b9655209e8e5773e72b001fc97dbf98f9d3.jpg)

![4bbe4774a42e659aae142cbcaf729d558b424dead0c1fb37b2c1a487b51f0557.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/tables/4bbe4774a42e659aae142cbcaf729d558b424dead0c1fb37b2c1a487b51f0557.jpg)

![627d4a701dd191e1174c54d1368ebcf82bc3367f8dc22ac107e3be7fd6ad2515.jpg](../icml_results/2590_Population-Aware%20Imitation%20Learning%20in%20Mean-field%20Games%20with%20Common%20Noise/tables/627d4a701dd191e1174c54d1368ebcf82bc3367f8dc22ac107e3be7fd6ad2515.jpg)

<a id="paper-2559"></a>

## ConTSG-Bench: A Unified Benchmark for Conditional Time Series Generation


### Images

![0863913bb4dfcef043300ee25d94dac02416f9c3b7043680d4436597ed0773e6.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/0863913bb4dfcef043300ee25d94dac02416f9c3b7043680d4436597ed0773e6.jpg)

![3203551735086c4728c1b28e1d24ace15d7911f34b1b326e1f1e1b4973f015a3.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/3203551735086c4728c1b28e1d24ace15d7911f34b1b326e1f1e1b4973f015a3.jpg)

![3292cd71369ca77a78483c37270af7c4428ade9bc80b3546fed97c86310ca308.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/3292cd71369ca77a78483c37270af7c4428ade9bc80b3546fed97c86310ca308.jpg)

![33b3697be82ce2910786343a7f4f270a66dd3f7f72dfbd511024b7ac5511e34e.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/33b3697be82ce2910786343a7f4f270a66dd3f7f72dfbd511024b7ac5511e34e.jpg)

![38b4bb94613ab4dee81fa3ac53a5b2883c2fc295990bb8cae5676dfb0d2c67e8.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/38b4bb94613ab4dee81fa3ac53a5b2883c2fc295990bb8cae5676dfb0d2c67e8.jpg)

![390f2448938a600d9eced9439f0f8530a99177e1aaf86a787ad0ba14ded9ef82.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/390f2448938a600d9eced9439f0f8530a99177e1aaf86a787ad0ba14ded9ef82.jpg)

![497919bf898042e07d961e55b589b2de5441d2f8ed74e0bd7c9692f9365252c6.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/497919bf898042e07d961e55b589b2de5441d2f8ed74e0bd7c9692f9365252c6.jpg)

![6651d7d68bca8138450342eb2a0c0665f6f90d819c0bf6e279928f8333dd3e2d.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/6651d7d68bca8138450342eb2a0c0665f6f90d819c0bf6e279928f8333dd3e2d.jpg)

![6f546a7be8793fc9d8927643946a3d089dc32384b25b0eae0ae6467e52ce80b2.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/6f546a7be8793fc9d8927643946a3d089dc32384b25b0eae0ae6467e52ce80b2.jpg)

![770f85648b3de997c1eca697150f9b4417995a01365546ab08a21e0e9356ef54.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/770f85648b3de997c1eca697150f9b4417995a01365546ab08a21e0e9356ef54.jpg)

![7d126ccce4a66394a175986ee894d49cdbadbc250e773366ebded91040ba6e92.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/7d126ccce4a66394a175986ee894d49cdbadbc250e773366ebded91040ba6e92.jpg)

![88974307c37cf239643abc5f62851bcc14faff12fe460974447268a6e9d7b8ea.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/88974307c37cf239643abc5f62851bcc14faff12fe460974447268a6e9d7b8ea.jpg)

![8ca1b45aa19fb9147d6e9c84cc4bb677fe767fa57eb409c236f4f4ae30101d3b.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/8ca1b45aa19fb9147d6e9c84cc4bb677fe767fa57eb409c236f4f4ae30101d3b.jpg)

![8f7462a7000de3eea0af4b891e4faaff85dfcdae4b65a5c287016364d38f391d.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/8f7462a7000de3eea0af4b891e4faaff85dfcdae4b65a5c287016364d38f391d.jpg)

![98630ae38743d0942b41b8dd8fa4a3da1cbd251ce202e36c37e2ea18b059b3b5.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/98630ae38743d0942b41b8dd8fa4a3da1cbd251ce202e36c37e2ea18b059b3b5.jpg)

![a42554f19e4c2049d13a646eb8070d95a508064a1dc7961b0316e9f9db0b0e76.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/a42554f19e4c2049d13a646eb8070d95a508064a1dc7961b0316e9f9db0b0e76.jpg)

![c455482ab83c8654238d6464c1c30c3d8c7e34942e580f5781029c8fc0c982fa.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/c455482ab83c8654238d6464c1c30c3d8c7e34942e580f5781029c8fc0c982fa.jpg)

![dc77fd83780bcf6206b7be07bea3c10ff8009c39931bc50fcaef2e96a2c947c6.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/dc77fd83780bcf6206b7be07bea3c10ff8009c39931bc50fcaef2e96a2c947c6.jpg)

![dd03e8aafa41685f1ece7e987aaa99c0c67f1273009b3d2cedeaeeb089226859.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/dd03e8aafa41685f1ece7e987aaa99c0c67f1273009b3d2cedeaeeb089226859.jpg)

![e5642b1bb48ee4650c2ae58f3c6ad186ef3cbfa0ec0e503ed7c077a65ab5e51b.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/e5642b1bb48ee4650c2ae58f3c6ad186ef3cbfa0ec0e503ed7c077a65ab5e51b.jpg)

![e7a458eea91734186beb8579254db8afe9bbe18e05cb29ea0fd9c92d752f5ece.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/e7a458eea91734186beb8579254db8afe9bbe18e05cb29ea0fd9c92d752f5ece.jpg)

![eec13f13bdbde73077d25bc3b46c3f377e2f635b20ffda3f744216db9df80a9b.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/eec13f13bdbde73077d25bc3b46c3f377e2f635b20ffda3f744216db9df80a9b.jpg)

![f284fa738f031fe279f0802dcdcb90de76f1cbe3f2a939bfb7b2fd755c0c1149.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/f284fa738f031fe279f0802dcdcb90de76f1cbe3f2a939bfb7b2fd755c0c1149.jpg)

![f90c0ed5168334dc73d6e70e7cf412caf47a217beb08768e5cfc9ae421d6e0ba.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/f90c0ed5168334dc73d6e70e7cf412caf47a217beb08768e5cfc9ae421d6e0ba.jpg)

![fa3612c3b661340f23ac06583ebc5faa54c4bc31f461e3563cc361384cd9da66.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/images/fa3612c3b661340f23ac06583ebc5faa54c4bc31f461e3563cc361384cd9da66.jpg)

### Tables

![15b127df8603fbcd552e417838f381810eddccc021c8c681c3a2b32be1431fb2.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/15b127df8603fbcd552e417838f381810eddccc021c8c681c3a2b32be1431fb2.jpg)

![1817a9ed5a9a431dee0adda40a3c8376c88f3c7a0e3b03f47590c8a7396f73ca.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/1817a9ed5a9a431dee0adda40a3c8376c88f3c7a0e3b03f47590c8a7396f73ca.jpg)

![1a4e8879a3385e7dca7b7f9c35e4a77282612c971f30e5d964d115aad4ed5b08.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/1a4e8879a3385e7dca7b7f9c35e4a77282612c971f30e5d964d115aad4ed5b08.jpg)

![1b21e5a8b484f4bed71d08d98f99ad61a9cc977eb7a63a5624ebd571afb3a16c.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/1b21e5a8b484f4bed71d08d98f99ad61a9cc977eb7a63a5624ebd571afb3a16c.jpg)

![3877711aeae077b2eff32545ef981489be41e94107166ff76f31f0a97daeea2c.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/3877711aeae077b2eff32545ef981489be41e94107166ff76f31f0a97daeea2c.jpg)

![48f25cefc653d4afa98595327c3202b786c4c923fa4c912b51686546ec688ba6.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/48f25cefc653d4afa98595327c3202b786c4c923fa4c912b51686546ec688ba6.jpg)

![4c4564d2ce62149cc6bf356633abb14de931e3a5f89706bda7e2cc02dfa495cc.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/4c4564d2ce62149cc6bf356633abb14de931e3a5f89706bda7e2cc02dfa495cc.jpg)

![4c635bcfdb877dbf7b29e555c3f3728292f7a186cf269e0d9def96ed8283742e.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/4c635bcfdb877dbf7b29e555c3f3728292f7a186cf269e0d9def96ed8283742e.jpg)

![4d17364cfce6ab03cff593a30fa95b72cd9ec1b56dbea3324e1f7821d7ebf4ef.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/4d17364cfce6ab03cff593a30fa95b72cd9ec1b56dbea3324e1f7821d7ebf4ef.jpg)

![540ca325c90cb423a61af32f61fe962b0c9a90ef9c56c4019be94eb7e31a0757.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/540ca325c90cb423a61af32f61fe962b0c9a90ef9c56c4019be94eb7e31a0757.jpg)

![62a2df080e41c5a7ee0d16b35bbdf90cf8b43c95dd66e050db3b999a1b18992e.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/62a2df080e41c5a7ee0d16b35bbdf90cf8b43c95dd66e050db3b999a1b18992e.jpg)

![63ee6222fb68113a642cefaad13eff4aa0d3707d1731e95bdbb5046cf31950cb.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/63ee6222fb68113a642cefaad13eff4aa0d3707d1731e95bdbb5046cf31950cb.jpg)

![69efe965657623c9bceef5d6f4bd1c0d2afe03424be745af560750de4c5b614c.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/69efe965657623c9bceef5d6f4bd1c0d2afe03424be745af560750de4c5b614c.jpg)

![75ff6b45e963cd9227951f8ec37192306146544681ed7abee6ca70634d650ca7.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/75ff6b45e963cd9227951f8ec37192306146544681ed7abee6ca70634d650ca7.jpg)

![773b52b452822880f3c6a3968dd2ccc47d19d7be5f1129713477470d88ffe54a.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/773b52b452822880f3c6a3968dd2ccc47d19d7be5f1129713477470d88ffe54a.jpg)

![7bfcf242edb8032559487a55d24575d0a1cde24ec288dabd9f8c312c04c0f278.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/7bfcf242edb8032559487a55d24575d0a1cde24ec288dabd9f8c312c04c0f278.jpg)

![81fa216aa23603be05da9aa060cf1287b6e0f5346769880da7ff31f5905b1a55.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/81fa216aa23603be05da9aa060cf1287b6e0f5346769880da7ff31f5905b1a55.jpg)

![86de34eea426f60adbbfd906c5ca3507e93e1ab0897208e9ce1aadd820ba1c6a.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/86de34eea426f60adbbfd906c5ca3507e93e1ab0897208e9ce1aadd820ba1c6a.jpg)

![a0b82b321ccc0dfed6696b1ab96f8c33ab8974572dd1e880fe3dad196dd08b0b.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/a0b82b321ccc0dfed6696b1ab96f8c33ab8974572dd1e880fe3dad196dd08b0b.jpg)

![a4a41838cf5b104944064315b996cb5aed22acbee88d6b59ed53dee9aa6f1f69.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/a4a41838cf5b104944064315b996cb5aed22acbee88d6b59ed53dee9aa6f1f69.jpg)

![a683d5e64718963ac48379b36c5e360b6bdbb9b17ebeb39f2acd19f882eded7b.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/a683d5e64718963ac48379b36c5e360b6bdbb9b17ebeb39f2acd19f882eded7b.jpg)

![ab41a7497c674300e5df365131beca362ffdb937198d918ee61430573110740e.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/ab41a7497c674300e5df365131beca362ffdb937198d918ee61430573110740e.jpg)

![adfc56e6d1fcb0599a10dc72b10c68c6500a3c2083cb5ed9901d7e84be919311.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/adfc56e6d1fcb0599a10dc72b10c68c6500a3c2083cb5ed9901d7e84be919311.jpg)

![b52ead912e71123dfbc4e984106a557f1f2f38149f52728169b0039d0a0d7aff.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/b52ead912e71123dfbc4e984106a557f1f2f38149f52728169b0039d0a0d7aff.jpg)

![b5876c4fdd9f5d57e98f3489c598a2166c983ad9f09421dd6ae870eab5e97014.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/b5876c4fdd9f5d57e98f3489c598a2166c983ad9f09421dd6ae870eab5e97014.jpg)

![bc70782653647e0f889efe875e0177462b2185d99ff8c9bf1574329aa325a35a.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/bc70782653647e0f889efe875e0177462b2185d99ff8c9bf1574329aa325a35a.jpg)

![bd7a9c3f0db681e5b53a94b97ab9a40f04f223ea3763ae67b27b6d4f92f8f0b1.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/bd7a9c3f0db681e5b53a94b97ab9a40f04f223ea3763ae67b27b6d4f92f8f0b1.jpg)

![cf609d8955010a294cf90ba5713e2745c00b8272d071427012ce55ec4898e302.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/cf609d8955010a294cf90ba5713e2745c00b8272d071427012ce55ec4898e302.jpg)

![d6138122baf0f184c91b1d721d8f4f0c061f3b84b0b643662cad19a677ea4e70.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/d6138122baf0f184c91b1d721d8f4f0c061f3b84b0b643662cad19a677ea4e70.jpg)

![e5c378d2f311d43028292105c5fbfc0f8fe3f42ef2e1438887660562fbab7245.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/e5c378d2f311d43028292105c5fbfc0f8fe3f42ef2e1438887660562fbab7245.jpg)

![eaaaad3bd01b6cc8586ff19125648460d884560c3202f53597bf25f1d93f28ea.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/eaaaad3bd01b6cc8586ff19125648460d884560c3202f53597bf25f1d93f28ea.jpg)

![eb7af3040fb7574f8b881312cdc0016d5aee1211af7f8aaf1b207452b15bb188.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/eb7af3040fb7574f8b881312cdc0016d5aee1211af7f8aaf1b207452b15bb188.jpg)

![f0ae477d0a88941dd4d2ad0ce64166e4dcf8ee12d4c0118f95ce2cd5450df1d8.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/f0ae477d0a88941dd4d2ad0ce64166e4dcf8ee12d4c0118f95ce2cd5450df1d8.jpg)

![f5a539d41e6fb6815e2cdee75eaa765e707987dc3403e9d0fff4bd26417ead80.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/f5a539d41e6fb6815e2cdee75eaa765e707987dc3403e9d0fff4bd26417ead80.jpg)

![f8d4231d32725dd19ec1237f04c144b8e6b67af63e7330460dc8b0439a88a19e.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/f8d4231d32725dd19ec1237f04c144b8e6b67af63e7330460dc8b0439a88a19e.jpg)

![fbd681096d05afe59caaa079d0109db71d90c0085492dc969aa3ca0829d85495.jpg](../icml_results/2591_ConTSG-Bench_%20A%20Unified%20Benchmark%20for%20Conditional%20Time%20Series%20Generation/tables/fbd681096d05afe59caaa079d0109db71d90c0085492dc969aa3ca0829d85495.jpg)

<a id="paper-2560"></a>

## Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy


### Images

![0bb288dc5b72e79f61d151644d2a68ac6cfa04dcd26a98de372fdbcf4a740207.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/0bb288dc5b72e79f61d151644d2a68ac6cfa04dcd26a98de372fdbcf4a740207.jpg)

![24cec49c73189c42069b1e71e1b11020a2bd830a47da69e1a3d94290faddf53d.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/24cec49c73189c42069b1e71e1b11020a2bd830a47da69e1a3d94290faddf53d.jpg)

![360c635dcdaf00de717655f1e166ce9b1b73a2862b9ab34c347c525438bbf8eb.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/360c635dcdaf00de717655f1e166ce9b1b73a2862b9ab34c347c525438bbf8eb.jpg)

![394169dbd972ef4669d3aa473f79e10dc39f67c194921a7622c3b446198bcdf7.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/394169dbd972ef4669d3aa473f79e10dc39f67c194921a7622c3b446198bcdf7.jpg)

![3c135d725beb73d148e0bdbc6a346269ada55a10b0277d364dc9b3cd15baa9a5.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/3c135d725beb73d148e0bdbc6a346269ada55a10b0277d364dc9b3cd15baa9a5.jpg)

![3d36ee635c637a38e473a69a868fae69888571284bd2f37748bd23d6735b6e8c.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/3d36ee635c637a38e473a69a868fae69888571284bd2f37748bd23d6735b6e8c.jpg)

![40adbcd2988910e544ddeef0a519a343ecf5137de4dcbc26332863cbb5b0f73e.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/40adbcd2988910e544ddeef0a519a343ecf5137de4dcbc26332863cbb5b0f73e.jpg)

![5c5b82058fedef5a44f08f2267f47d0cfabee2fe71bc683ac5f4b883242bac5c.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/5c5b82058fedef5a44f08f2267f47d0cfabee2fe71bc683ac5f4b883242bac5c.jpg)

![5cfcb8fa2248bac6874c4a137b82061c9d42741b25581d476d962426806884f8.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/5cfcb8fa2248bac6874c4a137b82061c9d42741b25581d476d962426806884f8.jpg)

![67ba4eecc2efdc10edd5426040353dc934e1da674a2fde48c1057c531cd9ffb4.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/67ba4eecc2efdc10edd5426040353dc934e1da674a2fde48c1057c531cd9ffb4.jpg)

![6f5dd4f666ae56bc5ef7c6a606fc054289d2380e17c8ec45a854bd7d130e5b4f.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/6f5dd4f666ae56bc5ef7c6a606fc054289d2380e17c8ec45a854bd7d130e5b4f.jpg)

![749b0b3b13a65f2efb0c8307d8d0e50b8f52eb97aad3006311a0a7f6bdf22a98.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/749b0b3b13a65f2efb0c8307d8d0e50b8f52eb97aad3006311a0a7f6bdf22a98.jpg)

![80186fcd9ab7d523f034df167c58a9bbfb7f4d35afc7b661195b08d0c4fd79b4.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/80186fcd9ab7d523f034df167c58a9bbfb7f4d35afc7b661195b08d0c4fd79b4.jpg)

![8281b6d1887b798f2a2aa649106637ea1287c09eded6df3b78c2aae6ec690730.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/8281b6d1887b798f2a2aa649106637ea1287c09eded6df3b78c2aae6ec690730.jpg)

![a5a4d0225f19d644e3af9a881e9a3989f8106a4396cbcba8080e997bdbd53445.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/a5a4d0225f19d644e3af9a881e9a3989f8106a4396cbcba8080e997bdbd53445.jpg)

![cff5bd8c37d8f84e610ea52c2ed452e22c37d904a3b75a807ce3a119f872df86.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/cff5bd8c37d8f84e610ea52c2ed452e22c37d904a3b75a807ce3a119f872df86.jpg)

![d8952af99c8ebf782afbcf3a91c280e0b74fd6490925add03e390d0e4b601f7c.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/d8952af99c8ebf782afbcf3a91c280e0b74fd6490925add03e390d0e4b601f7c.jpg)

![ede09d13f26bdf97da20e34cb2e3faf055964f818aeddf1659a7fcffb88d175b.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/images/ede09d13f26bdf97da20e34cb2e3faf055964f818aeddf1659a7fcffb88d175b.jpg)

### Tables

![1cd4d291a3fa3c6ff627f9c3676dd3e297ad59d6d827da4d4945f59bca8c4f04.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/1cd4d291a3fa3c6ff627f9c3676dd3e297ad59d6d827da4d4945f59bca8c4f04.jpg)

![3f6d6114bffe21d6896da280f10ee06cabd3989ab73912a14670e22c86fd3f0b.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/3f6d6114bffe21d6896da280f10ee06cabd3989ab73912a14670e22c86fd3f0b.jpg)

![507cc1f6875019ecb5619d4e57a951e897ce1befeff3fc40dde2a4a003fe251c.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/507cc1f6875019ecb5619d4e57a951e897ce1befeff3fc40dde2a4a003fe251c.jpg)

![761fe9e799138377bd0fc7f4cee23cd2380092862c4a8e193842dc832beee57e.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/761fe9e799138377bd0fc7f4cee23cd2380092862c4a8e193842dc832beee57e.jpg)

![81bfa2b32b3c0d05fc100d7d61b33795ca6d9fce6655e934c85fae248deaf835.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/81bfa2b32b3c0d05fc100d7d61b33795ca6d9fce6655e934c85fae248deaf835.jpg)

![98abc391298bd1e3d8638ad9e5b9dbb4917e974fd4c0dc76fbac1fe23b1ac323.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/98abc391298bd1e3d8638ad9e5b9dbb4917e974fd4c0dc76fbac1fe23b1ac323.jpg)

![9d24312e5bb70a71ba36c2f4430009029c55ca3433301a5ccf34ea70b4927777.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/9d24312e5bb70a71ba36c2f4430009029c55ca3433301a5ccf34ea70b4927777.jpg)

![b0a47791b1f2452d6286dbc87f0bc19d51dfa6ef7017e874e75cbd6e942879af.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/b0a47791b1f2452d6286dbc87f0bc19d51dfa6ef7017e874e75cbd6e942879af.jpg)

![b37be7d96d02460e4d8f700930ded52035d9840f1851d3c4d49ca3445ef5cd3d.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/b37be7d96d02460e4d8f700930ded52035d9840f1851d3c4d49ca3445ef5cd3d.jpg)

![be0baf7249fe5cf790a47029b2fbec130ff291baa97b7f36a2d00be63a9669c4.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/be0baf7249fe5cf790a47029b2fbec130ff291baa97b7f36a2d00be63a9669c4.jpg)

![c03a8f8d373c379ec40e171f4a4fd8d60d5369e9bcbc6f37b88a025e659ef6ca.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/c03a8f8d373c379ec40e171f4a4fd8d60d5369e9bcbc6f37b88a025e659ef6ca.jpg)

![c251eb8e021fa5bde23a83b64fb30ddd812dda8b221733a0fccde1195dc35794.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/c251eb8e021fa5bde23a83b64fb30ddd812dda8b221733a0fccde1195dc35794.jpg)

![ec310bed744be18a4fbaa30461a777850a01b2ea698fa9bb120bdc6f7f74fb85.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/ec310bed744be18a4fbaa30461a777850a01b2ea698fa9bb120bdc6f7f74fb85.jpg)

![eeb71bff1e7ecd935cebc79cec9ce1bb477038b2e6e676d050c357cc5dcd6fc1.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/eeb71bff1e7ecd935cebc79cec9ce1bb477038b2e6e676d050c357cc5dcd6fc1.jpg)

![f729061ee1fc12483931938194d44296ffb76cb66d4a0d851da0c593ed4e1848.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/f729061ee1fc12483931938194d44296ffb76cb66d4a0d851da0c593ed4e1848.jpg)

![f81d0d8a1fd8baf9ada17750f363108b94b61849125a670dad2f19ace4e165a0.jpg](../icml_results/2592_Towards%20Foundation%20Models%20for%20Zero-Shot%20Time%20Series%20Anomaly%20Detection_%20Leveraging%20Synthetic%20Data%20and/tables/f81d0d8a1fd8baf9ada17750f363108b94b61849125a670dad2f19ace4e165a0.jpg)

<a id="paper-2561"></a>

## FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates


### Images

![0b73dcc0da752e958e527778b5fd05fa95bb8ef73304c550135f35a11af3effd.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/0b73dcc0da752e958e527778b5fd05fa95bb8ef73304c550135f35a11af3effd.jpg)

![0f2da82ead241e0e807f0f5f628c524664c27a7502b32e153bb22e3f440046e9.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/0f2da82ead241e0e807f0f5f628c524664c27a7502b32e153bb22e3f440046e9.jpg)

![15cdfccce5c47f41ef23e0260b65e80d5c0dee47c52a56a636d0920015b2f646.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/15cdfccce5c47f41ef23e0260b65e80d5c0dee47c52a56a636d0920015b2f646.jpg)

![1a7f88176f94f70969fcf0d4cf61f8a25b9f0d6b612bb7b49cf323feba427c1b.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/1a7f88176f94f70969fcf0d4cf61f8a25b9f0d6b612bb7b49cf323feba427c1b.jpg)

![1c3c79e9cb8c68ef212fbc454ac77ca702b44c59f6f28915b49834d2ce59ad9e.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/1c3c79e9cb8c68ef212fbc454ac77ca702b44c59f6f28915b49834d2ce59ad9e.jpg)

![51ba009d47089d4365a3aed5339c392a2f3c8421e665b31a2bc5027586aafd61.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/51ba009d47089d4365a3aed5339c392a2f3c8421e665b31a2bc5027586aafd61.jpg)

![71a219c6601e730443f19031109707bc5aa85a230dee7e75f935e4c551fe286a.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/71a219c6601e730443f19031109707bc5aa85a230dee7e75f935e4c551fe286a.jpg)

![721437ba83648ef693b3881437a0581946db57b2c3e2171b71ead5a7e34f2aad.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/721437ba83648ef693b3881437a0581946db57b2c3e2171b71ead5a7e34f2aad.jpg)

![8598c531ba03fb222690d3194297ed7cc3022cf3e73c56cc40f1ddb9c34dd36b.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/8598c531ba03fb222690d3194297ed7cc3022cf3e73c56cc40f1ddb9c34dd36b.jpg)

![8c06b97016ec7c5f9fb357685cbab9ea2e0b9ec2f61c9ecdf58ab98b7f902bcc.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/8c06b97016ec7c5f9fb357685cbab9ea2e0b9ec2f61c9ecdf58ab98b7f902bcc.jpg)

![8f728b4f7642596e4e4782f0da2d17b4fea713ac68d4294ef4227b07a6d7bdc5.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/8f728b4f7642596e4e4782f0da2d17b4fea713ac68d4294ef4227b07a6d7bdc5.jpg)

![94ec79409810b9d0464aa480db7f69e0c0ad9e132ed7de5fede0f16ab2c4a2b0.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/94ec79409810b9d0464aa480db7f69e0c0ad9e132ed7de5fede0f16ab2c4a2b0.jpg)

![b7a609d69c313373654831c3022420316738b1f93597483391f536f1824fa873.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/b7a609d69c313373654831c3022420316738b1f93597483391f536f1824fa873.jpg)

![c9e407e6f9d81e3fccf334c6d5f460a0275240b4551c6eaf5c3221c7d2c5e462.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/c9e407e6f9d81e3fccf334c6d5f460a0275240b4551c6eaf5c3221c7d2c5e462.jpg)

![d8d0cebd6b780e94f09b903d246eff8e4f274f2147be0b6e6c91db74dbe358d5.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/d8d0cebd6b780e94f09b903d246eff8e4f274f2147be0b6e6c91db74dbe358d5.jpg)

![f0e7d4a70949939c96a247a6e9baa929573aff209c485c57b817bf3c395c2957.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/f0e7d4a70949939c96a247a6e9baa929573aff209c485c57b817bf3c395c2957.jpg)

![fe659490abbcc2b7aad07dfd76c32b5f567f7f6155e3f73a3401a0d495256a27.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/images/fe659490abbcc2b7aad07dfd76c32b5f567f7f6155e3f73a3401a0d495256a27.jpg)

### Tables

![058b6410ccfe5c6ff042218685dde8fd3b7564510d93f781eb32a1216ccf1404.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/058b6410ccfe5c6ff042218685dde8fd3b7564510d93f781eb32a1216ccf1404.jpg)

![2947a99db129dc6e389d200de8800bc29fcf88c63fcbe0f1f4225d96ad141d00.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/2947a99db129dc6e389d200de8800bc29fcf88c63fcbe0f1f4225d96ad141d00.jpg)

![61f8af69e99abbbaa944a365c57aebe7927f197b110d4cbd3030eb599c60b4d5.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/61f8af69e99abbbaa944a365c57aebe7927f197b110d4cbd3030eb599c60b4d5.jpg)

![69f1e7c8ca3b41f599bf33241004fe481498988f3d7ab1cd752f3d6e7d696bf5.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/69f1e7c8ca3b41f599bf33241004fe481498988f3d7ab1cd752f3d6e7d696bf5.jpg)

![6f1fd0ebde59c2b4c19c8422a3372add5c25239d100527c7d2185c18078089de.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/6f1fd0ebde59c2b4c19c8422a3372add5c25239d100527c7d2185c18078089de.jpg)

![a77ac4ac259e5cb428d0d2fc71cf4b3abe64a9acd6cfb03a89ac7ae22643880e.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/a77ac4ac259e5cb428d0d2fc71cf4b3abe64a9acd6cfb03a89ac7ae22643880e.jpg)

![d24e72ee8526f9728693f0a43ff86b5f219656d9d2408c43d9853c051eb5c51e.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/d24e72ee8526f9728693f0a43ff86b5f219656d9d2408c43d9853c051eb5c51e.jpg)

![d56d0acc4cc01d7aa65082529f3e1bc0eae81521a4eade976270b379922a7202.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/d56d0acc4cc01d7aa65082529f3e1bc0eae81521a4eade976270b379922a7202.jpg)

![ed7e4e98f0acb8688655b7c7815636c09f8f8e80cf950783ddfa1aca08764279.jpg](../icml_results/2593_FluxNet_%20Learning%20Capacity-Constrained%20Local%20Transport%20Operators%20for%20Conservative%20and%20Bounded%20PDE%20Su/tables/ed7e4e98f0acb8688655b7c7815636c09f8f8e80cf950783ddfa1aca08764279.jpg)

<a id="paper-2562"></a>

## Scaling Vision Transformers for Functional MRI with Flat Maps


### Images

![02412bbee3f78386698eb5188e9923fe465114d316cd062bc5adb156d3a7355e.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/02412bbee3f78386698eb5188e9923fe465114d316cd062bc5adb156d3a7355e.jpg)

![2c69ba7cfb8ecb885874d16ac26cef7becd9ae29e5058fe89bb51a9f8c014338.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/2c69ba7cfb8ecb885874d16ac26cef7becd9ae29e5058fe89bb51a9f8c014338.jpg)

![33234709a1596017c53098805dba2c86a49073843c589fefd271c067300294d5.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/33234709a1596017c53098805dba2c86a49073843c589fefd271c067300294d5.jpg)

![339b12f098859bbbb511cd889f5b8e4f1e00198bdb3a3cc2e0bf5701307f7ef4.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/339b12f098859bbbb511cd889f5b8e4f1e00198bdb3a3cc2e0bf5701307f7ef4.jpg)

![3cff325c1bfc5f257ee502ab5c66b1895c182090d308e198c0149991fde7b0fe.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/3cff325c1bfc5f257ee502ab5c66b1895c182090d308e198c0149991fde7b0fe.jpg)

![4d7cd7694b2a2cfecb909b631a2ba2bfb84187329fba80e1952f6e688f1dd6e4.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/4d7cd7694b2a2cfecb909b631a2ba2bfb84187329fba80e1952f6e688f1dd6e4.jpg)

![597f3f26ae0a39d354a0255d99a075ff16244dc4f15618d3067a7236d884dafe.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/597f3f26ae0a39d354a0255d99a075ff16244dc4f15618d3067a7236d884dafe.jpg)

![5dca53e5981955dd93bb7606f7005c34a6e306ac7d58139d7fa86f8ea877ece2.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/5dca53e5981955dd93bb7606f7005c34a6e306ac7d58139d7fa86f8ea877ece2.jpg)

![631498f3cfc44d75d4013a0dfc0f2be4dafcfed060a367d910a9e595e4578816.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/631498f3cfc44d75d4013a0dfc0f2be4dafcfed060a367d910a9e595e4578816.jpg)

![79f920937058667a2e20f1b5841d0f1af95edf5d616b792d111459b15205dbcf.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/79f920937058667a2e20f1b5841d0f1af95edf5d616b792d111459b15205dbcf.jpg)

![8fbaf19edee466a909c9faa9d16021ca82af71dcbce0f3c80efda2c464092408.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/8fbaf19edee466a909c9faa9d16021ca82af71dcbce0f3c80efda2c464092408.jpg)

![927f5f8f3e6ecde101faaf3ec0cc2bdcf0cc14e72fc298675e2a07a7a6fd97c6.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/927f5f8f3e6ecde101faaf3ec0cc2bdcf0cc14e72fc298675e2a07a7a6fd97c6.jpg)

![9e1d362ff1402d6b8536cfe8f83112e83a3c32b1542e9e7bd233dc6910d54706.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/9e1d362ff1402d6b8536cfe8f83112e83a3c32b1542e9e7bd233dc6910d54706.jpg)

![b6fa683e2b51a206331bfc3397b834b10cefcd681ff02dfcab85c74c00f5828b.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/b6fa683e2b51a206331bfc3397b834b10cefcd681ff02dfcab85c74c00f5828b.jpg)

![c322f8490c9849c413df00b71a52cb091778562526cdea6c1da1a2a0df1b509e.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/c322f8490c9849c413df00b71a52cb091778562526cdea6c1da1a2a0df1b509e.jpg)

![fb1f4deb88da6ca3c66a535ea315c4ea655250f8a259d260b80a9d20bbb7602c.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/images/fb1f4deb88da6ca3c66a535ea315c4ea655250f8a259d260b80a9d20bbb7602c.jpg)

### Tables

![092ef6fcd824bc44c7ebf28e551fde9771abb5c572ae314f2f96ce3b720e6ad7.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/092ef6fcd824bc44c7ebf28e551fde9771abb5c572ae314f2f96ce3b720e6ad7.jpg)

![1e7e421f34e08ba3d8181e532a2e8dde14c27d3b30f270cb5622f0c3420e335d.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/1e7e421f34e08ba3d8181e532a2e8dde14c27d3b30f270cb5622f0c3420e335d.jpg)

![230671e29e926df56c1a00ebef2ce60f2930b07d725bc3215689f9d1d4f30e11.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/230671e29e926df56c1a00ebef2ce60f2930b07d725bc3215689f9d1d4f30e11.jpg)

![2cda6ed5e692aee961c1fd35f37ebd9fec0dbbe6c42d340329ff71e0482969c0.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/2cda6ed5e692aee961c1fd35f37ebd9fec0dbbe6c42d340329ff71e0482969c0.jpg)

![48499baeccd20d60d4d3a08fb887558fe519154470eafc3efc131d2c2fb5ad10.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/48499baeccd20d60d4d3a08fb887558fe519154470eafc3efc131d2c2fb5ad10.jpg)

![6f1c86aeaea0c9defc5437365b5becd77c2e332c19c4da8b979c2d4e70e9b92b.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/6f1c86aeaea0c9defc5437365b5becd77c2e332c19c4da8b979c2d4e70e9b92b.jpg)

![73adf5f7fb1782aedb640768a07cac7329cbc1d0ed1df572cea89934ab31e211.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/73adf5f7fb1782aedb640768a07cac7329cbc1d0ed1df572cea89934ab31e211.jpg)

![8163ade7f5d758d6cb5f66d0ac52aa27bb201d0805ad142e4800d2f385f3caff.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/8163ade7f5d758d6cb5f66d0ac52aa27bb201d0805ad142e4800d2f385f3caff.jpg)

![844e3569424a942337dd00266670cca5a26aad518845a0c20648996a9c8c3b6a.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/844e3569424a942337dd00266670cca5a26aad518845a0c20648996a9c8c3b6a.jpg)

![8497043fab09d908812ea4f27c0b10ededfe9b9f729c3e5ac8c1923955e5ade5.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/8497043fab09d908812ea4f27c0b10ededfe9b9f729c3e5ac8c1923955e5ade5.jpg)

![857cf68153ec085617ac8af184b8d5a6c484cd714439799c0566fd2bb964c586.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/857cf68153ec085617ac8af184b8d5a6c484cd714439799c0566fd2bb964c586.jpg)

![afb2d155620dee24331059c3e09539a347d2a0b61fd7d0bf19de3f72e80ea2dc.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/afb2d155620dee24331059c3e09539a347d2a0b61fd7d0bf19de3f72e80ea2dc.jpg)

![b1dad90decadfedf2c2c68a64a3992a001875154b5cbdeb7627aea2f1dc36eec.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/b1dad90decadfedf2c2c68a64a3992a001875154b5cbdeb7627aea2f1dc36eec.jpg)

![c244087f7e4ede922324028cb9e4f042564c44e91bd2e03b98510d8b51a9ad81.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/c244087f7e4ede922324028cb9e4f042564c44e91bd2e03b98510d8b51a9ad81.jpg)

![c2d0db086ddd94073f58a357cc942b906caffb6ab38287a32417f597cf7a26d1.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/c2d0db086ddd94073f58a357cc942b906caffb6ab38287a32417f597cf7a26d1.jpg)

![c53211ac7b9090997dab6426c44833efcba5dfde75d7c158d2024fba16c1e162.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/c53211ac7b9090997dab6426c44833efcba5dfde75d7c158d2024fba16c1e162.jpg)

![ed78301ed30159aca500f0b2f774f3a767f7a17b5f7e167b1856ebe5b1d4f4de.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/ed78301ed30159aca500f0b2f774f3a767f7a17b5f7e167b1856ebe5b1d4f4de.jpg)

![ef3c3fa681fb97c1bb430900d2cea5c15bd0817adacf2be5aa0f593216c02286.jpg](../icml_results/2594_Scaling%20Vision%20Transformers%20for%20Functional%20MRI%20with%20Flat%20Maps/tables/ef3c3fa681fb97c1bb430900d2cea5c15bd0817adacf2be5aa0f593216c02286.jpg)

<a id="paper-2563"></a>

## AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning


### Images

![075d7d94c83961180cbc09a49a10ef3e9709c4f619168353589fe5e31e4560f8.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/075d7d94c83961180cbc09a49a10ef3e9709c4f619168353589fe5e31e4560f8.jpg)

![0e4465e337bb44aa7b33a462dc85801bec89b97aabfca3b48fd63f113dbc57ff.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/0e4465e337bb44aa7b33a462dc85801bec89b97aabfca3b48fd63f113dbc57ff.jpg)

![119d92086b5c9b69f71689234a34dfd812e8488ced2f4d693feefc7e387ad8ea.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/119d92086b5c9b69f71689234a34dfd812e8488ced2f4d693feefc7e387ad8ea.jpg)

![53efc1c1726d371f2e17a65c5a8565c477d96a77399cc68f36c8832599e3fbd9.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/53efc1c1726d371f2e17a65c5a8565c477d96a77399cc68f36c8832599e3fbd9.jpg)

![6a709b6d6856870321f3ea293ead3a294cdfaf97d83c36935fd504fea9753f8d.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/6a709b6d6856870321f3ea293ead3a294cdfaf97d83c36935fd504fea9753f8d.jpg)

![78dd0889ea96a296dcb4328861b06e76b7ed530decb22d7b53dd1fcfa00fe52e.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/78dd0889ea96a296dcb4328861b06e76b7ed530decb22d7b53dd1fcfa00fe52e.jpg)

![7b4143524a1550156e4c2c02ec52c29156d73a980f7d97fe66b078df12a1ef59.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/7b4143524a1550156e4c2c02ec52c29156d73a980f7d97fe66b078df12a1ef59.jpg)

![91c289f58b521b65b47d662287a54ef95aeab014fbea6b803f0f36290ce7336a.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/91c289f58b521b65b47d662287a54ef95aeab014fbea6b803f0f36290ce7336a.jpg)

![9c011169e9943cae8a9ed37adbd2f5b9450449c6646e4f86f592b2235c610c15.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/9c011169e9943cae8a9ed37adbd2f5b9450449c6646e4f86f592b2235c610c15.jpg)

![b2df295f446ded1c3913abea4ef5d6563e74c63151a9db92c7f2926853dec8ae.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/b2df295f446ded1c3913abea4ef5d6563e74c63151a9db92c7f2926853dec8ae.jpg)

![c5aef1b3f4137f3bbd816e618cefa99adce57e66c311ab0d68c2615602ca78a9.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/images/c5aef1b3f4137f3bbd816e618cefa99adce57e66c311ab0d68c2615602ca78a9.jpg)

### Tables

![1d2c2c536bf8eea13eb292d3c51cb8dd2d4d99dbac268b852de5e4a8b23d3d18.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/1d2c2c536bf8eea13eb292d3c51cb8dd2d4d99dbac268b852de5e4a8b23d3d18.jpg)

![28404ec69f4203286aa39c7be8483802a42721c0dcc33d6b01b6e8047f151e25.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/28404ec69f4203286aa39c7be8483802a42721c0dcc33d6b01b6e8047f151e25.jpg)

![2b7dc8b576971bb5b5f528c9b6afff7113caddd28dd0c193a8f536448c5d1fa9.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/2b7dc8b576971bb5b5f528c9b6afff7113caddd28dd0c193a8f536448c5d1fa9.jpg)

![68f0db6ae8462c81cfc56b7bb44fd60ac4dc8d090a4baf33f9fbdb203ccaf92d.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/68f0db6ae8462c81cfc56b7bb44fd60ac4dc8d090a4baf33f9fbdb203ccaf92d.jpg)

![7f12198690649d5df19816e1079432ee051d4d11f23915a08f6698a4f67fa6c1.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/7f12198690649d5df19816e1079432ee051d4d11f23915a08f6698a4f67fa6c1.jpg)

![8f9e1e06893f245ea52dc0cf61f51cd1d0bb56de563aeb39aa8b5a8ee751aefa.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/8f9e1e06893f245ea52dc0cf61f51cd1d0bb56de563aeb39aa8b5a8ee751aefa.jpg)

![b5946fd47d724ffebcaee13c5ec9665a6af6bdcf856484686496ef68f96118fb.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/b5946fd47d724ffebcaee13c5ec9665a6af6bdcf856484686496ef68f96118fb.jpg)

![f8b3e6c5e7ab997c8fb08ddbc62159bf05c62c79e9f19001f3e0e555cc2b8636.jpg](../icml_results/2595_AOEPT_%20Breaking%20the%20Implicit%20Modality-Reduction%20Bottleneck%20in%20Modality-Missing%20Prompt%20Tuning/tables/f8b3e6c5e7ab997c8fb08ddbc62159bf05c62c79e9f19001f3e0e555cc2b8636.jpg)

<a id="paper-2564"></a>

## Functional Decomposition and Shapley Interactions for Interpreting Survival Models


### Images

![15c8a2aec19050aa1587ef947b64978e804280adc89218d89960e4eb56a5d257.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/15c8a2aec19050aa1587ef947b64978e804280adc89218d89960e4eb56a5d257.jpg)

![22339fc493b3fdf489d61b8a44910d389fa7d2b8ff0c93b58500306fcf54fa47.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/22339fc493b3fdf489d61b8a44910d389fa7d2b8ff0c93b58500306fcf54fa47.jpg)

![2e406d8c375e08973b70b41db168b3c1f144b85e557df1d1bb70a63970405e33.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/2e406d8c375e08973b70b41db168b3c1f144b85e557df1d1bb70a63970405e33.jpg)

![34ec4f16b25fcabf788cdb1cd8d48681266ef2708ab8f1301ac76176ec9a7729.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/34ec4f16b25fcabf788cdb1cd8d48681266ef2708ab8f1301ac76176ec9a7729.jpg)

![478cc1439ed1708fea90d35cefb621a22d4b46c0537478bb217b6ceae1f033ef.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/478cc1439ed1708fea90d35cefb621a22d4b46c0537478bb217b6ceae1f033ef.jpg)

![4a0fa82fccc9bd03b108563c4dca21dfe3664a700f903d377ade1d2585bdbf5c.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/4a0fa82fccc9bd03b108563c4dca21dfe3664a700f903d377ade1d2585bdbf5c.jpg)

![5f5d3797f32e850eeed2a94e5eb652f3840f3c9e7663dfcd52236a4dbdbf6d30.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/5f5d3797f32e850eeed2a94e5eb652f3840f3c9e7663dfcd52236a4dbdbf6d30.jpg)

![6f9f6b9badc1710dba9e6c15629f6a05651451c869ee0f59584f2e8cb749f6c4.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/6f9f6b9badc1710dba9e6c15629f6a05651451c869ee0f59584f2e8cb749f6c4.jpg)

![9a7733421ec41e14d11948c54650c3e52ea8b0f83c3da56cd5eb2ee5599195d4.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/9a7733421ec41e14d11948c54650c3e52ea8b0f83c3da56cd5eb2ee5599195d4.jpg)

![9e26f5681db78da9beefa859665d8811008bbd22def4c62cbcc85e736b6448ad.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/9e26f5681db78da9beefa859665d8811008bbd22def4c62cbcc85e736b6448ad.jpg)

![b7288daeb25f46410316135ac4c376f010c8a40472b204354a56ae9b1d272371.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/b7288daeb25f46410316135ac4c376f010c8a40472b204354a56ae9b1d272371.jpg)

![b9d44b028b7764c7977e31f6a4c5b8a87d5e7d3b1b4baaa256fd6646a58eef8f.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/b9d44b028b7764c7977e31f6a4c5b8a87d5e7d3b1b4baaa256fd6646a58eef8f.jpg)

![c0b74fa2c66d0133e306cb7d4f7ff7630c2f12a16698e097f17e6123d7b36146.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/c0b74fa2c66d0133e306cb7d4f7ff7630c2f12a16698e097f17e6123d7b36146.jpg)

![c8b45815bd6b5cc11c9eae6abd64a33bfb149ecfb03b5af8eef898b240e982ae.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/c8b45815bd6b5cc11c9eae6abd64a33bfb149ecfb03b5af8eef898b240e982ae.jpg)

![ca4c4ae900b0cf4ed530b29485dd36f9421ef9de3941148354e43c8dbea04e56.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/ca4c4ae900b0cf4ed530b29485dd36f9421ef9de3941148354e43c8dbea04e56.jpg)

![d13f37f54ec43039df00999601321d958fdc733e8dda5bdcb6ef7ae37a64ab7b.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/d13f37f54ec43039df00999601321d958fdc733e8dda5bdcb6ef7ae37a64ab7b.jpg)

![e5185cf41ab2e55e3870378728e0cbde56b8faff7df2b1d5fa95285d0dd7f207.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/e5185cf41ab2e55e3870378728e0cbde56b8faff7df2b1d5fa95285d0dd7f207.jpg)

![e5954a4ca90cfda7ea5f29a41eeca5862f8feb0386cc9b5b6fa3407662504064.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/e5954a4ca90cfda7ea5f29a41eeca5862f8feb0386cc9b5b6fa3407662504064.jpg)

![eb1d53b6306b67ce00456810a697e63ac29f68fffe9461ed959ffea497cacec4.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/eb1d53b6306b67ce00456810a697e63ac29f68fffe9461ed959ffea497cacec4.jpg)

![f97bbd13764c42483d2b92a9167be81fa7e3efbcfa1dee703d166308523c8ad0.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/images/f97bbd13764c42483d2b92a9167be81fa7e3efbcfa1dee703d166308523c8ad0.jpg)

### Tables

![9025a5c83098363cc6e5b6766abdcb8b2963d25beb1e5ff67cc125130f5e9bd4.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/9025a5c83098363cc6e5b6766abdcb8b2963d25beb1e5ff67cc125130f5e9bd4.jpg)

![a57b12cd08e2c3748f5fe2d9418af2e90217f17d785feb107c035fc1f2cd9247.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/a57b12cd08e2c3748f5fe2d9418af2e90217f17d785feb107c035fc1f2cd9247.jpg)

![ad0325d790ca963e3b441e6c7026b504cb553fea5e1dc632cda08d15f1f488c2.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/ad0325d790ca963e3b441e6c7026b504cb553fea5e1dc632cda08d15f1f488c2.jpg)

![c3319c133a4110ff36b15399f8f65f7016c24fa28cc19c34545377ee2bdb345f.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/c3319c133a4110ff36b15399f8f65f7016c24fa28cc19c34545377ee2bdb345f.jpg)

![e394e3edb5e95ec24fbc63d06cc9a751129f29c0b6a749ba009c9034b176ea2c.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/e394e3edb5e95ec24fbc63d06cc9a751129f29c0b6a749ba009c9034b176ea2c.jpg)

![f65dbbe9ee5ba146a360f11ffc934be0374f3112a9d21001683f1ce41e75692d.jpg](../icml_results/2596_Functional%20Decomposition%20and%20Shapley%20Interactions%20for%20Interpreting%20Survival%20Models/tables/f65dbbe9ee5ba146a360f11ffc934be0374f3112a9d21001683f1ce41e75692d.jpg)

<a id="paper-2565"></a>

## OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data


### Images

![00eddb0a6684d1171dcec1c4d01c0ce481414966df87231e7f8954cf4fc539b5.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/00eddb0a6684d1171dcec1c4d01c0ce481414966df87231e7f8954cf4fc539b5.jpg)

![02fbc7b14ee5fd2a336c7ada21fab888da844026bae0fd14bfcf5f4e63073a5b.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/02fbc7b14ee5fd2a336c7ada21fab888da844026bae0fd14bfcf5f4e63073a5b.jpg)

![039dec6706d99e8c867820f9c0b117487e79d67c968fb19a9185d5234941c8b0.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/039dec6706d99e8c867820f9c0b117487e79d67c968fb19a9185d5234941c8b0.jpg)

![0565a2c5cd3e9bc64b49aaa141d8630bbae012b202e476b8c1e343080e4f30c3.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/0565a2c5cd3e9bc64b49aaa141d8630bbae012b202e476b8c1e343080e4f30c3.jpg)

![0e49e1f02ef2a6190b60ed516983473a8fb96486954282134e8eac042d4ffb1f.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/0e49e1f02ef2a6190b60ed516983473a8fb96486954282134e8eac042d4ffb1f.jpg)

![17a20bb793f0eb0e40cdad4968ee5e1c4a11ad0dcd1fa8d311fd2c36978f0d9a.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/17a20bb793f0eb0e40cdad4968ee5e1c4a11ad0dcd1fa8d311fd2c36978f0d9a.jpg)

![17e0f13f50438683a32dd47e9018f2c6dcb8e733fa8d72ddd92a6a545fa83b13.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/17e0f13f50438683a32dd47e9018f2c6dcb8e733fa8d72ddd92a6a545fa83b13.jpg)

![28f7e3b9c7f7140c25f66bae9135430efb11c29c373446f6380d65da36cb4fe1.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/28f7e3b9c7f7140c25f66bae9135430efb11c29c373446f6380d65da36cb4fe1.jpg)

![41c79cc621df20803d3ec025c7ac677f52a1ebba9e751d0d4faf44513970d0db.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/41c79cc621df20803d3ec025c7ac677f52a1ebba9e751d0d4faf44513970d0db.jpg)

![4d7e210e09bf84ed17b79cebb336d99ad2b3ef483852224c16a360e6a9a1b6ad.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/4d7e210e09bf84ed17b79cebb336d99ad2b3ef483852224c16a360e6a9a1b6ad.jpg)

![53e80d48447578fbc5a860d70601eabd1c784ecc83382639bfb1101b6e662362.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/53e80d48447578fbc5a860d70601eabd1c784ecc83382639bfb1101b6e662362.jpg)

![5cd740ba27dc770f1fc89f133a37fa90f51dce161c88e6cc05bda73364a843be.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/5cd740ba27dc770f1fc89f133a37fa90f51dce161c88e6cc05bda73364a843be.jpg)

![5d1ad5b86be1f99e5beca8084259120ac6d3f15b1d6bb8f5b8d7e7647f70e88b.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/5d1ad5b86be1f99e5beca8084259120ac6d3f15b1d6bb8f5b8d7e7647f70e88b.jpg)

![6135c2f77d4651897ea5bb5ce016fdfe1a091d213bde3cb0f55991c2b92a9d78.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/6135c2f77d4651897ea5bb5ce016fdfe1a091d213bde3cb0f55991c2b92a9d78.jpg)

![6159760609720d46ba740417f541cbe713d5ceddbb3879409312ba3cea1eb1e3.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/6159760609720d46ba740417f541cbe713d5ceddbb3879409312ba3cea1eb1e3.jpg)

![6b3a644658b8a3e054759dff92d707d9a62c1fd601f1d5638b4ce0b4d9838ae9.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/6b3a644658b8a3e054759dff92d707d9a62c1fd601f1d5638b4ce0b4d9838ae9.jpg)

![7041a5059d39085121adc60dc27143c3525fc4bccdda462d0bf09bafdf27de9b.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/7041a5059d39085121adc60dc27143c3525fc4bccdda462d0bf09bafdf27de9b.jpg)

![70dfe9b8781953be9ec7dd3c156ebabb47e2eea28fd0161347f59193f1a77e0b.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/70dfe9b8781953be9ec7dd3c156ebabb47e2eea28fd0161347f59193f1a77e0b.jpg)

![809f412bd493126bad2bc963d87836132fd55d2944782d28542c87b32c89af9a.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/809f412bd493126bad2bc963d87836132fd55d2944782d28542c87b32c89af9a.jpg)

![83407860e7fd8216290a5951ba6a191f1d0c2ec41ff2ff47946d88fee7a1d416.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/83407860e7fd8216290a5951ba6a191f1d0c2ec41ff2ff47946d88fee7a1d416.jpg)

![866ca7b3a7f15172c744f6f70a34b13409aea72092f2000bf3ba65a5065bbd65.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/866ca7b3a7f15172c744f6f70a34b13409aea72092f2000bf3ba65a5065bbd65.jpg)

![8c2506159acd1df7f74ff701a7af5bb8fb3e82f0eed79d0d5e58cc8fd7cd016c.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/8c2506159acd1df7f74ff701a7af5bb8fb3e82f0eed79d0d5e58cc8fd7cd016c.jpg)

![919d0a71855a69a98ece53af6338aadb313274f29ec633b1b26cfc148dfb7647.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/919d0a71855a69a98ece53af6338aadb313274f29ec633b1b26cfc148dfb7647.jpg)

![935b495b3258bf81d5a1bf5be007ade8e5a51bf5ef9f4c4338c858fd5dbe703c.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/935b495b3258bf81d5a1bf5be007ade8e5a51bf5ef9f4c4338c858fd5dbe703c.jpg)

![9767bd61568bee0dbf181a3d42c633ad42ed97dab39db12ac662916542609954.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/9767bd61568bee0dbf181a3d42c633ad42ed97dab39db12ac662916542609954.jpg)

![97bd1db8080f9c3089bab0f6e24d9644d209f365abc60fc168da49b36c0d2f07.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/97bd1db8080f9c3089bab0f6e24d9644d209f365abc60fc168da49b36c0d2f07.jpg)

![9d74d759514870122a4b3c15087adbbd8754c9544707eb35d2ade4dd49cdf7f6.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/9d74d759514870122a4b3c15087adbbd8754c9544707eb35d2ade4dd49cdf7f6.jpg)

![9eb2c3268d65eaac4b69a725b0502bb50e14351a9ff2b86c81e7c2d68980256e.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/9eb2c3268d65eaac4b69a725b0502bb50e14351a9ff2b86c81e7c2d68980256e.jpg)

![ab0bfd26d230ad6f39f7555bc09c9f0f7d463a1964746b2a92e5f64d4ae3187c.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/ab0bfd26d230ad6f39f7555bc09c9f0f7d463a1964746b2a92e5f64d4ae3187c.jpg)

![ae09b7b2db9fa13b5527ca1a2903fb019175e1b8cf8a6d24820708271db6eabd.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/ae09b7b2db9fa13b5527ca1a2903fb019175e1b8cf8a6d24820708271db6eabd.jpg)

![afe338e47b37e1bc9ecc11b4047dc67223f17de788c58f7d45d7eb8367947eac.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/afe338e47b37e1bc9ecc11b4047dc67223f17de788c58f7d45d7eb8367947eac.jpg)

![b0e141b832562b2ca3c1af48ac5dfca6f328636148ca4fb2bb3ef50d1aa649c1.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/b0e141b832562b2ca3c1af48ac5dfca6f328636148ca4fb2bb3ef50d1aa649c1.jpg)

![be5a22b2ac71960283babcf020b5a9930a78421f0d13e8f856e73e41e6ddc22f.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/be5a22b2ac71960283babcf020b5a9930a78421f0d13e8f856e73e41e6ddc22f.jpg)

![c426f1259ccead4aad6f9f94cffb202adb1230babc8bd109606a4c9bd22526a2.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/c426f1259ccead4aad6f9f94cffb202adb1230babc8bd109606a4c9bd22526a2.jpg)

![d1d6754c1db10e2119e65690076bd8c9c9e9de1e30345bfd726378a6425c85e9.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/d1d6754c1db10e2119e65690076bd8c9c9e9de1e30345bfd726378a6425c85e9.jpg)

![d442e9b5375c641566a6ad0ae3ef4ade22c58710a8b441c41ee1b676787e9f94.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/d442e9b5375c641566a6ad0ae3ef4ade22c58710a8b441c41ee1b676787e9f94.jpg)

![d8539800b7006e026d587b7df19addbd9cdcf46585b8b5bb23c2e6a74bee67a6.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/d8539800b7006e026d587b7df19addbd9cdcf46585b8b5bb23c2e6a74bee67a6.jpg)

![d8ee4125da9bc07754bb615d8eff312e262883c39934d5dafbaf03197c5dd661.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/d8ee4125da9bc07754bb615d8eff312e262883c39934d5dafbaf03197c5dd661.jpg)

![dd52114feac3ece9cbe4c5b43e9cc1847350c636a78a64a48b3d13c7bae396a2.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/dd52114feac3ece9cbe4c5b43e9cc1847350c636a78a64a48b3d13c7bae396a2.jpg)

![e2e07291c0af3412d6f5030afb1f790dca672292a88794032e8ba5599d99aeed.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/e2e07291c0af3412d6f5030afb1f790dca672292a88794032e8ba5599d99aeed.jpg)

![ee2efc3e989f1f5e85cf462d68e12a8f30fd680e568ee92a348b6d39bf3178bd.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/ee2efc3e989f1f5e85cf462d68e12a8f30fd680e568ee92a348b6d39bf3178bd.jpg)

![f3439177b5d7c742a8c6787d725393919ea0b5708173afd1110c44b4f21a1044.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/images/f3439177b5d7c742a8c6787d725393919ea0b5708173afd1110c44b4f21a1044.jpg)

### Tables

![0f74663d0e633a347fa0763cbe4c52a5043350ad5ff1323c8985fad0f1a64feb.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/0f74663d0e633a347fa0763cbe4c52a5043350ad5ff1323c8985fad0f1a64feb.jpg)

![636b0df18a753670d5793e832ca49f3db4cbb6a0cc224be783d190e2e36d853e.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/636b0df18a753670d5793e832ca49f3db4cbb6a0cc224be783d190e2e36d853e.jpg)

![6f6e7946a186dbe942a03df2fc6d574fc5605dc2621d02f8a8de2599268dbeaa.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/6f6e7946a186dbe942a03df2fc6d574fc5605dc2621d02f8a8de2599268dbeaa.jpg)

![7d92d522a8475896a73f891d893001c4fce0d6d6a5a64c2ed87a2113f5e3d629.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/7d92d522a8475896a73f891d893001c4fce0d6d6a5a64c2ed87a2113f5e3d629.jpg)

![808306b3b9b88a30b32dd75fd3baec3ca7fc8e75f98c210673be3b66e7f588db.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/808306b3b9b88a30b32dd75fd3baec3ca7fc8e75f98c210673be3b66e7f588db.jpg)

![98551ddee10851c92fa84e6f07323f65bebd1a2ba316c6df6e92d35bc00cde82.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/98551ddee10851c92fa84e6f07323f65bebd1a2ba316c6df6e92d35bc00cde82.jpg)

![ab101baac292a26d400d6b2ab6bc9c1cbcc52a1be239b9c13da612320df6f658.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/ab101baac292a26d400d6b2ab6bc9c1cbcc52a1be239b9c13da612320df6f658.jpg)

![c89e01a006a1dfc4cd79b920430d556116b9fe6f8bb32f891ecdb0b8a3e666c2.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/c89e01a006a1dfc4cd79b920430d556116b9fe6f8bb32f891ecdb0b8a3e666c2.jpg)

![cb7540e8e5d6743edc20b6690e17f1a0b3404ff86eb983897f0d8aec87ed6531.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/cb7540e8e5d6743edc20b6690e17f1a0b3404ff86eb983897f0d8aec87ed6531.jpg)

![de8472ece86067bf836523dd3dfe6f250829f5396549a52a0535206b8fd2d2d8.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/de8472ece86067bf836523dd3dfe6f250829f5396549a52a0535206b8fd2d2d8.jpg)

![e1bf19ec39535709bf53fd4f255211582edf9b4af9cabed6a1084542f8b54d5e.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/e1bf19ec39535709bf53fd4f255211582edf9b4af9cabed6a1084542f8b54d5e.jpg)

![e2bce5814f47576d3ff1c6b923cb8cbdb32d37aa5802ae82551295c92c2dbf1d.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/e2bce5814f47576d3ff1c6b923cb8cbdb32d37aa5802ae82551295c92c2dbf1d.jpg)

![ed16c3f0dc63f54f4abf64c5993379cdb91d9cf873ba65137b89369380eb30b9.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/ed16c3f0dc63f54f4abf64c5993379cdb91d9cf873ba65137b89369380eb30b9.jpg)

![f0ea9c8e69adcc7c3553b6858483a3f9bd389a593b84850eea53c99f842a9945.jpg](../icml_results/2597_OpenTSLM_%20Time-Series%20Language%20Models%20for%20Reasoning%20over%20Multivariate%20Medical%20Text-%20and%20Time-Series%20/tables/f0ea9c8e69adcc7c3553b6858483a3f9bd389a593b84850eea53c99f842a9945.jpg)

<a id="paper-2566"></a>

## Nested birth-death processes are competitive with neural networks as time-dependent models of protein evolution


### Images

![02a9b3a14ddcf8275e111d412aea0c199abf05e840d5eb8ae3a18499a9a05f24.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/images/02a9b3a14ddcf8275e111d412aea0c199abf05e840d5eb8ae3a18499a9a05f24.jpg)

### Tables

![02fb63f06a3143a3b4bfff4d2a9193c3e465b3675eac701d3f6ca8445b2208e5.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/02fb63f06a3143a3b4bfff4d2a9193c3e465b3675eac701d3f6ca8445b2208e5.jpg)

![1729ab5bd7b0cbde6f31dbd8fdb16b6190ee8d02b5f54e841a4d3a298e4b8f42.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/1729ab5bd7b0cbde6f31dbd8fdb16b6190ee8d02b5f54e841a4d3a298e4b8f42.jpg)

![280330e920d82c36160dbba8e8faec79b0751475447cdacb0ee7054d73ae55ab.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/280330e920d82c36160dbba8e8faec79b0751475447cdacb0ee7054d73ae55ab.jpg)

![28f491c8cd7e920fa8d16f4c920c57568cc22d75e5906fb98e249f94740c493d.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/28f491c8cd7e920fa8d16f4c920c57568cc22d75e5906fb98e249f94740c493d.jpg)

![2f6a949cf64f4688d1b92cbba157878eef53b5aaeea9489d038a92491b739a55.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/2f6a949cf64f4688d1b92cbba157878eef53b5aaeea9489d038a92491b739a55.jpg)

![3e49ff9c8f563e9138f60a438a77c546caba5d73ede7b5dc0a357bb7aea783d8.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/3e49ff9c8f563e9138f60a438a77c546caba5d73ede7b5dc0a357bb7aea783d8.jpg)

![4dda6c1ef420b4cac7e37817db07a7b6d23c817a55b4cf65574ea5abe15a2199.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/4dda6c1ef420b4cac7e37817db07a7b6d23c817a55b4cf65574ea5abe15a2199.jpg)

![533058c5c4e23363bf376ef24fca029bbc90705601fd2a32daa36e6668f193fa.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/533058c5c4e23363bf376ef24fca029bbc90705601fd2a32daa36e6668f193fa.jpg)

![54e08edfa6692efb671c92437073103f0c7ae94373826eea7870cac4970535b9.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/54e08edfa6692efb671c92437073103f0c7ae94373826eea7870cac4970535b9.jpg)

![6b6fbc1188ee8f4a9dcfd96a6be5d99aa58c8fcf052a2ecd9df773fcb6cd4e60.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/6b6fbc1188ee8f4a9dcfd96a6be5d99aa58c8fcf052a2ecd9df773fcb6cd4e60.jpg)

![70eb51358c6e53cd7591158aa73172cfd39e1c0abfb8ce1d8370872c1b7eb532.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/70eb51358c6e53cd7591158aa73172cfd39e1c0abfb8ce1d8370872c1b7eb532.jpg)

![74feda0ff6ff6b42d88565f70efcd0a2c1c5d938a48f0c8f40b89c9f61c77e9a.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/74feda0ff6ff6b42d88565f70efcd0a2c1c5d938a48f0c8f40b89c9f61c77e9a.jpg)

![780cbd910001d1e21f78f62766a63c64b957e56735c18204b80eec6128ebbeb3.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/780cbd910001d1e21f78f62766a63c64b957e56735c18204b80eec6128ebbeb3.jpg)

![7f2e869fc60da94707985d659542fbc19c973baf7d1105179257e54a37af653c.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/7f2e869fc60da94707985d659542fbc19c973baf7d1105179257e54a37af653c.jpg)

![980f9e3170111a3aae3033f70e93fee34b60ac303982e213f985e844d62f651a.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/980f9e3170111a3aae3033f70e93fee34b60ac303982e213f985e844d62f651a.jpg)

![afcc62d707d4783c63988985ffcc71c6f149e0f810751278cd73eadc40bc643e.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/afcc62d707d4783c63988985ffcc71c6f149e0f810751278cd73eadc40bc643e.jpg)

![c109578a10268f87c9ea408c1a807964c381123ed8870f7beb9e0ebc04debf54.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/c109578a10268f87c9ea408c1a807964c381123ed8870f7beb9e0ebc04debf54.jpg)

![d40350fdc37bae0af1533cbb104743bb4502cbb93de48c33620c58b61cdcacc9.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/d40350fdc37bae0af1533cbb104743bb4502cbb93de48c33620c58b61cdcacc9.jpg)

![d63741cabf08fb4e683e4ee6348b47b838f424f0c16440338fe2a4b8cafcf861.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/d63741cabf08fb4e683e4ee6348b47b838f424f0c16440338fe2a4b8cafcf861.jpg)

![daa335acc1de6ef4003d780ac842ac380190491f2e93e3a9672da67c93e918dc.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/daa335acc1de6ef4003d780ac842ac380190491f2e93e3a9672da67c93e918dc.jpg)

![dc04b26278a33fd49cce8f0a559495fb5e1e5b1601a3dd97a6bb0f64ddfe8d9b.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/dc04b26278a33fd49cce8f0a559495fb5e1e5b1601a3dd97a6bb0f64ddfe8d9b.jpg)

![ea394cde495a7958b57ed72a1cdae8cb61b5fd32bb052412854664bf334c915f.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/ea394cde495a7958b57ed72a1cdae8cb61b5fd32bb052412854664bf334c915f.jpg)

![ef190822d71b93fd3b3cf377a308a1bc0db2e86da1317b453dd421afb71903c6.jpg](../icml_results/2598_Nested%20birth-death%20processes%20are%20competitive%20with%20neural%20networks%20as%20time-dependent%20models%20of%20protei/tables/ef190822d71b93fd3b3cf377a308a1bc0db2e86da1317b453dd421afb71903c6.jpg)

<a id="paper-2567"></a>

## Learning High-Dimensional Parity Functions with Product Networks using Gradient Descent


### Images

![003de7f621361341271fe05eb41c249257bf67b9b08363db43f0b6f7cd869e77.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/003de7f621361341271fe05eb41c249257bf67b9b08363db43f0b6f7cd869e77.jpg)

![07364ff1f015466d14f2e7d70f10944b690822fd018f27bba00b3d1395e0db4b.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/07364ff1f015466d14f2e7d70f10944b690822fd018f27bba00b3d1395e0db4b.jpg)

![0aa8e731ef2e72f18fd28872657e62ebae5e90b43020d334cf6215bb10542b9c.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/0aa8e731ef2e72f18fd28872657e62ebae5e90b43020d334cf6215bb10542b9c.jpg)

![0c88f0e0bfe6863089d600270b79978b9e2a8e6e3a253fcb1e2f0149f71eabc1.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/0c88f0e0bfe6863089d600270b79978b9e2a8e6e3a253fcb1e2f0149f71eabc1.jpg)

![0e977aa4e5d4a8e6cc4490f29f717cff14a219b6f445e1cbfbf6ebdbb01b57ca.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/0e977aa4e5d4a8e6cc4490f29f717cff14a219b6f445e1cbfbf6ebdbb01b57ca.jpg)

![21218015a77e3889a3f77da5e9d81b21f2d549503d49cbb6ef58775a9e4661e7.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/21218015a77e3889a3f77da5e9d81b21f2d549503d49cbb6ef58775a9e4661e7.jpg)

![31cb77bfc92db0fabb5436accad3cb5a35ba80848d292624e278239802206e81.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/31cb77bfc92db0fabb5436accad3cb5a35ba80848d292624e278239802206e81.jpg)

![33e57d10a2788c1608d3bf982270dfa93b6a5093ae3f56bfc0bd34ade43562ba.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/33e57d10a2788c1608d3bf982270dfa93b6a5093ae3f56bfc0bd34ade43562ba.jpg)

![39036ea4afa32153dbfaad2ce93419804ffabe53d4d013f7b481952ac125b959.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/39036ea4afa32153dbfaad2ce93419804ffabe53d4d013f7b481952ac125b959.jpg)

![3d2d41627fd27551ee134a44856d86cd965dcbab8f995deb41932bbcf05d9dc3.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/3d2d41627fd27551ee134a44856d86cd965dcbab8f995deb41932bbcf05d9dc3.jpg)

![6dee00d0fbf9d70c086226b24350612696c957e9770eb0346d79ab68ede82234.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/6dee00d0fbf9d70c086226b24350612696c957e9770eb0346d79ab68ede82234.jpg)

![720ea88bb0d6c340a386be731a164342748bbce0cd98ab10c14788b97f71115f.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/720ea88bb0d6c340a386be731a164342748bbce0cd98ab10c14788b97f71115f.jpg)

![75102d322e3651832d17c65584d52ad9e8bcc4747134f4e191c6368b8c39beb7.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/75102d322e3651832d17c65584d52ad9e8bcc4747134f4e191c6368b8c39beb7.jpg)

![7b0e63b3ae2a1c3c43d2b5b2e9655782cbfa897c4a5b5ed50eea06b7a7b93108.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/7b0e63b3ae2a1c3c43d2b5b2e9655782cbfa897c4a5b5ed50eea06b7a7b93108.jpg)

![7e6e916385da2a7216328e6d4f6f821c7bafe8f0c7af98c11203dadceeb03f4f.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/7e6e916385da2a7216328e6d4f6f821c7bafe8f0c7af98c11203dadceeb03f4f.jpg)

![8157ffa557305a3592a1ddb514936cc8d4bc12a1f79640a48d764840b6fa5a10.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/8157ffa557305a3592a1ddb514936cc8d4bc12a1f79640a48d764840b6fa5a10.jpg)

![8327c74d7928b5862baf26c63c2ccc3df7cceb6749f2cdba8d60b44980d6f5ae.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/8327c74d7928b5862baf26c63c2ccc3df7cceb6749f2cdba8d60b44980d6f5ae.jpg)

![901742839ae17e76f57e5ca0ad2c4d786a39fa1ef9ef6f7a14d6fd7d2b2947e9.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/901742839ae17e76f57e5ca0ad2c4d786a39fa1ef9ef6f7a14d6fd7d2b2947e9.jpg)

![ae726c7161471e85ad173e5d8b3fa7e0871aa830ca3c86c0335fa4970b357f0c.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/ae726c7161471e85ad173e5d8b3fa7e0871aa830ca3c86c0335fa4970b357f0c.jpg)

![b643bac07d5ac8293b88791dc1a8e12a1ef24896b770dc661ac4cbdaed227d36.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/b643bac07d5ac8293b88791dc1a8e12a1ef24896b770dc661ac4cbdaed227d36.jpg)

![b669d564d0febd9486c033c621e394f95e3911f45c63afe2fb0179503f4b0202.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/b669d564d0febd9486c033c621e394f95e3911f45c63afe2fb0179503f4b0202.jpg)

![c303f085b176922c4e43c082c7ba79b0f4cbbae841b41d059a41babcdb3bb992.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/c303f085b176922c4e43c082c7ba79b0f4cbbae841b41d059a41babcdb3bb992.jpg)

![c9c4ebecdb92b821325bfe452b3aa17d9fbf601667cc74b69bdc498c44ef4915.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/c9c4ebecdb92b821325bfe452b3aa17d9fbf601667cc74b69bdc498c44ef4915.jpg)

![d4cc47d394cfeaa9a45f3cc9e87e0582ab3aacab59471012bf5f1b306900bb50.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/d4cc47d394cfeaa9a45f3cc9e87e0582ab3aacab59471012bf5f1b306900bb50.jpg)

![e020d776ec078706839f2547583df94d5c2a13165f7f9f9efb181c5941a21e72.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/e020d776ec078706839f2547583df94d5c2a13165f7f9f9efb181c5941a21e72.jpg)

![e085b18cda3a79844122224e4beab7c7f5c7fc1d6f7c1548c8630556952a44b5.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/e085b18cda3a79844122224e4beab7c7f5c7fc1d6f7c1548c8630556952a44b5.jpg)

![e29317c7055d087443b6e7e69b4e393cb8830df3563d55539ef3f76e92246971.jpg](../icml_results/2600_Learning%20High-Dimensional%20Parity%20Functions%20with%20Product%20Networks%20using%20Gradient%20Descent/images/e29317c7055d087443b6e7e69b4e393cb8830df3563d55539ef3f76e92246971.jpg)

<a id="paper-2568"></a>

## Translation Heads: Disentangling meaning from language in LLM-based machine translation


### Images

![0320426ba9dbab57f6017879c7dc9a2859450eb05ff8e661b18e8d931b7f878d.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/0320426ba9dbab57f6017879c7dc9a2859450eb05ff8e661b18e8d931b7f878d.jpg)

![033a37416cf1d6a1d9f3b8c6b703d718e74e13f7303d3f98b56e8724032bce99.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/033a37416cf1d6a1d9f3b8c6b703d718e74e13f7303d3f98b56e8724032bce99.jpg)

![05461c1d60a7cfe1c9f9017580aa553eef5509b406f777a8e4d59d43e4211cfd.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/05461c1d60a7cfe1c9f9017580aa553eef5509b406f777a8e4d59d43e4211cfd.jpg)

![0634e73a50d8588bae5f4bce234e572b756a4b9af6433cf1e2ed104fdcd7e1fa.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/0634e73a50d8588bae5f4bce234e572b756a4b9af6433cf1e2ed104fdcd7e1fa.jpg)

![09eddb4934dbce71bb9abe129d8a77a3cd14c84f0beaa9207ae4d212834e8f9e.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/09eddb4934dbce71bb9abe129d8a77a3cd14c84f0beaa9207ae4d212834e8f9e.jpg)

![0ea7cfb59046903019079d0084feade83c75275e81d6a12837198a25c5d98dbd.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/0ea7cfb59046903019079d0084feade83c75275e81d6a12837198a25c5d98dbd.jpg)

![121c7cd2d52dd7ca9947c9b339428551df3a117c26079917dbf24ed913788ff2.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/121c7cd2d52dd7ca9947c9b339428551df3a117c26079917dbf24ed913788ff2.jpg)

![1572d92269d8c0e1f04145313e818ca20222c73af76b2e1d57f6508de18302c2.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/1572d92269d8c0e1f04145313e818ca20222c73af76b2e1d57f6508de18302c2.jpg)

![1760cfdde6e05735d658e1dae1d47dfaa8d6a7f75480425e750ebcfb8395e16f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/1760cfdde6e05735d658e1dae1d47dfaa8d6a7f75480425e750ebcfb8395e16f.jpg)

![182d7556a8b1348f2d176daa5f383d5da691f4b1fd358fac4f0240e8d05e25f5.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/182d7556a8b1348f2d176daa5f383d5da691f4b1fd358fac4f0240e8d05e25f5.jpg)

![197c10f64f0d8186550320d12f078de8c64aa863011b5891340da3e623b8a1b0.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/197c10f64f0d8186550320d12f078de8c64aa863011b5891340da3e623b8a1b0.jpg)

![1c51747f8cc969bd26785829502e08460672695211cf7855f165d95f167167c3.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/1c51747f8cc969bd26785829502e08460672695211cf7855f165d95f167167c3.jpg)

![1ff3dcc5a6216ff58927a5c1d0177ae8890334b7d6d66a5c8c9a196c2fceb891.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/1ff3dcc5a6216ff58927a5c1d0177ae8890334b7d6d66a5c8c9a196c2fceb891.jpg)

![24216b02920a8f064b89bf323f9f64ff62dc1020b5cd6f82c3b87a43542a1d51.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/24216b02920a8f064b89bf323f9f64ff62dc1020b5cd6f82c3b87a43542a1d51.jpg)

![3119889d306e2720c8c99dd0e46ce2e60b163706a72dee20c84a8af7008625aa.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/3119889d306e2720c8c99dd0e46ce2e60b163706a72dee20c84a8af7008625aa.jpg)

![32161a0b627a86e6d6cfc194efa3af66e32926da0f87cb041450cca0a74439a3.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/32161a0b627a86e6d6cfc194efa3af66e32926da0f87cb041450cca0a74439a3.jpg)

![351ed2c548781d8b81f1ba809cf8c69ae3c61ee2e35f2f57f0ae9dbb21276295.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/351ed2c548781d8b81f1ba809cf8c69ae3c61ee2e35f2f57f0ae9dbb21276295.jpg)

![381c22b8f2ffbee297ff0d9296d6be2d84e6de96c252b65022a2f3d685a1cce9.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/381c22b8f2ffbee297ff0d9296d6be2d84e6de96c252b65022a2f3d685a1cce9.jpg)

![462e1ac5efec03bd283dd54e5f94a8a46474252e03cb84f17d1844a5e8514cd2.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/462e1ac5efec03bd283dd54e5f94a8a46474252e03cb84f17d1844a5e8514cd2.jpg)

![46b5d99002c7804cc4e2f220fb85a2c4b5dd189467fd033f762be64705a8b071.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/46b5d99002c7804cc4e2f220fb85a2c4b5dd189467fd033f762be64705a8b071.jpg)

![4c4defd3f09baf72f69d41cb56f21e66cc0fed850a1669de289c50612af19b1d.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/4c4defd3f09baf72f69d41cb56f21e66cc0fed850a1669de289c50612af19b1d.jpg)

![53e7a14b7b2b17eaf4a9ac69080a1f9798a776c1efb06ba5692c88176ed8d2ee.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/53e7a14b7b2b17eaf4a9ac69080a1f9798a776c1efb06ba5692c88176ed8d2ee.jpg)

![53f1543ce9b6298b831d5b90508cf6eb943164b6bb3982a692a6216e67ac633f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/53f1543ce9b6298b831d5b90508cf6eb943164b6bb3982a692a6216e67ac633f.jpg)

![5750c699a6f96796f8c14ee7441167c48cece81d75ba2a432514deadb3621f3b.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5750c699a6f96796f8c14ee7441167c48cece81d75ba2a432514deadb3621f3b.jpg)

![5967e873412488a37b0868a1bce4fc66a490e4c86ee6138de200830f86803262.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5967e873412488a37b0868a1bce4fc66a490e4c86ee6138de200830f86803262.jpg)

![5b101f643a6e7be32380f0204a6f427903376237dca74e8cdef60e9ab0f3af00.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5b101f643a6e7be32380f0204a6f427903376237dca74e8cdef60e9ab0f3af00.jpg)

![5dec778a544dac5157ad230f9bfeef4cf0351ec20621ca545960f05e0baf6e89.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5dec778a544dac5157ad230f9bfeef4cf0351ec20621ca545960f05e0baf6e89.jpg)

![5fa86efd2aa3182fdd50b0cc3105f313e6e2ace9ab065011d920557ea4a91d6f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5fa86efd2aa3182fdd50b0cc3105f313e6e2ace9ab065011d920557ea4a91d6f.jpg)

![5fdb0502228881a9ec3a8586056338e275fcf901937d93b9dd5136689a2d50af.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/5fdb0502228881a9ec3a8586056338e275fcf901937d93b9dd5136689a2d50af.jpg)

![635be166349f0e224e0a2160990dc0a5bf1869809735c5c339f37c8aa177126e.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/635be166349f0e224e0a2160990dc0a5bf1869809735c5c339f37c8aa177126e.jpg)

![641762438d6d125a980878cc1acfd4a2ea22a58161b00290cb9ba9a0b69c39eb.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/641762438d6d125a980878cc1acfd4a2ea22a58161b00290cb9ba9a0b69c39eb.jpg)

![68b01c53edc250030f8c47cdd5dda9cb7b5f14f3cc5a338df3019f554258130f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/68b01c53edc250030f8c47cdd5dda9cb7b5f14f3cc5a338df3019f554258130f.jpg)

![6d3f6bc3ed7889b5db24a47913a266db9ec5d50518d4edf79a29a324cead04e5.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/6d3f6bc3ed7889b5db24a47913a266db9ec5d50518d4edf79a29a324cead04e5.jpg)

![7a9555a1cfd97fa86f92553d12d7606438126db22da665b5dc7266c471607987.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/7a9555a1cfd97fa86f92553d12d7606438126db22da665b5dc7266c471607987.jpg)

![81a3279385ad6c66ef0475f6f46f4998670504d15d4ac8025a48bc215d1af147.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/81a3279385ad6c66ef0475f6f46f4998670504d15d4ac8025a48bc215d1af147.jpg)

![81c57721f25ba14755465c7b06ada4910923ad3d9d55281996075250fa00fbe9.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/81c57721f25ba14755465c7b06ada4910923ad3d9d55281996075250fa00fbe9.jpg)

![83c78eb901dee75fd9cea2cd06ec15eb10b0b187db790c5e54c008dd72a0dc35.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/83c78eb901dee75fd9cea2cd06ec15eb10b0b187db790c5e54c008dd72a0dc35.jpg)

![8707db849c887c922359ec81509ec5478b2ef0dd9fa5758614ab812163d0350a.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/8707db849c887c922359ec81509ec5478b2ef0dd9fa5758614ab812163d0350a.jpg)

![885e89f7fdcdc89dd2b54bca5e39552e600d5bec52a3b70af680a744ed3286f9.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/885e89f7fdcdc89dd2b54bca5e39552e600d5bec52a3b70af680a744ed3286f9.jpg)

![9108f185436f462731d718481e64e6b6cbd2a00de84ec3e7c7a902b914ec0240.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/9108f185436f462731d718481e64e6b6cbd2a00de84ec3e7c7a902b914ec0240.jpg)

![95210dc119d07e53d4df3dc6d7a740a362cc66a3be10bfa747f0d4b51a833902.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/95210dc119d07e53d4df3dc6d7a740a362cc66a3be10bfa747f0d4b51a833902.jpg)

![95d93e8dddf3f9d0f5c3f51ed9a86ab2ad1f3f191e7ea636b35ccd259c749218.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/95d93e8dddf3f9d0f5c3f51ed9a86ab2ad1f3f191e7ea636b35ccd259c749218.jpg)

![9a9e55522cb901ec60cabe38cccf3184b710a0889c9b8d716531f4af00eccbea.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/9a9e55522cb901ec60cabe38cccf3184b710a0889c9b8d716531f4af00eccbea.jpg)

![9e924cd60e7778565f418855844ce277bcbcf1e88bb6b92b162ccd535bda558f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/9e924cd60e7778565f418855844ce277bcbcf1e88bb6b92b162ccd535bda558f.jpg)

![9f3012b1aba9daccd54385a4494f82ddc08c2bc76639703d7d5ef0169101e43c.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/9f3012b1aba9daccd54385a4494f82ddc08c2bc76639703d7d5ef0169101e43c.jpg)

![a30f0f55f113adce226357424369e89325695500a0368d41871f3f78069a925d.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/a30f0f55f113adce226357424369e89325695500a0368d41871f3f78069a925d.jpg)

![a5ff3d76924865efa21ea06fde815eed140a6cd4511401f95d6929dc5e722108.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/a5ff3d76924865efa21ea06fde815eed140a6cd4511401f95d6929dc5e722108.jpg)

![aadfc63d6dd346449a661b1bbd72abaf3027fe5e72f6bb8f17d4fcec1ba6c9aa.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/aadfc63d6dd346449a661b1bbd72abaf3027fe5e72f6bb8f17d4fcec1ba6c9aa.jpg)

![aaf8a9a7985de24b0101d0b44be3a91e004795ef4c3290c93ec6951e17cde193.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/aaf8a9a7985de24b0101d0b44be3a91e004795ef4c3290c93ec6951e17cde193.jpg)

![ab6c68d4262491d239af8806a92fb5d95b6385b69d69646b829f9216d97971bd.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/ab6c68d4262491d239af8806a92fb5d95b6385b69d69646b829f9216d97971bd.jpg)

![b0b15f1f58900f33c785bd5fe7169085a2b627fa449b191d879c0686c7adefaf.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/b0b15f1f58900f33c785bd5fe7169085a2b627fa449b191d879c0686c7adefaf.jpg)

![b28a30b00c60fe2cd7176b5ba9e1a8fe44284113c10f64cf0a73cf21eb636aad.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/b28a30b00c60fe2cd7176b5ba9e1a8fe44284113c10f64cf0a73cf21eb636aad.jpg)

![b683e3f119b44205707f7a250cdd63f7d94559b39751e9535201f41f3984e0de.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/b683e3f119b44205707f7a250cdd63f7d94559b39751e9535201f41f3984e0de.jpg)

![b6a0a36053acddea50defb3ef11c2db5625a41f7700e3e349bf89f204e64cfa6.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/b6a0a36053acddea50defb3ef11c2db5625a41f7700e3e349bf89f204e64cfa6.jpg)

![bdc89e6c4879d33672034d1112c642801598d1483452cb2a19f3b7da9f9cf19e.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/bdc89e6c4879d33672034d1112c642801598d1483452cb2a19f3b7da9f9cf19e.jpg)

![bf9e94d005c1f24b852d107ff3c60fce57538276b28c035a3dcf1597d7aafebb.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/bf9e94d005c1f24b852d107ff3c60fce57538276b28c035a3dcf1597d7aafebb.jpg)

![c0c3a5b5af13fadecb96d30fca4d879320cc96681da7d8245b769f8a1a9056cc.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/c0c3a5b5af13fadecb96d30fca4d879320cc96681da7d8245b769f8a1a9056cc.jpg)

![c0c599d4ee5c1a378c6b7d05cff64b93f312d2089f0977f98203a38d10bebf5e.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/c0c599d4ee5c1a378c6b7d05cff64b93f312d2089f0977f98203a38d10bebf5e.jpg)

![c1d314e2cd1d6aea53b32c496ee8d2ed10f10eb2ce4e256d898beb17a5b9c6e6.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/c1d314e2cd1d6aea53b32c496ee8d2ed10f10eb2ce4e256d898beb17a5b9c6e6.jpg)

![c4964bf014bee566b14057d014c0162ba2e0986bd0bb29dddc26f6ef663d01cc.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/c4964bf014bee566b14057d014c0162ba2e0986bd0bb29dddc26f6ef663d01cc.jpg)

![c5ed4eb070bc0766d877b618e30053141313c1f5783249a916e77bf1895106eb.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/c5ed4eb070bc0766d877b618e30053141313c1f5783249a916e77bf1895106eb.jpg)

![cad8c6d1a7b08ec7904819a7991baef4ed7076d83624cf263559885f15480398.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/cad8c6d1a7b08ec7904819a7991baef4ed7076d83624cf263559885f15480398.jpg)

![cb2804acb8d492bbb9e6cad24e940c94911e6efc1143ddd1f8014d1729d58c58.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/cb2804acb8d492bbb9e6cad24e940c94911e6efc1143ddd1f8014d1729d58c58.jpg)

![cd2c9720f7740e1dbfad00c8b4600573e65f254622c6b0ea21449de111c01fd3.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/cd2c9720f7740e1dbfad00c8b4600573e65f254622c6b0ea21449de111c01fd3.jpg)

![e3a4cb82bfa825bcaca6041420b3bebad305f8a18f2deacc004511c704166d48.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/e3a4cb82bfa825bcaca6041420b3bebad305f8a18f2deacc004511c704166d48.jpg)

![f3b1ee2aaf8bae93122377da6d2dffb0cb233a77b93f8a238bce2425dc7e5c89.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/f3b1ee2aaf8bae93122377da6d2dffb0cb233a77b93f8a238bce2425dc7e5c89.jpg)

![f3e6af81c3fae8843ac805dfa5b556c8a250cdf7e0e5bf545c4a5145808c8679.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/f3e6af81c3fae8843ac805dfa5b556c8a250cdf7e0e5bf545c4a5145808c8679.jpg)

![f4768bdc2e7f1e9c0ecf711fed9886cf9516771823c96dde2f31874331f8c3a2.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/f4768bdc2e7f1e9c0ecf711fed9886cf9516771823c96dde2f31874331f8c3a2.jpg)

![fa6d097358edbde8a088ec5889b4a55b651851cd939366f189a0dabf188a6571.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/fa6d097358edbde8a088ec5889b4a55b651851cd939366f189a0dabf188a6571.jpg)

![ff01e801dd34d33d1fdd564fd6b5bcca914bd410135c7f091b2b2bcd57a2ce4c.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/ff01e801dd34d33d1fdd564fd6b5bcca914bd410135c7f091b2b2bcd57a2ce4c.jpg)

![ffc8a50660b10863a7f1214be203fad656e485918a41bd991d897b43c1f00367.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/images/ffc8a50660b10863a7f1214be203fad656e485918a41bd991d897b43c1f00367.jpg)

### Tables

![00acd9a706da37da849f06a412ea697f23ba493102b846b4567d6d0768e855f3.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/00acd9a706da37da849f06a412ea697f23ba493102b846b4567d6d0768e855f3.jpg)

![2816edd96b3f624007b5f8f73810d344bab8153e4bdcd37824efcdff32272b21.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/2816edd96b3f624007b5f8f73810d344bab8153e4bdcd37824efcdff32272b21.jpg)

![597577d503fbac1f0cef95e37bd23e411dd8d43e79ae35b1b35c1d5b78815a3a.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/597577d503fbac1f0cef95e37bd23e411dd8d43e79ae35b1b35c1d5b78815a3a.jpg)

![5ab0cceb44aaed81fd62205b4036dcce4058daa90beae5e29f675f550eeb86d0.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/5ab0cceb44aaed81fd62205b4036dcce4058daa90beae5e29f675f550eeb86d0.jpg)

![89b92e0384d89380c91ff937f479babb3c8f09100d155366e081c7297a887ab1.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/89b92e0384d89380c91ff937f479babb3c8f09100d155366e081c7297a887ab1.jpg)

![a5e04ded7ddd19072b5e68cc7fcb5b95039d35498410f7bb9226c847fd4a7402.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/a5e04ded7ddd19072b5e68cc7fcb5b95039d35498410f7bb9226c847fd4a7402.jpg)

![da955970bac386c2a6573c4887e4cad6a7714d50f139e29677e2a22d9db716ae.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/da955970bac386c2a6573c4887e4cad6a7714d50f139e29677e2a22d9db716ae.jpg)

![e483f285b9f30432bdadcf1dc375078aa9997a6a72b2f78fef0a5ccd098010c3.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/e483f285b9f30432bdadcf1dc375078aa9997a6a72b2f78fef0a5ccd098010c3.jpg)

![fd9e36be92d7b33ef38b6b8b780a0e043f9d17bdb1c4acc806c1dc2c31cb3e7f.jpg](../icml_results/2601_Translation%20Heads_%20Disentangling%20meaning%20from%20language%20in%20LLM-based%20machine%20translation/tables/fd9e36be92d7b33ef38b6b8b780a0e043f9d17bdb1c4acc806c1dc2c31cb3e7f.jpg)

<a id="paper-2569"></a>

## RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing


### Images

![36924f1111c716e9c57ded8783e3ef3bf3559ea4da554243181da487b54dbb49.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/36924f1111c716e9c57ded8783e3ef3bf3559ea4da554243181da487b54dbb49.jpg)

![39380554862108db18b32423945bca8d48ff3274adb9862e29d3a355c1db4851.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/39380554862108db18b32423945bca8d48ff3274adb9862e29d3a355c1db4851.jpg)

![4efd8ae835647c65bd93b979db444d9b044618553fbdb5aa50aaa7ebc556b953.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/4efd8ae835647c65bd93b979db444d9b044618553fbdb5aa50aaa7ebc556b953.jpg)

![5b12db0731fac17b18c31c392d4119d9371643d1ac326b7288e19e305fc7ab8a.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/5b12db0731fac17b18c31c392d4119d9371643d1ac326b7288e19e305fc7ab8a.jpg)

![67c3b7fd21e00e0cde4cec38b0ccbec91cb002cdb4f4e1fdec0dbdb6998707e5.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/67c3b7fd21e00e0cde4cec38b0ccbec91cb002cdb4f4e1fdec0dbdb6998707e5.jpg)

![6deb10bca7f055dc7d9e40c3ed83e99cb15256af784ebfcc734a0212cdc01995.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/6deb10bca7f055dc7d9e40c3ed83e99cb15256af784ebfcc734a0212cdc01995.jpg)

![76b77569e65c052dd21a3ced41001ef5df09e70a1c057f4b19095cf25589baf0.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/76b77569e65c052dd21a3ced41001ef5df09e70a1c057f4b19095cf25589baf0.jpg)

![84442b0e1b258ad182318ca846e36f0f6cb81432092ae3d5a2c26d62fefe0f6e.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/84442b0e1b258ad182318ca846e36f0f6cb81432092ae3d5a2c26d62fefe0f6e.jpg)

![d7c9693c0f5263a6bbc885afadd0700619e76a069da5c4dacfc95e7bdbe7633b.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/images/d7c9693c0f5263a6bbc885afadd0700619e76a069da5c4dacfc95e7bdbe7633b.jpg)

### Tables

![12f7f1caacdf58b77178d2c9af41e1d8d6ce0098ea6204d440e12d554011d4ec.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/12f7f1caacdf58b77178d2c9af41e1d8d6ce0098ea6204d440e12d554011d4ec.jpg)

![2b34785998b3a2fab221a0b198ce9c72824a5a5838839b98b1c7427aebaf9458.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/2b34785998b3a2fab221a0b198ce9c72824a5a5838839b98b1c7427aebaf9458.jpg)

![528555b293d09b59145f856da443c2f478678a2860fd36cc395872030f128749.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/528555b293d09b59145f856da443c2f478678a2860fd36cc395872030f128749.jpg)

![5926406756fabe360a01bc10bbfaa15e768111408fbe7b79e789e40c04e2867e.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/5926406756fabe360a01bc10bbfaa15e768111408fbe7b79e789e40c04e2867e.jpg)

![93d4584bd2f2963bfa3afc91fe6e432a376310f000695b50a0bf4908fc1de2f5.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/93d4584bd2f2963bfa3afc91fe6e432a376310f000695b50a0bf4908fc1de2f5.jpg)

![9618c5ca47620086fe771382f0e312ad9d013dfd70ab0d3a8589534ea202c65b.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/9618c5ca47620086fe771382f0e312ad9d013dfd70ab0d3a8589534ea202c65b.jpg)

![a425bad8acd152cf7164cb26efdd0c044f2b57b195fb79c5fd482d0ae820477a.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/a425bad8acd152cf7164cb26efdd0c044f2b57b195fb79c5fd482d0ae820477a.jpg)

![d8546ed723f7058344c03c7653444bd1387015fccad194cf759e858c06f201f8.jpg](../icml_results/2602_RouterInterp_%20Understanding%20Superposed%20Specialisation%20in%20Mixture%20of%20Experts%20Routing/tables/d8546ed723f7058344c03c7653444bd1387015fccad194cf759e858c06f201f8.jpg)

<a id="paper-2570"></a>

## Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections


### Images

![302a70af48cf99bc1f0ae93125381b71b2f4711822a5a2bf2c276803177d66cb.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/302a70af48cf99bc1f0ae93125381b71b2f4711822a5a2bf2c276803177d66cb.jpg)

![83dd42f83e92991faf63c57a920674ce9c3f08c937d355c2cdb0555ec5a161ed.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/83dd42f83e92991faf63c57a920674ce9c3f08c937d355c2cdb0555ec5a161ed.jpg)

![9e485c00dd901aeb9a93cba22c5419ededd67be13851f948651fe25258641917.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/9e485c00dd901aeb9a93cba22c5419ededd67be13851f948651fe25258641917.jpg)

![e02b126c471dc4654fd0d7991c30a4edca726cfd79788e1b598d14235a08b703.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/e02b126c471dc4654fd0d7991c30a4edca726cfd79788e1b598d14235a08b703.jpg)

![e43b95301a06566bfe705ec706d1b90259761f7914a3e57e89d60800d90f3d9d.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/e43b95301a06566bfe705ec706d1b90259761f7914a3e57e89d60800d90f3d9d.jpg)

![e5f9ec329a1aaf1c043f32c006eb184bb93fe5ac24a64f3c8fe772ff19546e77.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/images/e5f9ec329a1aaf1c043f32c006eb184bb93fe5ac24a64f3c8fe772ff19546e77.jpg)

### Tables

![1563f50e419ab5b8cecc10b8eaf41990e4e3b01db7ed43ecfc95de4315370f0c.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/tables/1563f50e419ab5b8cecc10b8eaf41990e4e3b01db7ed43ecfc95de4315370f0c.jpg)

![6162dec55cc4d9c09dd725ea3d98bf30de875d81b6184e3e16e6a3adca780caf.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/tables/6162dec55cc4d9c09dd725ea3d98bf30de875d81b6184e3e16e6a3adca780caf.jpg)

![87b87d26596d7d68ca4d7c08140c63a9ba059edb4c3753f463aac9605d6eddb8.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/tables/87b87d26596d7d68ca4d7c08140c63a9ba059edb4c3753f463aac9605d6eddb8.jpg)

![9416dbf5413cb92dfcb0efc4981e4298f7ea23535203ab69651401b9d36ee24e.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/tables/9416dbf5413cb92dfcb0efc4981e4298f7ea23535203ab69651401b9d36ee24e.jpg)

![c2c39c5b9423ce97cfe146e3b08f6035844b22f582aa25f44c280524cff67375.jpg](../icml_results/2603_Imitation%20Learning%20for%20Multi-turn%20LM%20Agents%20via%20On-policy%20Expert%20Corrections/tables/c2c39c5b9423ce97cfe146e3b08f6035844b22f582aa25f44c280524cff67375.jpg)

<a id="paper-2571"></a>

## MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations


### Images

![0d6d0753e5448ea523a6bbe68e1e329f03cda0ebfcddf27db536c8f2726d6535.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/images/0d6d0753e5448ea523a6bbe68e1e329f03cda0ebfcddf27db536c8f2726d6535.jpg)

![714096ac751f1b1d53046fc4f7ad135ce23be1f41796a0e57f0fc979eebcd58e.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/images/714096ac751f1b1d53046fc4f7ad135ce23be1f41796a0e57f0fc979eebcd58e.jpg)

![900cf7c5f2026ace035451827d200c5001536dc71e6e8f84b9c60beeadddf03d.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/images/900cf7c5f2026ace035451827d200c5001536dc71e6e8f84b9c60beeadddf03d.jpg)

![c63351bd5278684d2927c5eccbdd02aab1345f4aea6c55ddfebe95b37fa14131.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/images/c63351bd5278684d2927c5eccbdd02aab1345f4aea6c55ddfebe95b37fa14131.jpg)

### Tables

![0ad687c29cf95637538bffc104d3b1cd0460a54108efb697c2cb94f87717fc6f.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/0ad687c29cf95637538bffc104d3b1cd0460a54108efb697c2cb94f87717fc6f.jpg)

![270a30236c8335de6ab12b954e11757a49e3d5ded2318211d14e547360070a62.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/270a30236c8335de6ab12b954e11757a49e3d5ded2318211d14e547360070a62.jpg)

![36e1960e581080999922f68cc1df2fc2635a5987d8b5d9f6092c912e84560a73.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/36e1960e581080999922f68cc1df2fc2635a5987d8b5d9f6092c912e84560a73.jpg)

![50dedee1fffb61611e8372c3d20ee4d8ecce5f92aac4951964e86ea68d523493.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/50dedee1fffb61611e8372c3d20ee4d8ecce5f92aac4951964e86ea68d523493.jpg)

![70c013ae90d387b8bda1203e6f349bfb7d463b648ac32b6a74ab0b73c8e585e2.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/70c013ae90d387b8bda1203e6f349bfb7d463b648ac32b6a74ab0b73c8e585e2.jpg)

![7e10b3521c1c6979eb97f1c45916f7a950eb718a25f65800872cd1d910c5b62b.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/7e10b3521c1c6979eb97f1c45916f7a950eb718a25f65800872cd1d910c5b62b.jpg)

![7fc3e0357871fb901e901a1ac452b7ce5d14b17da921df17a228ca7073f0f77b.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/7fc3e0357871fb901e901a1ac452b7ce5d14b17da921df17a228ca7073f0f77b.jpg)

![8c106b7d1481ccfdc243962ba3dd872b2575cfde1ecf0567078adf939347f8de.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/8c106b7d1481ccfdc243962ba3dd872b2575cfde1ecf0567078adf939347f8de.jpg)

![8f3e6e263c5ffde4728c083af169d406db1c54209d2a6e5916b22b66bc461c20.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/8f3e6e263c5ffde4728c083af169d406db1c54209d2a6e5916b22b66bc461c20.jpg)

![a0caeaa2f90a8455adf000bdd7bf4ffd6a14d4959ff37b37a5501c127f76c7aa.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/a0caeaa2f90a8455adf000bdd7bf4ffd6a14d4959ff37b37a5501c127f76c7aa.jpg)

![a160c990330e4278a298a0f54743c475053ca7c4cf7393567cff02fe6247eff8.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/a160c990330e4278a298a0f54743c475053ca7c4cf7393567cff02fe6247eff8.jpg)

![a2c4f72dc346b16a3e2bcdac9ec895a8afe0e1a1d4d5db7df4f141b844f0cfcf.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/a2c4f72dc346b16a3e2bcdac9ec895a8afe0e1a1d4d5db7df4f141b844f0cfcf.jpg)

![a7efaf2f334ddfccc1acc2166646d78bb27b30e7773182134ffd60c719f91ff3.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/a7efaf2f334ddfccc1acc2166646d78bb27b30e7773182134ffd60c719f91ff3.jpg)

![acc51c9ac6b208913bb23532ba052d099897c7af837c36cd77005a898b5ce9b2.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/acc51c9ac6b208913bb23532ba052d099897c7af837c36cd77005a898b5ce9b2.jpg)

![ad3ca8cd42d9d4338ebd486081cc3a1aa7f8834150e2be3fb829da2f0f05b210.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/ad3ca8cd42d9d4338ebd486081cc3a1aa7f8834150e2be3fb829da2f0f05b210.jpg)

![bbadfa8c8a12cac8d424fd2f10db2507c199c59c7b00c761d46d433bef7e7d6a.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/bbadfa8c8a12cac8d424fd2f10db2507c199c59c7b00c761d46d433bef7e7d6a.jpg)

![d1243dae1edbf5bf6aba0e685ac8fe300b850668fa0b6b86189d4929192bb0ae.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/d1243dae1edbf5bf6aba0e685ac8fe300b850668fa0b6b86189d4929192bb0ae.jpg)

![e0b9035c77d50f09d8bfed5d5369156ce12798c26aea0f754d8d8008a69a061f.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/e0b9035c77d50f09d8bfed5d5369156ce12798c26aea0f754d8d8008a69a061f.jpg)

![ed1eda888b9fc128bd80e115985c280260405c753b495f965b7f4b2dc6e6aa61.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/ed1eda888b9fc128bd80e115985c280260405c753b495f965b7f4b2dc6e6aa61.jpg)

![f82c112f704d34eac9a30ec1e0abe6c27464b78de7a255b133445d611f205e72.jpg](../icml_results/2604_MultiHal_%20Multilingual%20Dataset%20for%20Knowledge-Graph%20Grounded%20Evaluation%20of%20LLM%20Hallucinations/tables/f82c112f704d34eac9a30ec1e0abe6c27464b78de7a255b133445d611f205e72.jpg)

<a id="paper-2572"></a>

## MAFE: Enabling Equitable Algorithm Design in Multi-Agent Multi-Stage Decision-Making Systems


### Images

![000c6ed90bde7cde7b13df4c523c5cdc3272d7c300c785f3621b52f030bd60a3.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/000c6ed90bde7cde7b13df4c523c5cdc3272d7c300c785f3621b52f030bd60a3.jpg)

![0324b9af369be5e9f6774faf208190d6eeaa0482400232a238c646703f44ecc2.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/0324b9af369be5e9f6774faf208190d6eeaa0482400232a238c646703f44ecc2.jpg)

![181ad77e9455539fe9134ae417c529b71de64a3c85e82e2b9e1f2fbc9cf9372d.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/181ad77e9455539fe9134ae417c529b71de64a3c85e82e2b9e1f2fbc9cf9372d.jpg)

![1ab1c090d0331811a347e510f0e89b2838a830089d43d02fe4f423ba029bbc50.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/1ab1c090d0331811a347e510f0e89b2838a830089d43d02fe4f423ba029bbc50.jpg)

![3762c96cab78b4656a798ce83a7d37340c58338e38e227c285529f3cd24155e6.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/3762c96cab78b4656a798ce83a7d37340c58338e38e227c285529f3cd24155e6.jpg)

![3a1fa1860d58c9a1a87e6ba0e6a461961cca15f5ad69b7c551e900c17a58a414.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/3a1fa1860d58c9a1a87e6ba0e6a461961cca15f5ad69b7c551e900c17a58a414.jpg)

![3b4e725bbabfd9427483deb7e8cc1f979c2f113eccfe689833572fe0ae73d6bf.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/3b4e725bbabfd9427483deb7e8cc1f979c2f113eccfe689833572fe0ae73d6bf.jpg)

![3e8a662e91505979892bbdc068177ee51e87155f651e5c8b2fcddfb5765ac1a7.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/3e8a662e91505979892bbdc068177ee51e87155f651e5c8b2fcddfb5765ac1a7.jpg)

![3fa32317949995d89845b56153a786166a764ab62daefce9aefada3db74e3dd5.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/3fa32317949995d89845b56153a786166a764ab62daefce9aefada3db74e3dd5.jpg)

![69bcfd010675aa4881667ebaf43ddf4bbd32241bda3c028aa56cd728817b3589.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/69bcfd010675aa4881667ebaf43ddf4bbd32241bda3c028aa56cd728817b3589.jpg)

![7eb3fe1efb6bd3425648b44032df9bf79dd7d49515b1b9d74f7321f121bf2647.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/7eb3fe1efb6bd3425648b44032df9bf79dd7d49515b1b9d74f7321f121bf2647.jpg)

![94b151c79a32840695d389c8dd2e30e5de64f6b0fba1da9274d10a03d64f41aa.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/94b151c79a32840695d389c8dd2e30e5de64f6b0fba1da9274d10a03d64f41aa.jpg)

![99e745433d062280ee0b5dfc1863506eb9790dff0db439fb4e0a059f8ea0569d.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/99e745433d062280ee0b5dfc1863506eb9790dff0db439fb4e0a059f8ea0569d.jpg)

![b3e75e0eca7b070651c4c77007c9939aad8f89c978877aa9b587ef6eb1d52e45.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/b3e75e0eca7b070651c4c77007c9939aad8f89c978877aa9b587ef6eb1d52e45.jpg)

![c234abe08e826ea78938359467984d7a375a689b154d7d078ea9eb0c45e5e709.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/c234abe08e826ea78938359467984d7a375a689b154d7d078ea9eb0c45e5e709.jpg)

![d707dd97456bc9b9d6e9b39535081e1b4b232427534d3fa8663cd32f43dafd30.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/d707dd97456bc9b9d6e9b39535081e1b4b232427534d3fa8663cd32f43dafd30.jpg)

![dd8344060e92e58e77f3a639f13d508cad8d5dc92e9cb3df53ed7c0fb56e159a.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/dd8344060e92e58e77f3a639f13d508cad8d5dc92e9cb3df53ed7c0fb56e159a.jpg)

![faba89c2254f9bbca0e4508c9b8fff0bb7d94aa9977d91be3b245a07d590301a.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/images/faba89c2254f9bbca0e4508c9b8fff0bb7d94aa9977d91be3b245a07d590301a.jpg)

### Tables

![015c1d8b2863135a927fc7e36e9814814525a878044006af8bb8c367fa2f6f52.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/015c1d8b2863135a927fc7e36e9814814525a878044006af8bb8c367fa2f6f52.jpg)

![16f09e5d70a85e815ae4860a85e659c3cb042fdc43f1b7a1e65a1d07a4fa27d8.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/16f09e5d70a85e815ae4860a85e659c3cb042fdc43f1b7a1e65a1d07a4fa27d8.jpg)

![200e42e13b17b324091e1f2d147353bf1e0cf54d316967b600cbb2208248f2fe.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/200e42e13b17b324091e1f2d147353bf1e0cf54d316967b600cbb2208248f2fe.jpg)

![2f49b5b21e16d00b19e4756f4b5d99a9007a16b4cec64df4451ed8df5ab5561f.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/2f49b5b21e16d00b19e4756f4b5d99a9007a16b4cec64df4451ed8df5ab5561f.jpg)

![643c6b474ed088b6134c3a265b4eb0d433966dbc9c6cf637607a053fd929960b.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/643c6b474ed088b6134c3a265b4eb0d433966dbc9c6cf637607a053fd929960b.jpg)

![7857a91959b687a08eee01e32064c9588b745f566c73a544e5d450ffd4bfb67a.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/7857a91959b687a08eee01e32064c9588b745f566c73a544e5d450ffd4bfb67a.jpg)

![bdd1a2e3342330e73e67c29013a50eb3c1f9efefd4ba0b4ed9b4cd0b6179e435.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/bdd1a2e3342330e73e67c29013a50eb3c1f9efefd4ba0b4ed9b4cd0b6179e435.jpg)

![d91da2219239b0a9dea8398aa5e94165f47edc7dbf311f5beec436e8d5dc6c55.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/d91da2219239b0a9dea8398aa5e94165f47edc7dbf311f5beec436e8d5dc6c55.jpg)

![e3d33eeab60e714800a6bfa1979ce2ef6bcdfc24e388c09c398a7085b09b2b93.jpg](../icml_results/2605_MAFE_%20Enabling%20Equitable%20Algorithm%20Design%20in%20Multi-Agent%20Multi-Stage%20Decision-Making%20Systems/tables/e3d33eeab60e714800a6bfa1979ce2ef6bcdfc24e388c09c398a7085b09b2b93.jpg)

<a id="paper-2573"></a>

## Trust-Region Diffusion Policies for Massively Parallel On-Policy RL


### Images

![00b278a195e607e397922ab03284f43cd5db9226c06471004c567bc7d307b452.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/00b278a195e607e397922ab03284f43cd5db9226c06471004c567bc7d307b452.jpg)

![1ee56a01f2f999cb9abbfc53b131c70a0f4500afcfdbbf10890f030796764c65.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/1ee56a01f2f999cb9abbfc53b131c70a0f4500afcfdbbf10890f030796764c65.jpg)

![37b22b30d15510d9190d2c4263d2620b2284210057506fdaa939348529d8bd2a.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/37b22b30d15510d9190d2c4263d2620b2284210057506fdaa939348529d8bd2a.jpg)

![42ca8813774acb2021aacd5ecb5f5d612f9f7cd33210beeb1dbb6cc5d7d805f3.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/42ca8813774acb2021aacd5ecb5f5d612f9f7cd33210beeb1dbb6cc5d7d805f3.jpg)

![56a05fb3a75f960621d69120b01e4332ea6c222d442853a318ef7e25a35f7cdf.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/56a05fb3a75f960621d69120b01e4332ea6c222d442853a318ef7e25a35f7cdf.jpg)

![7a2057fc186f2cfcdf9a301db12972d900ef162f48d575577aa4412f8f516d42.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/7a2057fc186f2cfcdf9a301db12972d900ef162f48d575577aa4412f8f516d42.jpg)

![7a8317d60934870870aae19b58f24e198b7045e72b1fa5285a241557a587dc33.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/7a8317d60934870870aae19b58f24e198b7045e72b1fa5285a241557a587dc33.jpg)

![7cbf85c90d6c3d8cb51640f080914b48a9126b94ebbe1c85908b2312a2dd44ee.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/7cbf85c90d6c3d8cb51640f080914b48a9126b94ebbe1c85908b2312a2dd44ee.jpg)

![80100b7d8e47d40c9258ecbbdf3902a2368a8da6f9beaf7fe71420b5f57c5ad5.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/80100b7d8e47d40c9258ecbbdf3902a2368a8da6f9beaf7fe71420b5f57c5ad5.jpg)

![8211ad191dcd0ecdc9a186aca0e25b81bc226b86ed2729026ad1b2bcbbd74ff2.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/8211ad191dcd0ecdc9a186aca0e25b81bc226b86ed2729026ad1b2bcbbd74ff2.jpg)

![aa4a6e7e0bfdbac3020b23497253c5bd91c603db7e62d705594b91cddb4dff3f.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/aa4a6e7e0bfdbac3020b23497253c5bd91c603db7e62d705594b91cddb4dff3f.jpg)

![cfa92ada71026a4b5e99b7eed9dc4d5f507920a4ab54e257fc68b0bb52e55e0c.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/cfa92ada71026a4b5e99b7eed9dc4d5f507920a4ab54e257fc68b0bb52e55e0c.jpg)

![d414a829fce5bc25860a56656eb5b736e836364774a304c48ac06d62242bec44.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/d414a829fce5bc25860a56656eb5b736e836364774a304c48ac06d62242bec44.jpg)

![ff01f25b68f2badaec95d5a98e5e1cf9f8c5659b264b18f5f632316933076377.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/images/ff01f25b68f2badaec95d5a98e5e1cf9f8c5659b264b18f5f632316933076377.jpg)

### Tables

![0afc90730fd10f29973b08ba9b2db5a0c28f170037ddfe06823bb0e9565cb766.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/tables/0afc90730fd10f29973b08ba9b2db5a0c28f170037ddfe06823bb0e9565cb766.jpg)

![10b0018472c1ac25eb656762245038fc2c72600a581bc33c303a30728f477def.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/tables/10b0018472c1ac25eb656762245038fc2c72600a581bc33c303a30728f477def.jpg)

![47bafe7bbece1e28d521a3be08d7e1846cc2ff384c1c6dde1debff27bb9f07fd.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/tables/47bafe7bbece1e28d521a3be08d7e1846cc2ff384c1c6dde1debff27bb9f07fd.jpg)

![581c1c062195ee1af5210a9baf42ad46b6a70b1edb38f2f3b415bd4876e39bc1.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/tables/581c1c062195ee1af5210a9baf42ad46b6a70b1edb38f2f3b415bd4876e39bc1.jpg)

![74e42aac2b7772fe25b8cbb13b3555e3900031e86c303f31d81c2820c53d570c.jpg](../icml_results/2606_Trust-Region%20Diffusion%20Policies%20for%20Massively%20Parallel%20On-Policy%20RL/tables/74e42aac2b7772fe25b8cbb13b3555e3900031e86c303f31d81c2820c53d570c.jpg)

<a id="paper-2574"></a>

## HieRD: Hierarchical Relational Distillation for Vision-Language Embedding Models

### Images

![195aacc37356b68cb4163262f1d6fcd5c32b0a20d9cf4ea5a9cd3bbf74c3c443.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/195aacc37356b68cb4163262f1d6fcd5c32b0a20d9cf4ea5a9cd3bbf74c3c443.jpg)

![9cafeb4da0a1586d04bb94aed447ea9653eb964ad68aa0ee154569334058fcc8.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/9cafeb4da0a1586d04bb94aed447ea9653eb964ad68aa0ee154569334058fcc8.jpg)

![d549172c9b28e3defd1b6041383f1d9e7f6f8c97e0b94b197e916af4f3e6f6b3.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/d549172c9b28e3defd1b6041383f1d9e7f6f8c97e0b94b197e916af4f3e6f6b3.jpg)

![dcd9b9d583d4e294de5325ab5e2387ee0e4ead0e0e0f46d27f6b15d203f3afc4.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/dcd9b9d583d4e294de5325ab5e2387ee0e4ead0e0e0f46d27f6b15d203f3afc4.jpg)

![efc24939a168259c2e7ff60769523be10974afc82add095b048fca017267e6fa.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/images/efc24939a168259c2e7ff60769523be10974afc82add095b048fca017267e6fa.jpg)

### Tables

![05a59777a1231fc5771c9778f8078920e83756577adc09854151e805b8e487ee.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/05a59777a1231fc5771c9778f8078920e83756577adc09854151e805b8e487ee.jpg)

![2bd23d250614c1b37b30725d69b4958a36783a7a51414f495f86cfde47cf4290.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/2bd23d250614c1b37b30725d69b4958a36783a7a51414f495f86cfde47cf4290.jpg)

![37389204f816c189bc5bf9978e93395e0697105c8df5709f1e924deab162ad20.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/37389204f816c189bc5bf9978e93395e0697105c8df5709f1e924deab162ad20.jpg)

![458879bc3a855f148fd4886901a40bd6ddc4fed9fdfef84f41dc9a3b2a023b71.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/458879bc3a855f148fd4886901a40bd6ddc4fed9fdfef84f41dc9a3b2a023b71.jpg)

![4856fed97035a1ca4537c74c9042caf6c951e7ef6c36ffdcf015529f531e7a09.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/4856fed97035a1ca4537c74c9042caf6c951e7ef6c36ffdcf015529f531e7a09.jpg)

![647d20ed80f9f634a09884e4e1c3d3b07faa5f7fd200560b544a212c726600dc.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/647d20ed80f9f634a09884e4e1c3d3b07faa5f7fd200560b544a212c726600dc.jpg)

![6b6c38d6efec4de0437df9c579c535a1af2342e4ea15e763f97dc4c5d4714a0e.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/6b6c38d6efec4de0437df9c579c535a1af2342e4ea15e763f97dc4c5d4714a0e.jpg)

![72c0c64b8ecee1953d41890016c9de44d407f9a9c468b5fed98187e5f619e209.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/72c0c64b8ecee1953d41890016c9de44d407f9a9c468b5fed98187e5f619e209.jpg)

![741fe895769cdd246ad8beaf689350e0d1d8cdc5ab97c41ae1c6d772af695d14.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/741fe895769cdd246ad8beaf689350e0d1d8cdc5ab97c41ae1c6d772af695d14.jpg)

![7e73b76756f18bc3c41b5351da47c94e4715d3042cf9118e9bf21b60eb1f3dfc.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/7e73b76756f18bc3c41b5351da47c94e4715d3042cf9118e9bf21b60eb1f3dfc.jpg)

![81a9859c6d2d6c985a99daeabb96a236c905067b07f9148cb896841d49fa9ecd.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/81a9859c6d2d6c985a99daeabb96a236c905067b07f9148cb896841d49fa9ecd.jpg)

![a7a039059411b40a9761b053d38c146808fab74d9ecdcd40efef9c062d2eb26f.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/a7a039059411b40a9761b053d38c146808fab74d9ecdcd40efef9c062d2eb26f.jpg)

![b9f28a16e00a7d543dba8c703ba3b1e90ff5c69f654854320256904df983f873.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/b9f28a16e00a7d543dba8c703ba3b1e90ff5c69f654854320256904df983f873.jpg)

![bd11f2afda6bf76e0e89f006c7abd9961d357ce9c52415485cdaf98ee37149c5.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/bd11f2afda6bf76e0e89f006c7abd9961d357ce9c52415485cdaf98ee37149c5.jpg)

![c946cc6f246737a28b9ecefabbbb7b7457952221d32de4ca88c28e170fc220f9.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/c946cc6f246737a28b9ecefabbbb7b7457952221d32de4ca88c28e170fc220f9.jpg)

![cc6efc6a0ed5974c03e2ae24d8e7fb2e04d3a6f492e7f7a8a56bf139c84aeb18.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/cc6efc6a0ed5974c03e2ae24d8e7fb2e04d3a6f492e7f7a8a56bf139c84aeb18.jpg)

![d32703f49f0e5e262faf45e21f95904fd5cbe2f7fe5797a6b8e7bbfe69cc77ab.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/d32703f49f0e5e262faf45e21f95904fd5cbe2f7fe5797a6b8e7bbfe69cc77ab.jpg)

![d4d1e8775bcb79470940caaf4600a85f6ae4b3c59c2de12bbbc925a80f5bf5c4.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/d4d1e8775bcb79470940caaf4600a85f6ae4b3c59c2de12bbbc925a80f5bf5c4.jpg)

![dd3f2b78bda220008e9abd4e2666dfa03598bf4bbb93a266c65019a788422c54.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/dd3f2b78bda220008e9abd4e2666dfa03598bf4bbb93a266c65019a788422c54.jpg)

![e9cb417a5fb9f11eac5c9ce116fa1bd54eb4b98e94fee45135ef9dc317d98986.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/e9cb417a5fb9f11eac5c9ce116fa1bd54eb4b98e94fee45135ef9dc317d98986.jpg)

![f600f18b410d3211e9686676bd364c0bc5de8ad43134210f29d35a74d6ccf469.jpg](../icml_results/2607_HieRD_%20Hierarchical%20Relational%20Distillation%20for%20Vision-Language%20Embedding%20Models/tables/f600f18b410d3211e9686676bd364c0bc5de8ad43134210f29d35a74d6ccf469.jpg)
