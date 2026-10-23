# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 3 of 197

## 目录 (Table of Contents)

1. [Protein Fold Classification at Scale: Benchmarking and Pretraining](#paper-0067)
2. [DOUBT: Decoupled Object-level Understanding and Bridging via vMF-based Trustworthiness for Hallucination Detection in MLLMs](#paper-0068)
3. [Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](#paper-0069)
4. [A Causal Decomposition Approach for Fair Contextual Multi-Armed Bandits](#paper-0070)
5. [Do LLMs Signal When They’re Right? Evidence from Neuron Agreement](#paper-0071)
6. [Real-World Unsupervised Models Generalize to Predict Brain Responses to Out-of-Distribution Stimuli](#paper-0072)
7. [Scaling Law for Quantization-Aware Training](#paper-0073)
8. [Towards Optimal Robustness in Learning-Augmented Paging](#paper-0074)
9. [Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](#paper-0075)
10. [Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization](#paper-0076)
11. [Unifying and Optimizing Data Values for Selection via Sequential Decision-Making](#paper-0077)
12. [Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms](#paper-0078)
13. [SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models](#paper-0079)
14. [Reinforced Sequential Monte Carlo for Amortised Sampling](#paper-0080)
15. [Concept Removal Guidance: Evidence-Calibrated Negative Guidance for Safe Diffusion Sampling](#paper-0081)
16. [End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](#paper-0082)
17. [Transformer Circuits Can Realize Clustering Algorithms](#paper-0083)
18. [Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models](#paper-0084)
19. [SVRG and Beyond via Posterior Correction](#paper-0085)
20. [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](#paper-0086)
21. [Causal Modeling of Selection in Evolution](#paper-0087)
22. [Second-Order Smooth Planning with Optimal-Transport Bellman Smoothing](#paper-0088)
23. [Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models](#paper-0089)
24. [On the Expressive Power of Permutation-Equivariant Weight-Space Networks](#paper-0090)
25. [Adaptive Memory Retention in Dynamic Graphs](#paper-0091)
26. [Mixture of Concept Bottleneck Experts](#paper-0092)
27. [Beyond First-order Asymptotics in Sequential Mean Testing](#paper-0093)
28. [A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps in Single- and Multi-Index Models](#paper-0094)
29. [BFTS: Thompson Sampling with Bayesian Additive Regression Trees](#paper-0095)
30. [LASER: Learning Active Sensing for Continuum Field Reconstruction](#paper-0096)
31. [FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications](#paper-0097)
32. [Listening Through the Noise: Cauchy-Driven Diffusion Bridges for Robust Gastrointestinal Auscultation and Clinical Benchmarking](#paper-0098)
33. [Disentangling Latent Risk Pathways via Bayesian Hypergraph Inference](#paper-0099)

---


<a id="paper-0067"></a>

## Protein Fold Classification at Scale: Benchmarking and Pretraining

### Images

![022de6fbc9130e7598be9293699c6e4bd954c10d31889408738508144b9d5063.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/022de6fbc9130e7598be9293699c6e4bd954c10d31889408738508144b9d5063.jpg)

![1add36664670ac49fa919b9945376685c61ff353c4a41aba2c25b4ff7de3228a.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/1add36664670ac49fa919b9945376685c61ff353c4a41aba2c25b4ff7de3228a.jpg)

![53f2d8d2ca34ccead80117788b73ce72d079a1e6dcc2d1eb27955e4ae4354b80.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/53f2d8d2ca34ccead80117788b73ce72d079a1e6dcc2d1eb27955e4ae4354b80.jpg)

![98615b7fadf48640352cbf8727a4e20d55b59569d17a71c80421160ba13774e3.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/98615b7fadf48640352cbf8727a4e20d55b59569d17a71c80421160ba13774e3.jpg)

![9af0e8ea9bc2ff63bb4b8c731e5d11ccd07bdbf5e04d1b208bd8782c2ea2a730.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/9af0e8ea9bc2ff63bb4b8c731e5d11ccd07bdbf5e04d1b208bd8782c2ea2a730.jpg)

![a770509983c82893a91fffd8c3f6106c1c70b85ba4d28cc02a3cb59dca333e54.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/a770509983c82893a91fffd8c3f6106c1c70b85ba4d28cc02a3cb59dca333e54.jpg)

![ae6817f79728acb0e9216ff40b06e6688a9e3a009fe7d331f1599e6bdfefef63.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/ae6817f79728acb0e9216ff40b06e6688a9e3a009fe7d331f1599e6bdfefef63.jpg)

![d1b08867050825379e8393608f63e5c85847f6200ea2d06e6b5cb2e788db8ea9.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/d1b08867050825379e8393608f63e5c85847f6200ea2d06e6b5cb2e788db8ea9.jpg)

![d8e52dda9905b750d17a204bfb51b5c44df474a7c4427ebf5ba5eb7ba582a024.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/d8e52dda9905b750d17a204bfb51b5c44df474a7c4427ebf5ba5eb7ba582a024.jpg)

![e89760e1a1733ed8c6ac33f39a1b6c2974f0ede2611e24429869884c282b4800.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/e89760e1a1733ed8c6ac33f39a1b6c2974f0ede2611e24429869884c282b4800.jpg)

![f81a543b6d7f17ecd5398962165a3a400b1acb78c581d2d338b6af49b27fc270.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/f81a543b6d7f17ecd5398962165a3a400b1acb78c581d2d338b6af49b27fc270.jpg)

### Tables

![0e4c80506b5a76adff0dc7c2a7fecfa474328147dc32a416b58774be17731377.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/0e4c80506b5a76adff0dc7c2a7fecfa474328147dc32a416b58774be17731377.jpg)

![27bb8ca4c9740ccc0c008a3bcc88432b860bba03f2713228b48829b50b4c2ea8.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/27bb8ca4c9740ccc0c008a3bcc88432b860bba03f2713228b48829b50b4c2ea8.jpg)

![2b5ec377343bae3d74ef4a306bf43eaeb2a23af9ffd2a8915cad4e9694d2da7b.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/2b5ec377343bae3d74ef4a306bf43eaeb2a23af9ffd2a8915cad4e9694d2da7b.jpg)

![39b629455cf0a31b9a86eb785c8a931952cc26862d3c8f97852ab1b98744b2bf.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/39b629455cf0a31b9a86eb785c8a931952cc26862d3c8f97852ab1b98744b2bf.jpg)

![60d661ff65d311d71529708465f8ec06ea6796f60837b8087c5cada4d66e5713.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/60d661ff65d311d71529708465f8ec06ea6796f60837b8087c5cada4d66e5713.jpg)

![64560f20f6e546bd88259cba023a7ac09d8bf26cec5499e4cabebe1db4deb8ef.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/64560f20f6e546bd88259cba023a7ac09d8bf26cec5499e4cabebe1db4deb8ef.jpg)

![66529fd12d33f57b48d0eea8aa1e3aa32e5cf3ee01bef3f36d6f26ad1042e86c.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/66529fd12d33f57b48d0eea8aa1e3aa32e5cf3ee01bef3f36d6f26ad1042e86c.jpg)

![97e04d732ebe63425163d9062f12590979c0db0c35947fb967490e6e08473613.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/97e04d732ebe63425163d9062f12590979c0db0c35947fb967490e6e08473613.jpg)

![a36e7efeef628cc514790df6dc0777fb50e8abc3e6983c5d74df576f0fefad43.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/a36e7efeef628cc514790df6dc0777fb50e8abc3e6983c5d74df576f0fefad43.jpg)

![f2a2e312c973dea73e72acb460748a2e0c25c06b2e8a344473b5ce4d25e8a91b.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/f2a2e312c973dea73e72acb460748a2e0c25c06b2e8a344473b5ce4d25e8a91b.jpg)

## Protein Fold Classification at Scale: Benchmarking and Pretraining


### Images

![05e8d6136f471aaf3028f1f650d1a46f44e6fd996087904d4b3b7e453664ae18.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/05e8d6136f471aaf3028f1f650d1a46f44e6fd996087904d4b3b7e453664ae18.jpg)

![1007a7dd7b785e10fce6e63fc0717f137088e3522980709b60b8a243731db825.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/1007a7dd7b785e10fce6e63fc0717f137088e3522980709b60b8a243731db825.jpg)

![1d96aefa673de2fc920fa99a43b2184100b8c5f555c675d343b45ab8062b0faf.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/1d96aefa673de2fc920fa99a43b2184100b8c5f555c675d343b45ab8062b0faf.jpg)

![523537dea20a1909b91a74c19d299c947645730d97e132cf5bd92a74b8495220.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/523537dea20a1909b91a74c19d299c947645730d97e132cf5bd92a74b8495220.jpg)

![62568708d5da57727251b3af7d4ca036d3da942a4d8bd25d4c66cc43134499a2.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/62568708d5da57727251b3af7d4ca036d3da942a4d8bd25d4c66cc43134499a2.jpg)

![62cbc7364c279ca605de29adee9947235e528157e9ca393b650420070a510978.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/62cbc7364c279ca605de29adee9947235e528157e9ca393b650420070a510978.jpg)

![77c87eba9484a01cc287b4394bbc003ca5a796be89857cc4ac6f7884e97b8e7f.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/77c87eba9484a01cc287b4394bbc003ca5a796be89857cc4ac6f7884e97b8e7f.jpg)

![8c6b98670a627081936b47a31cf754590440d4ffe760e36b0f2a4f051022f9a3.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/8c6b98670a627081936b47a31cf754590440d4ffe760e36b0f2a4f051022f9a3.jpg)

![947ec99ae27d1d023fdfb9ad0f3b386e73de53d68818c50388c215afa79e701a.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/947ec99ae27d1d023fdfb9ad0f3b386e73de53d68818c50388c215afa79e701a.jpg)

![9fb0758662d316e31c3143bbba2578bf3e32217b57685433fbe8b0e35e67ef9c.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/9fb0758662d316e31c3143bbba2578bf3e32217b57685433fbe8b0e35e67ef9c.jpg)

![fcf42cc8d8a01c48d5c626831cc7b11345e3cf1fd4fbadb5b2a48c3364a2e76b.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/images/fcf42cc8d8a01c48d5c626831cc7b11345e3cf1fd4fbadb5b2a48c3364a2e76b.jpg)

### Tables

![120ef9a4bf81ecb24190995f858fa604187a8cb88fd04fb5dbaa8e702d1338e4.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/120ef9a4bf81ecb24190995f858fa604187a8cb88fd04fb5dbaa8e702d1338e4.jpg)

![134bf8226bdf66e01964b155570d4f2789aa457b31eb1eceae7adb22a43a303a.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/134bf8226bdf66e01964b155570d4f2789aa457b31eb1eceae7adb22a43a303a.jpg)

![2c5efef56438d512e0973b5c71a717e8f8c938fcec35889c29c490c1df572e46.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/2c5efef56438d512e0973b5c71a717e8f8c938fcec35889c29c490c1df572e46.jpg)

![5a7617d6642eaee62d0b22d88c3cf02f0a678e7d8d6b9af5c25568b69651edc9.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/5a7617d6642eaee62d0b22d88c3cf02f0a678e7d8d6b9af5c25568b69651edc9.jpg)

![5cdce1c918ffbff258dea77e4e38791463be5525d0ee089d197c4f4d3b90139f.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/5cdce1c918ffbff258dea77e4e38791463be5525d0ee089d197c4f4d3b90139f.jpg)

![601b95f4a99655294f67cee38a120c9d6d8f91e3aaf22ce528fb51564c560d68.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/601b95f4a99655294f67cee38a120c9d6d8f91e3aaf22ce528fb51564c560d68.jpg)

![6a5ffac1cd78ecf695fb87215e54411109dc515da0ac0e441ac69e4b57ce720f.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/6a5ffac1cd78ecf695fb87215e54411109dc515da0ac0e441ac69e4b57ce720f.jpg)

![7788db40dcd48aaae81a46db3c9a22544657675ab932a7d4bfc37e95f05004a9.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/7788db40dcd48aaae81a46db3c9a22544657675ab932a7d4bfc37e95f05004a9.jpg)

![a50c7513f73e7661fd2aa3f67f40c6c3f5a73939d951da0c78f8a3232f31dd47.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/a50c7513f73e7661fd2aa3f67f40c6c3f5a73939d951da0c78f8a3232f31dd47.jpg)

![c63342ea07e0519eafa4bc4e49e5ab45327a9340eabe6d39f6f16733e86f15b3.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/c63342ea07e0519eafa4bc4e49e5ab45327a9340eabe6d39f6f16733e86f15b3.jpg)

![cc0370d3def5923c5ebb7b7a5110dd99d25384fdeebdd1ad8623e0463adf28bf.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/cc0370d3def5923c5ebb7b7a5110dd99d25384fdeebdd1ad8623e0463adf28bf.jpg)

![d415a182e4d72108e17a63dee013b7842be39ce2533f268bdc55f537820286cf.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/d415a182e4d72108e17a63dee013b7842be39ce2533f268bdc55f537820286cf.jpg)

![db430ad8734a4d6ce7efa9fda4f7b9edebe67adc7c4d05e2d65a82fb19746dd0.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/db430ad8734a4d6ce7efa9fda4f7b9edebe67adc7c4d05e2d65a82fb19746dd0.jpg)

![dcd4c4bf3694f8b9937090becfd497b9ac1df2b3a4889a9b56186fd5dcd21554.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/dcd4c4bf3694f8b9937090becfd497b9ac1df2b3a4889a9b56186fd5dcd21554.jpg)

![e69f01f7d7849612641a306abe22c8e14aec9f12f894e29b4b496e65614555e6.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/e69f01f7d7849612641a306abe22c8e14aec9f12f894e29b4b496e65614555e6.jpg)

![f73333dad30467078d5a0f4c348f59018b1d0b6f2c353acb055647075f9c1194.jpg](../icml_results/68_Protein%20Fold%20Classification%20at%20Scale_%20Benchmarking%20and%20Pretraining/tables/f73333dad30467078d5a0f4c348f59018b1d0b6f2c353acb055647075f9c1194.jpg)

<a id="paper-0068"></a>

## DOUBT: Decoupled Object-level Understanding and Bridging via vMF-based Trustworthiness for Hallucination Detection in MLLMs


### Images

![134aa9d5077d3a33b6d9c3feccc342460d6821d38a25a55489dd4c285384c47d.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/134aa9d5077d3a33b6d9c3feccc342460d6821d38a25a55489dd4c285384c47d.jpg)

![1ebd56a97827f26465f9301f2d36885f5156ac721f0e37abd39bad5e517c5211.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/1ebd56a97827f26465f9301f2d36885f5156ac721f0e37abd39bad5e517c5211.jpg)

![3aef82bd54ded5e3c7a0365a091109afe01324486330a68968826ffd2078aa26.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/3aef82bd54ded5e3c7a0365a091109afe01324486330a68968826ffd2078aa26.jpg)

![41bfa325065e75c602f9c35ca0bc826f2ff200461e946a8a585515f56bac59d9.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/41bfa325065e75c602f9c35ca0bc826f2ff200461e946a8a585515f56bac59d9.jpg)

![42a38c68a68ba0b97c10c664f35b8989ffaf3fb1c09f82b8895031b8efddfb37.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/42a38c68a68ba0b97c10c664f35b8989ffaf3fb1c09f82b8895031b8efddfb37.jpg)

![50923eb558a2fbb48ab7377e0599592b253b2ccfb5737a9285a39fa20feb59b7.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/50923eb558a2fbb48ab7377e0599592b253b2ccfb5737a9285a39fa20feb59b7.jpg)

![61b7a9ff11417837562fd172c9aeec6080b15770b2975b0d19f3f9c2fde03c17.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/61b7a9ff11417837562fd172c9aeec6080b15770b2975b0d19f3f9c2fde03c17.jpg)

![6a45dd77d5cfa109b0eba5102a56b722a2dc059d9b3b45cb3fd8bf988b02f424.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/6a45dd77d5cfa109b0eba5102a56b722a2dc059d9b3b45cb3fd8bf988b02f424.jpg)

![7a52ea3ef979cbb08ea9209242bfd181de854d7f84256384e6879497f4b8f002.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/7a52ea3ef979cbb08ea9209242bfd181de854d7f84256384e6879497f4b8f002.jpg)

![873f4290be7026e90ef44faf77f9fd6d9dfd0978c5f024b36e9bc933534f9a75.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/873f4290be7026e90ef44faf77f9fd6d9dfd0978c5f024b36e9bc933534f9a75.jpg)

![947cd5b7099023daae2948f223b1cebc29ed97c5a026ac0b14e2b863645faa2d.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/947cd5b7099023daae2948f223b1cebc29ed97c5a026ac0b14e2b863645faa2d.jpg)

![9b736289f4502bfd9a49c28e86409b19b37610cffeb32718835215dc54434986.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/9b736289f4502bfd9a49c28e86409b19b37610cffeb32718835215dc54434986.jpg)

![a7b4f8ad23cfffd3b8f38014ece53126fe366d265e6bb5d225ec547cad34a940.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/a7b4f8ad23cfffd3b8f38014ece53126fe366d265e6bb5d225ec547cad34a940.jpg)

![c90bfed10043d9bc097752e68376b8973d35b02866dc2b33bfd27f5285f0a45f.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/c90bfed10043d9bc097752e68376b8973d35b02866dc2b33bfd27f5285f0a45f.jpg)

![d70a96782bb56c2777f168bce014668f804f1ca09475282509210b159f01ad44.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/d70a96782bb56c2777f168bce014668f804f1ca09475282509210b159f01ad44.jpg)

![de570a0c630ca291fb7693334c51fb0e46d7a00c183d108bb2419c7088562aba.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/images/de570a0c630ca291fb7693334c51fb0e46d7a00c183d108bb2419c7088562aba.jpg)

### Tables

![165cb4bc739fe90fcb177de00d5a0da0392021c3eeb92037c295378fec196fd6.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/165cb4bc739fe90fcb177de00d5a0da0392021c3eeb92037c295378fec196fd6.jpg)

![1c2ece275446e183d70d13357083b7d078ea671709742cd6e9d4e6827d59242d.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/1c2ece275446e183d70d13357083b7d078ea671709742cd6e9d4e6827d59242d.jpg)

![2fb8c68c0be2ac3bc0766f6d9e3605d2960b0deb31839cb108f666f521c926aa.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/2fb8c68c0be2ac3bc0766f6d9e3605d2960b0deb31839cb108f666f521c926aa.jpg)

![3efa36120feb4ebaaaccf0a833c88d007791a8f6745e18cf19ce82e7db54769a.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/3efa36120feb4ebaaaccf0a833c88d007791a8f6745e18cf19ce82e7db54769a.jpg)

![8298459ae46585e4a227ffb2ed4f213bd5fe85b8dfb518d9b72118d20798528d.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/8298459ae46585e4a227ffb2ed4f213bd5fe85b8dfb518d9b72118d20798528d.jpg)

![83e4ef3d0d01652dca79a50adff837eb128b098aa584488f83c94d016d9cbd2f.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/83e4ef3d0d01652dca79a50adff837eb128b098aa584488f83c94d016d9cbd2f.jpg)

![a2e6dc31b9b6eb70187dfa99ef31d4f5c5fe0741003c220debff3d656b4f77a1.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/a2e6dc31b9b6eb70187dfa99ef31d4f5c5fe0741003c220debff3d656b4f77a1.jpg)

![b023d78849167c2520c211d6cff10f28f0aa442e65b584a8395ce1fb4fc06439.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/b023d78849167c2520c211d6cff10f28f0aa442e65b584a8395ce1fb4fc06439.jpg)

![c28847d40e282d1ebba0dfe851e1200c63276285a45cb766f190c18bad1fa5f1.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/c28847d40e282d1ebba0dfe851e1200c63276285a45cb766f190c18bad1fa5f1.jpg)

![daf565a7ecc81429e3fa5b2e3e40fcb61725b66ebe290d4f8c25074537872305.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/daf565a7ecc81429e3fa5b2e3e40fcb61725b66ebe290d4f8c25074537872305.jpg)

![e8152cc690ca02a8f46e980c3fd2b9fd6d250c40529b8048d00e7f66f95c6ca1.jpg](../icml_results/69_DOUBT_%20Decoupled%20Object-level%20Understanding%20and%20Bridging%20via%20vMF-based%20Trustworthiness%20for%20Hallucina/tables/e8152cc690ca02a8f46e980c3fd2b9fd6d250c40529b8048d00e7f66f95c6ca1.jpg)

<a id="paper-0069"></a>

## Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance


### Images

![087b6ed1b96e17448bb56b2ce8373ff380b734db29923c7190bf6ce1a0282c8f.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/087b6ed1b96e17448bb56b2ce8373ff380b734db29923c7190bf6ce1a0282c8f.jpg)

![204143a7221bb790cb821a7ef759916484cab467efda99ed2427395e50f5975b.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/204143a7221bb790cb821a7ef759916484cab467efda99ed2427395e50f5975b.jpg)

![4226f3a3e070a92c02aaaa999c97fd113c158fa133c0735d4a20edeeec1a62d4.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/4226f3a3e070a92c02aaaa999c97fd113c158fa133c0735d4a20edeeec1a62d4.jpg)

![45a180c94468ad2cfdc71078070ab71e8d63b7eecd842334aed8b393148cb31f.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/45a180c94468ad2cfdc71078070ab71e8d63b7eecd842334aed8b393148cb31f.jpg)

![45dc33717b86fe50785a3e811a2dabab2bc58ec5962307696314c75f6bfd3d7b.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/45dc33717b86fe50785a3e811a2dabab2bc58ec5962307696314c75f6bfd3d7b.jpg)

![869332e00ece97d83d1622a98e06ee8e7aa791ae698423a6032219dd679af47b.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/869332e00ece97d83d1622a98e06ee8e7aa791ae698423a6032219dd679af47b.jpg)

![88aac03fd39a0dbf882b542766b9952cd350ee22cf24e4fdab06a40468c3a54f.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/88aac03fd39a0dbf882b542766b9952cd350ee22cf24e4fdab06a40468c3a54f.jpg)

![a242a9f858ad441bbe566d5e3c5719601ae6c127617591cc870a1db7d42833b9.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/a242a9f858ad441bbe566d5e3c5719601ae6c127617591cc870a1db7d42833b9.jpg)

![c3bffcd90094b85204d684a05e18c047ae3c27f3fed708dbdd4938efa4c971d9.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/c3bffcd90094b85204d684a05e18c047ae3c27f3fed708dbdd4938efa4c971d9.jpg)

![d61948c0dc5980724b05bde16fb422be9d6806f664ecac0bde4bd371e24d02b2.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/d61948c0dc5980724b05bde16fb422be9d6806f664ecac0bde4bd371e24d02b2.jpg)

![d6c34222a2f40cb16d2db3b7e1f6f4e4b5c3928bf430843257bf8aea4cfd0d7a.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/d6c34222a2f40cb16d2db3b7e1f6f4e4b5c3928bf430843257bf8aea4cfd0d7a.jpg)

![ed84a2386881cfd2a3ab2cadb197b485765188bd7bd0681c61eeba6f5baab036.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/ed84a2386881cfd2a3ab2cadb197b485765188bd7bd0681c61eeba6f5baab036.jpg)

![ff61b06fd34c11fa35f5b9f04fdb3300cffb3516cc4500e71a4eb292467d465c.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/images/ff61b06fd34c11fa35f5b9f04fdb3300cffb3516cc4500e71a4eb292467d465c.jpg)

### Tables

![0a3890f6e080e8f35be336ab9f6389e60cde6db40dd02ff1bcfc36a3c36ac423.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/0a3890f6e080e8f35be336ab9f6389e60cde6db40dd02ff1bcfc36a3c36ac423.jpg)

![0c0262cb941843288bb1e4ccb0efb79612b9821fa30c18f3eafc1209f6b37ecf.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/0c0262cb941843288bb1e4ccb0efb79612b9821fa30c18f3eafc1209f6b37ecf.jpg)

![3d55652a755b76ce9d2266f6fcd7e95f63ee85405b630669f043a0e9ce0e4cc9.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/3d55652a755b76ce9d2266f6fcd7e95f63ee85405b630669f043a0e9ce0e4cc9.jpg)

![4ed6c1455cd979d782be7eb1d395deb54ec36b24873bd80e15a47f2498596fed.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/4ed6c1455cd979d782be7eb1d395deb54ec36b24873bd80e15a47f2498596fed.jpg)

![5d9bbf7f45268a95b5bd942accdb90c4d54cf6ba41d471a0eea1b63c9f81fe37.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/5d9bbf7f45268a95b5bd942accdb90c4d54cf6ba41d471a0eea1b63c9f81fe37.jpg)

![5fdf34090748934d1ead2367995dc1027b0692aa9bbd87d9c1c56d4b02b185d6.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/5fdf34090748934d1ead2367995dc1027b0692aa9bbd87d9c1c56d4b02b185d6.jpg)

![82df01e0f0e9621311cc6a1e97d896cb09d5b8aa81ada51cdab48b6d44b5e819.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/82df01e0f0e9621311cc6a1e97d896cb09d5b8aa81ada51cdab48b6d44b5e819.jpg)

![89d93134f5d7535d6252fa0cb2b91d76b87fd19ac54f6566f16079cfd66149bf.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/89d93134f5d7535d6252fa0cb2b91d76b87fd19ac54f6566f16079cfd66149bf.jpg)

![92456c0ef4f660915a708ef4e7fdd244c1df49fe0864f1dad1c4a04857995a6c.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/92456c0ef4f660915a708ef4e7fdd244c1df49fe0864f1dad1c4a04857995a6c.jpg)

![c6e423aebd5171caab6d5d572515ecf96fba27845a27e739a93fef8d9318e9df.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/c6e423aebd5171caab6d5d572515ecf96fba27845a27e739a93fef8d9318e9df.jpg)

![c8f9ac50f68e9e45edcdc4c7ec882d774c3d85532b83068e30a942265b07454e.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/c8f9ac50f68e9e45edcdc4c7ec882d774c3d85532b83068e30a942265b07454e.jpg)

![ca9f9f472a9df2788462b8f5cb42a92d8324764dc184389082df2a00de87d2c9.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/ca9f9f472a9df2788462b8f5cb42a92d8324764dc184389082df2a00de87d2c9.jpg)

![d4f9abf59204547186df7a18ee945caaa34de2de3776df115ddc038dac0e5872.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/d4f9abf59204547186df7a18ee945caaa34de2de3776df115ddc038dac0e5872.jpg)

![f17e3a74c28f02d6e36bbf23f85c15790c5a1e27b0fc822ac27826628ba4f11f.jpg](../icml_results/70_Toward%20Stable%20Value%20Alignment_%20Introducing%20Independent%20Modules%20for%20Consistent%20Value%20Guidance/tables/f17e3a74c28f02d6e36bbf23f85c15790c5a1e27b0fc822ac27826628ba4f11f.jpg)

<a id="paper-0070"></a>

## A Causal Decomposition Approach for Fair Contextual Multi-Armed Bandits


### Images

![3f66f6d57cd0719cade90d78a6af8f7141540bbc6776574df013746b2c026f4a.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/images/3f66f6d57cd0719cade90d78a6af8f7141540bbc6776574df013746b2c026f4a.jpg)

![6e2daa90e870d4d7a3381637b570612cadcbb2e793e2696aa38c652c1be4d956.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/images/6e2daa90e870d4d7a3381637b570612cadcbb2e793e2696aa38c652c1be4d956.jpg)

![986fed5927324e7ec540ac182418a50a1088a36416cffa744826a77c93057be8.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/images/986fed5927324e7ec540ac182418a50a1088a36416cffa744826a77c93057be8.jpg)

![c2c8a64447786d001d6de1037ab0fa0c8865b863d2584a3e69a4c1fde26a4a1f.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/images/c2c8a64447786d001d6de1037ab0fa0c8865b863d2584a3e69a4c1fde26a4a1f.jpg)

### Tables

![26e2de81cfa5ec992f1709abe47eda339ac7d5d080e78893921614b3e7a210cb.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/tables/26e2de81cfa5ec992f1709abe47eda339ac7d5d080e78893921614b3e7a210cb.jpg)

![cc8459db5e9fa975242f8ef5354f60eb6400208bd7037a88b2b3b677f7f2eea1.jpg](../icml_results/71_A%20Causal%20Decomposition%20Approach%20for%20Fair%20Contextual%20Multi-Armed%20Bandits/tables/cc8459db5e9fa975242f8ef5354f60eb6400208bd7037a88b2b3b677f7f2eea1.jpg)

<a id="paper-0071"></a>

## Do LLMs Signal When They’re Right? Evidence from Neuron Agreement


### Images

![0059b0d2fafa70dffb28adde86dc9685dcf7bf4796be19c7e15355f058339e58.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/0059b0d2fafa70dffb28adde86dc9685dcf7bf4796be19c7e15355f058339e58.jpg)

![38b0db95564af3216cf21205d9851a1827bcc6a5614e27850593077f17b9c958.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/38b0db95564af3216cf21205d9851a1827bcc6a5614e27850593077f17b9c958.jpg)

![477115ea7ef29eb42008b4dce15b8a2cae03a30244e23d3610af5326ff071d53.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/477115ea7ef29eb42008b4dce15b8a2cae03a30244e23d3610af5326ff071d53.jpg)

![4859a330930f1df54bc11ed8ad8a1f4b524f625c26d7d1a04d0367ae8045d2dd.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/4859a330930f1df54bc11ed8ad8a1f4b524f625c26d7d1a04d0367ae8045d2dd.jpg)

![4afd0e7cf060a77a7a10113ebc349b5538985238a78673a75826350ab4087486.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/4afd0e7cf060a77a7a10113ebc349b5538985238a78673a75826350ab4087486.jpg)

![4c1cdaa3ae9d7fbebcc4408cda42f8ae8583ea3cb62d3884d010a6f3cb52ddd8.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/4c1cdaa3ae9d7fbebcc4408cda42f8ae8583ea3cb62d3884d010a6f3cb52ddd8.jpg)

![5153099e86466dc42674da5a4603a135e79a373f8897d8b94695fcbeecef4dc9.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/5153099e86466dc42674da5a4603a135e79a373f8897d8b94695fcbeecef4dc9.jpg)

![6042e023c35a8d1d68fd7ba1b3b2b906037e3b21b06b682dbe869b442066ec8f.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/6042e023c35a8d1d68fd7ba1b3b2b906037e3b21b06b682dbe869b442066ec8f.jpg)

![72f600812551a81fedb37661703efc0b7af2f16af57e444002a0247008700130.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/72f600812551a81fedb37661703efc0b7af2f16af57e444002a0247008700130.jpg)

![b161ae51239d6694ea1e4a9ba83095bf4ac168fe144805723849972927c8f2aa.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/images/b161ae51239d6694ea1e4a9ba83095bf4ac168fe144805723849972927c8f2aa.jpg)

### Tables

![908f6cc9f4f5468960bb456d39c32ee39e547f722a2d37f9fc17dd760e17e614.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/908f6cc9f4f5468960bb456d39c32ee39e547f722a2d37f9fc17dd760e17e614.jpg)

![932cf73a3e5ec691d1bd6220e1c8fa016b80d7553fe2d717a447ccf3c65c8f6b.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/932cf73a3e5ec691d1bd6220e1c8fa016b80d7553fe2d717a447ccf3c65c8f6b.jpg)

![96b8e771783588a3e46fa3f1f08cdbd0cdc07448598cc37d720acdae6e592386.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/96b8e771783588a3e46fa3f1f08cdbd0cdc07448598cc37d720acdae6e592386.jpg)

![a512f968be2977eed9e9a4a9389bf511ded684a54fc3c75f353e60e562be2caa.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/a512f968be2977eed9e9a4a9389bf511ded684a54fc3c75f353e60e562be2caa.jpg)

![b1cd0c386e0889cc80389918dadd2e2d891404024b9c16e4a1ded757f38f273e.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/b1cd0c386e0889cc80389918dadd2e2d891404024b9c16e4a1ded757f38f273e.jpg)

![b56f56e4a8e8062c6dfd0ef470946ba5a21f07082ea4f23b10a9271f7628f3fc.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/b56f56e4a8e8062c6dfd0ef470946ba5a21f07082ea4f23b10a9271f7628f3fc.jpg)

![e7b19501c92d474231c32f0d54686937fe20e39a175df25ade8f438a0b6d09d7.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/e7b19501c92d474231c32f0d54686937fe20e39a175df25ade8f438a0b6d09d7.jpg)

![fa7a0b0dd3f220e380453eecc36f7dd5269cdb8faf85091c21300072dfa33fc4.jpg](../icml_results/72_Do%20LLMs%20Signal%20When%20They%E2%80%99re%20Right_%20Evidence%20from%20Neuron%20Agreement/tables/fa7a0b0dd3f220e380453eecc36f7dd5269cdb8faf85091c21300072dfa33fc4.jpg)

<a id="paper-0072"></a>

## Real-World Unsupervised Models Generalize to Predict Brain Responses to Out-of-Distribution Stimuli


### Images

![18e344c9487fd89d4cef38667b42a6e482ba99c1205430df48548d8c343a037f.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/18e344c9487fd89d4cef38667b42a6e482ba99c1205430df48548d8c343a037f.jpg)

![28ea9711a509d20d6456f6aaee38cfbd37a5346574f7c487496825f7e4838be9.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/28ea9711a509d20d6456f6aaee38cfbd37a5346574f7c487496825f7e4838be9.jpg)

![3751b8b447ee265b945efa45c640dbfa6a3afc1fa3efd6f1ee141fe17081aa1b.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/3751b8b447ee265b945efa45c640dbfa6a3afc1fa3efd6f1ee141fe17081aa1b.jpg)

![54abcc80e059f608e8599cb2aa631fd852f9b70388b73c1e01e8d715f7f866aa.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/54abcc80e059f608e8599cb2aa631fd852f9b70388b73c1e01e8d715f7f866aa.jpg)

![60fc7ea797a53c4cf5d67aa039e6459348cbeaa3a7efc25fe668c982ba61e21c.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/60fc7ea797a53c4cf5d67aa039e6459348cbeaa3a7efc25fe668c982ba61e21c.jpg)

![7a4301867487afe605a999ea7d728e1c0136d90fd7072beace6ae9670d1ce5bc.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/7a4301867487afe605a999ea7d728e1c0136d90fd7072beace6ae9670d1ce5bc.jpg)

![8b62555feb0ad1be6b1f3bebee56448666ad9fa75474a6a4e3a17b4810f60fb8.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/8b62555feb0ad1be6b1f3bebee56448666ad9fa75474a6a4e3a17b4810f60fb8.jpg)

![910df660119594768014e9f5635946a94aed47922ce364d1fe6026a31967d001.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/910df660119594768014e9f5635946a94aed47922ce364d1fe6026a31967d001.jpg)

![98b8e49dce3fe43dd967ba06d5c7487533137398c176c500964d0a6fa23150ce.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/98b8e49dce3fe43dd967ba06d5c7487533137398c176c500964d0a6fa23150ce.jpg)

![9bd01beae047adb3f5f4c8c2038ccbb1b34b6772c89504c6b492982ae62bd2dc.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/9bd01beae047adb3f5f4c8c2038ccbb1b34b6772c89504c6b492982ae62bd2dc.jpg)

![a2affcd465440859afa3d351aa6d7df0e39d276b832e4f20cf58baf313ebb6f8.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/a2affcd465440859afa3d351aa6d7df0e39d276b832e4f20cf58baf313ebb6f8.jpg)

![a51ce48f9a613b00ccb016caa30053cb27f9351f9a374c648a4aa8efc811515e.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/a51ce48f9a613b00ccb016caa30053cb27f9351f9a374c648a4aa8efc811515e.jpg)

![b6364af9869c3071d01f72d444dc1aca4b30572952d8bad13558948f9107ebd7.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/b6364af9869c3071d01f72d444dc1aca4b30572952d8bad13558948f9107ebd7.jpg)

![bc8670f13498e488408294c3854a40629f5e6b98bfeaf7a2a202a05f0afac8b2.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/bc8670f13498e488408294c3854a40629f5e6b98bfeaf7a2a202a05f0afac8b2.jpg)

![bdf0eaf6dba7f48eb89a30c5038f205564d51f7ca408f8a6f3f4e16f2dd6c1d8.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/bdf0eaf6dba7f48eb89a30c5038f205564d51f7ca408f8a6f3f4e16f2dd6c1d8.jpg)

![d139152bbd5a93d82494eb9e2a6e837049b7039e8a9c12536f558d17c705ee9a.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/d139152bbd5a93d82494eb9e2a6e837049b7039e8a9c12536f558d17c705ee9a.jpg)

![e94dbcba9cf3e98d5a3dbe0129aaebae57a612c4d53cfbb9f9ed6959dccb8e05.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/images/e94dbcba9cf3e98d5a3dbe0129aaebae57a612c4d53cfbb9f9ed6959dccb8e05.jpg)

### Tables

![023e7205e882b7a2166688a70e9f624feedadbfa1cfc14ccb09af6984153fcc1.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/tables/023e7205e882b7a2166688a70e9f624feedadbfa1cfc14ccb09af6984153fcc1.jpg)

![a30344e26c737b7f3f75dab8f4ac8fcb7126d5a26a4692a2da0a0a4a133c2f8b.jpg](../icml_results/73_Real-World%20Unsupervised%20Models%20Generalize%20to%20Predict%20Brain%20Responses%20to%20Out-of-Distribution%20Stimuli/tables/a30344e26c737b7f3f75dab8f4ac8fcb7126d5a26a4692a2da0a0a4a133c2f8b.jpg)

<a id="paper-0073"></a>

## Scaling Law for Quantization-Aware Training


### Images

![122379639cc3eb06c11698d21bcb007fd270830a47730bbf82753ca185ffdb14.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/122379639cc3eb06c11698d21bcb007fd270830a47730bbf82753ca185ffdb14.jpg)

![1cb5f95362758fcbda684b2b6ae1f8810fdbcced8f4838acb2c2cd84fddc9ec5.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/1cb5f95362758fcbda684b2b6ae1f8810fdbcced8f4838acb2c2cd84fddc9ec5.jpg)

![3a3c1659e50f87232c0ad683c8a91dd054d75d91be7434e0d284d7aeeb36bf04.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/3a3c1659e50f87232c0ad683c8a91dd054d75d91be7434e0d284d7aeeb36bf04.jpg)

![41f501d5a80d7e2c135b7fa2bbd3fb594df2cdcdb549a6db04bfc0321527b81c.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/41f501d5a80d7e2c135b7fa2bbd3fb594df2cdcdb549a6db04bfc0321527b81c.jpg)

![7b052578a833720d17309789b60b175472f894dbe41b758c05dfe6123fbaef9b.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/7b052578a833720d17309789b60b175472f894dbe41b758c05dfe6123fbaef9b.jpg)

![7c38f50e90a0a85843f42cd32b24870b7770cf893fe11eaa8e2649336368809c.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/7c38f50e90a0a85843f42cd32b24870b7770cf893fe11eaa8e2649336368809c.jpg)

![84fbb7147cbc32637a10a19e1a657c93d95d0e7bd2d409b878d4211efdb98064.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/84fbb7147cbc32637a10a19e1a657c93d95d0e7bd2d409b878d4211efdb98064.jpg)

![90ef9a1288d3ae0faf302d38c7ee6877c4bcdf111ae5965aede06518fad02b66.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/90ef9a1288d3ae0faf302d38c7ee6877c4bcdf111ae5965aede06518fad02b66.jpg)

![92e4dccf7534e64abdf23a7fe7a2621fb3651cf0a9ce7e1555977904c9c652f7.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/92e4dccf7534e64abdf23a7fe7a2621fb3651cf0a9ce7e1555977904c9c652f7.jpg)

![9c9a81855b683b2165d38fadb876a9eb76c6e261d460be31465db837064565ca.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/9c9a81855b683b2165d38fadb876a9eb76c6e261d460be31465db837064565ca.jpg)

![b754ef3a3a25988e5299626ea50e2573d110ab05f974319d16528f334097c8a2.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/b754ef3a3a25988e5299626ea50e2573d110ab05f974319d16528f334097c8a2.jpg)

![ce25ea096a64447d996cb13ad4165beddf59dc552bc58ab843ba05e27cb55303.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/ce25ea096a64447d996cb13ad4165beddf59dc552bc58ab843ba05e27cb55303.jpg)

![db0de9c08b9fc77eac7b711dbff231e172fb20f8451e092fca87de24b0ec202e.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/db0de9c08b9fc77eac7b711dbff231e172fb20f8451e092fca87de24b0ec202e.jpg)

![e13cbcc7577106f412be4dc2d6e7640c6c9014d50081631b70586474bf0efb4e.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/e13cbcc7577106f412be4dc2d6e7640c6c9014d50081631b70586474bf0efb4e.jpg)

![e781cfd6bc1ec07d3fe8be1f863821ebd596205d89a517439b7c7ab1ddf790b0.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/e781cfd6bc1ec07d3fe8be1f863821ebd596205d89a517439b7c7ab1ddf790b0.jpg)

![e95fd5b8140678dba5afcbc956a8729261a7ba0f1d0eab40f386f5178f480b0c.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/e95fd5b8140678dba5afcbc956a8729261a7ba0f1d0eab40f386f5178f480b0c.jpg)

![f05ae9ec1eef5086f1eeeb65b2ed39805a1e93125d4337494fff3ddccbbead19.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/images/f05ae9ec1eef5086f1eeeb65b2ed39805a1e93125d4337494fff3ddccbbead19.jpg)

### Tables

![0faa0425efc1c644a973f8a169f2ba9edee673781e640155f5e75e4a6452859c.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/0faa0425efc1c644a973f8a169f2ba9edee673781e640155f5e75e4a6452859c.jpg)

![16485da05440e80b9321a1fc4bc288ac19a39f14f4dfb84a7d4f33a409dec3bd.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/16485da05440e80b9321a1fc4bc288ac19a39f14f4dfb84a7d4f33a409dec3bd.jpg)

![1e2fb69935d831bba1680c751556a3dd4c4a6a8e7fcabe4bbd1e64e7d1a88590.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/1e2fb69935d831bba1680c751556a3dd4c4a6a8e7fcabe4bbd1e64e7d1a88590.jpg)

![3a9da261b158573cbcc35f2d550eec2c9386c589f071707c0df49a6c1d52b9c2.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/3a9da261b158573cbcc35f2d550eec2c9386c589f071707c0df49a6c1d52b9c2.jpg)

![6b96b3539d5cf6948941920caa78c57a608235bd95dd41c7d99a69f99fb110d2.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/6b96b3539d5cf6948941920caa78c57a608235bd95dd41c7d99a69f99fb110d2.jpg)

![c68711ad79e12c47f9ac92558360ca171c12977fa1f69f07baebee7c775f1ec0.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/c68711ad79e12c47f9ac92558360ca171c12977fa1f69f07baebee7c775f1ec0.jpg)

![f0a3f97dd764d9baadc139d346e41a57687d60697d201b1e47c5c614990d85bb.jpg](../icml_results/74_Scaling%20Law%20for%20Quantization-Aware%20Training/tables/f0a3f97dd764d9baadc139d346e41a57687d60697d201b1e47c5c614990d85bb.jpg)

<a id="paper-0074"></a>

## Towards Optimal Robustness in Learning-Augmented Paging


### Images

![02c1c86d861ecead6c3299c6f36f235dec67790c4df6ca4122eb18c410f5023c.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/images/02c1c86d861ecead6c3299c6f36f235dec67790c4df6ca4122eb18c410f5023c.jpg)

![c5cf197ef24fe3a56a56df8d86d011267bb3a2e1a2c617e5c6a171ad5baafa86.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/images/c5cf197ef24fe3a56a56df8d86d011267bb3a2e1a2c617e5c6a171ad5baafa86.jpg)

### Tables

![0927a6d3cd58cf3e3031bcf22fc7b9f41b473ff1d70664d8a864989b0c36cc8d.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/0927a6d3cd58cf3e3031bcf22fc7b9f41b473ff1d70664d8a864989b0c36cc8d.jpg)

![0fc60aa7922dfd130b7a14763fdb58a087278c5e6efdcdcf8111ada0a69a328c.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/0fc60aa7922dfd130b7a14763fdb58a087278c5e6efdcdcf8111ada0a69a328c.jpg)

![2bd1e6ac2e11ec64a07f69959dc4edd0c79ca4e29b253451ab4c2ac0baebc030.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/2bd1e6ac2e11ec64a07f69959dc4edd0c79ca4e29b253451ab4c2ac0baebc030.jpg)

![3bcf861529434c2dddf24bef8c726c80b8e191ea7cbc9dd67d94cb50952c15f2.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/3bcf861529434c2dddf24bef8c726c80b8e191ea7cbc9dd67d94cb50952c15f2.jpg)

![5a68a43d57b0d049670247ecdc3e02adfb91baaca0b743e3da2cad7b3a5609e4.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/5a68a43d57b0d049670247ecdc3e02adfb91baaca0b743e3da2cad7b3a5609e4.jpg)

![73c5de47d8fd1d3512789d71d3893bf7bf66d970c8cb799895ca1a1e8ae0c89e.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/73c5de47d8fd1d3512789d71d3893bf7bf66d970c8cb799895ca1a1e8ae0c89e.jpg)

![7c379d0ffcc07eb8bc2dc2a5d30169270923cfaaa84cc6642da44de2ba29ab8a.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/7c379d0ffcc07eb8bc2dc2a5d30169270923cfaaa84cc6642da44de2ba29ab8a.jpg)

![7e720f040d8025f0ada07a62e2b942c7fe99679f0bf439bfcd5581c10423cab7.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/7e720f040d8025f0ada07a62e2b942c7fe99679f0bf439bfcd5581c10423cab7.jpg)

![8ed4f4d8b59ff2bbde7065e964af912b9a1569bbd8557a5444780b3e382df9af.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/8ed4f4d8b59ff2bbde7065e964af912b9a1569bbd8557a5444780b3e382df9af.jpg)

![8fa4503619321031e0ada9076899ec5b7b084888ee325206cb0281e805d407e8.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/8fa4503619321031e0ada9076899ec5b7b084888ee325206cb0281e805d407e8.jpg)

![ad0ce3ee0b0659eefee8532b1dcd2c4575569c920f8a778c3039a1af5f5bfa9f.jpg](../icml_results/75_Towards%20Optimal%20Robustness%20in%20Learning-Augmented%20Paging/tables/ad0ce3ee0b0659eefee8532b1dcd2c4575569c920f8a778c3039a1af5f5bfa9f.jpg)

<a id="paper-0075"></a>

## Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating


### Images

![201df70a4b32ad2dc7f0f9a7af288d4529d716b2b505322130c0af87f0cbd757.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/201df70a4b32ad2dc7f0f9a7af288d4529d716b2b505322130c0af87f0cbd757.jpg)

![214e2f65f2b70a5a23d85b2b04ec41a34a579b2ad801db13add5eab5f57885f2.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/214e2f65f2b70a5a23d85b2b04ec41a34a579b2ad801db13add5eab5f57885f2.jpg)

![2bb65403871d27db8a8afbf9e3b0cc444a62a8cf757fc3d281c852ad2a173de3.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/2bb65403871d27db8a8afbf9e3b0cc444a62a8cf757fc3d281c852ad2a173de3.jpg)

![2ee309b6611758b0d51757db761e48dbc9cb4140cdc28b24948025cd27ab46e3.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/2ee309b6611758b0d51757db761e48dbc9cb4140cdc28b24948025cd27ab46e3.jpg)

![37a00e970d4da25751a483048b386d2d807db19bd088f7eb97e763b6d5d0f95b.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/37a00e970d4da25751a483048b386d2d807db19bd088f7eb97e763b6d5d0f95b.jpg)

![3f52379f764d7d72f53f05cd750cd85662d3487babca2158a9335b1aba9885e3.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/3f52379f764d7d72f53f05cd750cd85662d3487babca2158a9335b1aba9885e3.jpg)

![6f92fe207b1fde0b40768b4b0e82e0e4a269fdb14a264111b326dde5074469ef.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/6f92fe207b1fde0b40768b4b0e82e0e4a269fdb14a264111b326dde5074469ef.jpg)

![97210d5c957c0f17e7f83c8c2f9cf27dab3819b55ec4ce579d77eba5c85acf53.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/97210d5c957c0f17e7f83c8c2f9cf27dab3819b55ec4ce579d77eba5c85acf53.jpg)

![ab5220c7183c8e3cfc78c1040757ae9fface094873d61b5f3af117059856c0f3.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/ab5220c7183c8e3cfc78c1040757ae9fface094873d61b5f3af117059856c0f3.jpg)

![aff2079067214e64e217f09b5f48ffe39f29483ad1e9fecb6b6afe11a606d911.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/aff2079067214e64e217f09b5f48ffe39f29483ad1e9fecb6b6afe11a606d911.jpg)

![d44030b83a88afc2ea7b0a0641fbe7e0d73c2b4d88287a272682a71cf15c80b2.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/images/d44030b83a88afc2ea7b0a0641fbe7e0d73c2b4d88287a272682a71cf15c80b2.jpg)

### Tables

![06c043236faee6b8ee82fa8ecbfef5424c99c3c328b8dae8f7a36a41dc767ede.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/06c043236faee6b8ee82fa8ecbfef5424c99c3c328b8dae8f7a36a41dc767ede.jpg)

![081d39052600493e72af43ac50ee43b93219227c4d186031a5fbcb38ac33efad.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/081d39052600493e72af43ac50ee43b93219227c4d186031a5fbcb38ac33efad.jpg)

![3bed035b7903dfaae730342412dcf62bf12225b56a746290987786c2e894b698.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/3bed035b7903dfaae730342412dcf62bf12225b56a746290987786c2e894b698.jpg)

![46572e2b9eaea41527c8faa107da5384eac7e172158fc8a7d5610d15c36d204d.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/46572e2b9eaea41527c8faa107da5384eac7e172158fc8a7d5610d15c36d204d.jpg)

![4931a2b750c8aef4f7df1cb2fa7fd43de2f36a95d382903c537840d9a8daacd3.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/4931a2b750c8aef4f7df1cb2fa7fd43de2f36a95d382903c537840d9a8daacd3.jpg)

![590bc878c8fbdb8a45231042e3391ec3f82857c037551d305555d209981950df.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/590bc878c8fbdb8a45231042e3391ec3f82857c037551d305555d209981950df.jpg)

![5f65f67cf0e269e5f9cc24368c00019047dfb0d7e5aea20840f679a12ef2cd23.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/5f65f67cf0e269e5f9cc24368c00019047dfb0d7e5aea20840f679a12ef2cd23.jpg)

![61bb4a4454949a7a4088080bb2a7c87104f71e3c4c76d0fe9344c024d641be8c.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/61bb4a4454949a7a4088080bb2a7c87104f71e3c4c76d0fe9344c024d641be8c.jpg)

![65a0269bec694caf322298264b9aafa5166ae998f2b59a04add83ac456c84aff.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/65a0269bec694caf322298264b9aafa5166ae998f2b59a04add83ac456c84aff.jpg)

![7b9d40b9adfdc01a6d3bc0745b3867864ba8739126abe7d9886c99303d4528a1.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/7b9d40b9adfdc01a6d3bc0745b3867864ba8739126abe7d9886c99303d4528a1.jpg)

![7ece01445cc36729babc12c8da56d1bf2418dd3d208cba8c0d2e15a9b1015440.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/7ece01445cc36729babc12c8da56d1bf2418dd3d208cba8c0d2e15a9b1015440.jpg)

![8e11db133a7ccc25a34874c1941e3006e317ba5d6f44f64593d8c251b2baa44f.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/8e11db133a7ccc25a34874c1941e3006e317ba5d6f44f64593d8c251b2baa44f.jpg)

![945d7325179b6eb41534d51fac82c6938793fded7a177ec49790d871301584e1.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/945d7325179b6eb41534d51fac82c6938793fded7a177ec49790d871301584e1.jpg)

![9802d914cc0f2f9e12f480045843aebd8a96106aea00e3ca0b8c5627cefb0aca.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/9802d914cc0f2f9e12f480045843aebd8a96106aea00e3ca0b8c5627cefb0aca.jpg)

![9a261a7d4abbf099f71419d11e3f5c8a1b318a5b2c1e72392325fbe92f1070c1.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/9a261a7d4abbf099f71419d11e3f5c8a1b318a5b2c1e72392325fbe92f1070c1.jpg)

![9a837c588b2eb8f6b47996153b1e192d74bd5bb314b38ae3966dbbecdd043231.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/9a837c588b2eb8f6b47996153b1e192d74bd5bb314b38ae3966dbbecdd043231.jpg)

![bd557b6e980c5704b99e40b3484b711371dea8a6e17719122f7b9fc8df5dbffd.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/bd557b6e980c5704b99e40b3484b711371dea8a6e17719122f7b9fc8df5dbffd.jpg)

![c983108a71af042d3ab7abb7edd92b4bc23dd94172cca593f92d1c7f4f12e3df.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/c983108a71af042d3ab7abb7edd92b4bc23dd94172cca593f92d1c7f4f12e3df.jpg)

![d5267ba036c6dfab2ad1d2e7ef28f2d85ba6a11640f35cf9b30f2e21117e3e70.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/d5267ba036c6dfab2ad1d2e7ef28f2d85ba6a11640f35cf9b30f2e21117e3e70.jpg)

![f68460aadcd50b1104a69bc39d79be00dc6e05860655ec0260892992a2e7ddbb.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/f68460aadcd50b1104a69bc39d79be00dc6e05860655ec0260892992a2e7ddbb.jpg)

![ff330b3b0ad9b372711544aff043d1b8eb4b97dfeecb422e2c710ae73eb8d438.jpg](../icml_results/76_Mitigating%20Hallucinations%20in%20Large%20Vision-Language%20Models%20via%20Causal%20Route%20Gating/tables/ff330b3b0ad9b372711544aff043d1b8eb4b97dfeecb422e2c710ae73eb8d438.jpg)

<a id="paper-0076"></a>

## Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization


### Images

![1f8fd43bcc63034fcc7261a95f17266fe912023063d6c6f77ea02d96664cce56.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/1f8fd43bcc63034fcc7261a95f17266fe912023063d6c6f77ea02d96664cce56.jpg)

![30c11eb35a757a74e0fae59df11258a81fd19bcadca61e2c45f8ea281156740c.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/30c11eb35a757a74e0fae59df11258a81fd19bcadca61e2c45f8ea281156740c.jpg)

![8473e315a88b1d87ed08825ba227b30e60dd080f63ab5817045f634177bcc766.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/8473e315a88b1d87ed08825ba227b30e60dd080f63ab5817045f634177bcc766.jpg)

![85698571e09ac7d14c89b9b8bbd3da0c12c57f2dcb0fbfd4db11be9b08fe27d7.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/85698571e09ac7d14c89b9b8bbd3da0c12c57f2dcb0fbfd4db11be9b08fe27d7.jpg)

![c1659ddf3595b7a232367321ed280a93295a199dabe6a1b00898278f40a1257e.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/c1659ddf3595b7a232367321ed280a93295a199dabe6a1b00898278f40a1257e.jpg)

![d2f906f605d56510738c3833f34fddd9ac7ffe1c9f81bb4f06a2d4996e35ce52.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/d2f906f605d56510738c3833f34fddd9ac7ffe1c9f81bb4f06a2d4996e35ce52.jpg)

![eb5521c1e4adf12bf41ed0928d84c168aec3a49f5b08f1569d55871d679e2823.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/eb5521c1e4adf12bf41ed0928d84c168aec3a49f5b08f1569d55871d679e2823.jpg)

![edd986e5ea4fb478fa5573dd1965a39cf96e6f338fc2ef0042c62076b9d291c4.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/images/edd986e5ea4fb478fa5573dd1965a39cf96e6f338fc2ef0042c62076b9d291c4.jpg)

### Tables

![2c148f8209883f1f02e06231f432355f83e3e38f47aa2d9a7a4d6afe607c0ac8.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/tables/2c148f8209883f1f02e06231f432355f83e3e38f47aa2d9a7a4d6afe607c0ac8.jpg)

![71facaa86fcaf8de3d099cfba723c2bc42203a22687c0393318cb88a580b0143.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/tables/71facaa86fcaf8de3d099cfba723c2bc42203a22687c0393318cb88a580b0143.jpg)

![799e32b2878bb259f65d216e2e38dddfbec2c4dee3e7efcbed9dce7c50a07eef.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/tables/799e32b2878bb259f65d216e2e38dddfbec2c4dee3e7efcbed9dce7c50a07eef.jpg)

![a8d052014ec03c591cf9e722665a1f4bca899227ad7ff51cca7ca10fd7a3246f.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/tables/a8d052014ec03c591cf9e722665a1f4bca899227ad7ff51cca7ca10fd7a3246f.jpg)

![b819fda5e09debefbb056a692eef364a7ac4911daa3534a72ae89736966cf855.jpg](../icml_results/77_Local%20Redundancy_%20An%20Information-Theoretic%20Measure%20of%20Plasticity%20from%20Synthetic%20Memorization/tables/b819fda5e09debefbb056a692eef364a7ac4911daa3534a72ae89736966cf855.jpg)

<a id="paper-0077"></a>

## Unifying and Optimizing Data Values for Selection via Sequential Decision-Making


### Images

![016841db5e9d74595edae836ec00616b1b21f62218ab06938d20e662b4139ea9.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/images/016841db5e9d74595edae836ec00616b1b21f62218ab06938d20e662b4139ea9.jpg)

![061aeaa84926a6336605dd63e1d21fdc59ce5dbc8dad5802506ceeb4482a1d54.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/images/061aeaa84926a6336605dd63e1d21fdc59ce5dbc8dad5802506ceeb4482a1d54.jpg)

![cb6fb3082c5c28c0472bcea24dab8dfadf4ed4b0236d246ccf469f6207e491cc.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/images/cb6fb3082c5c28c0472bcea24dab8dfadf4ed4b0236d246ccf469f6207e491cc.jpg)

![cfeeb35640065b5d094b4d265377473e438e85eca56474288320425a6c66db1b.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/images/cfeeb35640065b5d094b4d265377473e438e85eca56474288320425a6c66db1b.jpg)

### Tables

![00629e3776c218b4fb3e8580ec2f382bda965599fb8ee28bb6400b84cdf3b36d.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/00629e3776c218b4fb3e8580ec2f382bda965599fb8ee28bb6400b84cdf3b36d.jpg)

![3156bf71ae9cb09c20121edde6bd39dfa6f55060fcda6d8fc37e177cba9e63f4.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/3156bf71ae9cb09c20121edde6bd39dfa6f55060fcda6d8fc37e177cba9e63f4.jpg)

![3f67645a3e04073b1b0f6eb9fe811b80a310ec307cb0a61644e6303f38d49768.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/3f67645a3e04073b1b0f6eb9fe811b80a310ec307cb0a61644e6303f38d49768.jpg)

![4b6abbc36013c5b2deabd03de98f32e59064ce560e0fbee9dd8262098a5c9899.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/4b6abbc36013c5b2deabd03de98f32e59064ce560e0fbee9dd8262098a5c9899.jpg)

![97699fb6703845a123a913454da094bf41f59f6b328ebe58d59b1468efd2a7c5.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/97699fb6703845a123a913454da094bf41f59f6b328ebe58d59b1468efd2a7c5.jpg)

![ad5940356cf1b9d85e9d62dbd38db9bc528145780967f58663e382e3fab9260d.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/ad5940356cf1b9d85e9d62dbd38db9bc528145780967f58663e382e3fab9260d.jpg)

![b016c6026d05d3bf9e484cf4843d8fda4b0363d3152682d1e7cc4e77edd319b2.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/b016c6026d05d3bf9e484cf4843d8fda4b0363d3152682d1e7cc4e77edd319b2.jpg)

![e2c8271398625dee437dfd3c74ce5e6d40296ac7029d49c29f98f4cd54ad7afa.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/e2c8271398625dee437dfd3c74ce5e6d40296ac7029d49c29f98f4cd54ad7afa.jpg)

![f651345cc6875d965faf6a6d9746497ddc902ed44d8cbb0278dff1160af21a05.jpg](../icml_results/78_Unifying%20and%20Optimizing%20Data%20Values%20for%20Selection%20via%20Sequential%20Decision-Making/tables/f651345cc6875d965faf6a6d9746497ddc902ed44d8cbb0278dff1160af21a05.jpg)

<a id="paper-0078"></a>

## Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms


### Images

![1692d38ec542e2da40fde9c3d55113eb5f5a4b09a195a08cc3e59336b652c37a.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/1692d38ec542e2da40fde9c3d55113eb5f5a4b09a195a08cc3e59336b652c37a.jpg)

![332ad99828741bbbcbd591897600f21e2da54af484a1a942c971e475718fd683.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/332ad99828741bbbcbd591897600f21e2da54af484a1a942c971e475718fd683.jpg)

![383bf5fc1008aa7450a1ceb149461b708d6fcbf800f78c4054f4b46c944fe27d.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/383bf5fc1008aa7450a1ceb149461b708d6fcbf800f78c4054f4b46c944fe27d.jpg)

![3ca8e92e5eebadcff0c4ddd84b317b6ff64944e8d8e07ace9cfe746b37cc9c53.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/3ca8e92e5eebadcff0c4ddd84b317b6ff64944e8d8e07ace9cfe746b37cc9c53.jpg)

![61ccb26ee70dbfa98cca1b6f331e1d5bea9a5b99dd7b639b66799a6c217b94b6.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/61ccb26ee70dbfa98cca1b6f331e1d5bea9a5b99dd7b639b66799a6c217b94b6.jpg)

![6316ad4f8eae891bae939ad793ee22a382dfd345eb1c10883a3b821f14da76dd.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/6316ad4f8eae891bae939ad793ee22a382dfd345eb1c10883a3b821f14da76dd.jpg)

![6cac91993403effcd5e376122d5b7fb906e4c6dea113c213419e86b3597e75d9.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/6cac91993403effcd5e376122d5b7fb906e4c6dea113c213419e86b3597e75d9.jpg)

![7e33d9c3a9430f033bf7acb8f1ecc34c7d791321638f8c0d3176e3a4e5ba64ad.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/7e33d9c3a9430f033bf7acb8f1ecc34c7d791321638f8c0d3176e3a4e5ba64ad.jpg)

![7fca71b8c474b7c76a69768fd4ef6ff83d4e7215b4ec521e2acf3e916730050d.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/7fca71b8c474b7c76a69768fd4ef6ff83d4e7215b4ec521e2acf3e916730050d.jpg)

![83f20268083738f59bf5512917d9817987d697871d3398bee35a62ad085eb29f.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/83f20268083738f59bf5512917d9817987d697871d3398bee35a62ad085eb29f.jpg)

![bab48c2eccf96739d82ecae50670acfdc87c1bb38623a00307aa207e6143eac6.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/bab48c2eccf96739d82ecae50670acfdc87c1bb38623a00307aa207e6143eac6.jpg)

![d0210d332148015dfe3da9e7713d83884300c2b683ad63dff70044a577977bee.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/d0210d332148015dfe3da9e7713d83884300c2b683ad63dff70044a577977bee.jpg)

![dcf25a7b976c974bea7830f01c7059b630c187b694242dcc101efaecf114230c.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/dcf25a7b976c974bea7830f01c7059b630c187b694242dcc101efaecf114230c.jpg)

![e1ec606e986762022692035b21545553015a50ae5ed938bdaa4bb69a76883af1.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/e1ec606e986762022692035b21545553015a50ae5ed938bdaa4bb69a76883af1.jpg)

![f003d492ca1396c117a803d2f4da649de1278eee8fa3f8af030278cbd45abd52.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/f003d492ca1396c117a803d2f4da649de1278eee8fa3f8af030278cbd45abd52.jpg)

![f307a18b8fc227b6acc031306e58bd12bce8f42cf4f14d41e720c1846a8a0c76.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/images/f307a18b8fc227b6acc031306e58bd12bce8f42cf4f14d41e720c1846a8a0c76.jpg)

### Tables

![0620151108b9f6952a1e192b4f75fc1f90860da1896eb8f181217f5ae4248c36.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/0620151108b9f6952a1e192b4f75fc1f90860da1896eb8f181217f5ae4248c36.jpg)

![0832bc87f95d1cee3f825b3d72bd0505aa5e0dc25b7ae50b341d49d06685ff0f.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/0832bc87f95d1cee3f825b3d72bd0505aa5e0dc25b7ae50b341d49d06685ff0f.jpg)

![0a14fef883c7e2518431795575318d9299a460b83597481747c38415198cd457.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/0a14fef883c7e2518431795575318d9299a460b83597481747c38415198cd457.jpg)

![2ba9e9628b61c5a8cfc733b7712955ad9fc39cafd26a6958a3a949810e7bbc71.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/2ba9e9628b61c5a8cfc733b7712955ad9fc39cafd26a6958a3a949810e7bbc71.jpg)

![33c54bbb8ed0363e654eb2d4631ebd3391fcbe35a9fe7497fb19451cee15cb54.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/33c54bbb8ed0363e654eb2d4631ebd3391fcbe35a9fe7497fb19451cee15cb54.jpg)

![34a7d87434b55c8931243bd5d65a1f5d3ab0ac66bb49a2a4f10006808fedf535.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/34a7d87434b55c8931243bd5d65a1f5d3ab0ac66bb49a2a4f10006808fedf535.jpg)

![4a3e82e4a40f7250721b7843eb19edaf333fc1286d5ee9b1f3b08dc67e5d7e7c.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/4a3e82e4a40f7250721b7843eb19edaf333fc1286d5ee9b1f3b08dc67e5d7e7c.jpg)

![50ca5c99432c3b3e862d5c42bff7e26d16805d78b6e1f739eb601ef106721d80.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/50ca5c99432c3b3e862d5c42bff7e26d16805d78b6e1f739eb601ef106721d80.jpg)

![5cd66901733fe332804074200a64199160d5eb03fac64c9823de446e8f64d2e1.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/5cd66901733fe332804074200a64199160d5eb03fac64c9823de446e8f64d2e1.jpg)

![6b22c5b126dde59f950039bed641d64e24c3f22b7986f2b12f0b3ce1b2e47f11.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/6b22c5b126dde59f950039bed641d64e24c3f22b7986f2b12f0b3ce1b2e47f11.jpg)

![88e00eb869eece2a27e7f41560461d6243376478c590390a460a555ce56cb171.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/88e00eb869eece2a27e7f41560461d6243376478c590390a460a555ce56cb171.jpg)

![8937233bcf003feabe5948895594eaefa0e3178ae0dfecd9c8f304ae6e72dcb6.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/8937233bcf003feabe5948895594eaefa0e3178ae0dfecd9c8f304ae6e72dcb6.jpg)

![8f9d3a4b34d768e235d24a80e0cccedb7e223d466e3e14d9d81a119098f467c9.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/8f9d3a4b34d768e235d24a80e0cccedb7e223d466e3e14d9d81a119098f467c9.jpg)

![a27aba2c14f5815a63fcf5bbf5da1028257286937b2fd3e64daa34c03be741d8.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/a27aba2c14f5815a63fcf5bbf5da1028257286937b2fd3e64daa34c03be741d8.jpg)

![b453121017302bc133c493aa236e7756f55b3b565fa2ed9fad34bbfd7028122d.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/b453121017302bc133c493aa236e7756f55b3b565fa2ed9fad34bbfd7028122d.jpg)

![bdc85c1c30a3ec5338a88b3b0c1c36c9e3d63943723fae77f599e2a2a9855606.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/bdc85c1c30a3ec5338a88b3b0c1c36c9e3d63943723fae77f599e2a2a9855606.jpg)

![e8d5a9af840869f8fad5ac514ca211eeb8d471758c2a04a2f276082f7cecc9db.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/e8d5a9af840869f8fad5ac514ca211eeb8d471758c2a04a2f276082f7cecc9db.jpg)

![ee04fa9b5be6e324e8d903f04d7df2059a42cb3fdc1f6a30a4241ddeae3d0ce3.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/ee04fa9b5be6e324e8d903f04d7df2059a42cb3fdc1f6a30a4241ddeae3d0ce3.jpg)

![fe46d8813e346662668d0f91b8b3e7f24e2b4cb8e85ebfd9949df0b4d376d066.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/fe46d8813e346662668d0f91b8b3e7f24e2b4cb8e85ebfd9949df0b4d376d066.jpg)

![ff511c5bd2fe2daf39cfa2d02829b3fc9fb691abce2d117268dad35e9829674e.jpg](../icml_results/79_Shared%20Semantics%2C%20Divergent%20Mechanisms_%20Unsupervised%20Feature%20Discovery%20by%20Aligning%20Semantics%20and%20Mec/tables/ff511c5bd2fe2daf39cfa2d02829b3fc9fb691abce2d117268dad35e9829674e.jpg)

<a id="paper-0079"></a>

## SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models


### Images

![010691893e55291f7620bd325d97bb0caa810ccc22e6b1294e8df267f8f89044.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/010691893e55291f7620bd325d97bb0caa810ccc22e6b1294e8df267f8f89044.jpg)

![105fb3ccfb344625bd421401e63f007eeaa2594af1672981850d9e6cec4ce4f1.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/105fb3ccfb344625bd421401e63f007eeaa2594af1672981850d9e6cec4ce4f1.jpg)

![646a7ab766cb255b453eb5cd52f3120e3745b41e4766ece24b6c640a3e2b8841.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/646a7ab766cb255b453eb5cd52f3120e3745b41e4766ece24b6c640a3e2b8841.jpg)

![9944e37123006e6d3d70176ff0a7072b966a15e747226ce76085e32db2fcd25f.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/9944e37123006e6d3d70176ff0a7072b966a15e747226ce76085e32db2fcd25f.jpg)

![9f7dd1fb510c10c4cc82e4dfbe8b22bf9762e9aaa84f476db0b60a75d06ded68.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/9f7dd1fb510c10c4cc82e4dfbe8b22bf9762e9aaa84f476db0b60a75d06ded68.jpg)

![a081de763f304a68efb9d46e7a3f14e442ee83bdf8c05f826c5a3f870ee9a0b5.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/a081de763f304a68efb9d46e7a3f14e442ee83bdf8c05f826c5a3f870ee9a0b5.jpg)

![b593adf1488be650272a9950431cc1bfd222518d403fbdb3b1e291000b5be273.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/b593adf1488be650272a9950431cc1bfd222518d403fbdb3b1e291000b5be273.jpg)

![c8326498e1e95a1a4dc780299ddde0a23268e39545ad5a145ba5f10669914ae2.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/images/c8326498e1e95a1a4dc780299ddde0a23268e39545ad5a145ba5f10669914ae2.jpg)

### Tables

![02fbfb172d021c5c4c2bbb645a5c2c8bf22abfe2228f1276f61ca2f1d71182ad.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/02fbfb172d021c5c4c2bbb645a5c2c8bf22abfe2228f1276f61ca2f1d71182ad.jpg)

![184273e8e05f794e68e592abf793dfd1ca706353e18430c4802f334cbf6a40d2.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/184273e8e05f794e68e592abf793dfd1ca706353e18430c4802f334cbf6a40d2.jpg)

![37afce660d9b4a31524bc12cca8db632a11967802b2fb957807a039ef7bc6ce1.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/37afce660d9b4a31524bc12cca8db632a11967802b2fb957807a039ef7bc6ce1.jpg)

![554aeb61a2d2a9b36defea9d6e756810d5a8032f4bacd8e7f420f9f87a8ced3f.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/554aeb61a2d2a9b36defea9d6e756810d5a8032f4bacd8e7f420f9f87a8ced3f.jpg)

![74395bcbf7dfe435296e4a06caa437e92e1ee631eb7a32885e9585cd7e2d5f60.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/74395bcbf7dfe435296e4a06caa437e92e1ee631eb7a32885e9585cd7e2d5f60.jpg)

![888429d86832fbf96eadc56ed1d7666213d73f2b8f6d351eebb6b695224e8a10.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/888429d86832fbf96eadc56ed1d7666213d73f2b8f6d351eebb6b695224e8a10.jpg)

![9535fc12791946c8da58b7925e2375ce4c607c084fa2da86aa24ce76333e5098.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/9535fc12791946c8da58b7925e2375ce4c607c084fa2da86aa24ce76333e5098.jpg)

![960ded3bc131ed11aa9d8d18e6deee5fe3605b1264d1f8d6daa7779b1af74d0a.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/960ded3bc131ed11aa9d8d18e6deee5fe3605b1264d1f8d6daa7779b1af74d0a.jpg)

![b69eb3856e22ebcb2475293ce03c62e20889d9e93a6a25d757a67980484179a7.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/b69eb3856e22ebcb2475293ce03c62e20889d9e93a6a25d757a67980484179a7.jpg)

![c01c3f09a25bc81677dfb1ad5fb35cfe41abb28e3771bba71aa25f03bea64f48.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/c01c3f09a25bc81677dfb1ad5fb35cfe41abb28e3771bba71aa25f03bea64f48.jpg)

![d0f4f927d4631b51c4d8e737cc5877f9279a2babea1202ff24d9eaabde6ce337.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/d0f4f927d4631b51c4d8e737cc5877f9279a2babea1202ff24d9eaabde6ce337.jpg)

![d15d2627810a81eff355f6bedda49a5e2c1f08f28583f6d93bea4f5f6acd0534.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/d15d2627810a81eff355f6bedda49a5e2c1f08f28583f6d93bea4f5f6acd0534.jpg)

![de7f6cb2428a87c36bcb24168cb86673e144f3da54f23bda1da76f4591d1717c.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/de7f6cb2428a87c36bcb24168cb86673e144f3da54f23bda1da76f4591d1717c.jpg)

![e0bf9d4d788055e467deaae03d44c186826275f85bab5ba2789e8ccd08e591d2.jpg](../icml_results/80_SCALE_%20Self-uncertainty%20Conditioned%20Adaptive%20Looking%20and%20Execution%20for%20Vision-Language-Action%20Models/tables/e0bf9d4d788055e467deaae03d44c186826275f85bab5ba2789e8ccd08e591d2.jpg)

<a id="paper-0080"></a>

## Reinforced Sequential Monte Carlo for Amortised Sampling


### Images

![11e6363bb3655c3f7357c7e710ecbb8329cc1590fad96f05030aef2134c335ab.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/11e6363bb3655c3f7357c7e710ecbb8329cc1590fad96f05030aef2134c335ab.jpg)

![4594c1bf9c1ae79315a28c3f32916ac28ec98237f169cce45c00b748d4740896.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/4594c1bf9c1ae79315a28c3f32916ac28ec98237f169cce45c00b748d4740896.jpg)

![4e417ce4a44b3819fe542a0748c2fcbd7f85542bc593bc9c92d46f0b15408a2d.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/4e417ce4a44b3819fe542a0748c2fcbd7f85542bc593bc9c92d46f0b15408a2d.jpg)

![68e367be4e51a6c123ce3d1de65fef5df90f35e0b8f80e128924951a9dadd5c9.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/68e367be4e51a6c123ce3d1de65fef5df90f35e0b8f80e128924951a9dadd5c9.jpg)

![780f8ccdf57ea0ed06761d4986ffc5b4dd00f28fff90a673cf095c283d0119d7.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/780f8ccdf57ea0ed06761d4986ffc5b4dd00f28fff90a673cf095c283d0119d7.jpg)

![7bed73f18e5665417dddaa1848f80a91b2c4c50319d7a15796c87936f116cd35.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/7bed73f18e5665417dddaa1848f80a91b2c4c50319d7a15796c87936f116cd35.jpg)

![9f78608b635b5c9e3c10e45c272548234edd105f1d4fede8823d0c64a1c97c43.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/9f78608b635b5c9e3c10e45c272548234edd105f1d4fede8823d0c64a1c97c43.jpg)

![a26b7e87bbd1d7a80db02dc292ab0cb96e526a717527d8427c04646e4d03119f.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/a26b7e87bbd1d7a80db02dc292ab0cb96e526a717527d8427c04646e4d03119f.jpg)

![dd4e6c0f2db1fe88eac7a238c7f8f274499ba94e7bf5406f32f9ba3352782ae2.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/dd4e6c0f2db1fe88eac7a238c7f8f274499ba94e7bf5406f32f9ba3352782ae2.jpg)

![f92e982e488dd042b1b06e41ec42db49a19c52a13ab45683a37695de572774eb.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/images/f92e982e488dd042b1b06e41ec42db49a19c52a13ab45683a37695de572774eb.jpg)

### Tables

![01aee8ee5d1bb793cac187b47d406ae0e2058cfa9fc6a0a9a362025a66927ab1.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/01aee8ee5d1bb793cac187b47d406ae0e2058cfa9fc6a0a9a362025a66927ab1.jpg)

![129d50a076fa4d9119fce18619b503e18db84b81186ae41f2f017727fe032f2c.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/129d50a076fa4d9119fce18619b503e18db84b81186ae41f2f017727fe032f2c.jpg)

![2ee51c0c65d1f760e1ffe7c4e47d825850b80a032d8732f1c19d3e11078f8ab4.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/2ee51c0c65d1f760e1ffe7c4e47d825850b80a032d8732f1c19d3e11078f8ab4.jpg)

![3d89b3a6bc67a8caf838b66f500a7be2d7082945cdc30373682468750f4a2c7b.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/3d89b3a6bc67a8caf838b66f500a7be2d7082945cdc30373682468750f4a2c7b.jpg)

![422c5056d837709ea8bbc912fb6e89fd1c1608a3cf56bfd7690fe88a29d86d6d.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/422c5056d837709ea8bbc912fb6e89fd1c1608a3cf56bfd7690fe88a29d86d6d.jpg)

![9c92f366c3e102f53cd7f3779d80f595bedafe50f5f3639a510b4f3ca2aecb8e.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/9c92f366c3e102f53cd7f3779d80f595bedafe50f5f3639a510b4f3ca2aecb8e.jpg)

![b94f17f5efd9876309d9d386cdee429a16210b0948f7fc7d5c8f5cfe394ffd78.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/b94f17f5efd9876309d9d386cdee429a16210b0948f7fc7d5c8f5cfe394ffd78.jpg)

![e06bad019039b891a4a0144a21784fb43ea386b379467c6d22c5c17ee2503086.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/e06bad019039b891a4a0144a21784fb43ea386b379467c6d22c5c17ee2503086.jpg)

![fc64ea5d7d2049c24d8feec672c47e38e40c987b81b99756c040f246aec78b54.jpg](../icml_results/81_Reinforced%20Sequential%20Monte%20Carlo%20for%20Amortised%20Sampling/tables/fc64ea5d7d2049c24d8feec672c47e38e40c987b81b99756c040f246aec78b54.jpg)

<a id="paper-0081"></a>

## Concept Removal Guidance: Evidence-Calibrated Negative Guidance for Safe Diffusion Sampling


### Images

![0c8861224b25c1b46852f589ebe6118ae83c85ac3a1e2b4d40570724276e9ca5.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/0c8861224b25c1b46852f589ebe6118ae83c85ac3a1e2b4d40570724276e9ca5.jpg)

![160ca78c135c160675d4d4215c15fee3e6ebfef29a29024f9f882b5592d4e37a.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/160ca78c135c160675d4d4215c15fee3e6ebfef29a29024f9f882b5592d4e37a.jpg)

![232a038564bfae43bbc67678bc354f7c553d6e2fc6008c49ec6108a8ee12c769.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/232a038564bfae43bbc67678bc354f7c553d6e2fc6008c49ec6108a8ee12c769.jpg)

![270739ce7ca1a99749d4a47bf8fbd34223b16e78d348426ccd422a2f8f8e6ba8.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/270739ce7ca1a99749d4a47bf8fbd34223b16e78d348426ccd422a2f8f8e6ba8.jpg)

![2eb0b69898cf8b183b83810838d1cc9c02ac742234d69135aeb75815fe2426d8.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/2eb0b69898cf8b183b83810838d1cc9c02ac742234d69135aeb75815fe2426d8.jpg)

![35c2c0b1ebb645c8508e19456c132f7b0abb4999ac705a77d4f25fd24fbb7da4.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/35c2c0b1ebb645c8508e19456c132f7b0abb4999ac705a77d4f25fd24fbb7da4.jpg)

![425c9494f040cc3e9550a087404bfc509f37641deb822db3b59a547871425a04.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/425c9494f040cc3e9550a087404bfc509f37641deb822db3b59a547871425a04.jpg)

![5ca39680071f4269ba9c50f9e2f5b25833c650c93caa906fa78234219f4233a4.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/5ca39680071f4269ba9c50f9e2f5b25833c650c93caa906fa78234219f4233a4.jpg)

![6dd9d2ac0eef43460a6970858db5ef75ba4ee7721a5ac7f5328d23cc2f5ae7e2.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/6dd9d2ac0eef43460a6970858db5ef75ba4ee7721a5ac7f5328d23cc2f5ae7e2.jpg)

![74ea3320aa799cf9d872040e558bf6551f1c913ddddd224151c503bf19901428.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/74ea3320aa799cf9d872040e558bf6551f1c913ddddd224151c503bf19901428.jpg)

![8189cd304a62c181fb4850333f0ea709c2dbe684bacebfaf16c97012014fc78b.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/8189cd304a62c181fb4850333f0ea709c2dbe684bacebfaf16c97012014fc78b.jpg)

![b6203580657990b0c095bcebcabe62c184e62d173d7348737cd6760c61389187.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/b6203580657990b0c095bcebcabe62c184e62d173d7348737cd6760c61389187.jpg)

![e3c257a4b73fa766061e6c1c05a8d07809033b815f6c3452286a986ca56882a1.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/e3c257a4b73fa766061e6c1c05a8d07809033b815f6c3452286a986ca56882a1.jpg)

![ece9724457db2eee08349ba9da430b2e20c648ce14410e0fbabf66e682f131fb.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/images/ece9724457db2eee08349ba9da430b2e20c648ce14410e0fbabf66e682f131fb.jpg)

### Tables

![001dadcb5dc6176b7f14e51cbf962aeed11f12ed0437e1559715865a634a87d3.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/001dadcb5dc6176b7f14e51cbf962aeed11f12ed0437e1559715865a634a87d3.jpg)

![0c0ae5809c2970327697724bdf6334fa04572824669ef1958fe90e60b09cc516.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/0c0ae5809c2970327697724bdf6334fa04572824669ef1958fe90e60b09cc516.jpg)

![2dd615e951368579612bdb955c38de333dc31e1cd33492bcacc4f483ecc321df.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/2dd615e951368579612bdb955c38de333dc31e1cd33492bcacc4f483ecc321df.jpg)

![6234f239320afb6a999a305f983baff0fdabff58c6cad425499d7d81c1a31190.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/6234f239320afb6a999a305f983baff0fdabff58c6cad425499d7d81c1a31190.jpg)

![64feec347cd8ca69f69027d92dd217f49056d0ff4afcb721ddfd6d3c43d86601.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/64feec347cd8ca69f69027d92dd217f49056d0ff4afcb721ddfd6d3c43d86601.jpg)

![76f928856abc7badd2b51bf73c9c02ef0215e06891e4e49a36cace12c4ec944e.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/76f928856abc7badd2b51bf73c9c02ef0215e06891e4e49a36cace12c4ec944e.jpg)

![7e641e1dbf0df78396da924b02008fbad221b085bf2580c2fcc0bf54b8c67b34.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/7e641e1dbf0df78396da924b02008fbad221b085bf2580c2fcc0bf54b8c67b34.jpg)

![969597e6de39a9f0153c8d7d2af1e0737a2d5c6e25f30dcd251953f6a64022e8.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/969597e6de39a9f0153c8d7d2af1e0737a2d5c6e25f30dcd251953f6a64022e8.jpg)

![ae887cae3f18923336639911d3b33f7ae39b312cd3f49c1730a7c445a97176e9.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/ae887cae3f18923336639911d3b33f7ae39b312cd3f49c1730a7c445a97176e9.jpg)

![b148645aa9e52cda3c64135b575b46b7511ac50f2d12ee2951acc2ac3cfcc240.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/b148645aa9e52cda3c64135b575b46b7511ac50f2d12ee2951acc2ac3cfcc240.jpg)

![bc656def8428e57f5313bbb838e593b265dce20aeff85416b7d3ba7191beb8be.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/bc656def8428e57f5313bbb838e593b265dce20aeff85416b7d3ba7191beb8be.jpg)

![bfa19de0a041b4ee8ab346e09607fbd157cf2f471d2d8f06b023c327f0f97050.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/bfa19de0a041b4ee8ab346e09607fbd157cf2f471d2d8f06b023c327f0f97050.jpg)

![c063aab771655d9f90e0947b378c31194780212e3707407d4e1843e0ca477b6b.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/c063aab771655d9f90e0947b378c31194780212e3707407d4e1843e0ca477b6b.jpg)

![c576283f92b6c51fa997e2fe9f537d1421299395ef9ea161ae49684ea57dc843.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/c576283f92b6c51fa997e2fe9f537d1421299395ef9ea161ae49684ea57dc843.jpg)

![c5fdef1b8b6a8e90db95415a5efaa4d727719b450eb81935ce1a930d144e6e65.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/c5fdef1b8b6a8e90db95415a5efaa4d727719b450eb81935ce1a930d144e6e65.jpg)

![ca690202ed76819948b301d3c9d06f074ff1e5dd53946d634c653700bb3dba24.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/ca690202ed76819948b301d3c9d06f074ff1e5dd53946d634c653700bb3dba24.jpg)

![d19decea6e357ffa4a7db63c8603ad98f1b16b14ebaa6abd1e276a6b6e171a8a.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/d19decea6e357ffa4a7db63c8603ad98f1b16b14ebaa6abd1e276a6b6e171a8a.jpg)

![de8d868f252dc1edabdced8c72e835973d68be105cb6fd72bc6fd27bc0c98e04.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/de8d868f252dc1edabdced8c72e835973d68be105cb6fd72bc6fd27bc0c98e04.jpg)

![ee9135c35102297d7a49a03e2aea2c717ac17846d339610303c9b017a65e6d00.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/ee9135c35102297d7a49a03e2aea2c717ac17846d339610303c9b017a65e6d00.jpg)

![fe178d7731f5358efb815812cb0a7057c1e1dc217525d81fcbef5844c4899383.jpg](../icml_results/82_Concept%20Removal%20Guidance_%20Evidence-Calibrated%20Negative%20Guidance%20for%20Safe%20Diffusion%20Sampling/tables/fe178d7731f5358efb815812cb0a7057c1e1dc217525d81fcbef5844c4899383.jpg)

<a id="paper-0082"></a>

## End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer


### Images

![1bf180f349750fca4ba60cb3c009eb4ad81bb17e0f5f53475b0a7ed48092f1ee.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/1bf180f349750fca4ba60cb3c009eb4ad81bb17e0f5f53475b0a7ed48092f1ee.jpg)

![2a60e0b00fab558bb0195bc20a6392af99fa6ec90ac290ee3de3a084082ae610.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/2a60e0b00fab558bb0195bc20a6392af99fa6ec90ac290ee3de3a084082ae610.jpg)

![2e4f7be625fd5b0efe7c75fcccef4fccae0e163d977df742407a7b6288d77944.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/2e4f7be625fd5b0efe7c75fcccef4fccae0e163d977df742407a7b6288d77944.jpg)

![357dbbebac49c62f5ee2200f48a290943c07931f0b75b97ace0131cad2ddc3fa.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/357dbbebac49c62f5ee2200f48a290943c07931f0b75b97ace0131cad2ddc3fa.jpg)

![3e7af6ab1ec20d38fd82740d4c6c354f272e0d7c5af51c42e0d5d36cb53fea5b.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/3e7af6ab1ec20d38fd82740d4c6c354f272e0d7c5af51c42e0d5d36cb53fea5b.jpg)

![44547aa8a0b69f2e7ad0bc63c322d561966b2ee6a0c71a45dbf554e48d838ba3.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/44547aa8a0b69f2e7ad0bc63c322d561966b2ee6a0c71a45dbf554e48d838ba3.jpg)

![89387eb450345c0e405215f5c538c7c95bbd779f8760fcb6fde02697431b5549.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/images/89387eb450345c0e405215f5c538c7c95bbd779f8760fcb6fde02697431b5549.jpg)

### Tables

![06b7868c40c203ad39639537d05a268a25ac4d746312f3f2c32c01de8cceec62.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/06b7868c40c203ad39639537d05a268a25ac4d746312f3f2c32c01de8cceec62.jpg)

![1b5a9a841e87d7c0040c60fa4fc7f37590379be32067c1045df6c585c4a59792.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/1b5a9a841e87d7c0040c60fa4fc7f37590379be32067c1045df6c585c4a59792.jpg)

![1dc4011127c76f52ef2d091cb536f631940fd7e0aab9f76287651cca424a410e.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/1dc4011127c76f52ef2d091cb536f631940fd7e0aab9f76287651cca424a410e.jpg)

![1f7a6fdca281ccb8834d949394ec8944bc4477a30a02a813124ba1ba4a339a88.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/1f7a6fdca281ccb8834d949394ec8944bc4477a30a02a813124ba1ba4a339a88.jpg)

![302d388cb8cc5fa979ef50ed4aae85f3ba2bdb77edbbf54729f632d4a3f8bf88.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/302d388cb8cc5fa979ef50ed4aae85f3ba2bdb77edbbf54729f632d4a3f8bf88.jpg)

![36438f3a277747786340e5988d5dcd228389233e84d555679f63c3632d908ac0.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/36438f3a277747786340e5988d5dcd228389233e84d555679f63c3632d908ac0.jpg)

![63e53c19754f7fa96344a4199bef24731e54f7f9cba124a49a027cc9bd0605ed.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/63e53c19754f7fa96344a4199bef24731e54f7f9cba124a49a027cc9bd0605ed.jpg)

![848b1b9292d317192483654db5bf6a7990c9f0248eb6c59dcb25e0fb4cdcff16.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/848b1b9292d317192483654db5bf6a7990c9f0248eb6c59dcb25e0fb4cdcff16.jpg)

![875ed073f270eef2ef985c0ad2f6ef19e1535046fd807ead1c76ffbb9f5d98c9.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/875ed073f270eef2ef985c0ad2f6ef19e1535046fd807ead1c76ffbb9f5d98c9.jpg)

![bcf088aabb399c0dbb4483342c5e58e135a192555fa1fb042d23fb93db66a683.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/bcf088aabb399c0dbb4483342c5e58e135a192555fa1fb042d23fb93db66a683.jpg)

![cc2befb9df69f11ae88c084d73d47116c170d3d687543b8b980a54018b823f52.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/cc2befb9df69f11ae88c084d73d47116c170d3d687543b8b980a54018b823f52.jpg)

![dc7da2e5cde30c2adeed25f8eada66024aaa0e37515de3add2c5ff1129c2c77c.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/dc7da2e5cde30c2adeed25f8eada66024aaa0e37515de3add2c5ff1129c2c77c.jpg)

![f2e3cded099907026a87c95c80ef5484fcdf2a5b6e98af396f323825df6a3c78.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/f2e3cded099907026a87c95c80ef5484fcdf2a5b6e98af396f323825df6a3c78.jpg)

![f32d3e480843c459ea6755a1442b61fa267a9c4364b7cb642f8431e3d91b7f50.jpg](../icml_results/83_End-to-End%20Autoregressive%20Image%20Generation%20with%201D%20Semantic%20Tokenizer/tables/f32d3e480843c459ea6755a1442b61fa267a9c4364b7cb642f8431e3d91b7f50.jpg)

<a id="paper-0083"></a>

## Transformer Circuits Can Realize Clustering Algorithms


### Images

![1689d23b9daefb9175b4ee705cfd71e2c0f5efb1e3cb99061c2c01a74947626a.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/1689d23b9daefb9175b4ee705cfd71e2c0f5efb1e3cb99061c2c01a74947626a.jpg)

![2430dc8139b0f2fa55dff445e9eb4eb74ae9acfbb5c7aa5c75f65723a1476604.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/2430dc8139b0f2fa55dff445e9eb4eb74ae9acfbb5c7aa5c75f65723a1476604.jpg)

![252fa83e2364d39431e0aba24940d3d7e9d793e90d161b508e74b883a92c450c.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/252fa83e2364d39431e0aba24940d3d7e9d793e90d161b508e74b883a92c450c.jpg)

![2a4c823aef1ccb17a289c2be7e3990949e2a2623fc5b9e77547e1778705637df.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/2a4c823aef1ccb17a289c2be7e3990949e2a2623fc5b9e77547e1778705637df.jpg)

![4af0ba921f7231aeb5a937c3c5d43be76c9b75fd4b118de985e141ba3a8ac629.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/4af0ba921f7231aeb5a937c3c5d43be76c9b75fd4b118de985e141ba3a8ac629.jpg)

![59f7ac50eb4b8b762927aae34f87f7a1f496d91bd95ed563710b72a94cfdeaf8.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/59f7ac50eb4b8b762927aae34f87f7a1f496d91bd95ed563710b72a94cfdeaf8.jpg)

![6e0f406f36750f5a41942e0d2f5d256f96a9e2ba3cc741bd7ba4f01b8f283ec3.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/6e0f406f36750f5a41942e0d2f5d256f96a9e2ba3cc741bd7ba4f01b8f283ec3.jpg)

![79534a148bad44b3ce7e780058debca47899d6e6e344b595592833c7eceeaabf.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/79534a148bad44b3ce7e780058debca47899d6e6e344b595592833c7eceeaabf.jpg)

![7f91e48e8c2260ba875ea5857193a5170e78281ea1194e40181c62d28b0a52b4.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/7f91e48e8c2260ba875ea5857193a5170e78281ea1194e40181c62d28b0a52b4.jpg)

![de83ab2b64fababb819a0624afcc613e3f3a3efce34141ae4dd044fe23963a8a.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/de83ab2b64fababb819a0624afcc613e3f3a3efce34141ae4dd044fe23963a8a.jpg)

![e83ddc1cf73132686922b9df52739619663d0b336bf6419301704c30f71e9fc1.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/e83ddc1cf73132686922b9df52739619663d0b336bf6419301704c30f71e9fc1.jpg)

![f9d0e2216820a11d41fc5bb32c3273522ad94c8648f6138e8c52d5b8069c7399.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/images/f9d0e2216820a11d41fc5bb32c3273522ad94c8648f6138e8c52d5b8069c7399.jpg)

### Tables

![2f0fa6dbd4af3924f249aab1ba3217351955b8335eef3dd736dc243da16150ed.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/tables/2f0fa6dbd4af3924f249aab1ba3217351955b8335eef3dd736dc243da16150ed.jpg)

![42c03d750f2a3fa993cc81f1f96da75b75c6529301513c3ab71be19655ee60ac.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/tables/42c03d750f2a3fa993cc81f1f96da75b75c6529301513c3ab71be19655ee60ac.jpg)

![4cfd42df9b8dc144e983bf5af185f5240f9c2d857f20db7424c4dfca8bb92966.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/tables/4cfd42df9b8dc144e983bf5af185f5240f9c2d857f20db7424c4dfca8bb92966.jpg)

![8887d4f9f671b23f92652105ccee1be499832dcce33c9a64112ac24ab2193ac5.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/tables/8887d4f9f671b23f92652105ccee1be499832dcce33c9a64112ac24ab2193ac5.jpg)

![e08b7977c1543a5b296f7f14110513c3936100db53146926e34123d34500c1cd.jpg](../icml_results/84_Transformer%20Circuits%20Can%20Realize%20Clustering%20Algorithms/tables/e08b7977c1543a5b296f7f14110513c3936100db53146926e34123d34500c1cd.jpg)

<a id="paper-0084"></a>

## Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models


### Images

![07407bf9b7e857f9833802b5cc423436f0be5ff8f70a292d213d41047bd0f7c6.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/07407bf9b7e857f9833802b5cc423436f0be5ff8f70a292d213d41047bd0f7c6.jpg)

![0f9d33e35cf1b00fff67d4f8184d2c9b4c697ec5322ae846a7556ea5a85f25af.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/0f9d33e35cf1b00fff67d4f8184d2c9b4c697ec5322ae846a7556ea5a85f25af.jpg)

![1636792bd2605e295ba6eb80e76bb9391fbf65e27dd9e7f1970c0988038f3430.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/1636792bd2605e295ba6eb80e76bb9391fbf65e27dd9e7f1970c0988038f3430.jpg)

![2247c042c562f3ae89075e1026d08b7b52edd5aa437bbd6c4665677867c44c08.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/2247c042c562f3ae89075e1026d08b7b52edd5aa437bbd6c4665677867c44c08.jpg)

![425c45e0abad480f274ceb3e4ee87f0a0716a6bcdbf4f09fc0db16508cad6e0c.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/425c45e0abad480f274ceb3e4ee87f0a0716a6bcdbf4f09fc0db16508cad6e0c.jpg)

![4f57dac20d4f8e7b3758eefc29f2df129e0bd0bc5c7d25f02013c92a2d1d84c3.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/4f57dac20d4f8e7b3758eefc29f2df129e0bd0bc5c7d25f02013c92a2d1d84c3.jpg)

![5db862b24c2af39ff5fcc5bc07b50d101ea12370df3b8d23c7e50edab806608e.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/5db862b24c2af39ff5fcc5bc07b50d101ea12370df3b8d23c7e50edab806608e.jpg)

![6477044bcff7205b8afd95c153417f5c2daf646db6a302f7d246087c97286e7c.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/6477044bcff7205b8afd95c153417f5c2daf646db6a302f7d246087c97286e7c.jpg)

![68dc01d73898151e0c787a6bd5021550d3f5644dce238d271936997549199b54.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/68dc01d73898151e0c787a6bd5021550d3f5644dce238d271936997549199b54.jpg)

![7b283bcc83a5a479ab6343c34873621248aedab4eda3c04fc6e63c72486acb0b.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/7b283bcc83a5a479ab6343c34873621248aedab4eda3c04fc6e63c72486acb0b.jpg)

![7d1ae8a25c38d1c5af77abd7fdc2b5c6aafce0c13e2a13eef448c27fe5152b1c.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/7d1ae8a25c38d1c5af77abd7fdc2b5c6aafce0c13e2a13eef448c27fe5152b1c.jpg)

![83c08615ed0ed5da5dca44011589c56574231559fa1cc4040e3544aeeaec2f44.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/83c08615ed0ed5da5dca44011589c56574231559fa1cc4040e3544aeeaec2f44.jpg)

![9032347cc1c75ea3bee897303f39df4ea91961eedf387ab04c4addce33c9df3a.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/9032347cc1c75ea3bee897303f39df4ea91961eedf387ab04c4addce33c9df3a.jpg)

![9c0b3082f00178e3f971560ab3fb8a74734fc179021bc4cc783b40eddb58b3a4.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/9c0b3082f00178e3f971560ab3fb8a74734fc179021bc4cc783b40eddb58b3a4.jpg)

![9eec9e8d68dd5c5ab97eab131605a9d8e80f4e52cf0f863c2997a007da4b55fa.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/9eec9e8d68dd5c5ab97eab131605a9d8e80f4e52cf0f863c2997a007da4b55fa.jpg)

![9f6fb94c8d80c073e4193685edaee134e330c4f8f83dd9b2638929a095ffc122.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/images/9f6fb94c8d80c073e4193685edaee134e330c4f8f83dd9b2638929a095ffc122.jpg)

### Tables

![10fd22e4c9857bf1f2965af143d12b59674508ab62a6cc3fb88faeb365d2c177.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/tables/10fd22e4c9857bf1f2965af143d12b59674508ab62a6cc3fb88faeb365d2c177.jpg)

![5b6b826098a730558829800084dfb958956d188016e5b8088b3596f0e2cb7820.jpg](../icml_results/85_Expressivity-Efficiency%20Tradeoffs%20for%20Hybrid%20Sequence%20Models/tables/5b6b826098a730558829800084dfb958956d188016e5b8088b3596f0e2cb7820.jpg)

<a id="paper-0085"></a>

## SVRG and Beyond via Posterior Correction


### Images

![204ec52a238184c536c34c76da67e7182e386743235302d44845f8a3b5f24a91.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/204ec52a238184c536c34c76da67e7182e386743235302d44845f8a3b5f24a91.jpg)

![6d08aac296605de6231527f908fb6c5dce9ea8337719d63d6f29510d641e833b.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/6d08aac296605de6231527f908fb6c5dce9ea8337719d63d6f29510d641e833b.jpg)

![905e228852a65020aa2b5fbb46fef9db9d63295cad4327b29160ad183a06c750.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/905e228852a65020aa2b5fbb46fef9db9d63295cad4327b29160ad183a06c750.jpg)

![93353f53e576af759746a6f3dbed4ff23bd6e474e53e6ba3169cf79307e1c8f1.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/93353f53e576af759746a6f3dbed4ff23bd6e474e53e6ba3169cf79307e1c8f1.jpg)

![a585b0b5cbf6099ac9a897a7e8a4781a01732607ff63f28febfe923411be248d.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/a585b0b5cbf6099ac9a897a7e8a4781a01732607ff63f28febfe923411be248d.jpg)

![ab181a9146d9b4b16f3ca698e06ef95c84651a94ad0c12e39eb43357484dac08.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/ab181a9146d9b4b16f3ca698e06ef95c84651a94ad0c12e39eb43357484dac08.jpg)

![b752bfa2898a2c42cd2f7bd3c28bc8f409450158113a66363ea77348c3bf6fb1.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/b752bfa2898a2c42cd2f7bd3c28bc8f409450158113a66363ea77348c3bf6fb1.jpg)

![b8615a56c6065724b2bf8552d2a391ae6cc6fc22d5093fe12dbd6ade84e0337c.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/images/b8615a56c6065724b2bf8552d2a391ae6cc6fc22d5093fe12dbd6ade84e0337c.jpg)

### Tables

![d9c59e8f55efa5e47651a51ea3488b7cde54a61999957e8ce928bfae107ce522.jpg](../icml_results/86_SVRG%20and%20Beyond%20via%20Posterior%20Correction/tables/d9c59e8f55efa5e47651a51ea3488b7cde54a61999957e8ce928bfae107ce522.jpg)

<a id="paper-0086"></a>

## RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies


### Images

![21ea8dab916ea0c802e86a9774ed22b4dbd1680616aec0782a546adeb9b57654.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/21ea8dab916ea0c802e86a9774ed22b4dbd1680616aec0782a546adeb9b57654.jpg)

![23b2817bb9046012d7744be3d4a484d39fa899b23c36218bc4dc5cdca3004638.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/23b2817bb9046012d7744be3d4a484d39fa899b23c36218bc4dc5cdca3004638.jpg)

![2ba391f2bb27db2de0ea239e9703c01105f858ddf6e998c0aa959497f1c0de5d.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/2ba391f2bb27db2de0ea239e9703c01105f858ddf6e998c0aa959497f1c0de5d.jpg)

![2eb47d5be6e8665a7c56380f0dc332224c147ac91431bd5f32ec11beee346468.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/2eb47d5be6e8665a7c56380f0dc332224c147ac91431bd5f32ec11beee346468.jpg)

![38532f1ee0846f56519468075f83be4f39f0baa822c754fc3f77901e685592fd.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/38532f1ee0846f56519468075f83be4f39f0baa822c754fc3f77901e685592fd.jpg)

![4606cc0e2e442a62008e3031036d6ea13c3ead61a68a3b03deb558a2ad80c43d.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/4606cc0e2e442a62008e3031036d6ea13c3ead61a68a3b03deb558a2ad80c43d.jpg)

![4cbfa774abc0d4b34c8a4d50b66d444cd009f2561f7b1462f112514796146d01.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/4cbfa774abc0d4b34c8a4d50b66d444cd009f2561f7b1462f112514796146d01.jpg)

![5085828e7239b56a5e8f15b17d85401daa56892cb80fd4336e06839ac7b4fa64.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/5085828e7239b56a5e8f15b17d85401daa56892cb80fd4336e06839ac7b4fa64.jpg)

![6a2b614e3abba124093a72cd87cd6aac6422b2173989013711f1310e0de84b01.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/6a2b614e3abba124093a72cd87cd6aac6422b2173989013711f1310e0de84b01.jpg)

![6f22666dd5ad6912ec4333f2e5912828aea48fda2f66af720a1ec72b6f3267e9.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/6f22666dd5ad6912ec4333f2e5912828aea48fda2f66af720a1ec72b6f3267e9.jpg)

![6feb340605882008812f106a316889305e049a0ec9ef2e6f7b3cee0226c80d68.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/6feb340605882008812f106a316889305e049a0ec9ef2e6f7b3cee0226c80d68.jpg)

![74f5eef38f48429d1d3c4041958e76b3cd65efe90127e33d356a46eaa5da2173.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/74f5eef38f48429d1d3c4041958e76b3cd65efe90127e33d356a46eaa5da2173.jpg)

![7815e1cee49e1757cb7e450eb714d421553fd3ba50edaf86e52cdb864f88145f.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/7815e1cee49e1757cb7e450eb714d421553fd3ba50edaf86e52cdb864f88145f.jpg)

![7fc93b39f6c4d94a7733ff36b7ef39ff6891985c40c72432dbeeb94953294c08.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/7fc93b39f6c4d94a7733ff36b7ef39ff6891985c40c72432dbeeb94953294c08.jpg)

![890b92a99ea1d88d3b3323aedcbb78f33aeb9ef58b6817362d3cebaf34b00a30.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/890b92a99ea1d88d3b3323aedcbb78f33aeb9ef58b6817362d3cebaf34b00a30.jpg)

![8bbace08143f817e2dea7dce7354d125fd6a2d37135acdd7bcd2cb21a1ea6f28.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/8bbace08143f817e2dea7dce7354d125fd6a2d37135acdd7bcd2cb21a1ea6f28.jpg)

![9947427d52c9a329e98584ab97a735b0a867a34b8e1b4040daa0d65c0ccc8c12.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/9947427d52c9a329e98584ab97a735b0a867a34b8e1b4040daa0d65c0ccc8c12.jpg)

![a198b80529f2f0f2e7ce0e38e7abf7663be3c93517ba8c9a00dcf49449269dd2.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/a198b80529f2f0f2e7ce0e38e7abf7663be3c93517ba8c9a00dcf49449269dd2.jpg)

![b80c92f9c335a21c2996fd672659fe6eb621ada9f12f9b7e09fcee5d24e4cff0.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/b80c92f9c335a21c2996fd672659fe6eb621ada9f12f9b7e09fcee5d24e4cff0.jpg)

![d87a068c76f70651a76f53b94d2bc6c90385a6c3567e10400b0a32b18ebe3287.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/d87a068c76f70651a76f53b94d2bc6c90385a6c3567e10400b0a32b18ebe3287.jpg)

![da08894838ee216fac19a800d7370b3d31509462da98d8e7364f4f686d84a6b1.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/da08894838ee216fac19a800d7370b3d31509462da98d8e7364f4f686d84a6b1.jpg)

![e2626e1b905682743acbbb7feb9c37342676feb7bfe093244a368f37c1c6b3ed.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/e2626e1b905682743acbbb7feb9c37342676feb7bfe093244a368f37c1c6b3ed.jpg)

![ee4a6823c7f2dd5b374ad8511b2745470f2cbda9c93fc4af676054c96f9905d5.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/ee4a6823c7f2dd5b374ad8511b2745470f2cbda9c93fc4af676054c96f9905d5.jpg)

![f0ee65d1e032617543bd555ae109a6c077bd9ee75ae76d8a720b64ad51343282.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/f0ee65d1e032617543bd555ae109a6c077bd9ee75ae76d8a720b64ad51343282.jpg)

![faeb5265af6798e151a7393e85c87aa652e55d7eb8028d33415a7cc2b5ae2bb0.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/faeb5265af6798e151a7393e85c87aa652e55d7eb8028d33415a7cc2b5ae2bb0.jpg)

![faf018fc473096a72a7d1ece11677ebf30c3e45daa1fa696bbe29f27e6c75edb.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/faf018fc473096a72a7d1ece11677ebf30c3e45daa1fa696bbe29f27e6c75edb.jpg)

![fd1b038d797736a7417c412259ba4b3f7544058d0b08c7d36048f0c67b16e489.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/images/fd1b038d797736a7417c412259ba4b3f7544058d0b08c7d36048f0c67b16e489.jpg)

### Tables

![07ca7c2ac2ce785f0cbfab3f18bccf74671522aca4a18f9d80eda62f4491f8c8.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/07ca7c2ac2ce785f0cbfab3f18bccf74671522aca4a18f9d80eda62f4491f8c8.jpg)

![0858d9992bdfb4f8552269bedfc581ededc3f3529af904e16d29eeec0c52a16f.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/0858d9992bdfb4f8552269bedfc581ededc3f3529af904e16d29eeec0c52a16f.jpg)

![199ca9e65200e8e61d241e842a72e6861cda3a3f369503d95fd0ede707cc538c.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/199ca9e65200e8e61d241e842a72e6861cda3a3f369503d95fd0ede707cc538c.jpg)

![371d38c491edf3e2660e950738d03d6fd9b128be3537126bfef0a624e5b012f4.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/371d38c491edf3e2660e950738d03d6fd9b128be3537126bfef0a624e5b012f4.jpg)

![39be3b388128671fb03b08482681caee649b3500a0a8b7907b0cabf587b84cb6.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/39be3b388128671fb03b08482681caee649b3500a0a8b7907b0cabf587b84cb6.jpg)

![3bb597a34db2a0a47c2628525df7d3db2239bff47afd2f37099657de1cb99750.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/3bb597a34db2a0a47c2628525df7d3db2239bff47afd2f37099657de1cb99750.jpg)

![42d3380413ea673d63c049d1a632fd898d5652145943809e223ac0acc6255e35.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/42d3380413ea673d63c049d1a632fd898d5652145943809e223ac0acc6255e35.jpg)

![467fcbe5fcd0022c0edac3c900762051ec30da715ad973a2d073969ea2304606.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/467fcbe5fcd0022c0edac3c900762051ec30da715ad973a2d073969ea2304606.jpg)

![53685f31609e6c8785d417218df0e8c0277a18b0cac36c894c770b791d53d125.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/53685f31609e6c8785d417218df0e8c0277a18b0cac36c894c770b791d53d125.jpg)

![593932265a5666d7cf599a7c954ed90fa1b738d540a145f36a0e10e53409783a.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/593932265a5666d7cf599a7c954ed90fa1b738d540a145f36a0e10e53409783a.jpg)

![5ae5b92ecee8a0db6fe7487971245f204f8be9d1f51d23da5018f1676f1177eb.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/5ae5b92ecee8a0db6fe7487971245f204f8be9d1f51d23da5018f1676f1177eb.jpg)

![5ce7bf5b184804b2c05cec3b63ab7c3485c97512f816ec90df1b2dfc75610f1b.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/5ce7bf5b184804b2c05cec3b63ab7c3485c97512f816ec90df1b2dfc75610f1b.jpg)

![667233bca8ddc6107bc332423e43d9d57bd4c5ceaffa8e63b4c6427de3301e34.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/667233bca8ddc6107bc332423e43d9d57bd4c5ceaffa8e63b4c6427de3301e34.jpg)

![69971014eb7c16d568e009872895a6b072414d5cc2062f4ed6786bd6245e8ef9.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/69971014eb7c16d568e009872895a6b072414d5cc2062f4ed6786bd6245e8ef9.jpg)

![6bd767e8acb4cb46661a8704bbd1af2664bd26c17d14b1c375a270c32f8c987a.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/6bd767e8acb4cb46661a8704bbd1af2664bd26c17d14b1c375a270c32f8c987a.jpg)

![6c9c51c53ee185183a3811f3a803afe8a7981aa5c3af45284ea480f7a7400aa2.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/6c9c51c53ee185183a3811f3a803afe8a7981aa5c3af45284ea480f7a7400aa2.jpg)

![7a640f4b3d42bca5eb20ae352b00f46159812d96f4766f5f7257ddb00e64d4fd.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/7a640f4b3d42bca5eb20ae352b00f46159812d96f4766f5f7257ddb00e64d4fd.jpg)

![940d66c7cd495465e6640eb2ecdb22a7fc74dd69150147ec28262735ec81b873.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/940d66c7cd495465e6640eb2ecdb22a7fc74dd69150147ec28262735ec81b873.jpg)

![972cdc1de856ec3ad91d4971d07358af60d4367d50f68f680d924dd943ef5bd4.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/972cdc1de856ec3ad91d4971d07358af60d4367d50f68f680d924dd943ef5bd4.jpg)

![9eb105fcfa6ac581c8e4ed2f4944481840eb75b1cb58f5dac4e41a51e5a2b633.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/9eb105fcfa6ac581c8e4ed2f4944481840eb75b1cb58f5dac4e41a51e5a2b633.jpg)

![9f090d896a79fabd75a03a30cb20eb7f4c73b85d04913c811d1e9e3f56474ddd.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/9f090d896a79fabd75a03a30cb20eb7f4c73b85d04913c811d1e9e3f56474ddd.jpg)

![b74516ae5739ae58845cc0b03b537acad5c2d9e368ca46702fbb62f8dc264ccd.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/b74516ae5739ae58845cc0b03b537acad5c2d9e368ca46702fbb62f8dc264ccd.jpg)

![b8834a16cbe5a8e0a285b65015203ef74228b6675b47041c7db24967bf4ca87b.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/b8834a16cbe5a8e0a285b65015203ef74228b6675b47041c7db24967bf4ca87b.jpg)

![c274ac93f4288528a3642f99dfdfb5a2dfa8df3872754066201f46f8b73e8847.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/c274ac93f4288528a3642f99dfdfb5a2dfa8df3872754066201f46f8b73e8847.jpg)

![c37bbfa912f911415e403bb071174a8df042c2d1325e1053ea3475dfe7581bee.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/c37bbfa912f911415e403bb071174a8df042c2d1325e1053ea3475dfe7581bee.jpg)

![cdc20d99c4486cc90fcf1e0f34a73973dc0f16600fc01aab79e82b12dde5d0ba.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/cdc20d99c4486cc90fcf1e0f34a73973dc0f16600fc01aab79e82b12dde5d0ba.jpg)

![d0d3c030ed346205d00fad486e84bede5224a1008767c60aed7b77bc81b2b9f7.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/d0d3c030ed346205d00fad486e84bede5224a1008767c60aed7b77bc81b2b9f7.jpg)

![defac302b5f42fe755114a08fac64bf3d24dda01729b98d33c7dffd92c65bf3d.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/defac302b5f42fe755114a08fac64bf3d24dda01729b98d33c7dffd92c65bf3d.jpg)

![f5eb6e1cffe0b3bdc451494f82fe7c70ab5988c1a5fb422b941dac6d74e9d514.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/f5eb6e1cffe0b3bdc451494f82fe7c70ab5988c1a5fb422b941dac6d74e9d514.jpg)

![fb9d4d07a43090fd91ed841f393d8e90e0488bade74936ba13f4a5c59957b332.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/fb9d4d07a43090fd91ed841f393d8e90e0488bade74936ba13f4a5c59957b332.jpg)

![fd6b1af2bf0d204cb4317c09a52734bc82f214be177073f4dc41d369232e3325.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/fd6b1af2bf0d204cb4317c09a52734bc82f214be177073f4dc41d369232e3325.jpg)

![ffb9170e8ed05cbb78da0712a64ea88bf6058ed693d0567677e79b9d425f6608.jpg](../icml_results/87_RoboMME_%20Benchmarking%20and%20Understanding%20Memory%20for%20Robotic%20Generalist%20Policies/tables/ffb9170e8ed05cbb78da0712a64ea88bf6058ed693d0567677e79b9d425f6608.jpg)

<a id="paper-0087"></a>

## Causal Modeling of Selection in Evolution


### Images

![1dabc6c2ca1d507b6f11eadda11c749a3fa5ebbdd4758009e6d730cc215eaaf9.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/1dabc6c2ca1d507b6f11eadda11c749a3fa5ebbdd4758009e6d730cc215eaaf9.jpg)

![24d18974d0f2ed95e34fb788fda6e3ace01869ea214416715a337240b8310132.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/24d18974d0f2ed95e34fb788fda6e3ace01869ea214416715a337240b8310132.jpg)

![2a56f079e89ad406d3cf6ba0060a27c8abb198f688e00e214d04e8b1ee173165.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/2a56f079e89ad406d3cf6ba0060a27c8abb198f688e00e214d04e8b1ee173165.jpg)

![2c8c1e560884f36963d4a5d1ab6dd24f3f32790d6fbb6eab8e10d0df8c817c81.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/2c8c1e560884f36963d4a5d1ab6dd24f3f32790d6fbb6eab8e10d0df8c817c81.jpg)

![3589a3442d44206c1061f62c26d848a8fa45d90cfa8c32a4a30a167e3a879554.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/3589a3442d44206c1061f62c26d848a8fa45d90cfa8c32a4a30a167e3a879554.jpg)

![3a6d46617d70f160df9749d415f355bf59b3fd67dd370d61f1a09bedd587adbd.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/3a6d46617d70f160df9749d415f355bf59b3fd67dd370d61f1a09bedd587adbd.jpg)

![43ad003d38b1030238eb76e2d406ed3288c170eda57ac8ba6445e7c883f45b28.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/43ad003d38b1030238eb76e2d406ed3288c170eda57ac8ba6445e7c883f45b28.jpg)

![56f8c26e4e7541d9bb12a7be537c0505437b90b41758a802d3207773739c090d.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/56f8c26e4e7541d9bb12a7be537c0505437b90b41758a802d3207773739c090d.jpg)

![785f9837af46df7fc191ee8df04268a89fbcbb61be39e058179191783b85a9ed.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/785f9837af46df7fc191ee8df04268a89fbcbb61be39e058179191783b85a9ed.jpg)

![7c6c533406c6aef0f6edbe723cac114d62a403d480b3d07a908ac170631aeaca.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/7c6c533406c6aef0f6edbe723cac114d62a403d480b3d07a908ac170631aeaca.jpg)

![84bebc3fa0b16240f5df09c01845f638d58fe76ea28c2d7ff153abb1f99419b2.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/84bebc3fa0b16240f5df09c01845f638d58fe76ea28c2d7ff153abb1f99419b2.jpg)

![853b529d76019479d00adb06ff5449df336f7a10c9c19f99f5c3654ffba0d12d.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/853b529d76019479d00adb06ff5449df336f7a10c9c19f99f5c3654ffba0d12d.jpg)

![89b024e1a2f875c030b693824a826cb1c7d024a8fde2aaa8d1e86f0094dd6444.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/89b024e1a2f875c030b693824a826cb1c7d024a8fde2aaa8d1e86f0094dd6444.jpg)

![a0ba5282cbdccdb0e539583e4771ba092c7c5be2e8f415d6e8545f26f92c6111.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/a0ba5282cbdccdb0e539583e4771ba092c7c5be2e8f415d6e8545f26f92c6111.jpg)

![a91e00b178c67acd1f2d52369b2e58170e6adac211d44669d64dfd9b8487a531.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/a91e00b178c67acd1f2d52369b2e58170e6adac211d44669d64dfd9b8487a531.jpg)

![b3454b768e3f3472a5f1c125b433836bd31217db16ba19ca55da278d9e3bb6ba.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/b3454b768e3f3472a5f1c125b433836bd31217db16ba19ca55da278d9e3bb6ba.jpg)

![bce35369ece611cef2680268fe1bffb0a4dc1af5de5cbb0426659039b40c4e87.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/bce35369ece611cef2680268fe1bffb0a4dc1af5de5cbb0426659039b40c4e87.jpg)

![bf8812fdab158c73212792265c80f2fb0a00f1788421801128f29fa1c77ff7d9.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/bf8812fdab158c73212792265c80f2fb0a00f1788421801128f29fa1c77ff7d9.jpg)

![e221aa7f649944ffe747b29c71e33ccbf461ed1858dc8207b5440be3ec376491.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/e221aa7f649944ffe747b29c71e33ccbf461ed1858dc8207b5440be3ec376491.jpg)

![e3f3206583c57ab318c901ac11c68c713ab20289104788a217a55df682696fc4.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/e3f3206583c57ab318c901ac11c68c713ab20289104788a217a55df682696fc4.jpg)

![ea9c0bb1b5e83ff80ac7815441b0298eb887d48989b96daed56d36e7a59d9fc8.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/ea9c0bb1b5e83ff80ac7815441b0298eb887d48989b96daed56d36e7a59d9fc8.jpg)

![f0cdb980a9a9c354d7caf96af072f185fd90ca10a6e43e6082acb64ab142065d.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/f0cdb980a9a9c354d7caf96af072f185fd90ca10a6e43e6082acb64ab142065d.jpg)

![f14a564c34359281d88ece6a940702862be98379f2c6172fc1d862ebe2e87b42.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/f14a564c34359281d88ece6a940702862be98379f2c6172fc1d862ebe2e87b42.jpg)

![f7561d8a8cc8dfcd2e54e13ae4859b4eee23cb85d3086d850803e10365e19aea.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/f7561d8a8cc8dfcd2e54e13ae4859b4eee23cb85d3086d850803e10365e19aea.jpg)

![fad8538a827860857ff9f8d50d8b400d9e39229c9617448e36e1f697b3f60f7d.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/images/fad8538a827860857ff9f8d50d8b400d9e39229c9617448e36e1f697b3f60f7d.jpg)

### Tables

![01df75a43a8178264a8a32b353a8921c8a965ae288733f3d45f9f8a20e365eb6.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/01df75a43a8178264a8a32b353a8921c8a965ae288733f3d45f9f8a20e365eb6.jpg)

![171c7f4893c637162494ad653b18599b8e473c7f30931d5dd4cb3a14c34cfce2.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/171c7f4893c637162494ad653b18599b8e473c7f30931d5dd4cb3a14c34cfce2.jpg)

![1dd16fbe2c8e18b8a649aeec49700caf6f7d27f4bb49eb767de6cb7bfb4b2a5f.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/1dd16fbe2c8e18b8a649aeec49700caf6f7d27f4bb49eb767de6cb7bfb4b2a5f.jpg)

![1e24e1e897d75408c846531533396a7133ce02bf8e74a187b6320db6b4987101.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/1e24e1e897d75408c846531533396a7133ce02bf8e74a187b6320db6b4987101.jpg)

![48aa711d0808f5fe194c7b6e1a16a7b1c974a1944e7846d21b02e8a6ac3dd3d1.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/48aa711d0808f5fe194c7b6e1a16a7b1c974a1944e7846d21b02e8a6ac3dd3d1.jpg)

![600c0afc9943ad123bfe3446205b9339c212a3e0bee7509b340d73bb38bf6ead.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/600c0afc9943ad123bfe3446205b9339c212a3e0bee7509b340d73bb38bf6ead.jpg)

![95854d0fd8a251b1570bfa750946840c89fcee09a3c6ba0d63dacc050facba21.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/95854d0fd8a251b1570bfa750946840c89fcee09a3c6ba0d63dacc050facba21.jpg)

![b071b58b7c6d172f3dc86132d1737da6f29cb007fed57bbc1e983906c487ef20.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/b071b58b7c6d172f3dc86132d1737da6f29cb007fed57bbc1e983906c487ef20.jpg)

![bbd5e252b1ca39432c41e6fe0f8ef7c2a4100e817324bcfc122779f036fab1cd.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/bbd5e252b1ca39432c41e6fe0f8ef7c2a4100e817324bcfc122779f036fab1cd.jpg)

![d1fde61b1c7d36aeeb570fee76ca9b912db648e1c43ef416dad2c1c3060eafb6.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/d1fde61b1c7d36aeeb570fee76ca9b912db648e1c43ef416dad2c1c3060eafb6.jpg)

![db5b2575daad77cf030bd8552b44a9db142a249f71f7e5d5fe6dcc6a54194a1f.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/db5b2575daad77cf030bd8552b44a9db142a249f71f7e5d5fe6dcc6a54194a1f.jpg)

![e01f500f6a2cf8979810c2ad165b8ad5bc80e00dbb70490e0e27db0293c48b04.jpg](../icml_results/88_Causal%20Modeling%20of%20Selection%20in%20Evolution/tables/e01f500f6a2cf8979810c2ad165b8ad5bc80e00dbb70490e0e27db0293c48b04.jpg)

<a id="paper-0088"></a>

## Second-Order Smooth Planning with Optimal-Transport Bellman Smoothing


### Tables

![0fc67b5ce25c8c02cd7737ae0c726c70e3376db889f8d5a192f8cf8a5e947396.jpg](../icml_results/89_Second-Order%20Smooth%20Planning%20with%20Optimal-Transport%20Bellman%20Smoothing/tables/0fc67b5ce25c8c02cd7737ae0c726c70e3376db889f8d5a192f8cf8a5e947396.jpg)

![3393bfdc90fd34fe9eca1867ff2f41026a339965d615ce6f5c08979d11d6f2fe.jpg](../icml_results/89_Second-Order%20Smooth%20Planning%20with%20Optimal-Transport%20Bellman%20Smoothing/tables/3393bfdc90fd34fe9eca1867ff2f41026a339965d615ce6f5c08979d11d6f2fe.jpg)

![390b36ef6b5c9445f5d7ea7840384e9e112bf2e134fc0b33e31ba7abb8a3e3f2.jpg](../icml_results/89_Second-Order%20Smooth%20Planning%20with%20Optimal-Transport%20Bellman%20Smoothing/tables/390b36ef6b5c9445f5d7ea7840384e9e112bf2e134fc0b33e31ba7abb8a3e3f2.jpg)

![482149f3e5d0f25a5782fddc1d8756ef4e65aa63754f6b9d349bdf7663685106.jpg](../icml_results/89_Second-Order%20Smooth%20Planning%20with%20Optimal-Transport%20Bellman%20Smoothing/tables/482149f3e5d0f25a5782fddc1d8756ef4e65aa63754f6b9d349bdf7663685106.jpg)

<a id="paper-0089"></a>

## Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models


### Images

![149949148891bd1acda72827b12f6c44e1033fc315d4c0e48fc5a7946b338e6c.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/images/149949148891bd1acda72827b12f6c44e1033fc315d4c0e48fc5a7946b338e6c.jpg)

![3c4930809c69ef8797ca8b2ead8078401275abf8537ab85ecdeb071cdbf5e137.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/images/3c4930809c69ef8797ca8b2ead8078401275abf8537ab85ecdeb071cdbf5e137.jpg)

![73dce8111f94dd4accbe696496f03a5f598a588dfb0a4ba39dc68a3ed28c6dbe.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/images/73dce8111f94dd4accbe696496f03a5f598a588dfb0a4ba39dc68a3ed28c6dbe.jpg)

### Tables

![059a7e9ddcccd9949b3ae7246ffe9fbe4da1c5ffb538f71cd92c08f7a9e0fb4b.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/059a7e9ddcccd9949b3ae7246ffe9fbe4da1c5ffb538f71cd92c08f7a9e0fb4b.jpg)

![1731568aee599b9bc7775bb5047845fe0c5073e23332e5b6480d5893d300255f.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/1731568aee599b9bc7775bb5047845fe0c5073e23332e5b6480d5893d300255f.jpg)

![243bf5037da419c781c5f9352f04f7e2c21ec91c0a7dc553b24b0e9b48680f75.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/243bf5037da419c781c5f9352f04f7e2c21ec91c0a7dc553b24b0e9b48680f75.jpg)

![4d142addd2183fae7d58c9e56b5d602eee0686f15c57f5d33a244aaea3685a12.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/4d142addd2183fae7d58c9e56b5d602eee0686f15c57f5d33a244aaea3685a12.jpg)

![c1d0952cdcfb152b2712babc8ed920e15167b76e72b4677ff9ba069a2696a557.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/c1d0952cdcfb152b2712babc8ed920e15167b76e72b4677ff9ba069a2696a557.jpg)

![d79afa8e4b7b7b92f5bd058cc8e8b35879a9e28a3a30fee87e31f3406670ef94.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/d79afa8e4b7b7b92f5bd058cc8e8b35879a9e28a3a30fee87e31f3406670ef94.jpg)

![f47dfab9e0c8ee436abf2520a5055ac61092e27e86b211f69de6b5c8f245b939.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/f47dfab9e0c8ee436abf2520a5055ac61092e27e86b211f69de6b5c8f245b939.jpg)

![fe13109c5ea399749d57a4ec5c459f282482adb83af2435e06c3c4c5e4f8e797.jpg](../icml_results/90_Geometry-Aware%20Decoding%20with%20Wasserstein-Regularized%20Truncation%20and%20Mass%20Penalties%20for%20Large%20Languag/tables/fe13109c5ea399749d57a4ec5c459f282482adb83af2435e06c3c4c5e4f8e797.jpg)

<a id="paper-0090"></a>

## On the Expressive Power of Permutation-Equivariant Weight-Space Networks


### Images

![1d97434d89af4a1f294442983458f0e5ae7f1c12d88ce387b5eec28d68905a56.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/images/1d97434d89af4a1f294442983458f0e5ae7f1c12d88ce387b5eec28d68905a56.jpg)

![217e99987c2a342865f72e32cd2a9b16561c18c0d80886a11ffe53d43ad2cdb9.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/images/217e99987c2a342865f72e32cd2a9b16561c18c0d80886a11ffe53d43ad2cdb9.jpg)

![244a78900488765cbc4cd33f90d31fbd0b45b060c3135d21ba0dbc44afe34aa9.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/images/244a78900488765cbc4cd33f90d31fbd0b45b060c3135d21ba0dbc44afe34aa9.jpg)

![29163c5ae089aac73c88fd24d823ab64eda8ebc816e8c9c4a1e72d96e0404985.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/images/29163c5ae089aac73c88fd24d823ab64eda8ebc816e8c9c4a1e72d96e0404985.jpg)

![6ce6ad5f2e80e1af0630cc26976d93c04620af7467196443f10568b0d4d46464.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/images/6ce6ad5f2e80e1af0630cc26976d93c04620af7467196443f10568b0d4d46464.jpg)

### Tables

![4b4d31310fc91e3b5895ca56c9fb7ddcb6318affe7ee7a0f7527a14ca87b0e6a.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/tables/4b4d31310fc91e3b5895ca56c9fb7ddcb6318affe7ee7a0f7527a14ca87b0e6a.jpg)

![7c5f5f48f317c286dd4a235b2d2e5eb906e797dc9a555eec07705b59e81e230c.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/tables/7c5f5f48f317c286dd4a235b2d2e5eb906e797dc9a555eec07705b59e81e230c.jpg)

![d113a19a248c0efb52f00b8d2d1201ad67e289eb152ba5a11f44ecd3521c1734.jpg](../icml_results/91_On%20the%20Expressive%20Power%20of%20Permutation-Equivariant%20Weight-Space%20Networks/tables/d113a19a248c0efb52f00b8d2d1201ad67e289eb152ba5a11f44ecd3521c1734.jpg)

<a id="paper-0091"></a>

## Adaptive Memory Retention in Dynamic Graphs


### Images

![4aa4e1856066510b4dbcce10eef67bcdb11c65d1b4a365a8e93b3694dcf1ad99.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/4aa4e1856066510b4dbcce10eef67bcdb11c65d1b4a365a8e93b3694dcf1ad99.jpg)

![6ae3ce04b47ba6f7ae794f38ccacd59cd7e975e4eed144bb97a65fdf460053e1.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/6ae3ce04b47ba6f7ae794f38ccacd59cd7e975e4eed144bb97a65fdf460053e1.jpg)

![9bd269046606758ef622472c2264fad56f9cc8b46eb0c462d2a77e9bacffda85.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/9bd269046606758ef622472c2264fad56f9cc8b46eb0c462d2a77e9bacffda85.jpg)

![d67e167d605576a1884abf8079df429b2465064bc47d6c1ab0cf1d425a09a314.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/d67e167d605576a1884abf8079df429b2465064bc47d6c1ab0cf1d425a09a314.jpg)

![d98baea7da8cc4e60901fc32a1ae73b7d52c55c64d4203c8cf57af4c38290ee7.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/d98baea7da8cc4e60901fc32a1ae73b7d52c55c64d4203c8cf57af4c38290ee7.jpg)

![f495f0a707fbc24d8e8c94745585cd5691e55539cb22e495006a712a2bb3b5f8.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/f495f0a707fbc24d8e8c94745585cd5691e55539cb22e495006a712a2bb3b5f8.jpg)

![f555bccb521c46e08148637c56ad4db750ad7f1f3a782ee370b67730dba5fcd2.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/images/f555bccb521c46e08148637c56ad4db750ad7f1f3a782ee370b67730dba5fcd2.jpg)

### Tables

![02b1721f974f4f8412d7452d6f0c03c9029e53a6d35c947e5af57908385630f4.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/02b1721f974f4f8412d7452d6f0c03c9029e53a6d35c947e5af57908385630f4.jpg)

![0927b0077e94ad1d9892f7a41b4c7aa381a438906012bbaf40b1fc9a36b7246e.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/0927b0077e94ad1d9892f7a41b4c7aa381a438906012bbaf40b1fc9a36b7246e.jpg)

![3039e17002aa198278e3952b7843ad4aaadfbd5ad93b80c70caf704b0bbba55b.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/3039e17002aa198278e3952b7843ad4aaadfbd5ad93b80c70caf704b0bbba55b.jpg)

![568a9c4cf90949c2b8c55f077fd09983e64f0ca22347562eb3ed7963852281c1.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/568a9c4cf90949c2b8c55f077fd09983e64f0ca22347562eb3ed7963852281c1.jpg)

![8807e4d1fa4fbb3950cf7e8a7248637a3f82c927a05632f31bf38c53c0ba0def.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/8807e4d1fa4fbb3950cf7e8a7248637a3f82c927a05632f31bf38c53c0ba0def.jpg)

![8fe7ec9f399989d8f40a5f8d01c862d157b50d451743db049d69d214cd358a4c.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/8fe7ec9f399989d8f40a5f8d01c862d157b50d451743db049d69d214cd358a4c.jpg)

![b9fde609d0edffeb445e3d0d4a354e81f4216e899dc668a7efbd7fcf3d859590.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/b9fde609d0edffeb445e3d0d4a354e81f4216e899dc668a7efbd7fcf3d859590.jpg)

![d239af01a1f225281f358d9c86b8d4221347d229fdc63bee249d86fdab178ef5.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/d239af01a1f225281f358d9c86b8d4221347d229fdc63bee249d86fdab178ef5.jpg)

![d2a7fea7451cbff4b8f30a1d3399605fd0a895c55ead70526bc4d7ea74d6d3bf.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/d2a7fea7451cbff4b8f30a1d3399605fd0a895c55ead70526bc4d7ea74d6d3bf.jpg)

![d778681e817d16c9d8fa8bd306d4b0978bd4d85f6e8dd5e82725e23392a722c7.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/d778681e817d16c9d8fa8bd306d4b0978bd4d85f6e8dd5e82725e23392a722c7.jpg)

![e4550dc0a95306206dfbaf0348e8deba2c99def94e6e12fb27aacbb2cb21d895.jpg](../icml_results/92_Adaptive%20Memory%20Retention%20in%20Dynamic%20Graphs/tables/e4550dc0a95306206dfbaf0348e8deba2c99def94e6e12fb27aacbb2cb21d895.jpg)

<a id="paper-0092"></a>

## Mixture of Concept Bottleneck Experts


### Images

![0c26b943ca936dc390d39d84dc2ca3f9ecc690d7cdbc28dd01a9e16968957db3.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/0c26b943ca936dc390d39d84dc2ca3f9ecc690d7cdbc28dd01a9e16968957db3.jpg)

![0f7bedec7ecbbc4c3dfda904aed28bc47e9d42795b26614ab8a5ffa5388fa224.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/0f7bedec7ecbbc4c3dfda904aed28bc47e9d42795b26614ab8a5ffa5388fa224.jpg)

![11c39481255d37afa41d3b763300a62712d6a119eb56fb2efe4fb369e32d4a0c.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/11c39481255d37afa41d3b763300a62712d6a119eb56fb2efe4fb369e32d4a0c.jpg)

![2be83c09128d47784ee78ef90304f44e433d5b2cd92e616e1a7eb1b1cee194f1.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/2be83c09128d47784ee78ef90304f44e433d5b2cd92e616e1a7eb1b1cee194f1.jpg)

![34124c06ac92400a71ee462051598c1e07dfdfab2d0342cd673afea4c944efcf.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/34124c06ac92400a71ee462051598c1e07dfdfab2d0342cd673afea4c944efcf.jpg)

![4b77552454bd147714aa11ff2008c81a559ac42320eeba5942056d10b5ee3b3e.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/4b77552454bd147714aa11ff2008c81a559ac42320eeba5942056d10b5ee3b3e.jpg)

![53624bfaa798f4af3012e201f8f857b0f93182a63990594bd5f61ad415595620.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/53624bfaa798f4af3012e201f8f857b0f93182a63990594bd5f61ad415595620.jpg)

![53a6434d5f1d6f9c7793964a556f4cce95fc9bc07346814c17627b632c4cc879.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/53a6434d5f1d6f9c7793964a556f4cce95fc9bc07346814c17627b632c4cc879.jpg)

![d4868b6307dc381aff53d1ea1928196154a0cfe5f7c606096e0da8570f49613b.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/images/d4868b6307dc381aff53d1ea1928196154a0cfe5f7c606096e0da8570f49613b.jpg)

### Tables

![040cdac351666e90a1cf491abf6f216eb548e8448dbd37db83f3612a17425c20.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/040cdac351666e90a1cf491abf6f216eb548e8448dbd37db83f3612a17425c20.jpg)

![0a997db69b72802b298d1c31a444f4fa49e2b6cb38d362ec8039d41fae153705.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/0a997db69b72802b298d1c31a444f4fa49e2b6cb38d362ec8039d41fae153705.jpg)

![464543b3c0cf495655ce30946e91599941ad883ab0eb4bf4bf0a1a5962ae8246.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/464543b3c0cf495655ce30946e91599941ad883ab0eb4bf4bf0a1a5962ae8246.jpg)

![49b2b45982cef53560c965769ccc397cfa13b9dccf51d66aa3640516f8c8ebb6.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/49b2b45982cef53560c965769ccc397cfa13b9dccf51d66aa3640516f8c8ebb6.jpg)

![549728525070f059f96e31bdd0c0a977ad4a510d7abb1a365b16ceb51d981cd9.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/549728525070f059f96e31bdd0c0a977ad4a510d7abb1a365b16ceb51d981cd9.jpg)

![6fdffb693534b39e082d926972a571b00b4926669cfeff0353e71ddea43fdd3d.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/6fdffb693534b39e082d926972a571b00b4926669cfeff0353e71ddea43fdd3d.jpg)

![8c11b171a58cd40b7e8a9b7702e429f716f2a646d291552a85554f8df9e898ae.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/8c11b171a58cd40b7e8a9b7702e429f716f2a646d291552a85554f8df9e898ae.jpg)

![a3a2a37f0c2ce1be0a7f2288b49eb2f00c377f10bd40448d29393a3cf946a2a9.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/a3a2a37f0c2ce1be0a7f2288b49eb2f00c377f10bd40448d29393a3cf946a2a9.jpg)

![ab30bc06da6b0c56e5ed668ae8efc2c62da35f58a52cba97a1eac1921e7a6806.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/ab30bc06da6b0c56e5ed668ae8efc2c62da35f58a52cba97a1eac1921e7a6806.jpg)

![aecc94a48171e8b9ebc07848d9e7bcdddf623cb3c9c85d2536cc1d6aca676dd4.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/aecc94a48171e8b9ebc07848d9e7bcdddf623cb3c9c85d2536cc1d6aca676dd4.jpg)

![beb8d767e46a14ddbea0995f5328a656c6c3458b3ac6965f85897ead0efcc62f.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/beb8d767e46a14ddbea0995f5328a656c6c3458b3ac6965f85897ead0efcc62f.jpg)

![bef849c47bd6e93aa998025313a14f019723b34a8cdb204e9395764c7a7018f5.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/bef849c47bd6e93aa998025313a14f019723b34a8cdb204e9395764c7a7018f5.jpg)

![e922c36b4fabc0d5525da523184cce14f73ba5ba962aa48180896252516303d3.jpg](../icml_results/93_Mixture%20of%20Concept%20Bottleneck%20Experts/tables/e922c36b4fabc0d5525da523184cce14f73ba5ba962aa48180896252516303d3.jpg)

<a id="paper-0093"></a>

## Beyond First-order Asymptotics in Sequential Mean Testing


### Images

![08c82f9eaca862a615e812c6f2e64b4946384b502521992756797c36d7ce7f6c.jpg](../icml_results/94_Beyond%20First-order%20Asymptotics%20in%20Sequential%20Mean%20Testing/images/08c82f9eaca862a615e812c6f2e64b4946384b502521992756797c36d7ce7f6c.jpg)

![4b932dc8192898f47f925322a4a116d22eab0dda70b87a5a316160cf5878026e.jpg](../icml_results/94_Beyond%20First-order%20Asymptotics%20in%20Sequential%20Mean%20Testing/images/4b932dc8192898f47f925322a4a116d22eab0dda70b87a5a316160cf5878026e.jpg)

![5924326ded5b118214ff982aa29afb21137efafc00599c131a90ea1a0b14be0e.jpg](../icml_results/94_Beyond%20First-order%20Asymptotics%20in%20Sequential%20Mean%20Testing/images/5924326ded5b118214ff982aa29afb21137efafc00599c131a90ea1a0b14be0e.jpg)

![848ba2f1f3a5d127e2340eb9dc7111beda03a7ad6429e448629522a0a911c2fb.jpg](../icml_results/94_Beyond%20First-order%20Asymptotics%20in%20Sequential%20Mean%20Testing/images/848ba2f1f3a5d127e2340eb9dc7111beda03a7ad6429e448629522a0a911c2fb.jpg)

![ddab3dfcae99bf56b61dd52277c3c01fa6126b9f2a01d7a700ea3a2c7dda75c1.jpg](../icml_results/94_Beyond%20First-order%20Asymptotics%20in%20Sequential%20Mean%20Testing/images/ddab3dfcae99bf56b61dd52277c3c01fa6126b9f2a01d7a700ea3a2c7dda75c1.jpg)

<a id="paper-0094"></a>

## A Noise Sensitivity Exponent Controls Large Statistical-to-Computational Gaps in Single- and Multi-Index Models


### Images

![f95738f367e133f3ccdf44105ac35026fc29d272bfb4dfadc2371965cd0549f2.jpg](../icml_results/95_A%20Noise%20Sensitivity%20Exponent%20Controls%20Large%20Statistical-to-Computational%20Gaps%20in%20Single-%20and%20Multi-I/images/f95738f367e133f3ccdf44105ac35026fc29d272bfb4dfadc2371965cd0549f2.jpg)

<a id="paper-0095"></a>

## BFTS: Thompson Sampling with Bayesian Additive Regression Trees


### Images

![024d36d48b6d609a1f72f8b6d9eb7bfa0d70366cff4954a9c4ebda3c686e5659.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/024d36d48b6d609a1f72f8b6d9eb7bfa0d70366cff4954a9c4ebda3c686e5659.jpg)

![14f53c5d5cd51038bf564d8343de7bd0d4ed06120837702a9ca323586e3937d4.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/14f53c5d5cd51038bf564d8343de7bd0d4ed06120837702a9ca323586e3937d4.jpg)

![240bc165b67394be76cd0e13118a982ca89d2e580e5b5974e334dccbc4fbb59e.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/240bc165b67394be76cd0e13118a982ca89d2e580e5b5974e334dccbc4fbb59e.jpg)

![389bfc27ef789d66ae8c50727b6355d8f7641aa54a1550ef751c6187d3c04544.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/389bfc27ef789d66ae8c50727b6355d8f7641aa54a1550ef751c6187d3c04544.jpg)

![701faa67796e114023387962ad46e55c3f5720f70bb1453d6a4e240d0b824d47.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/701faa67796e114023387962ad46e55c3f5720f70bb1453d6a4e240d0b824d47.jpg)

![7d3ef7e518c63f72e7cb7e285caba475dafe45d7e6e23a1b405970e403202c81.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/7d3ef7e518c63f72e7cb7e285caba475dafe45d7e6e23a1b405970e403202c81.jpg)

![80ff488b0d45edc9eee022c892c009a151b34067fa69973dfae65160f42a7467.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/80ff488b0d45edc9eee022c892c009a151b34067fa69973dfae65160f42a7467.jpg)

![824d52ba6d27b88e0a2da4ea114ca6c97e02e7bee19b98a9c1927fa4ed0468d3.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/824d52ba6d27b88e0a2da4ea114ca6c97e02e7bee19b98a9c1927fa4ed0468d3.jpg)

![93dc1f927d5af315a723ef62a23b8c32e527c5eadc6d59ae19d0b2ffdd5806be.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/93dc1f927d5af315a723ef62a23b8c32e527c5eadc6d59ae19d0b2ffdd5806be.jpg)

![9d44478189660af8712802dd562935e9605759f62f452778c886f3c01759e6ae.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/9d44478189660af8712802dd562935e9605759f62f452778c886f3c01759e6ae.jpg)

![a231b024071532db5291bb47ae170400986fc785eff2f1aa636ac618dff6b70d.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/a231b024071532db5291bb47ae170400986fc785eff2f1aa636ac618dff6b70d.jpg)

![aa85cdc9dc4b2303c873fd9abe4962627823aa81d39503ae2101a5f9e26080d0.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/aa85cdc9dc4b2303c873fd9abe4962627823aa81d39503ae2101a5f9e26080d0.jpg)

![c2ba39ba17fc7ad256da564f7f712f047c9ba03ec1d8c65e7ead9583efdbbd8c.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/images/c2ba39ba17fc7ad256da564f7f712f047c9ba03ec1d8c65e7ead9583efdbbd8c.jpg)

### Tables

![33066be5427775fcf64affa3441e2e7cdb5f1fb361e8820fd0bd7934b8a4c747.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/33066be5427775fcf64affa3441e2e7cdb5f1fb361e8820fd0bd7934b8a4c747.jpg)

![354f3d783e37feb7daaaed03a7fdd1f87abfab78ced22d660b89a43fea52f751.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/354f3d783e37feb7daaaed03a7fdd1f87abfab78ced22d660b89a43fea52f751.jpg)

![4d2700307e2e78d8ad2eedff13e07f89e0f693983122109af06bb82fdbebdac1.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/4d2700307e2e78d8ad2eedff13e07f89e0f693983122109af06bb82fdbebdac1.jpg)

![5ef91189c804cbebc20f3564a78e4cd764370b212586823ee6f1f4a6900efa5b.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/5ef91189c804cbebc20f3564a78e4cd764370b212586823ee6f1f4a6900efa5b.jpg)

![909655e074142405f7c139941b1678b68a8853ffb1d0d2b7a5fa14e6201a132e.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/909655e074142405f7c139941b1678b68a8853ffb1d0d2b7a5fa14e6201a132e.jpg)

![954a2e571d80c10868d7085428f02e3425cd7da45e0ee949eb3cb907222f1d48.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/954a2e571d80c10868d7085428f02e3425cd7da45e0ee949eb3cb907222f1d48.jpg)

![98032591469b0edfe3214a164939182a85d1cd86a5cb4ad6c3e1c98553109eea.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/98032591469b0edfe3214a164939182a85d1cd86a5cb4ad6c3e1c98553109eea.jpg)

![dd63888eda1320f7bcf7e23c1e8dc98b9498724686dadc6f4a54a5ba07bca9d0.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/dd63888eda1320f7bcf7e23c1e8dc98b9498724686dadc6f4a54a5ba07bca9d0.jpg)

![e74db73d5b95cc67ac363a9af3a3be3a95f9d65f21910a06df3bb1f44551f7c8.jpg](../icml_results/96_BFTS_%20Thompson%20Sampling%20with%20Bayesian%20Additive%20Regression%20Trees/tables/e74db73d5b95cc67ac363a9af3a3be3a95f9d65f21910a06df3bb1f44551f7c8.jpg)

<a id="paper-0096"></a>

## LASER: Learning Active Sensing for Continuum Field Reconstruction


### Images

![014c18619bc1d7ce25a1e0515c0618d278f753fa32f8ffda03b2cf1089b64152.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/014c18619bc1d7ce25a1e0515c0618d278f753fa32f8ffda03b2cf1089b64152.jpg)

![04d45679acd4baf7d06ceb6af8e9ad092c9885795a7ebf2d2b34b5f8a3114615.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/04d45679acd4baf7d06ceb6af8e9ad092c9885795a7ebf2d2b34b5f8a3114615.jpg)

![1322bdf365ebede0a24c6acb651c6d70be14aa06332dd021443bd85e5e27850b.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/1322bdf365ebede0a24c6acb651c6d70be14aa06332dd021443bd85e5e27850b.jpg)

![1e0307d98bf585ecb184780e94f3a384ed13c18a8277e6557622e535ef80eff6.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/1e0307d98bf585ecb184780e94f3a384ed13c18a8277e6557622e535ef80eff6.jpg)

![32ba16684455e304ceaa3f2d3ed1788fea646dbc27fd22101cb596349aec7bac.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/32ba16684455e304ceaa3f2d3ed1788fea646dbc27fd22101cb596349aec7bac.jpg)

![3511ff0653730eaa7d0dd49d0019f30cb90bcd592bec691baa9597c2e9123e28.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/3511ff0653730eaa7d0dd49d0019f30cb90bcd592bec691baa9597c2e9123e28.jpg)

![41b95fe5ad673c9ef3c494c44ad59b424e321ba25fe5058d423cfe3362ce0a20.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/41b95fe5ad673c9ef3c494c44ad59b424e321ba25fe5058d423cfe3362ce0a20.jpg)

![4b319d549773e32c31cb48100b58c6980f29804964fd6464a482f62bba65356e.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/4b319d549773e32c31cb48100b58c6980f29804964fd6464a482f62bba65356e.jpg)

![5c97037b18179bf9396486e95b1e0aca4aafa13ac145f0987d1c7e40767a7d4f.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/5c97037b18179bf9396486e95b1e0aca4aafa13ac145f0987d1c7e40767a7d4f.jpg)

![7c23940e886c297b3df990fa39d046008e17a36d384725b8e18a0828067c1351.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/7c23940e886c297b3df990fa39d046008e17a36d384725b8e18a0828067c1351.jpg)

![8175c2c7c4b602a2a5209785d4f99c33b0063c04f33ba05f4fe16e3e212d68dc.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/8175c2c7c4b602a2a5209785d4f99c33b0063c04f33ba05f4fe16e3e212d68dc.jpg)

![92d71d52072e120c5f1ddb9a284ceebad87ba1341db6231eef6fcedbf76ec36a.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/92d71d52072e120c5f1ddb9a284ceebad87ba1341db6231eef6fcedbf76ec36a.jpg)

![974912b80d1fdc9853ab57ff03cbf60efd8f1969629fbcb22882e8661af47e1f.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/974912b80d1fdc9853ab57ff03cbf60efd8f1969629fbcb22882e8661af47e1f.jpg)

![a1e504d9005a62d5a1a0e47bad53c8178220f9fe3fdd032ddda5d177d0f05d60.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/a1e504d9005a62d5a1a0e47bad53c8178220f9fe3fdd032ddda5d177d0f05d60.jpg)

![afb665fd5ce176e99e5c323874cf78126f4af654acb9f043f3fe4bf25d1f4a38.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/afb665fd5ce176e99e5c323874cf78126f4af654acb9f043f3fe4bf25d1f4a38.jpg)

![df741ae70becd78660b0fa0fa0c12339586492f616af5ceecd6fab449fefaf0e.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/df741ae70becd78660b0fa0fa0c12339586492f616af5ceecd6fab449fefaf0e.jpg)

![e162fba58d5d4c587d57cb0d9dc73de02da43c19f35b8123669b77efd45e8d91.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/e162fba58d5d4c587d57cb0d9dc73de02da43c19f35b8123669b77efd45e8d91.jpg)

![ea51470b47908d4162d6d6c8b36ccb6da9fff0df134c0480cc6e4d3233bfee8c.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/ea51470b47908d4162d6d6c8b36ccb6da9fff0df134c0480cc6e4d3233bfee8c.jpg)

![eb5b11789c0010fa8e4b789a048b25871cab5052092eec97c6e13f184b27e2ca.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/images/eb5b11789c0010fa8e4b789a048b25871cab5052092eec97c6e13f184b27e2ca.jpg)

### Tables

![1ec46a2e2d508c1cfddfc09dd8ddf2cd15b7c1d07a30ea8948deb81f65a6164c.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/1ec46a2e2d508c1cfddfc09dd8ddf2cd15b7c1d07a30ea8948deb81f65a6164c.jpg)

![431c2722431f0a5b9d3abc73003896859417267174cfd8482b9299ca01dbdda7.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/431c2722431f0a5b9d3abc73003896859417267174cfd8482b9299ca01dbdda7.jpg)

![6a3c3c109df0559437879bbab2cad533cdcbf59a78960ed2758180d9814db94e.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/6a3c3c109df0559437879bbab2cad533cdcbf59a78960ed2758180d9814db94e.jpg)

![73e752c1bdbdec98c3bd375bcca889952de627a4494b15384cc45c61edd2823b.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/73e752c1bdbdec98c3bd375bcca889952de627a4494b15384cc45c61edd2823b.jpg)

![946b1c19e24d10b1415d0263c3a12d1faba8a11b7ed96e1d14d7c1598bb5165c.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/946b1c19e24d10b1415d0263c3a12d1faba8a11b7ed96e1d14d7c1598bb5165c.jpg)

![a079c9e21fc0d41fdb532287ddb0776a6bea0716ebe29ea2a55f08c1637b5e76.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/a079c9e21fc0d41fdb532287ddb0776a6bea0716ebe29ea2a55f08c1637b5e76.jpg)

![ae618e9e53e4af0d703d5642cf3da1efd593e80640587a8fa01d2b218a4551e7.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/ae618e9e53e4af0d703d5642cf3da1efd593e80640587a8fa01d2b218a4551e7.jpg)

![b63ba347e804c8aea140d82770e0fd43994b431ae4df05084b0ae9021951eae8.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/b63ba347e804c8aea140d82770e0fd43994b431ae4df05084b0ae9021951eae8.jpg)

![b92f2a36037f70f3f3c1fbd2653907f43e4e16fd4b60f20838e97372c81bef9a.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/b92f2a36037f70f3f3c1fbd2653907f43e4e16fd4b60f20838e97372c81bef9a.jpg)

![b959c6ac67529d2e4e750ffa5e340885b1b96f7b81ac36408344941a91845efb.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/b959c6ac67529d2e4e750ffa5e340885b1b96f7b81ac36408344941a91845efb.jpg)

![cce2f2590028dd575dab19c8038d4b53de30e0ee3b8c6bc92827459ca6e7e57c.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/cce2f2590028dd575dab19c8038d4b53de30e0ee3b8c6bc92827459ca6e7e57c.jpg)

![e6fff535faae39285b5280eb2bf3351eae7211eef3dc79d87847e9e950207197.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/e6fff535faae39285b5280eb2bf3351eae7211eef3dc79d87847e9e950207197.jpg)

![f94a5fe0e7e9cb156ad17ba446a875db109e27a4d8df04737fb9e0ced2ec7df7.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/f94a5fe0e7e9cb156ad17ba446a875db109e27a4d8df04737fb9e0ced2ec7df7.jpg)

![fd71245eda0d2223622f542a70f564413581cd57833ce62548d9a83349b983c2.jpg](../icml_results/97_LASER_%20Learning%20Active%20Sensing%20for%20Continuum%20Field%20Reconstruction/tables/fd71245eda0d2223622f542a70f564413581cd57833ce62548d9a83349b983c2.jpg)

<a id="paper-0097"></a>

## FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications


### Images

![0cb60d3aa19f4a3af6e9799bc8bd23d6ffd295ffef99d3879ae8fe31eaca55c7.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/0cb60d3aa19f4a3af6e9799bc8bd23d6ffd295ffef99d3879ae8fe31eaca55c7.jpg)

![0d88255aa19a013dbbb067dd83cb9a0343fd4fc50875e3150841052db22fbd1f.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/0d88255aa19a013dbbb067dd83cb9a0343fd4fc50875e3150841052db22fbd1f.jpg)

![0e79e26c795056405bdd74c9e9ffcf715d1510d597b1ad10f8aec3c7193247ba.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/0e79e26c795056405bdd74c9e9ffcf715d1510d597b1ad10f8aec3c7193247ba.jpg)

![194ab2b4b559c6a5f3280de0e53aeef9171eb39e4662e9c25920c546105020f2.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/194ab2b4b559c6a5f3280de0e53aeef9171eb39e4662e9c25920c546105020f2.jpg)

![19705f1c11ff72503fb5c8b351db9ab20d4d086e3ae1114f6895801853f18f92.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/19705f1c11ff72503fb5c8b351db9ab20d4d086e3ae1114f6895801853f18f92.jpg)

![1daf03ee838d4cefcd0826a89bb47776104820d4e3027a9cdb60e781943b710a.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/1daf03ee838d4cefcd0826a89bb47776104820d4e3027a9cdb60e781943b710a.jpg)

![1ecc3b27dc1c690f86099ba06542b39aa2281353383e2ba07597080a85865cf6.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/1ecc3b27dc1c690f86099ba06542b39aa2281353383e2ba07597080a85865cf6.jpg)

![260bbf16b47d42008aa65c112e24a2a269d0cd12f260cb9a3910c12d4b92353d.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/260bbf16b47d42008aa65c112e24a2a269d0cd12f260cb9a3910c12d4b92353d.jpg)

![31325ad06a7dc8d224ebf12f714bda1c447beec9030c457418fe01ecc75d107a.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/31325ad06a7dc8d224ebf12f714bda1c447beec9030c457418fe01ecc75d107a.jpg)

![3d14e7bd34e8828e899d99b3f94b5c68d0d9e1f2d9931ddd21ceb72ddc113efc.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/3d14e7bd34e8828e899d99b3f94b5c68d0d9e1f2d9931ddd21ceb72ddc113efc.jpg)

![3ffda14d0a42f99fa390b37eee8be477acbd7c471721a3c755071f9ec2c604ba.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/3ffda14d0a42f99fa390b37eee8be477acbd7c471721a3c755071f9ec2c604ba.jpg)

![451cd44c610464cdda6933353bbb2ab1703724cd89882c0552a1e8fbbe114cca.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/451cd44c610464cdda6933353bbb2ab1703724cd89882c0552a1e8fbbe114cca.jpg)

![45c50c4e1ba825d453fef53c3cbd9679d2402ff38fac3e627a19c8ec25e74b30.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/45c50c4e1ba825d453fef53c3cbd9679d2402ff38fac3e627a19c8ec25e74b30.jpg)

![468a3afc974879a5b5eabf8a9bf7265d63141ab6a26d926933ec2a1997d7d3b6.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/468a3afc974879a5b5eabf8a9bf7265d63141ab6a26d926933ec2a1997d7d3b6.jpg)

![4a9964868fa35ab94aa6a61de94083eab564a2997ec7236ffd0918102ab8ee5b.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/4a9964868fa35ab94aa6a61de94083eab564a2997ec7236ffd0918102ab8ee5b.jpg)

![4efaee73a6c8deeb2bec938216f2281dfd2a8723e73525413cdb37d4c1ea3a9e.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/4efaee73a6c8deeb2bec938216f2281dfd2a8723e73525413cdb37d4c1ea3a9e.jpg)

![4efc06886b4f4dcd24f614ad53b971b7a298bea009d1ed8cfbbe3276559c71f9.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/4efc06886b4f4dcd24f614ad53b971b7a298bea009d1ed8cfbbe3276559c71f9.jpg)

![510e88ec9696f0da4a4cbe85d9f705a76ff2254903374724aa9db7f38c726ab0.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/510e88ec9696f0da4a4cbe85d9f705a76ff2254903374724aa9db7f38c726ab0.jpg)

![56d47d2f0c71e0daf38b5bca56d01690a460d23c656ca91e99274c0e016e6b3a.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/56d47d2f0c71e0daf38b5bca56d01690a460d23c656ca91e99274c0e016e6b3a.jpg)

![5cbe8bbbd656e55a2d9737f891071af6579f184d8181c81fec4a56888e3996c2.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/5cbe8bbbd656e55a2d9737f891071af6579f184d8181c81fec4a56888e3996c2.jpg)

![5db619257137a1b8c180e0bc756a5a41a6136e377ee732eb47427d1928cccb82.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/5db619257137a1b8c180e0bc756a5a41a6136e377ee732eb47427d1928cccb82.jpg)

![7d93b606034333261af6f7ca3dd66833bf8c0b9e9dd7391a24b3120596e739e6.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/7d93b606034333261af6f7ca3dd66833bf8c0b9e9dd7391a24b3120596e739e6.jpg)

![8042341a294b25294728a339004a95e4ac285d8fc02f1e7252f619063b6e2e6f.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/8042341a294b25294728a339004a95e4ac285d8fc02f1e7252f619063b6e2e6f.jpg)

![9a58755d9edf9312a73db3f388f3c374b48a6d08f4f1115cca3ec08580a5d5e4.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/9a58755d9edf9312a73db3f388f3c374b48a6d08f4f1115cca3ec08580a5d5e4.jpg)

![9cad1eecaef79c1e63ed93629d6ddc0281bb08b74c0bd058811fcaaa52ec7e9f.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/9cad1eecaef79c1e63ed93629d6ddc0281bb08b74c0bd058811fcaaa52ec7e9f.jpg)

![a34c5a0da056d92b86b1fba5c7a4f04776987e4d5d9aa942c6e6a51e700789b1.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/a34c5a0da056d92b86b1fba5c7a4f04776987e4d5d9aa942c6e6a51e700789b1.jpg)

![a7bcea7e59a778a4d2a6d638947ab5ef58c584414a9c2057f61d5a3443e9a7b7.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/a7bcea7e59a778a4d2a6d638947ab5ef58c584414a9c2057f61d5a3443e9a7b7.jpg)

![aa224ec712c7f9f6c6261a6be56b8901a38afa7d9f3fae9988c4e0d1b3e0bcf0.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/aa224ec712c7f9f6c6261a6be56b8901a38afa7d9f3fae9988c4e0d1b3e0bcf0.jpg)

![ae0a1d3c21a5cb0d31abff5618eaa76a5bf69f78e3f1bf489d24742ecd9681c3.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/ae0a1d3c21a5cb0d31abff5618eaa76a5bf69f78e3f1bf489d24742ecd9681c3.jpg)

![be6f8e09ad4507281cbfa485b3edc113842ba28d456125bc48753e36ae7e26a9.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/be6f8e09ad4507281cbfa485b3edc113842ba28d456125bc48753e36ae7e26a9.jpg)

![c088d5411273b87096325e2ca082eb9bfb7a0216bcbac6ce2bf7c134d4ebaf44.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/c088d5411273b87096325e2ca082eb9bfb7a0216bcbac6ce2bf7c134d4ebaf44.jpg)

![cea3830dd6ca6ce4252c4a270cbfe45d066365cbd3389f18e8ab3469c0e01773.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/cea3830dd6ca6ce4252c4a270cbfe45d066365cbd3389f18e8ab3469c0e01773.jpg)

![d0cae1ff1dc74d453319df5fb3649934222515900db8e98a6a4a71a5c66dba42.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/d0cae1ff1dc74d453319df5fb3649934222515900db8e98a6a4a71a5c66dba42.jpg)

![d7aba7bdc384f921518e0b3a91f8f97f2e1fdff6a6e290d8bc4d59aa8df13abc.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/d7aba7bdc384f921518e0b3a91f8f97f2e1fdff6a6e290d8bc4d59aa8df13abc.jpg)

![ec49ce0ec59fc60b7c7e432e30493ec723ad8e6ea1401817d927a29605b74323.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/ec49ce0ec59fc60b7c7e432e30493ec723ad8e6ea1401817d927a29605b74323.jpg)

![fcfd536a8a75e7011c0bdb7a6075387b6b999c443d746a6ee09696f00a5a340d.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/images/fcfd536a8a75e7011c0bdb7a6075387b6b999c443d746a6ee09696f00a5a340d.jpg)

### Tables

![05095843ceb32922ff208f21954ce725fc964e27407d38f03181c6623a4cec00.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/05095843ceb32922ff208f21954ce725fc964e27407d38f03181c6623a4cec00.jpg)

![09ba0ef5034c924d34f19c260d17caf77e2f101aa94f0154bc5818af90135696.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/09ba0ef5034c924d34f19c260d17caf77e2f101aa94f0154bc5818af90135696.jpg)

![1596d8e11229b7536d24893ee645e9d40993ff0aa66ce149159f866f285486e7.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/1596d8e11229b7536d24893ee645e9d40993ff0aa66ce149159f866f285486e7.jpg)

![4330cd30bec828796314c901080884d5cbc1b2bbba7a21996f984fefef11254c.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/4330cd30bec828796314c901080884d5cbc1b2bbba7a21996f984fefef11254c.jpg)

![626a643c3053a3bd3f61a601b3674293298edc18e36c1e951dacc5b289af3567.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/626a643c3053a3bd3f61a601b3674293298edc18e36c1e951dacc5b289af3567.jpg)

![6675d8e1def57b1dd6b0eb080f58ca03f07ca1cb5231d8d29a08dfef93bc769f.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/6675d8e1def57b1dd6b0eb080f58ca03f07ca1cb5231d8d29a08dfef93bc769f.jpg)

![73ff73e128cea2b58b036d9e2edda8ac67b0e6cd527f8b41606bb7b66403da08.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/73ff73e128cea2b58b036d9e2edda8ac67b0e6cd527f8b41606bb7b66403da08.jpg)

![7bdb7893f8819f45f2c3d2970aa45b07ad0f7fb752a5b44f4a0d40137a55c907.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/7bdb7893f8819f45f2c3d2970aa45b07ad0f7fb752a5b44f4a0d40137a55c907.jpg)

![8452e616400e2a0f0e7e22ba2215244cd05ae3920413f7a660a73b4514126843.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/8452e616400e2a0f0e7e22ba2215244cd05ae3920413f7a660a73b4514126843.jpg)

![9521020db498ece0cd7638c71f140bd30399097a7b24b7bed971ac640542dadc.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/9521020db498ece0cd7638c71f140bd30399097a7b24b7bed971ac640542dadc.jpg)

![9aad5654c2de5b9401b02ff4d7278afad6d99c05ad507f636e7b9be0a5ca3220.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/9aad5654c2de5b9401b02ff4d7278afad6d99c05ad507f636e7b9be0a5ca3220.jpg)

![9f3de10a4c1a8776c863ecd0e7dc035f6f0acebe17a0d52dc7288be1eaeb857e.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/9f3de10a4c1a8776c863ecd0e7dc035f6f0acebe17a0d52dc7288be1eaeb857e.jpg)

![a4be497c4c4b8f0389e593825e643bc5b1622d1fa95a558ee03a732594128470.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/a4be497c4c4b8f0389e593825e643bc5b1622d1fa95a558ee03a732594128470.jpg)

![a797ef69bb8564825c2a3053faac77bddf7f33d55dae0be502be4d7c9f4537f5.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/a797ef69bb8564825c2a3053faac77bddf7f33d55dae0be502be4d7c9f4537f5.jpg)

![a924e6c1aa194f3a4d00d0bfd824fbc1f18c12dc8b9b038c27289fc02c78cb8b.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/a924e6c1aa194f3a4d00d0bfd824fbc1f18c12dc8b9b038c27289fc02c78cb8b.jpg)

![c3e4fc5f94debe0255f30df58063fe611ea56b882957a00d15c97498222fd6e9.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/c3e4fc5f94debe0255f30df58063fe611ea56b882957a00d15c97498222fd6e9.jpg)

![ca225c24c3031fd9e961e34d59d94cd4a291476d436702094c6f0e4bab75ae9c.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/ca225c24c3031fd9e961e34d59d94cd4a291476d436702094c6f0e4bab75ae9c.jpg)

![e2f14d229c11fcdbf43f30729fc7e007df2413c88c9113b83fb33ab41d02ee1c.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/e2f14d229c11fcdbf43f30729fc7e007df2413c88c9113b83fb33ab41d02ee1c.jpg)

![e72f184f78b0d046aea16a0f79a68056e654f3d7a6f01282bef0624605f60acb.jpg](../icml_results/98_FLIP2_%20Expanding%20Protein%20Fitness%20Landscape%20Benchmarks%20for%20Real-World%20Machine%20Learning%20Applications/tables/e72f184f78b0d046aea16a0f79a68056e654f3d7a6f01282bef0624605f60acb.jpg)

<a id="paper-0098"></a>

## Listening Through the Noise: Cauchy-Driven Diffusion Bridges for Robust Gastrointestinal Auscultation and Clinical Benchmarking


### Images

![0809eb6ae71f961d3916717bfc7cc8b03e67a6223df3cad5af01fe94eaff03fa.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/0809eb6ae71f961d3916717bfc7cc8b03e67a6223df3cad5af01fe94eaff03fa.jpg)

![28e9b68c48bc5ff82db5d79315c1b36099d736ece766e5f1651480e3767400c6.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/28e9b68c48bc5ff82db5d79315c1b36099d736ece766e5f1651480e3767400c6.jpg)

![6f4bb17acdbff038158f1934803ff8139d4da6b41b47456ef4176afd926f903d.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/6f4bb17acdbff038158f1934803ff8139d4da6b41b47456ef4176afd926f903d.jpg)

![74de83837239600182a7fcdbdc5f92cdfe80e501965c196e3feb111b3ba22387.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/74de83837239600182a7fcdbdc5f92cdfe80e501965c196e3feb111b3ba22387.jpg)

![b9c93858b2a8cad1362d4667fa7f36f9b5e5b353e7ec0c3111b94bd2e1473642.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/b9c93858b2a8cad1362d4667fa7f36f9b5e5b353e7ec0c3111b94bd2e1473642.jpg)

![e34c16354e3f55a3ea1d0f4a26515a5a9a035c2ea962152c3b4bd7bd8fba76e8.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/images/e34c16354e3f55a3ea1d0f4a26515a5a9a035c2ea962152c3b4bd7bd8fba76e8.jpg)

### Tables

![166079794169873a44e5e36b450a479e7cc3478167b6320f044b6d02061bb0e4.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/166079794169873a44e5e36b450a479e7cc3478167b6320f044b6d02061bb0e4.jpg)

![2d0bf40012403ee7664befc091cc2136bdbf6cae5102598e5cc3be16f38661f9.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/2d0bf40012403ee7664befc091cc2136bdbf6cae5102598e5cc3be16f38661f9.jpg)

![2ea8f0179cf87d2de32ca5045b37338c9eab9071b416f728cccfc42683bc7ae3.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/2ea8f0179cf87d2de32ca5045b37338c9eab9071b416f728cccfc42683bc7ae3.jpg)

![33fb037350fd6cd3b8248c6ae9451796d109feb8c8aca2ede10ef10d5b52640d.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/33fb037350fd6cd3b8248c6ae9451796d109feb8c8aca2ede10ef10d5b52640d.jpg)

![47449c226bcc70576b3b5d707c59bf322c96c8985e883de837bae3f5d45ede24.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/47449c226bcc70576b3b5d707c59bf322c96c8985e883de837bae3f5d45ede24.jpg)

![50986c21924211a2ff9e1e46085ba84ca1e2c46b67f7f83237c40439ecdf166d.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/50986c21924211a2ff9e1e46085ba84ca1e2c46b67f7f83237c40439ecdf166d.jpg)

![76f14d0aaf697dcacd04d47543d65b670655cd1daa6ffb5099946fa34f7f53ba.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/76f14d0aaf697dcacd04d47543d65b670655cd1daa6ffb5099946fa34f7f53ba.jpg)

![8fb28dc9e6d54eea9cb6dc3d2ddcf3d41bfa39aa1256053ec645e135bef53b7a.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/8fb28dc9e6d54eea9cb6dc3d2ddcf3d41bfa39aa1256053ec645e135bef53b7a.jpg)

![acfd7b6a7fdd1a0ad604b101dccc44d54b67ab0da2d14bb3f6c73693ec837874.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/acfd7b6a7fdd1a0ad604b101dccc44d54b67ab0da2d14bb3f6c73693ec837874.jpg)

![b9cf6ed7bd144ec123411d69bb0641ea4dbcb3c00f2bf090b9227824705235dd.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/b9cf6ed7bd144ec123411d69bb0641ea4dbcb3c00f2bf090b9227824705235dd.jpg)

![d7fab6b5020f688d01127bac3045199bf0496f7237850e635b41cbe6afe32f3a.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/d7fab6b5020f688d01127bac3045199bf0496f7237850e635b41cbe6afe32f3a.jpg)

![f90378251a1d0a2a6b3604fdf9708bc47ead7e6930a20319f0c733d9e5c70bf2.jpg](../icml_results/99_Listening%20Through%20the%20Noise_%20Cauchy-Driven%20Diffusion%20Bridges%20for%20Robust%20Gastrointestinal%20Auscultatio/tables/f90378251a1d0a2a6b3604fdf9708bc47ead7e6930a20319f0c733d9e5c70bf2.jpg)

<a id="paper-0099"></a>

## Disentangling Latent Risk Pathways via Bayesian Hypergraph Inference

### Images

![0e426bcdfe8f504a9cf714626a9b64a35c9ce5d275b69321911a75ad28683cfc.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/0e426bcdfe8f504a9cf714626a9b64a35c9ce5d275b69321911a75ad28683cfc.jpg)

![1f61c121973b96913052c93664dda562f857884882fc26fc6a1d49380d9ed46e.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/1f61c121973b96913052c93664dda562f857884882fc26fc6a1d49380d9ed46e.jpg)

![2180e2b4cd682faa10ff998faddaf10ba80107905c86bf0b1618677ad1c87dcb.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/2180e2b4cd682faa10ff998faddaf10ba80107905c86bf0b1618677ad1c87dcb.jpg)

![5d884d5e38a3d7206af3f0d575ef617126d761ed4b12a33b523caf1be1ba721f.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/5d884d5e38a3d7206af3f0d575ef617126d761ed4b12a33b523caf1be1ba721f.jpg)

![7cf9d5bb37a3fd621431bd1485c2f645fdd1bda054a7b640735a9e4a1f80db55.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/7cf9d5bb37a3fd621431bd1485c2f645fdd1bda054a7b640735a9e4a1f80db55.jpg)

![a8e0a7263c10a340a09fa658319b60657521b65488d70040f98a551cf258102e.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/a8e0a7263c10a340a09fa658319b60657521b65488d70040f98a551cf258102e.jpg)

![c6fc26bd066b42f2b10c08e04c1c4f8904db4a3c103857265604dac5718eae40.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/c6fc26bd066b42f2b10c08e04c1c4f8904db4a3c103857265604dac5718eae40.jpg)

![e0ff7c1553de3d47a890217d819d1473b0c389eeac3629dc38685719b16a8a86.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/e0ff7c1553de3d47a890217d819d1473b0c389eeac3629dc38685719b16a8a86.jpg)

![e20b64499a3a1be86c1de45eb81bdf6ee58c1f1b6a4003b0e41c758998419215.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/e20b64499a3a1be86c1de45eb81bdf6ee58c1f1b6a4003b0e41c758998419215.jpg)

![ed6d1158ee0ec3abbbec193e484268a82a5f6577ef2211dc3b86b612a7652c6a.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/ed6d1158ee0ec3abbbec193e484268a82a5f6577ef2211dc3b86b612a7652c6a.jpg)

![ed94bbe9a4c818abdd51ac5dd3bfef3c966943f9ed4f218daaf50b210124e6d6.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/images/ed94bbe9a4c818abdd51ac5dd3bfef3c966943f9ed4f218daaf50b210124e6d6.jpg)

### Tables

![103597431153595eb96bb27ac15103d7195e3cfc53014fb445b12e5691c29a04.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/103597431153595eb96bb27ac15103d7195e3cfc53014fb445b12e5691c29a04.jpg)

![22961eea71e88366f821996af15508376a584202fa764002811357eb62c1c847.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/22961eea71e88366f821996af15508376a584202fa764002811357eb62c1c847.jpg)

![2ddec07ed883f1a6cc83b59f983d2968707a5c7ab592e46a83f35104bb6107aa.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/2ddec07ed883f1a6cc83b59f983d2968707a5c7ab592e46a83f35104bb6107aa.jpg)

![38b378aa09822d159a454f1bd2afa791972f9a7621f4bf7c5e1d557915900810.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/38b378aa09822d159a454f1bd2afa791972f9a7621f4bf7c5e1d557915900810.jpg)

![3d903430d95f6d392860f70c4840ceb3f3823f9186d3de92df5b32557c0262a2.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/3d903430d95f6d392860f70c4840ceb3f3823f9186d3de92df5b32557c0262a2.jpg)

![47e90165c6bea55863929b6f25e465f2712ceab6ebab9ca1ca1474fb23841ce7.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/47e90165c6bea55863929b6f25e465f2712ceab6ebab9ca1ca1474fb23841ce7.jpg)

![493ce74ae798b54ce74a7b902e7c4650ca386c9b42e03147dc79a68c4949265f.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/493ce74ae798b54ce74a7b902e7c4650ca386c9b42e03147dc79a68c4949265f.jpg)

![7c3ad1694e95409a92729940a29d72c46fd887103c5d66d014de1193c7b87520.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/7c3ad1694e95409a92729940a29d72c46fd887103c5d66d014de1193c7b87520.jpg)

![9705c8612e4de548207e96796a415d6a924ddacecf4a5baa409d575388cce3ec.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/9705c8612e4de548207e96796a415d6a924ddacecf4a5baa409d575388cce3ec.jpg)

![9b626b9f31eca30e719e71056cac3e5add1cdc2ef8f5dbb98e87a71d0201102d.jpg](../icml_results/100_Disentangling%20Latent%20Risk%20Pathways%20via%20Bayesian%20Hypergraph%20Inference/tables/9b626b9f31eca30e719e71056cac3e5add1cdc2ef8f5dbb98e87a71d0201102d.jpg)
