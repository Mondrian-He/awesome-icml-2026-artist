# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 95 of 197

## 目录 (Table of Contents)

1. [The Geometry of Narrow Fine-Tuning Degradation: Trajectory Lock-in and Spectral Bifurcation](#paper-3103)
2. [TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation](#paper-3104)
3. [BioProBench: A Corpus and Benchmark for Biological Protocol Reasoning in Autonomous Science](#paper-3105)
4. [Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens](#paper-3106)
5. [WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting](#paper-3107)
6. [Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space](#paper-3108)
7. [ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity](#paper-3109)
8. [Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding](#paper-3110)
9. [Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation](#paper-3111)
10. [From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](#paper-3112)
11. [Post-Training Language Models for Crosslingual Consistency](#paper-3113)
12. [On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](#paper-3114)
13. [From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide ML Interatomic Potential Architectures](#paper-3115)
14. [PETS: A Principled Framework Towards Optimal Trajectory Allocation for Efficient Test-Time Self-Consistency](#paper-3116)
15. [TimeLAVA: Learning-Agnostic Valuation for Time Series Data](#paper-3117)
16. [ProtDBench: A Unified Benchmark of Protein Binder Design and Evaluation](#paper-3118)
17. [Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane](#paper-3119)
18. [Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](#paper-3120)
19. [FairMerging: Rethinking Model Merging through the Lens of Fairness](#paper-3121)
20. [PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting](#paper-3122)
21. [LIF Recurrent Memory Enables Long-Horizon Spiking Computation](#paper-3123)
22. [ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning](#paper-3124)
23. [Hyperbolic Multimodal Continual Learning](#paper-3125)
24. [XPERT: Expert Knowledge Transfer for Effective Training of Language Models](#paper-3126)
25. [SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning](#paper-3127)
26. [RECTOR: Masked Region-Channel-Temporal Modeling for Affective and Cognitive Representation Learning](#paper-3128)
27. [AlignVid: Taming Visual Dominance via Training-Free Attention Modulation in Text-guided Image-to-Video Generation](#paper-3129)
28. [Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation](#paper-3130)
29. [Agora: Toward Autonomous Bug Detection in Production-Level Consensus Protocols with LLM Agents](#paper-3131)
30. [SimpleMem: Efficient Lifelong Memory for LLM Agents](#paper-3132)
31. [Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](#paper-3133)
32. [Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](#paper-3134)
33. [JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments](#paper-3135)

---


<a id="paper-3103"></a>

## The Geometry of Narrow Fine-Tuning Degradation: Trajectory Lock-in and Spectral Bifurcation

### Images

![0136d7c610c8c5763b877bf50bfd4dedd689ebbec7461516fc3db50be054ce02.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/0136d7c610c8c5763b877bf50bfd4dedd689ebbec7461516fc3db50be054ce02.jpg)

![098dcb2e2be6f6fa3b0346cfeaf7978ba28bf3d4155c7f67293f5452603e970e.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/098dcb2e2be6f6fa3b0346cfeaf7978ba28bf3d4155c7f67293f5452603e970e.jpg)

![0d2894c16232533b4a731e47ca18ffe284608e9330d4f9f293b5e03a9c851038.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/0d2894c16232533b4a731e47ca18ffe284608e9330d4f9f293b5e03a9c851038.jpg)

![24da2928f1be82de40e4e948fb0fabe815707f24b01b2cd4afdde1fe8c262519.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/24da2928f1be82de40e4e948fb0fabe815707f24b01b2cd4afdde1fe8c262519.jpg)

![62266a66c94b11d19f183cca2de5ba236baea6eb997867659065895f2d78b303.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/62266a66c94b11d19f183cca2de5ba236baea6eb997867659065895f2d78b303.jpg)

![662833443ecb20ee1db66ec7df896df4f89f635c7aefd2f4ab2de3bd17af6bbd.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/662833443ecb20ee1db66ec7df896df4f89f635c7aefd2f4ab2de3bd17af6bbd.jpg)

![8b62391541cb1497df9f6a6dd3849e9079d7a4c2e24f73e99d4d21e8c4893164.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/8b62391541cb1497df9f6a6dd3849e9079d7a4c2e24f73e99d4d21e8c4893164.jpg)

![9e8b00edbeee4037a8ede4e5d103bc1b87fafff23c84fdd574039349095cd8b7.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/9e8b00edbeee4037a8ede4e5d103bc1b87fafff23c84fdd574039349095cd8b7.jpg)

![a10d6f6fa1bcacc43e797c5c3fd64c4f982fccfcc5cad6e112cb2840fc7da35a.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/a10d6f6fa1bcacc43e797c5c3fd64c4f982fccfcc5cad6e112cb2840fc7da35a.jpg)

![a87683c1287b47867b0e2fa7838aa060895dc730474734b64d60a19c66614330.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/a87683c1287b47867b0e2fa7838aa060895dc730474734b64d60a19c66614330.jpg)

![ab15419c712cd68ef09072003c4dc04d85027a5e78f3d063418da0f249f967e7.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/ab15419c712cd68ef09072003c4dc04d85027a5e78f3d063418da0f249f967e7.jpg)

![f5a4aa4ac8ed684d276980f54e0f09d156735753c8f213ca5d3d6a19917ed565.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/images/f5a4aa4ac8ed684d276980f54e0f09d156735753c8f213ca5d3d6a19917ed565.jpg)

### Tables

![0d9ef0641dbfeff665959f1852c8ca925b7e9f9a3cb1d6d4d6e865adb628ae87.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/0d9ef0641dbfeff665959f1852c8ca925b7e9f9a3cb1d6d4d6e865adb628ae87.jpg)

![19ce8d139f6bbd6ca1440e39c68577f441366687d9f7d130203a6e212d8fc804.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/19ce8d139f6bbd6ca1440e39c68577f441366687d9f7d130203a6e212d8fc804.jpg)

![578c1f0fd06ea5d922cbdad5b2acf18017ec5d65597f2790e111d28caa023c72.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/578c1f0fd06ea5d922cbdad5b2acf18017ec5d65597f2790e111d28caa023c72.jpg)

![7c1d0abbad357d1f8cd6804448edecfb64d235667beecc49b6759d906366952e.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/7c1d0abbad357d1f8cd6804448edecfb64d235667beecc49b6759d906366952e.jpg)

![b0888192709d1e944bf705cbaa1e0c7aa6c7c3f5ed231cc9cc85127c46b78384.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/b0888192709d1e944bf705cbaa1e0c7aa6c7c3f5ed231cc9cc85127c46b78384.jpg)

![b64490d737c9c5f3eb11d8925ff5ee1beb0c1c6d907380723e2d5881726382da.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/b64490d737c9c5f3eb11d8925ff5ee1beb0c1c6d907380723e2d5881726382da.jpg)

![d02c0eac02570d0d4d7cb7341176594477e2dfb312545154856467f011cf79e1.jpg](../icml_results/3141_Next-Gen%20CAPTCHAs_%20Leveraging%20the%20Cognitive%20Gap%20for%20Scalable%20and%20Diverse%20GUI-Agent%20Defense/tables/d02c0eac02570d0d4d7cb7341176594477e2dfb312545154856467f011cf79e1.jpg)

## The Geometry of Narrow Fine-Tuning Degradation: Trajectory Lock-in and Spectral Bifurcation


### Images

![0e9e0c73eab9a6bbb54acf78c7d9f5f445bd282f15ea29fc920ec6695040c1f7.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/0e9e0c73eab9a6bbb54acf78c7d9f5f445bd282f15ea29fc920ec6695040c1f7.jpg)

![83a18486362fa178b2ff12d13b44aba581e17ad5146e18506eb63b362a001a3c.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/83a18486362fa178b2ff12d13b44aba581e17ad5146e18506eb63b362a001a3c.jpg)

![8887cfb78d18f8cc1240d69bfb60bba37945d76a089e99a135ff8af2a203099e.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/8887cfb78d18f8cc1240d69bfb60bba37945d76a089e99a135ff8af2a203099e.jpg)

![95aca93a448f439769a51cbde339ac94f50b08fa8e1378d2cd86b842b050912b.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/95aca93a448f439769a51cbde339ac94f50b08fa8e1378d2cd86b842b050912b.jpg)

![a44c48ca9788ee3dcfd52a9180c1c26533e30c19da822e77eddbd8690affc494.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/a44c48ca9788ee3dcfd52a9180c1c26533e30c19da822e77eddbd8690affc494.jpg)

![aad9372036e32c788f804e5ea7629e79ef800f5ab131782dda8d316824ead327.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/aad9372036e32c788f804e5ea7629e79ef800f5ab131782dda8d316824ead327.jpg)

![f3882bea11b3fd9b3bd46a591167569889247b3ef46e9518f0db68b83473a7fc.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/f3882bea11b3fd9b3bd46a591167569889247b3ef46e9518f0db68b83473a7fc.jpg)

![f9026fef85a806916f4ec64a237809fc7c67f7738636ec3d755cf6f28ef0706d.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/images/f9026fef85a806916f4ec64a237809fc7c67f7738636ec3d755cf6f28ef0706d.jpg)

### Tables

![01969e41380a78f7d9863729ada18792311462d8917d5a6a54ac3aecc00378af.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/01969e41380a78f7d9863729ada18792311462d8917d5a6a54ac3aecc00378af.jpg)

![1738c80c9fd77ab3712c4950f8baeda00d0859939076ab4550effd7ec851349a.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/1738c80c9fd77ab3712c4950f8baeda00d0859939076ab4550effd7ec851349a.jpg)

![225212143fbe4ce572a68ff75d95ee5bc5d49c8d5713ca484455afa596050e80.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/225212143fbe4ce572a68ff75d95ee5bc5d49c8d5713ca484455afa596050e80.jpg)

![33233bdf74430f14e971a48817f93b10f0ce709577fb1c831ea22e3fafd2761c.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/33233bdf74430f14e971a48817f93b10f0ce709577fb1c831ea22e3fafd2761c.jpg)

![3bdac13af6838c387e96c1b804274b36a316368c88a08718fc4c3ddd2507d8a8.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/3bdac13af6838c387e96c1b804274b36a316368c88a08718fc4c3ddd2507d8a8.jpg)

![5cbce7f2672b1ee5dcd7396e35450c079ba8d90aa5b12a5738059336b2babff6.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/5cbce7f2672b1ee5dcd7396e35450c079ba8d90aa5b12a5738059336b2babff6.jpg)

![5e73ab849fd04771fb41f2779c147f08a66df2095c9528d92c1d5fbadd26922d.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/5e73ab849fd04771fb41f2779c147f08a66df2095c9528d92c1d5fbadd26922d.jpg)

![6eba099acac9fb41f692a6357bf2f237c8bc92253f197fce64d34ccc1eee67d6.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/6eba099acac9fb41f692a6357bf2f237c8bc92253f197fce64d34ccc1eee67d6.jpg)

![9062b8a1835c64374faeb8ed59a87c60f9426fb403da41e4cd0717c9d9689dbd.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/9062b8a1835c64374faeb8ed59a87c60f9426fb403da41e4cd0717c9d9689dbd.jpg)

![95fdae1eb7796d44f56b041afbecd96acf04ad124ca6bcf89217629fafd0c396.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/95fdae1eb7796d44f56b041afbecd96acf04ad124ca6bcf89217629fafd0c396.jpg)

![c678b7b98f7ece62835d03090bd259795f897bce6a9ee9f8a2a8248441a0eaec.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/c678b7b98f7ece62835d03090bd259795f897bce6a9ee9f8a2a8248441a0eaec.jpg)

![c7d3937038d5f45036318a9f3e28c7f8b99a3221b2ca3e6bc954dac1423244ce.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/c7d3937038d5f45036318a9f3e28c7f8b99a3221b2ca3e6bc954dac1423244ce.jpg)

![e63168a622430c7c1c5e490e42194cd23a595eefae855410b9aad082edda385d.jpg](../icml_results/3142_The%20Geometry%20of%20Narrow%20Fine-Tuning%20Degradation_%20Trajectory%20Lock-in%20and%20Spectral%20Bifurcation/tables/e63168a622430c7c1c5e490e42194cd23a595eefae855410b9aad082edda385d.jpg)

<a id="paper-3104"></a>

## TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation


### Images

![119e2a68e5b6a63267ed69f8f0e02b68d2fe7b54ce65e0ae490dc82ad33ef48b.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/119e2a68e5b6a63267ed69f8f0e02b68d2fe7b54ce65e0ae490dc82ad33ef48b.jpg)

![398507036e905f02c96dbbfb38e2c3aed8101e5847efa695b7e76c44b6d6cd7e.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/398507036e905f02c96dbbfb38e2c3aed8101e5847efa695b7e76c44b6d6cd7e.jpg)

![62a6c7a4b57b4e701d4fc803ca0cc3161ca19e4aad5b8f49a91b980a05cff7ba.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/62a6c7a4b57b4e701d4fc803ca0cc3161ca19e4aad5b8f49a91b980a05cff7ba.jpg)

![946962c19a3b7ab634aa402f650b204c98459c8d2d488d5caf94a66df9893928.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/946962c19a3b7ab634aa402f650b204c98459c8d2d488d5caf94a66df9893928.jpg)

![98501e584f7f571752595fce48519b6f535f0bb63598b247a3c6e1088ec6a39a.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/98501e584f7f571752595fce48519b6f535f0bb63598b247a3c6e1088ec6a39a.jpg)

![a7ea937076976913f975a4eb62a4b76f63a845cc072fa52312f19d2df6e1d760.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/a7ea937076976913f975a4eb62a4b76f63a845cc072fa52312f19d2df6e1d760.jpg)

![b733c7b2ec2a46b4cb67183b72ec9cd59a51c122a97e7d35fc9000b836b284a1.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/b733c7b2ec2a46b4cb67183b72ec9cd59a51c122a97e7d35fc9000b836b284a1.jpg)

![bc632a8d83823f068806fe2f939a657725258af9925b9535972399f95a0a0d25.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/bc632a8d83823f068806fe2f939a657725258af9925b9535972399f95a0a0d25.jpg)

![c2ce6a7e66e0c059ff67bdaf13bf98af0794b789d2616db35ad2c6c3635b5621.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/c2ce6a7e66e0c059ff67bdaf13bf98af0794b789d2616db35ad2c6c3635b5621.jpg)

![cf4ddfb55f933d53fe50dc524c4a8b0946d20910ee7bf830603c5fa906a6ee8c.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/cf4ddfb55f933d53fe50dc524c4a8b0946d20910ee7bf830603c5fa906a6ee8c.jpg)

![d31e779bb2ce26f6a973e2399c748bdc16b816243102c8311ae11d5ec7d9c5d7.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/d31e779bb2ce26f6a973e2399c748bdc16b816243102c8311ae11d5ec7d9c5d7.jpg)

![dc36d9606defb604a762bdd4282c563e82a50346434ef83ba5799cfa0baf1ea7.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/dc36d9606defb604a762bdd4282c563e82a50346434ef83ba5799cfa0baf1ea7.jpg)

![e7ff4ff74594bc8be372e7d1da36f9dc78a9d19749263a4cb01ecefa0bb82ad8.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/e7ff4ff74594bc8be372e7d1da36f9dc78a9d19749263a4cb01ecefa0bb82ad8.jpg)

![fab12a886278f1e458ac7651f8954e6b9b7f6981e7eb3aa6b18c3b5e1da5ed7d.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/fab12a886278f1e458ac7651f8954e6b9b7f6981e7eb3aa6b18c3b5e1da5ed7d.jpg)

![fd49b8ecf71993845c70b70f74ed275ee137e92561446aa53fe90fe37781f366.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/images/fd49b8ecf71993845c70b70f74ed275ee137e92561446aa53fe90fe37781f366.jpg)

### Tables

![21543339f2b5511f946189707cd03d2267c9570fe6034526aea9d4980512a206.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/21543339f2b5511f946189707cd03d2267c9570fe6034526aea9d4980512a206.jpg)

![2567d8fb0646a21bdcbe4d5b4ea9c0cdd6d1b4cb143d5f3ef373bf946b05244c.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/2567d8fb0646a21bdcbe4d5b4ea9c0cdd6d1b4cb143d5f3ef373bf946b05244c.jpg)

![3470f188beddffe6f8cae5ad9c9d71e3b6061d0776072852621cca2854c84a56.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/3470f188beddffe6f8cae5ad9c9d71e3b6061d0776072852621cca2854c84a56.jpg)

![82ce83bb202e282fc63f14b9f4132f8710afed87ecedacfa2b34b3c20fb393d7.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/82ce83bb202e282fc63f14b9f4132f8710afed87ecedacfa2b34b3c20fb393d7.jpg)

![94344a2b1a232aa789036ec13f60186a1236578f046980227de582bf0ad50b75.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/94344a2b1a232aa789036ec13f60186a1236578f046980227de582bf0ad50b75.jpg)

![c4b30bd26b4118b97bf086e4074c6e6a04d7c06adaf89737555612acd0bcf6c3.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/c4b30bd26b4118b97bf086e4074c6e6a04d7c06adaf89737555612acd0bcf6c3.jpg)

![d7a8d765133cbb4267a88baffc8e27be4e62c5281f2d41e8cb0660f56a312775.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/d7a8d765133cbb4267a88baffc8e27be4e62c5281f2d41e8cb0660f56a312775.jpg)

![dff744c381997e260076fc7e37214fa2c32360bdc693e5cfb02f20548a3f78b9.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/dff744c381997e260076fc7e37214fa2c32360bdc693e5cfb02f20548a3f78b9.jpg)

![e9853d29d0e5e26af9760fffbc3ef96d44b27481c494ae18b4d29419de9f88e2.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/e9853d29d0e5e26af9760fffbc3ef96d44b27481c494ae18b4d29419de9f88e2.jpg)

![ee755420fd123dd293c4b8007099ec21537f86d9f3d5ce397db82b47d77ab19c.jpg](../icml_results/3143_TestExplora_%20Benchmarking%20LLMs%20for%20Proactive%20Bug%20Discovery%20via%20Repository-Level%20Test%20Generation/tables/ee755420fd123dd293c4b8007099ec21537f86d9f3d5ce397db82b47d77ab19c.jpg)

<a id="paper-3105"></a>

## BioProBench: A Corpus and Benchmark for Biological Protocol Reasoning in Autonomous Science


### Images

![0cc528dc45ece884cce2e55563b431214b176c91245688e2c0f7f0a3a723e49b.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/0cc528dc45ece884cce2e55563b431214b176c91245688e2c0f7f0a3a723e49b.jpg)

![16cc756183c009f499df0b8835de0d519bb8fa8e4186d3b424338e986e3bf293.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/16cc756183c009f499df0b8835de0d519bb8fa8e4186d3b424338e986e3bf293.jpg)

![40b6ee5ae67c45c628b7782c2266481c0a5fa26dbb22b00d8786d8ba35ad6941.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/40b6ee5ae67c45c628b7782c2266481c0a5fa26dbb22b00d8786d8ba35ad6941.jpg)

![4b129fa8f1878a19428f673e6d9c5f79ba5a3c18298f7719917b06f53dbc4b78.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/4b129fa8f1878a19428f673e6d9c5f79ba5a3c18298f7719917b06f53dbc4b78.jpg)

![8ac62632fcdc50365329c7a1075a62f9e1745bc3e799734eb4c0754ab9e1dbb5.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/8ac62632fcdc50365329c7a1075a62f9e1745bc3e799734eb4c0754ab9e1dbb5.jpg)

![8d51b9c3988493b6af49580cb4a7d0724791355caacb47c1fca508289d605ecd.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/8d51b9c3988493b6af49580cb4a7d0724791355caacb47c1fca508289d605ecd.jpg)

![c7a6b5977cab15145c0dbd67f67f3468a39392968c44cacf9d63ce9f1eabac9a.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/c7a6b5977cab15145c0dbd67f67f3468a39392968c44cacf9d63ce9f1eabac9a.jpg)

![e61cee8e00a6e4ac3fbe5222413033a9469b33a430849781d9db387817802a03.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/e61cee8e00a6e4ac3fbe5222413033a9469b33a430849781d9db387817802a03.jpg)

![eccaabbe1bc720fab361efba041b9d4b6d8c1f33212a799c490c021ff2d6ffbf.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/images/eccaabbe1bc720fab361efba041b9d4b6d8c1f33212a799c490c021ff2d6ffbf.jpg)

### Tables

![05a77348bfa8fcbcc5df5acc0225ec1fce8d5bc7e5d32e389845ae14080cde5c.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/05a77348bfa8fcbcc5df5acc0225ec1fce8d5bc7e5d32e389845ae14080cde5c.jpg)

![06356bb1190bba0e572a9cd0d3cf95c45682c164265f4fcaca53fdb0f504e569.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/06356bb1190bba0e572a9cd0d3cf95c45682c164265f4fcaca53fdb0f504e569.jpg)

![1277952a8a7a111b020dbee7072ee351584cf471d2c561afac50c8d2b95395dc.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/1277952a8a7a111b020dbee7072ee351584cf471d2c561afac50c8d2b95395dc.jpg)

![1acff7e38d43aa5300eda50c07307d52beaa0d6da4d22930ada43a175cad1908.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/1acff7e38d43aa5300eda50c07307d52beaa0d6da4d22930ada43a175cad1908.jpg)

![27b6d2acb15ea21c6646f70e71bf9da78aa52587671c8b701e8d09184efd1e54.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/27b6d2acb15ea21c6646f70e71bf9da78aa52587671c8b701e8d09184efd1e54.jpg)

![2fefeb252ada649b151dd3979bf6492af30db10d92ddc07cb4bfc0a25e8890e6.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/2fefeb252ada649b151dd3979bf6492af30db10d92ddc07cb4bfc0a25e8890e6.jpg)

![30ed5746bbdb7bdc3b06acce5359244576f159309b8e094bc5f8648ccdce4197.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/30ed5746bbdb7bdc3b06acce5359244576f159309b8e094bc5f8648ccdce4197.jpg)

![3274f60bc405616d30c8be423514cb59ec165671f91edfdf047b9f347f21b812.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/3274f60bc405616d30c8be423514cb59ec165671f91edfdf047b9f347f21b812.jpg)

![34592ed489b35a9edda89dcfd694ac05aeab551c772b8c8c6148b30db7100777.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/34592ed489b35a9edda89dcfd694ac05aeab551c772b8c8c6148b30db7100777.jpg)

![4cfcb150660b95b60adf377cc74f5397e6f7702edc44c80be8547a99ab6c9d6f.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/4cfcb150660b95b60adf377cc74f5397e6f7702edc44c80be8547a99ab6c9d6f.jpg)

![4d5fe677ef52d81d14b6f510dcf13967697772544c5d0b697e392aca194f6e9e.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/4d5fe677ef52d81d14b6f510dcf13967697772544c5d0b697e392aca194f6e9e.jpg)

![4da7761f102a6ce82dc35c365f3b4709afe145a0eadefe6747a54c41060467d3.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/4da7761f102a6ce82dc35c365f3b4709afe145a0eadefe6747a54c41060467d3.jpg)

![529ffd952e4bc9006ad2974dce541b1253c4535b085c99f9ac0e00b3f4950ba8.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/529ffd952e4bc9006ad2974dce541b1253c4535b085c99f9ac0e00b3f4950ba8.jpg)

![58b2e033e175854267b638f68f577c57a8c56f51cc88ff276ff51a0bbbd22538.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/58b2e033e175854267b638f68f577c57a8c56f51cc88ff276ff51a0bbbd22538.jpg)

![60bd695fbe61f0ecf4b78144cc0a6c8846fb9c554a35dbccdb98b3ce99c86152.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/60bd695fbe61f0ecf4b78144cc0a6c8846fb9c554a35dbccdb98b3ce99c86152.jpg)

![63089365e78a4880dea40f9df09237fb08f8c2dc88793e1f863e8fb6c02047ed.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/63089365e78a4880dea40f9df09237fb08f8c2dc88793e1f863e8fb6c02047ed.jpg)

![747016da3fc58120f34b52c5ed1f9cf50e1a54f91b49f355d9a2e635f35cb14b.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/747016da3fc58120f34b52c5ed1f9cf50e1a54f91b49f355d9a2e635f35cb14b.jpg)

![854083c1f6d1dca7f38f018a25105ba3905c1fe4e8920aa5fc9ea58de2fa5cf7.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/854083c1f6d1dca7f38f018a25105ba3905c1fe4e8920aa5fc9ea58de2fa5cf7.jpg)

![900814870720c437951db4cd44f96a26a77c69b06d6078d8fcdab6a2062f1af0.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/900814870720c437951db4cd44f96a26a77c69b06d6078d8fcdab6a2062f1af0.jpg)

![90c6b1a108a44c8b27e497cfe8c233effc74337494bb497f30d9215239247784.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/90c6b1a108a44c8b27e497cfe8c233effc74337494bb497f30d9215239247784.jpg)

![90cc5023ce10d675deb743d9e1e5b5b92f94dfcb08d939372fe283cbf1458af9.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/90cc5023ce10d675deb743d9e1e5b5b92f94dfcb08d939372fe283cbf1458af9.jpg)

![b619a6729d4e30a694f6314cb5038cbc58480c004e1ef7846a8deeb2a1419ed6.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/b619a6729d4e30a694f6314cb5038cbc58480c004e1ef7846a8deeb2a1419ed6.jpg)

![c04dc4e5ae7d02063f7192cf9d563d4c3d64a32622d1e0e77cebe6de82ff6391.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/c04dc4e5ae7d02063f7192cf9d563d4c3d64a32622d1e0e77cebe6de82ff6391.jpg)

![c4e2358fb81cd923c1ee62cbc81bef1d9b05c5ea00b94e49227ea7f8857c5166.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/c4e2358fb81cd923c1ee62cbc81bef1d9b05c5ea00b94e49227ea7f8857c5166.jpg)

![dc00b314a7da77aa35f0bf0d54e5c1901f167c1d72a530eac859b4a1101a9630.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/dc00b314a7da77aa35f0bf0d54e5c1901f167c1d72a530eac859b4a1101a9630.jpg)

![e06b3461a16f8a7d317343462c7b2f286edaa7ea2d25b63ee4ca20942328f265.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/e06b3461a16f8a7d317343462c7b2f286edaa7ea2d25b63ee4ca20942328f265.jpg)

![e3fbebd353c5e2874336ce03360756bba92e56f064d6ee74197a5f211f4940e2.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/e3fbebd353c5e2874336ce03360756bba92e56f064d6ee74197a5f211f4940e2.jpg)

![e67828af4240d383ea3545284d8cac727e0fce4c473eace600f8ee5dd7897f72.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/e67828af4240d383ea3545284d8cac727e0fce4c473eace600f8ee5dd7897f72.jpg)

![f3cb11e96286d405a6eadb965216881924bc00988384ee7a460fa27bc6a25fb9.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/f3cb11e96286d405a6eadb965216881924bc00988384ee7a460fa27bc6a25fb9.jpg)

![f60dd913bc09b6bd39a6ab8c6ee523716d67502143085147dc5f0d41412dd559.jpg](../icml_results/3144_BioProBench_%20A%20Corpus%20and%20Benchmark%20for%20Biological%20Protocol%20Reasoning%20in%20Autonomous%20Science/tables/f60dd913bc09b6bd39a6ab8c6ee523716d67502143085147dc5f0d41412dd559.jpg)

<a id="paper-3106"></a>

## Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens


### Images

![1aa2a8a154326307a6416e5cd8f1d63429a529452065f6b0562597170a300674.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/1aa2a8a154326307a6416e5cd8f1d63429a529452065f6b0562597170a300674.jpg)

![304b352b83d2de2320429de026887c8192e9ffde9062f301cf7f800a15a682d4.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/304b352b83d2de2320429de026887c8192e9ffde9062f301cf7f800a15a682d4.jpg)

![43d4cbbd86dde15e77c6070e38fd7f239b0dffbf6db89df6511e879c2c1fc6c4.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/43d4cbbd86dde15e77c6070e38fd7f239b0dffbf6db89df6511e879c2c1fc6c4.jpg)

![666773adcb2b6501722be6ddb937993e2d3505447ebc8aaf5618699e20910129.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/666773adcb2b6501722be6ddb937993e2d3505447ebc8aaf5618699e20910129.jpg)

![d59f6eef2cef34ddd07e801ea0e22e2682b9ba3626cd5e79b7e3929b405276c3.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/d59f6eef2cef34ddd07e801ea0e22e2682b9ba3626cd5e79b7e3929b405276c3.jpg)

![df16a84e3401188b3a0fd61a583f0253815cd93f35d81f2b8694645ebff18f09.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/images/df16a84e3401188b3a0fd61a583f0253815cd93f35d81f2b8694645ebff18f09.jpg)

### Tables

![137aa61af2d88de5f7e12c6d0ce2f2808ac14450dc05987c45e92d7cca51f920.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/137aa61af2d88de5f7e12c6d0ce2f2808ac14450dc05987c45e92d7cca51f920.jpg)

![236c9e14b917a4bc94a80e6d90379ec21f926f2e05b3087446a4693e46637b53.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/236c9e14b917a4bc94a80e6d90379ec21f926f2e05b3087446a4693e46637b53.jpg)

![2eb352932e891f124730c6b1198e9475ef38f91bbcad459965ea076afc2ff1ca.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/2eb352932e891f124730c6b1198e9475ef38f91bbcad459965ea076afc2ff1ca.jpg)

![4564456f3f8e55634212c5504360ddf2c20082c8a1f71e52d045953e7351753f.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/4564456f3f8e55634212c5504360ddf2c20082c8a1f71e52d045953e7351753f.jpg)

![45bf4a9103dfd13415c70f2ba537bbc8d7da332be6fd4cb50a7d8f06079534ac.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/45bf4a9103dfd13415c70f2ba537bbc8d7da332be6fd4cb50a7d8f06079534ac.jpg)

![536f38ba23a77c4e4944d9a97d2f9b2ad84bf6a0a7d26713b30d7b6038612517.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/536f38ba23a77c4e4944d9a97d2f9b2ad84bf6a0a7d26713b30d7b6038612517.jpg)

![70b47199f17d9d0c8c3aeed43c30351d2b578f08f05a4b81fd0a37bea5c26460.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/70b47199f17d9d0c8c3aeed43c30351d2b578f08f05a4b81fd0a37bea5c26460.jpg)

![977c46762ea67380185810f1181686331a885a54103d80be18e29fe1ca9f8a46.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/977c46762ea67380185810f1181686331a885a54103d80be18e29fe1ca9f8a46.jpg)

![c2c4ab29a5ac717d18f3eea9f19dcea5c8bce38a32441c2c88da725390fd1122.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/c2c4ab29a5ac717d18f3eea9f19dcea5c8bce38a32441c2c88da725390fd1122.jpg)

![d0078ede13ae830cacc09ff2fb1531db8e8f410d6b4391803ec577be2f21a615.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/d0078ede13ae830cacc09ff2fb1531db8e8f410d6b4391803ec577be2f21a615.jpg)

![dc80fe1939e699a326da387207e6c8f1e932c28ce2ef5ccb4830bc554d4ea6f5.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/dc80fe1939e699a326da387207e6c8f1e932c28ce2ef5ccb4830bc554d4ea6f5.jpg)

![f957c8e26363320639cb0314ece926678103a0f41ae8b2b53b69fb8393b29ed1.jpg](../icml_results/3145_Latent%20Thoughts%20Tuning_%20Bridging%20Context%20and%20Reasoning%20with%20Fused%20Information%20in%20Latent%20Tokens/tables/f957c8e26363320639cb0314ece926678103a0f41ae8b2b53b69fb8393b29ed1.jpg)

<a id="paper-3107"></a>

## WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting


### Images

![00dd45cac6b16fffce1ba48d81f80dcc8e1f401905eee5e7efeda68f632b0786.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/00dd45cac6b16fffce1ba48d81f80dcc8e1f401905eee5e7efeda68f632b0786.jpg)

![2c335301bedfd184fd4de76d7d22be9ce9c6c00df11b90c3685bd6908dfbbd39.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/2c335301bedfd184fd4de76d7d22be9ce9c6c00df11b90c3685bd6908dfbbd39.jpg)

![46fffae3569f6fd5aeb952615602ddb6ea30c9e212edf2237d1061da3368b4da.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/46fffae3569f6fd5aeb952615602ddb6ea30c9e212edf2237d1061da3368b4da.jpg)

![4e06dbbd74281700e4975cd262cb6ad5f9741a84205eda390d921a17785bedd8.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/4e06dbbd74281700e4975cd262cb6ad5f9741a84205eda390d921a17785bedd8.jpg)

![6a4b3ab41ff72b02c1fb873acddc42228c4b8a6319f21ef4df9cf457750526cc.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/6a4b3ab41ff72b02c1fb873acddc42228c4b8a6319f21ef4df9cf457750526cc.jpg)

![72608b44ceb7ec8d74929229dc50e8dec2a6e3f448a7053ecbb70a5949c9970e.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/72608b44ceb7ec8d74929229dc50e8dec2a6e3f448a7053ecbb70a5949c9970e.jpg)

![a3f284a68430639008e9094b965320521af88e762ffe66a383a8817171411566.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/a3f284a68430639008e9094b965320521af88e762ffe66a383a8817171411566.jpg)

![a76c75f971568610d388df17d494bcc062961df8178f339f519b632589c18fbc.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/a76c75f971568610d388df17d494bcc062961df8178f339f519b632589c18fbc.jpg)

![a8d4f0932b9948be0db03a1c89b4131cafc6bbbc7544cbb176f4e789f40ea6fc.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/a8d4f0932b9948be0db03a1c89b4131cafc6bbbc7544cbb176f4e789f40ea6fc.jpg)

![aa2d99400fd9b835ea97ac58e248c47a7fade0e5eafb8f665e7bdbab32c6433e.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/aa2d99400fd9b835ea97ac58e248c47a7fade0e5eafb8f665e7bdbab32c6433e.jpg)

![b550cb3d28747eae42f59fda1f13098ce5611d91e45d7be67efa0a0c4e36ed5c.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/b550cb3d28747eae42f59fda1f13098ce5611d91e45d7be67efa0a0c4e36ed5c.jpg)

![bdc9a36520a5f3511fce47aa68fd6a8c2576b3eff465c48e8aa7181f5b0f98ab.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/bdc9a36520a5f3511fce47aa68fd6a8c2576b3eff465c48e8aa7181f5b0f98ab.jpg)

![d382c2bf7b7a5aae190db1d16aa4b27076790dc15aec1b5b258ed1750c88ecd2.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/d382c2bf7b7a5aae190db1d16aa4b27076790dc15aec1b5b258ed1750c88ecd2.jpg)

![e4010171fdac6e83cac2a2a198d56caee2bbf6f5410cbf1fe883c2089525883e.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/e4010171fdac6e83cac2a2a198d56caee2bbf6f5410cbf1fe883c2089525883e.jpg)

![fe63996274f5da58c63e33bace8968f4ed2d37c08ab9ee94118157f08417312c.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/images/fe63996274f5da58c63e33bace8968f4ed2d37c08ab9ee94118157f08417312c.jpg)

### Tables

![067a56a8e30d75bcddac4fcc9973115b114519a5880605ea09f573d33d41d8db.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/067a56a8e30d75bcddac4fcc9973115b114519a5880605ea09f573d33d41d8db.jpg)

![183eaab21def1552f3225dabd54389465eac09b7a4577a1a705f11de6f1d6ed2.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/183eaab21def1552f3225dabd54389465eac09b7a4577a1a705f11de6f1d6ed2.jpg)

![221a7102157f4163195a0e5aaa31ee3ad35b42ca614ef27d1b272ee30148676a.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/221a7102157f4163195a0e5aaa31ee3ad35b42ca614ef27d1b272ee30148676a.jpg)

![2585a5d387de34691da04da00854673198819983c1ab8a2289b1a97f5fb64edb.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/2585a5d387de34691da04da00854673198819983c1ab8a2289b1a97f5fb64edb.jpg)

![27404fb8e45bda22d99ab472c9583a766a868dab7ff95e0dcad2f099e62d933f.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/27404fb8e45bda22d99ab472c9583a766a868dab7ff95e0dcad2f099e62d933f.jpg)

![475057ebfc8e8514eec23b967e13e3e6ece0a08f5687e9bc9d5d8b6f96e1cb9e.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/475057ebfc8e8514eec23b967e13e3e6ece0a08f5687e9bc9d5d8b6f96e1cb9e.jpg)

![4c20276fd6a628f4b5fd441d895160569470ac0beed9a1147f1a49ab3928440f.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/4c20276fd6a628f4b5fd441d895160569470ac0beed9a1147f1a49ab3928440f.jpg)

![5ae3fd0e79c249c62d18988df23512c6ddbb1c433a64cfa894cd2f1d7cf4a8e8.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/5ae3fd0e79c249c62d18988df23512c6ddbb1c433a64cfa894cd2f1d7cf4a8e8.jpg)

![601860e2b5c6b7686cc4bbd478a6b843f62b3426eac6fdaa85fe74957090cf14.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/601860e2b5c6b7686cc4bbd478a6b843f62b3426eac6fdaa85fe74957090cf14.jpg)

![70cef844ec92c7000c11977905d2603d9ce41917ab45ef65d9bb89968970abb8.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/70cef844ec92c7000c11977905d2603d9ce41917ab45ef65d9bb89968970abb8.jpg)

![78e16257ee2be35f6eaf89f8365c691a3ddb5e4dc76f0a60adc7b80696328e2b.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/78e16257ee2be35f6eaf89f8365c691a3ddb5e4dc76f0a60adc7b80696328e2b.jpg)

![8967a8d6c6baa711587907d8ddf040b08a4636671d6f70e9a7047f2dfdf760a3.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/8967a8d6c6baa711587907d8ddf040b08a4636671d6f70e9a7047f2dfdf760a3.jpg)

![a7d6bab53240b4425a7225ffff11c56ebdbf6f07eef1eeec82bf8bb7595b2a3f.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/a7d6bab53240b4425a7225ffff11c56ebdbf6f07eef1eeec82bf8bb7595b2a3f.jpg)

![b611b6f2f0ae462926c005471af41b0efc70917278cb85c1b096686007b92485.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/b611b6f2f0ae462926c005471af41b0efc70917278cb85c1b096686007b92485.jpg)

![b875f6d7187f367704b0effe5e633923afde6a40d91193bc957a31a743bb36f4.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/b875f6d7187f367704b0effe5e633923afde6a40d91193bc957a31a743bb36f4.jpg)

![bf200d0d98c683a23eaca6445d8ddabcc7b75b4ce4002cdaf0fb1aab204eee0c.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/bf200d0d98c683a23eaca6445d8ddabcc7b75b4ce4002cdaf0fb1aab204eee0c.jpg)

![cdf0f95cee44a924513714dca87da1626d9559a5f14f5d9f335008c6fc4daf8c.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/cdf0f95cee44a924513714dca87da1626d9559a5f14f5d9f335008c6fc4daf8c.jpg)

![d2251d05aae7b158913a3603918ef0b8634421f7037b8f2bf2f067fe98d344ed.jpg](../icml_results/3146_WorldMirror_%20Universal%203D%20World%20Reconstruction%20with%20Any-Prior%20Prompting/tables/d2251d05aae7b158913a3603918ef0b8634421f7037b8f2bf2f067fe98d344ed.jpg)

<a id="paper-3108"></a>

## Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space


### Images

![0d96d3ad950155f725fd552523016d98d42c2fc4e4eacf8ab01bf239c7d16e76.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/0d96d3ad950155f725fd552523016d98d42c2fc4e4eacf8ab01bf239c7d16e76.jpg)

![1f9e23adb7730e9357895939d8f12c018bcc3e0c1c3af6ef5579b825e20ba3cb.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/1f9e23adb7730e9357895939d8f12c018bcc3e0c1c3af6ef5579b825e20ba3cb.jpg)

![2398f884ceb279a0f1076d50b5efe56974b72e24c23dd245764d835b9005a960.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/2398f884ceb279a0f1076d50b5efe56974b72e24c23dd245764d835b9005a960.jpg)

![577f18d29fca744c60dde10c48c228d45cddf9704abaeb27f8d634d961998eec.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/577f18d29fca744c60dde10c48c228d45cddf9704abaeb27f8d634d961998eec.jpg)

![7098a87f88ccd6518437819c93d2a6d78cbe4396498d34df5badab52da14cac8.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/7098a87f88ccd6518437819c93d2a6d78cbe4396498d34df5badab52da14cac8.jpg)

![ea84b3df7371da180b68d805a34c8472c082eead0d9877aa36548167d44e5ddb.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/ea84b3df7371da180b68d805a34c8472c082eead0d9877aa36548167d44e5ddb.jpg)

![f57a45982853dd1c36fd8c26eedaa76d56645575ea6a8beac48d715c80ba9104.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/images/f57a45982853dd1c36fd8c26eedaa76d56645575ea6a8beac48d715c80ba9104.jpg)

### Tables

![05d860800afd022de26027a136de9c947bdcac2ffe79093ccc3aac78e2dd7140.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/05d860800afd022de26027a136de9c947bdcac2ffe79093ccc3aac78e2dd7140.jpg)

![231f7f30880ab618de61306a9da6fe399cb9f4e1f3a56bbbdcf9610c593a79a6.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/231f7f30880ab618de61306a9da6fe399cb9f4e1f3a56bbbdcf9610c593a79a6.jpg)

![259bc755576d3c71efd9a589a69e446a30ff01e734cecf5560eabb34d9018a23.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/259bc755576d3c71efd9a589a69e446a30ff01e734cecf5560eabb34d9018a23.jpg)

![3ece2d07365195c3cc758f922f6f49cf918c21480a3b262a1448aaa95f581e25.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/3ece2d07365195c3cc758f922f6f49cf918c21480a3b262a1448aaa95f581e25.jpg)

![719a84f3b3701562e00b6603c15370d5acfa57e02dfbcaad6e911850d901e386.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/719a84f3b3701562e00b6603c15370d5acfa57e02dfbcaad6e911850d901e386.jpg)

![aaa9fc9b3d06dbf256f7f349dda227ccf42b17f1a94faf0977e08e6162604988.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/aaa9fc9b3d06dbf256f7f349dda227ccf42b17f1a94faf0977e08e6162604988.jpg)

![ad2e4e477b3f99b4789e706dc5dededc0f57e8c764d581cc6f6d76671c90a125.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/ad2e4e477b3f99b4789e706dc5dededc0f57e8c764d581cc6f6d76671c90a125.jpg)

![bea0d65ce5e35b38e74f15915c05022e84d317aa83fcc15bc5691488fb98f779.jpg](../icml_results/3147_Thoughtbubbles_%20an%20Unsupervised%20Method%20for%20Parallel%20Thinking%20in%20Latent%20Space/tables/bea0d65ce5e35b38e74f15915c05022e84d317aa83fcc15bc5691488fb98f779.jpg)

<a id="paper-3109"></a>

## ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity


### Images

![2adaf8322a5bf9252071e787515c3467de80b678dbcdfc4a3d23e56685c91b89.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/images/2adaf8322a5bf9252071e787515c3467de80b678dbcdfc4a3d23e56685c91b89.jpg)

![37572d2cdae38d343105f8f8f923d985dec78ae7a384062ee8dab85e91cf4389.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/images/37572d2cdae38d343105f8f8f923d985dec78ae7a384062ee8dab85e91cf4389.jpg)

![859acb20325f433443021847242e10b37057e15c44763bb4284ef08206bcdb7b.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/images/859acb20325f433443021847242e10b37057e15c44763bb4284ef08206bcdb7b.jpg)

![f3122c8136ad78bba3d511cd1c693a03c5190c29c74cd6d03ee6942f2e103707.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/images/f3122c8136ad78bba3d511cd1c693a03c5190c29c74cd6d03ee6942f2e103707.jpg)

### Tables

![026c84d2a1750d964733bb7364f5efea5afc2416d8cba6a00c3ed1415a177ab8.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/026c84d2a1750d964733bb7364f5efea5afc2416d8cba6a00c3ed1415a177ab8.jpg)

![19f07d8fbe8454b1e15ea73d5f0f58100ccd4773d5fc3fe0260024d71e5ff143.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/19f07d8fbe8454b1e15ea73d5f0f58100ccd4773d5fc3fe0260024d71e5ff143.jpg)

![8d6101d391a7f1b7cc97532f73883407cf4479f8645a767dedf9b78d2d4d7229.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/8d6101d391a7f1b7cc97532f73883407cf4479f8645a767dedf9b78d2d4d7229.jpg)

![a0b1e6b794ec5c1ac60ea4dafc6f93b86bf0ea4692c565acddfd8905f57ffbad.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/a0b1e6b794ec5c1ac60ea4dafc6f93b86bf0ea4692c565acddfd8905f57ffbad.jpg)

![eb95342a10c09ab565307a09456959c68fd544d71657953119c9a295927a005a.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/eb95342a10c09ab565307a09456959c68fd544d71657953119c9a295927a005a.jpg)

![ff3e1c4c9e18bf4b1fe8e09d1ae075a2ead77b98cea770aa906040ce6f022e06.jpg](../icml_results/3148_ABC-Bench_%20An%20Agentic%20Bio-Capabilities%20Benchmark%20for%20Biosecurity/tables/ff3e1c4c9e18bf4b1fe8e09d1ae075a2ead77b98cea770aa906040ce6f022e06.jpg)

<a id="paper-3110"></a>

## Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding


### Images

![18c95dac09856000f575ab4463b8c2a38391cc06bfbdc86995a300be4e329f6b.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/18c95dac09856000f575ab4463b8c2a38391cc06bfbdc86995a300be4e329f6b.jpg)

![1921f2ff6992abcfd8cc05a63da133b0ccb8e2da36f091775ee1fbb1ef0a6ed0.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/1921f2ff6992abcfd8cc05a63da133b0ccb8e2da36f091775ee1fbb1ef0a6ed0.jpg)

![1f286465a94f24f01bcbd3a2bf9ae980fe0b4e3bb7d0c3a33148e697ebe3d524.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/1f286465a94f24f01bcbd3a2bf9ae980fe0b4e3bb7d0c3a33148e697ebe3d524.jpg)

![3b7323e5040aa6e57cc58cc8cf6c560ff4dcfeb91f63f651a218da5d581946f9.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/3b7323e5040aa6e57cc58cc8cf6c560ff4dcfeb91f63f651a218da5d581946f9.jpg)

![3e4ae791116930186af39994979ddc137653ca52272bf362e7c039c079bb08dc.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/3e4ae791116930186af39994979ddc137653ca52272bf362e7c039c079bb08dc.jpg)

![a1c4d0fdd0a3c3a29b097aa58400e64041103442db9a310ffd272d38cd37a038.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/a1c4d0fdd0a3c3a29b097aa58400e64041103442db9a310ffd272d38cd37a038.jpg)

![c804a024cf73fc02577beedad0b26d23897d2e62d9cd4050cfd6003d15ad5be6.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/c804a024cf73fc02577beedad0b26d23897d2e62d9cd4050cfd6003d15ad5be6.jpg)

![c81cef03bf4ce662d36867d43424b6031d80dcc314cc84f47b66170f94356b29.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/images/c81cef03bf4ce662d36867d43424b6031d80dcc314cc84f47b66170f94356b29.jpg)

### Tables

![2d85a4f3f381df4fd028b172b3d1cd75a3d824c6e0e74a97d7ad5e2651292ebf.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/2d85a4f3f381df4fd028b172b3d1cd75a3d824c6e0e74a97d7ad5e2651292ebf.jpg)

![45774a6057d0b1b80018989942773b9a3097fc9f0f4479db48c54248f5bc332f.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/45774a6057d0b1b80018989942773b9a3097fc9f0f4479db48c54248f5bc332f.jpg)

![4f94ede837efdbb3310a0bf6828a69cb99ece4c1974d1b33f240b417221f2c25.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/4f94ede837efdbb3310a0bf6828a69cb99ece4c1974d1b33f240b417221f2c25.jpg)

![526986c318bf73b92c885cdad1250f10558ca498c0845345c1547eb2fe5f5db3.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/526986c318bf73b92c885cdad1250f10558ca498c0845345c1547eb2fe5f5db3.jpg)

![75b7e970626bd8770869ff020e0b0edef88983e29c0eb2a168253b04ec00d550.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/75b7e970626bd8770869ff020e0b0edef88983e29c0eb2a168253b04ec00d550.jpg)

![94a4396c4b4f3c31a363b6e300a2a4b3156bc527474a89900b91965e7ac43e01.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/94a4396c4b4f3c31a363b6e300a2a4b3156bc527474a89900b91965e7ac43e01.jpg)

![d1032db8db3d7350a6500b6da274ca83a79f0e318054bf67cfb96c61e5089107.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/d1032db8db3d7350a6500b6da274ca83a79f0e318054bf67cfb96c61e5089107.jpg)

![d133f2dff676c52ccc5bfc4728befd3ccf7c25a73b8ffafa32fc9d7c613bcfab.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/d133f2dff676c52ccc5bfc4728befd3ccf7c25a73b8ffafa32fc9d7c613bcfab.jpg)

![ee29b55a0d4df1643fb0b29c0ca26701a03d570b9822702e85b3cde0f7a5acb4.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/ee29b55a0d4df1643fb0b29c0ca26701a03d570b9822702e85b3cde0f7a5acb4.jpg)

![fe860968524739175e840aabaa3afd2491461be3e6bfdb128472de23b9dbdcc4.jpg](../icml_results/3149_Faster%20Than%20Flash_%20Exploiting%20Attention%20Sparsity%20for%20Efficient%20Long-Context%20Decoding/tables/fe860968524739175e840aabaa3afd2491461be3e6bfdb128472de23b9dbdcc4.jpg)

<a id="paper-3111"></a>

## Sample from What You See: Visuomotor Policy Learning via Diffusion Bridge with Observation-Embedded Stochastic Differential Equation


### Images

![0277b09654d4ff6c402fe441b966048ec18a597d6dc40b983241b164b2120e6b.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/0277b09654d4ff6c402fe441b966048ec18a597d6dc40b983241b164b2120e6b.jpg)

![0316e8edf00e2ff6aabc98006434833ef3dc2e6b79f9ebbaadbedfaa0895c242.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/0316e8edf00e2ff6aabc98006434833ef3dc2e6b79f9ebbaadbedfaa0895c242.jpg)

![083e96871bb1eb8cfb9f620947977dede4983c71009e8e2175181ae999db3b0c.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/083e96871bb1eb8cfb9f620947977dede4983c71009e8e2175181ae999db3b0c.jpg)

![0db8d9c9c79ad51483e42485aaa3ea5b801fe500d692df437eaa38da5bae3524.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/0db8d9c9c79ad51483e42485aaa3ea5b801fe500d692df437eaa38da5bae3524.jpg)

![17fc12509af7dfc8ff1b3ba37e3ec598f4e62d59e98fcbad1aba0f79c6e1bd08.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/17fc12509af7dfc8ff1b3ba37e3ec598f4e62d59e98fcbad1aba0f79c6e1bd08.jpg)

![3b1d3b609f4ac2b0ca07f7e49d30fe4044bbea011a528471622a78e4449f457c.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/3b1d3b609f4ac2b0ca07f7e49d30fe4044bbea011a528471622a78e4449f457c.jpg)

![6baaa589fafbe312841ceb851fa7ac8dbcbd0abc70a3192e04415d51ac94347b.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/6baaa589fafbe312841ceb851fa7ac8dbcbd0abc70a3192e04415d51ac94347b.jpg)

![7058a5ce2259f8cf9f7054ec3e413daf2716519b0fc74c90b000e89148287faf.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/7058a5ce2259f8cf9f7054ec3e413daf2716519b0fc74c90b000e89148287faf.jpg)

![a7d62f02e9ebf3bad9dbc38e17f90c6fad3b1417efe9be25a721d9003457a799.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/a7d62f02e9ebf3bad9dbc38e17f90c6fad3b1417efe9be25a721d9003457a799.jpg)

![bde5d23ab8aa8132645db90361eec5f6f2fc2aaed79c6dd0662e3cfdc2e18c00.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/bde5d23ab8aa8132645db90361eec5f6f2fc2aaed79c6dd0662e3cfdc2e18c00.jpg)

![bec7d024fe93a959b9087ad1a815717c3883eda491f6ae5bf915534d91bc6761.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/bec7d024fe93a959b9087ad1a815717c3883eda491f6ae5bf915534d91bc6761.jpg)

![da5901f0b86a463f2a87dbeee603aeb6cf9c43eaffe1263bfe2f7ac088f11622.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/da5901f0b86a463f2a87dbeee603aeb6cf9c43eaffe1263bfe2f7ac088f11622.jpg)

![e16e3511b11a48ad4a59deaf3c1af4925f505add3d8c1d95ad76099c7a4e26f2.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/e16e3511b11a48ad4a59deaf3c1af4925f505add3d8c1d95ad76099c7a4e26f2.jpg)

![f817d291079a77547ccae79968b7222a1020a2adecec0eda3838f67175ec7381.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/f817d291079a77547ccae79968b7222a1020a2adecec0eda3838f67175ec7381.jpg)

![fc4b9cb68e987c86f7186a1398cf023fc594256441d4a1964ef499619f256f67.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/images/fc4b9cb68e987c86f7186a1398cf023fc594256441d4a1964ef499619f256f67.jpg)

### Tables

![0a3d857edac9c2fb6a829ede60e72de39fccae8b8db09871eba832df10fa26ae.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/0a3d857edac9c2fb6a829ede60e72de39fccae8b8db09871eba832df10fa26ae.jpg)

![0f3d29a62a72f6af5a2cdfaf89ba9ddb37d400c2cd78343be6bd163544f9458b.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/0f3d29a62a72f6af5a2cdfaf89ba9ddb37d400c2cd78343be6bd163544f9458b.jpg)

![2927cfb41ab133f544c488491556240e264585c85edc4c4a88c9403c9a11285c.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/2927cfb41ab133f544c488491556240e264585c85edc4c4a88c9403c9a11285c.jpg)

![322b07511e96b64c52ee275e28c78cd0c70bf66c44fefc673d17a4c155b59322.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/322b07511e96b64c52ee275e28c78cd0c70bf66c44fefc673d17a4c155b59322.jpg)

![45dd3a8db0ee5b36f8295f55906774d64bb81e10c21d2fc58d8b1673fb990465.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/45dd3a8db0ee5b36f8295f55906774d64bb81e10c21d2fc58d8b1673fb990465.jpg)

![598baa9a1c90825f332a5debb034f8b26b80358b8d5bed3c7435a2c4642edf52.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/598baa9a1c90825f332a5debb034f8b26b80358b8d5bed3c7435a2c4642edf52.jpg)

![5b9ca503a36d4c55b4f9190a6352e429de6014b9073fee0b3b58e0bcefbfafb8.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/5b9ca503a36d4c55b4f9190a6352e429de6014b9073fee0b3b58e0bcefbfafb8.jpg)

![6d1c4759c36e8a69bcb68e7ae7d8c7f5935a5577a7037e4ad96e02b1b1762e12.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/6d1c4759c36e8a69bcb68e7ae7d8c7f5935a5577a7037e4ad96e02b1b1762e12.jpg)

![761de52dcc979b6b75c61c9cbce470b5bbc0eb7ccb786d7c80ce9a89e547e5e0.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/761de52dcc979b6b75c61c9cbce470b5bbc0eb7ccb786d7c80ce9a89e547e5e0.jpg)

![7e75a2ebd644683315c5b56d037850597ce42c0b9c6da3ca012a6598a928a35b.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/7e75a2ebd644683315c5b56d037850597ce42c0b9c6da3ca012a6598a928a35b.jpg)

![82906e298c82a188d2504bcc1df8b64b2cba42ad3da21e7b8b1565b1e059d1c2.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/82906e298c82a188d2504bcc1df8b64b2cba42ad3da21e7b8b1565b1e059d1c2.jpg)

![8c14095d0de4cee003c0d4cd7b703ad5344abd3edd13d292b5d7a14095b84d40.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/8c14095d0de4cee003c0d4cd7b703ad5344abd3edd13d292b5d7a14095b84d40.jpg)

![8dbf4b35dacf634a169e6f4ce4d19bdff389085839ba60c76bc171c222380f63.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/8dbf4b35dacf634a169e6f4ce4d19bdff389085839ba60c76bc171c222380f63.jpg)

![94a592be00a08e1a151df2b5c7a8ac922c6495182fc877f3210aa4f761afaa27.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/94a592be00a08e1a151df2b5c7a8ac922c6495182fc877f3210aa4f761afaa27.jpg)

![af3e42c5e8f68f04a7a5f149e8b9d9c6aa8c02b15101b4e5198e14844f1b231f.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/af3e42c5e8f68f04a7a5f149e8b9d9c6aa8c02b15101b4e5198e14844f1b231f.jpg)

![b4b633529b7e4df7e333d848d2957e59492716bb31ee2417378876dd1c51c91e.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/b4b633529b7e4df7e333d848d2957e59492716bb31ee2417378876dd1c51c91e.jpg)

![d08770f23b6282068303a1bfbeaa14b193baa0f29bbcb6c230704550d64ba714.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/d08770f23b6282068303a1bfbeaa14b193baa0f29bbcb6c230704550d64ba714.jpg)

![d3520803f07dbadcc744f4222b8db89c221b7d54fc8db81b08999a22e760820b.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/d3520803f07dbadcc744f4222b8db89c221b7d54fc8db81b08999a22e760820b.jpg)

![e7c2ddbbc0c4fbf1576f2bffda0cf4eec4058faaf7db21dcba72416a7ebb15f8.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/e7c2ddbbc0c4fbf1576f2bffda0cf4eec4058faaf7db21dcba72416a7ebb15f8.jpg)

![fdeb4e843bf29f8879356e5c03e791db11be1dfcd6436a1b48ef92a24bf44290.jpg](../icml_results/3150_Sample%20from%20What%20You%20See_%20Visuomotor%20Policy%20Learning%20via%20Diffusion%20Bridge%20with%20Observation-Embedded%20/tables/fdeb4e843bf29f8879356e5c03e791db11be1dfcd6436a1b48ef92a24bf44290.jpg)

<a id="paper-3112"></a>

## From Out-of-Distribution Detection to Hallucination Detection: A Geometric View


### Images

![e7b4a35381c7f4b20c971c904df358d21bf01be494a5f12787c53bce304b9b64.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/images/e7b4a35381c7f4b20c971c904df358d21bf01be494a5f12787c53bce304b9b64.jpg)

![ebd38c58638e29c1f99e8097fd70f10671e1aa4935ea58fad9d8aef671916277.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/images/ebd38c58638e29c1f99e8097fd70f10671e1aa4935ea58fad9d8aef671916277.jpg)

### Tables

![0fe183ce807e327f5c3c72712964d3902c70e4a179de9737f4bdf27ec76745a7.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/0fe183ce807e327f5c3c72712964d3902c70e4a179de9737f4bdf27ec76745a7.jpg)

![275d2b49749f57a6641e9d9c9c134922ff1c742a15d23468ff27a2d12e9ac831.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/275d2b49749f57a6641e9d9c9c134922ff1c742a15d23468ff27a2d12e9ac831.jpg)

![28105ca023dac86b49821860b828aeb1ced772285947737dbecddeebb050a681.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/28105ca023dac86b49821860b828aeb1ced772285947737dbecddeebb050a681.jpg)

![4653009d003b1bc4c2306c314dcf97b6e968e283ff935c93d77eeb88c081a42e.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/4653009d003b1bc4c2306c314dcf97b6e968e283ff935c93d77eeb88c081a42e.jpg)

![63af1aad9485691aaf91d0f78d8552bffbb62f6be81b6a157d88ba7db7b840db.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/63af1aad9485691aaf91d0f78d8552bffbb62f6be81b6a157d88ba7db7b840db.jpg)

![6559462deeed1196f8341abcc59b90d75ca41fc2b5312e84a710db16bfe6e985.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/6559462deeed1196f8341abcc59b90d75ca41fc2b5312e84a710db16bfe6e985.jpg)

![6c23bfb7b07de286bedfcc99aa1e43b1243306f8801bdc5bb7160ec66f9893b4.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/6c23bfb7b07de286bedfcc99aa1e43b1243306f8801bdc5bb7160ec66f9893b4.jpg)

![aa81fe5596f750d9c446e739a0c3258add8c298842676dd553542f5420544b91.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/aa81fe5596f750d9c446e739a0c3258add8c298842676dd553542f5420544b91.jpg)

![afd3a753d8df54f5b7c9ffc8861cd0be19f0653c71f949750b5ca14441c8eb49.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/afd3a753d8df54f5b7c9ffc8861cd0be19f0653c71f949750b5ca14441c8eb49.jpg)

![b715af4fc13de294a63f9cd682e7321e12fb8d8f0d2065fae7815ca147dda51a.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/b715af4fc13de294a63f9cd682e7321e12fb8d8f0d2065fae7815ca147dda51a.jpg)

![ce6c561694253fb7bb560c9a52db6fa603113d9cec2dd6c298d5b17c67a5d5a1.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/ce6c561694253fb7bb560c9a52db6fa603113d9cec2dd6c298d5b17c67a5d5a1.jpg)

![dd01e7447958f8b9eb53eb4a357ed1ccd00ef1c353d5472fdd9fa90b77a6f743.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/dd01e7447958f8b9eb53eb4a357ed1ccd00ef1c353d5472fdd9fa90b77a6f743.jpg)

![e9178f605c46397ccded7f447fd9bbab37d5dfdf964c4a7c099001266e2226cf.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/e9178f605c46397ccded7f447fd9bbab37d5dfdf964c4a7c099001266e2226cf.jpg)

![f17a3ca73d26e8d026538c8656e5436a6d68f1ea6a3a1873e878a357200b4150.jpg](../icml_results/3151_From%20Out-of-Distribution%20Detection%20to%20Hallucination%20Detection_%20A%20Geometric%20View/tables/f17a3ca73d26e8d026538c8656e5436a6d68f1ea6a3a1873e878a357200b4150.jpg)

<a id="paper-3113"></a>

## Post-Training Language Models for Crosslingual Consistency


### Images

![034f5d42fd6147008eb9967724f8a98795f5ebc5a47cfd4fee8b16fabe2210ad.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/034f5d42fd6147008eb9967724f8a98795f5ebc5a47cfd4fee8b16fabe2210ad.jpg)

![124dba2a97424c561ebb0589b7e396c2e352e596eaaae915774ddd55b58f562c.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/124dba2a97424c561ebb0589b7e396c2e352e596eaaae915774ddd55b58f562c.jpg)

![184fed7c8f4e550f730ed917660ffa3506c0e369e763c1e494b608877a21ebf9.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/184fed7c8f4e550f730ed917660ffa3506c0e369e763c1e494b608877a21ebf9.jpg)

![3ad6bfc7b3a51078af59cad904302d567bd8fbd41cf58f9a0d55221b6634360b.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/3ad6bfc7b3a51078af59cad904302d567bd8fbd41cf58f9a0d55221b6634360b.jpg)

![90ecfa244989f68255fcfe547b3438e669ffd228343ca5d3016db9277722b75c.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/90ecfa244989f68255fcfe547b3438e669ffd228343ca5d3016db9277722b75c.jpg)

![beefd43a6d9a7543e0ac41852b2044bcb6972fbd6013d487686498560acb1899.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/beefd43a6d9a7543e0ac41852b2044bcb6972fbd6013d487686498560acb1899.jpg)

![dfc6dd49169c760c5d416adf309905fd5370799f4c82abfb2d6ec7073fa9a89d.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/dfc6dd49169c760c5d416adf309905fd5370799f4c82abfb2d6ec7073fa9a89d.jpg)

![e7d1d158f5646f26a542dee44d62ddd13bf852af24a1166475d65b34849c316b.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/e7d1d158f5646f26a542dee44d62ddd13bf852af24a1166475d65b34849c316b.jpg)

![eec48dc87877f3bbe08517699d8e311e2130d7888cdef3dab38bbee4b97753d0.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/images/eec48dc87877f3bbe08517699d8e311e2130d7888cdef3dab38bbee4b97753d0.jpg)

### Tables

![150887b935c5be281dd566e0a69eeb6ab223ab442cf3c338dfc8b95469f939ae.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/150887b935c5be281dd566e0a69eeb6ab223ab442cf3c338dfc8b95469f939ae.jpg)

![1a3ca67d4db937dddbea5e4da82ecb522a1802f1052e1e1fe46c64c284ce98aa.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/1a3ca67d4db937dddbea5e4da82ecb522a1802f1052e1e1fe46c64c284ce98aa.jpg)

![23ef5875e3326662f31d44a9cce03e9daa47e1763b4a9c13a8b39b9689c377d5.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/23ef5875e3326662f31d44a9cce03e9daa47e1763b4a9c13a8b39b9689c377d5.jpg)

![312095ec316fbaf227ac17dedb8c36b4dee7ff275c10e69b627a01b9c0796285.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/312095ec316fbaf227ac17dedb8c36b4dee7ff275c10e69b627a01b9c0796285.jpg)

![47c75b028122c77ec0002e7e2d9cd995b815e0c8670dcf6524e5d2fd9f70748b.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/47c75b028122c77ec0002e7e2d9cd995b815e0c8670dcf6524e5d2fd9f70748b.jpg)

![4b2aabe96d0aa2df6587f9cd5ea39714063985f69ab35d55dc4cabeecbb37fe6.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/4b2aabe96d0aa2df6587f9cd5ea39714063985f69ab35d55dc4cabeecbb37fe6.jpg)

![542827c3d38e264ca96ad8f6e4deba950bf9677d210d47a98d3f3fffc64f0dac.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/542827c3d38e264ca96ad8f6e4deba950bf9677d210d47a98d3f3fffc64f0dac.jpg)

![5594a4b654143085179001e86faa33efffeb520286869b7572d8ab3958ae4863.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/5594a4b654143085179001e86faa33efffeb520286869b7572d8ab3958ae4863.jpg)

![65413a4270e0708bb59630d766cdbd4d297b263e0bc3a5dace9d8ac2d016ea87.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/65413a4270e0708bb59630d766cdbd4d297b263e0bc3a5dace9d8ac2d016ea87.jpg)

![78dd5c2d5c9d7738f9c363d5c3d477709144648b0c20e8758a61d0f675c75cdf.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/78dd5c2d5c9d7738f9c363d5c3d477709144648b0c20e8758a61d0f675c75cdf.jpg)

![793c02b64520edc359333a58f33cd906d217a25c5f6b82e3dd8f3e7fa0169cbf.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/793c02b64520edc359333a58f33cd906d217a25c5f6b82e3dd8f3e7fa0169cbf.jpg)

![7ebbd0cbf7a88999344bd830b41d216fe6260c44224b5346252a6c7078631ba5.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/7ebbd0cbf7a88999344bd830b41d216fe6260c44224b5346252a6c7078631ba5.jpg)

![7f30cc6bd215ac34d8c153488bdba55ff4c98b3d582e6f4af95ce5dc2361161c.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/7f30cc6bd215ac34d8c153488bdba55ff4c98b3d582e6f4af95ce5dc2361161c.jpg)

![8719f69b3611625f66d5488000658c4fb9073b08d3aace12367377dc4a61d898.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/8719f69b3611625f66d5488000658c4fb9073b08d3aace12367377dc4a61d898.jpg)

![d91e2f6bad335d1140094b44d3cca20722c924cec773f5a3735b56926b0a176d.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/d91e2f6bad335d1140094b44d3cca20722c924cec773f5a3735b56926b0a176d.jpg)

![e523cf04422360aca3883a98e1c572a3404a4ee95ef983b19e4319023af500ee.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/e523cf04422360aca3883a98e1c572a3404a4ee95ef983b19e4319023af500ee.jpg)

![fc14dbf125ae540194785864105a890bfdc7ba0d65c1acc4f7f174130490c8f0.jpg](../icml_results/3152_Post-Training%20Language%20Models%20for%20Crosslingual%20Consistency/tables/fc14dbf125ae540194785864105a890bfdc7ba0d65c1acc4f7f174130490c8f0.jpg)

<a id="paper-3114"></a>

## On Effectiveness and Efficiency of Agentic Tool-calling and RL Training


### Images

![0266d54bda33be08a96c785aaa42b6f7f84cef33d3b7c6db116401044ba5f919.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/0266d54bda33be08a96c785aaa42b6f7f84cef33d3b7c6db116401044ba5f919.jpg)

![42a426bb5ca17bdba673b133034cc3ec1b4024702179787fd3ec344299374185.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/42a426bb5ca17bdba673b133034cc3ec1b4024702179787fd3ec344299374185.jpg)

![43f066d53a364553bee8b7944d21e73d2a025e8ee373b324b6027913e62353cf.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/43f066d53a364553bee8b7944d21e73d2a025e8ee373b324b6027913e62353cf.jpg)

![50d8cd1f60c23745c14b00f18f5eda4abe003884a7c1111cc05e80709e343c46.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/50d8cd1f60c23745c14b00f18f5eda4abe003884a7c1111cc05e80709e343c46.jpg)

![789e8ff68d2dd12873788ab20230978c0502d3f48822df35908e5e52cd57de67.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/789e8ff68d2dd12873788ab20230978c0502d3f48822df35908e5e52cd57de67.jpg)

![81810d6e298163cc86d09552545ea3274bcb9c7750e1d42ac3d795d4ef949176.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/81810d6e298163cc86d09552545ea3274bcb9c7750e1d42ac3d795d4ef949176.jpg)

![9552b8f9434ad1daa7c9c66a998babcaae9add849398fc1b1810b03938e241f7.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/9552b8f9434ad1daa7c9c66a998babcaae9add849398fc1b1810b03938e241f7.jpg)

![a6ac54237f6c76fa3031617a0afffc9763b6e0678b2c699d58499df719eaf1a3.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/a6ac54237f6c76fa3031617a0afffc9763b6e0678b2c699d58499df719eaf1a3.jpg)

![a9eb1f56969930dff0a60bb7b631e91a69ce21660479e252c404c4426ed09abc.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/a9eb1f56969930dff0a60bb7b631e91a69ce21660479e252c404c4426ed09abc.jpg)

![b90a54ad7cc5546e01af751cda1c9a220f62e03cdce2e841d94f4d8001e98a94.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/b90a54ad7cc5546e01af751cda1c9a220f62e03cdce2e841d94f4d8001e98a94.jpg)

![ba6bfd5cdc892166081f45bed73bf283936d41f40843074ad48a72aff64d340f.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/ba6bfd5cdc892166081f45bed73bf283936d41f40843074ad48a72aff64d340f.jpg)

![bf85e72dea9af6c7d2ccf5e3c05b914b60e21bdbd354326d22dd079b5040aac6.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/bf85e72dea9af6c7d2ccf5e3c05b914b60e21bdbd354326d22dd079b5040aac6.jpg)

![c18bcb1bd00694c51f61dcaf2106cf6022e09fd4c5983818b206de127e0b8959.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/c18bcb1bd00694c51f61dcaf2106cf6022e09fd4c5983818b206de127e0b8959.jpg)

![dc1a1132206920e0bf181e041e0e9b2be3a8d6c99626dc11a726944889927b68.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/images/dc1a1132206920e0bf181e041e0e9b2be3a8d6c99626dc11a726944889927b68.jpg)

### Tables

![2a9176c77165e3addca0f5ba7bf36362944d02b80d8a8c7756c3ebe01a9badc7.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/2a9176c77165e3addca0f5ba7bf36362944d02b80d8a8c7756c3ebe01a9badc7.jpg)

![56fb7d1b0aead3be7ff7761e78f67cd6d8dc15eb7e64cbe4d10f764e05762779.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/56fb7d1b0aead3be7ff7761e78f67cd6d8dc15eb7e64cbe4d10f764e05762779.jpg)

![9eaf946543c3cc5051f546b22d0d51246d6d16529d040add394ce596785b00d4.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/9eaf946543c3cc5051f546b22d0d51246d6d16529d040add394ce596785b00d4.jpg)

![a13d5e342feb295c12276eb217b76a4b0632d5664f8d2c5ea5a046293dd9127f.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/a13d5e342feb295c12276eb217b76a4b0632d5664f8d2c5ea5a046293dd9127f.jpg)

![cb4dd5eafdd9e982b0acb37573a983424a67f2b5479d5ace5a41b379cbbcaf29.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/cb4dd5eafdd9e982b0acb37573a983424a67f2b5479d5ace5a41b379cbbcaf29.jpg)

![d04db570bf0f88de668a9ba066d6cb9da21d87235a224555dbaad4ca950020e3.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/d04db570bf0f88de668a9ba066d6cb9da21d87235a224555dbaad4ca950020e3.jpg)

![d364be10dcd4a8a703847665ad16c92ad23af75427c0ab96d4b52666c41a757a.jpg](../icml_results/3153_On%20Effectiveness%20and%20Efficiency%20of%20Agentic%20Tool-calling%20and%20RL%20Training/tables/d364be10dcd4a8a703847665ad16c92ad23af75427c0ab96d4b52666c41a757a.jpg)

<a id="paper-3115"></a>

## From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide ML Interatomic Potential Architectures


### Images

![1756c0c1762c061ef46060c5bac804fda371b3ac26abf42f28e19b6d9b437399.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/1756c0c1762c061ef46060c5bac804fda371b3ac26abf42f28e19b6d9b437399.jpg)

![27994c4a7895476d248be9bf24c9a01e04b80d7dccb593fdea47e6a2ed0275f0.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/27994c4a7895476d248be9bf24c9a01e04b80d7dccb593fdea47e6a2ed0275f0.jpg)

![4aacc218aa4153b0923afc9ac929d830dbe2ef5ef15f91d47b5f5fa1a29a7e64.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/4aacc218aa4153b0923afc9ac929d830dbe2ef5ef15f91d47b5f5fa1a29a7e64.jpg)

![5cc934b69e2d0c4e653122c0c622a7cc73e14b1c46e3be284e8036e0fdb5abdb.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/5cc934b69e2d0c4e653122c0c622a7cc73e14b1c46e3be284e8036e0fdb5abdb.jpg)

![65e4757d65a894bb99ead89a78dceb83121ad58184ab2cf43dbc4bcf2982dffa.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/65e4757d65a894bb99ead89a78dceb83121ad58184ab2cf43dbc4bcf2982dffa.jpg)

![a92cc49ed8c54052050392a19e06fd1e35f37c2efc7e8f7002dfcc8990429cbd.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/a92cc49ed8c54052050392a19e06fd1e35f37c2efc7e8f7002dfcc8990429cbd.jpg)

![ceea0958f13c3ca38d624cd72e53e6cbec37a0f8f9ddc9c9808015bcb76f7fc9.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/ceea0958f13c3ca38d624cd72e53e6cbec37a0f8f9ddc9c9808015bcb76f7fc9.jpg)

![e43d0bc6957ca55b863d93f33d745a14c28245af7b5de8689b3d56b069e8372f.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/e43d0bc6957ca55b863d93f33d745a14c28245af7b5de8689b3d56b069e8372f.jpg)

![f8d669f254bd172fea6dc9be6208aa9a32747a8af88893c7c08018886fa75a98.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/images/f8d669f254bd172fea6dc9be6208aa9a32747a8af88893c7c08018886fa75a98.jpg)

### Tables

![07fddb740a6a926b3273c34df65d715e1f2d159ffe4f3ae938444e42a4140ab7.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/07fddb740a6a926b3273c34df65d715e1f2d159ffe4f3ae938444e42a4140ab7.jpg)

![1dff1b41a998b18a065631b81e1ceb8e0784a5a444c2b6905b6a6148e9488fd1.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/1dff1b41a998b18a065631b81e1ceb8e0784a5a444c2b6905b6a6148e9488fd1.jpg)

![337a10ca67bad40af0ebe1856d50e7252526a2228670d7d421812b47806fe168.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/337a10ca67bad40af0ebe1856d50e7252526a2228670d7d421812b47806fe168.jpg)

![7d0ab7fa05bce62a7489e2b6034ea9b61746be4fcdd05c615baf5adbb2c3c70b.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/7d0ab7fa05bce62a7489e2b6034ea9b61746be4fcdd05c615baf5adbb2c3c70b.jpg)

![a927d80fdbca649071df11f69b85fb07bfdd735071c1c569fae8279ceb78e0e2.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/a927d80fdbca649071df11f69b85fb07bfdd735071c1c569fae8279ceb78e0e2.jpg)

![c64f8944749aa64214690f80f1a037a79cd7dd27105b3baef86cf27a45950c87.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/c64f8944749aa64214690f80f1a037a79cd7dd27105b3baef86cf27a45950c87.jpg)

![ec8fb7d507d0a2b604363928d0ee98abf5060871c3aaaa25c2e7a756a27b0efa.jpg](../icml_results/3154_From%20Evaluation%20to%20Design_%20Using%20Potential%20Energy%20Surface%20Smoothness%20Metrics%20to%20Guide%20ML%20Interatomic/tables/ec8fb7d507d0a2b604363928d0ee98abf5060871c3aaaa25c2e7a756a27b0efa.jpg)

<a id="paper-3116"></a>

## PETS: A Principled Framework Towards Optimal Trajectory Allocation for Efficient Test-Time Self-Consistency


### Images

![07f954adc1e0991a93e0a36f957fb8469173614a1130e6a976199f6627b1dba8.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/07f954adc1e0991a93e0a36f957fb8469173614a1130e6a976199f6627b1dba8.jpg)

![15a5f903f65e35657c93453dabbdef035e1a1a6a31d4a4bb2b577b410a400d1c.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/15a5f903f65e35657c93453dabbdef035e1a1a6a31d4a4bb2b577b410a400d1c.jpg)

![1ceb69b3d67666cb88cd28c994e88496048d0c10b328b809fcee9e6ad8c5e32a.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/1ceb69b3d67666cb88cd28c994e88496048d0c10b328b809fcee9e6ad8c5e32a.jpg)

![2015e49ca5da4e9224ce7b1b87d3acb29d2388f15896c964001bfca01ef2cdb8.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/2015e49ca5da4e9224ce7b1b87d3acb29d2388f15896c964001bfca01ef2cdb8.jpg)

![2d394a59ab6e21e547f72522b12391ef11a67b0eafee423095c621c507e09e82.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/2d394a59ab6e21e547f72522b12391ef11a67b0eafee423095c621c507e09e82.jpg)

![301b5bbfc8071fee5f6b8cf9eab7e5abb927098cbd91813cbc9e6d4926f64be6.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/301b5bbfc8071fee5f6b8cf9eab7e5abb927098cbd91813cbc9e6d4926f64be6.jpg)

![45a1b9651e5ad54801ab5a0b6dabaf2c35259a8d237d386bc7e67946d53f9b89.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/45a1b9651e5ad54801ab5a0b6dabaf2c35259a8d237d386bc7e67946d53f9b89.jpg)

![6113fc9ae2eef412c40de0a9a95239a007247648db8295507becbe9a8c9a3c4c.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/6113fc9ae2eef412c40de0a9a95239a007247648db8295507becbe9a8c9a3c4c.jpg)

![64a4eee130ffcdc9f28216494b408f346a28dc67d70696232e998da3d709ddbb.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/64a4eee130ffcdc9f28216494b408f346a28dc67d70696232e998da3d709ddbb.jpg)

![67cd2bc261a99f1b87ebe4a027dde765d66cec32bbbf1bb0c9fde4ce7cd48b66.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/67cd2bc261a99f1b87ebe4a027dde765d66cec32bbbf1bb0c9fde4ce7cd48b66.jpg)

![833ec036e9cb49cbaa95af6e25d91896f9d34bb6deff65b6ba5c986fd70d264b.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/833ec036e9cb49cbaa95af6e25d91896f9d34bb6deff65b6ba5c986fd70d264b.jpg)

![8a7d2cbfe8065c0b5ec9af0027db68047a026292321467ec55aa4365bcc1f7e1.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/8a7d2cbfe8065c0b5ec9af0027db68047a026292321467ec55aa4365bcc1f7e1.jpg)

![963bf058915a8a60bfef9eb9b5fb825df55339a8cb46e723e6f40d1d37dcaded.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/963bf058915a8a60bfef9eb9b5fb825df55339a8cb46e723e6f40d1d37dcaded.jpg)

![bec74effb2cf9b94ed8fba17eac53cbafcbda286d62cc059f85787a14a9d3fd7.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/bec74effb2cf9b94ed8fba17eac53cbafcbda286d62cc059f85787a14a9d3fd7.jpg)

![cd2feaac2a2459c568dfeef37df599e0f0149fe5d34f5b1709b749cbfb3cb0f9.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/cd2feaac2a2459c568dfeef37df599e0f0149fe5d34f5b1709b749cbfb3cb0f9.jpg)

![f14d250dd2faa2034e5c36bb1f39b58ca10d8053acfce3e9b7c7d25b2105a5ed.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/f14d250dd2faa2034e5c36bb1f39b58ca10d8053acfce3e9b7c7d25b2105a5ed.jpg)

![fbb0569607d29adbc14ef7b19184944028da780fe42cc17ab86fda0b5f36eda5.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/images/fbb0569607d29adbc14ef7b19184944028da780fe42cc17ab86fda0b5f36eda5.jpg)

### Tables

![038b27804eb2ac067be5a1561fa9407d22f031399bf2ffc9a669fbac7fcfab84.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/038b27804eb2ac067be5a1561fa9407d22f031399bf2ffc9a669fbac7fcfab84.jpg)

![24343184a4f5965bf4f3e09af4495a8c2444efef03fde6c2a522ddc2f40512a0.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/24343184a4f5965bf4f3e09af4495a8c2444efef03fde6c2a522ddc2f40512a0.jpg)

![288a77351376dd3186137ca87c1d4b26238e88fa45afc5cfbcccff22e4dd09fa.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/288a77351376dd3186137ca87c1d4b26238e88fa45afc5cfbcccff22e4dd09fa.jpg)

![2a9bdc6533ad0cefd51600b87ae31d190adbebf92005897b9a2828b212228318.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/2a9bdc6533ad0cefd51600b87ae31d190adbebf92005897b9a2828b212228318.jpg)

![8b9b320c1d1eaa5e4d92eb5e8af43b290684f962ca3e51acd865c9cd39443340.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/8b9b320c1d1eaa5e4d92eb5e8af43b290684f962ca3e51acd865c9cd39443340.jpg)

![bae1b819434baba13c78d86c64ede26f6f88875417c922a4fc50ac6ac201cc4d.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/bae1b819434baba13c78d86c64ede26f6f88875417c922a4fc50ac6ac201cc4d.jpg)

![d5d6685c3c233da5f42bb8d0f136d6f7ef4deb7d3fca24219d2e5275dfefaea5.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/d5d6685c3c233da5f42bb8d0f136d6f7ef4deb7d3fca24219d2e5275dfefaea5.jpg)

![eb1fe9c364ddbd953890d4481d026cc3798ac34450c3c331a640b4a955efe0d8.jpg](../icml_results/3155_PETS_%20A%20Principled%20Framework%20Towards%20Optimal%20Trajectory%20Allocation%20for%20Efficient%20Test-Time%20Self-Cons/tables/eb1fe9c364ddbd953890d4481d026cc3798ac34450c3c331a640b4a955efe0d8.jpg)

<a id="paper-3117"></a>

## TimeLAVA: Learning-Agnostic Valuation for Time Series Data


### Images

![16567c50c34905aa7a4ba161ee58e1e58e0bc92f2bbdbaef041d684dddebdae2.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/16567c50c34905aa7a4ba161ee58e1e58e0bc92f2bbdbaef041d684dddebdae2.jpg)

![165d353b3aa1e015de30fa83d411973ec010f738575ce681e7734a2ed17d89c6.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/165d353b3aa1e015de30fa83d411973ec010f738575ce681e7734a2ed17d89c6.jpg)

![18ddde21c5c4a35ec8c6d3466d04a253f575f5ae2e5a4a8f055b5481552a07a4.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/18ddde21c5c4a35ec8c6d3466d04a253f575f5ae2e5a4a8f055b5481552a07a4.jpg)

![1a77a5f9d5617093161e3470c1a273f08330940152d1e71bbde92ac224cebf86.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/1a77a5f9d5617093161e3470c1a273f08330940152d1e71bbde92ac224cebf86.jpg)

![263dfb8ceb17381a5fdc8aafb6826539c66477db3d5f7b9ffae99d7f88ce6a63.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/263dfb8ceb17381a5fdc8aafb6826539c66477db3d5f7b9ffae99d7f88ce6a63.jpg)

![28f7bc245be1c389e6e320602c01dd99e35e48a8aa1d14eded4a7501049db4fe.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/28f7bc245be1c389e6e320602c01dd99e35e48a8aa1d14eded4a7501049db4fe.jpg)

![3a30c4adbeb1f671e803ab88a8d5fc26b69ab5a1ab65d171688042519c90b25c.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/3a30c4adbeb1f671e803ab88a8d5fc26b69ab5a1ab65d171688042519c90b25c.jpg)

![41d4c3122872053b626d05fec1b41b398cd0022a4a73afbfbd5fdefa917d7a36.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/41d4c3122872053b626d05fec1b41b398cd0022a4a73afbfbd5fdefa917d7a36.jpg)

![4d2ba5b6e7df25938cb93a7b4e5a96175ab9ba4227243b33261cf5a23da47615.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/4d2ba5b6e7df25938cb93a7b4e5a96175ab9ba4227243b33261cf5a23da47615.jpg)

![50462521bf0f5c5787c5a3538e766a73bc5f557c7594feafc5dac8c9286b2a6e.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/50462521bf0f5c5787c5a3538e766a73bc5f557c7594feafc5dac8c9286b2a6e.jpg)

![62c089ec8636c6d5e74c8cc9b87ecad38e6e3ff7a6312ce62f73d9cbbc68391a.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/62c089ec8636c6d5e74c8cc9b87ecad38e6e3ff7a6312ce62f73d9cbbc68391a.jpg)

![6bff25d2fc2868ff3ffc6bbb98b0c3be675d8fe8e3b1e01d05c24dd6aaec3709.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/6bff25d2fc2868ff3ffc6bbb98b0c3be675d8fe8e3b1e01d05c24dd6aaec3709.jpg)

![798a750d70d1d3f6536c0f54f42b9fdbd6c230aabe3dc3cb1ca0a81d16ffc839.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/798a750d70d1d3f6536c0f54f42b9fdbd6c230aabe3dc3cb1ca0a81d16ffc839.jpg)

![7c530a942ee5d7e6caa563fb59691f4d2519c4c9248c78de2f470a1bb77b1a25.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/7c530a942ee5d7e6caa563fb59691f4d2519c4c9248c78de2f470a1bb77b1a25.jpg)

![865a05c7f600f7ae0bd970c6c5ceb9813c69ee0fda47364eca3628bedca671c7.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/865a05c7f600f7ae0bd970c6c5ceb9813c69ee0fda47364eca3628bedca671c7.jpg)

![8b46ec22b94ce730561bd73ac0f8d9ef8b15b187ae38378707a28aa684d2a76b.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/8b46ec22b94ce730561bd73ac0f8d9ef8b15b187ae38378707a28aa684d2a76b.jpg)

![8c7153e4aad080fafc4175c15669c976f0820fe984eb4268e7fdff1d13a39ccb.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/8c7153e4aad080fafc4175c15669c976f0820fe984eb4268e7fdff1d13a39ccb.jpg)

![8fecf1e6158a41b2482db04af4518d84bf7ade193baf205f6d7cb391837fd795.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/8fecf1e6158a41b2482db04af4518d84bf7ade193baf205f6d7cb391837fd795.jpg)

![916f8e29b7b58c2a2e5061e269064dfc65b318a038b5d73042638fea2f60c2c3.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/916f8e29b7b58c2a2e5061e269064dfc65b318a038b5d73042638fea2f60c2c3.jpg)

![936e99b2263cf9d95b878d4eb4fa8b8eb988d5c88cffc156f4223e1e94e8a464.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/936e99b2263cf9d95b878d4eb4fa8b8eb988d5c88cffc156f4223e1e94e8a464.jpg)

![b468e656a419cc2e1a32a0c93a90c27ae530fc09e97932481d396371397c76d8.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/b468e656a419cc2e1a32a0c93a90c27ae530fc09e97932481d396371397c76d8.jpg)

![bb82672f570d7035a151e58e9abf4d3c3d5d12aa7c36c48955fbb72bbd446c0c.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/bb82672f570d7035a151e58e9abf4d3c3d5d12aa7c36c48955fbb72bbd446c0c.jpg)

![cad5d35f0faa58732e8dc73372971e0dc0e218e3f4297520836f4cf4eadb376c.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/cad5d35f0faa58732e8dc73372971e0dc0e218e3f4297520836f4cf4eadb376c.jpg)

![d6995368d272b4924b0aded670eab7fd45e53a55faa682164b05a6ac10732d67.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/d6995368d272b4924b0aded670eab7fd45e53a55faa682164b05a6ac10732d67.jpg)

![dbf5d1a83d64e2564ff81fe718ad409f3339dd365d98511b107bfbf7e22f21dc.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/dbf5d1a83d64e2564ff81fe718ad409f3339dd365d98511b107bfbf7e22f21dc.jpg)

![ffa54d6ed4d1d997928933f8d834caf91092e4e35e56198ec401b60032413ebd.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/images/ffa54d6ed4d1d997928933f8d834caf91092e4e35e56198ec401b60032413ebd.jpg)

### Tables

![1806b8e7cd432f8a75852195194c6703548432dccea9c7733c1449528bc16119.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/1806b8e7cd432f8a75852195194c6703548432dccea9c7733c1449528bc16119.jpg)

![37e7b8086bdf75d420f3fe88f2c45e589dfde343c01633a31459cbb2f504fb0a.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/37e7b8086bdf75d420f3fe88f2c45e589dfde343c01633a31459cbb2f504fb0a.jpg)

![4539b443a8c7f959bd328b9f1d67e2e6063f6077a09d89000e470096004ef2dc.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/4539b443a8c7f959bd328b9f1d67e2e6063f6077a09d89000e470096004ef2dc.jpg)

![71e7da258571f4de95280ba7b9267beca5785e55aea8831a987a4936abeb09ab.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/71e7da258571f4de95280ba7b9267beca5785e55aea8831a987a4936abeb09ab.jpg)

![9d5a21a40d9ba3a5d51dcd24685b6893f6385a74f860bfa89b2f599ec5457cf4.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/9d5a21a40d9ba3a5d51dcd24685b6893f6385a74f860bfa89b2f599ec5457cf4.jpg)

![d1a737b3c7cfbded975d4ae55247f887da88f66707a52f41f3ab6d8ce637fc39.jpg](../icml_results/3156_TimeLAVA_%20Learning-Agnostic%20Valuation%20for%20Time%20Series%20Data/tables/d1a737b3c7cfbded975d4ae55247f887da88f66707a52f41f3ab6d8ce637fc39.jpg)

<a id="paper-3118"></a>

## ProtDBench: A Unified Benchmark of Protein Binder Design and Evaluation


### Images

![28928f7aaf351b12e2bff2b4c4b332d18e87d944d1b477f0aed016dccd32bf49.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/28928f7aaf351b12e2bff2b4c4b332d18e87d944d1b477f0aed016dccd32bf49.jpg)

![326127387a4e25f437c868bc68ab362007525d0cae76b1f9635bfcefa9880d90.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/326127387a4e25f437c868bc68ab362007525d0cae76b1f9635bfcefa9880d90.jpg)

![449a188398e70122154c8553e24dcd682ef60b4e686a396e8d6ae09706f5eb9e.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/449a188398e70122154c8553e24dcd682ef60b4e686a396e8d6ae09706f5eb9e.jpg)

![487956ce2a36cfbed54239ed1ef8d4bcabb6025f82606191d64b1596e9ef505c.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/487956ce2a36cfbed54239ed1ef8d4bcabb6025f82606191d64b1596e9ef505c.jpg)

![51dc30be72126cba66e788b8b929f9804b071935d151d84ece609a3a5ae4c064.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/51dc30be72126cba66e788b8b929f9804b071935d151d84ece609a3a5ae4c064.jpg)

![72499946d0c6a41f20d1982bb28c02bf74649fe5de8220cc10dc50db81ed901e.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/72499946d0c6a41f20d1982bb28c02bf74649fe5de8220cc10dc50db81ed901e.jpg)

![7c7ef56abb263f18e49fc1362e21237dfb32cb2d71b6e924da3e62eb7c0dd546.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/7c7ef56abb263f18e49fc1362e21237dfb32cb2d71b6e924da3e62eb7c0dd546.jpg)

![96775bea3b526a479c61c4cae4785cd1db95a7952afb915d6135a2250001a6fa.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/96775bea3b526a479c61c4cae4785cd1db95a7952afb915d6135a2250001a6fa.jpg)

![a263275327c51422806a82209212767eb7596864c0896de47d5a7f523ef197b2.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/a263275327c51422806a82209212767eb7596864c0896de47d5a7f523ef197b2.jpg)

![a733204132f53e571ec50966bfc538c77a7f2c0fb96c1851aec8b4e15caaa9af.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/a733204132f53e571ec50966bfc538c77a7f2c0fb96c1851aec8b4e15caaa9af.jpg)

![c0de6a121b3c1974894a3fcad6598d038561cebca2c149dbaae59471f8e6d035.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/c0de6a121b3c1974894a3fcad6598d038561cebca2c149dbaae59471f8e6d035.jpg)

![c28892775afff2b7bb189bfc79c182f3bab54036e326d43ae00fef3446e748ce.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/c28892775afff2b7bb189bfc79c182f3bab54036e326d43ae00fef3446e748ce.jpg)

![cbebaf9caa87d6195a51a7ae2831aea40facaff512ebf6c1ea0fb82affd38d76.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/cbebaf9caa87d6195a51a7ae2831aea40facaff512ebf6c1ea0fb82affd38d76.jpg)

![d703e1e59bc0e2ea230a48875e8233e605bcf865d0bd577ab6e76a134c295343.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/d703e1e59bc0e2ea230a48875e8233e605bcf865d0bd577ab6e76a134c295343.jpg)

![d780f483f5d2a76cef53394035529c09f0a43179310c06b94c5a9d1acd6009b4.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/d780f483f5d2a76cef53394035529c09f0a43179310c06b94c5a9d1acd6009b4.jpg)

![dc1ade7627e9798788e07acf9d562a3d42e136eeef7e67812b0197bc83e9cc0d.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/dc1ade7627e9798788e07acf9d562a3d42e136eeef7e67812b0197bc83e9cc0d.jpg)

![dca53d3f56f5370b028706ac13b676bb98c8f64046bdc84e642a5648e051f405.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/dca53d3f56f5370b028706ac13b676bb98c8f64046bdc84e642a5648e051f405.jpg)

![eb1dcad4f6d437d29c531801d4bdd71a1666174c7031ff8d0a8acd6559f01a60.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/eb1dcad4f6d437d29c531801d4bdd71a1666174c7031ff8d0a8acd6559f01a60.jpg)

![f5f0981d3383974020705b1ff7d9c8b1b842cd387c10868a7832dc7c75163c9d.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/images/f5f0981d3383974020705b1ff7d9c8b1b842cd387c10868a7832dc7c75163c9d.jpg)

### Tables

![2d8be9e0fb7de20b408eeaf21672a68ab61d322248a7149098051745bec4e241.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/2d8be9e0fb7de20b408eeaf21672a68ab61d322248a7149098051745bec4e241.jpg)

![587e0613c8eb960354dca013b418007605bdc2b4e5e8b19a663a2788c2f5cacd.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/587e0613c8eb960354dca013b418007605bdc2b4e5e8b19a663a2788c2f5cacd.jpg)

![a3a8e008b1e513e90fa77c225cec80bd9674651afc55e98d2298d5644a927c52.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/a3a8e008b1e513e90fa77c225cec80bd9674651afc55e98d2298d5644a927c52.jpg)

![c61db81006a2fc5d3c4f5460f9c78e072f22a85480cfe0b80b24c37f326dd06c.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/c61db81006a2fc5d3c4f5460f9c78e072f22a85480cfe0b80b24c37f326dd06c.jpg)

![cb1d9b9788b42c107cd5edf867c121331804e803ff1e8b150c8ea4d8dc43296b.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/cb1d9b9788b42c107cd5edf867c121331804e803ff1e8b150c8ea4d8dc43296b.jpg)

![d698b6068c55b43a089acdd2dcabf7548a23a12c2770e2821920454c59e21bfc.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/d698b6068c55b43a089acdd2dcabf7548a23a12c2770e2821920454c59e21bfc.jpg)

![e2dceffb47bb1802558365e85bcecd06d66a111cfa726a3c1f9509526ef3a9f3.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/e2dceffb47bb1802558365e85bcecd06d66a111cfa726a3c1f9509526ef3a9f3.jpg)

![f77484b7f9ae35b8a65a453db92d2ae03a5fa245358f79c3cc3ea158e950362c.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/f77484b7f9ae35b8a65a453db92d2ae03a5fa245358f79c3cc3ea158e950362c.jpg)

![feb8d7aaa5d60ef3c0a066c613df8c324498e4c0f277ff9d6847cd8d91818573.jpg](../icml_results/3157_ProtDBench_%20A%20Unified%20Benchmark%20of%20Protein%20Binder%20Design%20and%20Evaluation/tables/feb8d7aaa5d60ef3c0a066c613df8c324498e4c0f277ff9d6847cd8d91818573.jpg)

<a id="paper-3119"></a>

## Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane


### Images

![0b2fcd6a0bc422fdcd6950dbf8c2a96005009c8beaf5c3f9c0130c82d7206933.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/0b2fcd6a0bc422fdcd6950dbf8c2a96005009c8beaf5c3f9c0130c82d7206933.jpg)

![12c42733f259d1ba7f0404d83f541da865c87eef2eb9afef392907e8eca6bf8f.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/12c42733f259d1ba7f0404d83f541da865c87eef2eb9afef392907e8eca6bf8f.jpg)

![2060fe49a7efd4dac94de3715ab67de2247cc7702154c4f819930c7ca2e6d2bd.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/2060fe49a7efd4dac94de3715ab67de2247cc7702154c4f819930c7ca2e6d2bd.jpg)

![212abb32d1d6a0153915e78d48360aabbe9c45b428d5b1fd5cdd7d75e7300159.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/212abb32d1d6a0153915e78d48360aabbe9c45b428d5b1fd5cdd7d75e7300159.jpg)

![2f850a3808e95111bac95a0a9b32e850236a6dadffb5f62bd00a6b900f4d8c48.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/2f850a3808e95111bac95a0a9b32e850236a6dadffb5f62bd00a6b900f4d8c48.jpg)

![4553170a0d9798d38b06a103d1756a44c061f7f9ce0d05c40b77fb9a870685f0.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/4553170a0d9798d38b06a103d1756a44c061f7f9ce0d05c40b77fb9a870685f0.jpg)

![46a4bb85cf7ece3ac1c2c74460c44115af568225e90f732911774087cc96512a.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/46a4bb85cf7ece3ac1c2c74460c44115af568225e90f732911774087cc96512a.jpg)

![4950161899ba1249fe93b3cf4dbbbc299f7041f7226580ba2a9acb8c923eae77.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/4950161899ba1249fe93b3cf4dbbbc299f7041f7226580ba2a9acb8c923eae77.jpg)

![626da989efe9224f28ccb5e66a257ab443a196c95d43c2ab8d331cd771f05132.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/626da989efe9224f28ccb5e66a257ab443a196c95d43c2ab8d331cd771f05132.jpg)

![6fea8376e0d8f23dacb9336a6199095a93654e00786075d620143aa8afc2601e.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/6fea8376e0d8f23dacb9336a6199095a93654e00786075d620143aa8afc2601e.jpg)

![7fd6ddf89e502b8a6cf394c924ca7468ecca77a5ba2a92feaf3aee573c13748d.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/7fd6ddf89e502b8a6cf394c924ca7468ecca77a5ba2a92feaf3aee573c13748d.jpg)

![808849ef8955e1cb0547dc5944def318bf06809356ea895bab61c8cace2f75ab.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/808849ef8955e1cb0547dc5944def318bf06809356ea895bab61c8cace2f75ab.jpg)

![996496fbf2167a4dcb3ec7ffb7cf44ba7afd559c646f970179a1ac8c331d169f.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/996496fbf2167a4dcb3ec7ffb7cf44ba7afd559c646f970179a1ac8c331d169f.jpg)

![a3f1a7cbb3f97f0519cc29e8025f5b6d789784d1b1d13eff39e558ff5515e511.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/a3f1a7cbb3f97f0519cc29e8025f5b6d789784d1b1d13eff39e558ff5515e511.jpg)

![a4915bb18ef94bd020c37645eb680467eb1154480c531b9637d17f6d9e11146e.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/a4915bb18ef94bd020c37645eb680467eb1154480c531b9637d17f6d9e11146e.jpg)

![a69d781187421af19eb5aa44df06203c59a8d83b5b376c6930cae6e0d2cefdfe.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/a69d781187421af19eb5aa44df06203c59a8d83b5b376c6930cae6e0d2cefdfe.jpg)

![c9ea06b7a20dd75b5829a70eae60aea79451af62154346b04b7afd49411d916c.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/c9ea06b7a20dd75b5829a70eae60aea79451af62154346b04b7afd49411d916c.jpg)

![cbe8ff51caf87582931d0258bc50604b2daa7d2fc17f31c8ae544da31dd2c27b.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/cbe8ff51caf87582931d0258bc50604b2daa7d2fc17f31c8ae544da31dd2c27b.jpg)

![fdde1be49b303a305e50c3f3742b7fc1ded03028a498987f10db2d6941a607b3.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/images/fdde1be49b303a305e50c3f3742b7fc1ded03028a498987f10db2d6941a607b3.jpg)

### Tables

![095b6176ac16ebb607dc63d9058c60b5ed4c831a7d3ce21a1549b6089f73d615.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/095b6176ac16ebb607dc63d9058c60b5ed4c831a7d3ce21a1549b6089f73d615.jpg)

![0dbc0e73f2d9f4405cfd3582b5cbd64961cf9b8c9c1eb5c7bdecf0ad9ec0051f.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/0dbc0e73f2d9f4405cfd3582b5cbd64961cf9b8c9c1eb5c7bdecf0ad9ec0051f.jpg)

![221918eb0380588f6d59c117e0c63d2e32490be8c4aa940a84d57c5c850cef90.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/221918eb0380588f6d59c117e0c63d2e32490be8c4aa940a84d57c5c850cef90.jpg)

![279c70d62e3a9c6952f9f1e5a9bfc3baabf3ed90824e738bd377f356ea283c4f.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/279c70d62e3a9c6952f9f1e5a9bfc3baabf3ed90824e738bd377f356ea283c4f.jpg)

![2ed5aeb02afbc3d34899f02354612b75cfce7e0cba3ec7b1cfb057c81086d5ba.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/2ed5aeb02afbc3d34899f02354612b75cfce7e0cba3ec7b1cfb057c81086d5ba.jpg)

![af5101c47a1f3574208beab48255a7d516b404bc120023a2419141229fc3ccb1.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/af5101c47a1f3574208beab48255a7d516b404bc120023a2419141229fc3ccb1.jpg)

![c2e32afe5ba38a7856f8e5c391c01ebb608d097053a71f6c7900fcedf57eb9b5.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/c2e32afe5ba38a7856f8e5c391c01ebb608d097053a71f6c7900fcedf57eb9b5.jpg)

![ca54d09bd4a06084534114fb68a9fe1b2a2af2f2f7652870c405a9cf6ea086da.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/ca54d09bd4a06084534114fb68a9fe1b2a2af2f2f7652870c405a9cf6ea086da.jpg)

![eef5d459239408df790362734cba21429e6da89b9fb60873d4910bdbf2335650.jpg](../icml_results/3158_Spiral%20RoPE_%20Rotate%20Your%20Rotary%20Positional%20Embeddings%20in%20the%202D%20Plane/tables/eef5d459239408df790362734cba21429e6da89b9fb60873d4910bdbf2335650.jpg)

<a id="paper-3120"></a>

## Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy


### Images

![0a651cca34f937e3d374b26c421be4560e2c0df795370551b2a338b5197e42dd.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/0a651cca34f937e3d374b26c421be4560e2c0df795370551b2a338b5197e42dd.jpg)

![23f78b03a29da9bafbe02250c9c1b8c1f31323526aea7c896302b45ec8e6eea9.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/23f78b03a29da9bafbe02250c9c1b8c1f31323526aea7c896302b45ec8e6eea9.jpg)

![5d61b161f46f8c6a412b0c5661399a583cb810162f0bdf8c57f0aa29a9031d25.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/5d61b161f46f8c6a412b0c5661399a583cb810162f0bdf8c57f0aa29a9031d25.jpg)

![5e775f650b573730e51e36351c8ba1cf11f40883c6a6b8a844b613a39c83d229.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/5e775f650b573730e51e36351c8ba1cf11f40883c6a6b8a844b613a39c83d229.jpg)

![a3a7ae9387784b15dc4a7e63bb883af1a0ea9741c7bfa38006b3e4f07331739f.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/a3a7ae9387784b15dc4a7e63bb883af1a0ea9741c7bfa38006b3e4f07331739f.jpg)

![b4860cf01b2e33a9b467f60b3ea6099658deb3988297f1c0548cd4831bd8b7f6.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/images/b4860cf01b2e33a9b467f60b3ea6099658deb3988297f1c0548cd4831bd8b7f6.jpg)

### Tables

![1be72a7f41b89dc89ffbe5df8f98f5b56255b51bc4821788af0eae26f69668f7.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/1be72a7f41b89dc89ffbe5df8f98f5b56255b51bc4821788af0eae26f69668f7.jpg)

![4c6a6211bbe771727a8afee71e78bb600ee5cf2b60d0504a5f81f131ebe6cd56.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/4c6a6211bbe771727a8afee71e78bb600ee5cf2b60d0504a5f81f131ebe6cd56.jpg)

![54faabc9b14bfbfb778627a3e0df0a9ade1f4493cacd1edbecd6aaf2b7c0fd5e.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/54faabc9b14bfbfb778627a3e0df0a9ade1f4493cacd1edbecd6aaf2b7c0fd5e.jpg)

![7bf06a80533090e303d0252c757057731a6d8ee9f48e4af0f1e0cdcb179344c8.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/7bf06a80533090e303d0252c757057731a6d8ee9f48e4af0f1e0cdcb179344c8.jpg)

![8245572a1d66e36c2465cc61fee14ae01dc3093b0649ff7ad8950b6396f520a1.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/8245572a1d66e36c2465cc61fee14ae01dc3093b0649ff7ad8950b6396f520a1.jpg)

![8a235a0c8a091c676c9cad167cd3eba686d9e896f58a3a83c4fcf2375a9de98f.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/8a235a0c8a091c676c9cad167cd3eba686d9e896f58a3a83c4fcf2375a9de98f.jpg)

![bf5be329e24d4cd393911cfe2168bc5d5a7d11ca176a78185d7ff61fa5265cfd.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/bf5be329e24d4cd393911cfe2168bc5d5a7d11ca176a78185d7ff61fa5265cfd.jpg)

![d1e4fe41f81893b098afd3dc320dc20669b66654f2b1811a9e2c8ff707d339c0.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/d1e4fe41f81893b098afd3dc320dc20669b66654f2b1811a9e2c8ff707d339c0.jpg)

![e8b3d04693e503d94ce1c5dd806f69e1048a80747b3acdcda1ca6da05f343bcd.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/e8b3d04693e503d94ce1c5dd806f69e1048a80747b3acdcda1ca6da05f343bcd.jpg)

![ecce1d1e20d81e0d6afb9927997446a9e0b8af1e6692b7071255fb2a4e966be6.jpg](../icml_results/3159_Mind%20the%20Gap_%20Mixtures%20of%20Gaussians%20in%20Approximate%20Differential%20Privacy/tables/ecce1d1e20d81e0d6afb9927997446a9e0b8af1e6692b7071255fb2a4e966be6.jpg)

<a id="paper-3121"></a>

## FairMerging: Rethinking Model Merging through the Lens of Fairness


### Images

![2214c436b895a84ebc2e960251ab7dcdf07bb1a429359354e8414bebec8c6565.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/2214c436b895a84ebc2e960251ab7dcdf07bb1a429359354e8414bebec8c6565.jpg)

![40e1d0ff9e63b3e04e7925c79a8c2ff19d5d7b25374602c80c833ffd564634a0.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/40e1d0ff9e63b3e04e7925c79a8c2ff19d5d7b25374602c80c833ffd564634a0.jpg)

![41f3a368940f0f248bb3146cc33995a98eeda52d481e9f85ade07e84c257968b.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/41f3a368940f0f248bb3146cc33995a98eeda52d481e9f85ade07e84c257968b.jpg)

![43d9161a30cd23b3f37af94c087fc3c25f2dfb2d5732ec0f69a445f8a69bc517.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/43d9161a30cd23b3f37af94c087fc3c25f2dfb2d5732ec0f69a445f8a69bc517.jpg)

![5167cad70151ff96e8981d8311a32a68ab53d06102f3c36468a21373521f4926.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/5167cad70151ff96e8981d8311a32a68ab53d06102f3c36468a21373521f4926.jpg)

![6206c5c696303e76b51ff3dddf8f7f2aaaaf73a2d3a1a5734fb4083650e5a0d8.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/6206c5c696303e76b51ff3dddf8f7f2aaaaf73a2d3a1a5734fb4083650e5a0d8.jpg)

![6904734d810615102105ff35f7f4b4e318cf93568243024ba9382b3754c18acb.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/6904734d810615102105ff35f7f4b4e318cf93568243024ba9382b3754c18acb.jpg)

![72ff79aa3e1dfbfe4b9a9a4626f2cb0336818bcb6e64ddd99c0d5f823da88db3.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/72ff79aa3e1dfbfe4b9a9a4626f2cb0336818bcb6e64ddd99c0d5f823da88db3.jpg)

![8d9c9548f3b6983cea7161586b2f9ecafab6536b664437dcaa9dd02d2eb1691f.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/8d9c9548f3b6983cea7161586b2f9ecafab6536b664437dcaa9dd02d2eb1691f.jpg)

![976f6f15f20d7e972e9e0a39ff97cb159cee8dda191a02b75098726a60f24a7d.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/976f6f15f20d7e972e9e0a39ff97cb159cee8dda191a02b75098726a60f24a7d.jpg)

![b8a49963cdbe3fa7d6b96a17cc0cbac7e8ac62567a13060deb6a1d309f4fadd8.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/b8a49963cdbe3fa7d6b96a17cc0cbac7e8ac62567a13060deb6a1d309f4fadd8.jpg)

![cb675f38a1c85a0ad1cbd41ce9e1d4a39a0280df09921293fcab9cf5e644f164.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/cb675f38a1c85a0ad1cbd41ce9e1d4a39a0280df09921293fcab9cf5e644f164.jpg)

![d1878768e4ab3cab0fdd79b48a1e0ea7b4cc80a0157b71bbe5b15079b2132c15.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/d1878768e4ab3cab0fdd79b48a1e0ea7b4cc80a0157b71bbe5b15079b2132c15.jpg)

![db0286cbcfeaf78e611f43b594b9fe64a1c6292c59a839b0525586372ce9d71a.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/images/db0286cbcfeaf78e611f43b594b9fe64a1c6292c59a839b0525586372ce9d71a.jpg)

### Tables

![0105c5ee530d060ee509ce50cdcbd53b09d088750861e40fc7406572cebf6cdf.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/0105c5ee530d060ee509ce50cdcbd53b09d088750861e40fc7406572cebf6cdf.jpg)

![0618bb776f5e77825c3de4678486c375fa07adce074ee9ae26e722ddce23a7c7.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/0618bb776f5e77825c3de4678486c375fa07adce074ee9ae26e722ddce23a7c7.jpg)

![270d7c70caa7a1b571e8081dd61e39859f08d51ac402e9dae35fdf6eb21dfcfc.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/270d7c70caa7a1b571e8081dd61e39859f08d51ac402e9dae35fdf6eb21dfcfc.jpg)

![44ee45f2302c7edf0538a13feb25ff1f8e8e7b22e1c5b0a5fb33692184b3da57.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/44ee45f2302c7edf0538a13feb25ff1f8e8e7b22e1c5b0a5fb33692184b3da57.jpg)

![4f0ef9bdf37d42815876f5c0eafc8ecb235ea0340d51b224cee65339e22b235e.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/4f0ef9bdf37d42815876f5c0eafc8ecb235ea0340d51b224cee65339e22b235e.jpg)

![5d60555dfae29a60f163cb8a30760eec275eb2a3ba67206bb04eeaf773817bfb.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/5d60555dfae29a60f163cb8a30760eec275eb2a3ba67206bb04eeaf773817bfb.jpg)

![69f225d4d2a3f13d1be4ef011b1cdaf46e533a89c302b460100ebeb07491b071.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/69f225d4d2a3f13d1be4ef011b1cdaf46e533a89c302b460100ebeb07491b071.jpg)

![8c183c928f056b1fcb4e208f74eb1f07ef1d8b165e5ce1fad0ab9aefb03ed259.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/8c183c928f056b1fcb4e208f74eb1f07ef1d8b165e5ce1fad0ab9aefb03ed259.jpg)

![8eca5d5b3545a7a30ea505ec50d96bbb705d98c9504f419bb654ade3a01d92b7.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/8eca5d5b3545a7a30ea505ec50d96bbb705d98c9504f419bb654ade3a01d92b7.jpg)

![925ae7fe14d43fd5fd53e7e48c0ce9430944f70205228034473f0f342d1693e8.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/925ae7fe14d43fd5fd53e7e48c0ce9430944f70205228034473f0f342d1693e8.jpg)

![b09014044616a3a017f0a441ba90b2ef8ac243ad2c2a758447a4ca6f178d4fac.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/b09014044616a3a017f0a441ba90b2ef8ac243ad2c2a758447a4ca6f178d4fac.jpg)

![b99363a84da91a3d9df66c4f7d4d9ea3a25f34e752ac4feceb3fb30b36f2a449.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/b99363a84da91a3d9df66c4f7d4d9ea3a25f34e752ac4feceb3fb30b36f2a449.jpg)

![bda759b18992dd26f60946d7a058e5f30e38953e8cffc2b6bb9f5db9e78a15da.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/bda759b18992dd26f60946d7a058e5f30e38953e8cffc2b6bb9f5db9e78a15da.jpg)

![ce7dabf0040e4309d0652abc5e5f0286a8607123c59c5663f9fc026a2d47defa.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/ce7dabf0040e4309d0652abc5e5f0286a8607123c59c5663f9fc026a2d47defa.jpg)

![e90d36c41d9e3dacbab48da8ce40fcc56d180dd7040b71bb7241fa847fe37e50.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/e90d36c41d9e3dacbab48da8ce40fcc56d180dd7040b71bb7241fa847fe37e50.jpg)

![f5d98280407d699abef9aad8f80fd651e99ca7868849d0cb6c72faf1f4906f34.jpg](../icml_results/3160_FairMerging_%20Rethinking%20Model%20Merging%20through%20the%20Lens%20of%20Fairness/tables/f5d98280407d699abef9aad8f80fd651e99ca7868849d0cb6c72faf1f4906f34.jpg)

<a id="paper-3122"></a>

## PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting


### Images

![1860a6583c1f524eb7d7423835bdfc7fcf52274f020ba1ac07e9ea27e8fcc081.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/1860a6583c1f524eb7d7423835bdfc7fcf52274f020ba1ac07e9ea27e8fcc081.jpg)

![3626de6f09b5cd07a8357ea70bff39faabcf17b7e89e7681b2d5aef0e274e9ec.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/3626de6f09b5cd07a8357ea70bff39faabcf17b7e89e7681b2d5aef0e274e9ec.jpg)

![45aa3849b10c96ccbe0ca539a8582114dabaf169be82b5e42ed74e5c77ccc994.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/45aa3849b10c96ccbe0ca539a8582114dabaf169be82b5e42ed74e5c77ccc994.jpg)

![49f41a5df1c35ecf6dff4a50778360ea042b4f5492a621ab4376cd8ae29c749f.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/49f41a5df1c35ecf6dff4a50778360ea042b4f5492a621ab4376cd8ae29c749f.jpg)

![6cd9f56b50ad8afe29162a0b4eb68128be174348029ff4657d514bba45bf666c.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/6cd9f56b50ad8afe29162a0b4eb68128be174348029ff4657d514bba45bf666c.jpg)

![83a53866937971ce61b6c53768d46ee4d97c04ff3703f0c218638ab814173c16.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/83a53866937971ce61b6c53768d46ee4d97c04ff3703f0c218638ab814173c16.jpg)

![8bbc466b4081f5312359850beaf909650b8d25bd50c93478c21362da3149b485.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/8bbc466b4081f5312359850beaf909650b8d25bd50c93478c21362da3149b485.jpg)

![af59cf5fd2b1df93be1fa507edfedc5b69c89dcff239a006b35144ad6b831edc.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/af59cf5fd2b1df93be1fa507edfedc5b69c89dcff239a006b35144ad6b831edc.jpg)

![b2eb40919e2e76c2dc05ce796328a02549520cfcceea1d443cc3145d0a70cbe2.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/b2eb40919e2e76c2dc05ce796328a02549520cfcceea1d443cc3145d0a70cbe2.jpg)

![c8e3a4ca70a1349531bb32d288f798c594a1ff0df4ca7759a40330cd7c1c362d.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/c8e3a4ca70a1349531bb32d288f798c594a1ff0df4ca7759a40330cd7c1c362d.jpg)

![e014a5c1e97493f359978b4502a5999c0cb3957c3a8e945491bec77534cf1147.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/e014a5c1e97493f359978b4502a5999c0cb3957c3a8e945491bec77534cf1147.jpg)

![ec1d016b3b5dbbdc18215ce9b35f877a45d2bfc2ddf15068b1f64c5cf9a56df0.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/ec1d016b3b5dbbdc18215ce9b35f877a45d2bfc2ddf15068b1f64c5cf9a56df0.jpg)

![f88c7bd3ef75ac7c143f64123e17e414cfbef1e84870aaf7a291a23fd94a4939.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/images/f88c7bd3ef75ac7c143f64123e17e414cfbef1e84870aaf7a291a23fd94a4939.jpg)

### Tables

![0809cce63c72063d1910d4e27ccfb8b630037b784e8587ea46408999508f3533.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/0809cce63c72063d1910d4e27ccfb8b630037b784e8587ea46408999508f3533.jpg)

![0d1c95b47c12d1e2ec8357a4483e6ec48f017db5f6b327813240caab96274754.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/0d1c95b47c12d1e2ec8357a4483e6ec48f017db5f6b327813240caab96274754.jpg)

![1f6eb3f1051c0087a89b62e4c22044dd3eaf33ad3eedf817917ffbdb5c422414.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/1f6eb3f1051c0087a89b62e4c22044dd3eaf33ad3eedf817917ffbdb5c422414.jpg)

![4200fe0156a2ebdf5753a773047112302c78d2b49a6f901a4fa271f5396ba862.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/4200fe0156a2ebdf5753a773047112302c78d2b49a6f901a4fa271f5396ba862.jpg)

![422b7bbc8316befbef3dc7877fbc1b96bca4b4c0dedabf61cfae25401e5a52fe.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/422b7bbc8316befbef3dc7877fbc1b96bca4b4c0dedabf61cfae25401e5a52fe.jpg)

![44d9cff56b4187e3d0423fc08ecb2b6ab16554dad12193d3b37208705c0269fe.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/44d9cff56b4187e3d0423fc08ecb2b6ab16554dad12193d3b37208705c0269fe.jpg)

![55d93ff971b385fd7bedc5ebdce320fa94846099048f07f83be178b4a2d6f294.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/55d93ff971b385fd7bedc5ebdce320fa94846099048f07f83be178b4a2d6f294.jpg)

![7aaf5855349a99a3fac3ed45ebd7fde2eda70762f334a2b58edc963b798ee1f8.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/7aaf5855349a99a3fac3ed45ebd7fde2eda70762f334a2b58edc963b798ee1f8.jpg)

![7c17f628ccc9271e85e843319abb99f0e3aba467a8e3446ce1866c98b64ca799.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/7c17f628ccc9271e85e843319abb99f0e3aba467a8e3446ce1866c98b64ca799.jpg)

![84b4d9614d190091282bb5799dd15ed55b66338bf78e5b3a626f380f7079d0fb.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/84b4d9614d190091282bb5799dd15ed55b66338bf78e5b3a626f380f7079d0fb.jpg)

![90278f7876405cebd33bfaf890d1f7d2be351fbcdad3c4706dfabbbecd93e05d.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/90278f7876405cebd33bfaf890d1f7d2be351fbcdad3c4706dfabbbecd93e05d.jpg)

![bdf8e6a49e8f237c0fcde788f8f73748fcd04c6f5fc7d07e832074bb4703da0c.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/bdf8e6a49e8f237c0fcde788f8f73748fcd04c6f5fc7d07e832074bb4703da0c.jpg)

![f081b8506fba4de6a3eb7506761cb5a1180d810c7d85afaf8157031d6cc8e288.jpg](../icml_results/3161_PULSE_%20Generative%20Phase%20Evolution%20for%20Non-Stationary%20Time%20Series%20Forecasting/tables/f081b8506fba4de6a3eb7506761cb5a1180d810c7d85afaf8157031d6cc8e288.jpg)

<a id="paper-3123"></a>

## LIF Recurrent Memory Enables Long-Horizon Spiking Computation


### Images

![14e56f7dfae2c5511b01e945ba0bf7f7e6aa00b8e067eb909ef8194bc2342b00.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/14e56f7dfae2c5511b01e945ba0bf7f7e6aa00b8e067eb909ef8194bc2342b00.jpg)

![16517efb220fd3e9b89d08d1d3c55341ca1e89dc04200581c663f7c80500ebb4.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/16517efb220fd3e9b89d08d1d3c55341ca1e89dc04200581c663f7c80500ebb4.jpg)

![7ab5c1c103ee3034fc12fb298b8d48f5b068b94789dd27820c9904d434589dfa.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/7ab5c1c103ee3034fc12fb298b8d48f5b068b94789dd27820c9904d434589dfa.jpg)

![9015411aabf34a11776bdcb0b822a2b2768d829a2a9f553e27df00b999145360.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/9015411aabf34a11776bdcb0b822a2b2768d829a2a9f553e27df00b999145360.jpg)

![d4f850e6f74366e9025e22a87dfac5e3b1752a24b456b3b39ab9778360a63833.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/d4f850e6f74366e9025e22a87dfac5e3b1752a24b456b3b39ab9778360a63833.jpg)

![fdadcbaed33397997c39e4ee44abde1c38beaa03db616095965ee6ab0b5102a0.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/images/fdadcbaed33397997c39e4ee44abde1c38beaa03db616095965ee6ab0b5102a0.jpg)

### Tables

![24522ef01b3bfbd877412fd482d13268a17c46268fddf053c214dcd42ef5bb2c.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/24522ef01b3bfbd877412fd482d13268a17c46268fddf053c214dcd42ef5bb2c.jpg)

![2bfd9c4b6a4da4a5affc228cd90e03391874829631fa89f3980a10814bee84a7.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/2bfd9c4b6a4da4a5affc228cd90e03391874829631fa89f3980a10814bee84a7.jpg)

![30de34bafdbd3264017cd69a8a7eaa97de7d2980625b3c6099a500935fc573a6.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/30de34bafdbd3264017cd69a8a7eaa97de7d2980625b3c6099a500935fc573a6.jpg)

![33298cc455c853fe59c3ff8048745a314153fc61678635706b31e611da1f7eb1.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/33298cc455c853fe59c3ff8048745a314153fc61678635706b31e611da1f7eb1.jpg)

![3366a2ab1cd07659a35ccbfbddab79deee93419990916a931e4a74c7f201a5de.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/3366a2ab1cd07659a35ccbfbddab79deee93419990916a931e4a74c7f201a5de.jpg)

![3ce427e61e4fbb58c625c2717f6bb1e128c129364bd48a8ba6ed220a63f2e1d3.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/3ce427e61e4fbb58c625c2717f6bb1e128c129364bd48a8ba6ed220a63f2e1d3.jpg)

![550440f95034fc9ba9cb0c9de33642044f45437c7a047871664d3bca1f50d27b.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/550440f95034fc9ba9cb0c9de33642044f45437c7a047871664d3bca1f50d27b.jpg)

![6d771690b9b17ac828a78cf373a2f9138a61279c55926e776f2cc99f7775f96d.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/6d771690b9b17ac828a78cf373a2f9138a61279c55926e776f2cc99f7775f96d.jpg)

![6ddae704ec408640a38f8498fa7053276467ad4d730eaffe423b62aba42f7e46.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/6ddae704ec408640a38f8498fa7053276467ad4d730eaffe423b62aba42f7e46.jpg)

![87017b6740cb664ddfcb987ce584f70f184c67544f10329c1fba60c762b2c2fc.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/87017b6740cb664ddfcb987ce584f70f184c67544f10329c1fba60c762b2c2fc.jpg)

![97b981c319aba68e3913ecf8e937db80c7e72ed1e97217dc5f07737bcd8fef21.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/97b981c319aba68e3913ecf8e937db80c7e72ed1e97217dc5f07737bcd8fef21.jpg)

![9e696dbe03944cac354075f376305bf2a30e1680cbc0c45f3899c2a1a6c59370.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/9e696dbe03944cac354075f376305bf2a30e1680cbc0c45f3899c2a1a6c59370.jpg)

![a36943e25c4dcecca916c17ef2a6ab132447c31dc38421c6b660cec498f3325f.jpg](../icml_results/3162_LIF%20Recurrent%20Memory%20Enables%20Long-Horizon%20Spiking%20Computation/tables/a36943e25c4dcecca916c17ef2a6ab132447c31dc38421c6b660cec498f3325f.jpg)

<a id="paper-3124"></a>

## ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning


### Images

![11c11831c7f2cf55b79a780a53c03cdfc214867991c526a9d5f7b9d0e7662f2e.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/11c11831c7f2cf55b79a780a53c03cdfc214867991c526a9d5f7b9d0e7662f2e.jpg)

![6c7c979e7c792d95bc9ddb7943adcdba526033a564f23dc618f1c28d76964642.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/6c7c979e7c792d95bc9ddb7943adcdba526033a564f23dc618f1c28d76964642.jpg)

![7f122677186198c47f28fef9b4e80c2262358a5fcf5c01559a6708479685ab1e.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/7f122677186198c47f28fef9b4e80c2262358a5fcf5c01559a6708479685ab1e.jpg)

![a1a0e1ccb7bc9e464e1ef402a35db9c6b173bb981afd9bca274add99549cc8b1.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/a1a0e1ccb7bc9e464e1ef402a35db9c6b173bb981afd9bca274add99549cc8b1.jpg)

![b6718ca9887e4b91724b21dc70640ecdc93106579c0e5e13ef70668636d487e9.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/b6718ca9887e4b91724b21dc70640ecdc93106579c0e5e13ef70668636d487e9.jpg)

![ba8895975a593c724b1971d7d0051a05b0dc59bce59c918bce72f524ff72b0d8.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/images/ba8895975a593c724b1971d7d0051a05b0dc59bce59c918bce72f524ff72b0d8.jpg)

### Tables

![3321c820889d93865f0ab4c05004c39105dfb41fdacd1a3b88655a0b5f4bd34a.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/tables/3321c820889d93865f0ab4c05004c39105dfb41fdacd1a3b88655a0b5f4bd34a.jpg)

![3797d7848973b424c17ac8c02692fe3b1f464594698cae6c7ea03a9f9ba14302.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/tables/3797d7848973b424c17ac8c02692fe3b1f464594698cae6c7ea03a9f9ba14302.jpg)

![6bf67b0b6702907c54f94ea5625bbf4aa3d6c2c5538f2396140b57aee070a5cf.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/tables/6bf67b0b6702907c54f94ea5625bbf4aa3d6c2c5538f2396140b57aee070a5cf.jpg)

![d79a2ebe72d6996e99ab3c3b247195a3e51a107f6bf72109d0f966428eb126bb.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/tables/d79a2ebe72d6996e99ab3c3b247195a3e51a107f6bf72109d0f966428eb126bb.jpg)

![f7fa91e6861eeccf1216cda310c60a7d8303d6d8086ba28872a191ead870b28f.jpg](../icml_results/3163_ThoughtFold_%20Folding%20Reasoning%20Chains%20via%20Introspective%20Preference%20Learning/tables/f7fa91e6861eeccf1216cda310c60a7d8303d6d8086ba28872a191ead870b28f.jpg)

<a id="paper-3125"></a>

## Hyperbolic Multimodal Continual Learning


### Images

![0f2215030c409dde4e7b89061afe1fdf64474342a1038dfc5b73a6e96e5d051e.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/0f2215030c409dde4e7b89061afe1fdf64474342a1038dfc5b73a6e96e5d051e.jpg)

![1e01cc0e9f82071fed3fdffe685f4db10a42993a65a274cc93e71e6d52481bb9.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/1e01cc0e9f82071fed3fdffe685f4db10a42993a65a274cc93e71e6d52481bb9.jpg)

![20ca6037820ed5a88ed74954fad0645698469647c4ca94fa1b73435ab5e604ab.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/20ca6037820ed5a88ed74954fad0645698469647c4ca94fa1b73435ab5e604ab.jpg)

![2867627f8b28f9efdbe1e7c85dc2ebad8a9eeb5344a671238368d5365a842a4e.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/2867627f8b28f9efdbe1e7c85dc2ebad8a9eeb5344a671238368d5365a842a4e.jpg)

![34256364b71a501b0f4a9a5f6be63ef3766f97bb831f449e17f791ac9a337ea2.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/34256364b71a501b0f4a9a5f6be63ef3766f97bb831f449e17f791ac9a337ea2.jpg)

![412e0ee42f80a27d107ea283086ec7c0fadcd9c46a3ad1a35fc67335504641fb.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/412e0ee42f80a27d107ea283086ec7c0fadcd9c46a3ad1a35fc67335504641fb.jpg)

![48dc3369d1466a503ffda410ed060fbdb4f69bf392c8f1c0438af240cc47cea1.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/48dc3369d1466a503ffda410ed060fbdb4f69bf392c8f1c0438af240cc47cea1.jpg)

![5213958fd426129034e3af75b7d6bf76853e7b0a02aba2d5a026cb4e99fa9eec.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/5213958fd426129034e3af75b7d6bf76853e7b0a02aba2d5a026cb4e99fa9eec.jpg)

![5b89c69dbf2570784cdfc8fc99bdb447a734a52c5982d472c6b75374732fc721.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/5b89c69dbf2570784cdfc8fc99bdb447a734a52c5982d472c6b75374732fc721.jpg)

![62e1b29a3fc644c81791f60a8a4d87040a82111e07288baf034aec74d9e4ba05.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/62e1b29a3fc644c81791f60a8a4d87040a82111e07288baf034aec74d9e4ba05.jpg)

![7b758293f90677fd99045849b8cddf31f52e45240fb7aa17da106c0b5442969a.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/7b758293f90677fd99045849b8cddf31f52e45240fb7aa17da106c0b5442969a.jpg)

![975673bea929e93385d07a324d5d9e87e72bb218b087645e557bc58b24c26459.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/975673bea929e93385d07a324d5d9e87e72bb218b087645e557bc58b24c26459.jpg)

![9b22d72429ca3b68f0406a68f8157927b37ea8b4b03b59ba8d505008aa5713a9.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/9b22d72429ca3b68f0406a68f8157927b37ea8b4b03b59ba8d505008aa5713a9.jpg)

![a747e2ad88a563a0a4760cfccaa74a900c29d3acddefb0c2527a06cca2021c45.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/a747e2ad88a563a0a4760cfccaa74a900c29d3acddefb0c2527a06cca2021c45.jpg)

![b680b6ad68c625e3f68e51e00b057bf39cccca3cce45dc10eff5fc8a36ec0269.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/b680b6ad68c625e3f68e51e00b057bf39cccca3cce45dc10eff5fc8a36ec0269.jpg)

![ba9c3bd98d1279ee10ff9e5e5913c094a2d06057de8581df3cef1ad2698846d3.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/ba9c3bd98d1279ee10ff9e5e5913c094a2d06057de8581df3cef1ad2698846d3.jpg)

![ca62ebb7e1420f1e95235f854317256a6f4aff146c768a522e172ae00692f157.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/ca62ebb7e1420f1e95235f854317256a6f4aff146c768a522e172ae00692f157.jpg)

![cb90155deab73b110574ca2c01fc0fb58547627bcee8aec4f2e0c62d5be79cdf.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/cb90155deab73b110574ca2c01fc0fb58547627bcee8aec4f2e0c62d5be79cdf.jpg)

![e858c0a8ccc8da9ebcfb6860c464d2c4811c77ae7564e1acb00569827da89d97.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/e858c0a8ccc8da9ebcfb6860c464d2c4811c77ae7564e1acb00569827da89d97.jpg)

![eefeef4b92b9db5a907d14d5a10cbf87ece11a865ee3752d8352375673aad70c.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/eefeef4b92b9db5a907d14d5a10cbf87ece11a865ee3752d8352375673aad70c.jpg)

![f0033515fb229967744ef0bc673739c0739303ed24726323465482c7d629bcd8.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/images/f0033515fb229967744ef0bc673739c0739303ed24726323465482c7d629bcd8.jpg)

### Tables

![195b78e89ce67e0e1e24cb7381ba550e6298c74ba40d9d92bef03d72d284ea26.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/195b78e89ce67e0e1e24cb7381ba550e6298c74ba40d9d92bef03d72d284ea26.jpg)

![56b4e888f3cb8be9db1fd6d5bc78b4d78d96d295f135597e9c515e1c58845390.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/56b4e888f3cb8be9db1fd6d5bc78b4d78d96d295f135597e9c515e1c58845390.jpg)

![5db003e6b4bdf6c0e395a63a318b71077b41bdd02c01b835e33b708b945dcc77.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/5db003e6b4bdf6c0e395a63a318b71077b41bdd02c01b835e33b708b945dcc77.jpg)

![6dca4c1b48225c2d7c3391e02b82d9a2e1edc12d9333acc210739bea6e498e1a.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/6dca4c1b48225c2d7c3391e02b82d9a2e1edc12d9333acc210739bea6e498e1a.jpg)

![7b56a5fede2b4af46d0c3a5b239234eff4d23bbbf6661912650867a18b707aa3.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/7b56a5fede2b4af46d0c3a5b239234eff4d23bbbf6661912650867a18b707aa3.jpg)

![c122a512a5ced3f54bd652f806db6d2365f5692bb2fa81fd26dfab3cae6003e3.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/c122a512a5ced3f54bd652f806db6d2365f5692bb2fa81fd26dfab3cae6003e3.jpg)

![c1fe583b28f7ca5a45026a0f62692ac49ac9f63eceacb94118d67b3d749ab101.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/c1fe583b28f7ca5a45026a0f62692ac49ac9f63eceacb94118d67b3d749ab101.jpg)

![c52b5bcacffa4746222c8dfd10f9d72f1907e85405c9e9cadb5e227f7515fd3d.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/c52b5bcacffa4746222c8dfd10f9d72f1907e85405c9e9cadb5e227f7515fd3d.jpg)

![cd9baeec5a0b6b7a6daeff6a17083d073d1351c0b67e0416ec5f9a27e0395647.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/cd9baeec5a0b6b7a6daeff6a17083d073d1351c0b67e0416ec5f9a27e0395647.jpg)

![d4a83083977a7e60df612c3ec3ec02a6ff1348e4a5822e51d797459b509781bc.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/d4a83083977a7e60df612c3ec3ec02a6ff1348e4a5822e51d797459b509781bc.jpg)

![d4b92366f731104c7bbaba27cf4c329883fb120875bbb7375a48b8ee0fad5d8b.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/d4b92366f731104c7bbaba27cf4c329883fb120875bbb7375a48b8ee0fad5d8b.jpg)

![dff1526faedd6c5f262183575aecb57c2719b943c5d58ae629305ded1dbeb246.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/dff1526faedd6c5f262183575aecb57c2719b943c5d58ae629305ded1dbeb246.jpg)

![ee4836fbed277d6ff10e8107e67a6afb5d5d28b0f0bb98c32a8095e4fc9e66f8.jpg](../icml_results/3164_Hyperbolic%20Multimodal%20Continual%20Learning/tables/ee4836fbed277d6ff10e8107e67a6afb5d5d28b0f0bb98c32a8095e4fc9e66f8.jpg)

<a id="paper-3126"></a>

## XPERT: Expert Knowledge Transfer for Effective Training of Language Models


### Images

![2fc413373f481cc6b51763129d76aa65677dd03c3e2f789d4f9964b264d18ee2.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/2fc413373f481cc6b51763129d76aa65677dd03c3e2f789d4f9964b264d18ee2.jpg)

![73f07c218bee40aa177f20ab22929683ee5759373e44bfd095c9808614fa6dee.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/73f07c218bee40aa177f20ab22929683ee5759373e44bfd095c9808614fa6dee.jpg)

![8072b7c2b4afe55bb587aaaddbecc2aff5cdb239b202091f49838ce4a532dd25.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/8072b7c2b4afe55bb587aaaddbecc2aff5cdb239b202091f49838ce4a532dd25.jpg)

![8160af16695930b37ba92b4c00257e03605dd239accf279e1f8d828f1279432b.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/8160af16695930b37ba92b4c00257e03605dd239accf279e1f8d828f1279432b.jpg)

![8908e189623f8851fa5c7ed858fff7486da2a296507515935ddb6f3faa42bc24.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/8908e189623f8851fa5c7ed858fff7486da2a296507515935ddb6f3faa42bc24.jpg)

![b027fb13d02eda49e1ee188a628b0867e717ee781931f93c97f161ec7ddab7e9.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/b027fb13d02eda49e1ee188a628b0867e717ee781931f93c97f161ec7ddab7e9.jpg)

![d0dd4a5b4fd2c73ea9ce723c85891357bf7d362319c68a482effd0ff287a93a3.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/images/d0dd4a5b4fd2c73ea9ce723c85891357bf7d362319c68a482effd0ff287a93a3.jpg)

### Tables

![0cbd5cc29a4a1570228e3a51be2c63b2170d2bc097bfe6b3a3b61669bc019963.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/0cbd5cc29a4a1570228e3a51be2c63b2170d2bc097bfe6b3a3b61669bc019963.jpg)

![0f665330fe7af8d3d6bd72a2cc0fbfe6cb386eb460be843c4031517cb7fd5f94.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/0f665330fe7af8d3d6bd72a2cc0fbfe6cb386eb460be843c4031517cb7fd5f94.jpg)

![10d44e45d8f4599eeb555b6e6cecb59d8077c40274dacc5091bddc987ced6757.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/10d44e45d8f4599eeb555b6e6cecb59d8077c40274dacc5091bddc987ced6757.jpg)

![6c8f4dbd67be71f263bbeaf397a033c1daf40434bdea032873cd98fd8b8f1631.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/6c8f4dbd67be71f263bbeaf397a033c1daf40434bdea032873cd98fd8b8f1631.jpg)

![8f62ee33146011ed50245cf6e873e5742e8b4d0a26195c30f7ed5a3bdcaaff06.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/8f62ee33146011ed50245cf6e873e5742e8b4d0a26195c30f7ed5a3bdcaaff06.jpg)

![91580ba514d4e35dd058c243176d6993a7cea501c0b986dd3fc27fb0539b6db4.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/91580ba514d4e35dd058c243176d6993a7cea501c0b986dd3fc27fb0539b6db4.jpg)

![c1e3b4912d530845fc3b52f86054f327354920dc2dec5ca17e66bfb6aaa9aa4e.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/c1e3b4912d530845fc3b52f86054f327354920dc2dec5ca17e66bfb6aaa9aa4e.jpg)

![ed9323f25f7ce31b7574f1a03a41a86e90e63c4d7ee2c24d9021a6721c59457a.jpg](../icml_results/3165_XPERT_%20Expert%20Knowledge%20Transfer%20for%20Effective%20Training%20of%20Language%20Models/tables/ed9323f25f7ce31b7574f1a03a41a86e90e63c4d7ee2c24d9021a6721c59457a.jpg)

<a id="paper-3127"></a>

## SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning


### Images

![018b5d21fb99c95c6c10b27dca7d0ff4371915880455f0db216a38c0d35c4fa9.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/018b5d21fb99c95c6c10b27dca7d0ff4371915880455f0db216a38c0d35c4fa9.jpg)

![08b41b9190b2f8b73ee9bf039af20fa2b2472c93e112070c4eb20523cc2d1286.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/08b41b9190b2f8b73ee9bf039af20fa2b2472c93e112070c4eb20523cc2d1286.jpg)

![1d5e2100982b9d95e85730795bef0297f6b07fa44c6bca5a48061fe55a2f8ce8.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/1d5e2100982b9d95e85730795bef0297f6b07fa44c6bca5a48061fe55a2f8ce8.jpg)

![718789ff064fc92bc178875104f8eb28db8719d43caa32bed59af9ca2723ca7d.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/718789ff064fc92bc178875104f8eb28db8719d43caa32bed59af9ca2723ca7d.jpg)

![8149b08938272404a623ae1000db5fba59670fe8876013cfac172b834740f831.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/8149b08938272404a623ae1000db5fba59670fe8876013cfac172b834740f831.jpg)

![87f415253dced78cb8275201c441e04181520046aacf263c82b34cc7c5c04495.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/87f415253dced78cb8275201c441e04181520046aacf263c82b34cc7c5c04495.jpg)

![b0de53fd4d801d0e9d52f1c1d4f186a69f57d0e5d786aa7a9c9cfa4809f8ac12.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/b0de53fd4d801d0e9d52f1c1d4f186a69f57d0e5d786aa7a9c9cfa4809f8ac12.jpg)

![d287505ab0677ac65a88d38f27cd3b48fb0c902509571a7590ea4bed846cd74d.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/d287505ab0677ac65a88d38f27cd3b48fb0c902509571a7590ea4bed846cd74d.jpg)

![ecbd9a26f0f36ceb33f1be1d78ec11123cef550d84668e5fa88d7849db1ff093.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/images/ecbd9a26f0f36ceb33f1be1d78ec11123cef550d84668e5fa88d7849db1ff093.jpg)

### Tables

![06aa989eaff4491f7ac3fc0c3b40bda641bf1222a51e098bcf3bf7872fca7955.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/06aa989eaff4491f7ac3fc0c3b40bda641bf1222a51e098bcf3bf7872fca7955.jpg)

![5397b6f473ee23d12d2a6ad3c70c575b802fb02f5d37efd3704265c5127bc4b7.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/5397b6f473ee23d12d2a6ad3c70c575b802fb02f5d37efd3704265c5127bc4b7.jpg)

![7a4e6e9bedf3b7e28463f4a7711ac1f2851cdaa44243cdc2f88f63b5ad4667bd.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/7a4e6e9bedf3b7e28463f4a7711ac1f2851cdaa44243cdc2f88f63b5ad4667bd.jpg)

![7b467af4e18925f2630351748071b23586dd0f662fa4751c7846d909bc25195f.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/7b467af4e18925f2630351748071b23586dd0f662fa4751c7846d909bc25195f.jpg)

![918ac7959f79df8847634a77bc42f5e48bf7d096cbba985847470e5905e9f9d8.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/918ac7959f79df8847634a77bc42f5e48bf7d096cbba985847470e5905e9f9d8.jpg)

![9c9bd5f21f13823f7267c07b6d9ca9aab0d4021e5e904e398ecbf7fcad7248da.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/9c9bd5f21f13823f7267c07b6d9ca9aab0d4021e5e904e398ecbf7fcad7248da.jpg)

![a5c598f0bde2127f69834f166a2c5c17da929bc58eacd6e8cf015149457f687a.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/a5c598f0bde2127f69834f166a2c5c17da929bc58eacd6e8cf015149457f687a.jpg)

![b58066194f40f550bcedb5b55cd34aa64a461e9df98a75691b21431d3be89b9a.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/b58066194f40f550bcedb5b55cd34aa64a461e9df98a75691b21431d3be89b9a.jpg)

![befa1bfcd623b3c1be29b2d5435f0f753d11e5e22e46be0ab24dc0711dcbbcf1.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/befa1bfcd623b3c1be29b2d5435f0f753d11e5e22e46be0ab24dc0711dcbbcf1.jpg)

![db7eb1f014dda6cafa952f068c2056067e66edc9a01b1bbf000248216a0b73df.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/db7eb1f014dda6cafa952f068c2056067e66edc9a01b1bbf000248216a0b73df.jpg)

![df6d0360aa799fa55adb14824954308aec90162efa79c58dbf9b81f9321d2be9.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/df6d0360aa799fa55adb14824954308aec90162efa79c58dbf9b81f9321d2be9.jpg)

![ebe0edf26396d17c1f6b0b9d2803b665a64d8f3c2297d770cab95402745de9fe.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/ebe0edf26396d17c1f6b0b9d2803b665a64d8f3c2297d770cab95402745de9fe.jpg)

![eee324f1e4fa18db5015993ecd3985f8a44347215951e5c575287aa0f76b51fd.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/eee324f1e4fa18db5015993ecd3985f8a44347215951e5c575287aa0f76b51fd.jpg)

![f745339fc632d8b2914e3a4831a1ddb2949fcfa5f6e276594b023282bcb858b3.jpg](../icml_results/3166_SRPO_%20Self-Reflective%20Policy%20Optimization%20for%20Long-Horizon%20Reasoning/tables/f745339fc632d8b2914e3a4831a1ddb2949fcfa5f6e276594b023282bcb858b3.jpg)

<a id="paper-3128"></a>

## RECTOR: Masked Region-Channel-Temporal Modeling for Affective and Cognitive Representation Learning


### Images

![14262fb71935f93db85d3dd49413ce5e33763a3e85658d15aa705e2efe5c84f7.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/14262fb71935f93db85d3dd49413ce5e33763a3e85658d15aa705e2efe5c84f7.jpg)

![2ed72f27a0f092ad8cd76471b1f8ecb9fac77254b28fe89986532f37c78ea46a.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/2ed72f27a0f092ad8cd76471b1f8ecb9fac77254b28fe89986532f37c78ea46a.jpg)

![367009d880fa348584f60bc45b12dd39c3089160988fc16da80043e8280422f9.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/367009d880fa348584f60bc45b12dd39c3089160988fc16da80043e8280422f9.jpg)

![4a6fd4ff882f4902d2f56f81a2467526b8a0999b8a202d44bb53442a7577bcf4.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/4a6fd4ff882f4902d2f56f81a2467526b8a0999b8a202d44bb53442a7577bcf4.jpg)

![847e879dcaad4afc9c255cf1647b30d3bd6332f17a103dc16cafd2a630700d69.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/847e879dcaad4afc9c255cf1647b30d3bd6332f17a103dc16cafd2a630700d69.jpg)

![94a889d3452c6c7b9a1a72dd642721facc511298962f9355334dba41c188021b.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/94a889d3452c6c7b9a1a72dd642721facc511298962f9355334dba41c188021b.jpg)

![9d16a51bd813b6210f9d54e440e3b833f14544062ca6bd4c045395b99658d42a.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/9d16a51bd813b6210f9d54e440e3b833f14544062ca6bd4c045395b99658d42a.jpg)

![9fd89c8a1b93ac4fc6d98d15664701dbcd1331b19bbc6577d3cce4c551bc2693.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/9fd89c8a1b93ac4fc6d98d15664701dbcd1331b19bbc6577d3cce4c551bc2693.jpg)

![a936e632a6137b5fce9f5f3003683068013b1c042d0dcd6762eab67ae9ca7fbe.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/a936e632a6137b5fce9f5f3003683068013b1c042d0dcd6762eab67ae9ca7fbe.jpg)

![aee163036bc674ede2970e4df4c66093c7d0d994deaacc3c843b9358dc6a7589.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/aee163036bc674ede2970e4df4c66093c7d0d994deaacc3c843b9358dc6a7589.jpg)

![b7118be5eff255bae2a164eb68abc6c8af1ab0064fb51f64c3e6e408a3a125f9.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/b7118be5eff255bae2a164eb68abc6c8af1ab0064fb51f64c3e6e408a3a125f9.jpg)

![ebe613e8e810f415ed3af45366835f5be8ea98b5c0a9e361729bf9d2af5c792c.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/ebe613e8e810f415ed3af45366835f5be8ea98b5c0a9e361729bf9d2af5c792c.jpg)

![fece75d90f99c5e053584e4ddbccfb78d15521e6c0e0914937b7748a5206579b.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/images/fece75d90f99c5e053584e4ddbccfb78d15521e6c0e0914937b7748a5206579b.jpg)

### Tables

![0b49b3ae8867932d934ec2e00dbfca87433dde7c1ddd80738209ad3a0575ee5e.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/0b49b3ae8867932d934ec2e00dbfca87433dde7c1ddd80738209ad3a0575ee5e.jpg)

![0d85a8019900502d968501677dc473b8331e3da16f1877a22ad59f79bb891075.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/0d85a8019900502d968501677dc473b8331e3da16f1877a22ad59f79bb891075.jpg)

![119637e89461b9815596ece4f4469e08bbb88bb1f78743c727e5ff722f88fc2c.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/119637e89461b9815596ece4f4469e08bbb88bb1f78743c727e5ff722f88fc2c.jpg)

![2420c286c8309b4b20b19315bf8e2a28d06ec9c43d363ac3893afc364931e686.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/2420c286c8309b4b20b19315bf8e2a28d06ec9c43d363ac3893afc364931e686.jpg)

![27e7b189b0525028d5be11f32ff02810c97f6062216593ac719d4fdd55057678.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/27e7b189b0525028d5be11f32ff02810c97f6062216593ac719d4fdd55057678.jpg)

![2bcf01c83ce71e4a08a135f2d67a12cafc00bdd88b939f91bdce3da31a3e86f8.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/2bcf01c83ce71e4a08a135f2d67a12cafc00bdd88b939f91bdce3da31a3e86f8.jpg)

![40f88b802ec98480ce494aada6a2e3960ae0de4789f15b086472dbab0cc27544.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/40f88b802ec98480ce494aada6a2e3960ae0de4789f15b086472dbab0cc27544.jpg)

![46825decea4ba93822ed0e2ecff3426f43aff767783898da2a175767b9d22432.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/46825decea4ba93822ed0e2ecff3426f43aff767783898da2a175767b9d22432.jpg)

![48fcb69486cd6b2f8aa664ed412587d3a74c0cfcd6539067f51163b71aa47915.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/48fcb69486cd6b2f8aa664ed412587d3a74c0cfcd6539067f51163b71aa47915.jpg)

![4ee99455184e7b2895ad55fdbb782560e1e8f6444f8092124981cdff1f52f413.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/4ee99455184e7b2895ad55fdbb782560e1e8f6444f8092124981cdff1f52f413.jpg)

![4efd4f2c51af2be7fadaf6e926dcb783237911d896e5ca5ca26d97cde0154841.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/4efd4f2c51af2be7fadaf6e926dcb783237911d896e5ca5ca26d97cde0154841.jpg)

![7a90d8aec028de3a93bd14fe46ce562ed62e37afd1aaf43a18970cd3fa6b83de.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/7a90d8aec028de3a93bd14fe46ce562ed62e37afd1aaf43a18970cd3fa6b83de.jpg)

![94acb1f6b1bfd1acb1b910fd1d85d7b0d9b1a4b7dbc2bc55c22ae524c57f7f33.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/94acb1f6b1bfd1acb1b910fd1d85d7b0d9b1a4b7dbc2bc55c22ae524c57f7f33.jpg)

![954b3df81d5cfd480c1400d0bdcaa9ff970a2403f129d119c1744a3ae3fd5ba5.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/954b3df81d5cfd480c1400d0bdcaa9ff970a2403f129d119c1744a3ae3fd5ba5.jpg)

![a68e5d8a08d089704b87ec6e7ba2e45aec3099303e86cbd7b0811d91c16a3c4e.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/a68e5d8a08d089704b87ec6e7ba2e45aec3099303e86cbd7b0811d91c16a3c4e.jpg)

![aed567241cef0c1aeeaf49a706d8f73e311334617d18bf785913f0cb4ee333f1.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/aed567241cef0c1aeeaf49a706d8f73e311334617d18bf785913f0cb4ee333f1.jpg)

![b429cce7113ca98196e3bf121da2e3e54830c179afbc31e857a8c9b7ea428cc8.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/b429cce7113ca98196e3bf121da2e3e54830c179afbc31e857a8c9b7ea428cc8.jpg)

![ccb8a2261e457ed3fd1d0cad70b4d7d82970488e3db6c24f953a26cfe5a9d686.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/ccb8a2261e457ed3fd1d0cad70b4d7d82970488e3db6c24f953a26cfe5a9d686.jpg)

![dff9d80522f4d4fc10d374d65d39c92ddbc646ea16758eab27eee9428a0ceeb3.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/dff9d80522f4d4fc10d374d65d39c92ddbc646ea16758eab27eee9428a0ceeb3.jpg)

![f1c13f9cca82f236288902103010073d9032683983d2d7199565f3040ba41f45.jpg](../icml_results/3167_RECTOR_%20Masked%20Region-Channel-Temporal%20Modeling%20for%20Affective%20and%20Cognitive%20Representation%20Learning/tables/f1c13f9cca82f236288902103010073d9032683983d2d7199565f3040ba41f45.jpg)

<a id="paper-3129"></a>

## AlignVid: Taming Visual Dominance via Training-Free Attention Modulation in Text-guided Image-to-Video Generation


### Images

![01018296a198745debff2e4c45e4e0f57334b0217dbe2b4a1852cb19d8e0f5f8.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/01018296a198745debff2e4c45e4e0f57334b0217dbe2b4a1852cb19d8e0f5f8.jpg)

![01aaef7508868f8319f8d06b5c1ee8b576d18bbd5a8ec8c9a25a48cd83d67ed4.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/01aaef7508868f8319f8d06b5c1ee8b576d18bbd5a8ec8c9a25a48cd83d67ed4.jpg)

![0867f3afb93f9894dd059f4d6ab9b623f09f84dc17a96ea20d18d4582bcbab76.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/0867f3afb93f9894dd059f4d6ab9b623f09f84dc17a96ea20d18d4582bcbab76.jpg)

![1a4533a58dc4b8c0e960a6a23a4f55155eff9a7d3802ca7c5ab54c3267e58519.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/1a4533a58dc4b8c0e960a6a23a4f55155eff9a7d3802ca7c5ab54c3267e58519.jpg)

![1e96e113253937f36535f66eca4070af9ad43e6a69262bb682c5516105f1cc14.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/1e96e113253937f36535f66eca4070af9ad43e6a69262bb682c5516105f1cc14.jpg)

![23441f44ef41c40d52e1a519d2e63e14b95679d9e7e5474460dca5cd1ce658d4.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/23441f44ef41c40d52e1a519d2e63e14b95679d9e7e5474460dca5cd1ce658d4.jpg)

![235461c05069ab14dce3ba5c7b091f2913eb9315857aa3d76095f1c0e37232bb.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/235461c05069ab14dce3ba5c7b091f2913eb9315857aa3d76095f1c0e37232bb.jpg)

![23fbd4506476a14b564b8041884ec7a874564c413b072260db67b96164f33903.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/23fbd4506476a14b564b8041884ec7a874564c413b072260db67b96164f33903.jpg)

![2ea852f2c4d4f3ada3b73044138e92fc9f6041b5a18072fa1883a44e9a5bd4ea.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/2ea852f2c4d4f3ada3b73044138e92fc9f6041b5a18072fa1883a44e9a5bd4ea.jpg)

![33475778ab0789ec24f2d55da41d1e29b661f28b46a31f4913330cf3fdded335.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/33475778ab0789ec24f2d55da41d1e29b661f28b46a31f4913330cf3fdded335.jpg)

![3bfd125aa78ac17c503c86de8395d297d7b98fafb6b2eacff7dcef69fad7e2af.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/3bfd125aa78ac17c503c86de8395d297d7b98fafb6b2eacff7dcef69fad7e2af.jpg)

![3e008e3001667f90c52dca9aadbb6bdb178ef11a40e4f1d2ea40029dd7272fe9.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/3e008e3001667f90c52dca9aadbb6bdb178ef11a40e4f1d2ea40029dd7272fe9.jpg)

![4a6792dd935e9be00cc7326cc6200121b6422983ff972a549736e3ffbf7a2a9a.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/4a6792dd935e9be00cc7326cc6200121b6422983ff972a549736e3ffbf7a2a9a.jpg)

![57595e8b1ab4be82f6709f80799ee406537d985c33fbb0fb16e9eb8bcf07bb9c.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/57595e8b1ab4be82f6709f80799ee406537d985c33fbb0fb16e9eb8bcf07bb9c.jpg)

![5cb6380ef92c290033a7ead1eb14c8bf8de95e2974537586558f71ffa1dd7cdb.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/5cb6380ef92c290033a7ead1eb14c8bf8de95e2974537586558f71ffa1dd7cdb.jpg)

![6278c50b8ffe4d6257bbdc60cfa817de808c50e97675423de5aada80583f0bca.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/6278c50b8ffe4d6257bbdc60cfa817de808c50e97675423de5aada80583f0bca.jpg)

![6a3f00fed73478109b90f693ab7e6c40171eac2cb475bdf56772c978c6acc226.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/6a3f00fed73478109b90f693ab7e6c40171eac2cb475bdf56772c978c6acc226.jpg)

![6d8914aedc81912122b6e7505aa86975dc77af242d4579f65387e57f953412f2.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/6d8914aedc81912122b6e7505aa86975dc77af242d4579f65387e57f953412f2.jpg)

![7c4f7048dbb69b6b6129db3765458257ff6152d2d6e710372b8f43d118f69ab9.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/7c4f7048dbb69b6b6129db3765458257ff6152d2d6e710372b8f43d118f69ab9.jpg)

![7ca11c956b382182e022abce8112833443f995f0a7d1b753dfba2d503217b2d9.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/7ca11c956b382182e022abce8112833443f995f0a7d1b753dfba2d503217b2d9.jpg)

![7ff8506089052843fba3de31727d0046ce39b6fe2f258193b8c0835dc56e9d7c.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/7ff8506089052843fba3de31727d0046ce39b6fe2f258193b8c0835dc56e9d7c.jpg)

![8749fd917418f92c047409104389e231ea76aa018b2b37f260b37b1e25ccc8f2.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/8749fd917418f92c047409104389e231ea76aa018b2b37f260b37b1e25ccc8f2.jpg)

![916c0c06cad05796aa844f523857c0b518192a695803757e8cd4a5a85be8cf13.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/916c0c06cad05796aa844f523857c0b518192a695803757e8cd4a5a85be8cf13.jpg)

![924b7bba982a139be294bffe72f138734e17ce556bac42d45c4ef107b8f4d862.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/924b7bba982a139be294bffe72f138734e17ce556bac42d45c4ef107b8f4d862.jpg)

![929ee8256d4c79f5b49ca5e012e3f348f55d0285513b9b54afa1a95b23374d8d.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/929ee8256d4c79f5b49ca5e012e3f348f55d0285513b9b54afa1a95b23374d8d.jpg)

![92d7aacf01d187c467bee257dc8d0a1ee308f18d4ac92da362cd9daf0c7bb0aa.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/92d7aacf01d187c467bee257dc8d0a1ee308f18d4ac92da362cd9daf0c7bb0aa.jpg)

![9aec77b4df3f4b0d50883592e337824eda43c77f6aad490e6a32acaa261eb922.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/9aec77b4df3f4b0d50883592e337824eda43c77f6aad490e6a32acaa261eb922.jpg)

![9c0ba50d01d5ef4e7d177241ac05a37d4b6dfcdaa69a04098ad778a97d463574.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/9c0ba50d01d5ef4e7d177241ac05a37d4b6dfcdaa69a04098ad778a97d463574.jpg)

![a1d898e23ba5c83cb865a6cfa155c8bcd57c78f5aa8d026cd693421838581eba.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/a1d898e23ba5c83cb865a6cfa155c8bcd57c78f5aa8d026cd693421838581eba.jpg)

![a31021a05489af3cdbf592ca76fc4b2dff30c1bd68833a1d76850763bcb94ce0.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/a31021a05489af3cdbf592ca76fc4b2dff30c1bd68833a1d76850763bcb94ce0.jpg)

![a341e9d2048697d6cea7e1c6ea057dddf312b94ad11bf7a22c1b9c557b48ca79.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/a341e9d2048697d6cea7e1c6ea057dddf312b94ad11bf7a22c1b9c557b48ca79.jpg)

![b0db1073569f8f0107c8cd13451b51beaca3ffb067c7fad2e43785e7cc4b7e8e.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/b0db1073569f8f0107c8cd13451b51beaca3ffb067c7fad2e43785e7cc4b7e8e.jpg)

![c285d69594e3f1f856f4a484c88bfe1f20cc0914f6b9c87e4b713564e910d926.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/c285d69594e3f1f856f4a484c88bfe1f20cc0914f6b9c87e4b713564e910d926.jpg)

![c3c5a181960fd461c470bf23c3bdcfee7b181183bed7edb3e68a8efce021d26d.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/c3c5a181960fd461c470bf23c3bdcfee7b181183bed7edb3e68a8efce021d26d.jpg)

![d44dec3e9f68ba58cd94378ad8dd6b3d092eb6a27316bf69cc50c687d46ffdf1.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/d44dec3e9f68ba58cd94378ad8dd6b3d092eb6a27316bf69cc50c687d46ffdf1.jpg)

![d5d8e085d3eb9ef489421b9d234f8382110d63579572e6ca7e7cf62c20e27ca5.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/d5d8e085d3eb9ef489421b9d234f8382110d63579572e6ca7e7cf62c20e27ca5.jpg)

![da23e50fdecb78a5e92c831cafc16e35b3dbd6571979d23cd969d935e7495f83.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/da23e50fdecb78a5e92c831cafc16e35b3dbd6571979d23cd969d935e7495f83.jpg)

![dd7cda05e11ce9501ca7e59de6728a583a6ea449193f5bcc4763f5128264c1aa.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/dd7cda05e11ce9501ca7e59de6728a583a6ea449193f5bcc4763f5128264c1aa.jpg)

![e1c3cb3dc5960eda2b10129db2a30ebeb4e8e298842a3dd81f019b8e94695ceb.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/e1c3cb3dc5960eda2b10129db2a30ebeb4e8e298842a3dd81f019b8e94695ceb.jpg)

![e273502cfbfd7238b3424568118a8cc7cc196874e8570c4d76900a52f2dfe49e.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/e273502cfbfd7238b3424568118a8cc7cc196874e8570c4d76900a52f2dfe49e.jpg)

![e2eb8d560f59655b2a6e0243cf09f0038a7fa7063f618f5583d15cfa2a867be0.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/e2eb8d560f59655b2a6e0243cf09f0038a7fa7063f618f5583d15cfa2a867be0.jpg)

![e9cd3888e3028889c741642c5b11e46259b8b6520c858da900f7098c6b9f7fff.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/e9cd3888e3028889c741642c5b11e46259b8b6520c858da900f7098c6b9f7fff.jpg)

![eefac4db9485e41529e05f57d9f5c981ac26d3e9564b321bd007cb7b6b064d24.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/eefac4db9485e41529e05f57d9f5c981ac26d3e9564b321bd007cb7b6b064d24.jpg)

![ef33f1eeba9aaded53330b6b963146fb6e8d694db8e89f0c0342720590088bfc.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/ef33f1eeba9aaded53330b6b963146fb6e8d694db8e89f0c0342720590088bfc.jpg)

![f04c8ab448e61cad81c47aaab38fef82296fc796ac1ee0458fe5c19f1b19e42d.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/f04c8ab448e61cad81c47aaab38fef82296fc796ac1ee0458fe5c19f1b19e42d.jpg)

![f3030bf5b59619ea6a46abdb5474fba4c06ec4ca97299be8d367c54ebfa82d03.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/f3030bf5b59619ea6a46abdb5474fba4c06ec4ca97299be8d367c54ebfa82d03.jpg)

![fdeb802d0c370cdff278e1d9b4580531de2c93a6beee1fb23648c0aa5999b1f2.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/fdeb802d0c370cdff278e1d9b4580531de2c93a6beee1fb23648c0aa5999b1f2.jpg)

![ff5b71b126b83e5c222da63eabe5f4b32462b6351288b1fb702b4db97c559260.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/images/ff5b71b126b83e5c222da63eabe5f4b32462b6351288b1fb702b4db97c559260.jpg)

### Tables

![14a5a064454490abc3e53e84af9c85f45607728fcf6705e14d9a8cdad9cc45da.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/14a5a064454490abc3e53e84af9c85f45607728fcf6705e14d9a8cdad9cc45da.jpg)

![1c5a52c22b390fd420199860b747fed86d4b1108c52a2b4e06f1c869350653d9.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/1c5a52c22b390fd420199860b747fed86d4b1108c52a2b4e06f1c869350653d9.jpg)

![37657d8ae1d228d868504b62483ce0580f6d9ffdc37383ec1f0ff0228101eb4e.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/37657d8ae1d228d868504b62483ce0580f6d9ffdc37383ec1f0ff0228101eb4e.jpg)

![3ed333d583cdfaa4a1d45a8967ecc6920a6d49f360f7eb2e5a1965a41fd42c87.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/3ed333d583cdfaa4a1d45a8967ecc6920a6d49f360f7eb2e5a1965a41fd42c87.jpg)

![68da636191530317fdeb062b285a6db3fed03c96f8d3b427ec9dce743b529f51.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/68da636191530317fdeb062b285a6db3fed03c96f8d3b427ec9dce743b529f51.jpg)

![6ef34b784879dfe89f81d88acafffa0affe82056bd9f235603b5291d2f0cccf0.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/6ef34b784879dfe89f81d88acafffa0affe82056bd9f235603b5291d2f0cccf0.jpg)

![84ca4ee785c4bd641ff2f918de85d0dd686bcb139c7d3a58a1bdab72d885d3b0.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/84ca4ee785c4bd641ff2f918de85d0dd686bcb139c7d3a58a1bdab72d885d3b0.jpg)

![8eb576e878cf4d39454203861618d5a3017b2a1388a4b00e807d2ca28d46d16e.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/8eb576e878cf4d39454203861618d5a3017b2a1388a4b00e807d2ca28d46d16e.jpg)

![95e67317f3e564221bb771f80f269c9f9df91b121f17aec8a2e37967cdccb251.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/95e67317f3e564221bb771f80f269c9f9df91b121f17aec8a2e37967cdccb251.jpg)

![9d7cd76900881da055a14f196b79eb7e8daa7bb7e52e04762d7c91e9f89e3164.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/9d7cd76900881da055a14f196b79eb7e8daa7bb7e52e04762d7c91e9f89e3164.jpg)

![b7edbb2d8f6424b7e96988100744edfcbb4525fa48768891826b82f5a5d83a62.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/b7edbb2d8f6424b7e96988100744edfcbb4525fa48768891826b82f5a5d83a62.jpg)

![b964039851541937b9302c45af0c912f5ad69a3b86fbc13709eb6fa441a7361c.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/b964039851541937b9302c45af0c912f5ad69a3b86fbc13709eb6fa441a7361c.jpg)

![c512884657ecbd966664d32a96d7a0643641fa10ff7f7553c84a1a1e67815835.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/c512884657ecbd966664d32a96d7a0643641fa10ff7f7553c84a1a1e67815835.jpg)

![c84b47a0dcad742b45df2c6c2a49cd298d611938fa92cd12e4fada853090f3b7.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/c84b47a0dcad742b45df2c6c2a49cd298d611938fa92cd12e4fada853090f3b7.jpg)

![c91ce83dbd1f2c89dc6346e79dcf2779daaa94b679f025d04e3d39258fbc030e.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/c91ce83dbd1f2c89dc6346e79dcf2779daaa94b679f025d04e3d39258fbc030e.jpg)

![f489215877843651d6945fe634b58e7d07c81b8934964e370a6a84775dca394a.jpg](../icml_results/3168_AlignVid_%20Taming%20Visual%20Dominance%20via%20Training-Free%20Attention%20Modulation%20in%20Text-guided%20Image-to-Vid/tables/f489215877843651d6945fe634b58e7d07c81b8934964e370a6a84775dca394a.jpg)

<a id="paper-3130"></a>

## Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation


### Images

![305bed9aa870d7097d3fc2b858f13ec5f2cdb7d9ab54d141632282a4e03c56c7.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/images/305bed9aa870d7097d3fc2b858f13ec5f2cdb7d9ab54d141632282a4e03c56c7.jpg)

![634b6d3495a7a849cfb7c33f29cc4e20c78e8f4413be2417448a6532d2ee5ccf.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/images/634b6d3495a7a849cfb7c33f29cc4e20c78e8f4413be2417448a6532d2ee5ccf.jpg)

![847806d96884a7f9019e7ba0b74c2186eb4486d8a2f555f42139d2fd42b54e8c.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/images/847806d96884a7f9019e7ba0b74c2186eb4486d8a2f555f42139d2fd42b54e8c.jpg)

### Tables

![01cc2e865e240b979fff0a8fafd287c74acc33cf63d91834471df45a96c13524.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/tables/01cc2e865e240b979fff0a8fafd287c74acc33cf63d91834471df45a96c13524.jpg)

![1d5a45e1e415d42ab1e493fbb3c14a16c2b987bfd454bfbf8ea733d03495df2b.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/tables/1d5a45e1e415d42ab1e493fbb3c14a16c2b987bfd454bfbf8ea733d03495df2b.jpg)

![b11d588654dc9786e7dc4fb7feee5c85707167ed239b07f26369e52d9dceb7bc.jpg](../icml_results/3169_Unison_%20Benchmarking%20Unified%20Multimodal%20Models%20via%20Synergistic%20Understanding%20and%20Generation/tables/b11d588654dc9786e7dc4fb7feee5c85707167ed239b07f26369e52d9dceb7bc.jpg)

<a id="paper-3131"></a>

## Agora: Toward Autonomous Bug Detection in Production-Level Consensus Protocols with LLM Agents


### Images

![422f6aada5ee7890cd458c07720eb752a2d1a887094d679619cdc589b275fd94.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/images/422f6aada5ee7890cd458c07720eb752a2d1a887094d679619cdc589b275fd94.jpg)

![63f7d6a551edf78890fd18bc567578c2f041d6c73b0b386c767dd84fe028683b.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/images/63f7d6a551edf78890fd18bc567578c2f041d6c73b0b386c767dd84fe028683b.jpg)

![c6ec896ce48bf0a84661b0f410f68b318c1291d62a9444cc08346f058f608222.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/images/c6ec896ce48bf0a84661b0f410f68b318c1291d62a9444cc08346f058f608222.jpg)

![f3b9eb4b22d247275e7f3fdeac292a6f1524bf743fc1ff2657850ddbff7a1e65.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/images/f3b9eb4b22d247275e7f3fdeac292a6f1524bf743fc1ff2657850ddbff7a1e65.jpg)

### Tables

![473ad62a38f0f9d0853a5b95278654a0a1bbaf7e6a28b176994d640b6eaab479.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/tables/473ad62a38f0f9d0853a5b95278654a0a1bbaf7e6a28b176994d640b6eaab479.jpg)

![67151d5c592e4ae21998b0849b8bdf724965917785c5aa2e52c8329801cbf77e.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/tables/67151d5c592e4ae21998b0849b8bdf724965917785c5aa2e52c8329801cbf77e.jpg)

![ccb705499e038939a436eebfbfcd85c584356e6b64de5a99db8277640c7ed3f4.jpg](../icml_results/3170_Agora_%20Toward%20Autonomous%20Bug%20Detection%20in%20Production-Level%20Consensus%20Protocols%20with%20LLM%20Agents/tables/ccb705499e038939a436eebfbfcd85c584356e6b64de5a99db8277640c7ed3f4.jpg)

<a id="paper-3132"></a>

## SimpleMem: Efficient Lifelong Memory for LLM Agents


### Images

![7dad8f5933d9fb3dd41889c51d144f24018e35024bfa7246d88a4aee8edfd62e.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/images/7dad8f5933d9fb3dd41889c51d144f24018e35024bfa7246d88a4aee8edfd62e.jpg)

![7fdaa30c94668becf481ac9d068df2d35366d404fbe2dc507d965a3d2f4004ac.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/images/7fdaa30c94668becf481ac9d068df2d35366d404fbe2dc507d965a3d2f4004ac.jpg)

![ec50083925a6e7ad480b6d916a23330e61c03b6cd703e77a63b41ffbda1d2217.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/images/ec50083925a6e7ad480b6d916a23330e61c03b6cd703e77a63b41ffbda1d2217.jpg)

### Tables

![102fa613f585abdbdb8c1c5116f1d13b7280c238b248cf04051be341fc9af6bc.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/102fa613f585abdbdb8c1c5116f1d13b7280c238b248cf04051be341fc9af6bc.jpg)

![1259e399b8301c5f88e78a85d4b2160fdfb5c38eb3fdcd058de2340ab7c6758c.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/1259e399b8301c5f88e78a85d4b2160fdfb5c38eb3fdcd058de2340ab7c6758c.jpg)

![2c8eb7a7b1b2ede31a3511f3a1c58b4acf6f70073918836ec276ffb6e7247f11.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/2c8eb7a7b1b2ede31a3511f3a1c58b4acf6f70073918836ec276ffb6e7247f11.jpg)

![61916c3bcbaa2081509d80d5a7f0651c28fa3959794c692e8aea9938ee4f74b5.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/61916c3bcbaa2081509d80d5a7f0651c28fa3959794c692e8aea9938ee4f74b5.jpg)

![93262320f1a5f7423071c39125bfd001d0b04ceb5dbbb03150c9271aa8b1cb20.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/93262320f1a5f7423071c39125bfd001d0b04ceb5dbbb03150c9271aa8b1cb20.jpg)

![b9180afed3871c9e0b0e1d31821672a58687b9755314272c192d22f1ce1924c7.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/b9180afed3871c9e0b0e1d31821672a58687b9755314272c192d22f1ce1924c7.jpg)

![d9306be9fed41ec7d170813ef094f4cceebe40da71f03692f124da98b5b363fa.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/d9306be9fed41ec7d170813ef094f4cceebe40da71f03692f124da98b5b363fa.jpg)

![f4200ff5dcaf9fdfa2db5dbaf2378cb591775196bcb57cef59ace118ea79a015.jpg](../icml_results/3171_SimpleMem_%20Efficient%20Lifelong%20Memory%20for%20LLM%20Agents/tables/f4200ff5dcaf9fdfa2db5dbaf2378cb591775196bcb57cef59ace118ea79a015.jpg)

<a id="paper-3133"></a>

## Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models


### Images

![2a583547c77d785f25368fad103b57fa428f5dab5d64fcb378edb74896a2442f.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/2a583547c77d785f25368fad103b57fa428f5dab5d64fcb378edb74896a2442f.jpg)

![53147c33fcbc9d925a3c5358be7da3d82acd2cc0b21f35a118f62f8ff6547402.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/53147c33fcbc9d925a3c5358be7da3d82acd2cc0b21f35a118f62f8ff6547402.jpg)

![5eccb8885c7f86a57fcd3967f81bebbec2a24c598a040ddd7e39aaddccf29653.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/5eccb8885c7f86a57fcd3967f81bebbec2a24c598a040ddd7e39aaddccf29653.jpg)

![8706bec80c15ae1a571c264bf1bc241a3af14151307b7f49b1fd22b1746273ad.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/8706bec80c15ae1a571c264bf1bc241a3af14151307b7f49b1fd22b1746273ad.jpg)

![8af32531e29baafd1d3f5763917a5308d2cf7df0bb76f63627d8e4c56be5e705.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/8af32531e29baafd1d3f5763917a5308d2cf7df0bb76f63627d8e4c56be5e705.jpg)

![a33adb4b6e99215e269ba3a79e1d97c437a8b29fdd689199ee84eb67a3d5e24d.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/a33adb4b6e99215e269ba3a79e1d97c437a8b29fdd689199ee84eb67a3d5e24d.jpg)

![ad4bec445aa2b3ec7d0ba0ce7790f2fad1ea9baf8443c55ca8562b241da3de95.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/ad4bec445aa2b3ec7d0ba0ce7790f2fad1ea9baf8443c55ca8562b241da3de95.jpg)

![b7616b10a9c09aad8674d909b59dc948f823c584954263fc6fe8bc14c035aa81.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/b7616b10a9c09aad8674d909b59dc948f823c584954263fc6fe8bc14c035aa81.jpg)

![bf2b61ad43005a4856bf78306e3e33537e61e9bbd0a67bb4c4970dd1c73d870f.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/bf2b61ad43005a4856bf78306e3e33537e61e9bbd0a67bb4c4970dd1c73d870f.jpg)

![e40a2bd4084f83a2f468fb7af5e2c7d4966a8661c75e4cfea14f9d417b97cfb7.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/images/e40a2bd4084f83a2f468fb7af5e2c7d4966a8661c75e4cfea14f9d417b97cfb7.jpg)

### Tables

![350339dd7ac260ed5eb5620265c9e09be19712744b88beaa9f8a80e5e6f8bfb0.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/350339dd7ac260ed5eb5620265c9e09be19712744b88beaa9f8a80e5e6f8bfb0.jpg)

![46b67ec66877a4713edd816c15acc3097d6e99c6b6aeb7cf8750e2dbe3ea2320.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/46b67ec66877a4713edd816c15acc3097d6e99c6b6aeb7cf8750e2dbe3ea2320.jpg)

![622cb0cc50e95054615dbfa63d7518492e5f17eed811689b401878c4ba0ff0cb.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/622cb0cc50e95054615dbfa63d7518492e5f17eed811689b401878c4ba0ff0cb.jpg)

![6e7db5858337e528fe07659681088d46ac6e59450240d89241f88d29badf4e8c.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/6e7db5858337e528fe07659681088d46ac6e59450240d89241f88d29badf4e8c.jpg)

![8b13d3919c93d248726b82389620c334d941c0429686fa2d6c5097fb792133b1.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/8b13d3919c93d248726b82389620c334d941c0429686fa2d6c5097fb792133b1.jpg)

![c54ff642f94206fc04cae6ff45184e48f2a3a00b7c8e4c76f584a0b26bee5f92.jpg](../icml_results/3172_Dispersion%20Loss%20Counteracts%20Embedding%20Condensation%20and%20Improves%20Generalization%20in%20Small%20Language%20Mod/tables/c54ff642f94206fc04cae6ff45184e48f2a3a00b7c8e4c76f584a0b26bee5f92.jpg)

<a id="paper-3134"></a>

## Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression


### Images

![2db6ea18c46977a09c7e62cd10d875996bbd8efe627b4a326d3c8f481879e9e3.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/2db6ea18c46977a09c7e62cd10d875996bbd8efe627b4a326d3c8f481879e9e3.jpg)

![51a2ead7ce8554251d3e28117c8e552e35671b011bff1bcf33ddeb63806215af.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/51a2ead7ce8554251d3e28117c8e552e35671b011bff1bcf33ddeb63806215af.jpg)

![5a3a12f4787074fed9b0f916306d66a5e9d306077b532bd2901577c1a7dcaa7e.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/5a3a12f4787074fed9b0f916306d66a5e9d306077b532bd2901577c1a7dcaa7e.jpg)

![5cf99f8ca5b152d86b79a120cce9c7dcd6f19ec3bcbd2620cc2884e9c2d3c41a.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/5cf99f8ca5b152d86b79a120cce9c7dcd6f19ec3bcbd2620cc2884e9c2d3c41a.jpg)

![9494b8f187e83f5574e13be437ebfda5fbb0e0e37440bfe8676c30afce08d79f.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/9494b8f187e83f5574e13be437ebfda5fbb0e0e37440bfe8676c30afce08d79f.jpg)

![d18a61a2f878b81d111565f11ebac2f6ca385942875ae6607772d6d2a7ec9e01.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/d18a61a2f878b81d111565f11ebac2f6ca385942875ae6607772d6d2a7ec9e01.jpg)

![d7af5366fef14c8e2386181a6b551ce2e2816c4c02be37fe0d575a97284edb43.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/d7af5366fef14c8e2386181a6b551ce2e2816c4c02be37fe0d575a97284edb43.jpg)

![ea98290e755ba39956c03049a49263b03bd7e0dc863c1431d1840aae6b21a14f.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/ea98290e755ba39956c03049a49263b03bd7e0dc863c1431d1840aae6b21a14f.jpg)

![ead81e3c8c4680564995c46022132cc5ba3a734a1192dfb3ab1371e222157bbc.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/ead81e3c8c4680564995c46022132cc5ba3a734a1192dfb3ab1371e222157bbc.jpg)

![efd62d036b0f1c82398ca986a32c792dea6a413a734ac43f89f7680d98395f1e.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/efd62d036b0f1c82398ca986a32c792dea6a413a734ac43f89f7680d98395f1e.jpg)

![f43e5462eb08d41e65e8be84889502504f5bd2c0dcf923d19b71b8d38f9d74c0.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/images/f43e5462eb08d41e65e8be84889502504f5bd2c0dcf923d19b71b8d38f9d74c0.jpg)

### Tables

![0f366e61be84491b27c7df134bcb690d40965524e278f3d85e78b8f26f94bb36.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/0f366e61be84491b27c7df134bcb690d40965524e278f3d85e78b8f26f94bb36.jpg)

![152fe0b7f48b6eebc85dfbf358a557051fb63c7cf183daebc3e31b09c9829a9c.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/152fe0b7f48b6eebc85dfbf358a557051fb63c7cf183daebc3e31b09c9829a9c.jpg)

![47ce5aa7c9ef07e77152c5a0c932d1aac455bdf304b309234cf9b7f4c7f40c8c.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/47ce5aa7c9ef07e77152c5a0c932d1aac455bdf304b309234cf9b7f4c7f40c8c.jpg)

![4988788bcfc7c453a2ac139ca92aa4cf4f7bdb3792558cebbf52f563e1909e6c.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/4988788bcfc7c453a2ac139ca92aa4cf4f7bdb3792558cebbf52f563e1909e6c.jpg)

![4d89b8ee508efbec40d136a792e6b0e45f2765a67ea4df707e38818609e3dc62.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/4d89b8ee508efbec40d136a792e6b0e45f2765a67ea4df707e38818609e3dc62.jpg)

![59fe5de7c9d4e4cd9bca66c449a5b8259cfcf4be2369544f40803d5b7e132d80.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/59fe5de7c9d4e4cd9bca66c449a5b8259cfcf4be2369544f40803d5b7e132d80.jpg)

![6e96065e37c3b0706722df07df1ceed7892abc219fe1651fea52bd6d6ded6cb3.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/6e96065e37c3b0706722df07df1ceed7892abc219fe1651fea52bd6d6ded6cb3.jpg)

![8b5218a34a7967f1cf824ae64d8c8512e9d21480b6198db632c08e3844cbd599.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/8b5218a34a7967f1cf824ae64d8c8512e9d21480b6198db632c08e3844cbd599.jpg)

![990116c03da5c9497752b73bb4c16438ab09abbbfdeee56c9487ae17715b126a.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/990116c03da5c9497752b73bb4c16438ab09abbbfdeee56c9487ae17715b126a.jpg)

![bba259686dc05be2f8ddb3479d217375142219234702dc4f06907648f8086e66.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/bba259686dc05be2f8ddb3479d217375142219234702dc4f06907648f8086e66.jpg)

![c5b84383d9922d40dc5985e4e5d05a26b512f7df30f1725b3768587ce0a67d9b.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/c5b84383d9922d40dc5985e4e5d05a26b512f7df30f1725b3768587ce0a67d9b.jpg)

![c6d0eec07ef1d4083f9945a210e21a37e7243a50c6e491d48489fae59c1762f8.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/c6d0eec07ef1d4083f9945a210e21a37e7243a50c6e491d48489fae59c1762f8.jpg)

![cf09afc7986a2d603e2e4512a94d6a1a80d24cd3f8c19f541e2f960593529063.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/cf09afc7986a2d603e2e4512a94d6a1a80d24cd3f8c19f541e2f960593529063.jpg)

![d1cfc1323de08c3763658c808129917c90d167c744ab7c7850f39ebe70d6a159.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/d1cfc1323de08c3763658c808129917c90d167c744ab7c7850f39ebe70d6a159.jpg)

![d456ec540695ea5cfb8214029f458cb2861396b8fdb20972c1a9d666c68f0db2.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/d456ec540695ea5cfb8214029f458cb2861396b8fdb20972c1a9d666c68f0db2.jpg)

![e8b381125c210079ccf8570be8028ccef353a3fe380ceb2975f27da7773cb4f2.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/e8b381125c210079ccf8570be8028ccef353a3fe380ceb2975f27da7773cb4f2.jpg)

![f6fb540f645ad6c977226f9bbdbbb25ec24955f2700aba1cd9685cd54f2b7518.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/f6fb540f645ad6c977226f9bbdbbb25ec24955f2700aba1cd9685cd54f2b7518.jpg)

![f7e9badf0be0fe4aab88f21fca12fb7dd7d518b3278afc870524f8ce1fdc4642.jpg](../icml_results/3173_Semantic%20Integrity%20Matters_%20Benchmarking%20and%20Preserving%20High-Density%20Reasoning%20in%20KV%20Cache%20Compressi/tables/f7e9badf0be0fe4aab88f21fca12fb7dd7d518b3278afc870524f8ce1fdc4642.jpg)

<a id="paper-3135"></a>

## JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments

### Images

![2dc70e623a90b0747ea9d7d64248e6433084989aea77ab69d9796930b41b82e0.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/images/2dc70e623a90b0747ea9d7d64248e6433084989aea77ab69d9796930b41b82e0.jpg)

![47a94f86034f72c0b5b1e37e0cd034414f5fd971b9e4f460f3b213fe0050f7a8.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/images/47a94f86034f72c0b5b1e37e0cd034414f5fd971b9e4f460f3b213fe0050f7a8.jpg)

![5cf8b320ec42ae676a95c432433e8158738ee11af8d52040eada88ccf70168ac.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/images/5cf8b320ec42ae676a95c432433e8158738ee11af8d52040eada88ccf70168ac.jpg)

![79304a75a8d14ff4122ce06eea9732482d64db25148a13a0f71a417a7edf4cfa.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/images/79304a75a8d14ff4122ce06eea9732482d64db25148a13a0f71a417a7edf4cfa.jpg)

### Tables

![42ebf4703ff9901de68c78f5a8630833ed4bfaf6edbd317893511694ffa2ad0a.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/42ebf4703ff9901de68c78f5a8630833ed4bfaf6edbd317893511694ffa2ad0a.jpg)

![6c5a22ddd79801684dbcc4d9db05a7ee05ce3f712c7efe71d323853f7e6408b9.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/6c5a22ddd79801684dbcc4d9db05a7ee05ce3f712c7efe71d323853f7e6408b9.jpg)

![7e8f0c99078546ea720963c109b3a23237117580fa41edff82374d039369a67f.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/7e8f0c99078546ea720963c109b3a23237117580fa41edff82374d039369a67f.jpg)

![97c45f18497e325d7d6ad712d58850ad88f745346d7b37997aff5396253ccd86.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/97c45f18497e325d7d6ad712d58850ad88f745346d7b37997aff5396253ccd86.jpg)

![b7fc1a0442bcce3df2a0906b5a48a69d1982332f9ed8fdc49429d99466f3a2a9.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/b7fc1a0442bcce3df2a0906b5a48a69d1982332f9ed8fdc49429d99466f3a2a9.jpg)

![bbd58a10cdb4d3d139ee54235b034384dff0b2a7f5e4ed4d2797e4b7294cd999.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/bbd58a10cdb4d3d139ee54235b034384dff0b2a7f5e4ed4d2797e4b7294cd999.jpg)

![c7250a06a686a801799a51e2237b3105839a1bcc6c1332fd47d8c118c7be8d7a.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/c7250a06a686a801799a51e2237b3105839a1bcc6c1332fd47d8c118c7be8d7a.jpg)

![f13e1f0c32ebec0779e9498715533072bf17d6f379ddf1b8ff9f9b5641d5761f.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/f13e1f0c32ebec0779e9498715533072bf17d6f379ddf1b8ff9f9b5641d5761f.jpg)

![f983ce3165ce2a37dd5ff1c31c40acad7eccae1e98b9cb08d77bc71f6aea5164.jpg](../icml_results/3174_JAEGER_%20Joint%203D%20Audio-Visual%20Grounding%20and%20Reasoning%20in%20Simulated%20Physical%20Environments/tables/f983ce3165ce2a37dd5ff1c31c40acad7eccae1e98b9cb08d77bc71f6aea5164.jpg)
