# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 96 of 197

## 目录 (Table of Contents)

1. [Resolving Blind Inverse Problems under Dynamic Range Compression via Structured Forward Operator Modeling](#paper-3136)
2. [TiME: Test-Time Mixture-of-Experts Routing via Asymmetric CO-Optimal Transport for Continual Test-Time Adaptation](#paper-3137)
3. [Estimating Correlation Clustering Cost in Node-Arrival Stream](#paper-3138)
4. [Pseudo-Mallows for Efficient Probabilistic Preference Learning](#paper-3139)
5. [Scaling Inference-Time Computation via Opponent Simulation: Enabling Online Strategic Adaptation in Repeated Negotiation](#paper-3140)
6. [HVAE: Hyperbolic Variational Autoencoder For Flexible Knowledge Transfer Across Multiple Domains](#paper-3141)
7. [Condition Number Based Low-Bit Quantization for Image Super-Resolution](#paper-3142)
8. [STAR-VAE: Structured Topology-Aware Regularization for Audio Reconstruction and Generation](#paper-3143)
9. [VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos](#paper-3144)
10. [SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass](#paper-3145)
11. [Opt-Verifier: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification](#paper-3146)
12. [AnalogVerifier: A Neuro-Symbolic Framework for Analog Circuit Verification](#paper-3147)
13. [Rethinking 3D Shape Generation: Diffusion over Superquadrics](#paper-3148)
14. [Ramba: Selective State-Space Models for Relational Deep Learning](#paper-3149)
15. [Causal Disentangled Anchor Learning for Scalable Fair Multi-view Clustering](#paper-3150)
16. [Decoupling Universal Laws and Environmental Heterogeneity: A Physics-Inspired Framework for Robust Spatio-Temporal Forecasting](#paper-3151)
17. [From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](#paper-3152)
18. [Beyond Reactivity: Proactive Adaptive Conformal Inference for Online LLM Factuality](#paper-3153)
19. [Distillation Models are Good Samplers for Diffusion Reinforcement Learning](#paper-3154)
20. [Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning](#paper-3155)
21. [Calibrated Multimodal Representation Learning with Missing Modalities](#paper-3156)
22. [Cardio-mmFlow: A Gaussian-Prior-Free Physics-Informed Flow Matching Framework for Electrocardiogram to mmWave Radar Synthesis](#paper-3157)
23. [MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics](#paper-3158)
24. [Clipped Q-Learning: Your Value Clipping Is Secretly A Robust Operator](#paper-3159)
25. [CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning](#paper-3160)
26. [Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations](#paper-3161)
27. [Sparse Relaxed-Lasso Steering: Automatic Sparse Autoencoder Feature Selection for Precise Image Editing](#paper-3162)
28. [Anchor-Final Self-Supervision Drives Hallucination-Aware Optimization in Large Vision-Language Models](#paper-3163)
29. [M-IDoL: Information Decomposition for Modality-Specific and Diverse Representation Learning in Medical Foundation Model](#paper-3164)
30. [Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](#paper-3165)
31. [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](#paper-3166)
32. [Lookahead Path Likelihood Optimization for Diffusion LLMs](#paper-3167)
33. [FlowCloud: Learning Continuous Spatiotemporal Dynamics from Unpaired Sparse Point Cloud Snapshots](#paper-3168)

---


<a id="paper-3136"></a>

## Resolving Blind Inverse Problems under Dynamic Range Compression via Structured Forward Operator Modeling

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

## Resolving Blind Inverse Problems under Dynamic Range Compression via Structured Forward Operator Modeling


### Images

![164d36d77b856cab140cb828d355a08155be113b9510ed7145f7749258cbd5d1.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/164d36d77b856cab140cb828d355a08155be113b9510ed7145f7749258cbd5d1.jpg)

![1cd1c481c38f5d6de3f11006a3bff80619b34ea1041bd676ea985d70d520dcf9.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/1cd1c481c38f5d6de3f11006a3bff80619b34ea1041bd676ea985d70d520dcf9.jpg)

![21ff0c9977d48b782099d9d7ea5390d49b70c731fa7b3ae75fb6ab002b8817e9.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/21ff0c9977d48b782099d9d7ea5390d49b70c731fa7b3ae75fb6ab002b8817e9.jpg)

![4e224190417f37cebacdb9748cd7e1107f3fc81c811b76548d5a2226589457fd.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/4e224190417f37cebacdb9748cd7e1107f3fc81c811b76548d5a2226589457fd.jpg)

![5d8f38f3307234059b8e27df0b7011a9c1c7fe269c9083edc9f080cc2032e8f2.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/5d8f38f3307234059b8e27df0b7011a9c1c7fe269c9083edc9f080cc2032e8f2.jpg)

![67caf8f494c123bc8f02c2d092d36606182b6177a360ec0060acec12683f1922.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/67caf8f494c123bc8f02c2d092d36606182b6177a360ec0060acec12683f1922.jpg)

![68704b4e9c7092777351e7c27202a84e96946b165056c7ee09ac5c2dae396a29.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/68704b4e9c7092777351e7c27202a84e96946b165056c7ee09ac5c2dae396a29.jpg)

![85dc6c07d20b695556b5bcf92f70ac44e8609f6821903d0cebf66a3ca78fd530.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/85dc6c07d20b695556b5bcf92f70ac44e8609f6821903d0cebf66a3ca78fd530.jpg)

![ca5e15f004fb46c33e7472765e0245f32a572ea374adbf447a400f708cd5aebc.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/ca5e15f004fb46c33e7472765e0245f32a572ea374adbf447a400f708cd5aebc.jpg)

![ceaf033a764a6acad97728c1b1c27ff6ff7023391bf2f8fbf3e6c2050f1773bb.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/ceaf033a764a6acad97728c1b1c27ff6ff7023391bf2f8fbf3e6c2050f1773bb.jpg)

![dd518d8d3560b587f8928be137c0b73b8c097305369f7544b88ea4c07a160e86.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/dd518d8d3560b587f8928be137c0b73b8c097305369f7544b88ea4c07a160e86.jpg)

![fc0a6ca1dee905fd742a74aca92eef0d3078fd6fd4f17227514b093c891370c7.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/images/fc0a6ca1dee905fd742a74aca92eef0d3078fd6fd4f17227514b093c891370c7.jpg)

### Tables

![3d0cc75f989c7d6d9841970ed77b8a206a1ce50b507c9adbba7401aeb8a1fb85.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/3d0cc75f989c7d6d9841970ed77b8a206a1ce50b507c9adbba7401aeb8a1fb85.jpg)

![466f44346d076438cc895dfcfa1875fd6d7c189c9474ed834e5e3a05a904ec5c.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/466f44346d076438cc895dfcfa1875fd6d7c189c9474ed834e5e3a05a904ec5c.jpg)

![681976bebd5720179271615dc32ededd003adf334635361b46a9ea341ce35430.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/681976bebd5720179271615dc32ededd003adf334635361b46a9ea341ce35430.jpg)

![9d0b803067e6e0692e9e457728b25a771e731988e62d860c41cd7b2db3b6de97.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/9d0b803067e6e0692e9e457728b25a771e731988e62d860c41cd7b2db3b6de97.jpg)

![cc25d445c2e93238c0b105c445368e74c312fa8e7d2bb0f636cecad6a896f33b.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/cc25d445c2e93238c0b105c445368e74c312fa8e7d2bb0f636cecad6a896f33b.jpg)

![d3f85c92e03346382da89d8a78b6757106acc54bc0660773baf971784e19a803.jpg](../icml_results/3175_Resolving%20Blind%20Inverse%20Problems%20under%20Dynamic%20Range%20Compression%20via%20Structured%20Forward%20Operator%20Mod/tables/d3f85c92e03346382da89d8a78b6757106acc54bc0660773baf971784e19a803.jpg)

<a id="paper-3137"></a>

## TiME: Test-Time Mixture-of-Experts Routing via Asymmetric CO-Optimal Transport for Continual Test-Time Adaptation


### Images

![313278e9b661e3d2479b214a9a4bfac680abf739ec2e833aa746495865fd194d.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/313278e9b661e3d2479b214a9a4bfac680abf739ec2e833aa746495865fd194d.jpg)

![4a559939bce632c4abe53924a8f757f8ce66b3e3864a416fa4c6fbc50b74bfe6.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/4a559939bce632c4abe53924a8f757f8ce66b3e3864a416fa4c6fbc50b74bfe6.jpg)

![59b4695b91e2218ca1989b64ecad0e547a7e5c1dcc5c4d34f420d71f5dfd0ffe.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/59b4695b91e2218ca1989b64ecad0e547a7e5c1dcc5c4d34f420d71f5dfd0ffe.jpg)

![672727dadbe81854497fd0220b98f4c4e6aa1186d6e44ab96ba70a5412dd7dbc.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/672727dadbe81854497fd0220b98f4c4e6aa1186d6e44ab96ba70a5412dd7dbc.jpg)

![7a80ce66ad8fd9d3f3642cc30b5030b5e0e20ee7755cd8e294af0d46e6111bf8.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/7a80ce66ad8fd9d3f3642cc30b5030b5e0e20ee7755cd8e294af0d46e6111bf8.jpg)

![92f74c4aa7dcedb785f86cf345b8d8e22fef35e45c555e674d49726706b14eed.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/92f74c4aa7dcedb785f86cf345b8d8e22fef35e45c555e674d49726706b14eed.jpg)

![d17e959a273cdac1bd11cdb3d8695a95b634f3cfe55ce8d9259d1bdbace73a49.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/d17e959a273cdac1bd11cdb3d8695a95b634f3cfe55ce8d9259d1bdbace73a49.jpg)

![d7734637058a37d70ef607a89cc616f6d86d2b96a6e79473a8cf439b78c964a8.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/d7734637058a37d70ef607a89cc616f6d86d2b96a6e79473a8cf439b78c964a8.jpg)

![e1dde7d04260f29bd738a49680cbd6be2cef12c6f5d2a0a08a3b5212b564dcfa.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/e1dde7d04260f29bd738a49680cbd6be2cef12c6f5d2a0a08a3b5212b564dcfa.jpg)

![fe34e667b756bfc19fe157a30a02c80d0a42d355cdc44354ce041ffadca7484b.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/images/fe34e667b756bfc19fe157a30a02c80d0a42d355cdc44354ce041ffadca7484b.jpg)

### Tables

![0ccfef2be10f1c1e001414e04b1b9e7b70f33c237a726e4a4c7ffb4ee2f1d495.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/0ccfef2be10f1c1e001414e04b1b9e7b70f33c237a726e4a4c7ffb4ee2f1d495.jpg)

![3d4ca3fcc3b4cc7a39a29f6bd5a0497f389453b26e55b25da7e0bdaab9be6be7.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/3d4ca3fcc3b4cc7a39a29f6bd5a0497f389453b26e55b25da7e0bdaab9be6be7.jpg)

![4fcb19ca0db2438b98b0a38d079a22279b5d9c696e77471d637cd290b02ab119.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/4fcb19ca0db2438b98b0a38d079a22279b5d9c696e77471d637cd290b02ab119.jpg)

![604940c4dcd55cad3935d66c2a11555f97852526415701c2d5adb4ba3eadb09d.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/604940c4dcd55cad3935d66c2a11555f97852526415701c2d5adb4ba3eadb09d.jpg)

![61c73d4a7a2235f649476b91f94801832476691b859c59725633a1cb7f28cfac.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/61c73d4a7a2235f649476b91f94801832476691b859c59725633a1cb7f28cfac.jpg)

![925c4c54faaeb7e2494481aceaead00dd6bf307f7830e625ed28996b049a8c0c.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/925c4c54faaeb7e2494481aceaead00dd6bf307f7830e625ed28996b049a8c0c.jpg)

![b2bac572c9f452c11e3d13c0016c7e4baee501dc57e1bfabc20393532fba97e4.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/b2bac572c9f452c11e3d13c0016c7e4baee501dc57e1bfabc20393532fba97e4.jpg)

![c9b06ac7de757e7292ccf4ec5f9f7c90206242821c5c12834f9614ede866d880.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/c9b06ac7de757e7292ccf4ec5f9f7c90206242821c5c12834f9614ede866d880.jpg)

![cf6efcfb1a07ebf367acc4c4114fad9e2ca255959209fa49fe536506c686e54a.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/cf6efcfb1a07ebf367acc4c4114fad9e2ca255959209fa49fe536506c686e54a.jpg)

![df3fb8ae479b08d6389aa076566806c44cd3a74b60a55e71f581083106b32b21.jpg](../icml_results/3176_TiME_%20Test-Time%20Mixture-of-Experts%20Routing%20via%20Asymmetric%20CO-Optimal%20Transport%20for%20Continual%20Test-Ti/tables/df3fb8ae479b08d6389aa076566806c44cd3a74b60a55e71f581083106b32b21.jpg)

<a id="paper-3138"></a>

## Estimating Correlation Clustering Cost in Node-Arrival Stream


### Images

![38878fc182ff4e0eebf9ef2abef8c17914ec010db02dff8f65af5a5f3dccf196.jpg](../icml_results/3177_Estimating%20Correlation%20Clustering%20Cost%20in%20Node-Arrival%20Stream/images/38878fc182ff4e0eebf9ef2abef8c17914ec010db02dff8f65af5a5f3dccf196.jpg)

![5ce107b6492f5893cd85a26527cef640078a868db8b40d423e9cd1cb78e77b0a.jpg](../icml_results/3177_Estimating%20Correlation%20Clustering%20Cost%20in%20Node-Arrival%20Stream/images/5ce107b6492f5893cd85a26527cef640078a868db8b40d423e9cd1cb78e77b0a.jpg)

![8a2f7a59015a93f39b75db5eec2492d45904ecbbe5b716cc002900596c928af5.jpg](../icml_results/3177_Estimating%20Correlation%20Clustering%20Cost%20in%20Node-Arrival%20Stream/images/8a2f7a59015a93f39b75db5eec2492d45904ecbbe5b716cc002900596c928af5.jpg)

![c152f13c2d4613a7fc70ba182d7d29dca0fa4c64d049ad0398e087a5ff9d80b8.jpg](../icml_results/3177_Estimating%20Correlation%20Clustering%20Cost%20in%20Node-Arrival%20Stream/images/c152f13c2d4613a7fc70ba182d7d29dca0fa4c64d049ad0398e087a5ff9d80b8.jpg)

### Tables

![c46cb537c870857edfb4552ac95680a9212dd04e69daa6e46ca4fbd766962e48.jpg](../icml_results/3177_Estimating%20Correlation%20Clustering%20Cost%20in%20Node-Arrival%20Stream/tables/c46cb537c870857edfb4552ac95680a9212dd04e69daa6e46ca4fbd766962e48.jpg)

<a id="paper-3139"></a>

## Pseudo-Mallows for Efficient Probabilistic Preference Learning


### Images

![1a3cf2a7126f216421c92128c8f79dbda8ce15683f63017e7cf9f0d3ed65f59d.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/1a3cf2a7126f216421c92128c8f79dbda8ce15683f63017e7cf9f0d3ed65f59d.jpg)

![3da1399165e6ed7c21c2c761e8f20fe5cb5277c1a320cb328075b79392c90aad.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/3da1399165e6ed7c21c2c761e8f20fe5cb5277c1a320cb328075b79392c90aad.jpg)

![4ecf6e423cf1b442b14f84b4a36d39e7510c2fed4d926bf30246fc81169d8f01.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/4ecf6e423cf1b442b14f84b4a36d39e7510c2fed4d926bf30246fc81169d8f01.jpg)

![7ed2f463853f95d93e8fd34c8b85b3caf3f061e73103b435bd50a0948a6933cd.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/7ed2f463853f95d93e8fd34c8b85b3caf3f061e73103b435bd50a0948a6933cd.jpg)

![8a6d5fd60256f230e11e0e6bbadf343c14c31507df91e78749826e2bf35c8946.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/8a6d5fd60256f230e11e0e6bbadf343c14c31507df91e78749826e2bf35c8946.jpg)

![971cdbf3dd8e472f364e3dd80b0d2d9f7cb577f69314fa66975bc39891567740.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/971cdbf3dd8e472f364e3dd80b0d2d9f7cb577f69314fa66975bc39891567740.jpg)

![a3a964cfa17956ecd88680e67f10060d9afc974051c3c30994db06bdad2b4e27.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/a3a964cfa17956ecd88680e67f10060d9afc974051c3c30994db06bdad2b4e27.jpg)

![b197932f705bc7aaffd40ea8d080eb1e4d74b98cc7456d06b3e4f3d99f1efb34.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/b197932f705bc7aaffd40ea8d080eb1e4d74b98cc7456d06b3e4f3d99f1efb34.jpg)

![b310e77cfb647e4ebcdb55be54392ec9fea2ebf46680af5219ce8b721e602d45.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/b310e77cfb647e4ebcdb55be54392ec9fea2ebf46680af5219ce8b721e602d45.jpg)

![bf0c6eae92a754dc4516e8e553a55b2cc04ff0b1708e1356401e63f9ebf13680.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/bf0c6eae92a754dc4516e8e553a55b2cc04ff0b1708e1356401e63f9ebf13680.jpg)

![c458b641edcb5ac979c5615a9da534860709981fe30f6f8fd60a2f62f167e33f.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/c458b641edcb5ac979c5615a9da534860709981fe30f6f8fd60a2f62f167e33f.jpg)

![e523b62b79cd27c3871829feec7775cbd397fca86ae38cc71c9b84034c10e67e.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/images/e523b62b79cd27c3871829feec7775cbd397fca86ae38cc71c9b84034c10e67e.jpg)

### Tables

![9e47198786a7c4205b0a6a8e7ffc25b3f6264409c83f5eca8fe9b42d7eea50cc.jpg](../icml_results/3178_Pseudo-Mallows%20for%20Efficient%20Probabilistic%20Preference%20Learning/tables/9e47198786a7c4205b0a6a8e7ffc25b3f6264409c83f5eca8fe9b42d7eea50cc.jpg)

<a id="paper-3140"></a>

## Scaling Inference-Time Computation via Opponent Simulation: Enabling Online Strategic Adaptation in Repeated Negotiation


### Images

![03c4abd17a38a701df48b110a366ef5cd0965392a56e91f7715dd879d7eba89e.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/03c4abd17a38a701df48b110a366ef5cd0965392a56e91f7715dd879d7eba89e.jpg)

![0eab73893314cb4bbbaef3751adb4fbff2a193585c76aadf4e1e9b06ef03d77d.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/0eab73893314cb4bbbaef3751adb4fbff2a193585c76aadf4e1e9b06ef03d77d.jpg)

![126fd48479af71611c4b050d04f42405c365e3b5e4df8cf2302d5697a1943d07.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/126fd48479af71611c4b050d04f42405c365e3b5e4df8cf2302d5697a1943d07.jpg)

![140a64493d922625b40f730690df5bece7c534f9e3d27f0407bb8b5c52735b82.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/140a64493d922625b40f730690df5bece7c534f9e3d27f0407bb8b5c52735b82.jpg)

![2af8b4991a75b278ffaf224aedcb6c018f1930394cd68c0cdb55f39fd84850d6.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/2af8b4991a75b278ffaf224aedcb6c018f1930394cd68c0cdb55f39fd84850d6.jpg)

![36275c43b6c8351e16caee8dcb22313da5c5a7f77eb0ca957d88f9b3cc585587.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/36275c43b6c8351e16caee8dcb22313da5c5a7f77eb0ca957d88f9b3cc585587.jpg)

![5049493e9aa6e58d1bf9607a731b1ed09d4d00b9b9c726ca754e34bee74d548c.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/5049493e9aa6e58d1bf9607a731b1ed09d4d00b9b9c726ca754e34bee74d548c.jpg)

![55ad2722291f79c11ff083a3b5b0d0e4533673138107f24eee07c3c9b666f5b4.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/55ad2722291f79c11ff083a3b5b0d0e4533673138107f24eee07c3c9b666f5b4.jpg)

![9194bf59955dca67d92d95ab3e8b08a6b591409e8c8311d532b6b3d1f185d8e0.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/9194bf59955dca67d92d95ab3e8b08a6b591409e8c8311d532b6b3d1f185d8e0.jpg)

![ad3eb020d10880681ede35a3f245316c21c3c3f2d2229bed362811c73080505c.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/ad3eb020d10880681ede35a3f245316c21c3c3f2d2229bed362811c73080505c.jpg)

![d946bc8ab5473ebe283b6b976f15c0179dfb6a6a2a9f0569e01d23cf091cc9c6.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/d946bc8ab5473ebe283b6b976f15c0179dfb6a6a2a9f0569e01d23cf091cc9c6.jpg)

![ee039b13b1dbd8a37c5b13c7a26123ec24308500182e29ab315c72623eab306d.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/ee039b13b1dbd8a37c5b13c7a26123ec24308500182e29ab315c72623eab306d.jpg)

![ee96984848b8160e83dcac3860209645ae4fa9e256e54552bf1b4a850373c8e5.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/ee96984848b8160e83dcac3860209645ae4fa9e256e54552bf1b4a850373c8e5.jpg)

![f11ba104de8431c9a02f0f80593c28be06e620bf9b0919c21c222be6dcca08e9.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/f11ba104de8431c9a02f0f80593c28be06e620bf9b0919c21c222be6dcca08e9.jpg)

![f256b882789430b09790fda18b35389c7f04ae5aca76f9ad8e1b5a807e665ef5.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/f256b882789430b09790fda18b35389c7f04ae5aca76f9ad8e1b5a807e665ef5.jpg)

![f2586c64b56382ba1840728d532829d320a2ab1f3bc7b779a664d8572dee6d2b.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/images/f2586c64b56382ba1840728d532829d320a2ab1f3bc7b779a664d8572dee6d2b.jpg)

### Tables

![22d711c633d551fab1d193905b62a93639070f68ed4d36ca8b61655568eb0040.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/tables/22d711c633d551fab1d193905b62a93639070f68ed4d36ca8b61655568eb0040.jpg)

![2937d43b3cf4362bb347810cedea73fa2639454b935169db22a78633f72d6201.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/tables/2937d43b3cf4362bb347810cedea73fa2639454b935169db22a78633f72d6201.jpg)

![8fe3659deaef55182df0ccca9b313e619dfbe9e7bdaa8c5e44c6143f8740c411.jpg](../icml_results/3179_Scaling%20Inference-Time%20Computation%20via%20Opponent%20Simulation_%20Enabling%20Online%20Strategic%20Adaptation%20in%20/tables/8fe3659deaef55182df0ccca9b313e619dfbe9e7bdaa8c5e44c6143f8740c411.jpg)

<a id="paper-3141"></a>

## HVAE: Hyperbolic Variational Autoencoder For Flexible Knowledge Transfer Across Multiple Domains


### Images

![2a3badf25e71c97a2b10036d1dd251692add0d8aefe97e00645c7ccd9e985918.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/2a3badf25e71c97a2b10036d1dd251692add0d8aefe97e00645c7ccd9e985918.jpg)

![313b5be48e0cc13d956b5107fe8c03b87b08c0b402f5bf620cb47187ffa95280.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/313b5be48e0cc13d956b5107fe8c03b87b08c0b402f5bf620cb47187ffa95280.jpg)

![3ebc268c8bfcd0ae58bc748656751cd2f5d410d4aa962208a4ac48df7a9523f3.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/3ebc268c8bfcd0ae58bc748656751cd2f5d410d4aa962208a4ac48df7a9523f3.jpg)

![57be6edccd6a8e296090228e3180c57610f29f987608707e34d8fa9b12b43c11.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/57be6edccd6a8e296090228e3180c57610f29f987608707e34d8fa9b12b43c11.jpg)

![91cffa917275587827e3becaa1df287863a0fb9beb542cc3028c7d591d55e577.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/91cffa917275587827e3becaa1df287863a0fb9beb542cc3028c7d591d55e577.jpg)

![b34be3135095202dd59442bc1f1df97dd55970cb91560037871ae479b919b218.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/b34be3135095202dd59442bc1f1df97dd55970cb91560037871ae479b919b218.jpg)

![f4c1a13fa6e4406b9f4ed7394338aa9aaf7d75aefcecaa43f6f6e5d6794ea2ac.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/f4c1a13fa6e4406b9f4ed7394338aa9aaf7d75aefcecaa43f6f6e5d6794ea2ac.jpg)

![f77c45d6340c5e98086d4ed9586e2bd422470551ae07d59957ba374119d6f089.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/images/f77c45d6340c5e98086d4ed9586e2bd422470551ae07d59957ba374119d6f089.jpg)

### Tables

![13f763ae93d6dec92c54f5c0517ccec82b1962b800ed6995de6bad382a56d263.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/tables/13f763ae93d6dec92c54f5c0517ccec82b1962b800ed6995de6bad382a56d263.jpg)

![513e17145576d5155e3a4e7c41f554a9e727cee51d1e643878f2fb385e371574.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/tables/513e17145576d5155e3a4e7c41f554a9e727cee51d1e643878f2fb385e371574.jpg)

![c06b74351777733de579cff09e1708a0f534db8e20c0f99566fc26cf8ed7027a.jpg](../icml_results/3180_HVAE_%20Hyperbolic%20Variational%20Autoencoder%20For%20Flexible%20Knowledge%20Transfer%20Across%20Multiple%20Domains/tables/c06b74351777733de579cff09e1708a0f534db8e20c0f99566fc26cf8ed7027a.jpg)

<a id="paper-3142"></a>

## Condition Number Based Low-Bit Quantization for Image Super-Resolution


### Images

![081b5fbee69959458cd6a36aa58b0957d2fc382841fd73becbc6300ec1111d39.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/081b5fbee69959458cd6a36aa58b0957d2fc382841fd73becbc6300ec1111d39.jpg)

![21c3e991533381a4b4a1e163e53bcd5be75db6c73f45474d0dedec91024774c8.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/21c3e991533381a4b4a1e163e53bcd5be75db6c73f45474d0dedec91024774c8.jpg)

![2802a4fbd224d59637a875cb8392a348c09cfca4c8cef43da84285316d28b4c5.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/2802a4fbd224d59637a875cb8392a348c09cfca4c8cef43da84285316d28b4c5.jpg)

![75cda3b9c928a47519cc16cbbc42faaf33356feef85232e108a07e234f3e1667.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/75cda3b9c928a47519cc16cbbc42faaf33356feef85232e108a07e234f3e1667.jpg)

![79a37bc8ad0913122d577f8f93ae4c052f4a3bfc464c29469f37a52a23ae0b47.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/79a37bc8ad0913122d577f8f93ae4c052f4a3bfc464c29469f37a52a23ae0b47.jpg)

![84bd0659cbc250fd726979bb168e7c5cb2bb692978d4a0c4d180983a3062e307.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/84bd0659cbc250fd726979bb168e7c5cb2bb692978d4a0c4d180983a3062e307.jpg)

![d978bf2e77757c592dae6d9948cb96450a25b320343f83e8a4de574fe124b8cc.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/d978bf2e77757c592dae6d9948cb96450a25b320343f83e8a4de574fe124b8cc.jpg)

![ed5f525fdadec3ceb419548145b8bba86f336f0a65d9cb33873016994c9cd8b3.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/ed5f525fdadec3ceb419548145b8bba86f336f0a65d9cb33873016994c9cd8b3.jpg)

![fdbec33cecd6c4f21bd9395a66e673fa284e566438be6f2de575de8d84a89761.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/images/fdbec33cecd6c4f21bd9395a66e673fa284e566438be6f2de575de8d84a89761.jpg)

### Tables

![1bca10c4b1417aaaa4994fb3c72d0072ec024ccdb0eb374f34e580604a03efc7.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/1bca10c4b1417aaaa4994fb3c72d0072ec024ccdb0eb374f34e580604a03efc7.jpg)

![84bb7d305f4864fd9542ad45507bd7c2b3596b3d1dc6eebf46f4312b33de14a0.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/84bb7d305f4864fd9542ad45507bd7c2b3596b3d1dc6eebf46f4312b33de14a0.jpg)

![a5156bff2e29e8e2783945ace6505b5a8bfb9c91d0baded289e0e59fc00c6a4a.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/a5156bff2e29e8e2783945ace6505b5a8bfb9c91d0baded289e0e59fc00c6a4a.jpg)

![c0fb400a9826416c1371174d7d8a0fe5137b6c2c3393818197c2a620706bfb32.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/c0fb400a9826416c1371174d7d8a0fe5137b6c2c3393818197c2a620706bfb32.jpg)

![df7ece4c0de3a28f45558da48ed5ac0983073243ea213ad4042c5b9910a2d291.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/df7ece4c0de3a28f45558da48ed5ac0983073243ea213ad4042c5b9910a2d291.jpg)

![ee3f7ba47c62e2dbaf8d4180bce7a9e3121b0a9d450fa525d129bf7dc6825a81.jpg](../icml_results/3181_Condition%20Number%20Based%20Low-Bit%20Quantization%20for%20Image%20Super-Resolution/tables/ee3f7ba47c62e2dbaf8d4180bce7a9e3121b0a9d450fa525d129bf7dc6825a81.jpg)

<a id="paper-3143"></a>

## STAR-VAE: Structured Topology-Aware Regularization for Audio Reconstruction and Generation


### Images

![04c6bb6e4f1551e74e45ff0d2aad222e4bf9c0036154f025868291b11c7af21f.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/images/04c6bb6e4f1551e74e45ff0d2aad222e4bf9c0036154f025868291b11c7af21f.jpg)

![2feca73c4f454af1b855baf119c904101bafdec9f71c1a8f7de596b318a47f53.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/images/2feca73c4f454af1b855baf119c904101bafdec9f71c1a8f7de596b318a47f53.jpg)

![83083d0184c55ecf1c38b411bbf64ced5a4ea5ac396b4c3b27c97e165d5946e5.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/images/83083d0184c55ecf1c38b411bbf64ced5a4ea5ac396b4c3b27c97e165d5946e5.jpg)

![ce63fdc15fc0baaa1f5150d8b90ee0524d94e7a208970bf5b16f67ec78ea9916.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/images/ce63fdc15fc0baaa1f5150d8b90ee0524d94e7a208970bf5b16f67ec78ea9916.jpg)

![fa33a14939a762c684a06f06cbc6a3fe7fcaa402259a3faa766a9b298ba3f74e.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/images/fa33a14939a762c684a06f06cbc6a3fe7fcaa402259a3faa766a9b298ba3f74e.jpg)

### Tables

![08e0062c2b76c102931bbc06a084f19f8d3807685bf41bdd0899995e2a6260d5.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/08e0062c2b76c102931bbc06a084f19f8d3807685bf41bdd0899995e2a6260d5.jpg)

![0c9d0b03bc3ca8dbd590eaf2d9b7f0ec03a6ec2f839f241c727422e9d073d79b.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/0c9d0b03bc3ca8dbd590eaf2d9b7f0ec03a6ec2f839f241c727422e9d073d79b.jpg)

![0ed0b191a77aa56ea8716423777be5d481d0f265de12fc7160966c8d6cd7f789.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/0ed0b191a77aa56ea8716423777be5d481d0f265de12fc7160966c8d6cd7f789.jpg)

![140f4e8851a6af653906f11972c83ff7a499035fb6a080583d7591001f359216.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/140f4e8851a6af653906f11972c83ff7a499035fb6a080583d7591001f359216.jpg)

![3d8582126bb721ea27a8768971acbf8d9ce258a6e9de2bd9a8a5242a9015b99e.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/3d8582126bb721ea27a8768971acbf8d9ce258a6e9de2bd9a8a5242a9015b99e.jpg)

![3ef54dbe1c24f6120a79dc8bca0d1e42e545e1a2da2ba2920f2c959a53d50f8a.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/3ef54dbe1c24f6120a79dc8bca0d1e42e545e1a2da2ba2920f2c959a53d50f8a.jpg)

![ac1e0e3773fedd2e1d981d2719a08de6da3e48fb4a0537a065ffd5904644dcff.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/ac1e0e3773fedd2e1d981d2719a08de6da3e48fb4a0537a065ffd5904644dcff.jpg)

![d1e92427d07f9de6600f8881b9cde399ead394df44f6ee0b99e3c2c9fce7b0e6.jpg](../icml_results/3182_STAR-VAE_%20Structured%20Topology-Aware%20Regularization%20for%20Audio%20Reconstruction%20and%20Generation/tables/d1e92427d07f9de6600f8881b9cde399ead394df44f6ee0b99e3c2c9fce7b0e6.jpg)

<a id="paper-3144"></a>

## VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos


### Images

![053b2fb882c1d4ec733999bbe640a0ec56f03ffa445807f7ab348990468488d9.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/053b2fb882c1d4ec733999bbe640a0ec56f03ffa445807f7ab348990468488d9.jpg)

![11a66f485ed5cd0ce1c9b463e250895aa6ee19202d0bbb6095adb7eb500ebb4e.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/11a66f485ed5cd0ce1c9b463e250895aa6ee19202d0bbb6095adb7eb500ebb4e.jpg)

![175608fbda102b7497dbb4700a2eca65af8a521ade40cbd0d411234d14811989.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/175608fbda102b7497dbb4700a2eca65af8a521ade40cbd0d411234d14811989.jpg)

![20643197defe578364a4ab0a707804bd0568d36c757dbc244928c8e7872af79b.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/20643197defe578364a4ab0a707804bd0568d36c757dbc244928c8e7872af79b.jpg)

![28003b31314423c1160edef481a458f1d4adb57ae0fb5a85a8cb5df7f4e7d965.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/28003b31314423c1160edef481a458f1d4adb57ae0fb5a85a8cb5df7f4e7d965.jpg)

![31234844b8b107a8a197d765524943a86acd9fd49f80ca3aa56808c69cd7f74e.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/31234844b8b107a8a197d765524943a86acd9fd49f80ca3aa56808c69cd7f74e.jpg)

![3cb2e949561584f2a15a29eb3ab49aded492401753595a9ca1195158c062ddbb.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/3cb2e949561584f2a15a29eb3ab49aded492401753595a9ca1195158c062ddbb.jpg)

![3e82737ba62a17de6ac0420630df2371dcc9678321147de227c9366e5c0f6dfe.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/3e82737ba62a17de6ac0420630df2371dcc9678321147de227c9366e5c0f6dfe.jpg)

![3e95093ef87a710171873365cc11e5d1919618e7682705433be652361278cf2b.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/3e95093ef87a710171873365cc11e5d1919618e7682705433be652361278cf2b.jpg)

![40cefc04a66b80209760de988b970fc4cf3af3ba34454acfa0008f90e05516c3.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/40cefc04a66b80209760de988b970fc4cf3af3ba34454acfa0008f90e05516c3.jpg)

![57c1bb4bc5328f0a3d214c2e3f481df28e5ea131eeb5b5581a25e8fea9415deb.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/57c1bb4bc5328f0a3d214c2e3f481df28e5ea131eeb5b5581a25e8fea9415deb.jpg)

![5cc949c2d58a84741aa1f3d422a9bf278a70f4348e7fc5179905b5e73d9886d5.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/5cc949c2d58a84741aa1f3d422a9bf278a70f4348e7fc5179905b5e73d9886d5.jpg)

![5dbe69f79e353551d7c2e6f24a289fda48af9c94301dca4f57c4d8ba68f4690b.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/5dbe69f79e353551d7c2e6f24a289fda48af9c94301dca4f57c4d8ba68f4690b.jpg)

![61c7444f8db20477381d687c318d46b17f63c183090fd7fac3dd790da4f68b7b.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/61c7444f8db20477381d687c318d46b17f63c183090fd7fac3dd790da4f68b7b.jpg)

![62b19d54edbce317c7576ed941ee62e52de7fb3af852421f9b3db6c15aa843ff.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/62b19d54edbce317c7576ed941ee62e52de7fb3af852421f9b3db6c15aa843ff.jpg)

![638487251f14e604692a3d23a3af6bc655bd6062dcc364982b135feac2456c18.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/638487251f14e604692a3d23a3af6bc655bd6062dcc364982b135feac2456c18.jpg)

![68e517d7dc0322fbee1eb58d91ed15806d0418a88008d9fcdb6349511eeaddb2.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/68e517d7dc0322fbee1eb58d91ed15806d0418a88008d9fcdb6349511eeaddb2.jpg)

![7d6cd6b5571171bd4be39aed9f4879e9a8be403243cd8a2ab1f876b289118e16.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/7d6cd6b5571171bd4be39aed9f4879e9a8be403243cd8a2ab1f876b289118e16.jpg)

![804739a07ad4efb4b2ff24acfcd9247fcf8b4d0e486f974d63466096313d2173.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/804739a07ad4efb4b2ff24acfcd9247fcf8b4d0e486f974d63466096313d2173.jpg)

![a37546d1aa4a8fca7a7c8f2095bcde36a443775c7616456a1ffaaa77d2cec34b.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/a37546d1aa4a8fca7a7c8f2095bcde36a443775c7616456a1ffaaa77d2cec34b.jpg)

![c2272f9e867b579b8d55c5b6715619a4cb92da9ae0cf5555f87fc2e25949157f.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/c2272f9e867b579b8d55c5b6715619a4cb92da9ae0cf5555f87fc2e25949157f.jpg)

![e2fa9cf2a4c94ddd12540465c2d46c54adb9ab193c09cad7b2125f8271d044d3.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/e2fa9cf2a4c94ddd12540465c2d46c54adb9ab193c09cad7b2125f8271d044d3.jpg)

![ebc32d2756025d192b91cbad876bbf3c368f7a3ba9bde7c249a7dcf4f93a9ff1.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/ebc32d2756025d192b91cbad876bbf3c368f7a3ba9bde7c249a7dcf4f93a9ff1.jpg)

![fe1163ca5d2fa685b9fd4b5eb32a637e81629f0aa74d7618c8f2d4570dd44204.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/images/fe1163ca5d2fa685b9fd4b5eb32a637e81629f0aa74d7618c8f2d4570dd44204.jpg)

### Tables

![20174aa6be43a2b16c91b62a1329f7a90338adf7198d1b7ccb2fe8752314c739.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/20174aa6be43a2b16c91b62a1329f7a90338adf7198d1b7ccb2fe8752314c739.jpg)

![5adccd5d7b3dbcbe7762c6745b2f0e2debc466b5c14f88289b46e3772888f8fe.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/5adccd5d7b3dbcbe7762c6745b2f0e2debc466b5c14f88289b46e3772888f8fe.jpg)

![5bce4ea3e365af54c8da3da4175f60d9580b858e1a99517c4425aaf7fa6a75f8.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/5bce4ea3e365af54c8da3da4175f60d9580b858e1a99517c4425aaf7fa6a75f8.jpg)

![92f51b6f82b614eee862652867a7134327bbf3e7adc387e804e7e9d8837b8415.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/92f51b6f82b614eee862652867a7134327bbf3e7adc387e804e7e9d8837b8415.jpg)

![a39636b8f81c605f8ca16f40dc314bb6cfc50333712c8874051018fad7ebbea8.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/a39636b8f81c605f8ca16f40dc314bb6cfc50333712c8874051018fad7ebbea8.jpg)

![badc85a43ec0f73d80aeef4f69b79926f9ffd5567dd292bf0ef5c6e897e365ff.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/badc85a43ec0f73d80aeef4f69b79926f9ffd5567dd292bf0ef5c6e897e365ff.jpg)

![c06c6525b68aea308f59aa53cbbf080052e9478f58bb6f171e360be4fff4004a.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/c06c6525b68aea308f59aa53cbbf080052e9478f58bb6f171e360be4fff4004a.jpg)

![eff7aaa1c1cc1eaed5d87bb3b6566c35cc0c3e4a942aa9c55670b8dacdeddc52.jpg](../icml_results/3183_VideoTemp-o3_%20Harmonizing%20Temporal%20Grounding%20and%20Video%20Understanding%20in%20Agentic%20Thinking-with-Videos/tables/eff7aaa1c1cc1eaed5d87bb3b6566c35cc0c3e4a942aa9c55670b8dacdeddc52.jpg)

<a id="paper-3145"></a>

## SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass


### Images

![028fd822ccbe279dd8d06b09de86faf0d0ded32e3cef15d83374fd0c0546735b.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/028fd822ccbe279dd8d06b09de86faf0d0ded32e3cef15d83374fd0c0546735b.jpg)

![09f9946e049ac1c2df2ae1a461b8e57737fefb1e278cab28e6460ebe220bde24.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/09f9946e049ac1c2df2ae1a461b8e57737fefb1e278cab28e6460ebe220bde24.jpg)

![10003650e11f35a8aa245cd01dd5b47d1e1b08bb8b783ec5cfe598f918fbd679.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/10003650e11f35a8aa245cd01dd5b47d1e1b08bb8b783ec5cfe598f918fbd679.jpg)

![34a626c70d537d7822f5cb38a0e70a4d820c77456410046a99233248fcb91c77.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/34a626c70d537d7822f5cb38a0e70a4d820c77456410046a99233248fcb91c77.jpg)

![3509f22110c8ef6d44fdb7f6d7a00f8fb4f6516ffcdfc77903ac4f910d0c273d.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/3509f22110c8ef6d44fdb7f6d7a00f8fb4f6516ffcdfc77903ac4f910d0c273d.jpg)

![3e02a6fea6957fcab6d313d9349a16faa2754d3c57f2cd00115fb00b734b09d6.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/3e02a6fea6957fcab6d313d9349a16faa2754d3c57f2cd00115fb00b734b09d6.jpg)

![683c3436e2e82ba8ea5f0e9996958b67b2625018ccd3f7113c0fadea7d03ee30.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/683c3436e2e82ba8ea5f0e9996958b67b2625018ccd3f7113c0fadea7d03ee30.jpg)

![89fd113d8deda95233107477bb41258186f1fd2a92f2b6a6b80faad5b4006ed5.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/89fd113d8deda95233107477bb41258186f1fd2a92f2b6a6b80faad5b4006ed5.jpg)

![8d53d909e6ec807ff7730a29ab1f0adc7e3482b840935fefc9cc8794418e6287.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/8d53d909e6ec807ff7730a29ab1f0adc7e3482b840935fefc9cc8794418e6287.jpg)

![92160cd74d2bc2cd3d2b90dbd898d84c45cffdfd7a8f966932e95679808c8f9e.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/92160cd74d2bc2cd3d2b90dbd898d84c45cffdfd7a8f966932e95679808c8f9e.jpg)

![9289cdeacb3de9d615d7e06c6fcd6e91ce589d958a52dd6a97547ee1ed0280df.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/9289cdeacb3de9d615d7e06c6fcd6e91ce589d958a52dd6a97547ee1ed0280df.jpg)

![963bfcce690a0ebb1f1fa847e11258fc9844cd612ba10f6f3f216b313c10fdc9.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/963bfcce690a0ebb1f1fa847e11258fc9844cd612ba10f6f3f216b313c10fdc9.jpg)

![adf29b4375e32adbff02b1d8f8e5c9b6ad956c26f8753742b01c977d1a774819.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/adf29b4375e32adbff02b1d8f8e5c9b6ad956c26f8753742b01c977d1a774819.jpg)

![ae3cffb8a9ee3cce18b6a2d17fb5a3499d621fbaf283aa370a4ebdc3d4aae7f4.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/ae3cffb8a9ee3cce18b6a2d17fb5a3499d621fbaf283aa370a4ebdc3d4aae7f4.jpg)

![b070ed3acf7c8e5fd7dde747e87be24461dae287896858753b6c911a7c57a7cc.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/b070ed3acf7c8e5fd7dde747e87be24461dae287896858753b6c911a7c57a7cc.jpg)

![bd4e0f82809ac62ad5f973a7cf03b8a3a27fd51c61bc55a2d2646d572ea52d92.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/bd4e0f82809ac62ad5f973a7cf03b8a3a27fd51c61bc55a2d2646d572ea52d92.jpg)

![bf8f90f31cc11ce2795c6cc3477805adb03e145755817611e154b14ae7d3e853.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/bf8f90f31cc11ce2795c6cc3477805adb03e145755817611e154b14ae7d3e853.jpg)

![df42a8a941e43721d9be083461f148dd824f4d13a9ac6c21672c5d43e539b7c5.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/images/df42a8a941e43721d9be083461f148dd824f4d13a9ac6c21672c5d43e539b7c5.jpg)

### Tables

![1ecb42c1a09a54c6ae99ee7d03abf4389e3da7045c98e679762b10022e9f3c84.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/1ecb42c1a09a54c6ae99ee7d03abf4389e3da7045c98e679762b10022e9f3c84.jpg)

![1f7144b3447dbe2139305faeaa065b51a9c21ad3393e0258a5b7f4c06e7941dd.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/1f7144b3447dbe2139305faeaa065b51a9c21ad3393e0258a5b7f4c06e7941dd.jpg)

![23579b6b271f974096cfc6da0482e1e955a5909933c931e3af9e1efee589c4cd.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/23579b6b271f974096cfc6da0482e1e955a5909933c931e3af9e1efee589c4cd.jpg)

![33d3c7526989a2a5a8029b3ffb9c43af0b73121c7514bff05c001352b944f944.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/33d3c7526989a2a5a8029b3ffb9c43af0b73121c7514bff05c001352b944f944.jpg)

![3964348076e1bde75a4550fad724a9e8d078942c75dd35bf3375a94fb27bb253.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/3964348076e1bde75a4550fad724a9e8d078942c75dd35bf3375a94fb27bb253.jpg)

![54cb9d3c8c90616a0e101b8f82005fc206b0d21294bcf70c96e9efe9b09c0e00.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/54cb9d3c8c90616a0e101b8f82005fc206b0d21294bcf70c96e9efe9b09c0e00.jpg)

![8acc4d5573f8d200209970a27b38f991ce6f396c3c4e83d7a73f3a9c9f6a61f6.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/8acc4d5573f8d200209970a27b38f991ce6f396c3c4e83d7a73f3a9c9f6a61f6.jpg)

![8afbe6923806e393f625b8d4fa63d7f304b3fa21e2cb9b2343bc1aa115e0f714.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/8afbe6923806e393f625b8d4fa63d7f304b3fa21e2cb9b2343bc1aa115e0f714.jpg)

![bfd73d6b499773cd3cfc2f31a2ed6ef93a36b85238fee93381b88378576b5d99.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/bfd73d6b499773cd3cfc2f31a2ed6ef93a36b85238fee93381b88378576b5d99.jpg)

![ce8944d35eee7659ffc51051abc0efa4478546689a3c3c9e86ee563cefccab1e.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/ce8944d35eee7659ffc51051abc0efa4478546689a3c3c9e86ee563cefccab1e.jpg)

![d47beb59d98e4e87afb1b68ac5f160f557d111ff6853ffd6bcb9f3c059b2b720.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/d47beb59d98e4e87afb1b68ac5f160f557d111ff6853ffd6bcb9f3c059b2b720.jpg)

![dcbd93a6d588bd0496980c6dc0ced46d66c0f30465204e5f68d4556e3a0b5dca.jpg](../icml_results/3184_SHINE_%20A%20Scalable%20In-Context%20Hypernetwork%20for%20Mapping%20Context%20to%20LoRA%20in%20a%20Single%20Pass/tables/dcbd93a6d588bd0496980c6dc0ced46d66c0f30465204e5f68d4556e3a0b5dca.jpg)

<a id="paper-3146"></a>

## Opt-Verifier: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification


### Images

![1eb71b7002a98040d90a65fb499d174e2ded3021d0b2983b5cda1bb209ac71df.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/1eb71b7002a98040d90a65fb499d174e2ded3021d0b2983b5cda1bb209ac71df.jpg)

![3c0dd020f27273090e466c62fcee0ec8fc46c5d3cf76a7ddfe741a8330e5b855.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/3c0dd020f27273090e466c62fcee0ec8fc46c5d3cf76a7ddfe741a8330e5b855.jpg)

![6c130fc8b2636ce4d11d0ca11c0d32d15ed2f9b19ee0ded83afe2b64f83f6ba8.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/6c130fc8b2636ce4d11d0ca11c0d32d15ed2f9b19ee0ded83afe2b64f83f6ba8.jpg)

![6f6695c666c53fb0ffdb211824cefce79cc1ce78cf12aacb43757fc3bd4090c4.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/6f6695c666c53fb0ffdb211824cefce79cc1ce78cf12aacb43757fc3bd4090c4.jpg)

![9068158bc189f51464c635ecc65dbefeb69b073e06c18a227c2df9fc0e6ad972.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/9068158bc189f51464c635ecc65dbefeb69b073e06c18a227c2df9fc0e6ad972.jpg)

![a4073ac38fb41e7165f68f01497bc4550977cebed733f2f8adf79c149cf78bab.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/a4073ac38fb41e7165f68f01497bc4550977cebed733f2f8adf79c149cf78bab.jpg)

![c14b286ffd676a02a4774974ef7e201c5c3eab3094dd4a31bb2cfd527351562b.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/c14b286ffd676a02a4774974ef7e201c5c3eab3094dd4a31bb2cfd527351562b.jpg)

![c7741e1f79596bf0841cb0ae0f7cfc5e1abb3708866204ca7c1583934e1fc949.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/c7741e1f79596bf0841cb0ae0f7cfc5e1abb3708866204ca7c1583934e1fc949.jpg)

![f7168db18472b86151de13b767dbbce999c95b836ad94eb280333f94925b4b8f.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/images/f7168db18472b86151de13b767dbbce999c95b836ad94eb280333f94925b4b8f.jpg)

### Tables

![06c22897e7c6465c5a3bd140fe69be6d6ffd145b1b89fa62fcd63ba366cf0bbd.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/06c22897e7c6465c5a3bd140fe69be6d6ffd145b1b89fa62fcd63ba366cf0bbd.jpg)

![0f24e39ed7c0f2e4e03978cb2cdc740035d3bb5bb1fdb36539619803ec01133f.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/0f24e39ed7c0f2e4e03978cb2cdc740035d3bb5bb1fdb36539619803ec01133f.jpg)

![3412386f93049bf4db36bc7281e2915158a5cfdcf289179b849aeef1a429f02f.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/3412386f93049bf4db36bc7281e2915158a5cfdcf289179b849aeef1a429f02f.jpg)

![37b4167fc2f8dbc459db358485b55f6d87b8d2b85ad2469e6fc8ec5da71fd77d.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/37b4167fc2f8dbc459db358485b55f6d87b8d2b85ad2469e6fc8ec5da71fd77d.jpg)

![5c4b5422c54e7179c40d03c6158e3c564b477e8a8bdb092faab744bb906022cc.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/5c4b5422c54e7179c40d03c6158e3c564b477e8a8bdb092faab744bb906022cc.jpg)

![65386acfa9b9cffebcec5175a9673fbc5d135dc13b5b6ef90fef36b60b1d792f.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/65386acfa9b9cffebcec5175a9673fbc5d135dc13b5b6ef90fef36b60b1d792f.jpg)

![681b62f3504ccf43de4142edf29cab5fba184c2427e2e953cc5539231f8f5373.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/681b62f3504ccf43de4142edf29cab5fba184c2427e2e953cc5539231f8f5373.jpg)

![7b9fb1df6b9877581482fa5f20ff22a632d0f1c6dd6284cd578e090f430fa7d7.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/7b9fb1df6b9877581482fa5f20ff22a632d0f1c6dd6284cd578e090f430fa7d7.jpg)

![a4484396d6e63e15594d6210ecfc519be06b0a051e9a6b3ff4abb55119f4befa.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/a4484396d6e63e15594d6210ecfc519be06b0a051e9a6b3ff4abb55119f4befa.jpg)

![a9fb6ff99fda8ccdb229c67194c9d1311d1303f59ab5f5723d5e17094f73e991.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/a9fb6ff99fda8ccdb229c67194c9d1311d1303f59ab5f5723d5e17094f73e991.jpg)

![b81faadd07ac19456ad6a7cde6562e75bf859c676516b65f9a73767aa5384b26.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/b81faadd07ac19456ad6a7cde6562e75bf859c676516b65f9a73767aa5384b26.jpg)

![c4bdf2f8f2c37213790ddb8ce316bff47fff29cd57c179ad24d6d19b566a3e05.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/c4bdf2f8f2c37213790ddb8ce316bff47fff29cd57c179ad24d6d19b566a3e05.jpg)

![e3f2d3c9f37a7a490a69c43429bc2cccbe672ccbe942338a670ec711b4dbb60e.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/e3f2d3c9f37a7a490a69c43429bc2cccbe672ccbe942338a670ec711b4dbb60e.jpg)

![e712e9f1d9dad43d7ec62acf6ee354a40aea675ce4e7cb28a5da806f9d2f17ed.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/e712e9f1d9dad43d7ec62acf6ee354a40aea675ce4e7cb28a5da806f9d2f17ed.jpg)

![e95d37b63bd6c4c9e04b0e89834bd764d0a26f3f6fa90aab0bc5209345f95558.jpg](../icml_results/3185_Opt-Verifier_%20Unleashing%20the%20Power%20of%20LLMs%20for%20Optimization%20Modeling%20via%20Dual-Side%20Verification/tables/e95d37b63bd6c4c9e04b0e89834bd764d0a26f3f6fa90aab0bc5209345f95558.jpg)

<a id="paper-3147"></a>

## AnalogVerifier: A Neuro-Symbolic Framework for Analog Circuit Verification


### Images

![04ceec6cc27ae8684fe68ec101204cbc0d59557d92d523a24fcc8654748fe80c.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/04ceec6cc27ae8684fe68ec101204cbc0d59557d92d523a24fcc8654748fe80c.jpg)

![1263246b92218689ca8396ab987275c0e1acc0a48395037dfda1d299f368a0b1.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/1263246b92218689ca8396ab987275c0e1acc0a48395037dfda1d299f368a0b1.jpg)

![1927cc3bf62abc5a8269a6b319a30bdb64901e294c05ee4a51005ef8f44ae782.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/1927cc3bf62abc5a8269a6b319a30bdb64901e294c05ee4a51005ef8f44ae782.jpg)

![21ba1cca0f37d7c3c961dae2c4967b60a7bfc30abc1b191213d0f713a18c5b0c.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/21ba1cca0f37d7c3c961dae2c4967b60a7bfc30abc1b191213d0f713a18c5b0c.jpg)

![6c8cec49930e204ff7cf959171596c0ef2064f3a954f9602c6f39db0e75c3fea.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/6c8cec49930e204ff7cf959171596c0ef2064f3a954f9602c6f39db0e75c3fea.jpg)

![7dcbdc535832db3fd42a6e794026b8d68aba027a2fc1aee24203f1284cb8bb10.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/7dcbdc535832db3fd42a6e794026b8d68aba027a2fc1aee24203f1284cb8bb10.jpg)

![800325a6e1ecdf0a7d14e06bd94fc39a877358bdabe5038981e860077c8c7ada.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/800325a6e1ecdf0a7d14e06bd94fc39a877358bdabe5038981e860077c8c7ada.jpg)

![8fb40bbe4a7da7c0e7fb61f4dc4f520a9090aa25f0f3cd68cd6f5b53df8ded8d.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/8fb40bbe4a7da7c0e7fb61f4dc4f520a9090aa25f0f3cd68cd6f5b53df8ded8d.jpg)

![93d041031bc9a74ec141cfab3d5a965af955b181244a2a23d3b21f6a32465354.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/93d041031bc9a74ec141cfab3d5a965af955b181244a2a23d3b21f6a32465354.jpg)

![98e5341f7a0ea20593d51b921a87f1da3916ce3b3827ead005c23f7d1bdc35bc.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/98e5341f7a0ea20593d51b921a87f1da3916ce3b3827ead005c23f7d1bdc35bc.jpg)

![9f79c0a4b1cb2cea9586b186f12e33326aaa799de104ef842177e5645225848f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/9f79c0a4b1cb2cea9586b186f12e33326aaa799de104ef842177e5645225848f.jpg)

![a179f6e25e617acb5fcb836ba5fb93fd1ccbb5bb5b421759f892aceac1b87d94.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/a179f6e25e617acb5fcb836ba5fb93fd1ccbb5bb5b421759f892aceac1b87d94.jpg)

![a695f76eeb25e14a27bb1817e8fb979fdee1470db551012169bf2121d15221e5.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/a695f76eeb25e14a27bb1817e8fb979fdee1470db551012169bf2121d15221e5.jpg)

![b71c356a9a577ab83bb26a2237404127e39adceef1c326e0e18f77054d4af81b.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/b71c356a9a577ab83bb26a2237404127e39adceef1c326e0e18f77054d4af81b.jpg)

![d30ea96000936dbfeb64c006b4cdf54fbd0d4da3f3ecee4cf486c5a9fd819a5f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/d30ea96000936dbfeb64c006b4cdf54fbd0d4da3f3ecee4cf486c5a9fd819a5f.jpg)

![f223aceecc53bebdbb3afbe751dceb28ed42cd9b1c636c1a056f84fe9a51a0de.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/f223aceecc53bebdbb3afbe751dceb28ed42cd9b1c636c1a056f84fe9a51a0de.jpg)

![fecfba2fe6f9856da923c203d43ee6441fc4dc1967dfe39622b440109307f9ef.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/images/fecfba2fe6f9856da923c203d43ee6441fc4dc1967dfe39622b440109307f9ef.jpg)

### Tables

![0540d4b4b9bb89f53a0be8a11410012e05aefa3b54d4fac3aecae45841a4d84f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/0540d4b4b9bb89f53a0be8a11410012e05aefa3b54d4fac3aecae45841a4d84f.jpg)

![0e1018daeb299aa44545bfc9197ffa9f392d5a89fbd6d187cad64732725c18d9.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/0e1018daeb299aa44545bfc9197ffa9f392d5a89fbd6d187cad64732725c18d9.jpg)

![14a775b1e29b2664459741615647ecb6a3a7ac9e1312674eee3dae35e3f2c480.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/14a775b1e29b2664459741615647ecb6a3a7ac9e1312674eee3dae35e3f2c480.jpg)

![176eb7ce4a026df65a5918aa78580c8cd7745fb2c9a855d99944ea49c4c17cae.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/176eb7ce4a026df65a5918aa78580c8cd7745fb2c9a855d99944ea49c4c17cae.jpg)

![1842d8b38f669b3f6b2aecd23ee052c21a2dce594bb65f02f23406e77646fd84.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/1842d8b38f669b3f6b2aecd23ee052c21a2dce594bb65f02f23406e77646fd84.jpg)

![18445864beba8bb7c5358c1665225ee460a98e11bb0748d8ece7dab5b7b5aa99.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/18445864beba8bb7c5358c1665225ee460a98e11bb0748d8ece7dab5b7b5aa99.jpg)

![228f929cd988dd00b3fbed94e38517ca4e1dae1bbfc80241856b3d9879f1282f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/228f929cd988dd00b3fbed94e38517ca4e1dae1bbfc80241856b3d9879f1282f.jpg)

![25eac988184111e864836a9a4cb6088a0f8556f14b03f21cddab329898633da6.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/25eac988184111e864836a9a4cb6088a0f8556f14b03f21cddab329898633da6.jpg)

![2c96406047f29fd058f27767a8a228c3a0d7152128dd348fd60c987df9f1123a.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/2c96406047f29fd058f27767a8a228c3a0d7152128dd348fd60c987df9f1123a.jpg)

![2d1a91c464535468ea0cc2c4b7e5530587af9c96cc66e9c29f32b76f87e74bd0.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/2d1a91c464535468ea0cc2c4b7e5530587af9c96cc66e9c29f32b76f87e74bd0.jpg)

![3143fa1c85b95526aaccac68539e9474e306ade88041d6fa92218775adee80aa.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/3143fa1c85b95526aaccac68539e9474e306ade88041d6fa92218775adee80aa.jpg)

![36acf44737183060473a417f7cef954a0618dd15e113aec835f43bfe6dff859a.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/36acf44737183060473a417f7cef954a0618dd15e113aec835f43bfe6dff859a.jpg)

![37cc2ecb34e0e894052a10c9fad29e3bf2a11943ee90d6018a489ce28ef2fc58.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/37cc2ecb34e0e894052a10c9fad29e3bf2a11943ee90d6018a489ce28ef2fc58.jpg)

![44bae1b89473e93ce6d5ca028174727dc93091a83fe786329eb253c47d936b9d.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/44bae1b89473e93ce6d5ca028174727dc93091a83fe786329eb253c47d936b9d.jpg)

![476a4c09a138270640d00f4c24ec1231396c383c3ef495ba0b44968f762a4b33.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/476a4c09a138270640d00f4c24ec1231396c383c3ef495ba0b44968f762a4b33.jpg)

![49a59bef97512ceb137256f6c06bc52cf86a8b42e6325b8b583678deb40377cb.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/49a59bef97512ceb137256f6c06bc52cf86a8b42e6325b8b583678deb40377cb.jpg)

![4acb54accb1337cd3353ec9cc7630a6699986efc0817521b06116785af06ab26.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/4acb54accb1337cd3353ec9cc7630a6699986efc0817521b06116785af06ab26.jpg)

![4da34ba84580ee67d387ef2b674ac6950d20bab980899e426dd09baaa364c794.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/4da34ba84580ee67d387ef2b674ac6950d20bab980899e426dd09baaa364c794.jpg)

![54f6378a952ce68c347157c056ae1ea9909f9848208b6f9a6c793b879d41912b.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/54f6378a952ce68c347157c056ae1ea9909f9848208b6f9a6c793b879d41912b.jpg)

![5986cbac96518a0aa521eef94e4f494e3d221cf28f18be1ab94486d25a511128.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/5986cbac96518a0aa521eef94e4f494e3d221cf28f18be1ab94486d25a511128.jpg)

![5aaa821b7bc6abbca1a643f840ab76a4ff9a1da62c6ac5f9c344cc3950e13685.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/5aaa821b7bc6abbca1a643f840ab76a4ff9a1da62c6ac5f9c344cc3950e13685.jpg)

![60c9cd1bd2697e9eec27897c7926c5867aff30020d561a6f0f153c98c3a5f7b5.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/60c9cd1bd2697e9eec27897c7926c5867aff30020d561a6f0f153c98c3a5f7b5.jpg)

![6bf2b8307b3c5a7fc183edaef9859788e938366ca3b7414154060c7f831a9410.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/6bf2b8307b3c5a7fc183edaef9859788e938366ca3b7414154060c7f831a9410.jpg)

![6f3da46334128da8835ec9ae43e2f9bac32c7c8e93dd8488affe32e0fd98e2ed.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/6f3da46334128da8835ec9ae43e2f9bac32c7c8e93dd8488affe32e0fd98e2ed.jpg)

![73c800e79aace4534ef2a2a7303caac131c5abe037145e2fba5ded08f04b8aee.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/73c800e79aace4534ef2a2a7303caac131c5abe037145e2fba5ded08f04b8aee.jpg)

![855082dbd5efc289d89f4e655938a06900b1b543f598398a91bbeee704679986.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/855082dbd5efc289d89f4e655938a06900b1b543f598398a91bbeee704679986.jpg)

![95e045d16b98e6afd18ae0d7b86f4c962620d3bdfb7a105fe5e9462af8bff168.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/95e045d16b98e6afd18ae0d7b86f4c962620d3bdfb7a105fe5e9462af8bff168.jpg)

![aa6f2f2f5f8fa9f527d885c316722e55d651aec522064cb4abf291793e35b92d.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/aa6f2f2f5f8fa9f527d885c316722e55d651aec522064cb4abf291793e35b92d.jpg)

![b4193bba89e85efb6373c9c5febf012beb69baba7bf0382988fe2c0fbf92a903.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/b4193bba89e85efb6373c9c5febf012beb69baba7bf0382988fe2c0fbf92a903.jpg)

![b43f75a2e2e583bdc414c313e1ab8b1a8b67c6efac4a7c7309c76caa337998d3.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/b43f75a2e2e583bdc414c313e1ab8b1a8b67c6efac4a7c7309c76caa337998d3.jpg)

![b49b31a7a4da6cfbf89101b1dc38cc136502c133ec2392cacbbcd7c1e94656f3.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/b49b31a7a4da6cfbf89101b1dc38cc136502c133ec2392cacbbcd7c1e94656f3.jpg)

![b7800330fdc63d0a313c7afb4f6e6e6e6cd43511718eec5d9c6366b6d9ecff7d.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/b7800330fdc63d0a313c7afb4f6e6e6e6cd43511718eec5d9c6366b6d9ecff7d.jpg)

![c93f14de8ee9bf99c0966fdfca61d37a6177b484e070265ea015072bdcf70394.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/c93f14de8ee9bf99c0966fdfca61d37a6177b484e070265ea015072bdcf70394.jpg)

![d22eabaaed5df51fc20794b8a8628d830aa41ff1282dd549070f6df3e13e9d04.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/d22eabaaed5df51fc20794b8a8628d830aa41ff1282dd549070f6df3e13e9d04.jpg)

![d819141a74649523d45ee959aba5e9d7a951d6105253823bfe4cdcafd199999f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/d819141a74649523d45ee959aba5e9d7a951d6105253823bfe4cdcafd199999f.jpg)

![d85fb0e3bc961ecae37550bf2d66418e8410e620587455d08fcb935e4f55906d.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/d85fb0e3bc961ecae37550bf2d66418e8410e620587455d08fcb935e4f55906d.jpg)

![d93f9781f166a213e3b5ec929236cc2fb6e85df3b313721ab261541f6be7dccb.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/d93f9781f166a213e3b5ec929236cc2fb6e85df3b313721ab261541f6be7dccb.jpg)

![de3f4bfac23459795738f0a11a32ef3512df7a50d87c64709e9db5331d61ed59.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/de3f4bfac23459795738f0a11a32ef3512df7a50d87c64709e9db5331d61ed59.jpg)

![e0a0476ef497ae2e732fc35feba01b26774ca0fbb50186e772a6b5646a1246f7.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/e0a0476ef497ae2e732fc35feba01b26774ca0fbb50186e772a6b5646a1246f7.jpg)

![e441f939754ebb0e989af87a93598f1aa244a8496c87110e55d415bdd5e14432.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/e441f939754ebb0e989af87a93598f1aa244a8496c87110e55d415bdd5e14432.jpg)

![e9c437a8dd89efc6da706b741e364db23eaca7ebcb1e85187fe7def088b0796a.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/e9c437a8dd89efc6da706b741e364db23eaca7ebcb1e85187fe7def088b0796a.jpg)

![eb2ece99d475cb645aa9eaec1373edeebd88aee37be86b130e36d9f5f9ca80e6.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/eb2ece99d475cb645aa9eaec1373edeebd88aee37be86b130e36d9f5f9ca80e6.jpg)

![eefb4deb15aec4a0e9f0201d82238d304fb244a2c8cfc71b877fa003884a57b8.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/eefb4deb15aec4a0e9f0201d82238d304fb244a2c8cfc71b877fa003884a57b8.jpg)

![f0fcd49223aa5892ea6371f6e4aa30da95b8f8cd268bf19b40993ebafef496df.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/f0fcd49223aa5892ea6371f6e4aa30da95b8f8cd268bf19b40993ebafef496df.jpg)

![f2e6f1cd39faaf28160f0e89abad390d4d80dea18d847eaf3a959ba9812641d2.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/f2e6f1cd39faaf28160f0e89abad390d4d80dea18d847eaf3a959ba9812641d2.jpg)

![f348c6ebcf3b1cf80c763d16ff83ad3ef28b6d0c102e3c62521de98aefe88eba.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/f348c6ebcf3b1cf80c763d16ff83ad3ef28b6d0c102e3c62521de98aefe88eba.jpg)

![f4dd4586c2ac4fa33308f69246f152720f63f18cc6f33959aca42c1c7a05a7a5.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/f4dd4586c2ac4fa33308f69246f152720f63f18cc6f33959aca42c1c7a05a7a5.jpg)

![ff63d6a3b288f1071e89ea3e8e3248547cf3884f40a87a2c861bde4c3f31d97f.jpg](../icml_results/3186_AnalogVerifier_%20A%20Neuro-Symbolic%20Framework%20for%20Analog%20Circuit%20Verification/tables/ff63d6a3b288f1071e89ea3e8e3248547cf3884f40a87a2c861bde4c3f31d97f.jpg)

<a id="paper-3148"></a>

## Rethinking 3D Shape Generation: Diffusion over Superquadrics


### Images

![0ff94a1e6fcc3ca64aefff543060a10016d10b635631aa42521ce417fc97e60e.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/0ff94a1e6fcc3ca64aefff543060a10016d10b635631aa42521ce417fc97e60e.jpg)

![200b116885745fcb8d7732e14dcc4d6d132cb65be16376a1b15268e5b10a087d.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/200b116885745fcb8d7732e14dcc4d6d132cb65be16376a1b15268e5b10a087d.jpg)

![3c262ea9fd455e114dc17a5f5e06cf3c785d0966f0a19554d329c62d946242bd.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/3c262ea9fd455e114dc17a5f5e06cf3c785d0966f0a19554d329c62d946242bd.jpg)

![52108ebd0b882f9a9f4cd81a71ff5691a2e8ebe2289b4f98f7da788f0b3ec872.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/52108ebd0b882f9a9f4cd81a71ff5691a2e8ebe2289b4f98f7da788f0b3ec872.jpg)

![66aacf54c1b7d20fdd9a27633ccd689cf17aded92dabdd1486f053f63b341e05.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/66aacf54c1b7d20fdd9a27633ccd689cf17aded92dabdd1486f053f63b341e05.jpg)

![7069444238375c330b5422581c546bd63b0b5e73c0b6e058d47524253b7d3cf9.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/7069444238375c330b5422581c546bd63b0b5e73c0b6e058d47524253b7d3cf9.jpg)

![7e1c76bc2e6969dce9cb65a10b13de6bc5645ede5018c811b6024b6830604c14.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/7e1c76bc2e6969dce9cb65a10b13de6bc5645ede5018c811b6024b6830604c14.jpg)

![a177853b6706d9c7c242ce88956253871e9fa223a26d0d20d94ab01eb489b93e.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/a177853b6706d9c7c242ce88956253871e9fa223a26d0d20d94ab01eb489b93e.jpg)

![a60a089d2dbc588e34c11e60b9df1c4f16740d10150fee4ca96bd9674e09aa37.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/a60a089d2dbc588e34c11e60b9df1c4f16740d10150fee4ca96bd9674e09aa37.jpg)

![ad7b0362e7a6872de7a60c448b0167c78648bea6c5732eb3418e983d5641ff90.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/ad7b0362e7a6872de7a60c448b0167c78648bea6c5732eb3418e983d5641ff90.jpg)

![f71fce1ca1ecb3e4a01b34fb5afc35c5e491cda6b01cd45d9bbae15cc156922c.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/images/f71fce1ca1ecb3e4a01b34fb5afc35c5e491cda6b01cd45d9bbae15cc156922c.jpg)

### Tables

![08f015f75638afd0a17d2ecd242c7595f86e878a983a5602bae9784c21e1aa12.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/tables/08f015f75638afd0a17d2ecd242c7595f86e878a983a5602bae9784c21e1aa12.jpg)

![101f0c5f22bf6b1f0188a304bea6ff4c54cb9a4c234aef957a9900e2c933010d.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/tables/101f0c5f22bf6b1f0188a304bea6ff4c54cb9a4c234aef957a9900e2c933010d.jpg)

![a859fdc64718e644d29635b389214f0ba14761695ce8140b73e8cfd114c740f2.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/tables/a859fdc64718e644d29635b389214f0ba14761695ce8140b73e8cfd114c740f2.jpg)

![d4ead4a9187bd573c859d14b65b3369904b399a5b17e53e0971747aa2a8f568c.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/tables/d4ead4a9187bd573c859d14b65b3369904b399a5b17e53e0971747aa2a8f568c.jpg)

![e1ac3dd6f75bc503e05af32e3a8c54fd661f7c5ed43a3b1e760c4f450bdef825.jpg](../icml_results/3187_Rethinking%203D%20Shape%20Generation_%20Diffusion%20over%20Superquadrics/tables/e1ac3dd6f75bc503e05af32e3a8c54fd661f7c5ed43a3b1e760c4f450bdef825.jpg)

<a id="paper-3149"></a>

## Ramba: Selective State-Space Models for Relational Deep Learning


### Images

![029c45b63282a48603be22038b33847346c803cbcf6089b9d80eb6bb4f6741ba.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/029c45b63282a48603be22038b33847346c803cbcf6089b9d80eb6bb4f6741ba.jpg)

![6fff1f5a992a6d72acf5d9325fd05867f58ad2d2750a9d1c2768ccfd208d4332.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/6fff1f5a992a6d72acf5d9325fd05867f58ad2d2750a9d1c2768ccfd208d4332.jpg)

![874377a83bb3fe8640bd60da3d2e93f07048fe2843dad216ef4f05942e7a3055.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/874377a83bb3fe8640bd60da3d2e93f07048fe2843dad216ef4f05942e7a3055.jpg)

![9832bbe51e797d108621c7a8269c451fba082ac9bffb9f4368955d96fd7fb154.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/9832bbe51e797d108621c7a8269c451fba082ac9bffb9f4368955d96fd7fb154.jpg)

![9f29b798ed6e583b956a6b59659f0db53c3651e171d22b027aa59b0c3d3d9e98.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/9f29b798ed6e583b956a6b59659f0db53c3651e171d22b027aa59b0c3d3d9e98.jpg)

![b84437cf82319ed19f68f47b3d81d2bd96c972fbed36779c74db4edf2e16c13e.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/b84437cf82319ed19f68f47b3d81d2bd96c972fbed36779c74db4edf2e16c13e.jpg)

![d4826044f5ebb7422bfe56316e9afcb07cbeab966e598c1bc8a50655a123f467.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/d4826044f5ebb7422bfe56316e9afcb07cbeab966e598c1bc8a50655a123f467.jpg)

![f54d2726be47ded59a3fc7c92fcf3437e784eb7964002e696e3bae2f27c75f6c.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/f54d2726be47ded59a3fc7c92fcf3437e784eb7964002e696e3bae2f27c75f6c.jpg)

![f6b71d435fb03ffc5a32c530506100113807894c4d013919af8ff29a700ac967.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/images/f6b71d435fb03ffc5a32c530506100113807894c4d013919af8ff29a700ac967.jpg)

### Tables

![09000e3a5d4b7debe035575be9834413d2a959a0dab56c15c80e67f9c53a45b1.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/09000e3a5d4b7debe035575be9834413d2a959a0dab56c15c80e67f9c53a45b1.jpg)

![0f2592feffc4d6119194a77a2e2c58e4d2e6c728ca9ad7a1bad0faf719611f88.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/0f2592feffc4d6119194a77a2e2c58e4d2e6c728ca9ad7a1bad0faf719611f88.jpg)

![2ce6e0f2c4fdf90449a7665d79974c06d3f874e0f2c2f00e87d79d4bd1977cae.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/2ce6e0f2c4fdf90449a7665d79974c06d3f874e0f2c2f00e87d79d4bd1977cae.jpg)

![3046d77b8a4bdce5374ec223c70e457b044312b813872ee883dd973159fe8e2e.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/3046d77b8a4bdce5374ec223c70e457b044312b813872ee883dd973159fe8e2e.jpg)

![333e76dcb91170df1351e10751d6b1d00c8c6bdadede8bfd7b0bf6d9c8457578.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/333e76dcb91170df1351e10751d6b1d00c8c6bdadede8bfd7b0bf6d9c8457578.jpg)

![3b67455b36d1eae60edc0f825238062a81d086aff78d18c4c4f28e31677dd288.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/3b67455b36d1eae60edc0f825238062a81d086aff78d18c4c4f28e31677dd288.jpg)

![3e2d25aa10baedae3d9950bdecc95bfc7ec2ce83a0f3bd87daf9fae60004e2c8.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/3e2d25aa10baedae3d9950bdecc95bfc7ec2ce83a0f3bd87daf9fae60004e2c8.jpg)

![4b4411f894dab816d3462fe4e1ed93592dde43eb6364cba0b25adcd4fd5475f8.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/4b4411f894dab816d3462fe4e1ed93592dde43eb6364cba0b25adcd4fd5475f8.jpg)

![4cebf31e21cab970231c74459d9ac966a0ae9c57a6fd4d840d7e227181002ec6.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/4cebf31e21cab970231c74459d9ac966a0ae9c57a6fd4d840d7e227181002ec6.jpg)

![6354df8822f867f79a1766f8c03191b5bc2ce513453da95483a5c55aba615a02.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/6354df8822f867f79a1766f8c03191b5bc2ce513453da95483a5c55aba615a02.jpg)

![6ff1f91695eaca28e6952d8be08d3b4a27c7ce043bbc2c4b8b9a979d7a3e295b.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/6ff1f91695eaca28e6952d8be08d3b4a27c7ce043bbc2c4b8b9a979d7a3e295b.jpg)

![943e246d5f20751ba847f11d5a4e7e9b84d42ecd2a64054f592e84016ae9cd18.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/943e246d5f20751ba847f11d5a4e7e9b84d42ecd2a64054f592e84016ae9cd18.jpg)

![97e395dce80026920d51a6882638e4165175bf3426675da13e22a64b7c47a541.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/97e395dce80026920d51a6882638e4165175bf3426675da13e22a64b7c47a541.jpg)

![983438cc4da642f8a79ad2f823e54271c958962ffeb25a7c863511938d7f2606.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/983438cc4da642f8a79ad2f823e54271c958962ffeb25a7c863511938d7f2606.jpg)

![afa4a2e7e677bc162444ce693fce2fd419fea2f38dfef79a258fa525af183647.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/afa4a2e7e677bc162444ce693fce2fd419fea2f38dfef79a258fa525af183647.jpg)

![b358fa9b98c0909b2b402db27df6492ffd182b3caafd238174a7d7011201f7ba.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/b358fa9b98c0909b2b402db27df6492ffd182b3caafd238174a7d7011201f7ba.jpg)

![bf718ce69b362764244cc6e68a31cfa6658b1677ad592e812a9481344e95d1b1.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/bf718ce69b362764244cc6e68a31cfa6658b1677ad592e812a9481344e95d1b1.jpg)

![c44e624d8259ff707ace449fc963bdf1b9b211d23130fcc079ecffbaf533c08b.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/c44e624d8259ff707ace449fc963bdf1b9b211d23130fcc079ecffbaf533c08b.jpg)

![d62185226db52768c43a90c0b73da20bdabea2658aa43a106c198d754cd81e9a.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/d62185226db52768c43a90c0b73da20bdabea2658aa43a106c198d754cd81e9a.jpg)

![df1ee2f9094b5c31885bf1258163c8dfdc4512ffadaee5ab1a145c6ffde26503.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/df1ee2f9094b5c31885bf1258163c8dfdc4512ffadaee5ab1a145c6ffde26503.jpg)

![e1ef5822cb5cb1eabda6f4e45835018101cc98b4a58a93752746b4b70d017f77.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/e1ef5822cb5cb1eabda6f4e45835018101cc98b4a58a93752746b4b70d017f77.jpg)

![e7b259d97ccbe89a1f0552f0bf627984d223ef5a5310b3a37463b05733102ecc.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/e7b259d97ccbe89a1f0552f0bf627984d223ef5a5310b3a37463b05733102ecc.jpg)

![f8223054cb25b498e4451f6e998e93a450256c525f9d3b693e9bee847dc8c327.jpg](../icml_results/3188_Ramba_%20Selective%20State-Space%20Models%20for%20Relational%20Deep%20Learning/tables/f8223054cb25b498e4451f6e998e93a450256c525f9d3b693e9bee847dc8c327.jpg)

<a id="paper-3150"></a>

## Causal Disentangled Anchor Learning for Scalable Fair Multi-view Clustering


### Images

![0267abb00ded80d0887ee0b486aac7b488a3e80f499fc5f6cfbec6fc1adf5a5e.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/0267abb00ded80d0887ee0b486aac7b488a3e80f499fc5f6cfbec6fc1adf5a5e.jpg)

![4916a1d721e11514804b017471ae16fd34c0517cd128a8fe3e42d42d80f7d6ce.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/4916a1d721e11514804b017471ae16fd34c0517cd128a8fe3e42d42d80f7d6ce.jpg)

![609c3a6d8184d29620989ad50a00021c288fb83d2285c595b7a66a6f99aafd89.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/609c3a6d8184d29620989ad50a00021c288fb83d2285c595b7a66a6f99aafd89.jpg)

![7b3beae6a3aad02257be9f470e45b9f2d00a4477c3d6d6fe7e05efd8128cfdda.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/7b3beae6a3aad02257be9f470e45b9f2d00a4477c3d6d6fe7e05efd8128cfdda.jpg)

![7c3d359284f00439cf5ac2db94d1333ca9379732a0d48941d8b87e7ade1d4da3.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/7c3d359284f00439cf5ac2db94d1333ca9379732a0d48941d8b87e7ade1d4da3.jpg)

![951ed030a8e05a5479569f4decd3e1aed383a8f15aa69a52af19b6b71f319522.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/951ed030a8e05a5479569f4decd3e1aed383a8f15aa69a52af19b6b71f319522.jpg)

![b3c1dbe60efe89e645b57fe337013951e18b8bc24de01ea371c88b32ee5c5bc7.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/b3c1dbe60efe89e645b57fe337013951e18b8bc24de01ea371c88b32ee5c5bc7.jpg)

![f221e9f889c674d127136514e186b21116296e66dfb312727e18e4aa6acabf9e.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/images/f221e9f889c674d127136514e186b21116296e66dfb312727e18e4aa6acabf9e.jpg)

### Tables

![3a34a6adc496d38ce2022e48469ef37efe810a724542fe86de3477eea5ac3c0c.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/tables/3a34a6adc496d38ce2022e48469ef37efe810a724542fe86de3477eea5ac3c0c.jpg)

![9420da3a3df69ed287ea5500e262242739692febe36314d90c59212c83db7254.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/tables/9420da3a3df69ed287ea5500e262242739692febe36314d90c59212c83db7254.jpg)

![9d085d4de9180159173edade6db4701a207098fcffb28cf001a1f2b24dc60286.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/tables/9d085d4de9180159173edade6db4701a207098fcffb28cf001a1f2b24dc60286.jpg)

![d60771de036468d74253e32e0353689df66a141b290d361d1b62b88bf08231cb.jpg](../icml_results/3189_Causal%20Disentangled%20Anchor%20Learning%20for%20Scalable%20Fair%20Multi-view%20Clustering/tables/d60771de036468d74253e32e0353689df66a141b290d361d1b62b88bf08231cb.jpg)

<a id="paper-3151"></a>

## Decoupling Universal Laws and Environmental Heterogeneity: A Physics-Inspired Framework for Robust Spatio-Temporal Forecasting


### Images

![07ec76e4eedfd0171dca019b882dfe7a330dc2f9c91b0789386dc44a4dd65f98.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/07ec76e4eedfd0171dca019b882dfe7a330dc2f9c91b0789386dc44a4dd65f98.jpg)

![2858b1951db5a83f7fbb650a2cce02238fadb0852931195d852223b2c1130675.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/2858b1951db5a83f7fbb650a2cce02238fadb0852931195d852223b2c1130675.jpg)

![2e8a8bdac75ccb2e708425bed07ce8efb077ec6b1bd76fa2d1ad2bad1b79cc8b.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/2e8a8bdac75ccb2e708425bed07ce8efb077ec6b1bd76fa2d1ad2bad1b79cc8b.jpg)

![3e683eb5d6505a648a3ff0a3681f4f777fb56670d49d405930711a65d99536a4.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/3e683eb5d6505a648a3ff0a3681f4f777fb56670d49d405930711a65d99536a4.jpg)

![569c1c54f631de3470fbcc1a86401b01fdd5b4b6f5343a7387390bffccdbc041.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/569c1c54f631de3470fbcc1a86401b01fdd5b4b6f5343a7387390bffccdbc041.jpg)

![5939e945691530223af00d178cc9fb30f80d468c1efcf4258681423c38949c13.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/5939e945691530223af00d178cc9fb30f80d468c1efcf4258681423c38949c13.jpg)

![78e78aab9cdb0be1feffbfa97184944ecf3bd83989d9c1ede58835d9fa16bfb2.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/78e78aab9cdb0be1feffbfa97184944ecf3bd83989d9c1ede58835d9fa16bfb2.jpg)

![b275a02dae92cfe8c03d99c4fd66586fc75586398c663d584efb0da48f4c6f56.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/b275a02dae92cfe8c03d99c4fd66586fc75586398c663d584efb0da48f4c6f56.jpg)

![b4cb65bd1f4e6978e6f2e2c959841c4cf5d80e3c1cf130ebc945c63827352f4f.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/b4cb65bd1f4e6978e6f2e2c959841c4cf5d80e3c1cf130ebc945c63827352f4f.jpg)

![b985d3220312f84dd4997c86255d136838350cb53aaaab19d416877831e44d8c.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/b985d3220312f84dd4997c86255d136838350cb53aaaab19d416877831e44d8c.jpg)

![c7c91fff83d0381a080fc8f5f0d2898eddf964bc358effd90b8ec3a56ce2c542.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/c7c91fff83d0381a080fc8f5f0d2898eddf964bc358effd90b8ec3a56ce2c542.jpg)

![db0c9ce9774e603976c516bc1647d202cadc465082d4a93e5a7b3b919590cf10.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/images/db0c9ce9774e603976c516bc1647d202cadc465082d4a93e5a7b3b919590cf10.jpg)

### Tables

![0d5a7b6dab5156a7d7bc66e55a74db1c8ee82e2e66ce514161a1715776939485.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/0d5a7b6dab5156a7d7bc66e55a74db1c8ee82e2e66ce514161a1715776939485.jpg)

![1fbc4320a95154cbed648e9106c095ae99fad6e6def92c2330bb78831eb3da23.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/1fbc4320a95154cbed648e9106c095ae99fad6e6def92c2330bb78831eb3da23.jpg)

![208ae2069876e32f61b3aae991ee4b3aa7b5ff10f07e95a3f39f9b5b501a0e1d.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/208ae2069876e32f61b3aae991ee4b3aa7b5ff10f07e95a3f39f9b5b501a0e1d.jpg)

![2dc5cd234808f8d8e232b29787924a12a1f0ebdead2e514304c50bd75edb3936.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/2dc5cd234808f8d8e232b29787924a12a1f0ebdead2e514304c50bd75edb3936.jpg)

![38d6840605ac91bf56ea3a4bf2ee84bd810cdbd652f193358a4bb916e5137aee.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/38d6840605ac91bf56ea3a4bf2ee84bd810cdbd652f193358a4bb916e5137aee.jpg)

![3effd960c34df88c8ecc010a3511f617c0bbaece84f8fbabb23239b1e78dde31.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/3effd960c34df88c8ecc010a3511f617c0bbaece84f8fbabb23239b1e78dde31.jpg)

![95ba36dc5de9c7f67930c7efbe3be9d92af0ed1f4dd048f199fada277fca5adf.jpg](../icml_results/3190_Decoupling%20Universal%20Laws%20and%20Environmental%20Heterogeneity_%20A%20Physics-Inspired%20Framework%20for%20Robust%20S/tables/95ba36dc5de9c7f67930c7efbe3be9d92af0ed1f4dd048f199fada277fca5adf.jpg)

<a id="paper-3152"></a>

## From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection


### Images

![134cb1f2e056506f8246153243dcbc8fa3b0ae9516466d8222beaa037d03c2a0.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/134cb1f2e056506f8246153243dcbc8fa3b0ae9516466d8222beaa037d03c2a0.jpg)

![1b3deb933154f41f5d74299fe28174cd1c78304791cf616bdff6ac5fb1672416.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/1b3deb933154f41f5d74299fe28174cd1c78304791cf616bdff6ac5fb1672416.jpg)

![49e9b0fc512b23e3b642e2870ae0b30cd92b713b3e4459bd6ba06b032c61f558.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/49e9b0fc512b23e3b642e2870ae0b30cd92b713b3e4459bd6ba06b032c61f558.jpg)

![5653a9006b6a589b2769a4e15cdd7a2c1ae2395427160660de67ed0e312b02a5.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/5653a9006b6a589b2769a4e15cdd7a2c1ae2395427160660de67ed0e312b02a5.jpg)

![7f977e96f9f55c0f3eff94ff2fedd4d5ced56eae3522045c244f0ab14413667a.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/7f977e96f9f55c0f3eff94ff2fedd4d5ced56eae3522045c244f0ab14413667a.jpg)

![d3e0746d9d7ff7cfb4fcfd5141245bb260ad4a2521564c628ebd2173fed0f435.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/d3e0746d9d7ff7cfb4fcfd5141245bb260ad4a2521564c628ebd2173fed0f435.jpg)

![ea4cb0e7ce7f2d167365242af821071f8af80e001f0527c936daf54245cf9406.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/images/ea4cb0e7ce7f2d167365242af821071f8af80e001f0527c936daf54245cf9406.jpg)

### Tables

![0c1715e369cd34f5b18fe4339ec8bdcc79b5e758a01e38653caa7d07e52b1f49.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/0c1715e369cd34f5b18fe4339ec8bdcc79b5e758a01e38653caa7d07e52b1f49.jpg)

![13f46b4f24d7c3253aea4e64098fa786f02210f0bc3da137a1718f7f1cf76898.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/13f46b4f24d7c3253aea4e64098fa786f02210f0bc3da137a1718f7f1cf76898.jpg)

![17e5a9db5394125ccc3b7628d85e885e6e781439fbb5ecd5ab47c48a937f191d.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/17e5a9db5394125ccc3b7628d85e885e6e781439fbb5ecd5ab47c48a937f191d.jpg)

![211f9041137637be542d8519ac401b0041eea6fbd97aa43371c18fbc316bb069.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/211f9041137637be542d8519ac401b0041eea6fbd97aa43371c18fbc316bb069.jpg)

![26a479e19b50369c6f8f4e83546bdc870eab821118f58ac59c5210f15090a38f.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/26a479e19b50369c6f8f4e83546bdc870eab821118f58ac59c5210f15090a38f.jpg)

![295d6feb37e4786731dcdd110acd60a48b635f6784c1e818d37ad95ff038d609.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/295d6feb37e4786731dcdd110acd60a48b635f6784c1e818d37ad95ff038d609.jpg)

![497c76e75b4c58d6eeed9b56222dd3b37f59fc432fdd2a3fed47966977340047.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/497c76e75b4c58d6eeed9b56222dd3b37f59fc432fdd2a3fed47966977340047.jpg)

![4be8acd46423c0aa3785e4789385b834b6e28472fcdc03b639c7626a7588978f.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/4be8acd46423c0aa3785e4789385b834b6e28472fcdc03b639c7626a7588978f.jpg)

![5c5284a0d5efc019defda2f092e9d1ab729a920ec5edbeba1d81683f83a46171.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/5c5284a0d5efc019defda2f092e9d1ab729a920ec5edbeba1d81683f83a46171.jpg)

![74f177c37f7ba8e12668e0e344d45c9183278f9ff2d1b6a87bc6c695303e5716.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/74f177c37f7ba8e12668e0e344d45c9183278f9ff2d1b6a87bc6c695303e5716.jpg)

![80c728fe4a3c0613afe3230254f08c127503dfc6c74c52bc8620f237567c6eaa.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/80c728fe4a3c0613afe3230254f08c127503dfc6c74c52bc8620f237567c6eaa.jpg)

![9e7d089e9a28d28378776e5d1f9ac33d930d4abedcf74cc37fba03ec26373909.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/9e7d089e9a28d28378776e5d1f9ac33d930d4abedcf74cc37fba03ec26373909.jpg)

![9f67e7faedcff0911cea76e564e3e582e4ce415208e292862c2cb171552e8beb.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/9f67e7faedcff0911cea76e564e3e582e4ce415208e292862c2cb171552e8beb.jpg)

![b9cc60374d1373072bb3fcce27df48452f8bc8a3730de6cca3c259fbc4c0f5b9.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/b9cc60374d1373072bb3fcce27df48452f8bc8a3730de6cca3c259fbc4c0f5b9.jpg)

![d748c21a4ac886c5c4d9fda01339a37bfc39ff9004b3c1d10ab0cd890b7cb764.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/d748c21a4ac886c5c4d9fda01339a37bfc39ff9004b3c1d10ab0cd890b7cb764.jpg)

![d8ebc231779f24a46c86e6f93e7e859d5cc91f632c4b71ecfdf5e7b47b9d7d98.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/d8ebc231779f24a46c86e6f93e7e859d5cc91f632c4b71ecfdf5e7b47b9d7d98.jpg)

![f8c13a6cd5827f649905eb7d6f46066e08e41521222f3056f2b40e6efe3373f8.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/f8c13a6cd5827f649905eb7d6f46066e08e41521222f3056f2b40e6efe3373f8.jpg)

![fe35208db66e1f1b9cd40865051cd684bd4408ab9ff8c56f56e8bd401bda040c.jpg](../icml_results/3191_From%20Talking%20to%20Singing_%20A%20New%20Challenge%20for%20Audio-Visual%20Deepfake%20Detection/tables/fe35208db66e1f1b9cd40865051cd684bd4408ab9ff8c56f56e8bd401bda040c.jpg)

<a id="paper-3153"></a>

## Beyond Reactivity: Proactive Adaptive Conformal Inference for Online LLM Factuality


### Images

![2e1103a8da3cd354c10e63ba108d5787322f54591c2603d2d0011c231a51916f.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/2e1103a8da3cd354c10e63ba108d5787322f54591c2603d2d0011c231a51916f.jpg)

![5926a2a8d6b0293737416b596dbc4f2e0b9aee8fa78fee14ae25258bc152e8d0.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/5926a2a8d6b0293737416b596dbc4f2e0b9aee8fa78fee14ae25258bc152e8d0.jpg)

![5ba52f753578900379faef29249e640b67bcc3bb10972eba358b81a24a7c90be.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/5ba52f753578900379faef29249e640b67bcc3bb10972eba358b81a24a7c90be.jpg)

![611767987e88419c2db2c638215bdd9ab3d2e1902758401a940d07e945067ae4.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/611767987e88419c2db2c638215bdd9ab3d2e1902758401a940d07e945067ae4.jpg)

![85baa5ad17e4329cc3a6681f961ca97e698fd58863a191420e6ba961919fc81d.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/85baa5ad17e4329cc3a6681f961ca97e698fd58863a191420e6ba961919fc81d.jpg)

![a90b1f369981be8c974fb3fd96a5c521cc449611deeb797c638bd3a881cac20b.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/a90b1f369981be8c974fb3fd96a5c521cc449611deeb797c638bd3a881cac20b.jpg)

![c76e4263b0759b5801f927b18e20a77291159ecddfafb069fb62bb68e33a456a.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/c76e4263b0759b5801f927b18e20a77291159ecddfafb069fb62bb68e33a456a.jpg)

![cc43a27f9b8074a2d545966c4c4e0e0c631dadbd2f682775a1fd6e1105e9afb5.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/cc43a27f9b8074a2d545966c4c4e0e0c631dadbd2f682775a1fd6e1105e9afb5.jpg)

![d1f25ccc0d0f9a4a4a81d3bd2ad0616d9d501a11a73c993e1928f2aa64669f2c.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/d1f25ccc0d0f9a4a4a81d3bd2ad0616d9d501a11a73c993e1928f2aa64669f2c.jpg)

![edec699b3c8a477eef2af7480e70cd2062ace8e1219fe1422243948133486d39.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/edec699b3c8a477eef2af7480e70cd2062ace8e1219fe1422243948133486d39.jpg)

![ee1bf2ff4539aeab3a5f03673b5a9bf1df9103ae0bfa59002a4896ca24c25362.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/images/ee1bf2ff4539aeab3a5f03673b5a9bf1df9103ae0bfa59002a4896ca24c25362.jpg)

### Tables

![08c02b4a27145a188f5b1522ecc56d4240090454a919d525b54eeb7e34b7bc5b.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/08c02b4a27145a188f5b1522ecc56d4240090454a919d525b54eeb7e34b7bc5b.jpg)

![141477bbfd2311b52c73c1cad905221d1dee2c5db84b19a3384617b2e20a0a5b.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/141477bbfd2311b52c73c1cad905221d1dee2c5db84b19a3384617b2e20a0a5b.jpg)

![3620c8cc45872cd03c75a88375e12d71bd620d86809f3815df13993e59940348.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/3620c8cc45872cd03c75a88375e12d71bd620d86809f3815df13993e59940348.jpg)

![5716dc2c3971a1944ed394853c22f65b2aac20bc72808089c6d707c1d0528628.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/5716dc2c3971a1944ed394853c22f65b2aac20bc72808089c6d707c1d0528628.jpg)

![718ceb1aac0255b58a9eb7fede1e2d5406db4b318f39cc1d48d01aa9bf8b6864.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/718ceb1aac0255b58a9eb7fede1e2d5406db4b318f39cc1d48d01aa9bf8b6864.jpg)

![7388fc9a7a5ae3d5a7fa93997976af7e10391709946de5006cdb372fbab7c67a.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/7388fc9a7a5ae3d5a7fa93997976af7e10391709946de5006cdb372fbab7c67a.jpg)

![7c2d37114576b897cb5ba9c781258c602b4565988378f5b0be13772ecddd984f.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/7c2d37114576b897cb5ba9c781258c602b4565988378f5b0be13772ecddd984f.jpg)

![a88d630ab7b3340e653ff6ffce4f0812239f649a566e06931acad2796742a627.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/a88d630ab7b3340e653ff6ffce4f0812239f649a566e06931acad2796742a627.jpg)

![afc42b01e8aec2983a3ed20905f56898e7f962bde0290eb9d8eb41cacab47882.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/afc42b01e8aec2983a3ed20905f56898e7f962bde0290eb9d8eb41cacab47882.jpg)

![c035a9d2f13d5fe758de04278e0fdccc66ad8eccf749abec2e8b813fb33cf6af.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/c035a9d2f13d5fe758de04278e0fdccc66ad8eccf749abec2e8b813fb33cf6af.jpg)

![f6e393db5676b5aca9d39f00668483bec645a8d0b9a09a6e80f409cdc54ae754.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/f6e393db5676b5aca9d39f00668483bec645a8d0b9a09a6e80f409cdc54ae754.jpg)

![fb7170c7e0cf2a986ad77d76fcb2c6529d49b8774c9f4113710f5906b0236ab6.jpg](../icml_results/3192_Beyond%20Reactivity_%20Proactive%20Adaptive%20Conformal%20Inference%20for%20Online%20LLM%20Factuality/tables/fb7170c7e0cf2a986ad77d76fcb2c6529d49b8774c9f4113710f5906b0236ab6.jpg)

<a id="paper-3154"></a>

## Distillation Models are Good Samplers for Diffusion Reinforcement Learning


### Images

![018dda3697d203b2260daa80623051cc3d8229d9eda94e8c543121daab19e3de.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/018dda3697d203b2260daa80623051cc3d8229d9eda94e8c543121daab19e3de.jpg)

![02589e136e1fad44994a1d47507592211eba77b4974446f07d9337856c20f8b1.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/02589e136e1fad44994a1d47507592211eba77b4974446f07d9337856c20f8b1.jpg)

![173b521849b49a3a14b638b70d73f5f076ef8794276751d062d56c73cfc307c7.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/173b521849b49a3a14b638b70d73f5f076ef8794276751d062d56c73cfc307c7.jpg)

![194bee38134f2131799218d9ac0bbbc4128a30c0cfcc702eacbf51ece7f45635.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/194bee38134f2131799218d9ac0bbbc4128a30c0cfcc702eacbf51ece7f45635.jpg)

![1b2ca9f72bdaa3c217330d313dba30b62bff8c174066f7d634ae4f02c5676b42.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/1b2ca9f72bdaa3c217330d313dba30b62bff8c174066f7d634ae4f02c5676b42.jpg)

![1b37c6722a7a4f41a093d27f64de1aa23001158d96b11babc3eb9b7434824403.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/1b37c6722a7a4f41a093d27f64de1aa23001158d96b11babc3eb9b7434824403.jpg)

![21f67fc0617a0066c65e7b065dc322dc17212f80f7688323e0a9bda41546b46a.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/21f67fc0617a0066c65e7b065dc322dc17212f80f7688323e0a9bda41546b46a.jpg)

![22a65d0af1aa4c0ab3a5d0f9e72de54c7a59f4845f43d2f079f2afe1b417f3bf.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/22a65d0af1aa4c0ab3a5d0f9e72de54c7a59f4845f43d2f079f2afe1b417f3bf.jpg)

![22a84fec1d7f7155126cdae21d78cdc11246c62b156b81d13ded76ddd8a85fb8.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/22a84fec1d7f7155126cdae21d78cdc11246c62b156b81d13ded76ddd8a85fb8.jpg)

![23263fffebe7ea1a9381efa204acbcb79acbd46a084c0effdea0cd8a3b69a568.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/23263fffebe7ea1a9381efa204acbcb79acbd46a084c0effdea0cd8a3b69a568.jpg)

![2cf9b89e2f6bc5fd96bb6ea2ba195646ea709f4b25d7e1d2ead5cbf56b9cb543.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/2cf9b89e2f6bc5fd96bb6ea2ba195646ea709f4b25d7e1d2ead5cbf56b9cb543.jpg)

![351492edc4c3b41dab0ea16af7ca6b7503505f5d94ab247d25fd463f594ded32.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/351492edc4c3b41dab0ea16af7ca6b7503505f5d94ab247d25fd463f594ded32.jpg)

![556b0eddd9adf322a5f930faf7646163ac9f58ebde5a81435e0c60a4fce6f951.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/556b0eddd9adf322a5f930faf7646163ac9f58ebde5a81435e0c60a4fce6f951.jpg)

![5806967dcf99bc27e7c8e004edde48c2c29ba27c91aa503e7dd3187d003b155f.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/5806967dcf99bc27e7c8e004edde48c2c29ba27c91aa503e7dd3187d003b155f.jpg)

![60a08115224f403fa288a71e8e905921faf68869910c4ed08e48bea8f4960b96.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/60a08115224f403fa288a71e8e905921faf68869910c4ed08e48bea8f4960b96.jpg)

![6533c1fa38ce173b44d3e75992a63cf9e92cab09661e847957a522839da87de8.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/6533c1fa38ce173b44d3e75992a63cf9e92cab09661e847957a522839da87de8.jpg)

![67c7bc96d9b446d1af53d9cfd83d85621361398e71e650c9870654de2ca067ba.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/67c7bc96d9b446d1af53d9cfd83d85621361398e71e650c9870654de2ca067ba.jpg)

![6f021196a44e80164d4d93a230fbb5e0d9aecd0d151fd9644115f1abe5990b59.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/6f021196a44e80164d4d93a230fbb5e0d9aecd0d151fd9644115f1abe5990b59.jpg)

![7df0f354e4ea36e43252059dc7fef72038b14b38945caae9c3c8789662ab9a2c.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/7df0f354e4ea36e43252059dc7fef72038b14b38945caae9c3c8789662ab9a2c.jpg)

![9277e8cd5c266bee18fbb74563b1b6950784ec65f7d59e00434c27f3c24e48ac.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/9277e8cd5c266bee18fbb74563b1b6950784ec65f7d59e00434c27f3c24e48ac.jpg)

![930b00758854f41c8fd9e8022a6de95b14362e8ad1d30e9f507bc4d6db58a812.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/930b00758854f41c8fd9e8022a6de95b14362e8ad1d30e9f507bc4d6db58a812.jpg)

![9b06f326199fd05ff43be6615f2e4e6ed5b3af535873007e9f1b9ff7fedb2fa2.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/9b06f326199fd05ff43be6615f2e4e6ed5b3af535873007e9f1b9ff7fedb2fa2.jpg)

![9b80bd9f2c80b66088797098fd6dc7cbaa39a5c13e8bc490628d2cd5dca647db.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/9b80bd9f2c80b66088797098fd6dc7cbaa39a5c13e8bc490628d2cd5dca647db.jpg)

![afef5a1857960a41b85da5c525cc2850421a55cdd86bef06de83d89269fbe6c4.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/afef5a1857960a41b85da5c525cc2850421a55cdd86bef06de83d89269fbe6c4.jpg)

![b4af5f6ef17f957f48f843e0173a1dd37972f19532ed9e13f7d166d650b7431c.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/b4af5f6ef17f957f48f843e0173a1dd37972f19532ed9e13f7d166d650b7431c.jpg)

![c413ef79869bd70ff751d20382ae34f6d27cc2fe22dab4bd355ee0b8421e1428.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/c413ef79869bd70ff751d20382ae34f6d27cc2fe22dab4bd355ee0b8421e1428.jpg)

![f6bdf3b63d805c25d896e77c6c77ade34aa53c7c1a5ba18168dd5bb8a45b5a8b.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/f6bdf3b63d805c25d896e77c6c77ade34aa53c7c1a5ba18168dd5bb8a45b5a8b.jpg)

![feab890abe25d1d37cf72443285c29e5ff6fb18a5dd32e02f5828a1936301888.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/images/feab890abe25d1d37cf72443285c29e5ff6fb18a5dd32e02f5828a1936301888.jpg)

### Tables

![24f30c76dd09a72ada6080804be361976551b978db68c5bee9b28c52bdb2b509.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/24f30c76dd09a72ada6080804be361976551b978db68c5bee9b28c52bdb2b509.jpg)

![2fb1671591d737837ab97c2dbe06594fe549db352d2b3d4c26dba6550cab5f85.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/2fb1671591d737837ab97c2dbe06594fe549db352d2b3d4c26dba6550cab5f85.jpg)

![3e6631c93cee1e3642d4df93398584b8d5c18aa9fb97a54f987c21fa7dc2ca63.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/3e6631c93cee1e3642d4df93398584b8d5c18aa9fb97a54f987c21fa7dc2ca63.jpg)

![610255215d60451b764b362bd2f573c3c29f3f924692552d2689871d60d31470.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/610255215d60451b764b362bd2f573c3c29f3f924692552d2689871d60d31470.jpg)

![672ea8b00a38f605e6c40dd0be52e81e0174a4037a80822ea1b0764fe15762f2.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/672ea8b00a38f605e6c40dd0be52e81e0174a4037a80822ea1b0764fe15762f2.jpg)

![6f8acbe191b9660895533570bcd9d51c4309d949d2cedd59b66da637926081f7.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/6f8acbe191b9660895533570bcd9d51c4309d949d2cedd59b66da637926081f7.jpg)

![7232f064701df1cbafa4f80a5500646c1f70f3c22d5ebf8b3061370e2306e758.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/7232f064701df1cbafa4f80a5500646c1f70f3c22d5ebf8b3061370e2306e758.jpg)

![7f160f5b3afa15aa92e1d1b6aba69a568c8a8f6c5d7eea8e2888022c015fa631.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/7f160f5b3afa15aa92e1d1b6aba69a568c8a8f6c5d7eea8e2888022c015fa631.jpg)

![bddd22c2281de2b592d57a99d68c8be1076ed23859e0a37b2d7aaaa5e8f2db4e.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/bddd22c2281de2b592d57a99d68c8be1076ed23859e0a37b2d7aaaa5e8f2db4e.jpg)

![bf6e9944ad8118cbdcf4c9ea27f51ffa5c943fb34071a56d5f3aa7e876de69a2.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/bf6e9944ad8118cbdcf4c9ea27f51ffa5c943fb34071a56d5f3aa7e876de69a2.jpg)

![e8aa53b8e451605766a82bd90cd4bbfea9916d9defee0ddd8b21cf3a6957e0de.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/e8aa53b8e451605766a82bd90cd4bbfea9916d9defee0ddd8b21cf3a6957e0de.jpg)

![f3a68d23b19c1c28e103b99217df671a6cde57b48e29fc1e817ae4bde8f130ac.jpg](../icml_results/3193_Distillation%20Models%20are%20Good%20Samplers%20for%20Diffusion%20Reinforcement%20Learning/tables/f3a68d23b19c1c28e103b99217df671a6cde57b48e29fc1e817ae4bde8f130ac.jpg)

<a id="paper-3155"></a>

## Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning


### Images

![142500e156e7cb684c67c08291234e652974feaab227ad8c0faeec772e6b7e2d.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/142500e156e7cb684c67c08291234e652974feaab227ad8c0faeec772e6b7e2d.jpg)

![2bd8d4f9a85c64fb3f383c735dc03791dad71107d8ec704924cab175be50e680.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/2bd8d4f9a85c64fb3f383c735dc03791dad71107d8ec704924cab175be50e680.jpg)

![5d13a73ba0fcd863724263723b744164585deaaa2b851dbed72915ef588c4123.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/5d13a73ba0fcd863724263723b744164585deaaa2b851dbed72915ef588c4123.jpg)

![63330958bc689e00f4ebcb8c03c953ac1a30e5b5f2d2f6042d58f0bbe565f115.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/63330958bc689e00f4ebcb8c03c953ac1a30e5b5f2d2f6042d58f0bbe565f115.jpg)

![64a327dbd645cdaaac5df4eeb2e2f3bd8b147cf247e4f8537cee08d8570b3045.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/64a327dbd645cdaaac5df4eeb2e2f3bd8b147cf247e4f8537cee08d8570b3045.jpg)

![6841e8a56a5aba3405574da9deb24c397ab77f65bd5bfdd5988f87744e5a6776.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/6841e8a56a5aba3405574da9deb24c397ab77f65bd5bfdd5988f87744e5a6776.jpg)

![7b8fe37eba5b9e49945334e79340f301dec94d4119185865ab8455b3a3b9c8ed.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/7b8fe37eba5b9e49945334e79340f301dec94d4119185865ab8455b3a3b9c8ed.jpg)

![7e66e54a1042c389a0a4e702e0428fa910afd8dce1882391df336669e9e372c7.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/7e66e54a1042c389a0a4e702e0428fa910afd8dce1882391df336669e9e372c7.jpg)

![8496ba5ec511b52b9739eda653db5980a43368d14a5ed2d620ae1e6ee0574e73.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/8496ba5ec511b52b9739eda653db5980a43368d14a5ed2d620ae1e6ee0574e73.jpg)

![8ffd8758ad72696ca59b9b2f57df03bfc7dc40987bc0fd0a5bf25d0d424cf117.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/8ffd8758ad72696ca59b9b2f57df03bfc7dc40987bc0fd0a5bf25d0d424cf117.jpg)

![94aabbdfbc85fe95744b599c23c400e38453d49caa87b1b23f119c3e6e656b44.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/94aabbdfbc85fe95744b599c23c400e38453d49caa87b1b23f119c3e6e656b44.jpg)

![a4f1ffee72b89d9d68c48ca0ffcde8efcd515da2fa8c1c634e160dbb0fc62827.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/a4f1ffee72b89d9d68c48ca0ffcde8efcd515da2fa8c1c634e160dbb0fc62827.jpg)

![b9014cbcdd4f2a689f64056eba825638ff5f27a30db720b2e794b890fd2fca22.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/b9014cbcdd4f2a689f64056eba825638ff5f27a30db720b2e794b890fd2fca22.jpg)

![bd850adea1df48f667c620c0ce439697cb3beca003d7596561c5627d810d1a24.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/bd850adea1df48f667c620c0ce439697cb3beca003d7596561c5627d810d1a24.jpg)

![cb1391962268c3f45da9824206f47a21e3b7f77f07befd58a069ca16dd835054.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/cb1391962268c3f45da9824206f47a21e3b7f77f07befd58a069ca16dd835054.jpg)

![d42d33b80da199fe92bf2fb3f4a9a6ecc7c36cd87dc936a9582818eb7aeeb54f.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/d42d33b80da199fe92bf2fb3f4a9a6ecc7c36cd87dc936a9582818eb7aeeb54f.jpg)

![d4bf906c915a81b8d061d7332f35f5f6050017097787d9913ac3ad7cf061b424.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/d4bf906c915a81b8d061d7332f35f5f6050017097787d9913ac3ad7cf061b424.jpg)

![e038e66be92a5fba2370b796820df367098af64f3121ec7b68dc7304b57be0af.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/e038e66be92a5fba2370b796820df367098af64f3121ec7b68dc7304b57be0af.jpg)

![ee21a0159380c4bc82585b2dc5e51c254779a19a4b7867dc0e557ca83ac7209f.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/ee21a0159380c4bc82585b2dc5e51c254779a19a4b7867dc0e557ca83ac7209f.jpg)

![f1b630cf7db32299ae921b3bbc16c0aea05053a93c2f0b6a3a10863307f00c7f.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/f1b630cf7db32299ae921b3bbc16c0aea05053a93c2f0b6a3a10863307f00c7f.jpg)

![fbdc01e876118d7b01f93ed029bc4f7110507f02d2701897ff7749ab718aed06.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/images/fbdc01e876118d7b01f93ed029bc4f7110507f02d2701897ff7749ab718aed06.jpg)

### Tables

![01171cba2213f209a8290dc818b7b12b749d7298fd875e44557298eb630087f3.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/01171cba2213f209a8290dc818b7b12b749d7298fd875e44557298eb630087f3.jpg)

![032d4ebf261cd6797b2406be15184c2a4f1ffbbce5bdaeb8555b5eb8504ddfe3.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/032d4ebf261cd6797b2406be15184c2a4f1ffbbce5bdaeb8555b5eb8504ddfe3.jpg)

![14d19abf10dbaccf61c503b5fbad83fe7fccb17e257af2458724ba808ed84aad.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/14d19abf10dbaccf61c503b5fbad83fe7fccb17e257af2458724ba808ed84aad.jpg)

![28cd12c4a6c2a72e365c670d259e4be1935f68181c0634607eebe489c1042d5a.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/28cd12c4a6c2a72e365c670d259e4be1935f68181c0634607eebe489c1042d5a.jpg)

![313c95c555563b63a9e6b16229e4a31c06ad024275a72e27026f9eedd9b44775.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/313c95c555563b63a9e6b16229e4a31c06ad024275a72e27026f9eedd9b44775.jpg)

![33310843355b5c190c143b03c3011c99b0684a8c287b4dde792df122ef63bbeb.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/33310843355b5c190c143b03c3011c99b0684a8c287b4dde792df122ef63bbeb.jpg)

![39d9e8961e7413914af8c42a678802ea03d5f20011a1ed6e388ab53fcc684bec.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/39d9e8961e7413914af8c42a678802ea03d5f20011a1ed6e388ab53fcc684bec.jpg)

![3a8b94d6055cbb089a76dc5f9ac933f207e72a686d2ea153506c967f2b63fa08.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/3a8b94d6055cbb089a76dc5f9ac933f207e72a686d2ea153506c967f2b63fa08.jpg)

![3c1505dbf385dde13a0705312edecaf78cc9dcb0bb39c767fba8c9ddeffbbd36.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/3c1505dbf385dde13a0705312edecaf78cc9dcb0bb39c767fba8c9ddeffbbd36.jpg)

![3f1c3a8abe161af8533c9aa9d998c9147d61f991b9eeb4520e11139097c29c28.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/3f1c3a8abe161af8533c9aa9d998c9147d61f991b9eeb4520e11139097c29c28.jpg)

![45d2bdd74c827bbeb4f096825fb9550c65f034daff04f703f690658cf46dc3b1.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/45d2bdd74c827bbeb4f096825fb9550c65f034daff04f703f690658cf46dc3b1.jpg)

![486f9faeb220ee5c04955f1aa78badebe9fd27f8580cffa27e56756129235ab4.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/486f9faeb220ee5c04955f1aa78badebe9fd27f8580cffa27e56756129235ab4.jpg)

![4c6dad9c434bde59ed19b6c9742a03b65de9a19ac9d5098f5f8f6fe09e08f205.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/4c6dad9c434bde59ed19b6c9742a03b65de9a19ac9d5098f5f8f6fe09e08f205.jpg)

![9e8b1639374172cb3872bee9bea344b93a910db4248ba121729846b39b132ea4.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/9e8b1639374172cb3872bee9bea344b93a910db4248ba121729846b39b132ea4.jpg)

![ac3a49f076b40a96319c92533df7ca3504aa571a3aac70a57a17e9a34523a3d6.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/ac3a49f076b40a96319c92533df7ca3504aa571a3aac70a57a17e9a34523a3d6.jpg)

![c943a02acdd333e327bbfbc1c9de05eeee5d9779fb8115b8e5f1b172bfd56872.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/c943a02acdd333e327bbfbc1c9de05eeee5d9779fb8115b8e5f1b172bfd56872.jpg)

![cbfcbdc6d51ce6d6fcbc8fda5778b891124790695da45ae819713763953c0279.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/cbfcbdc6d51ce6d6fcbc8fda5778b891124790695da45ae819713763953c0279.jpg)

![ef68d08b3aa502e68fbb82276cfe1b1487621db5ffcc89541ddf8cd6bc886d79.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/ef68d08b3aa502e68fbb82276cfe1b1487621db5ffcc89541ddf8cd6bc886d79.jpg)

![f10cfd5bee71cc7b4e017663f857e241989be7abeb879231a3dd92e2c04245d3.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/f10cfd5bee71cc7b4e017663f857e241989be7abeb879231a3dd92e2c04245d3.jpg)

![f230308a33a891c87ad3dfee77e26e17d996e52ff26fc406d2164ad4bd1a779f.jpg](../icml_results/3194_Scaling%20the%20Scaling%20Logic_%20Agentic%20Meta-Synthesis%20of%20Logic%20Reasoning/tables/f230308a33a891c87ad3dfee77e26e17d996e52ff26fc406d2164ad4bd1a779f.jpg)

<a id="paper-3156"></a>

## Calibrated Multimodal Representation Learning with Missing Modalities


### Images

![20512e13263cd6d42c96093f22bd9733885da343f8cd83623caf3d486fb14e39.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/20512e13263cd6d42c96093f22bd9733885da343f8cd83623caf3d486fb14e39.jpg)

![2fe9fd7f5de0e31e40c688e382415d7f1644ce0ef43951d925affa6a6d240efc.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/2fe9fd7f5de0e31e40c688e382415d7f1644ce0ef43951d925affa6a6d240efc.jpg)

![5d2378328394393fcdce4d3c0478da717782ee3b7f9b31717201f3119684314e.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/5d2378328394393fcdce4d3c0478da717782ee3b7f9b31717201f3119684314e.jpg)

![6092ef894efc7208a5f55f1beee9d5a7416616b9eeda84a025d8219c5372de8e.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/6092ef894efc7208a5f55f1beee9d5a7416616b9eeda84a025d8219c5372de8e.jpg)

![bda38b5a5101b9eeb94a1623ef0b3189b9096845f2853e88f6a5f55067b6652b.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/bda38b5a5101b9eeb94a1623ef0b3189b9096845f2853e88f6a5f55067b6652b.jpg)

![c1c8bb5cee43150f05f7018bfb9879f741728ef8cc388c93be4ab7bd12fe15a5.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/c1c8bb5cee43150f05f7018bfb9879f741728ef8cc388c93be4ab7bd12fe15a5.jpg)

![f17d5bfcd37b0cbbd733a2888a022bc34b87d0e8d56c43158233b0a44a51d8a9.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/images/f17d5bfcd37b0cbbd733a2888a022bc34b87d0e8d56c43158233b0a44a51d8a9.jpg)

### Tables

![19a401226fd79c441276047da484e550480b2ef0a611d63ae96dbd86fe1b54ee.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/19a401226fd79c441276047da484e550480b2ef0a611d63ae96dbd86fe1b54ee.jpg)

![2f958b6d801191778bd6f056376347c1aabdf7e662705df69ef7c63a2a915272.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/2f958b6d801191778bd6f056376347c1aabdf7e662705df69ef7c63a2a915272.jpg)

![44d770b783acaabf07d0dbdb46b35ed3d454dd4624e8d2731a3da1950dec931c.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/44d770b783acaabf07d0dbdb46b35ed3d454dd4624e8d2731a3da1950dec931c.jpg)

![6bc3afe55ebb98494c702d0e1776461ca2cd7a6dab07d0dcb2db5c2855220891.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/6bc3afe55ebb98494c702d0e1776461ca2cd7a6dab07d0dcb2db5c2855220891.jpg)

![9585d5ba05b889b34de9d8c879bdd3086197dc0c9dee5cd55dc269c6d9dbfe2d.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/9585d5ba05b889b34de9d8c879bdd3086197dc0c9dee5cd55dc269c6d9dbfe2d.jpg)

![99ca770c2818f8b85029844d56d4a6e7bd62575bd0410e0c714efedae01eab11.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/99ca770c2818f8b85029844d56d4a6e7bd62575bd0410e0c714efedae01eab11.jpg)

![d61eb615c49b759b3c0f6da2d2e23f3687e3a816e31767ad3c3e540615267e4d.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/d61eb615c49b759b3c0f6da2d2e23f3687e3a816e31767ad3c3e540615267e4d.jpg)

![dcbae01cce72acfbbb9b6d931d9f609e82c4a273341d3ee199f729e5629a84d9.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/dcbae01cce72acfbbb9b6d931d9f609e82c4a273341d3ee199f729e5629a84d9.jpg)

![dfb990c2770d3a3e0c1a9ff16103dc25891cf497a67ba908e9a79093ae53e055.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/dfb990c2770d3a3e0c1a9ff16103dc25891cf497a67ba908e9a79093ae53e055.jpg)

![f0b6b7f0fae1337fa4e2eaad9b600fc94f1b65b94972d98d8d2209d4c9e9a8fd.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/f0b6b7f0fae1337fa4e2eaad9b600fc94f1b65b94972d98d8d2209d4c9e9a8fd.jpg)

![f2c1dab9bcb808f871a58cdff7f527dc84265afd7cdc2856604d9f9a69940ce3.jpg](../icml_results/3195_Calibrated%20Multimodal%20Representation%20Learning%20with%20Missing%20Modalities/tables/f2c1dab9bcb808f871a58cdff7f527dc84265afd7cdc2856604d9f9a69940ce3.jpg)

<a id="paper-3157"></a>

## Cardio-mmFlow: A Gaussian-Prior-Free Physics-Informed Flow Matching Framework for Electrocardiogram to mmWave Radar Synthesis


### Images

![0548e2cd5ac40f749eff540b994e4e5dd5d90eb8214a9eb29d0ff1bbbe394e98.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/0548e2cd5ac40f749eff540b994e4e5dd5d90eb8214a9eb29d0ff1bbbe394e98.jpg)

![08aca162271c58abf3d3e26682ce87969357eac1b17e971e74ac045fe38ff03f.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/08aca162271c58abf3d3e26682ce87969357eac1b17e971e74ac045fe38ff03f.jpg)

![2c28ce76e75adea3f04df5aab46f05cd66107354e75511b7e8f444deb5c08927.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/2c28ce76e75adea3f04df5aab46f05cd66107354e75511b7e8f444deb5c08927.jpg)

![77815b3a738dcdb48ab050fb5935744c0b8691be3e0687458f45d065c5d23f6f.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/77815b3a738dcdb48ab050fb5935744c0b8691be3e0687458f45d065c5d23f6f.jpg)

![98ed253dc60c7c2c886772fb3b3a7ec2eac264cb06eb04cd206558b8e964d69e.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/98ed253dc60c7c2c886772fb3b3a7ec2eac264cb06eb04cd206558b8e964d69e.jpg)

![a0da772012b61c2938f6ef47b19f603303fa8fb9d3c3b40e46979b8bdbcf3145.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/a0da772012b61c2938f6ef47b19f603303fa8fb9d3c3b40e46979b8bdbcf3145.jpg)

![c457ec2619a6949b420f6219340d067f41f1dc32259d6771356d0fa084f885ac.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/c457ec2619a6949b420f6219340d067f41f1dc32259d6771356d0fa084f885ac.jpg)

![cf55bb582af457cb37d79fd5e5c886c557cf74b05680c45c13f57ef652572c37.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/images/cf55bb582af457cb37d79fd5e5c886c557cf74b05680c45c13f57ef652572c37.jpg)

### Tables

![33ae27d57acf7b1ce3a1431f7833d758f2e0430514e8c16a15bd13edf2435336.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/33ae27d57acf7b1ce3a1431f7833d758f2e0430514e8c16a15bd13edf2435336.jpg)

![4a687872a8af9b59d9afed8a4d718dc436a805e25636daef1465399c647ff069.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/4a687872a8af9b59d9afed8a4d718dc436a805e25636daef1465399c647ff069.jpg)

![60121107fdb2e887d46276051f3c008150b2139279e62506292fe4270f4e456d.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/60121107fdb2e887d46276051f3c008150b2139279e62506292fe4270f4e456d.jpg)

![8f992c014f0341a6a58257489ab99e3bdfd131b468ca275de9f8cffaa364c7f6.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/8f992c014f0341a6a58257489ab99e3bdfd131b468ca275de9f8cffaa364c7f6.jpg)

![98544ca33deaf45e2d961b170eb9df707a9a08ccbfb6f42202ef866b668d1df8.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/98544ca33deaf45e2d961b170eb9df707a9a08ccbfb6f42202ef866b668d1df8.jpg)

![de52f9b7db823cf3f805a2a30fa05462e989cab6556302951c9e2fd965e54848.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/de52f9b7db823cf3f805a2a30fa05462e989cab6556302951c9e2fd965e54848.jpg)

![f95ddd6a051f81ec8e9ddf74f49946ac45da3789365ec486a9fee9f859c02660.jpg](../icml_results/3196_Cardio-mmFlow_%20A%20Gaussian-Prior-Free%20Physics-Informed%20Flow%20Matching%20Framework%20for%20Electrocardiogram%20/tables/f95ddd6a051f81ec8e9ddf74f49946ac45da3789365ec486a9fee9f859c02660.jpg)

<a id="paper-3158"></a>

## MathlibLemma: Folklore Lemma Generation and Benchmark for Formal Mathematics


### Images

![0988435c044fb62297799e5543dfdd1f6ce1641376b68ceb14d9d0da49490194.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/images/0988435c044fb62297799e5543dfdd1f6ce1641376b68ceb14d9d0da49490194.jpg)

![34c1d772301779a62a5ae0499d521e86bf7850180ff29ad133100dc82cf014e2.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/images/34c1d772301779a62a5ae0499d521e86bf7850180ff29ad133100dc82cf014e2.jpg)

![4a13aa8658aec1784c29f47a7e93189018a32ff4e866241f03e72da2695db5b5.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/images/4a13aa8658aec1784c29f47a7e93189018a32ff4e866241f03e72da2695db5b5.jpg)

![565a17ac709059b106715d4cfd75e0dc30773264fc85936a440fd55bf3c4c57b.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/images/565a17ac709059b106715d4cfd75e0dc30773264fc85936a440fd55bf3c4c57b.jpg)

![e182d712dc8731abe932fcfb479dd2b0964a740ae0ae8e3ecfb204a80514891c.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/images/e182d712dc8731abe932fcfb479dd2b0964a740ae0ae8e3ecfb204a80514891c.jpg)

### Tables

![3a204d5b262ef666e188a4d0e5a55727002b66a884b514b7344d273e99396b1e.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/3a204d5b262ef666e188a4d0e5a55727002b66a884b514b7344d273e99396b1e.jpg)

![83f0a7953169938b2c58d80240ab0bab1ed91c46237cac6f320b73b549d78f2e.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/83f0a7953169938b2c58d80240ab0bab1ed91c46237cac6f320b73b549d78f2e.jpg)

![b686d40ee9db63b3c877ca03541693b50c45016a99b78f747ed2271e7d7a3640.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/b686d40ee9db63b3c877ca03541693b50c45016a99b78f747ed2271e7d7a3640.jpg)

![c07a4fe28974651fa60fd959b19be702d47d9151a1aeda8203b07198f06a7b52.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/c07a4fe28974651fa60fd959b19be702d47d9151a1aeda8203b07198f06a7b52.jpg)

![d2044f159f6cd9173f9894bed8e669e560ac67f6fcccadbfc6fe8ee4c0e9f8af.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/d2044f159f6cd9173f9894bed8e669e560ac67f6fcccadbfc6fe8ee4c0e9f8af.jpg)

![e0d128710522e7695c3579c46f2ec9a84ef44d77ee3cc828ed711e546fe97330.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/e0d128710522e7695c3579c46f2ec9a84ef44d77ee3cc828ed711e546fe97330.jpg)

![f49e4e8ecf6dd20f351c64eaf3ea454d115ee263ff7271a811b13f945947b412.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/f49e4e8ecf6dd20f351c64eaf3ea454d115ee263ff7271a811b13f945947b412.jpg)

![ff4259c7fb84b497ed7497f80ad13230e1b855911f6453d59c62a09743726009.jpg](../icml_results/3197_MathlibLemma_%20Folklore%20Lemma%20Generation%20and%20Benchmark%20for%20Formal%20Mathematics/tables/ff4259c7fb84b497ed7497f80ad13230e1b855911f6453d59c62a09743726009.jpg)

<a id="paper-3159"></a>

## Clipped Q-Learning: Your Value Clipping Is Secretly A Robust Operator


### Images

![0c6177c79780c708ffdcd7bff1f90f7928b92255e6703c6aea83d831eb46f023.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/0c6177c79780c708ffdcd7bff1f90f7928b92255e6703c6aea83d831eb46f023.jpg)

![1713c5b6da7fa7927ae3081b6a7166b594ee3bd8b52cb0abb9d7f48d346b589e.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/1713c5b6da7fa7927ae3081b6a7166b594ee3bd8b52cb0abb9d7f48d346b589e.jpg)

![3aaa61206f8fbdfe5fe14b58651bfbab6a1f4c267d57a01dec624a07c7fa6ce2.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/3aaa61206f8fbdfe5fe14b58651bfbab6a1f4c267d57a01dec624a07c7fa6ce2.jpg)

![5a506a6db378289c8c9de796b3d62a68457050df327ca8b3cb5711c9e18974a2.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/5a506a6db378289c8c9de796b3d62a68457050df327ca8b3cb5711c9e18974a2.jpg)

![5b02d35ac04c5a25de001d8d880d23972d4a9c75ec7027d6cbc126998fcf108d.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/5b02d35ac04c5a25de001d8d880d23972d4a9c75ec7027d6cbc126998fcf108d.jpg)

![629a975d9fd33a16061c275c5bd9de648411337523b3d0a72a3f56f152a79ac9.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/629a975d9fd33a16061c275c5bd9de648411337523b3d0a72a3f56f152a79ac9.jpg)

![6b9a7fbe673c014887dc3b8282881b74d8bbf9c18153c34f6846365afcd4feb9.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/6b9a7fbe673c014887dc3b8282881b74d8bbf9c18153c34f6846365afcd4feb9.jpg)

![77ab560c4750a278987d3b7cef2020f8a92ac97240e3c4a0ce62805cca1718b8.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/77ab560c4750a278987d3b7cef2020f8a92ac97240e3c4a0ce62805cca1718b8.jpg)

![79c5c120e63231d0683a6829385cbed3904c0926a6e47cd02d9b72eb26f862ae.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/79c5c120e63231d0683a6829385cbed3904c0926a6e47cd02d9b72eb26f862ae.jpg)

![8c341364cbf62e831e71b97df4d50f858e6ddfa7d9d8c0a259bfa15ad2d3d90d.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/8c341364cbf62e831e71b97df4d50f858e6ddfa7d9d8c0a259bfa15ad2d3d90d.jpg)

![9181fb9fca5c4a26c4f9a526625c812e0e113944ad145736711608486ba23d2f.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/9181fb9fca5c4a26c4f9a526625c812e0e113944ad145736711608486ba23d2f.jpg)

![93e9a5fb826ab1c6005f2feb3af7915ed9c04b84b7a32789256f2cc304f9b23a.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/93e9a5fb826ab1c6005f2feb3af7915ed9c04b84b7a32789256f2cc304f9b23a.jpg)

![a2cabb3962d1cc54906faecf68cf6736886cffa5892a5887a8705fe975ae2e4f.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/a2cabb3962d1cc54906faecf68cf6736886cffa5892a5887a8705fe975ae2e4f.jpg)

![af350afa33d807b2f9bc8a04f003e37609578e623ed939b8a9132ad17811dd91.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/af350afa33d807b2f9bc8a04f003e37609578e623ed939b8a9132ad17811dd91.jpg)

![b5feaa52580682ff40642662ce289892fb0865202d742d809ff59bc94a702a97.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/b5feaa52580682ff40642662ce289892fb0865202d742d809ff59bc94a702a97.jpg)

![da561025c185ceccd09347e4b56ae951ef6c9a05618c53faf719fd28c53b76d1.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/da561025c185ceccd09347e4b56ae951ef6c9a05618c53faf719fd28c53b76d1.jpg)

![ef509e7c0035852af14d963c230a2d9f9a6e84cc83d959e54d028c0d520db661.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/ef509e7c0035852af14d963c230a2d9f9a6e84cc83d959e54d028c0d520db661.jpg)

![f8616169f560a1ba6f9b601a6aa2af13469e8299cceea8af51f515242e4e37c8.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/images/f8616169f560a1ba6f9b601a6aa2af13469e8299cceea8af51f515242e4e37c8.jpg)

### Tables

![a799e668bdd03b0f06befbce2b9ba4ae548f685b2ee51cda4a3e94020bd63cb8.jpg](../icml_results/3198_Clipped%20Q-Learning_%20Your%20Value%20Clipping%20Is%20Secretly%20A%20Robust%20Operator/tables/a799e668bdd03b0f06befbce2b9ba4ae548f685b2ee51cda4a3e94020bd63cb8.jpg)

<a id="paper-3160"></a>

## CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning


### Images

![078ab30b057cdf3614e2c47fbdd2157f5681c3a0c3ffd5afbac048c899fc7737.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/078ab30b057cdf3614e2c47fbdd2157f5681c3a0c3ffd5afbac048c899fc7737.jpg)

![0e7064f9f998c6d6d1aef0e1bd0ee4ca432cdd0198a9175453e84921ad902e72.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/0e7064f9f998c6d6d1aef0e1bd0ee4ca432cdd0198a9175453e84921ad902e72.jpg)

![23c321415fce4ef117d7d9c0adf0fad6585133a39359c6bba571c0d90d760c29.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/23c321415fce4ef117d7d9c0adf0fad6585133a39359c6bba571c0d90d760c29.jpg)

![2ff1ac300b468d21f0fc58d70b744ba0179790e7da3be44c14e843cfb9efb047.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/2ff1ac300b468d21f0fc58d70b744ba0179790e7da3be44c14e843cfb9efb047.jpg)

![64110fc977e3753ec1e400fd7e7dc5f7100530b09798e9c56833128d272c10b0.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/64110fc977e3753ec1e400fd7e7dc5f7100530b09798e9c56833128d272c10b0.jpg)

![823b7b79c87073e6b12bdbfd56c5cef1956010c4899ef84a5b4f95fd64a77f23.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/823b7b79c87073e6b12bdbfd56c5cef1956010c4899ef84a5b4f95fd64a77f23.jpg)

![8aa730ab6afd79a6dbe770ecd6781d47ae779e08de745a21e2c12026b800251b.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/8aa730ab6afd79a6dbe770ecd6781d47ae779e08de745a21e2c12026b800251b.jpg)

![8ac9b85301d4971012634b7fac64784382e372289ca7a2b6735c9bbb798f0fa3.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/8ac9b85301d4971012634b7fac64784382e372289ca7a2b6735c9bbb798f0fa3.jpg)

![a7d797584cb2d8c7b1876d209f3e3881614db9b97261ad088baba3eec29651b2.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/a7d797584cb2d8c7b1876d209f3e3881614db9b97261ad088baba3eec29651b2.jpg)

![b5fe5a149f0aca922cc987e21468df9f772a7ff4bc856aa6d1595f2247398912.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/b5fe5a149f0aca922cc987e21468df9f772a7ff4bc856aa6d1595f2247398912.jpg)

![f277410c26fe3fae88fa4ac039ac818788784200b3b25596fbe9a395dab62ed0.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/images/f277410c26fe3fae88fa4ac039ac818788784200b3b25596fbe9a395dab62ed0.jpg)

### Tables

![32f10d64c1ac27dc2a3752ec4e5c0b6a42ea205bfd7e4977b5aadec14646e11f.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/32f10d64c1ac27dc2a3752ec4e5c0b6a42ea205bfd7e4977b5aadec14646e11f.jpg)

![34d00122a692018a7f85c736a6437bb1905a9faadf67b80259d267d1aa14a8b2.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/34d00122a692018a7f85c736a6437bb1905a9faadf67b80259d267d1aa14a8b2.jpg)

![38259a5b61b86aace9069f3e17c54c27c6e41c4e24012568a84ad1c169a347ff.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/38259a5b61b86aace9069f3e17c54c27c6e41c4e24012568a84ad1c169a347ff.jpg)

![47f121abccc04e5c27885cac69cc6786883fcbb9e63f25bb989f2dca360b318f.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/47f121abccc04e5c27885cac69cc6786883fcbb9e63f25bb989f2dca360b318f.jpg)

![49d18a5f669216d4cd5a9846952542d30153cd926f325959c34483c858a255b0.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/49d18a5f669216d4cd5a9846952542d30153cd926f325959c34483c858a255b0.jpg)

![5054e4a8ab0c53f1336b342366f3b05759b3ddbb7424734769aebbed2cf6ba78.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/5054e4a8ab0c53f1336b342366f3b05759b3ddbb7424734769aebbed2cf6ba78.jpg)

![604cf43cab73abc35e04444f78f51aa04d168876c78ebd5c89ccef7e5ec4e50c.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/604cf43cab73abc35e04444f78f51aa04d168876c78ebd5c89ccef7e5ec4e50c.jpg)

![7b1971dd074b3d409c7c4133703aa968ed6a3b49271db84421df3e0bbfa81d8d.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/7b1971dd074b3d409c7c4133703aa968ed6a3b49271db84421df3e0bbfa81d8d.jpg)

![7e9d5c6bde31212af2d30e29ab74ccc8b1b65e16efaab22e02d4127250341626.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/7e9d5c6bde31212af2d30e29ab74ccc8b1b65e16efaab22e02d4127250341626.jpg)

![81c65db5e5a335626f60dc38e398c262b4456524f5539ebbc99280de3f5330db.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/81c65db5e5a335626f60dc38e398c262b4456524f5539ebbc99280de3f5330db.jpg)

![826057d5e96fa15ad8e61939473ac5087d098beb42fd9be68026229960280e1f.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/826057d5e96fa15ad8e61939473ac5087d098beb42fd9be68026229960280e1f.jpg)

![942a6561366b476c33982cedd0268014413d908fd1abe2574fe816705c9d074a.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/942a6561366b476c33982cedd0268014413d908fd1abe2574fe816705c9d074a.jpg)

![b0837b2f2ff8282582f6261ba17e6e05d6513d3d6feeb8cbad7c0f8d721d4e62.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/b0837b2f2ff8282582f6261ba17e6e05d6513d3d6feeb8cbad7c0f8d721d4e62.jpg)

![e55c42eb3c2612ae7f070bd3039d4fee992fa22b17a202cf5a96d322e9346c28.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/e55c42eb3c2612ae7f070bd3039d4fee992fa22b17a202cf5a96d322e9346c28.jpg)

![ed99b033769e75ace439095504e270f1e12ee04cd19dfdc60b8cd9c59a46eb97.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/ed99b033769e75ace439095504e270f1e12ee04cd19dfdc60b8cd9c59a46eb97.jpg)

![f2232df6ca0053de657a004739f08df625cd60fd9a563f3ab24a13f367553871.jpg](../icml_results/3199_CoME_%20Empowering%20Channel-of-Mobile-Experts%20with%20Informative%20Hybrid-Capabilities%20Reasoning/tables/f2232df6ca0053de657a004739f08df625cd60fd9a563f3ab24a13f367553871.jpg)

<a id="paper-3161"></a>

## Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations


### Images

![3bebcee9b6cb7a85e75afe1d7fa707d1e98023f2c2f1c45879ded558407bb3ca.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/3bebcee9b6cb7a85e75afe1d7fa707d1e98023f2c2f1c45879ded558407bb3ca.jpg)

![53e5ac1fa0f885e2a91482d8dcf2d5bd3b9c461806fd0dbfb5e8370805078f5e.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/53e5ac1fa0f885e2a91482d8dcf2d5bd3b9c461806fd0dbfb5e8370805078f5e.jpg)

![6789ffcf144a9cdee4641c926a80d233d46d1f443ef065caa4a485dedc940923.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/6789ffcf144a9cdee4641c926a80d233d46d1f443ef065caa4a485dedc940923.jpg)

![6ca3798ad35db2cb560bd3c42d692d083132361e47d421da62b8105d6d37438f.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/6ca3798ad35db2cb560bd3c42d692d083132361e47d421da62b8105d6d37438f.jpg)

![6e7a9ce808a8bcea0ff9d986c0d52c9007a23977a750ba0be437271a97d98b1a.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/6e7a9ce808a8bcea0ff9d986c0d52c9007a23977a750ba0be437271a97d98b1a.jpg)

![91a56538a25a60f6d61008112b28c91cdc509407eb46bc2f388db7457b3f0fc3.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/91a56538a25a60f6d61008112b28c91cdc509407eb46bc2f388db7457b3f0fc3.jpg)

![925e69ce4912be6100915fd4e1b1abb7ef2675228100896dfead255e360056a1.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/925e69ce4912be6100915fd4e1b1abb7ef2675228100896dfead255e360056a1.jpg)

![9280a242f4f57d83aea80d7162ba8c2d7195720691b29a1c7144306f8b8bd27f.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/9280a242f4f57d83aea80d7162ba8c2d7195720691b29a1c7144306f8b8bd27f.jpg)

![b0ba68a9395c67c762877aba38cb667d0cefb91e93646a9e90785a7daeff9ab7.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/b0ba68a9395c67c762877aba38cb667d0cefb91e93646a9e90785a7daeff9ab7.jpg)

![bfa78057722e8b012e05304589e8ddd05fcb884b4a69b2820f5590d296b180b2.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/bfa78057722e8b012e05304589e8ddd05fcb884b4a69b2820f5590d296b180b2.jpg)

![cd3dd7cd21235851e3c60dac15372be9e18be1fb9f736959c30fc0d578d7a30d.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/cd3dd7cd21235851e3c60dac15372be9e18be1fb9f736959c30fc0d578d7a30d.jpg)

![ceb5548bc0ac782f030b17f538912e35fd8e2d8a339ee4d5fc6e72b136fc4f29.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/ceb5548bc0ac782f030b17f538912e35fd8e2d8a339ee4d5fc6e72b136fc4f29.jpg)

![ed8eb4f8556894b6d0560eaf266149b18f3e04a824d2e190c4c9f7c6fbf06b5c.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/images/ed8eb4f8556894b6d0560eaf266149b18f3e04a824d2e190c4c9f7c6fbf06b5c.jpg)

### Tables

![a9444d7e7c6c8218b919f0e82e078420b46551bba17be5c57b6626ebe1886add.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/tables/a9444d7e7c6c8218b919f0e82e078420b46551bba17be5c57b6626ebe1886add.jpg)

![f504cfcb8287d0d8053f2b73e9eb3631e64940e278ac9e48990c466acbb0b408.jpg](../icml_results/3200_Dr.%20Kernel_%20Reinforcement%20Learning%20Done%20Right%20for%20Triton%20Kernel%20Generations/tables/f504cfcb8287d0d8053f2b73e9eb3631e64940e278ac9e48990c466acbb0b408.jpg)

<a id="paper-3162"></a>

## Sparse Relaxed-Lasso Steering: Automatic Sparse Autoencoder Feature Selection for Precise Image Editing


### Images

![079271c81cce9ae8c0014f1818fd61454bac5e448151cda7a1f81318df588b4c.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/079271c81cce9ae8c0014f1818fd61454bac5e448151cda7a1f81318df588b4c.jpg)

![0adea330ece9fd9ea263100eb82c1cfd58e90a5c66029b4d89e9a08663db4830.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/0adea330ece9fd9ea263100eb82c1cfd58e90a5c66029b4d89e9a08663db4830.jpg)

![0c4fedba46f26843a7cfe9d1138db5cb0e06606136f132671897df527076eda8.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/0c4fedba46f26843a7cfe9d1138db5cb0e06606136f132671897df527076eda8.jpg)

![12c5a974e8306411a211e81090afd45f8b054b1a81862594ab3e4b4c424bac47.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/12c5a974e8306411a211e81090afd45f8b054b1a81862594ab3e4b4c424bac47.jpg)

![35c2786cb91762faa533ea955d82940f44b2575c92ca141583b83781775aa89a.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/35c2786cb91762faa533ea955d82940f44b2575c92ca141583b83781775aa89a.jpg)

![3961e0329e86fa9ee74bc97a8f732e934130c9f6d190ec0d6dffd8b1aeaf4b2f.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/3961e0329e86fa9ee74bc97a8f732e934130c9f6d190ec0d6dffd8b1aeaf4b2f.jpg)

![3d798aa44cba255621dacb79e9d62b19a4a74d39a8bf689f06b30f9fd44f5cf9.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/3d798aa44cba255621dacb79e9d62b19a4a74d39a8bf689f06b30f9fd44f5cf9.jpg)

![3de58fe619005ab5e69548151bd75d2d097f8740528026c8d41c3a25dad35dc1.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/3de58fe619005ab5e69548151bd75d2d097f8740528026c8d41c3a25dad35dc1.jpg)

![44d590a3d4fc46436ec92cdf325b5e37367665a087d59cf8aa8952e838cb2544.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/44d590a3d4fc46436ec92cdf325b5e37367665a087d59cf8aa8952e838cb2544.jpg)

![59fd64e9934de9945f34e1627a46db8c4aa3c18d6c80b1e492ae853e1220d590.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/59fd64e9934de9945f34e1627a46db8c4aa3c18d6c80b1e492ae853e1220d590.jpg)

![7942a06ef7ed6acf77c33090e4344809fc5dc975a74d321c6d6f29e505d20d48.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/7942a06ef7ed6acf77c33090e4344809fc5dc975a74d321c6d6f29e505d20d48.jpg)

![81e36253132eb2058e2fa37b295a6e49486d7c04e59f21b60608a3b46a942f4c.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/81e36253132eb2058e2fa37b295a6e49486d7c04e59f21b60608a3b46a942f4c.jpg)

![96b2d394e8d6a36b96c82800f79f2af2da4b9bf3bd7675df5f7b6da8a4c9854a.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/96b2d394e8d6a36b96c82800f79f2af2da4b9bf3bd7675df5f7b6da8a4c9854a.jpg)

![96befa8cbbac2bc1a5a6a427fc80aab2e527b2c8cc8225e2c207f2ccc7828a7f.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/96befa8cbbac2bc1a5a6a427fc80aab2e527b2c8cc8225e2c207f2ccc7828a7f.jpg)

![99d58e154421af51459c675e54d14596546e797a6f9787123aff03ae3c29f282.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/99d58e154421af51459c675e54d14596546e797a6f9787123aff03ae3c29f282.jpg)

![ab7576bb96e3aed367a99c468aec77f52d6c73e3236f5395e51a66ea524a0eff.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/ab7576bb96e3aed367a99c468aec77f52d6c73e3236f5395e51a66ea524a0eff.jpg)

![d0a1349b972e6972600da170a7b28bfbb3a1d27fb93f4d69c5d01b504237b12f.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/d0a1349b972e6972600da170a7b28bfbb3a1d27fb93f4d69c5d01b504237b12f.jpg)

![dce66c837a34496be50340838b0d3b6c5be07b040890482a16cc1a3d31eb88db.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/dce66c837a34496be50340838b0d3b6c5be07b040890482a16cc1a3d31eb88db.jpg)

![f5856da52ecc3bb87df29ce5522ecee809499aca31c178b6d83c328460ce3bef.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/f5856da52ecc3bb87df29ce5522ecee809499aca31c178b6d83c328460ce3bef.jpg)

![f9a1238f4141f0dd55170397f95fee68cb6cd48fdaa9d6963792f48acf295f34.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/images/f9a1238f4141f0dd55170397f95fee68cb6cd48fdaa9d6963792f48acf295f34.jpg)

### Tables

![29959004dce1e8047b56b855ac595dfd413841f95e6feec3b1940c2b9973ff9c.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/29959004dce1e8047b56b855ac595dfd413841f95e6feec3b1940c2b9973ff9c.jpg)

![555e0264694c5187adb2c56f24a607f35f6c66f691cec31d32db095ba25dab37.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/555e0264694c5187adb2c56f24a607f35f6c66f691cec31d32db095ba25dab37.jpg)

![64c7a7ef71fdd74da438ddc9b0c46a1ceb3e9af9c3023cafbc14431bc6b7df28.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/64c7a7ef71fdd74da438ddc9b0c46a1ceb3e9af9c3023cafbc14431bc6b7df28.jpg)

![738bdd1164782c527595486f7ea2ccc90559a13b1042a31cd69706ff752d11b3.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/738bdd1164782c527595486f7ea2ccc90559a13b1042a31cd69706ff752d11b3.jpg)

![74eec3c04756de3bd6fdc83f3b85fc2263420e2736784d15018bd6fb38254d54.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/74eec3c04756de3bd6fdc83f3b85fc2263420e2736784d15018bd6fb38254d54.jpg)

![98652345d473717cbfcec815e7e989cd0edc831eb5b18ab55bb9f2b52ef5ebb1.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/98652345d473717cbfcec815e7e989cd0edc831eb5b18ab55bb9f2b52ef5ebb1.jpg)

![b49d0db277c8ed99e4597bb456eeb91c26bb87c9a4c6b9342a72eb0178bdd61a.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/b49d0db277c8ed99e4597bb456eeb91c26bb87c9a4c6b9342a72eb0178bdd61a.jpg)

![c682a943b6307c30a5674fcb44cf8a8f86e2d7827da01d34e18b661e3ec92972.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/c682a943b6307c30a5674fcb44cf8a8f86e2d7827da01d34e18b661e3ec92972.jpg)

![e70f99f23a57c663c19e1e8a18b3c628fc5e7e22ba634aca34cec2b74f7d18f4.jpg](../icml_results/3201_Sparse%20Relaxed-Lasso%20Steering_%20Automatic%20Sparse%20Autoencoder%20Feature%20Selection%20for%20Precise%20Image%20Edit/tables/e70f99f23a57c663c19e1e8a18b3c628fc5e7e22ba634aca34cec2b74f7d18f4.jpg)

<a id="paper-3163"></a>

## Anchor-Final Self-Supervision Drives Hallucination-Aware Optimization in Large Vision-Language Models


### Images

![0bdb1dfc610a906335bbe431d000cc37ae66fcb1f948786b068b7c6b27a096f6.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/images/0bdb1dfc610a906335bbe431d000cc37ae66fcb1f948786b068b7c6b27a096f6.jpg)

![66144cc93f21bd04ea119e87b3764789b58dab41957f2027501f0d273710d2a5.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/images/66144cc93f21bd04ea119e87b3764789b58dab41957f2027501f0d273710d2a5.jpg)

### Tables

![1cdf91b72e8ff5e6f19569157441a0b643e5acedae6be494e8d056e70be4b688.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/1cdf91b72e8ff5e6f19569157441a0b643e5acedae6be494e8d056e70be4b688.jpg)

![39c7f6db61ef099a3a3fba5c3865b11bc3c20c15c6be38a32c753006cd7fc4ac.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/39c7f6db61ef099a3a3fba5c3865b11bc3c20c15c6be38a32c753006cd7fc4ac.jpg)

![4fec0aa17472753a9d4d72a1c623ccfd7ff0842109cf8631ac9004c3429d7a3e.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/4fec0aa17472753a9d4d72a1c623ccfd7ff0842109cf8631ac9004c3429d7a3e.jpg)

![59a6df1e6be76075f47a1b7a379ffbc0210471530c0e98ba9810288c32605f11.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/59a6df1e6be76075f47a1b7a379ffbc0210471530c0e98ba9810288c32605f11.jpg)

![6ddc5950c57c6ff0e2f0609354a3fc7d1235184967a1be8f9ba02995f9b87fa2.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/6ddc5950c57c6ff0e2f0609354a3fc7d1235184967a1be8f9ba02995f9b87fa2.jpg)

![8b5567ce4fcaec8790aa2ec42f514969a34897e3a96e52053ca543a854338e66.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/8b5567ce4fcaec8790aa2ec42f514969a34897e3a96e52053ca543a854338e66.jpg)

![9014c89c4e140f4aed78dad101d86b8d5cc83a830a1eb7b4b3de8d8405454031.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/9014c89c4e140f4aed78dad101d86b8d5cc83a830a1eb7b4b3de8d8405454031.jpg)

![9325ce3920a8546ba8899702d5929bed62346ca2cef6a6eb83097fda5f5e15ac.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/9325ce3920a8546ba8899702d5929bed62346ca2cef6a6eb83097fda5f5e15ac.jpg)

![99eb8c2d778cc1c0f8e5b0e13d08c223936aa598a4b531d14082760513ea31b3.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/99eb8c2d778cc1c0f8e5b0e13d08c223936aa598a4b531d14082760513ea31b3.jpg)

![a67a0284d395f0cc1cc945be9b2459f1f797458491aebb8947edfb6f5834fd14.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/a67a0284d395f0cc1cc945be9b2459f1f797458491aebb8947edfb6f5834fd14.jpg)

![b7863c3b14b3a42defed7e3af1ea66c888ef79e6fcf13b5cb576a2aaf93f6954.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/b7863c3b14b3a42defed7e3af1ea66c888ef79e6fcf13b5cb576a2aaf93f6954.jpg)

![bd2148d9e9a43305d100b636b6f57a8f8ca686f65123088752640cc06020d24d.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/bd2148d9e9a43305d100b636b6f57a8f8ca686f65123088752640cc06020d24d.jpg)

![e32931d1c7132477d331cff933107bfc4332bde9cf5693ad71425a5ffb9dd150.jpg](../icml_results/3202_Anchor-Final%20Self-Supervision%20Drives%20Hallucination-Aware%20Optimization%20in%20Large%20Vision-Language%20Model/tables/e32931d1c7132477d331cff933107bfc4332bde9cf5693ad71425a5ffb9dd150.jpg)

<a id="paper-3164"></a>

## M-IDoL: Information Decomposition for Modality-Specific and Diverse Representation Learning in Medical Foundation Model


### Images

![0031a257534eb9e518f172d79d7aa23e7dabeb4ccd0fe793be5e32b31e9e273b.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/0031a257534eb9e518f172d79d7aa23e7dabeb4ccd0fe793be5e32b31e9e273b.jpg)

![01452d2202e635f24575b27343a10602438b7eca3d4f4ccc68e7c1c9f172d025.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/01452d2202e635f24575b27343a10602438b7eca3d4f4ccc68e7c1c9f172d025.jpg)

![045908581f1e1aac49839664bfa09a8c44b04ed7089438d4ef643deb720b326c.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/045908581f1e1aac49839664bfa09a8c44b04ed7089438d4ef643deb720b326c.jpg)

![329bc2e0fb187c489bd4e7ef82c8f1c72bbd7bcc5c2fb6749b026bebcc6eda4c.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/329bc2e0fb187c489bd4e7ef82c8f1c72bbd7bcc5c2fb6749b026bebcc6eda4c.jpg)

![bd847c322907e5b8278e88eba981a5d9c707e327c360f46e2a17f41b4a999ad7.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/bd847c322907e5b8278e88eba981a5d9c707e327c360f46e2a17f41b4a999ad7.jpg)

![e625d313dc90a667f8952d821c3932a80755c20a32269e409b83e547ae6954ca.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/e625d313dc90a667f8952d821c3932a80755c20a32269e409b83e547ae6954ca.jpg)

![f58d7d6e40a8894d92f5fc825c5632ee406607ee9876f48263139c1e5e8a79f2.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/f58d7d6e40a8894d92f5fc825c5632ee406607ee9876f48263139c1e5e8a79f2.jpg)

![f7f71a9c681182b57aa72252e697e8f72ebee71e57bb1f7f8894f7e48d5df9bb.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/images/f7f71a9c681182b57aa72252e697e8f72ebee71e57bb1f7f8894f7e48d5df9bb.jpg)

### Tables

![50b0e538c57b8506041e0bf588f93d5a8552439059e1d4b74b1640488c709d28.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/tables/50b0e538c57b8506041e0bf588f93d5a8552439059e1d4b74b1640488c709d28.jpg)

![a2ea2d31abf5a0dd4fdb10748b73cfb9d0df0aa0152d5778edd500845da1c1ed.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/tables/a2ea2d31abf5a0dd4fdb10748b73cfb9d0df0aa0152d5778edd500845da1c1ed.jpg)

![e38bbc0eccc56908e1e70e82377a954ff14c7a6987d8ef323bc72a2891851393.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/tables/e38bbc0eccc56908e1e70e82377a954ff14c7a6987d8ef323bc72a2891851393.jpg)

![ed34d35733426074af14c29a1f4e0b0851a32d3f32c0f6766bace8e35877334d.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/tables/ed34d35733426074af14c29a1f4e0b0851a32d3f32c0f6766bace8e35877334d.jpg)

![f9943987af88de83dfff6bf600c821fda2a7f4931612c790837baced87e1b155.jpg](../icml_results/3203_M-IDoL_%20Information%20Decomposition%20for%20Modality-Specific%20and%20Diverse%20Representation%20Learning%20in%20Medic/tables/f9943987af88de83dfff6bf600c821fda2a7f4931612c790837baced87e1b155.jpg)

<a id="paper-3165"></a>

## Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling


### Images

![02dd837f5327b077905a1d290e60adf8348268b416f2fc891a31f39c239f9186.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/02dd837f5327b077905a1d290e60adf8348268b416f2fc891a31f39c239f9186.jpg)

![26f94983dd9e03b237d24cfed2fce19bcfa8ed03b758ec2ef397aed08829c62c.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/26f94983dd9e03b237d24cfed2fce19bcfa8ed03b758ec2ef397aed08829c62c.jpg)

![4e8632c316c82d6a1b0e26152ccd6401939a4e6e9a517cc0d625947bd8368f7c.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/4e8632c316c82d6a1b0e26152ccd6401939a4e6e9a517cc0d625947bd8368f7c.jpg)

![54fb4a88fcc25b5e1c031f33aae7c7f919e2ee499d5b96ac299324c9b1904dcc.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/54fb4a88fcc25b5e1c031f33aae7c7f919e2ee499d5b96ac299324c9b1904dcc.jpg)

![6abbd9e8a92e42cf7804ca72aed4980088b7ad56a1766180e8254f1b86961f37.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/6abbd9e8a92e42cf7804ca72aed4980088b7ad56a1766180e8254f1b86961f37.jpg)

![839ea3ed1591784b8a4d59a672b1a6e60ad96a278007770a075881c7420235ca.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/839ea3ed1591784b8a4d59a672b1a6e60ad96a278007770a075881c7420235ca.jpg)

![87b3f576bd98e2b7aa0e625be456019909cc832687409fc9618e3ca6748334b8.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/87b3f576bd98e2b7aa0e625be456019909cc832687409fc9618e3ca6748334b8.jpg)

![b5f3b99af5bffbb1686f38cc06491be695907da5554464f9e46a0fbb6588950f.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/b5f3b99af5bffbb1686f38cc06491be695907da5554464f9e46a0fbb6588950f.jpg)

![c516d8aa6643ff46420ef1bcb0c760ee7fdaedc850dc2519c686b2f695822680.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/c516d8aa6643ff46420ef1bcb0c760ee7fdaedc850dc2519c686b2f695822680.jpg)

![d2f6b9c3790a422db1f12145462e4a0a680c32dc08476e0f1926ef543485f31d.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/d2f6b9c3790a422db1f12145462e4a0a680c32dc08476e0f1926ef543485f31d.jpg)

![e1b2ff58d17e0c2d338ec785f3740b0f40dea8adf8022f9984fb76c51c3ca54e.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/images/e1b2ff58d17e0c2d338ec785f3740b0f40dea8adf8022f9984fb76c51c3ca54e.jpg)

### Tables

![0ba3729b320587d0d4ab431d5afb67f2a4b370d80613284f2c1a7da0dabffd4c.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/0ba3729b320587d0d4ab431d5afb67f2a4b370d80613284f2c1a7da0dabffd4c.jpg)

![569edb20c95c4922d9fe13481a866b270248c86520197d0f9bed3ec995489185.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/569edb20c95c4922d9fe13481a866b270248c86520197d0f9bed3ec995489185.jpg)

![59d7c427357c91b293e60472999afab25b24a288c2860698a9b7aacd0915dce0.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/59d7c427357c91b293e60472999afab25b24a288c2860698a9b7aacd0915dce0.jpg)

![59ee2823e6e9c7ad0f0f7a297ef9f22bf2f2d541d0692daf3dfc584bb7b2c940.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/59ee2823e6e9c7ad0f0f7a297ef9f22bf2f2d541d0692daf3dfc584bb7b2c940.jpg)

![690bcf972fca7ca1a88966bf1f76892866dda8541f32495c1ab39d4f28bad1c7.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/690bcf972fca7ca1a88966bf1f76892866dda8541f32495c1ab39d4f28bad1c7.jpg)

![6bbef1e7aa1f785c7cbc484f4294fb5261fc9a67519839576622bd401f74edc8.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/6bbef1e7aa1f785c7cbc484f4294fb5261fc9a67519839576622bd401f74edc8.jpg)

![7f9d1bcae363e7cd3e1e23eefe45519432510ecc8ad4ec2261a8c3409f2f0e82.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/7f9d1bcae363e7cd3e1e23eefe45519432510ecc8ad4ec2261a8c3409f2f0e82.jpg)

![89ad14062586f71fcb0a2d13edb3e5964193bf1f4a16a3b8b2cbdc8c2c0f1007.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/89ad14062586f71fcb0a2d13edb3e5964193bf1f4a16a3b8b2cbdc8c2c0f1007.jpg)

![92e1b76e17bbaac4ce583f33c69ef86adc7bae1cdf505aa77c9687024b9afa51.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/92e1b76e17bbaac4ce583f33c69ef86adc7bae1cdf505aa77c9687024b9afa51.jpg)

![a131b235f66dfaea6f2d8f5f1bb94c585da6a1782e3723f78288c145cb4b5bae.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/a131b235f66dfaea6f2d8f5f1bb94c585da6a1782e3723f78288c145cb4b5bae.jpg)

![a4335de42e6fb8913ece3c2f381c3261a156f9628c67c8f9fa378065bee77fc2.jpg](../icml_results/3204_Beyond%20VLM-Based%20Rewards_%20Diffusion-Native%20Latent%20Reward%20Modeling/tables/a4335de42e6fb8913ece3c2f381c3261a156f9628c67c8f9fa378065bee77fc2.jpg)

<a id="paper-3166"></a>

## dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching


### Images

![4defe3d6305fc31463efc535b7238f72b50563a216df28537eb23b9efc50d508.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/4defe3d6305fc31463efc535b7238f72b50563a216df28537eb23b9efc50d508.jpg)

![63f4b23d417fa84d2b1cf7e07b1f600bada0ce763098791c70d4974ea9686a57.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/63f4b23d417fa84d2b1cf7e07b1f600bada0ce763098791c70d4974ea9686a57.jpg)

![704eb59ca10cfbc9a67169d97d3e25ff1f4d379f88fd8462159b0add81347ae6.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/704eb59ca10cfbc9a67169d97d3e25ff1f4d379f88fd8462159b0add81347ae6.jpg)

![e4719e85fc1fe53952f6bbfb4366c0188e6f480f842b97ff9235e78cc6763ea7.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/e4719e85fc1fe53952f6bbfb4366c0188e6f480f842b97ff9235e78cc6763ea7.jpg)

![e69b189f058ba94ab411187396db65266d8ebcd67968ed826a61e2b2f89d53f4.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/e69b189f058ba94ab411187396db65266d8ebcd67968ed826a61e2b2f89d53f4.jpg)

![fe278205fcdd1a08b22395f42df7633d31d724926461ab27a8792c0f78c647bd.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/images/fe278205fcdd1a08b22395f42df7633d31d724926461ab27a8792c0f78c647bd.jpg)

### Tables

![08e92030848599468563ad8b778bec6be8dab0a017b6893d0ccec970fd3a70a0.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/08e92030848599468563ad8b778bec6be8dab0a017b6893d0ccec970fd3a70a0.jpg)

![13635c6370f2a7355ece2eb457f5f77737daa396612ee562ee4a8959205fc40c.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/13635c6370f2a7355ece2eb457f5f77737daa396612ee562ee4a8959205fc40c.jpg)

![163d67a958db2b25a303a2738d65e40551ef390eac8f034b9681111c43110efe.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/163d67a958db2b25a303a2738d65e40551ef390eac8f034b9681111c43110efe.jpg)

![219fc84b4a9ef13e33fdb914c96e7cb42650a9ae262ce034e762f9e6b6f0eef4.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/219fc84b4a9ef13e33fdb914c96e7cb42650a9ae262ce034e762f9e6b6f0eef4.jpg)

![32b4f389cb50096f81b4807d259d851776648786ed186120149a223b83f1ef8f.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/32b4f389cb50096f81b4807d259d851776648786ed186120149a223b83f1ef8f.jpg)

![36ee0e4a1b37442c77db3d229d4978ad2ea3b962eb95de06d9f734d66c0123c2.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/36ee0e4a1b37442c77db3d229d4978ad2ea3b962eb95de06d9f734d66c0123c2.jpg)

![3d410dac48ac2e3bce5a5f3e55ee5674efe0ee3bcf72fd7c6e0d0df4ee1bae69.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/3d410dac48ac2e3bce5a5f3e55ee5674efe0ee3bcf72fd7c6e0d0df4ee1bae69.jpg)

![6e81ad5f90d3db2a6d1f70e7ca9d9aea025f46b03b1985fc01202a2d848800f4.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/6e81ad5f90d3db2a6d1f70e7ca9d9aea025f46b03b1985fc01202a2d848800f4.jpg)

![7ea6601e961d9e9f4bf689f0205bc17611afa30b2b9999325537961d73b77a29.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/7ea6601e961d9e9f4bf689f0205bc17611afa30b2b9999325537961d73b77a29.jpg)

![7ecb609b9b7ab451fc9fbe619d3fde5b514244c50570f26869cfffbf16d83636.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/7ecb609b9b7ab451fc9fbe619d3fde5b514244c50570f26869cfffbf16d83636.jpg)

![7fee6cfad4be2d5c5f13712e080ebed6da4e2751e3497ff140912bc395960c7b.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/7fee6cfad4be2d5c5f13712e080ebed6da4e2751e3497ff140912bc395960c7b.jpg)

![90577d7f69f775d719921d60a3f70fe6c1a1646411280279b51aa41992383c80.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/90577d7f69f775d719921d60a3f70fe6c1a1646411280279b51aa41992383c80.jpg)

![95d4a57f8f68ac82a911cc424213bbaab6fc7f4f290d9140b1eddfb70b71321d.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/95d4a57f8f68ac82a911cc424213bbaab6fc7f4f290d9140b1eddfb70b71321d.jpg)

![a78dd6fb0e0247ed777b1e56d50396388af80be03f62f008d13206c0aeb5b2a7.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/a78dd6fb0e0247ed777b1e56d50396388af80be03f62f008d13206c0aeb5b2a7.jpg)

![a9601d6fbec494f6d6ac1b222282c0eff2d78bf5313a56bb267d5f0dfdfee711.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/a9601d6fbec494f6d6ac1b222282c0eff2d78bf5313a56bb267d5f0dfdfee711.jpg)

![e923ec1d95c9852b862f580e681d89d8cacf3659942490212c891c513d403631.jpg](../icml_results/3205_dLLM-Cache_%20Accelerating%20Diffusion%20Large%20Language%20Models%20with%20Adaptive%20Caching/tables/e923ec1d95c9852b862f580e681d89d8cacf3659942490212c891c513d403631.jpg)

<a id="paper-3167"></a>

## Lookahead Path Likelihood Optimization for Diffusion LLMs


### Images

![166899bb440f49331dddf3df0ebd438d890f318abebf45169f501d74b33f2c88.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/166899bb440f49331dddf3df0ebd438d890f318abebf45169f501d74b33f2c88.jpg)

![2fe5744db44d7ad664093340fc50147192f0492c4283727f5175f2417c7996b4.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/2fe5744db44d7ad664093340fc50147192f0492c4283727f5175f2417c7996b4.jpg)

![3bc90d60bd2fc00149ebf2edd3752c4604df7e9fd11eb1a9963af716b7bb6c8a.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/3bc90d60bd2fc00149ebf2edd3752c4604df7e9fd11eb1a9963af716b7bb6c8a.jpg)

![556b2eb9707b702e079c8dee987109e6bb9329967b05a2a5c3e0a686a85a9248.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/556b2eb9707b702e079c8dee987109e6bb9329967b05a2a5c3e0a686a85a9248.jpg)

![6ed6fe1edfeb7b8a5f17b9352fb098849f72959650824ea3e6ddd1c3da9ea2cf.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/6ed6fe1edfeb7b8a5f17b9352fb098849f72959650824ea3e6ddd1c3da9ea2cf.jpg)

![75b5fb2dbfc034315a43a50003ff166c66d3427e1b7765892e70b2d9438a796f.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/75b5fb2dbfc034315a43a50003ff166c66d3427e1b7765892e70b2d9438a796f.jpg)

![9976107ff21cdd5ad8632dca5d43025a58b01481d48c70f861319639e6ffe560.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/9976107ff21cdd5ad8632dca5d43025a58b01481d48c70f861319639e6ffe560.jpg)

![a2acdf8d016c8ebdfa9ae78cc29972e954f3ba6fdaef23aafdb2a2962042e134.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/a2acdf8d016c8ebdfa9ae78cc29972e954f3ba6fdaef23aafdb2a2962042e134.jpg)

![b596838b331899a04c238406fecdc74011955860ce6d17f5fb78b1a7ca9e04ba.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/b596838b331899a04c238406fecdc74011955860ce6d17f5fb78b1a7ca9e04ba.jpg)

![e133669e8a5ad59131b2bf8da877e8fbed7d7e56fa58b62fd7abf658b357f783.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/e133669e8a5ad59131b2bf8da877e8fbed7d7e56fa58b62fd7abf658b357f783.jpg)

![f0b3da711b7d8f55883df7ef0396fdc2c460138669f79c781e57a68118db55b1.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/f0b3da711b7d8f55883df7ef0396fdc2c460138669f79c781e57a68118db55b1.jpg)

![f1d8940c77d3ea848c58a1600a9020c73b042c5b9b4455ab1d7dd5cfbd1fa048.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/images/f1d8940c77d3ea848c58a1600a9020c73b042c5b9b4455ab1d7dd5cfbd1fa048.jpg)

### Tables

![0357ccb3460c77f94dfe25511f4e7564827359de4a13312827afd8d38654025a.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/0357ccb3460c77f94dfe25511f4e7564827359de4a13312827afd8d38654025a.jpg)

![3dab94128574ea2163c92bdd11c394d43524d5c767e7625dcbad7574fc5045b3.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/3dab94128574ea2163c92bdd11c394d43524d5c767e7625dcbad7574fc5045b3.jpg)

![45d9b3cac68fa5786774fcaf5eb54b7edf553a435c44641b907b8eb0ca924a0a.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/45d9b3cac68fa5786774fcaf5eb54b7edf553a435c44641b907b8eb0ca924a0a.jpg)

![503740631ceb63c089b310e1e79895cc1d8fd47779f16297420026e4e80e255e.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/503740631ceb63c089b310e1e79895cc1d8fd47779f16297420026e4e80e255e.jpg)

![7f5e14769e7a30932850d008d5ca176505ba8c30c8b67703d6db74cbcf8d7ae5.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/7f5e14769e7a30932850d008d5ca176505ba8c30c8b67703d6db74cbcf8d7ae5.jpg)

![8c67071613aea2e92cfc5638a2cad33ee8b55318a31591e18e2d3f9d959d0a98.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/8c67071613aea2e92cfc5638a2cad33ee8b55318a31591e18e2d3f9d959d0a98.jpg)

![a47689cdef839a1c778de23473ea69a9d31ac3b9ea06f7fe5b465ee22aba7648.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/a47689cdef839a1c778de23473ea69a9d31ac3b9ea06f7fe5b465ee22aba7648.jpg)

![ac8b88da8ef34157b9f5b785d373a547f837093b064aebfd1a65518b3cbc7759.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/ac8b88da8ef34157b9f5b785d373a547f837093b064aebfd1a65518b3cbc7759.jpg)

![ad88775680d1cd376d209ac3037823d8ea1c32595b16b65e491d4ea3ea220e86.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/ad88775680d1cd376d209ac3037823d8ea1c32595b16b65e491d4ea3ea220e86.jpg)

![b7d0a4fd79900206dd1b3289340bcb5f3c4e12baa571bfc6d8cd3671de0c00ed.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/b7d0a4fd79900206dd1b3289340bcb5f3c4e12baa571bfc6d8cd3671de0c00ed.jpg)

![b80100eda501071d478526e3cc982c599701e48e79c60256aa4803591346046a.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/b80100eda501071d478526e3cc982c599701e48e79c60256aa4803591346046a.jpg)

![c360d4aa1def55d5b9ac357c0cda1f48baeb191cc865becceaafe999518388e0.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/c360d4aa1def55d5b9ac357c0cda1f48baeb191cc865becceaafe999518388e0.jpg)

![c5ada212b8223452c572641e8d3611645ab0c2a41358b4104c67c5fc5a436dbf.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/c5ada212b8223452c572641e8d3611645ab0c2a41358b4104c67c5fc5a436dbf.jpg)

![cb85dbd75533a304fa908cd635a4b2f672f8de0b3d8e81b4442f4295247a5b43.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/cb85dbd75533a304fa908cd635a4b2f672f8de0b3d8e81b4442f4295247a5b43.jpg)

![d05c214465edf45c368f76dff9651baa5604a94fb01a340652a8da301f9b7f55.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/d05c214465edf45c368f76dff9651baa5604a94fb01a340652a8da301f9b7f55.jpg)

![d641370b8810145863681c4c686e1c9d592180f138316137a471f905465eeed3.jpg](../icml_results/3206_Lookahead%20Path%20Likelihood%20Optimization%20for%20Diffusion%20LLMs/tables/d641370b8810145863681c4c686e1c9d592180f138316137a471f905465eeed3.jpg)

<a id="paper-3168"></a>

## FlowCloud: Learning Continuous Spatiotemporal Dynamics from Unpaired Sparse Point Cloud Snapshots

### Images

![134914d305e7bffac6a98ff41e2f7e7d7eb6ea2df888bdfbce7229f6e69223b4.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/134914d305e7bffac6a98ff41e2f7e7d7eb6ea2df888bdfbce7229f6e69223b4.jpg)

![1c3a48d482a5557aaa068bef935b390243cf90a208eab753de0bdd1b7cea9836.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/1c3a48d482a5557aaa068bef935b390243cf90a208eab753de0bdd1b7cea9836.jpg)

![36fba06a39506a628d7ff44b44ab46ac40cc5158259671277bc7266dfa26a175.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/36fba06a39506a628d7ff44b44ab46ac40cc5158259671277bc7266dfa26a175.jpg)

![486a20b20c0efce64a30d9413d0ffc61208dc93655a2995b89e9b3e89aa9321f.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/486a20b20c0efce64a30d9413d0ffc61208dc93655a2995b89e9b3e89aa9321f.jpg)

![4d577fc5e5c8f8ffec3d44ec6dcd3758b78a3e467e70e311d9468d1a1eef5123.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/4d577fc5e5c8f8ffec3d44ec6dcd3758b78a3e467e70e311d9468d1a1eef5123.jpg)

![4e40f11d65cbd80ab3d6fd9fa82963f96de0a5fb99cf2012464f996ea38cb17a.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/4e40f11d65cbd80ab3d6fd9fa82963f96de0a5fb99cf2012464f996ea38cb17a.jpg)

![5097f446880b4a6e2153f596f67d56c7b5f5cd9b808a07fa460c62ee634f5ae5.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/5097f446880b4a6e2153f596f67d56c7b5f5cd9b808a07fa460c62ee634f5ae5.jpg)

![561e3f118f0c497743218c728840c055eb2d41b111f0d32d818896f50e95cb88.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/561e3f118f0c497743218c728840c055eb2d41b111f0d32d818896f50e95cb88.jpg)

![81eef2719c25fa3b9123712171d53ca23288572fa5d4bf349da9a19cdd819447.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/81eef2719c25fa3b9123712171d53ca23288572fa5d4bf349da9a19cdd819447.jpg)

![8d3cfee614102e7840c0a987133bb45f7f23c60a73ceb99c13ba89e280890d43.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/8d3cfee614102e7840c0a987133bb45f7f23c60a73ceb99c13ba89e280890d43.jpg)

![9640592c1498c09412f4af018d03535cb406aa6c1b70c950624997c3c467e451.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/9640592c1498c09412f4af018d03535cb406aa6c1b70c950624997c3c467e451.jpg)

![984a7e6cc97554eb59b0d15c1883a7601f6891c7e6c8b3ec6335459093b5c4fc.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/984a7e6cc97554eb59b0d15c1883a7601f6891c7e6c8b3ec6335459093b5c4fc.jpg)

![9eb923098248c5411e398ad6616bb9d7d2dce00d35b76cdd38245475709eb89c.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/9eb923098248c5411e398ad6616bb9d7d2dce00d35b76cdd38245475709eb89c.jpg)

![9ec6ff37b77db42265d64a68685339614cbda030d8dd54df2c6da592d412b4da.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/9ec6ff37b77db42265d64a68685339614cbda030d8dd54df2c6da592d412b4da.jpg)

![d8a7804633e062c98f9a74cca785ae64a4ed903a4a240480503191ccdb214348.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/d8a7804633e062c98f9a74cca785ae64a4ed903a4a240480503191ccdb214348.jpg)

![db81502f394d2f2c345737404adbd97ab650b0c69f37f51eadc4d2b69c7c72bb.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/db81502f394d2f2c345737404adbd97ab650b0c69f37f51eadc4d2b69c7c72bb.jpg)

![f16a94f539c49ab472363fbf17deff7d2ecf64bafbb534c11b0f949e71458529.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/images/f16a94f539c49ab472363fbf17deff7d2ecf64bafbb534c11b0f949e71458529.jpg)

### Tables

![05dda1a46209f0812b16f65b56dae2b3122733123703ada488fd5da8fab267ab.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/05dda1a46209f0812b16f65b56dae2b3122733123703ada488fd5da8fab267ab.jpg)

![0c4cdce39390521b11cb3b342bfaa5bd6402e28d1c13370a76775c888e1d4f3b.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/0c4cdce39390521b11cb3b342bfaa5bd6402e28d1c13370a76775c888e1d4f3b.jpg)

![5241e7e2cb52727416fc2b554c7a344c799841d62623ddef7287f979391dfc9c.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/5241e7e2cb52727416fc2b554c7a344c799841d62623ddef7287f979391dfc9c.jpg)

![66f566ecfed4135c2e02b4bd1f65272aefe64ba7fa245da9ceabb5efbca26e35.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/66f566ecfed4135c2e02b4bd1f65272aefe64ba7fa245da9ceabb5efbca26e35.jpg)

![684f4cb4ec7d48310f62c3e371a9a1c11334839b48d565cb5de714a581165afa.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/684f4cb4ec7d48310f62c3e371a9a1c11334839b48d565cb5de714a581165afa.jpg)

![75609a537a90cc596613e5224758162e0c591fc8a40e9f0d87147eb7c1a013f6.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/75609a537a90cc596613e5224758162e0c591fc8a40e9f0d87147eb7c1a013f6.jpg)

![88335ea8b3a8de70c81e1f441dbf316ed27023e1228dfbfe3c6bb9327657bebd.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/88335ea8b3a8de70c81e1f441dbf316ed27023e1228dfbfe3c6bb9327657bebd.jpg)

![9ced2bce489aa8bb55ea885036baf809ed550a535878202b6a4f5c675169ed39.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/9ced2bce489aa8bb55ea885036baf809ed550a535878202b6a4f5c675169ed39.jpg)

![a4bb267aacaca29bce28477b2a832c2d96f180a5f13a8a7b7be2ed55700925d2.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/a4bb267aacaca29bce28477b2a832c2d96f180a5f13a8a7b7be2ed55700925d2.jpg)

![c9e3e5ae6ee5dbc11c2a25e66aa2ba0079e375dcd603376710e1cd3f1632517f.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/c9e3e5ae6ee5dbc11c2a25e66aa2ba0079e375dcd603376710e1cd3f1632517f.jpg)

![d13465940055f89883762f639df732627ffb797ba1d50cf245bb339deacf96d8.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/d13465940055f89883762f639df732627ffb797ba1d50cf245bb339deacf96d8.jpg)

![d26f21c9c0d970b652fdd74b6837474000566952aa8bf0806686a458e3e35d66.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/d26f21c9c0d970b652fdd74b6837474000566952aa8bf0806686a458e3e35d66.jpg)

![e752889db4bf0155936b18214ca9698d74e260ac33b6f58a702c88434eadd2e6.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/e752889db4bf0155936b18214ca9698d74e260ac33b6f58a702c88434eadd2e6.jpg)

![f301fcf3d68b195be708f7db1fbeb52d675b6ed2915e814cef73c0e55efe53a0.jpg](../icml_results/3207_FlowCloud_%20Learning%20Continuous%20Spatiotemporal%20Dynamics%20from%20Unpaired%20Sparse%20Point%20Cloud%20Snapshots/tables/f301fcf3d68b195be708f7db1fbeb52d675b6ed2915e814cef73c0e55efe53a0.jpg)
