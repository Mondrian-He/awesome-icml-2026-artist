# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 46 of 197

## 目录 (Table of Contents)

1. [Compositional Planning with Jumpy World Models](#paper-1486)
2. [Enhancing Conformal Prediction via Class Similarity](#paper-1487)
3. [Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control](#paper-1488)
4. [Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach](#paper-1489)
5. [$f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](#paper-1490)
6. [COFT: Counterfactual–Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](#paper-1491)
7. [Dissecting Quantization Error: A Concentration-Alignment Perspective](#paper-1492)
8. [Learning Interpretable Options by Identifying Reward Diffusion Bottlenecks in Reinforcement Learning](#paper-1493)
9. [Olivia: Harmonizing Time Series Foundation Models with Power Spectral Density](#paper-1494)
10. [One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](#paper-1495)
11. [Efficient privacy loss accounting for subsampling and random allocation](#paper-1496)
12. [Distributionally Robust Causal Abstractions](#paper-1497)
13. [Optimal Self-Consistency for Efficient Reasoning with Large Language Models](#paper-1498)
14. [LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification](#paper-1499)
15. [SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems](#paper-1500)
16. [WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](#paper-1501)
17. [What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT](#paper-1502)
18. [Leveraging Machine Unlearning for Cost-Efficient Preference Alignment](#paper-1503)
19. [Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](#paper-1504)
20. [Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning](#paper-1505)
21. [ExpWeaver: LLM Agents Learn from Experience via Latent RAG](#paper-1506)
22. [CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](#paper-1507)
23. [Understanding Dynamics of Adam in Zero-Sum Games: An ODE Approach](#paper-1508)
24. [Convex Optimization for Alignment and Preference Learning on a Single GPU](#paper-1509)
25. [Knowledge Diversion for Efficient Morphology Control and Policy Transfer](#paper-1510)
26. [Unsupervised Diffusion Solver for Combinatorial Optimization via Combinatorial Adjoint Matching](#paper-1511)
27. [Convex Low-resource Accent-Robust Language Detection in Speech Recognition](#paper-1512)
28. [Offline Reinforcement Learning with Generative Trajectory Policies](#paper-1513)
29. [HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](#paper-1514)
30. [TelecomTS: A Multi-Modal Observability Dataset for Time Series and Language Analysis](#paper-1515)
31. [Fast-SAM3D: 3Dfy Anything in Images but Faster](#paper-1516)
32. [Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting](#paper-1517)
33. [DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](#paper-1518)

---


<a id="paper-1486"></a>

## Compositional Planning with Jumpy World Models

### Images

![4a80ee1e9595b4b38eb7ffc2aae150de7fbf9d034c5e55cb21a0c5083c1b4371.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/4a80ee1e9595b4b38eb7ffc2aae150de7fbf9d034c5e55cb21a0c5083c1b4371.jpg)

![5e3e3ca45e19ad8de37def048daf27925c60d731529f3948e172e28e3b68fc4c.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/5e3e3ca45e19ad8de37def048daf27925c60d731529f3948e172e28e3b68fc4c.jpg)

![615550ed01c8ff41a9cd524a865ad12049e73cb044aad6e34494451001d6f830.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/615550ed01c8ff41a9cd524a865ad12049e73cb044aad6e34494451001d6f830.jpg)

![6ea5f5ff599c33abff3e2a945c366014b3a2e3e12fdd03676ee62067512cf2fa.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/6ea5f5ff599c33abff3e2a945c366014b3a2e3e12fdd03676ee62067512cf2fa.jpg)

![6efb7f959a92d76fb222e68a1024f0374bd599850092fa6276946e3bc96ff09b.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/6efb7f959a92d76fb222e68a1024f0374bd599850092fa6276946e3bc96ff09b.jpg)

![7b6abe670a5ad30c178a9469af7e712c55ec3a37ac0788e96e4141c3e50adea2.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/7b6abe670a5ad30c178a9469af7e712c55ec3a37ac0788e96e4141c3e50adea2.jpg)

![8089003aad0d6fc20468471e65190265d70b587a6b8047a53aa952de80cf3329.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/8089003aad0d6fc20468471e65190265d70b587a6b8047a53aa952de80cf3329.jpg)

![aa68606d7812121dbc2f14a6e2d99b5168a4f121392023fa49335e979df9a24b.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/images/aa68606d7812121dbc2f14a6e2d99b5168a4f121392023fa49335e979df9a24b.jpg)

### Tables

![3b895f9cbce488ead7dc2ba6514bd8927a9820b3734972777f82994e178b602c.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/tables/3b895f9cbce488ead7dc2ba6514bd8927a9820b3734972777f82994e178b602c.jpg)

![6c33c72abaf9571a40acd3fab8209e246a676a4d59054fc2f7d5afc04e126df5.jpg](../icml_results/1506_StreamFlow_%20Theory%2C%20Algorithm%2C%20and%20Implementation%20for%20High-Efficiency%20Rectified%20Flow%20Generation/tables/6c33c72abaf9571a40acd3fab8209e246a676a4d59054fc2f7d5afc04e126df5.jpg)

## Compositional Planning with Jumpy World Models


### Images

![014fc8bf45d849db1150e1dbef84cde9d76a2f7f0253393feed7b57679c71a48.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/014fc8bf45d849db1150e1dbef84cde9d76a2f7f0253393feed7b57679c71a48.jpg)

![0c4f7db119ecc1dca7963d01301cc989a56344abc96ce042e6f238edb8dee738.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/0c4f7db119ecc1dca7963d01301cc989a56344abc96ce042e6f238edb8dee738.jpg)

![4fb647a48aaf1601167e455406c355572d4e2f5b72f8e910620bd152ffc0d1f9.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/4fb647a48aaf1601167e455406c355572d4e2f5b72f8e910620bd152ffc0d1f9.jpg)

![5fb8e5b38a01861b0e91c1ba3c4273bfb5b8207aabff454056959a30dc2988c6.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/5fb8e5b38a01861b0e91c1ba3c4273bfb5b8207aabff454056959a30dc2988c6.jpg)

![777553faed0c03bcb79149a8038cd0ab993ce03b0bb85b7a304674a4336d6479.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/777553faed0c03bcb79149a8038cd0ab993ce03b0bb85b7a304674a4336d6479.jpg)

![b53c24858ff603727da7590f7ee40abb3a3ba0cfbbe6afe6d60c6aef0d84b601.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/b53c24858ff603727da7590f7ee40abb3a3ba0cfbbe6afe6d60c6aef0d84b601.jpg)

![c906a0821eed79d756d2672a356991a175c1938460ddbeac76128a02234ff06b.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/c906a0821eed79d756d2672a356991a175c1938460ddbeac76128a02234ff06b.jpg)

![d5a65db47dbb02d1cc8bcf0ecec7c3e8a6935b8df623c37c3b1f67a7e186bb32.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/d5a65db47dbb02d1cc8bcf0ecec7c3e8a6935b8df623c37c3b1f67a7e186bb32.jpg)

![e25e98650fba59ef51d789e484d6ca8c7c98054681a5f5d69a681a495cbf7359.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/e25e98650fba59ef51d789e484d6ca8c7c98054681a5f5d69a681a495cbf7359.jpg)

![e496f1d7ce0aee09546e0a469439cc144a015e0f13b2df665c3ad8bce60d4475.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/images/e496f1d7ce0aee09546e0a469439cc144a015e0f13b2df665c3ad8bce60d4475.jpg)

### Tables

![09fac6827c5dce15ba4366adcce5a934f53bdbf4b58e32a0c0bf3a5833232703.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/09fac6827c5dce15ba4366adcce5a934f53bdbf4b58e32a0c0bf3a5833232703.jpg)

![1462dfb510e33e8f612db02d44101baf8b76f8fe51654f26ae831b029653b8b0.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/1462dfb510e33e8f612db02d44101baf8b76f8fe51654f26ae831b029653b8b0.jpg)

![1a518f77d49abf9ad439e1ab17587c6b42adef2b7e9c7f567f6c6d71f7e73ba8.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/1a518f77d49abf9ad439e1ab17587c6b42adef2b7e9c7f567f6c6d71f7e73ba8.jpg)

![27196dfe867de1e766bf140c404a2c46c622ae7b3ce01de91318ae54ef99093e.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/27196dfe867de1e766bf140c404a2c46c622ae7b3ce01de91318ae54ef99093e.jpg)

![277215186bb710d52ade6596822e399c63a6a4d4a48d6986738a6fde86842dbf.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/277215186bb710d52ade6596822e399c63a6a4d4a48d6986738a6fde86842dbf.jpg)

![398484e4c3e81d864fa54fd669f6823dc47b8ff61b2cb85fcbe3a22450f040d9.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/398484e4c3e81d864fa54fd669f6823dc47b8ff61b2cb85fcbe3a22450f040d9.jpg)

![7ff1a2f151eb39e7dbf928e75019b5e44c91b2cdabbe77057d72842a85dde709.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/7ff1a2f151eb39e7dbf928e75019b5e44c91b2cdabbe77057d72842a85dde709.jpg)

![894c966da5e0a49f1055d098cb96384ed2fd575e3de5866c2c306cdb2188c3c7.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/894c966da5e0a49f1055d098cb96384ed2fd575e3de5866c2c306cdb2188c3c7.jpg)

![909f9e7ad24dd66c9d44703af77875fdf12ec191253876e3845620e25fa7bcee.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/909f9e7ad24dd66c9d44703af77875fdf12ec191253876e3845620e25fa7bcee.jpg)

![92286406662474e3052d54ad3666ece216ec4dbae1a16e474215aa61747c48fc.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/92286406662474e3052d54ad3666ece216ec4dbae1a16e474215aa61747c48fc.jpg)

![ab92095c907238dfa694ae65a10707eddfbe30142ebd487579918016d722a467.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/ab92095c907238dfa694ae65a10707eddfbe30142ebd487579918016d722a467.jpg)

![d1ea195abe393ab7e790d05e0d3de682bb281076c406bfc17d43f1e84c260437.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/d1ea195abe393ab7e790d05e0d3de682bb281076c406bfc17d43f1e84c260437.jpg)

![ec1930bae7641c242576dd9b4a6571f68bba428beb2bba9ca6359ee135a7b781.jpg](../icml_results/1507_Compositional%20Planning%20with%20Jumpy%20World%20Models/tables/ec1930bae7641c242576dd9b4a6571f68bba428beb2bba9ca6359ee135a7b781.jpg)

<a id="paper-1487"></a>

## Enhancing Conformal Prediction via Class Similarity


### Images

![11007181b8c3500663a69cba60f5ddc9a3dc9d9d19a8086ba3d23a4ff93f6460.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/11007181b8c3500663a69cba60f5ddc9a3dc9d9d19a8086ba3d23a4ff93f6460.jpg)

![49b71806b2fa9c203f11513b5527c1880f6ee9a59e62de7dfa4df09d5603e9d7.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/49b71806b2fa9c203f11513b5527c1880f6ee9a59e62de7dfa4df09d5603e9d7.jpg)

![4bb08b8123b31c0531f9e00d9418ddbb88ae167804ba560c5eb2d291127c7193.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/4bb08b8123b31c0531f9e00d9418ddbb88ae167804ba560c5eb2d291127c7193.jpg)

![7ea6e375afdeb26242915cf8afc0db9cf8ddef76acf4c2d122f68c32236ddfea.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/7ea6e375afdeb26242915cf8afc0db9cf8ddef76acf4c2d122f68c32236ddfea.jpg)

![851eff296383e46044e900ab838afb5c6f59f6594ec7bb49757522cea945d344.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/851eff296383e46044e900ab838afb5c6f59f6594ec7bb49757522cea945d344.jpg)

![c93a78a65cbc69698ed19c1c91bfa567004861bfa1ebfd2392c55e3b0a61f8fc.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/c93a78a65cbc69698ed19c1c91bfa567004861bfa1ebfd2392c55e3b0a61f8fc.jpg)

![e379960de1c4b7104d88c500d62b701e99adbb129bde5cc2cac2ff4a7f741f5b.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/images/e379960de1c4b7104d88c500d62b701e99adbb129bde5cc2cac2ff4a7f741f5b.jpg)

### Tables

![02cbf1f1d870b93f7071ff57fd634a566529d1ba28e3c62044b88173df5442d8.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/02cbf1f1d870b93f7071ff57fd634a566529d1ba28e3c62044b88173df5442d8.jpg)

![162c3ee06ea13d78d3dbbcb9de4d954c3510ba4c3861140088de6cc3d9dc9836.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/162c3ee06ea13d78d3dbbcb9de4d954c3510ba4c3861140088de6cc3d9dc9836.jpg)

![36d3d57fbf6a99110082d61b87d786b61437ae99de9c5895bbe9431598e56b5a.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/36d3d57fbf6a99110082d61b87d786b61437ae99de9c5895bbe9431598e56b5a.jpg)

![44874cf4d9e03c02b4a3436acb19bc9b1249eab4526892358068f2f16d9db999.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/44874cf4d9e03c02b4a3436acb19bc9b1249eab4526892358068f2f16d9db999.jpg)

![5d979398dd2e788fd3546df80a3a087b9171d0ea34c9b6fdd166f23b3bec80e7.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/5d979398dd2e788fd3546df80a3a087b9171d0ea34c9b6fdd166f23b3bec80e7.jpg)

![809dee86f77b9ade0ae81c5d9e35e3de88faa93c37d62ca848bee495fc3ff2b5.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/809dee86f77b9ade0ae81c5d9e35e3de88faa93c37d62ca848bee495fc3ff2b5.jpg)

![91324ef68ea64a634bc3a3f7773a9e3db73b82c4082c6f7103b08a1fe31843cc.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/91324ef68ea64a634bc3a3f7773a9e3db73b82c4082c6f7103b08a1fe31843cc.jpg)

![bb40b09d9e65e2bc7038c6da9bf8d47423412f3c594442cb532cb84c908c2169.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/bb40b09d9e65e2bc7038c6da9bf8d47423412f3c594442cb532cb84c908c2169.jpg)

![d51d1daebe6daf9ded91e5179fe4f0fa8e7439638020432dcf76437cef0fcd69.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/d51d1daebe6daf9ded91e5179fe4f0fa8e7439638020432dcf76437cef0fcd69.jpg)

![e8ac27ce1ab56b1a1c62485f821a1031ac37e15ecf0d1030a57fc2a82387b742.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/e8ac27ce1ab56b1a1c62485f821a1031ac37e15ecf0d1030a57fc2a82387b742.jpg)

![efab3ad84100a0f8c0d78d0a0fbc55666638df24c637ec83c139330cb56c4d30.jpg](../icml_results/1508_Enhancing%20Conformal%20Prediction%20via%20Class%20Similarity/tables/efab3ad84100a0f8c0d78d0a0fbc55666638df24c637ec83c139330cb56c4d30.jpg)

<a id="paper-1488"></a>

## Should I Have Expressed a Different Intent? Counterfactual Generation for LLM-Based Autonomous Control


### Images

![0267adca8e2925fca0d96154170850bfd66db1e29ea180203ef2380600c1feb8.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/0267adca8e2925fca0d96154170850bfd66db1e29ea180203ef2380600c1feb8.jpg)

![0de0bc3ccec01677b98e244f191fe218ba23bf01e21b925d03a090b053c60a6c.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/0de0bc3ccec01677b98e244f191fe218ba23bf01e21b925d03a090b053c60a6c.jpg)

![142e5158ada69e53365ecdd1b548bf1991fd73bc8ae0de578e4c334890df44cd.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/142e5158ada69e53365ecdd1b548bf1991fd73bc8ae0de578e4c334890df44cd.jpg)

![30c371cf7bbbd925d7118cb4d032dc7636c23f98808a1c6e79ea3c1df5d06ee1.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/30c371cf7bbbd925d7118cb4d032dc7636c23f98808a1c6e79ea3c1df5d06ee1.jpg)

![8fb3e07ae42b88f2cfc8e374fcc9bc24ba63468fc822c610b8c9ce1e6b623e22.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/8fb3e07ae42b88f2cfc8e374fcc9bc24ba63468fc822c610b8c9ce1e6b623e22.jpg)

![b4e074359744f95291458efc9db42c62711aa2e3a15297867116ce2ba576c1b7.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/b4e074359744f95291458efc9db42c62711aa2e3a15297867116ce2ba576c1b7.jpg)

![b6ba9c2fcebab449694960e7915e7692734e49ca58d6914eeb22c9df09d2adaf.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/b6ba9c2fcebab449694960e7915e7692734e49ca58d6914eeb22c9df09d2adaf.jpg)

![d268f2a9259fba4f8c4f85ebe1c4be30243d794014398f6e7c0cc05c22cc0ef7.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/d268f2a9259fba4f8c4f85ebe1c4be30243d794014398f6e7c0cc05c22cc0ef7.jpg)

![fe432d5d1702fbc0ddda21e1eaa60fc8f5b6047e378a5151c5bfd969daf2e382.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/images/fe432d5d1702fbc0ddda21e1eaa60fc8f5b6047e378a5151c5bfd969daf2e382.jpg)

### Tables

![f6d576737f3509f0de4db44b4ff3a91524afae6d72de27cb74c78ec0639269da.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/tables/f6d576737f3509f0de4db44b4ff3a91524afae6d72de27cb74c78ec0639269da.jpg)

![ff888ab45d42a03dabab18dfbd5f49ef2a99cc90141ca0a691562337f5e49e33.jpg](../icml_results/1509_Should%20I%20Have%20Expressed%20a%20Different%20Intent_%20Counterfactual%20Generation%20for%20LLM-Based%20Autonomous%20Contr/tables/ff888ab45d42a03dabab18dfbd5f49ef2a99cc90141ca0a691562337f5e49e33.jpg)

<a id="paper-1489"></a>

## Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach


### Images

![00f6be35ff803494df422e83ce639ee577758a4e920bd57344a70be4064e205a.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/00f6be35ff803494df422e83ce639ee577758a4e920bd57344a70be4064e205a.jpg)

![260557abcd449e652dae14e589af060e0d3f6dbe1e90fac97283a39e586236da.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/260557abcd449e652dae14e589af060e0d3f6dbe1e90fac97283a39e586236da.jpg)

![43330f64c58c8f6f827f108c32c29320a4a3dc6e02c19351deb879057f2c8c44.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/43330f64c58c8f6f827f108c32c29320a4a3dc6e02c19351deb879057f2c8c44.jpg)

![44278f2f96868c5336829a1193edabed54db36d12df7c6d6a98956e77c4cda88.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/44278f2f96868c5336829a1193edabed54db36d12df7c6d6a98956e77c4cda88.jpg)

![465abf4c0a6913e8a0db2d0eab62df200621ee5d4e13b5679ec799ebabdd0abc.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/465abf4c0a6913e8a0db2d0eab62df200621ee5d4e13b5679ec799ebabdd0abc.jpg)

![4952560c96accfb03c525ab938e99b29cd2a33eaee0f94a18a5e55b2da2f3971.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/4952560c96accfb03c525ab938e99b29cd2a33eaee0f94a18a5e55b2da2f3971.jpg)

![825b3f904814c98c70983d6753ad47e2ce65a2fb69a64c8a501f5c2d325771ee.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/825b3f904814c98c70983d6753ad47e2ce65a2fb69a64c8a501f5c2d325771ee.jpg)

![b09b30426b9bc2eb9fcf9b268c04c41a343e4558ef6fc8fae61bfe90a78e5799.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/b09b30426b9bc2eb9fcf9b268c04c41a343e4558ef6fc8fae61bfe90a78e5799.jpg)

![d617cbd5a6645fd8374a50df0c98f8541a5faf695d32a6ed3d0bc007aaf9cb52.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/d617cbd5a6645fd8374a50df0c98f8541a5faf695d32a6ed3d0bc007aaf9cb52.jpg)

![fa265c4340cf56e080f88756bafcac18702e57744ad2a1d9e574c4ba1aea76cc.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/fa265c4340cf56e080f88756bafcac18702e57744ad2a1d9e574c4ba1aea76cc.jpg)

![fbb1132f66f9080cdc84614f025a52504ffa05db543d5abada6b75feaafa76c1.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/fbb1132f66f9080cdc84614f025a52504ffa05db543d5abada6b75feaafa76c1.jpg)

![fe18522ac2278c8bee94034c36f597fb61dc4901ee767e5f77074d2322cd402b.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/images/fe18522ac2278c8bee94034c36f597fb61dc4901ee767e5f77074d2322cd402b.jpg)

### Tables

![55223896858fdb79a91507f4ddd11e27e3de907c8b348bd8eb94de8a5991bc30.jpg](../icml_results/1510_Solving%20the%20Offline%20and%20Online%20Min-Max%20Problem%20of%20Non-smooth%20Submodular-Concave%20Functions_%20A%20Zeroth-/tables/55223896858fdb79a91507f4ddd11e27e3de907c8b348bd8eb94de8a5991bc30.jpg)

<a id="paper-1490"></a>

## $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data


### Images

![1611b10f0bcd09610b3ec3f2f500e9905fa9f6bd17de2d2a29f363db6f57a933.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/1611b10f0bcd09610b3ec3f2f500e9905fa9f6bd17de2d2a29f363db6f57a933.jpg)

![334056cb1a5e4d2699381e59de50946421be7c70388cd6f920b92ef173b6e322.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/334056cb1a5e4d2699381e59de50946421be7c70388cd6f920b92ef173b6e322.jpg)

![57d187a75f66e1ba298aee17ec9b740107e062d117573ae59bd4bc1115c3ae8f.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/57d187a75f66e1ba298aee17ec9b740107e062d117573ae59bd4bc1115c3ae8f.jpg)

![645c8e0d214dcfebdce757d6a81259d6020fd3054fe4bbf697f8cde442f9222c.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/645c8e0d214dcfebdce757d6a81259d6020fd3054fe4bbf697f8cde442f9222c.jpg)

![73c75910968f62d932f4edafd96c5955e83753e9d76ae1dfb2204a2ece71029c.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/73c75910968f62d932f4edafd96c5955e83753e9d76ae1dfb2204a2ece71029c.jpg)

![765cf656945ab7744f72a55b68b67f2cf5cc1805f5cf2bcd8ccfb9ce87152892.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/765cf656945ab7744f72a55b68b67f2cf5cc1805f5cf2bcd8ccfb9ce87152892.jpg)

![789391e8f951685ee8bb1e29de3b5704e7891bde02da85dcfeb448fa3a43a783.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/789391e8f951685ee8bb1e29de3b5704e7891bde02da85dcfeb448fa3a43a783.jpg)

![797bbdf7cc061a49de2e46a67534e5decbf96f2968846b3f12777f7934693044.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/797bbdf7cc061a49de2e46a67534e5decbf96f2968846b3f12777f7934693044.jpg)

![837a6cf574ba383f13f8313e3a702a266464fc9260e61d367be0a47357e42fca.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/837a6cf574ba383f13f8313e3a702a266464fc9260e61d367be0a47357e42fca.jpg)

![88c7e343ba5342013e70c3be0dd9cf8f776078d2c811ce8bc43c1d1d1ca1d7fb.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/88c7e343ba5342013e70c3be0dd9cf8f776078d2c811ce8bc43c1d1d1ca1d7fb.jpg)

![906c15001a954a96838a4a94334fdac7ecf046184c17e561d0971abe253859ad.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/906c15001a954a96838a4a94334fdac7ecf046184c17e561d0971abe253859ad.jpg)

![9126fc4ba75cf9e9b9bd889d5566798a3964b2d0a633e7ec050474f3a3140eeb.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/9126fc4ba75cf9e9b9bd889d5566798a3964b2d0a633e7ec050474f3a3140eeb.jpg)

![92238fb680712731664dc39d5e677cf138fbe740035749441582d1d102cab44c.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/92238fb680712731664dc39d5e677cf138fbe740035749441582d1d102cab44c.jpg)

![962d846e5cc4bc2dc206e1c02644434a6646eb9ab4f6ebbe78fc17abeee60d1c.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/962d846e5cc4bc2dc206e1c02644434a6646eb9ab4f6ebbe78fc17abeee60d1c.jpg)

![998d366855aaa0c6ead0e1411b269979b2318b078e2cf55b62401fb8251729b2.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/998d366855aaa0c6ead0e1411b269979b2318b078e2cf55b62401fb8251729b2.jpg)

![9c4df172335b69141f02a700f1d44c726f16a3fc42af1cb9615a0e20be915f80.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/9c4df172335b69141f02a700f1d44c726f16a3fc42af1cb9615a0e20be915f80.jpg)

![a1f318784e7eec9ee2a4f62bb58c4f36c1c41e1975313759fe92350bf8993337.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/a1f318784e7eec9ee2a4f62bb58c4f36c1c41e1975313759fe92350bf8993337.jpg)

![a681a3dbf523bea2b08717b0c652c57675ac170c6db1031eb3eb4a9df19e0018.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/a681a3dbf523bea2b08717b0c652c57675ac170c6db1031eb3eb4a9df19e0018.jpg)

![ab82f2cd701a32e1bb07bd93afec219f80048efcb0ab91d1110511e1d6a53b54.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/ab82f2cd701a32e1bb07bd93afec219f80048efcb0ab91d1110511e1d6a53b54.jpg)

![cbf8f743155fd5274ec73adbca02a2b28c64ccf408aca66c30bcaa8208f2ae97.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/cbf8f743155fd5274ec73adbca02a2b28c64ccf408aca66c30bcaa8208f2ae97.jpg)

![d26a8d6c3ccdc8a7cf243883368c3b03879a209149e5a637531afe67c061cf48.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/d26a8d6c3ccdc8a7cf243883368c3b03879a209149e5a637531afe67c061cf48.jpg)

![d65292fe1511961961ef6f44d62f72f3f93d385d0b6282da46e950e52d6bb901.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/d65292fe1511961961ef6f44d62f72f3f93d385d0b6282da46e950e52d6bb901.jpg)

![e03331f8808bf7ef51ca623653e13cbab2771e245a483bfaed01feba6bc9e5d9.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/e03331f8808bf7ef51ca623653e13cbab2771e245a483bfaed01feba6bc9e5d9.jpg)

![ef4fa0251cf1e1d10cee1d120ba4572439fbca2ff9c5574986db31c7c5512b73.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/images/ef4fa0251cf1e1d10cee1d120ba4572439fbca2ff9c5574986db31c7c5512b73.jpg)

### Tables

![652014a049a161b89b3e00c4c5a3d198d5b85dd78b53c5f155db87d6e2d4d485.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/tables/652014a049a161b89b3e00c4c5a3d198d5b85dd78b53c5f155db87d6e2d4d485.jpg)

![827736024584f3312885f5e24176c59237d845644301a5710866f577002acddd.jpg](../icml_results/1511_%24f%24-Trajectory%20Balance_%20A%20Loss%20Family%20for%20Tuning%20GFlowNets%2C%20Generative%20Models%2C%20and%20LLMs%20with%20Off-%20an/tables/827736024584f3312885f5e24176c59237d845644301a5710866f577002acddd.jpg)

<a id="paper-1491"></a>

## COFT: Counterfactual–Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models


### Images

![0ca85e0f07be6ba8ece88935edcd6a392ad62626ed307fb6d6d2488e34230d32.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/0ca85e0f07be6ba8ece88935edcd6a392ad62626ed307fb6d6d2488e34230d32.jpg)

![106df6a1a2bd8c25f3969d37a8c752b5fb31a4d2da02182824945196763abf72.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/106df6a1a2bd8c25f3969d37a8c752b5fb31a4d2da02182824945196763abf72.jpg)

![2e4dda415d396bc9d198e2a9695623f8a1ad4366982297d21947733a71d824cf.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/2e4dda415d396bc9d198e2a9695623f8a1ad4366982297d21947733a71d824cf.jpg)

![6660eefef548d468a73cc92ef23d4fd6adcebcd895cf0a5c685c410c7f914d44.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/6660eefef548d468a73cc92ef23d4fd6adcebcd895cf0a5c685c410c7f914d44.jpg)

![6b22c4c632a57c66d724b75a87409e3979e22ee0b6f7b4cb70e93ca9c90bbe04.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/6b22c4c632a57c66d724b75a87409e3979e22ee0b6f7b4cb70e93ca9c90bbe04.jpg)

![6f67efaf4f3912d34822a4e48bfbe540a52bba0b01595627801909a67dc90efc.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/6f67efaf4f3912d34822a4e48bfbe540a52bba0b01595627801909a67dc90efc.jpg)

![84ac178b8b4adc98c7724aea0efac0f74a84e76102da3529b271f0db90feb5e5.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/84ac178b8b4adc98c7724aea0efac0f74a84e76102da3529b271f0db90feb5e5.jpg)

![b711658a9df9f3d93825d5ffd83e4a5896e94f2d331baa62c9c4afb27c77afb5.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/b711658a9df9f3d93825d5ffd83e4a5896e94f2d331baa62c9c4afb27c77afb5.jpg)

![c07175b408844cfefa16b492d79d2ff8f2a8d94aa55cebba95206364efbc5ba2.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/c07175b408844cfefa16b492d79d2ff8f2a8d94aa55cebba95206364efbc5ba2.jpg)

![e7a19a215b9ac8303d7de8e9a7b4e4488d997c5d7098c41bb4291cda84f3f98c.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/images/e7a19a215b9ac8303d7de8e9a7b4e4488d997c5d7098c41bb4291cda84f3f98c.jpg)

### Tables

![0a05314de037e671c62f2e4a7fd4b9b35f81c168704fcc1bf70b47481dc0a24d.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/0a05314de037e671c62f2e4a7fd4b9b35f81c168704fcc1bf70b47481dc0a24d.jpg)

![122fc11df5dffc3584600da68ab8c207f77ff916b3475dc309b71422f8f225e7.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/122fc11df5dffc3584600da68ab8c207f77ff916b3475dc309b71422f8f225e7.jpg)

![2557fd3d9cc4557e5869fd384e9dde80c96e11cdacbf8dcdceda00316f9cc8c1.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/2557fd3d9cc4557e5869fd384e9dde80c96e11cdacbf8dcdceda00316f9cc8c1.jpg)

![2fba43ed24eca58fa8236366cb217dc7e246a2fa9cfb66710a1a3a3aaf8c25b9.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/2fba43ed24eca58fa8236366cb217dc7e246a2fa9cfb66710a1a3a3aaf8c25b9.jpg)

![5d24287e087b6825c206fec434c07c0b9eedb8b68add9db1642726862c25767b.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/5d24287e087b6825c206fec434c07c0b9eedb8b68add9db1642726862c25767b.jpg)

![60461d65111aab6913b59f9ffe92f52b88bd355973920c9ae00e9141ae8a718f.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/60461d65111aab6913b59f9ffe92f52b88bd355973920c9ae00e9141ae8a718f.jpg)

![63887a9a22102c34c7420d3f810ba01353acc89616d5fb3aa1cd1bee571c4564.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/63887a9a22102c34c7420d3f810ba01353acc89616d5fb3aa1cd1bee571c4564.jpg)

![6fdf88367ed2c5ac33c4e987705459034e41609601a5cdf7e0a61646f95718f5.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/6fdf88367ed2c5ac33c4e987705459034e41609601a5cdf7e0a61646f95718f5.jpg)

![7282518b3917c454eab7effa3944c5d5961b9769e917e512a49fac294570940c.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/7282518b3917c454eab7effa3944c5d5961b9769e917e512a49fac294570940c.jpg)

![76b5bccc05d9264fb14af13c91ed3a1c52a1d9ffcfd2b4e7e4bf79cd56010d52.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/76b5bccc05d9264fb14af13c91ed3a1c52a1d9ffcfd2b4e7e4bf79cd56010d52.jpg)

![87666a2ce53eae2f0b97156006de3302be001be91838ffd731276b5eecd1e0d3.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/87666a2ce53eae2f0b97156006de3302be001be91838ffd731276b5eecd1e0d3.jpg)

![8ccc43beb9df994629ff712d165b3c96fbf15bf683d7dd16e9c82f145273d417.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/8ccc43beb9df994629ff712d165b3c96fbf15bf683d7dd16e9c82f145273d417.jpg)

![940f1f7a932cc6c96f82221752051073ef765f7cd1d272aef364d3c024acda61.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/940f1f7a932cc6c96f82221752051073ef765f7cd1d272aef364d3c024acda61.jpg)

![9f25939aba4801b701b93faeb6387e93f5a5199d745561888f851cf05678f3d9.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/9f25939aba4801b701b93faeb6387e93f5a5199d745561888f851cf05678f3d9.jpg)

![a5aec3ab7daf769a3c9c0808aeb4f81a29162b37ba0c7701a009f8843b967838.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/a5aec3ab7daf769a3c9c0808aeb4f81a29162b37ba0c7701a009f8843b967838.jpg)

![a73f91fc6facd3385b5215cb47a63bf27288b05c5820bf937191b036fe606f25.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/a73f91fc6facd3385b5215cb47a63bf27288b05c5820bf937191b036fe606f25.jpg)

![a96de56563ed8162629d7c9aa81976bac631fd30579bc5ada5d4edbdc3121862.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/a96de56563ed8162629d7c9aa81976bac631fd30579bc5ada5d4edbdc3121862.jpg)

![ad83d761a434a20f0e5a45aa78ebf1eeb33ecbaa9dc3dedcffc7f92f2a8089f9.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/ad83d761a434a20f0e5a45aa78ebf1eeb33ecbaa9dc3dedcffc7f92f2a8089f9.jpg)

![b074a77718aca62baab1c6b185d0efefe1592ca332ca660df1a8739ff9927fe1.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/b074a77718aca62baab1c6b185d0efefe1592ca332ca660df1a8739ff9927fe1.jpg)

![bdfbe23856b42c1785e8fa40fc6f3fbf79e785fcf9e10e6a256dc0049c069ea8.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/bdfbe23856b42c1785e8fa40fc6f3fbf79e785fcf9e10e6a256dc0049c069ea8.jpg)

![bf678fbb22a309c028eafafc981424c37050c2136c2a75a42547e4607394e873.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/bf678fbb22a309c028eafafc981424c37050c2136c2a75a42547e4607394e873.jpg)

![c0eb2d3dbdfaac9ebd094e8ad51c3a985062aa30787e5fddbfeb5b6d218c03b3.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/c0eb2d3dbdfaac9ebd094e8ad51c3a985062aa30787e5fddbfeb5b6d218c03b3.jpg)

![c35182a437c48a52fbaf1a548ee4320c839d6e8c63d3eb496c6a2c002283dce3.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/c35182a437c48a52fbaf1a548ee4320c839d6e8c63d3eb496c6a2c002283dce3.jpg)

![c6ad8f5a6424f31c3f1df3a94aa1bd7f92fc26b800361d95673da9eb31d25de9.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/c6ad8f5a6424f31c3f1df3a94aa1bd7f92fc26b800361d95673da9eb31d25de9.jpg)

![c6be16e36d0ba7a0cf8b7734b9a7349ed290f1d554bd93002ea229e5fc76843e.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/c6be16e36d0ba7a0cf8b7734b9a7349ed290f1d554bd93002ea229e5fc76843e.jpg)

![cc8ab62ed20cb5b621c64e89f8ebafa6ae8a0729607926694fe308f5b99dcf0c.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/cc8ab62ed20cb5b621c64e89f8ebafa6ae8a0729607926694fe308f5b99dcf0c.jpg)

![d951f8683cd991470fd3162581706ad173fc72688862c796dbf8504f5a050273.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/d951f8683cd991470fd3162581706ad173fc72688862c796dbf8504f5a050273.jpg)

![fab18e89fcfbf91e843705c019540844329d7f470e38980bba37c8bdf2eab095.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/fab18e89fcfbf91e843705c019540844329d7f470e38980bba37c8bdf2eab095.jpg)

![fc5c27ab18c56b275d99161f58adf4c4267efb099db39401a44ded7676c70d47.jpg](../icml_results/1512_COFT_%20Counterfactual%E2%80%93Conformal%20Decoding%20for%20Fair%20Chain-of-Thought%20Reasoning%20in%20Large%20Language%20Models/tables/fc5c27ab18c56b275d99161f58adf4c4267efb099db39401a44ded7676c70d47.jpg)

<a id="paper-1492"></a>

## Dissecting Quantization Error: A Concentration-Alignment Perspective


### Images

![2e4e4f8de43cf1d9cf9dc6db6302d263024339e0893f62c24f04acaecf2a488b.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/2e4e4f8de43cf1d9cf9dc6db6302d263024339e0893f62c24f04acaecf2a488b.jpg)

![3bf1e2413eb57f59cb91e404a1d4e07cce6e6b376560bd797008f07c116b1386.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/3bf1e2413eb57f59cb91e404a1d4e07cce6e6b376560bd797008f07c116b1386.jpg)

![4f0c0af2702a4bd8c72946ce0ff9a10545a0f5a3cf39d756adc47a0ee5344e4b.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/4f0c0af2702a4bd8c72946ce0ff9a10545a0f5a3cf39d756adc47a0ee5344e4b.jpg)

![62ddc54b7282e4cd64e1f0d8a4705b3c6f0655e5e17b75cafd6345beaeb48b97.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/62ddc54b7282e4cd64e1f0d8a4705b3c6f0655e5e17b75cafd6345beaeb48b97.jpg)

![692143a16def7dea519e1ba2d140c1479db79ec5bfad5917f5bb0b0464d7c8b2.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/692143a16def7dea519e1ba2d140c1479db79ec5bfad5917f5bb0b0464d7c8b2.jpg)

![8e4a8d4c6895b343839dcc58ed88db24e3c0fa398be06beb3d6a2392e1140d5c.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/8e4a8d4c6895b343839dcc58ed88db24e3c0fa398be06beb3d6a2392e1140d5c.jpg)

![97c413e5d779498e6073c5f0b8cc28a2a3ce95d867f97d519a7c84d7ea1c0933.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/97c413e5d779498e6073c5f0b8cc28a2a3ce95d867f97d519a7c84d7ea1c0933.jpg)

![a28dc11e7364ec37ef057bda552aa7768d1a7b86219f9cdcccc76cf6adfcb17d.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/a28dc11e7364ec37ef057bda552aa7768d1a7b86219f9cdcccc76cf6adfcb17d.jpg)

![c183aaa9d62a2e0a08cb08a5b45b4d7d822393c963f017e5a084a04b05b27438.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/c183aaa9d62a2e0a08cb08a5b45b4d7d822393c963f017e5a084a04b05b27438.jpg)

![c36425abf07ced8256f4ae47cbf276d34bf696938425cf9cdf554e02cb5eecce.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/c36425abf07ced8256f4ae47cbf276d34bf696938425cf9cdf554e02cb5eecce.jpg)

![ece088626759103d7be690a4d2e97d0ca30a7ca127e12ed982ddf9687963d4bf.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/images/ece088626759103d7be690a4d2e97d0ca30a7ca127e12ed982ddf9687963d4bf.jpg)

### Tables

![7c78ec5d5c8160fbb8136e3e03099af6af5f349737a5be459cafab26f84a3eda.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/tables/7c78ec5d5c8160fbb8136e3e03099af6af5f349737a5be459cafab26f84a3eda.jpg)

![c6e09a3a2a356664d8207c08a581ca50513c878b1b10fef04fd78052f79a2d64.jpg](../icml_results/1513_Dissecting%20Quantization%20Error_%20A%20Concentration-Alignment%20Perspective/tables/c6e09a3a2a356664d8207c08a581ca50513c878b1b10fef04fd78052f79a2d64.jpg)

<a id="paper-1493"></a>

## Learning Interpretable Options by Identifying Reward Diffusion Bottlenecks in Reinforcement Learning


### Images

![01fd8a9967b9d7d7239107bf3ad31afacf0275ff0e83c74047dc4b9d3f7d8af7.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/01fd8a9967b9d7d7239107bf3ad31afacf0275ff0e83c74047dc4b9d3f7d8af7.jpg)

![07e1a29899fdccc7c8d6aa3c791d4b00c50e127bf1e755f6f552a3f2a9947d34.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/07e1a29899fdccc7c8d6aa3c791d4b00c50e127bf1e755f6f552a3f2a9947d34.jpg)

![1176d9eb95da21b52a35b7b9b8092fd9ab5a55e0fb9f449a298a57799bb86c3f.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/1176d9eb95da21b52a35b7b9b8092fd9ab5a55e0fb9f449a298a57799bb86c3f.jpg)

![1dc912aebb78bea93d59cbce7ffa5cb595d111f9c26be3e570ce1e7f9819a99a.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/1dc912aebb78bea93d59cbce7ffa5cb595d111f9c26be3e570ce1e7f9819a99a.jpg)

![37635b592ed15d75fcaabbedc7cfabd0e496b83d6b90af9f5759a5e131ec72cc.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/37635b592ed15d75fcaabbedc7cfabd0e496b83d6b90af9f5759a5e131ec72cc.jpg)

![4bc3060b637f4e68b9c545abd4d0d8caa70b6a1cb9bb81dcd59e6551fd9cbe64.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/4bc3060b637f4e68b9c545abd4d0d8caa70b6a1cb9bb81dcd59e6551fd9cbe64.jpg)

![4f28e7759fc404f2a0ac0d73f2238c3e1175dd381b1cb172d692b22491558700.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/4f28e7759fc404f2a0ac0d73f2238c3e1175dd381b1cb172d692b22491558700.jpg)

![61ff7cdda0fa8c0a4917aaf420756fb3aa17270ac6b966d9cf891e999d22119b.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/61ff7cdda0fa8c0a4917aaf420756fb3aa17270ac6b966d9cf891e999d22119b.jpg)

![67cf28bd16be3bc2ed7990676083a9e36acecb040e96b96d51c586bea47d43dd.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/67cf28bd16be3bc2ed7990676083a9e36acecb040e96b96d51c586bea47d43dd.jpg)

![6ad2e15400bc2e0ac81508d7073ec4db70d74730750ef1d82fd0f8ec723319f1.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/6ad2e15400bc2e0ac81508d7073ec4db70d74730750ef1d82fd0f8ec723319f1.jpg)

![844c15ca0e18e1d499d3b9d53efbc8bee99053d81d82cb2eca3c16f2afdbd34b.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/844c15ca0e18e1d499d3b9d53efbc8bee99053d81d82cb2eca3c16f2afdbd34b.jpg)

![92073c17fc6f794073a2c78c83dd9049ca57f971f6d244756ef08a81fb146049.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/92073c17fc6f794073a2c78c83dd9049ca57f971f6d244756ef08a81fb146049.jpg)

![9244a1db31aeb527700fac96c6fe6ce07d266224d54c2c9b138d300502dca686.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/9244a1db31aeb527700fac96c6fe6ce07d266224d54c2c9b138d300502dca686.jpg)

![9be6ecc290ea90ecfc43c231bbb9407fb86d7b7a558bb66e77cc85f4dac2f492.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/9be6ecc290ea90ecfc43c231bbb9407fb86d7b7a558bb66e77cc85f4dac2f492.jpg)

![af4ba8490a908f62193a1d1622ea47500948351ee24b54697b1826fb17a27754.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/af4ba8490a908f62193a1d1622ea47500948351ee24b54697b1826fb17a27754.jpg)

![b23852b39bf314cd4fb04ca9e6246d39919f6e26513f8872d995b787818c06bc.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/b23852b39bf314cd4fb04ca9e6246d39919f6e26513f8872d995b787818c06bc.jpg)

![bb4ae6ec18970f6458509b3236d86ed0f6f0c6244e0dc3227dc528f599a688c1.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/bb4ae6ec18970f6458509b3236d86ed0f6f0c6244e0dc3227dc528f599a688c1.jpg)

![cdec20e1d0c3a2ccbbe2f4297cd0391a255a03042d0cc6f44634781db75ee178.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/cdec20e1d0c3a2ccbbe2f4297cd0391a255a03042d0cc6f44634781db75ee178.jpg)

![e70d01e8780ff866d6b93bfc25a48be7c4341c117ce847b08e6457aa69133d6d.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/e70d01e8780ff866d6b93bfc25a48be7c4341c117ce847b08e6457aa69133d6d.jpg)

![f4d1c6f3c7789139a35b09f63b4a95f77d1163203b8c0a321d5c0f541a95e861.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/images/f4d1c6f3c7789139a35b09f63b4a95f77d1163203b8c0a321d5c0f541a95e861.jpg)

### Tables

![085808bc9115fa470f85b4e49e1a9ad21c9a1f8808763df334ba727b12b9eec3.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/085808bc9115fa470f85b4e49e1a9ad21c9a1f8808763df334ba727b12b9eec3.jpg)

![2c7b00d0b8ed55dc077715cbe541680dff18122adababc7360a922f9c201c681.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/2c7b00d0b8ed55dc077715cbe541680dff18122adababc7360a922f9c201c681.jpg)

![898b00be270c092f79536f291cb0b0ec51df7083e6d5985593628815eeabb64c.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/898b00be270c092f79536f291cb0b0ec51df7083e6d5985593628815eeabb64c.jpg)

![8c735b4b4a9adaf660041cc59fc716b9e8159b26cc4987968c4f5071dd9000fc.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/8c735b4b4a9adaf660041cc59fc716b9e8159b26cc4987968c4f5071dd9000fc.jpg)

![8c9f81f769e5bb554a738c0018f8c8d5531814f3f04abb35b3f6c9b44c42584d.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/8c9f81f769e5bb554a738c0018f8c8d5531814f3f04abb35b3f6c9b44c42584d.jpg)

![8f7503366605faf40b3a84fa6b941106b618dabc69bcefdbf99c1ca93697d0b7.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/8f7503366605faf40b3a84fa6b941106b618dabc69bcefdbf99c1ca93697d0b7.jpg)

![9301096b10e50919b726a0e1424f2116731080bac110649a6d1b1d2f4103ae60.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/9301096b10e50919b726a0e1424f2116731080bac110649a6d1b1d2f4103ae60.jpg)

![9bacf7daba079bef5ea4e77cc377eb9c7806fcb91ea175a43bc442694346485a.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/9bacf7daba079bef5ea4e77cc377eb9c7806fcb91ea175a43bc442694346485a.jpg)

![b5b167e2a92e85c4131d873d365f92fdf5280a4250d7a7eb6c47588d4197b28c.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/b5b167e2a92e85c4131d873d365f92fdf5280a4250d7a7eb6c47588d4197b28c.jpg)

![bf4d187edb3356e3a59b158380d691bf3027b5f0d7ee1822b9d636797e84e8a3.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/bf4d187edb3356e3a59b158380d691bf3027b5f0d7ee1822b9d636797e84e8a3.jpg)

![e2269c91bd2f539b36edbf16d7bf6205441cc5ec1471e040de15714b94f634f4.jpg](../icml_results/1514_Learning%20Interpretable%20Options%20by%20Identifying%20Reward%20Diffusion%20Bottlenecks%20in%20Reinforcement%20Learning/tables/e2269c91bd2f539b36edbf16d7bf6205441cc5ec1471e040de15714b94f634f4.jpg)

<a id="paper-1494"></a>

## Olivia: Harmonizing Time Series Foundation Models with Power Spectral Density


### Images

![0421bd8c6517864a9343898e4a4ec8c83fefd53e8740ff48cb9d024f447fa319.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/0421bd8c6517864a9343898e4a4ec8c83fefd53e8740ff48cb9d024f447fa319.jpg)

![12169250b96c2bb2e6f609447774f269cd15b1c5dcc60e452d22c349a91a2cd0.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/12169250b96c2bb2e6f609447774f269cd15b1c5dcc60e452d22c349a91a2cd0.jpg)

![203814866fe9febace8a1a70a40953334376a8dacf1d917c7910e9de03683dd3.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/203814866fe9febace8a1a70a40953334376a8dacf1d917c7910e9de03683dd3.jpg)

![28a6af56fc708b8a9d32151c262642ebebe28a6f9e393e5240074659d1c50702.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/28a6af56fc708b8a9d32151c262642ebebe28a6f9e393e5240074659d1c50702.jpg)

![28d9d7bfae1861e6953cc145dc42ff2d0279472de836504f6da93d6a15b97ba2.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/28d9d7bfae1861e6953cc145dc42ff2d0279472de836504f6da93d6a15b97ba2.jpg)

![732b1b5e41a2a8fd732a621c02b69a7dc45189757b35f0df2cc7b49532d18355.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/732b1b5e41a2a8fd732a621c02b69a7dc45189757b35f0df2cc7b49532d18355.jpg)

![8a90e5b705b79415d9bf2a79ddcc99c21608cc1e1752393ed4b820def98ac126.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/8a90e5b705b79415d9bf2a79ddcc99c21608cc1e1752393ed4b820def98ac126.jpg)

![9e3d61896385bdf3b31326e88666b10ae9e9a365ab25d945d86b4887b40efae8.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/9e3d61896385bdf3b31326e88666b10ae9e9a365ab25d945d86b4887b40efae8.jpg)

![a292a4b8470e6263fb62434d7f671da67b49a2f8bec0c388df1cf3cec6182f30.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/a292a4b8470e6263fb62434d7f671da67b49a2f8bec0c388df1cf3cec6182f30.jpg)

![a4496f1c83231b00b8ae8c1cab92799c8ee8005b80f3979f4bcaf373b4c44516.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/a4496f1c83231b00b8ae8c1cab92799c8ee8005b80f3979f4bcaf373b4c44516.jpg)

![cfe59a37745a3f302301713703c1e25610036c429b4003b8f95d558b26f3f28c.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/cfe59a37745a3f302301713703c1e25610036c429b4003b8f95d558b26f3f28c.jpg)

![d342529408737070f796ac65576bec5176319b6c6e9e22bb09007971c7dbba6e.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/d342529408737070f796ac65576bec5176319b6c6e9e22bb09007971c7dbba6e.jpg)

![d749f7e9775b1d608f81b078a55ef52d9b072c4369115a0a399445936e341c38.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/d749f7e9775b1d608f81b078a55ef52d9b072c4369115a0a399445936e341c38.jpg)

![ddcc0e68000d0787bfdba4a6e914a0e9184f68a8233daed11f4b7697259d7a1d.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/ddcc0e68000d0787bfdba4a6e914a0e9184f68a8233daed11f4b7697259d7a1d.jpg)

![e700951ebd7d64cae387ea63501f4150fa023c667e55f0e40d6916d205d11d71.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/images/e700951ebd7d64cae387ea63501f4150fa023c667e55f0e40d6916d205d11d71.jpg)

### Tables

![0107fc88550b370c736a4bb300affc74422ce3961205056d7f38ba21f0726b7b.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/0107fc88550b370c736a4bb300affc74422ce3961205056d7f38ba21f0726b7b.jpg)

![03973a6ac1899f4571387d221042cb556eda8763df59d6096e344c3168313eed.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/03973a6ac1899f4571387d221042cb556eda8763df59d6096e344c3168313eed.jpg)

![1140091e941ed93b19c70d001f22d8e9b517ffcfd94ae8207a1fa815f21adf87.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/1140091e941ed93b19c70d001f22d8e9b517ffcfd94ae8207a1fa815f21adf87.jpg)

![1f69071afe4d7919f768a829b0ff4706ee165ecd71ab6f079c49cb2717da0a62.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/1f69071afe4d7919f768a829b0ff4706ee165ecd71ab6f079c49cb2717da0a62.jpg)

![3a5ed4c5552170534241f1e03d5de63315a374897ce9daf2d3f409fd54f63385.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/3a5ed4c5552170534241f1e03d5de63315a374897ce9daf2d3f409fd54f63385.jpg)

![3d8060a13f6c073634dee607953c96ab822fc21a4f6e38cfa9da118e8e0fd5c1.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/3d8060a13f6c073634dee607953c96ab822fc21a4f6e38cfa9da118e8e0fd5c1.jpg)

![3f1afea251db60d5663ee48ebab4f8889b9d7d0168694ecbc688fb2b20f69b70.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/3f1afea251db60d5663ee48ebab4f8889b9d7d0168694ecbc688fb2b20f69b70.jpg)

![40654f26eb564e7ece56f6eec71a9a665bd0a917cf5c92983675ed23b987bf00.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/40654f26eb564e7ece56f6eec71a9a665bd0a917cf5c92983675ed23b987bf00.jpg)

![450b8d780aba0de8b1d12a322c7c82c56b2fe9722edae4df8b6f5b6fa37e88e4.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/450b8d780aba0de8b1d12a322c7c82c56b2fe9722edae4df8b6f5b6fa37e88e4.jpg)

![4ca1c7dbef909324da52b76a724cfcead127e2d19abc8af13c272e15b909291f.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/4ca1c7dbef909324da52b76a724cfcead127e2d19abc8af13c272e15b909291f.jpg)

![545e8a678500d1274b76645384f1bce2f3d8b7639c9f5da56e826c6969bcf216.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/545e8a678500d1274b76645384f1bce2f3d8b7639c9f5da56e826c6969bcf216.jpg)

![6f071454ca97801cbccb57e92ed52f96d29997db49203f6a1b662bb673f283e7.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/6f071454ca97801cbccb57e92ed52f96d29997db49203f6a1b662bb673f283e7.jpg)

![8466f582d9594724922db4cb4c00f716bac3f22d85ddd918794e403aa73fe108.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/8466f582d9594724922db4cb4c00f716bac3f22d85ddd918794e403aa73fe108.jpg)

![90b476a7c23084b9d38950501ccd57cc8ce666314a31f92fa7bbf728d7ca1f02.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/90b476a7c23084b9d38950501ccd57cc8ce666314a31f92fa7bbf728d7ca1f02.jpg)

![a0100e88daa7d8b2977e80df61b1335119114b6abce8d9b6190f729e77d44178.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/a0100e88daa7d8b2977e80df61b1335119114b6abce8d9b6190f729e77d44178.jpg)

![ab51803a92de0d366da54bc34a7ede50582f47bd7259a40eeff9962e9c71b552.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/ab51803a92de0d366da54bc34a7ede50582f47bd7259a40eeff9962e9c71b552.jpg)

![bacd9c0be15978d3c0e5884a6ed2b43d9e105e11b0ae7afda06b64b58608af1b.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/bacd9c0be15978d3c0e5884a6ed2b43d9e105e11b0ae7afda06b64b58608af1b.jpg)

![bc319533d48dde206b569fd964960c58fa18e92d18bb57af7838e0090a984b55.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/bc319533d48dde206b569fd964960c58fa18e92d18bb57af7838e0090a984b55.jpg)

![bc4a3854e1c067e142cee5c1d795133c371b6a49377fdb68c3744d76c1dbf6f4.jpg](../icml_results/1515_Olivia_%20Harmonizing%20Time%20Series%20Foundation%20Models%20with%20Power%20Spectral%20Density/tables/bc4a3854e1c067e142cee5c1d795133c371b6a49377fdb68c3744d76c1dbf6f4.jpg)

<a id="paper-1495"></a>

## One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models


### Images

![1abaa181ac302298b1e5b8c02ada0c87f8b43c14f2710931fe4018f877a0d094.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/1abaa181ac302298b1e5b8c02ada0c87f8b43c14f2710931fe4018f877a0d094.jpg)

![823efee0fdd86906452342d0724e62ecdf9cfe32d648d04cd2c12b52dc2d6e80.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/823efee0fdd86906452342d0724e62ecdf9cfe32d648d04cd2c12b52dc2d6e80.jpg)

![8402567cd12c32caa7f1b43241999db7dbcb2d17e80e61d980dbc4256dd621df.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/8402567cd12c32caa7f1b43241999db7dbcb2d17e80e61d980dbc4256dd621df.jpg)

![a8b4d55359d29c080963cb89e1c093953d7a3f6ffe38316f666663ba840b3cfc.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/a8b4d55359d29c080963cb89e1c093953d7a3f6ffe38316f666663ba840b3cfc.jpg)

![b5e0c3da609426575c73303ffb9378aca8d7e7b77cb771368d547356674d05f1.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/b5e0c3da609426575c73303ffb9378aca8d7e7b77cb771368d547356674d05f1.jpg)

![bf81e9371e8c7e877811b6f489b49aebbf8c07bd03a6078b2fb7dac72fd76aaa.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/bf81e9371e8c7e877811b6f489b49aebbf8c07bd03a6078b2fb7dac72fd76aaa.jpg)

![d4a01127cacf6ad12968c98352e1d13058fc68455a0a9844a9f3087a61f07ac7.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/d4a01127cacf6ad12968c98352e1d13058fc68455a0a9844a9f3087a61f07ac7.jpg)

![ddb5003afac8b4c72aeabb9e2077c8c57586b429a77f35cbba09dbca5a69fddd.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/ddb5003afac8b4c72aeabb9e2077c8c57586b429a77f35cbba09dbca5a69fddd.jpg)

![e5b417da7d9d4645736ffc9cddd0c33d903271da64ee60263bff9313aca3cd98.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/e5b417da7d9d4645736ffc9cddd0c33d903271da64ee60263bff9313aca3cd98.jpg)

![f0b907abdfebf3d4c239e1c51730a21c758882801d7dd666a7b6a1fb414c3c12.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/images/f0b907abdfebf3d4c239e1c51730a21c758882801d7dd666a7b6a1fb414c3c12.jpg)

### Tables

![0298d66357afa2c7825c17cb84cb35b19cb3b09f9716edb56e89866507a4eb55.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/0298d66357afa2c7825c17cb84cb35b19cb3b09f9716edb56e89866507a4eb55.jpg)

![040fe5a9d59a345f188a67f3290f99752f4ca66fbf45cd58fe93a50a0bf6e36a.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/040fe5a9d59a345f188a67f3290f99752f4ca66fbf45cd58fe93a50a0bf6e36a.jpg)

![056433131f12b8dc5d4456b8dcfd6ebab09eac27e096f5776534f3db14970519.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/056433131f12b8dc5d4456b8dcfd6ebab09eac27e096f5776534f3db14970519.jpg)

![06ab4c2375e4ac838587ccd4ba8f4de1880c2369505c84a24d169fbb4277e73c.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/06ab4c2375e4ac838587ccd4ba8f4de1880c2369505c84a24d169fbb4277e73c.jpg)

![25c786bf1ec1346b16ba3e99af3576d220a7af6d0319e8ce03df7a8e199735f8.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/25c786bf1ec1346b16ba3e99af3576d220a7af6d0319e8ce03df7a8e199735f8.jpg)

![2c253b6bb2f939b5fc242c7211bfc4a2eda3f6dcd56a8aeee0e9d66dcaa5fc77.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/2c253b6bb2f939b5fc242c7211bfc4a2eda3f6dcd56a8aeee0e9d66dcaa5fc77.jpg)

![32d81c229ab41a6897df47c833c98eeb28075b30798700288a695d49d7c416a3.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/32d81c229ab41a6897df47c833c98eeb28075b30798700288a695d49d7c416a3.jpg)

![43cfde6314497629ee3753e91541cfa03a2fdd8bb699a6e8443f7b03105fcb30.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/43cfde6314497629ee3753e91541cfa03a2fdd8bb699a6e8443f7b03105fcb30.jpg)

![5873418dfedd97311f93ebdf75cd9ddd46814a1b2ead5e53ce7d5020b6db85c9.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/5873418dfedd97311f93ebdf75cd9ddd46814a1b2ead5e53ce7d5020b6db85c9.jpg)

![6983fef3e6c39fb9b04c0f856ec2846e9c8609a12ad9438d1360f53e267672f8.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/6983fef3e6c39fb9b04c0f856ec2846e9c8609a12ad9438d1360f53e267672f8.jpg)

![6d0c317ce31fc27976c62cebb716bf83e2e9e887479ed95d056e1a2f0f4bbf54.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/6d0c317ce31fc27976c62cebb716bf83e2e9e887479ed95d056e1a2f0f4bbf54.jpg)

![6fd39ca4da56ddf5d1fc0de36709fb58a732e59d8e57c175a396a11a5eafa3d9.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/6fd39ca4da56ddf5d1fc0de36709fb58a732e59d8e57c175a396a11a5eafa3d9.jpg)

![7e113fb49078928c493215f0288537d3497408d7cc22e4f306d1d93c596fca26.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/7e113fb49078928c493215f0288537d3497408d7cc22e4f306d1d93c596fca26.jpg)

![81d50043cc120c05f6777d34d2890f220ad7073d196ecf23d874401730edd21f.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/81d50043cc120c05f6777d34d2890f220ad7073d196ecf23d874401730edd21f.jpg)

![8a223ca1079040d0cde9631e52d94fc8dbd8d59e52ae4a0f38fb7be12300f7a3.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/8a223ca1079040d0cde9631e52d94fc8dbd8d59e52ae4a0f38fb7be12300f7a3.jpg)

![9c63b42a0ee5cba575a2f39c835f78af863865994af07c769e294f13f266eb75.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/9c63b42a0ee5cba575a2f39c835f78af863865994af07c769e294f13f266eb75.jpg)

![ca85f35714bc9f53ad2e88b40fea4049f3f8309cb679e37bdec81e8bb47c0ff2.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/ca85f35714bc9f53ad2e88b40fea4049f3f8309cb679e37bdec81e8bb47c0ff2.jpg)

![d414e379984f260bce685f086bf5565d86f6bc54e4b5a48de7ec1a640cb742f9.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/d414e379984f260bce685f086bf5565d86f6bc54e4b5a48de7ec1a640cb742f9.jpg)

![da0e3b4cf9d255fc3a3091003f0878d1278e583ceee075809dac24086554ebf2.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/da0e3b4cf9d255fc3a3091003f0878d1278e583ceee075809dac24086554ebf2.jpg)

![e32851ec3e2aef67e849eda93d2eb2813c79cdcb06b1db4205da05cc7e73be8a.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/e32851ec3e2aef67e849eda93d2eb2813c79cdcb06b1db4205da05cc7e73be8a.jpg)

![f256f01fd792f5e1f3dfaf7fff004a8b72d4a04d9e7480db5391563b44e10486.jpg](../icml_results/1516_One%20Bias%20After%20Another_%20Mechanistic%20Reward%20Shaping%20and%20Persistent%20Biases%20in%20Language%20Reward%20Models/tables/f256f01fd792f5e1f3dfaf7fff004a8b72d4a04d9e7480db5391563b44e10486.jpg)

<a id="paper-1496"></a>

## Efficient privacy loss accounting for subsampling and random allocation


### Images

![03723ca416433dbb8c3b2ed0fb15af4cc2a117ef22fbf931cf5b55fa00550b81.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/03723ca416433dbb8c3b2ed0fb15af4cc2a117ef22fbf931cf5b55fa00550b81.jpg)

![22809315e5bbb757d468bd87dea02260827e5fd05a3c08f64d7bc0914dd53657.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/22809315e5bbb757d468bd87dea02260827e5fd05a3c08f64d7bc0914dd53657.jpg)

![37236972acc730a2369e3662b1e53e93a23a058c4e5c5b5377b093422a4b2eca.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/37236972acc730a2369e3662b1e53e93a23a058c4e5c5b5377b093422a4b2eca.jpg)

![37399048e8c0c5b93d1d550afe85038ff45d8a8f6331031b6e93039bbec228fd.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/37399048e8c0c5b93d1d550afe85038ff45d8a8f6331031b6e93039bbec228fd.jpg)

![3da2dd3dc05abf563fa883281f07ae2b0c94c1c9448531d2ffea996bf28cf6dd.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/3da2dd3dc05abf563fa883281f07ae2b0c94c1c9448531d2ffea996bf28cf6dd.jpg)

![4b79db67395cbe0870ee7a69315c12e7bee1e039ccc1ce14ae9a590c4b83de8a.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/4b79db67395cbe0870ee7a69315c12e7bee1e039ccc1ce14ae9a590c4b83de8a.jpg)

![5d072ed268f3eb2ee2a7bdef445a7f4a99a15a15bd902b3c980111f20d0aab50.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/5d072ed268f3eb2ee2a7bdef445a7f4a99a15a15bd902b3c980111f20d0aab50.jpg)

![5f62e1a2441a3f5318d7f8cdcf57cb39fd8f9888b980fba2af8d46d3009ab6c9.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/5f62e1a2441a3f5318d7f8cdcf57cb39fd8f9888b980fba2af8d46d3009ab6c9.jpg)

![72a190f360ee3db06f7e84b129a6c41e3732a71865ffc0e7412da1d090ae8096.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/72a190f360ee3db06f7e84b129a6c41e3732a71865ffc0e7412da1d090ae8096.jpg)

![89a06c5fef7d809a1239ea52132c8504fcd0de6b1c3130df8b3aa8ddbb063604.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/89a06c5fef7d809a1239ea52132c8504fcd0de6b1c3130df8b3aa8ddbb063604.jpg)

![b8499b57b1211c7f7ae674f76e18d6db0ca1325835b2567cc9d94fb497d126f0.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/b8499b57b1211c7f7ae674f76e18d6db0ca1325835b2567cc9d94fb497d126f0.jpg)

![bc8af074bbc631e7378f49d8f1d1f67c456b824204b82e6ad6b6a1b5a1ef74c2.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/bc8af074bbc631e7378f49d8f1d1f67c456b824204b82e6ad6b6a1b5a1ef74c2.jpg)

![c1bf8aa589133d4ec981f0672b9e7267fc24aa609261fbebda95f6381eab2eec.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/c1bf8aa589133d4ec981f0672b9e7267fc24aa609261fbebda95f6381eab2eec.jpg)

![cd9af2594de798054f37d95abfe9b91caf19a7ffcb49e1c8383169bb89e2d2ce.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/cd9af2594de798054f37d95abfe9b91caf19a7ffcb49e1c8383169bb89e2d2ce.jpg)

![dae1d68ec0a18bf731db999d58436e636df2d71f11535ddd434e0d09ac862225.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/dae1d68ec0a18bf731db999d58436e636df2d71f11535ddd434e0d09ac862225.jpg)

![eb75d4b537337380c73ce5a9e132406f1f794506b925175f7ebea4e8068554f6.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/images/eb75d4b537337380c73ce5a9e132406f1f794506b925175f7ebea4e8068554f6.jpg)

### Tables

![475713fb88f3cda3f6d1ebf046b7fed58ebae2b62b92e837e682f05d752dc7eb.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/475713fb88f3cda3f6d1ebf046b7fed58ebae2b62b92e837e682f05d752dc7eb.jpg)

![9b21869d7adbc3f7f895829f3b6def59538ea39b8e83e35afdf35e7221f2d61d.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/9b21869d7adbc3f7f895829f3b6def59538ea39b8e83e35afdf35e7221f2d61d.jpg)

![9e5e67c21de44282f86527488e4fc3c042ec67870e63b97cc228a67e49dea80b.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/9e5e67c21de44282f86527488e4fc3c042ec67870e63b97cc228a67e49dea80b.jpg)

![c311d0055f2548c7deed8d355b7171c2d2ba43328e40940901d2c14ba2b470b3.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/c311d0055f2548c7deed8d355b7171c2d2ba43328e40940901d2c14ba2b470b3.jpg)

![d7d63fab1bcdba8c593dff9e3923568c9100c322c0faeb1324ecf905d3bd5790.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/d7d63fab1bcdba8c593dff9e3923568c9100c322c0faeb1324ecf905d3bd5790.jpg)

![de9eacd56d00c257f012956c54e7a5858d11b5c9d161fbc42555f078a065cda6.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/de9eacd56d00c257f012956c54e7a5858d11b5c9d161fbc42555f078a065cda6.jpg)

![eb8e3b91be0e92559ce03031da7b050581701235213a3cadeb1691861d118962.jpg](../icml_results/1517_Efficient%20privacy%20loss%20accounting%20for%20subsampling%20and%20random%20allocation/tables/eb8e3b91be0e92559ce03031da7b050581701235213a3cadeb1691861d118962.jpg)

<a id="paper-1497"></a>

## Distributionally Robust Causal Abstractions


### Images

![1012e289c95b934586e1320c72b4560fb95e282f560c725db7aa466779b1ee52.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/1012e289c95b934586e1320c72b4560fb95e282f560c725db7aa466779b1ee52.jpg)

![18ad895b6e5645c48e7dd8c26b3ae708370051155d4d4dbdea4318bbebbf32d3.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/18ad895b6e5645c48e7dd8c26b3ae708370051155d4d4dbdea4318bbebbf32d3.jpg)

![1dc03f94a10e8ac7dfb641d028b00093e98d07273f03adda1b1bf1c4631a30e9.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/1dc03f94a10e8ac7dfb641d028b00093e98d07273f03adda1b1bf1c4631a30e9.jpg)

![1dd17cca027237020edf4acd42bc00a81ec2ad9b758075bdef01614526ac055f.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/1dd17cca027237020edf4acd42bc00a81ec2ad9b758075bdef01614526ac055f.jpg)

![2f6928dd77b7574bb888428f5cbf8cf34ed5188eee502ed1630c790aef77d983.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/2f6928dd77b7574bb888428f5cbf8cf34ed5188eee502ed1630c790aef77d983.jpg)

![31d33427476be7b162ace0631655ebead856f75256edf95f902aebe2960e07d3.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/31d33427476be7b162ace0631655ebead856f75256edf95f902aebe2960e07d3.jpg)

![3355d8229a2cdce3680e95add049d47d53a2b1e860fc7a12338a2a7c172345f0.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/3355d8229a2cdce3680e95add049d47d53a2b1e860fc7a12338a2a7c172345f0.jpg)

![3453b89f55d7e74e00f907a8301d4c67c18fcccb49b62d55ca696bcf2d65dda0.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/3453b89f55d7e74e00f907a8301d4c67c18fcccb49b62d55ca696bcf2d65dda0.jpg)

![34f89fdfc29ed411cb31ec7896c56d6515d93e2700e93a2feb92cb2704e65023.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/34f89fdfc29ed411cb31ec7896c56d6515d93e2700e93a2feb92cb2704e65023.jpg)

![3bfae4e7c5c9cfe38d8f8aa965326ab745ff84f6110a92508e31d7c89caac958.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/3bfae4e7c5c9cfe38d8f8aa965326ab745ff84f6110a92508e31d7c89caac958.jpg)

![450235b1d088dc4e5c4daebf03edc596991de0077ab119f761d504a27c7f11a5.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/450235b1d088dc4e5c4daebf03edc596991de0077ab119f761d504a27c7f11a5.jpg)

![48fd6c974d1cb1186f5cf08ff453becc0b81fb8d3cb384db79c97b10329b5442.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/48fd6c974d1cb1186f5cf08ff453becc0b81fb8d3cb384db79c97b10329b5442.jpg)

![68bf076ecc61f8146adb8ecdb7002fff0dc0cd11f092db9a30263ebd81eaf87f.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/68bf076ecc61f8146adb8ecdb7002fff0dc0cd11f092db9a30263ebd81eaf87f.jpg)

![6c91e9b89636f9accbc1b059eafaf2222d1121af37dfec2bdc07afe8f73f2ba7.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/6c91e9b89636f9accbc1b059eafaf2222d1121af37dfec2bdc07afe8f73f2ba7.jpg)

![6fe78d1c338eef3972aa918f793273a1df3e79a2096cfa61b6998ed28e3d41b3.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/6fe78d1c338eef3972aa918f793273a1df3e79a2096cfa61b6998ed28e3d41b3.jpg)

![733680c9c4375502cd5bb56c3e33d2963525661f87b3fe2249aee2770e9fe3eb.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/733680c9c4375502cd5bb56c3e33d2963525661f87b3fe2249aee2770e9fe3eb.jpg)

![73ef0d2b4e74355d98c31641f52d1f01ddabfb3d046d891824d5669be5ace6fb.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/73ef0d2b4e74355d98c31641f52d1f01ddabfb3d046d891824d5669be5ace6fb.jpg)

![b4c6cfeba1c8d93e6e19c2dcd5794199576ae7231b0f1584081541d52cb8b1f7.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/b4c6cfeba1c8d93e6e19c2dcd5794199576ae7231b0f1584081541d52cb8b1f7.jpg)

![ba39e5ed77e1207c538f1ff30641e8fcaf66782bc1c8b25a2b1e5c7e0507d09b.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/ba39e5ed77e1207c538f1ff30641e8fcaf66782bc1c8b25a2b1e5c7e0507d09b.jpg)

![ba4db4f6e263051f7f925abcad59bbc51e617475d81096dfc6272e0b63da5832.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/ba4db4f6e263051f7f925abcad59bbc51e617475d81096dfc6272e0b63da5832.jpg)

![bd8c5d9fc50d3047002defc3454303fd71801197c969cb7a219d126842b4296e.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/bd8c5d9fc50d3047002defc3454303fd71801197c969cb7a219d126842b4296e.jpg)

![c3b8dfa8526f4316786c5d4c212e482b4dc005b0a3dd9598e89e61fb1fb99145.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/c3b8dfa8526f4316786c5d4c212e482b4dc005b0a3dd9598e89e61fb1fb99145.jpg)

![cc490fd174e0ad23aee6ec1d2458d52459aaa00f43ea05c1afffccfc75d0ba09.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/cc490fd174e0ad23aee6ec1d2458d52459aaa00f43ea05c1afffccfc75d0ba09.jpg)

![ce15464e2843e0b02bd5c9561fb034aadccabf7788ba736893ef80a0fc863bdc.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/ce15464e2843e0b02bd5c9561fb034aadccabf7788ba736893ef80a0fc863bdc.jpg)

![d76c3fffebb481ba49180bdf4ce1e18d6f56e929409a0131232443c3ffe94966.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/d76c3fffebb481ba49180bdf4ce1e18d6f56e929409a0131232443c3ffe94966.jpg)

![d94c21b0193aea6d9040875143842c612895b66d51c9d47d3b19ff227ff9f85e.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/d94c21b0193aea6d9040875143842c612895b66d51c9d47d3b19ff227ff9f85e.jpg)

![dd1148ac1025594fbba819c8bc9a665762ef7f74c68133de3d55f6b1838bede5.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/dd1148ac1025594fbba819c8bc9a665762ef7f74c68133de3d55f6b1838bede5.jpg)

![dd87ec38043fce01569db14a321fd9aa722b4c48e0ef1ef33011632d22695268.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/dd87ec38043fce01569db14a321fd9aa722b4c48e0ef1ef33011632d22695268.jpg)

![e6c06c8a50cc59bf40fa2c95bb6795e7afd38967a5e120a40a31dc6a6d263537.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/e6c06c8a50cc59bf40fa2c95bb6795e7afd38967a5e120a40a31dc6a6d263537.jpg)

![e975fe652b9c066605434cfe70c901482a0b982dd840efbdb6a8d62a07568188.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/e975fe652b9c066605434cfe70c901482a0b982dd840efbdb6a8d62a07568188.jpg)

![f2de1dbe62f52f7fecf8d860e77d391f52dc3090982ffb2f13bd884bda6e855a.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/f2de1dbe62f52f7fecf8d860e77d391f52dc3090982ffb2f13bd884bda6e855a.jpg)

![fff31351b1555718e20620c7c8c1fd8637c7c34ed54bd6c3c5c9fea993f33e8e.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/images/fff31351b1555718e20620c7c8c1fd8637c7c34ed54bd6c3c5c9fea993f33e8e.jpg)

### Tables

![0740ca317383d05f52693ee503d24133a8256b4472148604dff7fcc260de1e82.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/0740ca317383d05f52693ee503d24133a8256b4472148604dff7fcc260de1e82.jpg)

![1fdbbf798db9531d93b1298f5f9d71aff14868e56603c4c7fdc40a15e192bd79.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/1fdbbf798db9531d93b1298f5f9d71aff14868e56603c4c7fdc40a15e192bd79.jpg)

![21ab2b89fd800e2618bc914fddd0b808553edf34b744388eb62f8af860fd4bbf.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/21ab2b89fd800e2618bc914fddd0b808553edf34b744388eb62f8af860fd4bbf.jpg)

![266c7361ec71bc171bcddd777426bd6e3566e0d5e292993afd6d186d141c76b0.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/266c7361ec71bc171bcddd777426bd6e3566e0d5e292993afd6d186d141c76b0.jpg)

![514e61d5bfc0a7a073cb593dd936168500826123b30d3149fc4b4e83c059f959.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/514e61d5bfc0a7a073cb593dd936168500826123b30d3149fc4b4e83c059f959.jpg)

![9a19814c3e8207a4c62e5e3d397c391faddf1693bc1ead710879e01616fbfa18.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/9a19814c3e8207a4c62e5e3d397c391faddf1693bc1ead710879e01616fbfa18.jpg)

![acebe3d273b7b117806fa7efb21ff2c9d535cb802be6873b11fc0fd7ec746d61.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/acebe3d273b7b117806fa7efb21ff2c9d535cb802be6873b11fc0fd7ec746d61.jpg)

![b42bddf1adf2e2caf89022b6bd39720a45957076f8ed5c0bf7769f96131cef65.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/b42bddf1adf2e2caf89022b6bd39720a45957076f8ed5c0bf7769f96131cef65.jpg)

![b802ffcea1bc4b242ceaea111a836ad54a61a7d21f99d482899a5e8411fbb9c8.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/b802ffcea1bc4b242ceaea111a836ad54a61a7d21f99d482899a5e8411fbb9c8.jpg)

![bbbe55c6f14bd7404f2ef0935554c97f5c8e0e341e43416326e1942aab842b58.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/bbbe55c6f14bd7404f2ef0935554c97f5c8e0e341e43416326e1942aab842b58.jpg)

![c0f578603a9a3bcce3b3c27ae0fab03e35d132846b7d0a10191dfc2f6d6da7cf.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/c0f578603a9a3bcce3b3c27ae0fab03e35d132846b7d0a10191dfc2f6d6da7cf.jpg)

![c6328cff2f669179d0d0197a1e0e184be3e6a34cb2875ce647d21efe08c46ceb.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/c6328cff2f669179d0d0197a1e0e184be3e6a34cb2875ce647d21efe08c46ceb.jpg)

![e1ab5f37fff3eb2718459bdda42389aeb95218c21ee7134d548fc87077af41d1.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/e1ab5f37fff3eb2718459bdda42389aeb95218c21ee7134d548fc87077af41d1.jpg)

![e20c80da886472f269cc01cf389bfb719c78d789353b45ab7a91b6f3b305615b.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/e20c80da886472f269cc01cf389bfb719c78d789353b45ab7a91b6f3b305615b.jpg)

![e239ddcc6aef915e8a43e5626a2a6002de7f327252f97c51bf6537512373a394.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/e239ddcc6aef915e8a43e5626a2a6002de7f327252f97c51bf6537512373a394.jpg)

![e6f6ed1ac3236d8baf48915e25cbdc2c0f0e0c76c34800b0e970a0a6d9d539ee.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/e6f6ed1ac3236d8baf48915e25cbdc2c0f0e0c76c34800b0e970a0a6d9d539ee.jpg)

![f8fd28f79ba05f079709be328dcec46d19aae5cf82003be10fad263b5e4a65cf.jpg](../icml_results/1518_Distributionally%20Robust%20Causal%20Abstractions/tables/f8fd28f79ba05f079709be328dcec46d19aae5cf82003be10fad263b5e4a65cf.jpg)

<a id="paper-1498"></a>

## Optimal Self-Consistency for Efficient Reasoning with Large Language Models


### Images

![049984f768ed63a1adbeb50f5dd0f5855e92b83ec479eeb4d6732d69a6e92e77.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/049984f768ed63a1adbeb50f5dd0f5855e92b83ec479eeb4d6732d69a6e92e77.jpg)

![1555e251fca1ca77c2ae7f227f3c65b75adf0afb89bddf9c764076343322189f.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/1555e251fca1ca77c2ae7f227f3c65b75adf0afb89bddf9c764076343322189f.jpg)

![19eb29940a8409d968cf8c19fbb1a26a32682f1fa30a4075844fe399a044dcb2.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/19eb29940a8409d968cf8c19fbb1a26a32682f1fa30a4075844fe399a044dcb2.jpg)

![2502ad7e3f12ec678ddc1629d811b604bf10c938c624a0b01df45f0bf2d30423.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/2502ad7e3f12ec678ddc1629d811b604bf10c938c624a0b01df45f0bf2d30423.jpg)

![3194672d8a365f6e26bb696ae7ce8fc33a0268ceb3587652a091d957af532967.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/3194672d8a365f6e26bb696ae7ce8fc33a0268ceb3587652a091d957af532967.jpg)

![33c8c1d468af9bf09394099a31afa3f2179ed53b181b09ad54e221c86b8d5b77.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/33c8c1d468af9bf09394099a31afa3f2179ed53b181b09ad54e221c86b8d5b77.jpg)

![42340ca7718a2f7c6278165f1d951d9c8410b14a33342cffdf2f2d7929bbbf42.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/42340ca7718a2f7c6278165f1d951d9c8410b14a33342cffdf2f2d7929bbbf42.jpg)

![79fc8dd5fb9359b37970d13cf31d4e22a5cd749e2d75d9bcaea0ae6236598b8b.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/79fc8dd5fb9359b37970d13cf31d4e22a5cd749e2d75d9bcaea0ae6236598b8b.jpg)

![83d46372398fddcffc664f137502c68685cb7770ad692159516b72272a0097ad.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/83d46372398fddcffc664f137502c68685cb7770ad692159516b72272a0097ad.jpg)

![840b6cfff53c96c6604f4b08d5bfb035128d0866b8a9b6b978a388709962db69.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/840b6cfff53c96c6604f4b08d5bfb035128d0866b8a9b6b978a388709962db69.jpg)

![a9c67b673c308029d64a5206144cfd46e0dc8900cf16b6789517d66a5070dc69.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/a9c67b673c308029d64a5206144cfd46e0dc8900cf16b6789517d66a5070dc69.jpg)

![cf8a3a9c11455fd856f97fc876b546ed9bc49c55a5bc05a5870b7d873a2ace88.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/cf8a3a9c11455fd856f97fc876b546ed9bc49c55a5bc05a5870b7d873a2ace88.jpg)

![efbc73e9ceb99b4bf52fad3c0dc8c035963c7e7857d657244ccf113dca220e56.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/images/efbc73e9ceb99b4bf52fad3c0dc8c035963c7e7857d657244ccf113dca220e56.jpg)

### Tables

![169e193452da3e0fd51fc9fcbf210936893713897de784f715627d9d3bbe1ffc.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/tables/169e193452da3e0fd51fc9fcbf210936893713897de784f715627d9d3bbe1ffc.jpg)

![42501e05fd6c84f4d53af8da826f55ab5c63a85afb227f759c59e62c08122a2d.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/tables/42501e05fd6c84f4d53af8da826f55ab5c63a85afb227f759c59e62c08122a2d.jpg)

![59c08a55bd72909d253a3c095a829081ac750e76d5524a5676bbd4fa95386f5c.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/tables/59c08a55bd72909d253a3c095a829081ac750e76d5524a5676bbd4fa95386f5c.jpg)

![d2c1eb9890833abc545436b1cf492a6cc38cc84a14b2bb8c92354c7cc8594928.jpg](../icml_results/1519_Optimal%20Self-Consistency%20for%20Efficient%20Reasoning%20with%20Large%20Language%20Models/tables/d2c1eb9890833abc545436b1cf492a6cc38cc84a14b2bb8c92354c7cc8594928.jpg)

<a id="paper-1499"></a>

## LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification


### Images

![67ad4521b89c04fcb4a5b2760ca181c6b10b2a2b1611f165091da217d1ebd454.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/images/67ad4521b89c04fcb4a5b2760ca181c6b10b2a2b1611f165091da217d1ebd454.jpg)

![6833e7aec773fea0de5d047154e5585db0506ea033dd3c1250f73acbe83a4fb6.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/images/6833e7aec773fea0de5d047154e5585db0506ea033dd3c1250f73acbe83a4fb6.jpg)

![b1bb7c8cdc220e9d329cc252542ff391573ee544dfa83fe2f199f6290dbb5bb8.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/images/b1bb7c8cdc220e9d329cc252542ff391573ee544dfa83fe2f199f6290dbb5bb8.jpg)

![e79480c317a80db0a320775a91996cc40932f8a06c0e402ce051447c7c5821df.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/images/e79480c317a80db0a320775a91996cc40932f8a06c0e402ce051447c7c5821df.jpg)

### Tables

![07f3488c91ade6522c537c2d63206d687742895300c917c85e817c5e90a5a6aa.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/07f3488c91ade6522c537c2d63206d687742895300c917c85e817c5e90a5a6aa.jpg)

![51b3acb69cf73b0afccfb82125583145d5a119fcf7041f0b4fdf18afa9fb8b7a.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/51b3acb69cf73b0afccfb82125583145d5a119fcf7041f0b4fdf18afa9fb8b7a.jpg)

![6c72d54b8755678a53c4668d0f98112f8243466392bc7fac30ad0201074bc9c4.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/6c72d54b8755678a53c4668d0f98112f8243466392bc7fac30ad0201074bc9c4.jpg)

![6d4c7326549280ac9a70c814a5c0804d27ff4a073750c4763b48018b84b1708a.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/6d4c7326549280ac9a70c814a5c0804d27ff4a073750c4763b48018b84b1708a.jpg)

![87d8d50d67501415823d13289dc297fbcf361f6a90613b672acd70a18f1613bc.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/87d8d50d67501415823d13289dc297fbcf361f6a90613b672acd70a18f1613bc.jpg)

![a1cd15a77ee629e7da3a9f88173c20aa7ee1c6541dee47c8c02a886059ed674d.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/a1cd15a77ee629e7da3a9f88173c20aa7ee1c6541dee47c8c02a886059ed674d.jpg)

![bf0901c6054d2b3e7d1ac62be7151d7098e1b5e921ed110983eba41561bfd1b4.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/bf0901c6054d2b3e7d1ac62be7151d7098e1b5e921ed110983eba41561bfd1b4.jpg)

![c09951854fbf716ffc31a8e3ca21b2591064ad389e7b04d69f3dc49b04c75d8b.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/c09951854fbf716ffc31a8e3ca21b2591064ad389e7b04d69f3dc49b04c75d8b.jpg)

![e37c8800f513f63d76e22966b8f1eccc73d38412f3ce0e0bf3a42d438ef88a46.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/e37c8800f513f63d76e22966b8f1eccc73d38412f3ce0e0bf3a42d438ef88a46.jpg)

![f7f98bb4f8809435e5880d19e4ffdaa02ce95618f0b4d31b166d9bde371387fb.jpg](../icml_results/1520_LERD_%20Latent%20Event-Relational%20Dynamics%20for%20Neurodegenerative%20Classification/tables/f7f98bb4f8809435e5880d19e4ffdaa02ce95618f0b4d31b166d9bde371387fb.jpg)

<a id="paper-1500"></a>

## SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems


### Images

![0524c43c7c4e43e4e0a42e8e92fc5b32bf5939380592d96748c572042c937768.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/0524c43c7c4e43e4e0a42e8e92fc5b32bf5939380592d96748c572042c937768.jpg)

![0e3e83a6602cd3cb009b85b9e54756cec6321a56369e545eebf00de5875c1898.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/0e3e83a6602cd3cb009b85b9e54756cec6321a56369e545eebf00de5875c1898.jpg)

![4243b3710e248ec4e9064b51ee1e6064ce572098c2f786634188a2d6ab1d7478.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/4243b3710e248ec4e9064b51ee1e6064ce572098c2f786634188a2d6ab1d7478.jpg)

![9ee2a0bf47ba3cd37488134ff20fe036a1b0b0d58f9d579ef36a62b5c22ec7f3.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/9ee2a0bf47ba3cd37488134ff20fe036a1b0b0d58f9d579ef36a62b5c22ec7f3.jpg)

![d81b511561a4985f0128ccd1fb3424360a7104d7436c740b47ed7904b996ac2b.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/d81b511561a4985f0128ccd1fb3424360a7104d7436c740b47ed7904b996ac2b.jpg)

![ff8da1a1eca935f03be13fbb108afbc5e62b5ab5320512544e9cff32458611e6.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/images/ff8da1a1eca935f03be13fbb108afbc5e62b5ab5320512544e9cff32458611e6.jpg)

### Tables

![0bfc8c0e6b710c0220a93eb7af10a31c4f89c9d1529a9db65f25ff007ec8b45e.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/0bfc8c0e6b710c0220a93eb7af10a31c4f89c9d1529a9db65f25ff007ec8b45e.jpg)

![600ffbf4d82fcb1962552ab5626e30e6a98de67441c4e382b25f6815f2634d4f.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/600ffbf4d82fcb1962552ab5626e30e6a98de67441c4e382b25f6815f2634d4f.jpg)

![7d044512be07eb8266c834cab4ff1440ebe21c013842c948e6534a1f1ce02ee6.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/7d044512be07eb8266c834cab4ff1440ebe21c013842c948e6534a1f1ce02ee6.jpg)

![a15c06b16f5848efe314748dbfb08a3c7ea17597bd6243a75c2c23eb161840c8.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/a15c06b16f5848efe314748dbfb08a3c7ea17597bd6243a75c2c23eb161840c8.jpg)

![b3753c7a4a00c12f3a3db4e4663cf54d6205bdee098c4f5ae6e73a7c81874c88.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/b3753c7a4a00c12f3a3db4e4663cf54d6205bdee098c4f5ae6e73a7c81874c88.jpg)

![b386d8e21c4d2320091925e226cd1405b862a8bcbec28e7c241b49a5ddea11b1.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/b386d8e21c4d2320091925e226cd1405b862a8bcbec28e7c241b49a5ddea11b1.jpg)

![b80bf38d19ef35bf0944c620e97289064dde12a39bfcf1355d75f174974e9b9c.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/b80bf38d19ef35bf0944c620e97289064dde12a39bfcf1355d75f174974e9b9c.jpg)

![eee20e4b4da190a72564c20ba12df7bfebf763cc6174caf85f97dc29a9911bf5.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/eee20e4b4da190a72564c20ba12df7bfebf763cc6174caf85f97dc29a9911bf5.jpg)

![fdb6d1e02773971000ef66b30d7d72ae21c9ee0446777bf72ad3e2bdfd00b82a.jpg](../icml_results/1521_SkillTrojan_%20Backdoor%20Attacks%20on%20Skill-Based%20Agent%20Systems/tables/fdb6d1e02773971000ef66b30d7d72ae21c9ee0446777bf72ad3e2bdfd00b82a.jpg)

<a id="paper-1501"></a>

## WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching


### Images

![17978db3993170a1ff7fa58d9894f3a0b963da75db610aec80dadf2c54e2f90a.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/17978db3993170a1ff7fa58d9894f3a0b963da75db610aec80dadf2c54e2f90a.jpg)

![1f3659c056fe94a81f0277b6ade2c6ac989aa4a9fe96ac967208204016a90b1c.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/1f3659c056fe94a81f0277b6ade2c6ac989aa4a9fe96ac967208204016a90b1c.jpg)

![29362aff900a584c6c216f37fbde6856813977a12617202df0b1d4ca82ac6723.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/29362aff900a584c6c216f37fbde6856813977a12617202df0b1d4ca82ac6723.jpg)

![4dad571c9540359a678971f1d2c851df8f1094f842147202d9eb297c6cf4d13b.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/4dad571c9540359a678971f1d2c851df8f1094f842147202d9eb297c6cf4d13b.jpg)

![82b6aebbe66af35b78c92c78103694eef4163d08d3e6f62810e15d302ad751b2.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/82b6aebbe66af35b78c92c78103694eef4163d08d3e6f62810e15d302ad751b2.jpg)

![9652604a4e5f02eafa1e174ddabaa77253007ca65692e6fa9daf5a9bfbda21fd.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/9652604a4e5f02eafa1e174ddabaa77253007ca65692e6fa9daf5a9bfbda21fd.jpg)

![ae841dc0c34baf02938ece99083611b3a96070f30dd8871e4f979026b61aabba.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/ae841dc0c34baf02938ece99083611b3a96070f30dd8871e4f979026b61aabba.jpg)

![bb3db158c39c96e03918ddca7853f22aefbc3560ce231e86603e9aad606b60db.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/bb3db158c39c96e03918ddca7853f22aefbc3560ce231e86603e9aad606b60db.jpg)

![bd476d8484c5488658bfaa8e4ff7a6c870b58e6353b0594e619c19d8da72fa1b.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/bd476d8484c5488658bfaa8e4ff7a6c870b58e6353b0594e619c19d8da72fa1b.jpg)

![c831e001f8034c13cf9c6edc767799f8b6fde50e2a9d63d48615cee6abf075f0.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/c831e001f8034c13cf9c6edc767799f8b6fde50e2a9d63d48615cee6abf075f0.jpg)

![d0950438a8a97022d72d33e720a63bdfb6010b04ad67abba1618ba6231cf0f4e.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/d0950438a8a97022d72d33e720a63bdfb6010b04ad67abba1618ba6231cf0f4e.jpg)

![e017ef4514133df606ec74bae51443682f6313323094abb1df216baad0b80c86.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/e017ef4514133df606ec74bae51443682f6313323094abb1df216baad0b80c86.jpg)

![e80da957dca220eae7eae89cff2446c78349909279f2ae41e2db350030928838.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/images/e80da957dca220eae7eae89cff2446c78349909279f2ae41e2db350030928838.jpg)

### Tables

![0025616379e049f153281c10ed30f458b42c7a4624c4d48049fcc16f6916a645.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/0025616379e049f153281c10ed30f458b42c7a4624c4d48049fcc16f6916a645.jpg)

![0650e35175ada50a60b285b1b7bd51ea140907e9d25066b22e245aa493340338.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/0650e35175ada50a60b285b1b7bd51ea140907e9d25066b22e245aa493340338.jpg)

![3f248b6517792bc091a755e75f8e714bba2d4f42cb863eae9ae8da1099d2700e.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/3f248b6517792bc091a755e75f8e714bba2d4f42cb863eae9ae8da1099d2700e.jpg)

![4f6730dcc324394fcb636fcda0b77d1481a41e8eb83b2ac657fc5ab8cec9722b.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/4f6730dcc324394fcb636fcda0b77d1481a41e8eb83b2ac657fc5ab8cec9722b.jpg)

![674657c6661d9555d75e2651c8b88fe17700243981284680ebcca69839462afc.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/674657c6661d9555d75e2651c8b88fe17700243981284680ebcca69839462afc.jpg)

![722f8d7adc4fb921df7b414be1da6fb77eb8cd935256bcc64033bfd8ae282fa7.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/722f8d7adc4fb921df7b414be1da6fb77eb8cd935256bcc64033bfd8ae282fa7.jpg)

![8ae6f26540d9f86c88447b75e063b0ed900bfc91ebb4b43e984ed07e3fffe400.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/8ae6f26540d9f86c88447b75e063b0ed900bfc91ebb4b43e984ed07e3fffe400.jpg)

![b63788081be211a5832f49475dec878fbc77ae15b924f013243a6bfe1500c4c6.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/b63788081be211a5832f49475dec878fbc77ae15b924f013243a6bfe1500c4c6.jpg)

![c223826fc059fe3e3ce811d08916b8beccb96f91da8b68970826a1f9aae874ac.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/c223826fc059fe3e3ce811d08916b8beccb96f91da8b68970826a1f9aae874ac.jpg)

![e32ed5b58f52ae384e27db7b6ba5ed704f2700da124898bcb2fd756e4d787adb.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/e32ed5b58f52ae384e27db7b6ba5ed704f2700da124898bcb2fd756e4d787adb.jpg)

![f556d67c12c42890539981b2b19c2ef93fbd51102271e9f128e4a412ff03dd29.jpg](../icml_results/1522_WorldCache_%20Accelerating%20World%20Models%20for%20Free%20via%20Heterogeneous%20Token%20Caching/tables/f556d67c12c42890539981b2b19c2ef93fbd51102271e9f128e4a412ff03dd29.jpg)

<a id="paper-1502"></a>

## What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT


### Images

![06b23b85603a68ee6edb29d9cc21079b241e296c2c9387c251ecea16138d2241.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/06b23b85603a68ee6edb29d9cc21079b241e296c2c9387c251ecea16138d2241.jpg)

![13f2f03c70d1a77643833c959ad6686317a31ba3da7b0bf8e2e5d375d89fc139.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/13f2f03c70d1a77643833c959ad6686317a31ba3da7b0bf8e2e5d375d89fc139.jpg)

![19b3c8e2bb48b7ab824aaf3927f3eeec21edcd81581b33e24ad2757fa2064c8b.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/19b3c8e2bb48b7ab824aaf3927f3eeec21edcd81581b33e24ad2757fa2064c8b.jpg)

![31fe80c731275750adb87d9e954d3cfec6b76416a61b7d2c7d5a2a6fecda8a9a.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/31fe80c731275750adb87d9e954d3cfec6b76416a61b7d2c7d5a2a6fecda8a9a.jpg)

![35b53a775130e83a4b37ce6c053f0db4a0de9aee378abceb38b1ca6aaa1238c0.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/35b53a775130e83a4b37ce6c053f0db4a0de9aee378abceb38b1ca6aaa1238c0.jpg)

![38a5126367fd4451cfa9d7a94c34878340d07f50c448b94bd4e77e2caa27d442.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/38a5126367fd4451cfa9d7a94c34878340d07f50c448b94bd4e77e2caa27d442.jpg)

![39239ff0c9f3966350b7fad6d2a9b65b65370027dcfd5c1558d68d29fd00ff47.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/39239ff0c9f3966350b7fad6d2a9b65b65370027dcfd5c1558d68d29fd00ff47.jpg)

![486415218af164e400134a170422bbd271e02d81366d0d5bbdc3a956116e5bc3.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/486415218af164e400134a170422bbd271e02d81366d0d5bbdc3a956116e5bc3.jpg)

![4f09368a246f804699cc07ca1377e9611d51ff4bc21c3c59ed78b0ad13246903.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/4f09368a246f804699cc07ca1377e9611d51ff4bc21c3c59ed78b0ad13246903.jpg)

![717779acdfd1681cd9a8ba4550c0e1ecca4481f4c6d6a58ea8907bee085e9c59.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/717779acdfd1681cd9a8ba4550c0e1ecca4481f4c6d6a58ea8907bee085e9c59.jpg)

![9fa1b083ae4abd7dc0cecb33a5bfa6cedc5e872e232bcea96f1fb1814d526443.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/9fa1b083ae4abd7dc0cecb33a5bfa6cedc5e872e232bcea96f1fb1814d526443.jpg)

![a655050351244d5869e4635ec04e8802391beca7ad153d7438bb98a2f4dd286f.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/a655050351244d5869e4635ec04e8802391beca7ad153d7438bb98a2f4dd286f.jpg)

![ac3ae705f4cc795f1ac23bc2132dadbb14e5d374886e4b2e7f7a5f5cfd05cc7b.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/ac3ae705f4cc795f1ac23bc2132dadbb14e5d374886e4b2e7f7a5f5cfd05cc7b.jpg)

![fd2a90bd7ba60f4b5a9a70c336329ab79f9f34531de2371e36d643eacd27bff5.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/images/fd2a90bd7ba60f4b5a9a70c336329ab79f9f34531de2371e36d643eacd27bff5.jpg)

### Tables

![7bcba9db91f2a7e5a0c35efe832cec590b803847bfb9c29f166b4d27fc661e34.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/7bcba9db91f2a7e5a0c35efe832cec590b803847bfb9c29f166b4d27fc661e34.jpg)

![7d22c0d3463d34a62bf7ba7ba8aabc88d20806d0d2bd47d272ea2e66c161341c.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/7d22c0d3463d34a62bf7ba7ba8aabc88d20806d0d2bd47d272ea2e66c161341c.jpg)

![a45d8b5be36d5d28741303feec169ed9378a6523d5437790938f0863c5c484cc.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/a45d8b5be36d5d28741303feec169ed9378a6523d5437790938f0863c5c484cc.jpg)

![b1fefa0834d5ec13346bc70f91e6b433cabea75dba45fb7498611ffaa825effe.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/b1fefa0834d5ec13346bc70f91e6b433cabea75dba45fb7498611ffaa825effe.jpg)

![c18d037c06eb7615a05d93482c4cd689f3e774092554d365baaeef77232f39f1.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/c18d037c06eb7615a05d93482c4cd689f3e774092554d365baaeef77232f39f1.jpg)

![df042589341c824e059d1b35026de29ac17cbf9130bd2940015578147a23219a.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/df042589341c824e059d1b35026de29ac17cbf9130bd2940015578147a23219a.jpg)

![f5eebf24e08df253a186db64e584f05576830366e1a8bcd2decedf45224e7855.jpg](../icml_results/1523_What%20Characterizes%20Effective%20Reasoning_%20Revisiting%20Length%2C%20Review%2C%20and%20Structure%20of%20CoT/tables/f5eebf24e08df253a186db64e584f05576830366e1a8bcd2decedf45224e7855.jpg)

<a id="paper-1503"></a>

## Leveraging Machine Unlearning for Cost-Efficient Preference Alignment


### Images

![0b26b637f25833cb83af6bdc7a17bcb423f2a64bd012c453d67b7bb248bfd96f.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/images/0b26b637f25833cb83af6bdc7a17bcb423f2a64bd012c453d67b7bb248bfd96f.jpg)

![43089c5b9148932e8ac9e65103c0b931b8683297756f184ee4c3186755b8f860.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/images/43089c5b9148932e8ac9e65103c0b931b8683297756f184ee4c3186755b8f860.jpg)

![65a0f74025f12e80c7e2e46477e4c77a5637fed618d678d509d6d506aa55a59d.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/images/65a0f74025f12e80c7e2e46477e4c77a5637fed618d678d509d6d506aa55a59d.jpg)

![898200679ff61d6220ddd7027732ba33ce835e125c052697c70d8ba81abb415b.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/images/898200679ff61d6220ddd7027732ba33ce835e125c052697c70d8ba81abb415b.jpg)

![bd4e04e1bc248dceaeae915e2c4588ce4af02660cc7be683e15e85180f4dcd59.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/images/bd4e04e1bc248dceaeae915e2c4588ce4af02660cc7be683e15e85180f4dcd59.jpg)

### Tables

![0394de6fe691634f0f2ac0dd8ee9618f09e4e5952a597b0b910fc95392b7ad84.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/0394de6fe691634f0f2ac0dd8ee9618f09e4e5952a597b0b910fc95392b7ad84.jpg)

![0a5fd27355852dcc2795c021a83fc0effa759cec160f1c801a964f049a1e584d.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/0a5fd27355852dcc2795c021a83fc0effa759cec160f1c801a964f049a1e584d.jpg)

![377bb4bb35a61e51901998569f1faf80b5e73d04eb487864b0d36c03eef78d04.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/377bb4bb35a61e51901998569f1faf80b5e73d04eb487864b0d36c03eef78d04.jpg)

![635de5c6a392ca7a1a5cda0266a96032d688507581f11aecb35c7491b7e4a41b.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/635de5c6a392ca7a1a5cda0266a96032d688507581f11aecb35c7491b7e4a41b.jpg)

![79b23e4b47e1fd5e8f2bfbb349fae6a831820090c43803736b08b2608283bfd0.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/79b23e4b47e1fd5e8f2bfbb349fae6a831820090c43803736b08b2608283bfd0.jpg)

![af2f89d0aa53c761fe36c1e3f9c77ce529deb123b11346d9116b06448258133f.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/af2f89d0aa53c761fe36c1e3f9c77ce529deb123b11346d9116b06448258133f.jpg)

![b1f6009ae4221de9089e15fd200ecaa7036642d12f506caa8a4d343835c968c4.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/b1f6009ae4221de9089e15fd200ecaa7036642d12f506caa8a4d343835c968c4.jpg)

![ec78fdd01d6dec99fa3b86f42f301f897f93ede4eeea29976edf280463f32ef1.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/ec78fdd01d6dec99fa3b86f42f301f897f93ede4eeea29976edf280463f32ef1.jpg)

![f145d5606d4289e7bb21d692eb266fd1ea7dab9faa1375f8764386cf5aec9f35.jpg](../icml_results/1524_Leveraging%20Machine%20Unlearning%20for%20Cost-Efficient%20Preference%20Alignment/tables/f145d5606d4289e7bb21d692eb266fd1ea7dab9faa1375f8764386cf5aec9f35.jpg)

<a id="paper-1504"></a>

## Explainable Forensics of Manipulated Segments in Untrimmed Long Videos


### Images

![1e16ca791bb57edd5705d32b584856abb50b4230d7c3f7f84b7843b838babcf3.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/1e16ca791bb57edd5705d32b584856abb50b4230d7c3f7f84b7843b838babcf3.jpg)

![414981799febdf7541323bf15bdaa2bf185bb162265d637146b9194bfc73046f.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/414981799febdf7541323bf15bdaa2bf185bb162265d637146b9194bfc73046f.jpg)

![476d16cde4669e41edd3e355830ba1ea7d343002877ba0b4a29af32fb7074794.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/476d16cde4669e41edd3e355830ba1ea7d343002877ba0b4a29af32fb7074794.jpg)

![50cf9e150122240c47d43c32b96ed08f30c1892247443b596a76d5103089c8e9.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/50cf9e150122240c47d43c32b96ed08f30c1892247443b596a76d5103089c8e9.jpg)

![7dde82d8884b5759e4d5ed456e2ee5fe99e11f13a2bb0c6df1bc158cb1bd4268.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/7dde82d8884b5759e4d5ed456e2ee5fe99e11f13a2bb0c6df1bc158cb1bd4268.jpg)

![9a4641c4eb2b1fc3450780557f905d64a078b2f4584a3c67bd58b253e43fe91d.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/9a4641c4eb2b1fc3450780557f905d64a078b2f4584a3c67bd58b253e43fe91d.jpg)

![c1f865b5a1320cb0a3f0d2f43d45029d830c2f1f58c1a079ce1e444c8265df34.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/c1f865b5a1320cb0a3f0d2f43d45029d830c2f1f58c1a079ce1e444c8265df34.jpg)

![cc2320d6f22a57b286266ca53f1c7048434b387c8f6979219bf9661dc95244de.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/cc2320d6f22a57b286266ca53f1c7048434b387c8f6979219bf9661dc95244de.jpg)

![f2dcdb619787dfc9e41ce658122debc8d7b2221609882abf842babe599b64883.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/f2dcdb619787dfc9e41ce658122debc8d7b2221609882abf842babe599b64883.jpg)

![f62cf5469e0b68b14ed2d6b15b8a5425d62e58a7248d1a441da799e3ec3f6144.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/images/f62cf5469e0b68b14ed2d6b15b8a5425d62e58a7248d1a441da799e3ec3f6144.jpg)

### Tables

![20a5a1b412049f9c31f49c3bea9f6eab438dec82c6b2134d79cf45c4d343c48f.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/20a5a1b412049f9c31f49c3bea9f6eab438dec82c6b2134d79cf45c4d343c48f.jpg)

![291e5f0740c928b883424a3c98cd858fc4837fab034b364c4bc82d6063784960.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/291e5f0740c928b883424a3c98cd858fc4837fab034b364c4bc82d6063784960.jpg)

![41690abcd891dab3adf7bf4ef196999396235a181f341d196b8c18b3df5284b0.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/41690abcd891dab3adf7bf4ef196999396235a181f341d196b8c18b3df5284b0.jpg)

![7a87d6ab1a32bf8a5f3577e857338b923f7972fc9561a8a0530823160f71ff73.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/7a87d6ab1a32bf8a5f3577e857338b923f7972fc9561a8a0530823160f71ff73.jpg)

![7be3e50cdedfd8bd9376056b13a129b9d82f7def7fcf3f9f01a61002efa42e10.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/7be3e50cdedfd8bd9376056b13a129b9d82f7def7fcf3f9f01a61002efa42e10.jpg)

![84cdb5a75ef86b6cfbac681a34874b2ee8c9f0847898af850300d0a2be40ae46.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/84cdb5a75ef86b6cfbac681a34874b2ee8c9f0847898af850300d0a2be40ae46.jpg)

![e313d848c5e535b65958a28dda76e15ac1b1f03be2cf4c5643688bbc55ade0fe.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/e313d848c5e535b65958a28dda76e15ac1b1f03be2cf4c5643688bbc55ade0fe.jpg)

![e3bce4fbf5763a6a8714b329ede0a66ed30d009c5ea8085e84dd38786312ea94.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/e3bce4fbf5763a6a8714b329ede0a66ed30d009c5ea8085e84dd38786312ea94.jpg)

![f1b4aaa2d91e53ddb625406d329212a6e29034c3d48247411a449a2e20c499b2.jpg](../icml_results/1525_Explainable%20Forensics%20of%20Manipulated%20Segments%20in%20Untrimmed%20Long%20Videos/tables/f1b4aaa2d91e53ddb625406d329212a6e29034c3d48247411a449a2e20c499b2.jpg)

<a id="paper-1505"></a>

## Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning


### Images

![0d331e575d42be807e52fbd31c4e990222707a4d1a013b12ef0e26342f2d9fe4.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/0d331e575d42be807e52fbd31c4e990222707a4d1a013b12ef0e26342f2d9fe4.jpg)

![19a73e2a62d72f9abc64e158e9ae021699dff6d9af8f0a068422149f4d3e90c5.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/19a73e2a62d72f9abc64e158e9ae021699dff6d9af8f0a068422149f4d3e90c5.jpg)

![38fb362aeabe54b4f60d63592777f2b70f46131d0e2b170f7068e247198334e6.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/38fb362aeabe54b4f60d63592777f2b70f46131d0e2b170f7068e247198334e6.jpg)

![4455dd8b9842992fce5273423fc84682287e2a3dcbfd84b3074f96a0f83cee59.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/4455dd8b9842992fce5273423fc84682287e2a3dcbfd84b3074f96a0f83cee59.jpg)

![7871ca33819aa7b0229937741771cedfb2f37f8889ce3931d4003c79f6b417bb.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/7871ca33819aa7b0229937741771cedfb2f37f8889ce3931d4003c79f6b417bb.jpg)

![c367c0210dc26e54cc0eade12fef04aed0cb1335f6d5f8ea497047a1b9b9f287.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/c367c0210dc26e54cc0eade12fef04aed0cb1335f6d5f8ea497047a1b9b9f287.jpg)

![c69eab479594aeb1d1782ba6d308d95ff8169908bf107f0898397817b25c1ed3.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/c69eab479594aeb1d1782ba6d308d95ff8169908bf107f0898397817b25c1ed3.jpg)

![cb4316561f6d3542f58b70b668e13228745e92a4824b56a898ef2cdf4827d35a.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/cb4316561f6d3542f58b70b668e13228745e92a4824b56a898ef2cdf4827d35a.jpg)

![e6f36d6d0084b76d597ecfa522f265fde86b1848ac5c4d7a977bbae8db57dcc0.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/images/e6f36d6d0084b76d597ecfa522f265fde86b1848ac5c4d7a977bbae8db57dcc0.jpg)

### Tables

![2c979040af0c6affd12717ccf9743c59d388f2355a1baff2e9390145eccd182e.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/2c979040af0c6affd12717ccf9743c59d388f2355a1baff2e9390145eccd182e.jpg)

![471be717fd4bdcf37ea8a366eab7fc72af34930ba28e1be78bb091a4d7d5ddcd.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/471be717fd4bdcf37ea8a366eab7fc72af34930ba28e1be78bb091a4d7d5ddcd.jpg)

![53f22e17da2fd7357d050262489c86108af7b681976e73ba741ae474f3607fa1.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/53f22e17da2fd7357d050262489c86108af7b681976e73ba741ae474f3607fa1.jpg)

![77696e6ff07cf82be528f89612a7cb129fff068c2196f5477cb58d2752809ae1.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/77696e6ff07cf82be528f89612a7cb129fff068c2196f5477cb58d2752809ae1.jpg)

![7d0bc855a40cefab9cb625552215816d34d3edd4e8cdd6efab687e0a984c09fe.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/7d0bc855a40cefab9cb625552215816d34d3edd4e8cdd6efab687e0a984c09fe.jpg)

![7e623bb712e6b40be52ab2ef2f01372a2b1e9e4320376f33135a8a4c117281a2.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/7e623bb712e6b40be52ab2ef2f01372a2b1e9e4320376f33135a8a4c117281a2.jpg)

![8860021067ebffaffc857d30e8c6d768a7238bf96e2e14086e1ea9cf3424b863.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/8860021067ebffaffc857d30e8c6d768a7238bf96e2e14086e1ea9cf3424b863.jpg)

![8b37c50c3dea66c7ce0def1d03d89413c5f18cd782f387f1977ce9f7bbdc8a31.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/8b37c50c3dea66c7ce0def1d03d89413c5f18cd782f387f1977ce9f7bbdc8a31.jpg)

![9c77676b985c21e6d216a9a2189e6354c010413052922567296bf4b54f5e7635.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/9c77676b985c21e6d216a9a2189e6354c010413052922567296bf4b54f5e7635.jpg)

![b0d259127e818e37713e2e3b78ba72265e5d2cee2ccfb46952b7b7db16d2db58.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/b0d259127e818e37713e2e3b78ba72265e5d2cee2ccfb46952b7b7db16d2db58.jpg)

![b86e487a16961856f97d71ee0346407e84250064c545f97950c43fc298f76b42.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/b86e487a16961856f97d71ee0346407e84250064c545f97950c43fc298f76b42.jpg)

![c0d4179edd1cf9761a6f8ae9d8be8676d55ff3f8bd84fed31e2e96a14d155474.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/c0d4179edd1cf9761a6f8ae9d8be8676d55ff3f8bd84fed31e2e96a14d155474.jpg)

![d0d0fc42b73958635517b6b4a9ae8a82634a81297a8457df3dc36b3c5935605f.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/d0d0fc42b73958635517b6b4a9ae8a82634a81297a8457df3dc36b3c5935605f.jpg)

![fe58813508b659495421b5eceae852681a743ef1086c85d650e1c059a2d7c2ad.jpg](../icml_results/1526_Learning%20in%20the%20Fisher%20Subspace_%20A%20Guided%20Initialization%20for%20LoRA%20Fine-Tuning/tables/fe58813508b659495421b5eceae852681a743ef1086c85d650e1c059a2d7c2ad.jpg)

<a id="paper-1506"></a>

## ExpWeaver: LLM Agents Learn from Experience via Latent RAG


### Images

![58e49c9eb2c0db2744260800c678c5e340a1b8efba000c87aa1be587d2b888bb.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/images/58e49c9eb2c0db2744260800c678c5e340a1b8efba000c87aa1be587d2b888bb.jpg)

![60366801757582ac4932160172f26a029f1ce8e4a42bf80a075b62f49e8c3f95.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/images/60366801757582ac4932160172f26a029f1ce8e4a42bf80a075b62f49e8c3f95.jpg)

![8d4d9f6100034fc584dabe751ac22154bc6b6a3e1b48d3b2839cce722466d806.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/images/8d4d9f6100034fc584dabe751ac22154bc6b6a3e1b48d3b2839cce722466d806.jpg)

![facd45e42ab215e8f4afb6696c92a04317c00a4965f18e25d70079ecc2091bf7.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/images/facd45e42ab215e8f4afb6696c92a04317c00a4965f18e25d70079ecc2091bf7.jpg)

### Tables

![059964607228fc92168631fc6012c482ba906a853b6732b5ad9957f6211e73a0.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/059964607228fc92168631fc6012c482ba906a853b6732b5ad9957f6211e73a0.jpg)

![0931029791bdf494b4ebf12c0c0eb3702aef73db019628df74324379b3ab8e64.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/0931029791bdf494b4ebf12c0c0eb3702aef73db019628df74324379b3ab8e64.jpg)

![1496c2bd8532f6d0a19f9c6f51c7dd4a8b550cc0c4ed0f109d4bdd99e66dac57.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/1496c2bd8532f6d0a19f9c6f51c7dd4a8b550cc0c4ed0f109d4bdd99e66dac57.jpg)

![1c72c54159184c409890e341d44087a23f3632d84d532e45a012d416f1c80425.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/1c72c54159184c409890e341d44087a23f3632d84d532e45a012d416f1c80425.jpg)

![294bc5a3bc739aca600dde6e3ea5ddc354ba080714b6bb472d2923498ccfd32f.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/294bc5a3bc739aca600dde6e3ea5ddc354ba080714b6bb472d2923498ccfd32f.jpg)

![2becff77954077fe3757eb19d4e9e52f5b6d5ee8c9931caeef232f3068196e7c.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/2becff77954077fe3757eb19d4e9e52f5b6d5ee8c9931caeef232f3068196e7c.jpg)

![2d2b0655983f13a2555c248678b0c78030efbbdd97a9530d1b5a2dffd1eaa416.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/2d2b0655983f13a2555c248678b0c78030efbbdd97a9530d1b5a2dffd1eaa416.jpg)

![32b663119a6859c17443babb6e073201c6f58307029945718f9f406862ab378e.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/32b663119a6859c17443babb6e073201c6f58307029945718f9f406862ab378e.jpg)

![348ded5f9d295256b4425cbf4ebe82303785dd0eb3d4d4a128bddd8519c68619.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/348ded5f9d295256b4425cbf4ebe82303785dd0eb3d4d4a128bddd8519c68619.jpg)

![436fab52c0d029a4047b7529ce5d2ab4cf041cb247a81faa0fb4c289671261e3.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/436fab52c0d029a4047b7529ce5d2ab4cf041cb247a81faa0fb4c289671261e3.jpg)

![4bee758a98bbdf262f3495a7b869ab338e1e60ac0c8a6753fe09af3115b63fcb.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/4bee758a98bbdf262f3495a7b869ab338e1e60ac0c8a6753fe09af3115b63fcb.jpg)

![7b30f2103c32875a52614e750e40bad15ffca4b76e97c57470ae7c9f6a12a227.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/7b30f2103c32875a52614e750e40bad15ffca4b76e97c57470ae7c9f6a12a227.jpg)

![7c85bfa87452c9c2404ae3b96f6fe649b22c0c6d4fc18d4c718aff0fc3f400e7.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/7c85bfa87452c9c2404ae3b96f6fe649b22c0c6d4fc18d4c718aff0fc3f400e7.jpg)

![7cdc0eb69b0d0feb57d3c1f251e1142e283f1282cd0f4eb8378726775791e495.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/7cdc0eb69b0d0feb57d3c1f251e1142e283f1282cd0f4eb8378726775791e495.jpg)

![8c7213a5a7152283c386ac78a1f3a5e5b28415fa3975f11088b7e7505f3e96ee.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/8c7213a5a7152283c386ac78a1f3a5e5b28415fa3975f11088b7e7505f3e96ee.jpg)

![9371d620d27efba080620ac231c0c1ff6fd9bca014805d9a42edf62771697346.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/9371d620d27efba080620ac231c0c1ff6fd9bca014805d9a42edf62771697346.jpg)

![9f3063a9ad89ab08017b790817b00d9fc5c43dbd574197c2a25675c0872cc5c5.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/9f3063a9ad89ab08017b790817b00d9fc5c43dbd574197c2a25675c0872cc5c5.jpg)

![a3298bb5ee1e9bc999217849fe8d67b277cea8e5fb1f3b7ecd8c3aa29e9f8b71.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/a3298bb5ee1e9bc999217849fe8d67b277cea8e5fb1f3b7ecd8c3aa29e9f8b71.jpg)

![a9b18fe37641034b94ef4d9fd030a2d905154dbcdc150f360b93461efdb459c4.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/a9b18fe37641034b94ef4d9fd030a2d905154dbcdc150f360b93461efdb459c4.jpg)

![aed7c7038df3fbe100b894ec69ea363afc01f682fc4fbe85f9587e2e7abbb4d4.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/aed7c7038df3fbe100b894ec69ea363afc01f682fc4fbe85f9587e2e7abbb4d4.jpg)

![b18d9031db4135cc5a18c6f0416ae0926d06de854c18d3f6ca5503d85233a3bd.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/b18d9031db4135cc5a18c6f0416ae0926d06de854c18d3f6ca5503d85233a3bd.jpg)

![b1a4138a946ae4ca61648c7514dbb02e36527a51f76af42a9f171cf42b4d0ab8.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/b1a4138a946ae4ca61648c7514dbb02e36527a51f76af42a9f171cf42b4d0ab8.jpg)

![b22b8588129a1ab362157da992e1e03e137589fe876d91af1c59a33dbeade746.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/b22b8588129a1ab362157da992e1e03e137589fe876d91af1c59a33dbeade746.jpg)

![c079ec07330af611e5dc1da761dc692f7424f46f45d4be82e83fac1aa912e99d.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/c079ec07330af611e5dc1da761dc692f7424f46f45d4be82e83fac1aa912e99d.jpg)

![c4878a18fea1c43894054ee51f69f14bb23c14e59d97a399099ae33e75ff58ce.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/c4878a18fea1c43894054ee51f69f14bb23c14e59d97a399099ae33e75ff58ce.jpg)

![c6146991f87c1c0a37ad7fdccd87b1226215d3881921166df7f9c59a5e2e99b2.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/c6146991f87c1c0a37ad7fdccd87b1226215d3881921166df7f9c59a5e2e99b2.jpg)

![c7b9b05ffae3a9ffbcd270f561dce0b7e8574926cae20014f19490fefddb47cc.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/c7b9b05ffae3a9ffbcd270f561dce0b7e8574926cae20014f19490fefddb47cc.jpg)

![c8581cb15b7226b5cac9a34700d9d49831116941e68bb1d9516b6c975d26a873.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/c8581cb15b7226b5cac9a34700d9d49831116941e68bb1d9516b6c975d26a873.jpg)

![d156399064f15c7a55f1a7db31289ba8efbe0aab9fb51497c5a7a741e010fe9b.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/d156399064f15c7a55f1a7db31289ba8efbe0aab9fb51497c5a7a741e010fe9b.jpg)

![e0cedb6ce00b638d3a01802bc55b133bf4d98b656b7f9936a1c9321f1f0fc2be.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/e0cedb6ce00b638d3a01802bc55b133bf4d98b656b7f9936a1c9321f1f0fc2be.jpg)

![e76492cfb91f2e8c3916724ce94a816a652708ca4b4ebe87c47c049b93a2c5cc.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/e76492cfb91f2e8c3916724ce94a816a652708ca4b4ebe87c47c049b93a2c5cc.jpg)

![fddc7c01240cc51eeac244023a9aae1a2aea112abe13ea801035262a3e0ec913.jpg](../icml_results/1527_ExpWeaver_%20LLM%20Agents%20Learn%20from%20Experience%20via%20Latent%20RAG/tables/fddc7c01240cc51eeac244023a9aae1a2aea112abe13ea801035262a3e0ec913.jpg)

<a id="paper-1507"></a>

## CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective


### Images

![4921aae7f84ce4baf71e81b729938b1393cbfb330aceec8bb7cb4c4389c6f222.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/4921aae7f84ce4baf71e81b729938b1393cbfb330aceec8bb7cb4c4389c6f222.jpg)

![570be367d862f6fd248ad285d9fc8fc92d2b19b59ab30c554b261e8477670869.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/570be367d862f6fd248ad285d9fc8fc92d2b19b59ab30c554b261e8477670869.jpg)

![60b7e3ad46cc27f0f255dca2de8c1a22ae913110b6ee9435ed0c3f8de6e229c2.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/60b7e3ad46cc27f0f255dca2de8c1a22ae913110b6ee9435ed0c3f8de6e229c2.jpg)

![71175626bf31b7d6e0bb323ae548fa03ec07e8d3370de83c37b6689791ecc7a1.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/71175626bf31b7d6e0bb323ae548fa03ec07e8d3370de83c37b6689791ecc7a1.jpg)

![9134061cfccfb4304489419986fe694e823a4d4222d048419e27d33f4adff08b.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/9134061cfccfb4304489419986fe694e823a4d4222d048419e27d33f4adff08b.jpg)

![c9904aa16ab3f815ea49294d85e8c3801937c4036f451f69d4fb5fb3d3e61224.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/c9904aa16ab3f815ea49294d85e8c3801937c4036f451f69d4fb5fb3d3e61224.jpg)

![cdeccf3889aaaa276a9200c10092396894fa04092262add367d1cd76275d790a.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/cdeccf3889aaaa276a9200c10092396894fa04092262add367d1cd76275d790a.jpg)

![d9c9d5e493cfb54e163a35c0e567f411f90f08df5726197ceda795896fcf04b6.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/d9c9d5e493cfb54e163a35c0e567f411f90f08df5726197ceda795896fcf04b6.jpg)

![ddc269f72e641a112e75190f492d4dbecd2b97efa93819ae2df9a9eb8165a836.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/images/ddc269f72e641a112e75190f492d4dbecd2b97efa93819ae2df9a9eb8165a836.jpg)

### Tables

![090cb4dc6b813d1ed250fad3dd5c6e7f0f4fa6adcb34562df23949138f51ab24.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/090cb4dc6b813d1ed250fad3dd5c6e7f0f4fa6adcb34562df23949138f51ab24.jpg)

![10150ec251fbfe06eec84aec3bcf7813b863795247caa1d3ae9d46e38893e45a.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/10150ec251fbfe06eec84aec3bcf7813b863795247caa1d3ae9d46e38893e45a.jpg)

![21eb0f506996a61235816c15957a4cea130d5d071c527911b0eb0297ff0d3c33.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/21eb0f506996a61235816c15957a4cea130d5d071c527911b0eb0297ff0d3c33.jpg)

![2236a0985fae42da2b700de0786089592580379f0397911cd5d173b100007584.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/2236a0985fae42da2b700de0786089592580379f0397911cd5d173b100007584.jpg)

![25947ffbf6cb5cd60b13b8d9c930d1b6436d875e792c96536bd8b49cf8d25c24.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/25947ffbf6cb5cd60b13b8d9c930d1b6436d875e792c96536bd8b49cf8d25c24.jpg)

![2ccaf450543058754b386944f0555e988c35a9125db9c76cd523afcb03132288.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/2ccaf450543058754b386944f0555e988c35a9125db9c76cd523afcb03132288.jpg)

![3dad8ad51dec2a94dc3baaf0a3227a34774d54afe232dfe8d58a9fd1b54615c5.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/3dad8ad51dec2a94dc3baaf0a3227a34774d54afe232dfe8d58a9fd1b54615c5.jpg)

![3feadc6198ffa70c527dfb73af0d9ec0c97dffdb5e29d9d67844ea28710241d1.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/3feadc6198ffa70c527dfb73af0d9ec0c97dffdb5e29d9d67844ea28710241d1.jpg)

![4994129b1c2a2dd5a214e824580d04b3c0c01142c1ce443cb7816cdf7663e7f0.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/4994129b1c2a2dd5a214e824580d04b3c0c01142c1ce443cb7816cdf7663e7f0.jpg)

![5744c1890ac8cd3835bc1489c661cd1babc94be722d900db0921f492120f072e.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/5744c1890ac8cd3835bc1489c661cd1babc94be722d900db0921f492120f072e.jpg)

![677a8684ff4eefc77eb5633ff89f6d7dcb2d9bf1b3b2f6dd01cc8608a8866751.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/677a8684ff4eefc77eb5633ff89f6d7dcb2d9bf1b3b2f6dd01cc8608a8866751.jpg)

![85f8023707aa664ab33945a4a3e3b14a13be4b6b08b6153372de560daa61d8a2.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/85f8023707aa664ab33945a4a3e3b14a13be4b6b08b6153372de560daa61d8a2.jpg)

![8d43c94e39f383cc0d385d73edb8bc2812850937ef8a44988800d2c36b3a390c.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/8d43c94e39f383cc0d385d73edb8bc2812850937ef8a44988800d2c36b3a390c.jpg)

![903f6ff9e7250a5a9cc7960c7ab225bcb30a88d84e365ad486a7c841a9d7e46e.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/903f6ff9e7250a5a9cc7960c7ab225bcb30a88d84e365ad486a7c841a9d7e46e.jpg)

![9bb21ced707a76568015314b8c2b660b781f3d1b98b48335f0665306793e48ad.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/9bb21ced707a76568015314b8c2b660b781f3d1b98b48335f0665306793e48ad.jpg)

![aed009344cfa75e57fcde721fd09cc95840babe162ce1b41e20789f2ad8d57f7.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/aed009344cfa75e57fcde721fd09cc95840babe162ce1b41e20789f2ad8d57f7.jpg)

![c5eed45a7574dcd5979f12d2ce7946196f26dfeffc1bbf4323cd853f2e7ae7ef.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/c5eed45a7574dcd5979f12d2ce7946196f26dfeffc1bbf4323cd853f2e7ae7ef.jpg)

![c83e0f41481d5d4732e4c60b7cdc9a0701c3b3b38a847387ec51e25636b69fd0.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/c83e0f41481d5d4732e4c60b7cdc9a0701c3b3b38a847387ec51e25636b69fd0.jpg)

![dc2d09b10e6202b09b8482ef8a71e8771a44eab31c39e8d2026b8fdcf30f323c.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/dc2d09b10e6202b09b8482ef8a71e8771a44eab31c39e8d2026b8fdcf30f323c.jpg)

![e177a3a2e197ea6614e221b19126292bfd948759b7183bca68028db3bf23d287.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/e177a3a2e197ea6614e221b19126292bfd948759b7183bca68028db3bf23d287.jpg)

![f9608e4fa593956f5d963bfde467843048bcaf0dddff8bd094acc6cdd614e6ab.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/f9608e4fa593956f5d963bfde467843048bcaf0dddff8bd094acc6cdd614e6ab.jpg)

![fd7d05a2e2429f34ea010c4f33a8167f66e6f6f3f2f62980d5c6d998fea1a9b6.jpg](../icml_results/1528_CriticalKV_%20Optimizing%20KV%20Cache%20Eviction%20from%20an%20Output%20Perturbation%20Perspective/tables/fd7d05a2e2429f34ea010c4f33a8167f66e6f6f3f2f62980d5c6d998fea1a9b6.jpg)

<a id="paper-1508"></a>

## Understanding Dynamics of Adam in Zero-Sum Games: An ODE Approach


### Images

![1babf2570cffba9ce9e7dc0b39db4be70e927ba9db82b75d6e5190c287aa1289.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/1babf2570cffba9ce9e7dc0b39db4be70e927ba9db82b75d6e5190c287aa1289.jpg)

![1d061c9f04bbab07d76a942e3f02018d35a44e2271b921bc826af70ed7c03a7b.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/1d061c9f04bbab07d76a942e3f02018d35a44e2271b921bc826af70ed7c03a7b.jpg)

![25e5eb4b81eb9e28c3023246c5b922c485451d0bab1dff8eff57774eecbdc01f.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/25e5eb4b81eb9e28c3023246c5b922c485451d0bab1dff8eff57774eecbdc01f.jpg)

![312919ce49a6309eb371257972fb6d520fb22f0815b7fa74f38c59fb0a2f6be0.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/312919ce49a6309eb371257972fb6d520fb22f0815b7fa74f38c59fb0a2f6be0.jpg)

![41ac3c482724dc5f59a9ab2b4085bb49f423cdd2200ea71d9d2dee4c201b91f0.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/41ac3c482724dc5f59a9ab2b4085bb49f423cdd2200ea71d9d2dee4c201b91f0.jpg)

![5277dcf6862baafa68e3b71be33d88101dd0b5a3318032a840cb5d00c51d7c6b.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/5277dcf6862baafa68e3b71be33d88101dd0b5a3318032a840cb5d00c51d7c6b.jpg)

![59153982799ff8d9c7f512fdfa50b78ca1bb2694d03ef899574e423a0a9a6d34.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/59153982799ff8d9c7f512fdfa50b78ca1bb2694d03ef899574e423a0a9a6d34.jpg)

![ac68b26c948d7db7ca7ac33b38a5fc3b95cca2c636b4ad81a0a12631d9ba8400.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/ac68b26c948d7db7ca7ac33b38a5fc3b95cca2c636b4ad81a0a12631d9ba8400.jpg)

![bc58ff48747fe9690c1e8381e0fdc9345aff96bbb91d0f1836b2811233c4b49d.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/bc58ff48747fe9690c1e8381e0fdc9345aff96bbb91d0f1836b2811233c4b49d.jpg)

![cac3053f01efed0e9070b62baa168f6f9b138222fb21fac719a589542feeff34.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/cac3053f01efed0e9070b62baa168f6f9b138222fb21fac719a589542feeff34.jpg)

![cb6f4f4e48157aaca466ca301c52d004e8f66e3b580ffd19b640a5b23f302b9c.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/cb6f4f4e48157aaca466ca301c52d004e8f66e3b580ffd19b640a5b23f302b9c.jpg)

![d8ea2bef9056a8448ec53be4b3753cefe416336905fac7965f3689df57e6a914.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/d8ea2bef9056a8448ec53be4b3753cefe416336905fac7965f3689df57e6a914.jpg)

![e0a53edf76656ab163d290d76f463847fd15c9c6584b8e301c62b19664d89a53.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/e0a53edf76656ab163d290d76f463847fd15c9c6584b8e301c62b19664d89a53.jpg)

![f8d5895acaf9956fc3ef980592344b9e9a1dee455c1c89c4c350ed7cd2e4a4ae.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/images/f8d5895acaf9956fc3ef980592344b9e9a1dee455c1c89c4c350ed7cd2e4a4ae.jpg)

### Tables

![36e1209ffc320395fd8dc31cc70d8312e1d1f59956bce15f6fec992e54ae99a7.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/tables/36e1209ffc320395fd8dc31cc70d8312e1d1f59956bce15f6fec992e54ae99a7.jpg)

![b0ea15c72f95c4f75a66c00803b0fdc476850aab86ae87451192b456a95ed57e.jpg](../icml_results/1529_Understanding%20Dynamics%20of%20Adam%20in%20Zero-Sum%20Games_%20An%20ODE%20Approach/tables/b0ea15c72f95c4f75a66c00803b0fdc476850aab86ae87451192b456a95ed57e.jpg)

<a id="paper-1509"></a>

## Convex Optimization for Alignment and Preference Learning on a Single GPU


### Images

![2e62ab81b2b75ff6bddd14bea97ee7645f4dd08ba1c4b025431f0ae55615c795.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/images/2e62ab81b2b75ff6bddd14bea97ee7645f4dd08ba1c4b025431f0ae55615c795.jpg)

![46a0297ed343af12ac9dad813d3ae7b591592cd6a30a963a4518d99f4af70e8f.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/images/46a0297ed343af12ac9dad813d3ae7b591592cd6a30a963a4518d99f4af70e8f.jpg)

![5510bba31f523fa38246fe2213d1ed9b073cbaf947114cdc520faec2e6fcc5d8.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/images/5510bba31f523fa38246fe2213d1ed9b073cbaf947114cdc520faec2e6fcc5d8.jpg)

![68a9590167ddd0bedd80b09ddbec095984c6ce5d687dfbf43948cd979c9507f7.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/images/68a9590167ddd0bedd80b09ddbec095984c6ce5d687dfbf43948cd979c9507f7.jpg)

![dca140e3345693512fa125ee37409f0ad00b6b86e6270804af260cb6ef76a902.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/images/dca140e3345693512fa125ee37409f0ad00b6b86e6270804af260cb6ef76a902.jpg)

### Tables

![0c7ed764158784bd8d7acf759fd9347639805b18829206ce2c161bda9d94601c.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/0c7ed764158784bd8d7acf759fd9347639805b18829206ce2c161bda9d94601c.jpg)

![32df1fb07d10e3a5901222c16823521b562c72e8eaff9c3d9a123c8691020b0a.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/32df1fb07d10e3a5901222c16823521b562c72e8eaff9c3d9a123c8691020b0a.jpg)

![3e6bcff3e8e1d5233dd175cc6b4feabde38ba198ec87976d427c2171a930750f.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/3e6bcff3e8e1d5233dd175cc6b4feabde38ba198ec87976d427c2171a930750f.jpg)

![3fd431109e5cf9ecec5129ec7a3d278724e72ecc97cc3fdaa196bb5587edd3c2.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/3fd431109e5cf9ecec5129ec7a3d278724e72ecc97cc3fdaa196bb5587edd3c2.jpg)

![5c1fba882e3baaa73878f5b5ead468959d8a6685ed6c480dd83183107530ddc2.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/5c1fba882e3baaa73878f5b5ead468959d8a6685ed6c480dd83183107530ddc2.jpg)

![645eeec4f1869483668de756a396f670c6a9c6c57f948aeffe1e8fb1bf260df7.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/645eeec4f1869483668de756a396f670c6a9c6c57f948aeffe1e8fb1bf260df7.jpg)

![85eb2fdfbb56a1fdd3e4281baf141449addd45d5ecf1893ecf939e29d1511846.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/85eb2fdfbb56a1fdd3e4281baf141449addd45d5ecf1893ecf939e29d1511846.jpg)

![9bffae5da75292c2deab20042f3d70106132ab8fde6ddb155299febca2bb9f22.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/9bffae5da75292c2deab20042f3d70106132ab8fde6ddb155299febca2bb9f22.jpg)

![b30a0726d15e3327b4b44341b05ebde87103476b6132e6741ce987074c4be92e.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/b30a0726d15e3327b4b44341b05ebde87103476b6132e6741ce987074c4be92e.jpg)

![c7ecee0a5e2b90654f59c3bb47f732feba6cca7b97c88dfe2c1d10f2596aa000.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/c7ecee0a5e2b90654f59c3bb47f732feba6cca7b97c88dfe2c1d10f2596aa000.jpg)

![e3d643b6cf1fc952cf1759db94f7b18c06dd1b66ad2a08202441f3739e1973a6.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/e3d643b6cf1fc952cf1759db94f7b18c06dd1b66ad2a08202441f3739e1973a6.jpg)

![ec9d115de44fb91b51dc0d0d0dba50e806f4231db097eab92f18e6b58c22e7a1.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/ec9d115de44fb91b51dc0d0d0dba50e806f4231db097eab92f18e6b58c22e7a1.jpg)

![eed7e53c4f75ab0bd0c1d4c955f6f53fc62b5a5b630a2db7ea20e67a97ca6d07.jpg](../icml_results/1530_Convex%20Optimization%20for%20Alignment%20and%20Preference%20Learning%20on%20a%20Single%20GPU/tables/eed7e53c4f75ab0bd0c1d4c955f6f53fc62b5a5b630a2db7ea20e67a97ca6d07.jpg)

<a id="paper-1510"></a>

## Knowledge Diversion for Efficient Morphology Control and Policy Transfer


### Images

![21ec8daede53449b0f1b25fd5a8ed7bebdbfa6697e061a83c420769ff2623329.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/21ec8daede53449b0f1b25fd5a8ed7bebdbfa6697e061a83c420769ff2623329.jpg)

![3f5ef092a5c92a312f1b9ea9ee7a1b9ce7ab2026cdbb461f07aeb6f39f753f58.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/3f5ef092a5c92a312f1b9ea9ee7a1b9ce7ab2026cdbb461f07aeb6f39f753f58.jpg)

![4aad19a382968dc253fa6044564d09a2c25422213c2c5d4b83fc5be298024632.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/4aad19a382968dc253fa6044564d09a2c25422213c2c5d4b83fc5be298024632.jpg)

![6d7687a8e9334034faabe2f2d877cbd40066220a4aee89a1134128657c7e6ab8.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/6d7687a8e9334034faabe2f2d877cbd40066220a4aee89a1134128657c7e6ab8.jpg)

![822eeff2c7af9bf3bf05435620599508c7009364aa678372495d51e40217b812.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/822eeff2c7af9bf3bf05435620599508c7009364aa678372495d51e40217b812.jpg)

![8edfc1fc9315333031619e1061f06474258402c81a81e5399114377ef05a872e.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/8edfc1fc9315333031619e1061f06474258402c81a81e5399114377ef05a872e.jpg)

![cba58c0710414a2c225620fb2292eed918f4c7a1557c477a32184bfe6c62200c.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/cba58c0710414a2c225620fb2292eed918f4c7a1557c477a32184bfe6c62200c.jpg)

![e41d4c5d07643c21c001543d4541d2366f427a5dc363b1ec0c9cfce7256ef6d6.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/images/e41d4c5d07643c21c001543d4541d2366f427a5dc363b1ec0c9cfce7256ef6d6.jpg)

### Tables

![db72326e2ddab1c4a8136785acb8b72ac57618377f46e11afa6409f8956abdbb.jpg](../icml_results/1531_Knowledge%20Diversion%20for%20Efficient%20Morphology%20Control%20and%20Policy%20Transfer/tables/db72326e2ddab1c4a8136785acb8b72ac57618377f46e11afa6409f8956abdbb.jpg)

<a id="paper-1511"></a>

## Unsupervised Diffusion Solver for Combinatorial Optimization via Combinatorial Adjoint Matching


### Images

![03dae6d40883926a7d2aa66002a6d330c746c0a2797c395ffdc99f356f31af64.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/images/03dae6d40883926a7d2aa66002a6d330c746c0a2797c395ffdc99f356f31af64.jpg)

![1346070512406bdab49f60ff2a7efc02a14a8d47024f5accbbd3493adcd740b7.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/images/1346070512406bdab49f60ff2a7efc02a14a8d47024f5accbbd3493adcd740b7.jpg)

### Tables

![17f5447818688e6a48118811018e14463e718e27febc3472333ef71be0e11803.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/tables/17f5447818688e6a48118811018e14463e718e27febc3472333ef71be0e11803.jpg)

![61602b4f8e8051118c395608073c1d6172b0b1acf50bb7ee8ada7cb3e062cb3a.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/tables/61602b4f8e8051118c395608073c1d6172b0b1acf50bb7ee8ada7cb3e062cb3a.jpg)

![944fb2e71ab3112b913e6cc636c90f431d5c07ed1cc758064c8b1e4967af95f2.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/tables/944fb2e71ab3112b913e6cc636c90f431d5c07ed1cc758064c8b1e4967af95f2.jpg)

![bd0b8405a31553bb8872c52d3634f005a0e6bb639a7aaed3ef99681ffd336211.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/tables/bd0b8405a31553bb8872c52d3634f005a0e6bb639a7aaed3ef99681ffd336211.jpg)

![ecc11c7ee47d6debb143c7e1d4d1349f74c42757cc79b00fd5dd29fe40180c6e.jpg](../icml_results/1532_Unsupervised%20Diffusion%20Solver%20for%20Combinatorial%20Optimization%20via%20Combinatorial%20Adjoint%20Matching/tables/ecc11c7ee47d6debb143c7e1d4d1349f74c42757cc79b00fd5dd29fe40180c6e.jpg)

<a id="paper-1512"></a>

## Convex Low-resource Accent-Robust Language Detection in Speech Recognition


### Images

![0a6a5dfa274fe3511de68b5b6eb60882357a6fef1febac2cb7733ca30299b227.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/images/0a6a5dfa274fe3511de68b5b6eb60882357a6fef1febac2cb7733ca30299b227.jpg)

![1ad830ce042aacaf0ab0594fd818597652ffbe2d4b76f0ebe344d0a86f182c87.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/images/1ad830ce042aacaf0ab0594fd818597652ffbe2d4b76f0ebe344d0a86f182c87.jpg)

![3368bf8eae1540bdb60f2f4f95b2df815b3af7596117ab14b1668e4b83bac848.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/images/3368bf8eae1540bdb60f2f4f95b2df815b3af7596117ab14b1668e4b83bac848.jpg)

![57d7c01531e4f05f7e5c78933263eb066a461ee739de66af5d4b0939532199b2.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/images/57d7c01531e4f05f7e5c78933263eb066a461ee739de66af5d4b0939532199b2.jpg)

![7261f3e7b6ac2173d0636beb674d33c402c0ee0f0a89494430cf5940c96fe984.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/images/7261f3e7b6ac2173d0636beb674d33c402c0ee0f0a89494430cf5940c96fe984.jpg)

### Tables

![22e81a2fb46f316952a4cfba05adee7ff2eab5569bbe4cf332cda37c089ffc93.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/22e81a2fb46f316952a4cfba05adee7ff2eab5569bbe4cf332cda37c089ffc93.jpg)

![55def7a74483ed88d75d9186086e1b115317dc210402af0a9bd6673244c3ee23.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/55def7a74483ed88d75d9186086e1b115317dc210402af0a9bd6673244c3ee23.jpg)

![727f09ea71df295a7d51567d9639e5ebafa5df84cc00f5780b68417f11118fb6.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/727f09ea71df295a7d51567d9639e5ebafa5df84cc00f5780b68417f11118fb6.jpg)

![7c81f3c4146c7b2c375f61a301feec093710daf596b32547cf3aa775f479d37f.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/7c81f3c4146c7b2c375f61a301feec093710daf596b32547cf3aa775f479d37f.jpg)

![92cca333fd9bac03f1459980853d39815c3592f38002aa28f0bd8ef9b0b4d2d1.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/92cca333fd9bac03f1459980853d39815c3592f38002aa28f0bd8ef9b0b4d2d1.jpg)

![93439b5f33f6e61e8b8f03cae1069a3070ee422a24a7ff78f679ae76a0f5bc62.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/93439b5f33f6e61e8b8f03cae1069a3070ee422a24a7ff78f679ae76a0f5bc62.jpg)

![96ab808558650ee0090fb12092d3f551344e486329153413d44b1b92c8a081e0.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/96ab808558650ee0090fb12092d3f551344e486329153413d44b1b92c8a081e0.jpg)

![a0d507478254dd3be22c21d5180812577a0f3b70c168926ad3682ef8f4422951.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/a0d507478254dd3be22c21d5180812577a0f3b70c168926ad3682ef8f4422951.jpg)

![caf539c8f16fafde8b6cb3bc92cff85d4554af2ecc4741c6297372a5f942a01b.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/caf539c8f16fafde8b6cb3bc92cff85d4554af2ecc4741c6297372a5f942a01b.jpg)

![e25b65a88fb7348c41ceeaf6a6c20488a74b8d5beeaa9585df949b56ae4f3031.jpg](../icml_results/1533_Convex%20Low-resource%20Accent-Robust%20Language%20Detection%20in%20Speech%20Recognition/tables/e25b65a88fb7348c41ceeaf6a6c20488a74b8d5beeaa9585df949b56ae4f3031.jpg)

<a id="paper-1513"></a>

## Offline Reinforcement Learning with Generative Trajectory Policies


### Images

![391ceebb6441449be04e540be57948cf519ec51642ab2405d83640f505e6f233.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/391ceebb6441449be04e540be57948cf519ec51642ab2405d83640f505e6f233.jpg)

![4d13107ed3b2cf8e716052bb2822f290c4763c4da510009efc123dc5fb3d9973.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/4d13107ed3b2cf8e716052bb2822f290c4763c4da510009efc123dc5fb3d9973.jpg)

![7695a9825fcccefd56bf38170fbe2bd27487135cd6fcc52d9b96feb28ed09d03.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/7695a9825fcccefd56bf38170fbe2bd27487135cd6fcc52d9b96feb28ed09d03.jpg)

![8b3f3b59b9297d48348bc8fe8c9ce10df905f8a6ac37e0d1023471fda404da5a.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/8b3f3b59b9297d48348bc8fe8c9ce10df905f8a6ac37e0d1023471fda404da5a.jpg)

![98ce09d1921ca1dc90a33fd3b33a11e55ea9920202aa626406ec745006d4704c.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/98ce09d1921ca1dc90a33fd3b33a11e55ea9920202aa626406ec745006d4704c.jpg)

![997b51b2f59795be0367acd597b662605258286d6fc6608d23a15dff4bec52e2.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/997b51b2f59795be0367acd597b662605258286d6fc6608d23a15dff4bec52e2.jpg)

![9e756ce02a4fa296e5278f2c5945ba0b1a17f13cc1733dc62dca9c6cd040a1a3.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/9e756ce02a4fa296e5278f2c5945ba0b1a17f13cc1733dc62dca9c6cd040a1a3.jpg)

![e52489db17cc6246132af3b737ddf9d52593cb8378f3cf0f51791a649509c3b3.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/e52489db17cc6246132af3b737ddf9d52593cb8378f3cf0f51791a649509c3b3.jpg)

![e64599a270b01cad61db4c0b2d6f5415491d7abda9037a8152f2c38ffb0c2043.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/images/e64599a270b01cad61db4c0b2d6f5415491d7abda9037a8152f2c38ffb0c2043.jpg)

### Tables

![23b48cc9f0b01176b46d347da5703c35922fe510def12dba99d14a2b9894106b.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/23b48cc9f0b01176b46d347da5703c35922fe510def12dba99d14a2b9894106b.jpg)

![3d20027f6552ed8a2a2b4c22f952eb0717b41ea500faad6984c8b57dcd25d8ac.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/3d20027f6552ed8a2a2b4c22f952eb0717b41ea500faad6984c8b57dcd25d8ac.jpg)

![40bda0795a5dda2b63b981f1aa63231a36c53da8c0a704922479f6b9699c4301.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/40bda0795a5dda2b63b981f1aa63231a36c53da8c0a704922479f6b9699c4301.jpg)

![9563ed8deabc6e00ad67ea48bd810c431a7a5c68971cdd19da15b32bc65a2a3c.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/9563ed8deabc6e00ad67ea48bd810c431a7a5c68971cdd19da15b32bc65a2a3c.jpg)

![ad2f84aad68f9cba2d4cbe34cca54d77ad498a559278386b0ef731ccc9ffb5f1.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/ad2f84aad68f9cba2d4cbe34cca54d77ad498a559278386b0ef731ccc9ffb5f1.jpg)

![af55780fc0e6e32718b2789c1c3893960532832b0f5cf28258e718c3231e2ec1.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/af55780fc0e6e32718b2789c1c3893960532832b0f5cf28258e718c3231e2ec1.jpg)

![bc99c856a42e54886ec027d1e02d56a9da212b30ac1ab3edf708b1ee8724a028.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/bc99c856a42e54886ec027d1e02d56a9da212b30ac1ab3edf708b1ee8724a028.jpg)

![d78d5092c6b35fc8d931f2d2185cd80e9e99d24d7690503de6d6285e6349f87d.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/d78d5092c6b35fc8d931f2d2185cd80e9e99d24d7690503de6d6285e6349f87d.jpg)

![da29f3ea2a1ae73dd9108464431c7523e6c1e334ec8de83159445b8462d31af8.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/da29f3ea2a1ae73dd9108464431c7523e6c1e334ec8de83159445b8462d31af8.jpg)

![fe9f2bbe3991d20f65ca0caa630f824acc2260a86237975a5518cd578be2ff02.jpg](../icml_results/1534_Offline%20Reinforcement%20Learning%20with%20Generative%20Trajectory%20Policies/tables/fe9f2bbe3991d20f65ca0caa630f824acc2260a86237975a5518cd578be2ff02.jpg)

<a id="paper-1514"></a>

## HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning


### Images

![398c2f80d1c08674e25b24b91cac9f2d551eb448ad53496518b6e89328cdcf73.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/398c2f80d1c08674e25b24b91cac9f2d551eb448ad53496518b6e89328cdcf73.jpg)

![3da5a66ad0f26c3a3366272bcd759e3b82438c824cb22449d5325966ebe07b00.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/3da5a66ad0f26c3a3366272bcd759e3b82438c824cb22449d5325966ebe07b00.jpg)

![788181296b236f6b61c0797baef3b578333e6a68b305bf4053965f4c5fd95887.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/788181296b236f6b61c0797baef3b578333e6a68b305bf4053965f4c5fd95887.jpg)

![896f9b96af1985a31c534c1d920d5b344b3f77de6132fd93ec7f5490bd8906ec.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/896f9b96af1985a31c534c1d920d5b344b3f77de6132fd93ec7f5490bd8906ec.jpg)

![a96dff57b2049b69335458fc7efcfa38930af71ebb401844699e966f26caf99f.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/a96dff57b2049b69335458fc7efcfa38930af71ebb401844699e966f26caf99f.jpg)

![c8d144fa1715b8a3d6990bb02989d5d12296ab66a652bc7ae50a289f8e4e60e1.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/images/c8d144fa1715b8a3d6990bb02989d5d12296ab66a652bc7ae50a289f8e4e60e1.jpg)

### Tables

![0f8a8775c676692b0c1a69a2418205c7f2b9d3183ca6ee17ba7483cf9b65f73a.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/tables/0f8a8775c676692b0c1a69a2418205c7f2b9d3183ca6ee17ba7483cf9b65f73a.jpg)

![1dd4481f363dfc1e7b084e95ffa568867ea946227dfeba577e74ede044e0c31c.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/tables/1dd4481f363dfc1e7b084e95ffa568867ea946227dfeba577e74ede044e0c31c.jpg)

![6ddcad02022b6b4700db828e1c1a169e540d00c9eef22c1b6e7c0cf3173ac1da.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/tables/6ddcad02022b6b4700db828e1c1a169e540d00c9eef22c1b6e7c0cf3173ac1da.jpg)

![9338b5b89dfb7904db54e08b66c4f5b20b0395d6a2cb2f860ffcf39924adc8c3.jpg](../icml_results/1535_HEDP_%20A%20Hybrid%20Energy-Distance%20Prompt-based%20Framework%20for%20Domain%20Incremental%20Learning/tables/9338b5b89dfb7904db54e08b66c4f5b20b0395d6a2cb2f860ffcf39924adc8c3.jpg)

<a id="paper-1515"></a>

## TelecomTS: A Multi-Modal Observability Dataset for Time Series and Language Analysis


### Images

![0b838a56de46b51ade748ae82c6ad96472684b2dbe5a24b06f97a321f0dcb6fc.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/0b838a56de46b51ade748ae82c6ad96472684b2dbe5a24b06f97a321f0dcb6fc.jpg)

![0d154ee5a92bc537fe4eb87ccd1fd84e7f2212601accb6bf50b5ce3cf00dbb69.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/0d154ee5a92bc537fe4eb87ccd1fd84e7f2212601accb6bf50b5ce3cf00dbb69.jpg)

![0e58fe102d02a8793767ca59e6a73d9c35162ef6a7a75076058bf8931024a60f.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/0e58fe102d02a8793767ca59e6a73d9c35162ef6a7a75076058bf8931024a60f.jpg)

![115955fab5c916ad5b6d124bde557e765fccab0c6f51e6b0b1e2ec006b2917bf.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/115955fab5c916ad5b6d124bde557e765fccab0c6f51e6b0b1e2ec006b2917bf.jpg)

![3328ae3e4e7f69d49cb6abcc73a52d93a7dbb7c331d6a16e012a3e9294e6db18.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/3328ae3e4e7f69d49cb6abcc73a52d93a7dbb7c331d6a16e012a3e9294e6db18.jpg)

![49956af8e4d90700d5b02478d60f8c758746a383dfe92b37e1020b1bef3f0976.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/49956af8e4d90700d5b02478d60f8c758746a383dfe92b37e1020b1bef3f0976.jpg)

![4f69fa79b864d60e4cd4cd255a2b93d500a35653a911822ae1c6a6cb18522d31.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/4f69fa79b864d60e4cd4cd255a2b93d500a35653a911822ae1c6a6cb18522d31.jpg)

![672171679e9a0855f1274ebef2265ebe2cdb077fd3c53ec293271cea54ded53a.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/672171679e9a0855f1274ebef2265ebe2cdb077fd3c53ec293271cea54ded53a.jpg)

![6824b54d361a974227e2d6b001ec51b311e9deb3e2718511f6e3cd1668436f8f.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/6824b54d361a974227e2d6b001ec51b311e9deb3e2718511f6e3cd1668436f8f.jpg)

![78b4d317288e27cc546203be7cba50895f9096e08897772831f671509137cb62.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/78b4d317288e27cc546203be7cba50895f9096e08897772831f671509137cb62.jpg)

![82b70708ce66ae69e4bad113631ca78a963e62071cb97e69e99275181f157322.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/82b70708ce66ae69e4bad113631ca78a963e62071cb97e69e99275181f157322.jpg)

![977daa8bc88379f3263521b8786503202df452d287278790a77e321effe82326.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/977daa8bc88379f3263521b8786503202df452d287278790a77e321effe82326.jpg)

![b34a479d9d9a56a53e1cfafb573898b617c682c1a0b6e8de3d8ad56b658ee031.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/b34a479d9d9a56a53e1cfafb573898b617c682c1a0b6e8de3d8ad56b658ee031.jpg)

![be11070d9f6190e1d821bb805a058237a9f092914a4e079de6f87f0c0b5a15ff.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/be11070d9f6190e1d821bb805a058237a9f092914a4e079de6f87f0c0b5a15ff.jpg)

![c2ddb4a19081d8167d6d0fbd8ce7d1cd8d16d9e2cbe5bc2103456599fb6a5bbe.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/c2ddb4a19081d8167d6d0fbd8ce7d1cd8d16d9e2cbe5bc2103456599fb6a5bbe.jpg)

![d41cc6359a1aa32af9cb11865ba1ce89f6e0a02221225b901a2f6036376ea76a.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/d41cc6359a1aa32af9cb11865ba1ce89f6e0a02221225b901a2f6036376ea76a.jpg)

![ea0ae433e61fa24068eb915cc0105b2173be4c3c6ab1f63c7ceb17345ebd18ed.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/ea0ae433e61fa24068eb915cc0105b2173be4c3c6ab1f63c7ceb17345ebd18ed.jpg)

![ff40938cb15500c52f8bef595d23b5b8fcf36525020c728676da8c7c83db65b6.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/images/ff40938cb15500c52f8bef595d23b5b8fcf36525020c728676da8c7c83db65b6.jpg)

### Tables

![032a9dba5aa3315f64cf16814e66c43c36ee450cb1159301271d131d41549d7f.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/032a9dba5aa3315f64cf16814e66c43c36ee450cb1159301271d131d41549d7f.jpg)

![15ef8cbea63c70426f36bfd98722c6b9d10cbda56b335b39c54fd3a1083c1182.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/15ef8cbea63c70426f36bfd98722c6b9d10cbda56b335b39c54fd3a1083c1182.jpg)

![24a9cd363aab696f4530e51c88f5a5c1cf2c2fe32842c9f90c025d394ba5ae69.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/24a9cd363aab696f4530e51c88f5a5c1cf2c2fe32842c9f90c025d394ba5ae69.jpg)

![6a6112b932727f52e92c7f5eb6bed3d6ce05da5c57245d99571a6b0b919d0b4f.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/6a6112b932727f52e92c7f5eb6bed3d6ce05da5c57245d99571a6b0b919d0b4f.jpg)

![73392ca10309302d75c119848fd2c02ef1c5ec5a79f052b1e118c2bbef2eaeda.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/73392ca10309302d75c119848fd2c02ef1c5ec5a79f052b1e118c2bbef2eaeda.jpg)

![7957db2d330828290a08bca203baae37b2f9d2bd46fe4a6ecdef50021438b2d8.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/7957db2d330828290a08bca203baae37b2f9d2bd46fe4a6ecdef50021438b2d8.jpg)

![94e1761cd4bf5f1c861a9da4dae70e58d59fea30f1f308f47731aa3e535197b3.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/94e1761cd4bf5f1c861a9da4dae70e58d59fea30f1f308f47731aa3e535197b3.jpg)

![a542ec84d254f8d42bf965ca315664a1f5b203c5d5df8cbe0e52bb3305a9d993.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/a542ec84d254f8d42bf965ca315664a1f5b203c5d5df8cbe0e52bb3305a9d993.jpg)

![a79df31a8068d6fa981a7297434078938949f4e586426162d5d811f28a66d5ba.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/a79df31a8068d6fa981a7297434078938949f4e586426162d5d811f28a66d5ba.jpg)

![d4cd305ee5d5d8b8c745bd11618975ddb414d93f26fd01626de83eb5c35a3cb8.jpg](../icml_results/1536_TelecomTS_%20A%20Multi-Modal%20Observability%20Dataset%20for%20Time%20Series%20and%20Language%20Analysis/tables/d4cd305ee5d5d8b8c745bd11618975ddb414d93f26fd01626de83eb5c35a3cb8.jpg)

<a id="paper-1516"></a>

## Fast-SAM3D: 3Dfy Anything in Images but Faster


### Images

![15d835baf1ac3c4e974d31dfd7f9ae5251084ab0999d76d35270092e4574477b.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/15d835baf1ac3c4e974d31dfd7f9ae5251084ab0999d76d35270092e4574477b.jpg)

![1e48dfecd0343d9204852d833ab1a44107a1f36bd2bf2bfb9de8312afce6365f.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/1e48dfecd0343d9204852d833ab1a44107a1f36bd2bf2bfb9de8312afce6365f.jpg)

![1f0852e18d77daaf8bffd4a98b8f22eb4d8d31ee23d2df370b6e9c24762e3b1d.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/1f0852e18d77daaf8bffd4a98b8f22eb4d8d31ee23d2df370b6e9c24762e3b1d.jpg)

![244b4688a44364845c85421e3df06f6252fd82385b8b264e646769d79fc053f9.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/244b4688a44364845c85421e3df06f6252fd82385b8b264e646769d79fc053f9.jpg)

![292760ef8a5bdda98714cab8bb8678b638e7468524ee6704322e94714211bb96.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/292760ef8a5bdda98714cab8bb8678b638e7468524ee6704322e94714211bb96.jpg)

![363c9a13fc3cb5d4631a1cb6505c7a6d71f35fb17cc1d510bba910a65daf7fc3.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/363c9a13fc3cb5d4631a1cb6505c7a6d71f35fb17cc1d510bba910a65daf7fc3.jpg)

![6c5a16b669851ba581fd53c903d6e310bf1ae8e1bb69b7c48659926e9d06f75a.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/6c5a16b669851ba581fd53c903d6e310bf1ae8e1bb69b7c48659926e9d06f75a.jpg)

![6fe3b8da2d6569247cb5551bfee328e4e2cbd809de5f92971f950fd5d2d4af94.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/6fe3b8da2d6569247cb5551bfee328e4e2cbd809de5f92971f950fd5d2d4af94.jpg)

![7b4331f1718e2f3bc2569652e0055e792058ae3faa5970273dccd19619563ecf.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/7b4331f1718e2f3bc2569652e0055e792058ae3faa5970273dccd19619563ecf.jpg)

![92053a6667b12b3630f191cc8d1bc02bc6965e1f6c3898804311a15604dcf9f0.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/92053a6667b12b3630f191cc8d1bc02bc6965e1f6c3898804311a15604dcf9f0.jpg)

![bc681d339fd6f2b350686da49815d4c4c20167618cea20c2a01168d4e2f43a1c.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/bc681d339fd6f2b350686da49815d4c4c20167618cea20c2a01168d4e2f43a1c.jpg)

![db761f45e3a26c8678408502598322a41d6bcbcb4b30d5d00a55c73f353a17ce.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/db761f45e3a26c8678408502598322a41d6bcbcb4b30d5d00a55c73f353a17ce.jpg)

![eadda7f84d0d3ccfdfadfe94e584901b04a588ca06257c9216a398c4ceae0948.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/images/eadda7f84d0d3ccfdfadfe94e584901b04a588ca06257c9216a398c4ceae0948.jpg)

### Tables

![2658c2b6e7ef0733f07511e3c9cad99802ac633987d4286ce81f610f173610cf.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/2658c2b6e7ef0733f07511e3c9cad99802ac633987d4286ce81f610f173610cf.jpg)

![2f93d4a1ed18096d31cb81d7c389b52a2fbd933559a131aa98afc96167e2903a.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/2f93d4a1ed18096d31cb81d7c389b52a2fbd933559a131aa98afc96167e2903a.jpg)

![311f32b3281331b3ec58f0e81562ccb5fd127604c36e9b8b9d9f0147929db286.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/311f32b3281331b3ec58f0e81562ccb5fd127604c36e9b8b9d9f0147929db286.jpg)

![3143b0dc4b8f3635ea6f0bcef1fd1220f8c2d421e0d21458d7295e5bc4c9e3b2.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/3143b0dc4b8f3635ea6f0bcef1fd1220f8c2d421e0d21458d7295e5bc4c9e3b2.jpg)

![778393353698a86c9e0ffe788ee472cb898efa54b3fa94a5909072443b8ac65e.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/778393353698a86c9e0ffe788ee472cb898efa54b3fa94a5909072443b8ac65e.jpg)

![7ece1a09fa30e82a50ddaa7344c75d114b1f5e32b2ee3b5d539fe87a347b5e0d.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/7ece1a09fa30e82a50ddaa7344c75d114b1f5e32b2ee3b5d539fe87a347b5e0d.jpg)

![8fca65e2badbf0b5f9bf4264403f6eb1ce2e5830deb18e535c9a891703617e19.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/8fca65e2badbf0b5f9bf4264403f6eb1ce2e5830deb18e535c9a891703617e19.jpg)

![ade4bf5dc78a255ae44ef2928137c3f7505c7971db36a4ac8962d0dcfd4d57e7.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/ade4bf5dc78a255ae44ef2928137c3f7505c7971db36a4ac8962d0dcfd4d57e7.jpg)

![b017cdd18120014d4246fbd7f81ab65c194f35e90d8bc8a8dcf048dffa7f0dfb.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/b017cdd18120014d4246fbd7f81ab65c194f35e90d8bc8a8dcf048dffa7f0dfb.jpg)

![b7528f36d023f460ff3b5762bd17869db6a7efed8f08497c83025c65b6815c5d.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/b7528f36d023f460ff3b5762bd17869db6a7efed8f08497c83025c65b6815c5d.jpg)

![c0381267b378fd8c32ff1665a27dfb03b394598c04f9d54e29fdf81b032fcbc6.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/c0381267b378fd8c32ff1665a27dfb03b394598c04f9d54e29fdf81b032fcbc6.jpg)

![cc2accd96b63e6f06c70ed6f07d265b1bde0f06c57bf9b2e1d3bda29d31cdee0.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/cc2accd96b63e6f06c70ed6f07d265b1bde0f06c57bf9b2e1d3bda29d31cdee0.jpg)

![f3ac951719eeaf4d4dc18b5dd935496dd68808f238ecc2a48e33a64cc07795c9.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/f3ac951719eeaf4d4dc18b5dd935496dd68808f238ecc2a48e33a64cc07795c9.jpg)

![f56c44bd731826b8070ef4aa8c1e4d40b8616c738746658779c4ac3742cf064e.jpg](../icml_results/1537_Fast-SAM3D_%203Dfy%20Anything%20in%20Images%20but%20Faster/tables/f56c44bd731826b8070ef4aa8c1e4d40b8616c738746658779c4ac3742cf064e.jpg)

<a id="paper-1517"></a>

## Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting


### Images

![1333ea8ff11c19f90128682ca836af1fcd66ed2df49a6c7e9e04aa91c5268a6e.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/1333ea8ff11c19f90128682ca836af1fcd66ed2df49a6c7e9e04aa91c5268a6e.jpg)

![77d3d4f6e96aff92afab4613143e546d3555c77eb84f1c79c22740211ea10dd5.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/77d3d4f6e96aff92afab4613143e546d3555c77eb84f1c79c22740211ea10dd5.jpg)

![8add880a803c62918f893eaaa3f64af47473897ff70ab7326e2fbc4ba7684630.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/8add880a803c62918f893eaaa3f64af47473897ff70ab7326e2fbc4ba7684630.jpg)

![92b6aedb5be04c1a0b9dcc9cb24d5a5e48dfb36d72fbc3cb13da9dbb93ffb10e.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/92b6aedb5be04c1a0b9dcc9cb24d5a5e48dfb36d72fbc3cb13da9dbb93ffb10e.jpg)

![97b01dda84cf9a86a3bf69c96a2bf56203d962f358598f12b35172db661da166.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/97b01dda84cf9a86a3bf69c96a2bf56203d962f358598f12b35172db661da166.jpg)

![ab72d30291602c5ed365f434f25423b4ca4db9867445dcdd1b59da50513146bc.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/ab72d30291602c5ed365f434f25423b4ca4db9867445dcdd1b59da50513146bc.jpg)

![d5ebcc445f23f463213395adb51e49580f34ddf2b45d84479444baca936e6201.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/images/d5ebcc445f23f463213395adb51e49580f34ddf2b45d84479444baca936e6201.jpg)

### Tables

![05cddbfcc4272995e0d72e565d53fbb8f8384962d01acc31cad7bd4996a76c48.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/05cddbfcc4272995e0d72e565d53fbb8f8384962d01acc31cad7bd4996a76c48.jpg)

![0f849385ddca54965ad8b0f87267437559f4c6fcdd856841ee35220df062833b.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/0f849385ddca54965ad8b0f87267437559f4c6fcdd856841ee35220df062833b.jpg)

![345df38a084f3d3fc9e60d093d8ea5c2a03db24c9a32f56b7a1d885384c604db.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/345df38a084f3d3fc9e60d093d8ea5c2a03db24c9a32f56b7a1d885384c604db.jpg)

![50f7f7c81cbf94f8b6c29722e87198a58b4ed219a8a853d9d091e6a4d2455004.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/50f7f7c81cbf94f8b6c29722e87198a58b4ed219a8a853d9d091e6a4d2455004.jpg)

![ad57518f038df7aacebbeb8b369a1ca40b11d76d67b465903457da7bcc3ab644.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/ad57518f038df7aacebbeb8b369a1ca40b11d76d67b465903457da7bcc3ab644.jpg)

![c195f07b9a92997a2cedbbfa97b74fd6ccb2dcce545bd95256c5d117bee29a45.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/c195f07b9a92997a2cedbbfa97b74fd6ccb2dcce545bd95256c5d117bee29a45.jpg)

![ebfc627527e43ce307bd7e88b475b2d0006031053a64ebb9df896b9069043182.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/ebfc627527e43ce307bd7e88b475b2d0006031053a64ebb9df896b9069043182.jpg)

![ee2fc8e3e368d81be5c67985d9cae4c2ec1e10f531d04096e1c9d60ac5d1f5f2.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/ee2fc8e3e368d81be5c67985d9cae4c2ec1e10f531d04096e1c9d60ac5d1f5f2.jpg)

![f2308a0136a877787dfacfd27e3fb8b46161027609924e8362a9dd290c9ec882.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/f2308a0136a877787dfacfd27e3fb8b46161027609924e8362a9dd290c9ec882.jpg)

![f5da63e74fcf6d745d78d9b748def3de13767dd47d7eb8626dac77e7f1d29d29.jpg](../icml_results/1538_Beyond%20Model%20Ranking_%20Predictability-Aligned%20Evaluation%20for%20Time%20Series%20Forecasting/tables/f5da63e74fcf6d745d78d9b748def3de13767dd47d7eb8626dac77e7f1d29d29.jpg)

<a id="paper-1518"></a>

## DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts

### Images

![239854c65d37d3af7c7049debfa43d0a9aeb2f94004bd9970075bb36c76d7ff6.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/239854c65d37d3af7c7049debfa43d0a9aeb2f94004bd9970075bb36c76d7ff6.jpg)

![36bd97427d87e8d4e332412d765e8684750be4a5208dd84770abf7977f0af105.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/36bd97427d87e8d4e332412d765e8684750be4a5208dd84770abf7977f0af105.jpg)

![59b2aa2946935837326857da2859201e582f29db7e5aca796aee767fe793c2ad.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/59b2aa2946935837326857da2859201e582f29db7e5aca796aee767fe793c2ad.jpg)

![6188b2d9dd9e1e71393ffadc5e554f5d6b7b285cb0da5edf6287b94fc983c9ef.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/6188b2d9dd9e1e71393ffadc5e554f5d6b7b285cb0da5edf6287b94fc983c9ef.jpg)

![63218f480d07b9027f18ad1216347388274c2fd1f3a1c82342d630840651411e.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/63218f480d07b9027f18ad1216347388274c2fd1f3a1c82342d630840651411e.jpg)

![aefead4b25640a8856a5cc877bc446dd93739d5a737a1badbdfc4b93e0c3d17c.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/aefead4b25640a8856a5cc877bc446dd93739d5a737a1badbdfc4b93e0c3d17c.jpg)

![f29f7c1b3eabfc0e22221f32403f99ed7b3c987fc84bfde73e9512290982f429.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/f29f7c1b3eabfc0e22221f32403f99ed7b3c987fc84bfde73e9512290982f429.jpg)

![ff7421c3e2f82a249a50d73d1e51b14b18efd44d59d8aaf59047d759c8fe34bf.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/images/ff7421c3e2f82a249a50d73d1e51b14b18efd44d59d8aaf59047d759c8fe34bf.jpg)

### Tables

![160d5e83dce15177eb871f5672280c3a6f072c5c3036d4f4ab6d336701b17f36.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/160d5e83dce15177eb871f5672280c3a6f072c5c3036d4f4ab6d336701b17f36.jpg)

![1e8127fc8a877c9547c3ecccd39d212c0c5b3ff7025e7c76566f8190210110b6.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/1e8127fc8a877c9547c3ecccd39d212c0c5b3ff7025e7c76566f8190210110b6.jpg)

![31aa27d29d8f138d44df84aaa8f335f94beae87115d46fe592ec1f15374105e5.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/31aa27d29d8f138d44df84aaa8f335f94beae87115d46fe592ec1f15374105e5.jpg)

![687bc00a6150fed1dbdde9478940a2643bbb6de5d1086c23eb1a427345eacc6c.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/687bc00a6150fed1dbdde9478940a2643bbb6de5d1086c23eb1a427345eacc6c.jpg)

![99746b701a057ae3f7530a247bc9088677112e2460c83071df648710733957d7.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/99746b701a057ae3f7530a247bc9088677112e2460c83071df648710733957d7.jpg)

![b4af1dafa76bd305956401b6a1d6cfdae71a4bca5b0bd609f02a4409b5af912e.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/b4af1dafa76bd305956401b6a1d6cfdae71a4bca5b0bd609f02a4409b5af912e.jpg)

![f42e4783ed6c6575de53ead3e3cd0fe430ae0a2f1051efc729db0de622b629d5.jpg](../icml_results/1539_DAG-MoE_%20From%20Simple%20Mixture%20to%20Structural%20Aggregation%20in%20Mixture-of-Experts/tables/f42e4783ed6c6575de53ead3e3cd0fe430ae0a2f1051efc729db0de622b629d5.jpg)
