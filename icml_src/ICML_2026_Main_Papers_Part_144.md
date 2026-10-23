# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 144 of 197

## 目录 (Table of Contents)

1. [TopAdapter: Topology-Aware Prompt Tuning for Efficient Point Cloud Understanding](#paper-4720)
2. [World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](#paper-4721)
3. [Harnessing Spectrum Video for Subject-Level Few-Shot and Cross-Montage EEG Generalization](#paper-4722)
4. [Beyond Heuristic Tuning: Power-Calibrated LLM Watermarking](#paper-4723)
5. [MoSA: Motion-constrained Stress Adaptation for Mitigating Real-to-Sim Gap in Continuum Dynamics via Learning Residual Anisotropy](#paper-4724)
6. [WET: Mitigating World-Conditioned Knowledge Conflicts via World Entropy Tethering](#paper-4725)
7. [Towards A Generative Protein Evolution Machine with DPLM-Evo](#paper-4726)
8. [Intervene When It Doubts: Conjunction-Guided Interactive Reasoning](#paper-4727)
9. [GeoEvo: Identity-Aware Potential Game with Geometric Evolution for Personalized Multimodal Federated Learning](#paper-4728)
10. [Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence](#paper-4729)
11. [Unveiling Multi-regime Patterns in SciML: Distinct Failure Modes and Regime-specific Optimization](#paper-4730)
12. [Dual-Calibration Multi-View Clustering via Compact Anchor Learning](#paper-4731)
13. [Spherical Procrustes Alignment for Reliable Medical Audio Diagnosis](#paper-4732)
14. [Acoustic Interference: A New Paradigm Weaponizing Acoustic Latent Semantic for Universal Jailbreak against Large Audio Language Models](#paper-4733)
15. [FormalRx: Rectify and eXamine Semantic Failures in Autoformalization](#paper-4734)
16. [Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution](#paper-4735)
17. [Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO](#paper-4736)
18. [TranX-Adapter: Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection](#paper-4737)
19. [Large-scale Uncertainty Quantification for Latent Variable Models Using Subsampling Markov Chain Monte Carlo](#paper-4738)
20. [Learning $U$-Statistics with Active Inference](#paper-4739)
21. [The Truth Lies Somewhere in the Middle (of the Generated Tokens)](#paper-4740)
22. [PhaseAlign: Complex Phase Alignment for Stable Open-Vocabulary Semantic Segmentation](#paper-4741)
23. [ProtoVAR: Efficient Dataset Distillation via Prototype-Guided Visual Autoregressive Modeling](#paper-4742)
24. [ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression](#paper-4743)
25. [GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation](#paper-4744)
26. [DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders](#paper-4745)
27. [PESD-TSF: A Period-Aware and Explicit Structured Decomposition Framework for Long-Term Time Series Forecasting](#paper-4746)
28. [When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems](#paper-4747)
29. [AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving](#paper-4748)
30. [LightWM: Training-Free Hierarchical Working Memory for Small Language Model Agents](#paper-4749)
31. [The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](#paper-4750)
32. [EvReflection: Event-Driven Micro-Dynamics for Reflection Removal](#paper-4751)
33. [Discovering Differences in Strategic Behavior between Humans and LLMs](#paper-4752)

---


<a id="paper-4720"></a>

## TopAdapter: Topology-Aware Prompt Tuning for Efficient Point Cloud Understanding

### Images

![059666fe62209b6d700b27280471c8dd8538c3517cbe2c0ec245348488420cc5.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/059666fe62209b6d700b27280471c8dd8538c3517cbe2c0ec245348488420cc5.jpg)

![09190872aaef286fa33fb3337690e95ba765ee6b190040c1c9286c1f80ad7232.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/09190872aaef286fa33fb3337690e95ba765ee6b190040c1c9286c1f80ad7232.jpg)

![0e58e952874d7cee0b67e4ad3d3124be7989bc9dfa3479fcfda0a9fa8ca56de9.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/0e58e952874d7cee0b67e4ad3d3124be7989bc9dfa3479fcfda0a9fa8ca56de9.jpg)

![23bb5172cc666807e6b6048ba711a39009015de822ac9abe5fc0b64032ad9ebb.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/23bb5172cc666807e6b6048ba711a39009015de822ac9abe5fc0b64032ad9ebb.jpg)

![5c9d6a1909f3153bb36f4adb17da4811c87c39d3d81b74e73de78fc00159c8de.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/5c9d6a1909f3153bb36f4adb17da4811c87c39d3d81b74e73de78fc00159c8de.jpg)

![73f0889654b7d0955477887647e4779f157e94682d9193501f0e03fbd484d376.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/73f0889654b7d0955477887647e4779f157e94682d9193501f0e03fbd484d376.jpg)

![75039233a16a1c91b568ed2caec0a01566111cf9731c7d029be9979a137b4893.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/75039233a16a1c91b568ed2caec0a01566111cf9731c7d029be9979a137b4893.jpg)

![8a4f8a14baed737bf88e79074dfcc237fa5a3cea869d3dfed124d6bc20f248cd.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/8a4f8a14baed737bf88e79074dfcc237fa5a3cea869d3dfed124d6bc20f248cd.jpg)

![f3a0d95ed4f956be41e732a9abf77d05e05063387447ae654668f83a40f27bb0.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/f3a0d95ed4f956be41e732a9abf77d05e05063387447ae654668f83a40f27bb0.jpg)

![f7ed48e6569405299557cefbbb0b99e1e5673bfb42970613f84e803165fa40f8.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/f7ed48e6569405299557cefbbb0b99e1e5673bfb42970613f84e803165fa40f8.jpg)

![ffac79c6fbdedeafda3f2c58c984146f6ffb0d7ccab3b313079baac0425ad96a.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/images/ffac79c6fbdedeafda3f2c58c984146f6ffb0d7ccab3b313079baac0425ad96a.jpg)

### Tables

![45b972e0c3b63e4eef232bdb3458087e3a9638cfbfa5c265b70bca47710ef8eb.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/45b972e0c3b63e4eef232bdb3458087e3a9638cfbfa5c265b70bca47710ef8eb.jpg)

![7c0c1dc9722cb8b27ed9bafc70b835d9e488391ca279c455e11f653e0335953f.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/7c0c1dc9722cb8b27ed9bafc70b835d9e488391ca279c455e11f653e0335953f.jpg)

![807077409f7968757a68761c2bb5bada2e0693668fb141944df3ee603e24d420.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/807077409f7968757a68761c2bb5bada2e0693668fb141944df3ee603e24d420.jpg)

![a79be0b55f886aae45dd0714a4d0ff708c02832dcd235abfa4b68929be7b55d5.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/a79be0b55f886aae45dd0714a4d0ff708c02832dcd235abfa4b68929be7b55d5.jpg)

![ee477597b8dab02ab43e62349eb89a6d5727a34c8f61ddf9cc600baabe9197e2.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/ee477597b8dab02ab43e62349eb89a6d5727a34c8f61ddf9cc600baabe9197e2.jpg)

![f78e139ad3fd1ac2da35f13ba3bbc6fd43a4c30454355cc58f2d51f2792ec4d5.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/f78e139ad3fd1ac2da35f13ba3bbc6fd43a4c30454355cc58f2d51f2792ec4d5.jpg)

![fb9e75f8b0b7d4e1ab489e7287993b9386cacf00a80b07d6c044ad63cbc91e0d.jpg](../icml_results/4767_FaPS_%20A%20General%20and%20Fast%20Training%20Method%20for%20Diffusion%20Models/tables/fb9e75f8b0b7d4e1ab489e7287993b9386cacf00a80b07d6c044ad63cbc91e0d.jpg)

## TopAdapter: Topology-Aware Prompt Tuning for Efficient Point Cloud Understanding


### Images

![2fb8c66c60d518841bd5bc0b5321adeb99f07e27586400ee5019e11b4692d742.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/images/2fb8c66c60d518841bd5bc0b5321adeb99f07e27586400ee5019e11b4692d742.jpg)

![534d17a1dab01f3e5332b5e12250486f62c31b4b6c157eb1afbe166305175ce2.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/images/534d17a1dab01f3e5332b5e12250486f62c31b4b6c157eb1afbe166305175ce2.jpg)

![986b238406b15a7e664b27bb7def91649870fa3e3f0004402eda81285cd19f33.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/images/986b238406b15a7e664b27bb7def91649870fa3e3f0004402eda81285cd19f33.jpg)

![e61479ff11dcfc5d6217031cf25b6ea03e5b79e5d75a8b80ba10ecb9ce43acdb.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/images/e61479ff11dcfc5d6217031cf25b6ea03e5b79e5d75a8b80ba10ecb9ce43acdb.jpg)

### Tables

![1b0954ab553ea111452e14b91e39a5da665ca68b2ff04732f2714ffc7d7231f9.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/tables/1b0954ab553ea111452e14b91e39a5da665ca68b2ff04732f2714ffc7d7231f9.jpg)

![44510812fdc873febda9696caa6f5c62c882e1681fba8068ef75091881240a5f.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/tables/44510812fdc873febda9696caa6f5c62c882e1681fba8068ef75091881240a5f.jpg)

![cb342b51edce16a73ea2afbb9b2d725594e8c785d1358505cbafd740d4b2d2f8.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/tables/cb342b51edce16a73ea2afbb9b2d725594e8c785d1358505cbafd740d4b2d2f8.jpg)

![e7fd94f42c004f731bd42ce3f794f266f7340e4ffacac30e292c0ed56d9d6baf.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/tables/e7fd94f42c004f731bd42ce3f794f266f7340e4ffacac30e292c0ed56d9d6baf.jpg)

![f5e294599c624e72a593cb787953b35d0f46093993f83ee757cb5688f6834616.jpg](../icml_results/4768_TopAdapter_%20Topology-Aware%20Prompt%20Tuning%20for%20Efficient%20Point%20Cloud%20Understanding/tables/f5e294599c624e72a593cb787953b35d0f46093993f83ee757cb5688f6834616.jpg)

<a id="paper-4721"></a>

## World-R1: Reinforcing 3D Constraints for Text-to-Video Generation


### Images

![0a242569328480ddf9d465d04755938d5f2f5b0fbbcb4dcddf9d677ca69f1bb1.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/0a242569328480ddf9d465d04755938d5f2f5b0fbbcb4dcddf9d677ca69f1bb1.jpg)

![1a08831c58e3e7c92f29c6ba314f0cc4020c4f3e65d2c2c96177507b2c32cf6f.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/1a08831c58e3e7c92f29c6ba314f0cc4020c4f3e65d2c2c96177507b2c32cf6f.jpg)

![27b561eb34a3424d3a3e18da1f84300f100b2f4bc23086b26ee3d7a1a0def9e3.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/27b561eb34a3424d3a3e18da1f84300f100b2f4bc23086b26ee3d7a1a0def9e3.jpg)

![72d30cdcd97a1bb17d21195a9549cbedc06dc390335c7e0229031300610e70d0.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/72d30cdcd97a1bb17d21195a9549cbedc06dc390335c7e0229031300610e70d0.jpg)

![735816c89f3a56db4a14a68fcd8635cba75645b97b5347a5b4abb39042959926.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/735816c89f3a56db4a14a68fcd8635cba75645b97b5347a5b4abb39042959926.jpg)

![8076aa2a95b91e9915c0cd280f8858318566bd19a2cbe9e441d45548ba11a453.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/8076aa2a95b91e9915c0cd280f8858318566bd19a2cbe9e441d45548ba11a453.jpg)

![a5f4e1ba5042f23d869215e2f43e98e0ecc84ab9e4c80ec81d8e1a2da0906beb.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/a5f4e1ba5042f23d869215e2f43e98e0ecc84ab9e4c80ec81d8e1a2da0906beb.jpg)

![c4938018d6bf8fa3ed47eae8d51c233c9083ef07d8fd1acaedae72723407b83b.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/images/c4938018d6bf8fa3ed47eae8d51c233c9083ef07d8fd1acaedae72723407b83b.jpg)

### Tables

![0996d8e7b26d76bae5469f2da36559eb2a3de7bca94be3c107b1f464547e1836.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/0996d8e7b26d76bae5469f2da36559eb2a3de7bca94be3c107b1f464547e1836.jpg)

![10e194df728cb7e027bdd2aaccc66dc14e51c459647462e20a33f3784b0e0ba9.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/10e194df728cb7e027bdd2aaccc66dc14e51c459647462e20a33f3784b0e0ba9.jpg)

![14ee172dc6f80fa850099394f6ae1e8a969495671fa563baeebc03e8f7aa500d.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/14ee172dc6f80fa850099394f6ae1e8a969495671fa563baeebc03e8f7aa500d.jpg)

![2d07bc80fd215059a035eb2f96dc231f7add8599b854d424fff0ca262f0726eb.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/2d07bc80fd215059a035eb2f96dc231f7add8599b854d424fff0ca262f0726eb.jpg)

![3e3a819d6f746a720c0a50e7a4ad87df40ac66e0fb60f9f047f7cc3453a30768.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/3e3a819d6f746a720c0a50e7a4ad87df40ac66e0fb60f9f047f7cc3453a30768.jpg)

![893ab38fd2c3c4b9bb3d7915d812701babe04b565b56180bd90f1b7a5838efa8.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/893ab38fd2c3c4b9bb3d7915d812701babe04b565b56180bd90f1b7a5838efa8.jpg)

![be3a17d63f0080641646b0a33ee38546294e974c33acfdf606c3bc51a46ca875.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/be3a17d63f0080641646b0a33ee38546294e974c33acfdf606c3bc51a46ca875.jpg)

![cbb7314c2d40d9f5756d712a375c56874a7702a098326d468e550bcb6579fbc1.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/cbb7314c2d40d9f5756d712a375c56874a7702a098326d468e550bcb6579fbc1.jpg)

![d3480be69d0a461d3cb4ada58f88fa064e274f1ed97789ed5840f558de85707a.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/d3480be69d0a461d3cb4ada58f88fa064e274f1ed97789ed5840f558de85707a.jpg)

![d4a10b5bd836962207a2f99e01d961a859edcafba623ff225f9a66cb248276ea.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/d4a10b5bd836962207a2f99e01d961a859edcafba623ff225f9a66cb248276ea.jpg)

![deb871c798a71ac1494eee0fe33d0d99b0b5323de106718fab20a46455ad7a7a.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/deb871c798a71ac1494eee0fe33d0d99b0b5323de106718fab20a46455ad7a7a.jpg)

![fccde664607bdde47cd540ed99dcff205d096f8b224778ba5a5edeff8cdc01dc.jpg](../icml_results/4769_World-R1_%20Reinforcing%203D%20Constraints%20for%20Text-to-Video%20Generation/tables/fccde664607bdde47cd540ed99dcff205d096f8b224778ba5a5edeff8cdc01dc.jpg)

<a id="paper-4722"></a>

## Harnessing Spectrum Video for Subject-Level Few-Shot and Cross-Montage EEG Generalization


### Images

![03a57681ef79bb931287d77bdfb4e1ac7bec9693ca45308fe1e54b454ee30396.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/03a57681ef79bb931287d77bdfb4e1ac7bec9693ca45308fe1e54b454ee30396.jpg)

![10c933d59a2dc9dd979bd695f4eb40f52e998a3903611b28b4925ba32dc67db1.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/10c933d59a2dc9dd979bd695f4eb40f52e998a3903611b28b4925ba32dc67db1.jpg)

![5c675ed5a53186c768e9a58b4e8dceb8b1471f4e3c0a2da5c3da218861c10b0c.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/5c675ed5a53186c768e9a58b4e8dceb8b1471f4e3c0a2da5c3da218861c10b0c.jpg)

![6e5235e686818e189604ecfcc352631b404a7c43fbdf9290230f88147a4df2ff.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/6e5235e686818e189604ecfcc352631b404a7c43fbdf9290230f88147a4df2ff.jpg)

![97442306f8659981680f4feca19ab05cd7447553f6689213e3d3ea05b46f5042.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/97442306f8659981680f4feca19ab05cd7447553f6689213e3d3ea05b46f5042.jpg)

![a3d95d4c378206d6786895ec8da3f783560d9d2ca81b40d4eb35ff8915f00625.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/a3d95d4c378206d6786895ec8da3f783560d9d2ca81b40d4eb35ff8915f00625.jpg)

![ad36b61f61682ff492ec0359cfcf0c9e9f580c2ee8a3f115322c8f7ea84b21c7.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/ad36b61f61682ff492ec0359cfcf0c9e9f580c2ee8a3f115322c8f7ea84b21c7.jpg)

![d818502f0fe604f252bca05d783c669ee2c0536fb7893f832cd63525e7886b29.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/images/d818502f0fe604f252bca05d783c669ee2c0536fb7893f832cd63525e7886b29.jpg)

### Tables

![103f2ffa242aa267683b749366b9bbd7fd3a28a27865b60617936b37a0984276.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/103f2ffa242aa267683b749366b9bbd7fd3a28a27865b60617936b37a0984276.jpg)

![2ba5a5d244baee81cef9a0cd2639085679a64ca728391d8aaab29e8d982f2344.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/2ba5a5d244baee81cef9a0cd2639085679a64ca728391d8aaab29e8d982f2344.jpg)

![358fe80d45891384a14d88aef965180b49397fb767206e881901c21359e8466d.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/358fe80d45891384a14d88aef965180b49397fb767206e881901c21359e8466d.jpg)

![6d5e014ab182589a49d3206863cdbaf6f99b686fec8ef738421f9c92e4717ea1.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/6d5e014ab182589a49d3206863cdbaf6f99b686fec8ef738421f9c92e4717ea1.jpg)

![72266542932679321fc3f010b8f58d4a5b8ae3881d0a80d25dc4579d66f32a66.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/72266542932679321fc3f010b8f58d4a5b8ae3881d0a80d25dc4579d66f32a66.jpg)

![a5199fea62050757611fa265b4c3760f7dc983fddf587859b01d192a16ab1ba3.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/a5199fea62050757611fa265b4c3760f7dc983fddf587859b01d192a16ab1ba3.jpg)

![f1a59f003e59dfdfbe73f478a4b1c0c7b1b1bae25f337af88a7f1de655690c5d.jpg](../icml_results/4770_Harnessing%20Spectrum%20Video%20for%20Subject-Level%20Few-Shot%20and%20Cross-Montage%20EEG%20Generalization/tables/f1a59f003e59dfdfbe73f478a4b1c0c7b1b1bae25f337af88a7f1de655690c5d.jpg)

<a id="paper-4723"></a>

## Beyond Heuristic Tuning: Power-Calibrated LLM Watermarking


### Images

![0e4021a91e9066e36f73004b722fe1413341c394a28952ad808813f16388e87f.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/0e4021a91e9066e36f73004b722fe1413341c394a28952ad808813f16388e87f.jpg)

![0e6037a664d94fbfa97f923adce59f50025e96aeb7e2e78c153fe511a3087b74.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/0e6037a664d94fbfa97f923adce59f50025e96aeb7e2e78c153fe511a3087b74.jpg)

![16b723652f5780d093a31879e661a27c436278cb21f156e045894b1a181e86c2.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/16b723652f5780d093a31879e661a27c436278cb21f156e045894b1a181e86c2.jpg)

![1d7bc1302718c77cb193ebf98fd21f3625bd55b851df231352513ac421723c73.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/1d7bc1302718c77cb193ebf98fd21f3625bd55b851df231352513ac421723c73.jpg)

![5d659e376f7a804e3bf9f812b83b383b7b262c3c30cfcf4bbe18943167a306ff.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/5d659e376f7a804e3bf9f812b83b383b7b262c3c30cfcf4bbe18943167a306ff.jpg)

![9158f467649f444e464f9827c59fef1ee35f4f384d1ca9e615abd11c2d3e6378.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/9158f467649f444e464f9827c59fef1ee35f4f384d1ca9e615abd11c2d3e6378.jpg)

![a975c6edf122ee8d040a51597eec3a770c7642161f97feaaa8f226389e087494.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/a975c6edf122ee8d040a51597eec3a770c7642161f97feaaa8f226389e087494.jpg)

![bca334bc6327691585cb3223a21ccd8b44b9e6ddd6d5fc205634031242a781df.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/bca334bc6327691585cb3223a21ccd8b44b9e6ddd6d5fc205634031242a781df.jpg)

![c38543b138c40a9f8ac2f909c031faa3b4040acad4a1be976b4552703d298fab.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/c38543b138c40a9f8ac2f909c031faa3b4040acad4a1be976b4552703d298fab.jpg)

![d782a8df624d6c6d22879853fda031a04638fe8657e580c7d66060749286d836.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/d782a8df624d6c6d22879853fda031a04638fe8657e580c7d66060749286d836.jpg)

![e370113c55575be49a834599ed0bc16493155b7bd04e187c07e029d4f9419388.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/e370113c55575be49a834599ed0bc16493155b7bd04e187c07e029d4f9419388.jpg)

![f9355f5422d462c1399878da101440b8601792a65c8d754c55b21b6bb5012aaa.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/images/f9355f5422d462c1399878da101440b8601792a65c8d754c55b21b6bb5012aaa.jpg)

### Tables

![3df902091c5586a9504905551432ddfd11ac01841ae38684950052b848457699.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/tables/3df902091c5586a9504905551432ddfd11ac01841ae38684950052b848457699.jpg)

![650f8adb37266460bd697e73f88f3c8c21792018f88f95d09069052b86d14341.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/tables/650f8adb37266460bd697e73f88f3c8c21792018f88f95d09069052b86d14341.jpg)

![8aae732b6336f5c186b44c9f081082d7f430096081a56047f6d8fd7d06ae7f72.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/tables/8aae732b6336f5c186b44c9f081082d7f430096081a56047f6d8fd7d06ae7f72.jpg)

![d18e34e3a28a7ef076d7bad11717769fd7d95aede53856a27123b9291d87cb1f.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/tables/d18e34e3a28a7ef076d7bad11717769fd7d95aede53856a27123b9291d87cb1f.jpg)

![f4f385ee41cb43dc826276d3f730bc10d677470a0cd86d2d1103739141dd691b.jpg](../icml_results/4771_Beyond%20Heuristic%20Tuning_%20Power-Calibrated%20LLM%20Watermarking/tables/f4f385ee41cb43dc826276d3f730bc10d677470a0cd86d2d1103739141dd691b.jpg)

<a id="paper-4724"></a>

## MoSA: Motion-constrained Stress Adaptation for Mitigating Real-to-Sim Gap in Continuum Dynamics via Learning Residual Anisotropy


### Images

![06f227a934ce1528755660c72b2eac21d8629cdc491694e8114557a5d52d8b23.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/06f227a934ce1528755660c72b2eac21d8629cdc491694e8114557a5d52d8b23.jpg)

![11ac0a968cf17bfeaf9dbee44b05399a8fcf0013780f13d315b8131964519c35.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/11ac0a968cf17bfeaf9dbee44b05399a8fcf0013780f13d315b8131964519c35.jpg)

![1e4812d621fb8b81fc1501a111e5ac92b0ea607e660e970c6507b432e3aef34d.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/1e4812d621fb8b81fc1501a111e5ac92b0ea607e660e970c6507b432e3aef34d.jpg)

![32d05116b27b2e547d8116f161df9689fbfff70797da30d4224490046831c2e0.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/32d05116b27b2e547d8116f161df9689fbfff70797da30d4224490046831c2e0.jpg)

![3371819bb48d2c12727046a3c930aeff25db70b3b24bcf260105d5b2f4f3ad77.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/3371819bb48d2c12727046a3c930aeff25db70b3b24bcf260105d5b2f4f3ad77.jpg)

![4e579475d5626778cb697013c082e157ad43ccd69207089e5b424193e6cbecec.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/4e579475d5626778cb697013c082e157ad43ccd69207089e5b424193e6cbecec.jpg)

![53c179656ab678f1586d14f5a7bc40cff6566230f53a41ab5f331de5d347eca3.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/53c179656ab678f1586d14f5a7bc40cff6566230f53a41ab5f331de5d347eca3.jpg)

![70739ce1de5bc83e86d4ddee6af3f8b04c6f271098151554856376fdd043cb0d.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/70739ce1de5bc83e86d4ddee6af3f8b04c6f271098151554856376fdd043cb0d.jpg)

![709d51e4d583d7052f23cbeefb1678e18fff6f9f617b3069579826bcf4d712e6.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/709d51e4d583d7052f23cbeefb1678e18fff6f9f617b3069579826bcf4d712e6.jpg)

![72846c4cefe0d96cacdcf965fe2ecc5073b811516b7a2e807829eb98653db4b9.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/72846c4cefe0d96cacdcf965fe2ecc5073b811516b7a2e807829eb98653db4b9.jpg)

![7bc140aafff24f887968e3a4c489d486595d2c1fc1a58aa58d0fab8b64e3db76.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/7bc140aafff24f887968e3a4c489d486595d2c1fc1a58aa58d0fab8b64e3db76.jpg)

![7fbe0bcae8119833cc9b98a76881075ed1a28a19525c4e6b4039a6f2c4745c7a.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/7fbe0bcae8119833cc9b98a76881075ed1a28a19525c4e6b4039a6f2c4745c7a.jpg)

![a15794dcf3b0f8c1954df9f8778894d0b0a43519be788f404f520b791bb44a90.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/a15794dcf3b0f8c1954df9f8778894d0b0a43519be788f404f520b791bb44a90.jpg)

![a22d82547bec70dc48c8d81ea3251404e62f2fc2bbd5252638df67f94f02abe7.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/a22d82547bec70dc48c8d81ea3251404e62f2fc2bbd5252638df67f94f02abe7.jpg)

![bb78e326bbf4ca768a6f5e115e05bb63e76d10716eb48df31081a4906d7da87a.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/bb78e326bbf4ca768a6f5e115e05bb63e76d10716eb48df31081a4906d7da87a.jpg)

![d11629f542159f78ae20e4cfbb1b7b739021d57a343b5ff24f7c71e28214641e.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/d11629f542159f78ae20e4cfbb1b7b739021d57a343b5ff24f7c71e28214641e.jpg)

![f49601634b87a5ffb47dc4244d95143533ca1c9ce1b13e89f790928c9b6bfbed.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/f49601634b87a5ffb47dc4244d95143533ca1c9ce1b13e89f790928c9b6bfbed.jpg)

![fa23bc787e3002bab731ba3697e179064d1b5a09f1462f309f02f13dbc6763b3.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/images/fa23bc787e3002bab731ba3697e179064d1b5a09f1462f309f02f13dbc6763b3.jpg)

### Tables

![39ce390359d3340b16d0687cbd4f1398f9522f231921b27fe6b75a1f1c866059.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/39ce390359d3340b16d0687cbd4f1398f9522f231921b27fe6b75a1f1c866059.jpg)

![46944b95f0e2fc45e31a55376d75be2a40015ec9931d67b18b07f9d7eebed62e.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/46944b95f0e2fc45e31a55376d75be2a40015ec9931d67b18b07f9d7eebed62e.jpg)

![4ce13895019f3bb8c7de08b51d37089091379f7f8916f7eb8afcc9f6793a7ce7.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/4ce13895019f3bb8c7de08b51d37089091379f7f8916f7eb8afcc9f6793a7ce7.jpg)

![7a62dcfc62d9689e3a7d308a1f52704af38b4ba2e7543116eb8ccb820df1bfbb.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/7a62dcfc62d9689e3a7d308a1f52704af38b4ba2e7543116eb8ccb820df1bfbb.jpg)

![85b5f4d1493d0909ba5e875255303fc33e80cf9aded7967d7ccd8c8d627787fc.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/85b5f4d1493d0909ba5e875255303fc33e80cf9aded7967d7ccd8c8d627787fc.jpg)

![ac728ca1b1ab86b4d86247522ed4236d957dfedebf3460d948624899a3750725.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/ac728ca1b1ab86b4d86247522ed4236d957dfedebf3460d948624899a3750725.jpg)

![bf98e0d71abd11e70ef467670bce0ea6f9ce5634f64ee2bd13a1aad4a9f17bca.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/bf98e0d71abd11e70ef467670bce0ea6f9ce5634f64ee2bd13a1aad4a9f17bca.jpg)

![cf37cb71f195c55b7612f6a65428981e53ffec8d0ed38e9adfc8b494672cc44b.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/cf37cb71f195c55b7612f6a65428981e53ffec8d0ed38e9adfc8b494672cc44b.jpg)

![df7a51c6f7614817099b79dfa6d09e9411fbbb6ad2ee541a4f6a5fe9d5fbdf10.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/df7a51c6f7614817099b79dfa6d09e9411fbbb6ad2ee541a4f6a5fe9d5fbdf10.jpg)

![efa53b823971a9580c8f06c0dbbb8069e247578673af0c3fb360209459c0b10e.jpg](../icml_results/4772_MoSA_%20Motion-constrained%20Stress%20Adaptation%20for%20Mitigating%20Real-to-Sim%20Gap%20in%20Continuum%20Dynamics%20via%20/tables/efa53b823971a9580c8f06c0dbbb8069e247578673af0c3fb360209459c0b10e.jpg)

<a id="paper-4725"></a>

## WET: Mitigating World-Conditioned Knowledge Conflicts via World Entropy Tethering


### Images

![05e4d936c4aae6d9bdae05376279684610b8c069d99d7859e2c49e0cac014a7f.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/05e4d936c4aae6d9bdae05376279684610b8c069d99d7859e2c49e0cac014a7f.jpg)

![060ae0618fb5534070a9bb649df63864ff96232d220733749b813aa56138710f.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/060ae0618fb5534070a9bb649df63864ff96232d220733749b813aa56138710f.jpg)

![0c776ce6adacac85be942a4c5ac342bc3e547513b6ab5ed5d4e0482267e4e968.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/0c776ce6adacac85be942a4c5ac342bc3e547513b6ab5ed5d4e0482267e4e968.jpg)

![0ccf8bdf3a0f4d010b608b64147bf07098eeb68bddb206cd77db553d104a1380.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/0ccf8bdf3a0f4d010b608b64147bf07098eeb68bddb206cd77db553d104a1380.jpg)

![39d5821c87f8fa60ea5a3c8a7684135846415c5591027688a2aa0fb5f1cb1455.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/39d5821c87f8fa60ea5a3c8a7684135846415c5591027688a2aa0fb5f1cb1455.jpg)

![447e0aea3e6955d15477b28dd9b83b83b0f50238febe654ec07ee73bcc574960.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/447e0aea3e6955d15477b28dd9b83b83b0f50238febe654ec07ee73bcc574960.jpg)

![6c420e19a764ef7fcd10d3f74e17206abe7519bd7ca9e9202edc220940141059.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/6c420e19a764ef7fcd10d3f74e17206abe7519bd7ca9e9202edc220940141059.jpg)

![7accce0715e5db180be578ba87baf4eff20a6500ea0242ab4e3ff7128e4114e6.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/7accce0715e5db180be578ba87baf4eff20a6500ea0242ab4e3ff7128e4114e6.jpg)

![91b3a88ad5f8be46094b163e665eee9a2d43082b772f3fdcd3692748dafeca76.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/91b3a88ad5f8be46094b163e665eee9a2d43082b772f3fdcd3692748dafeca76.jpg)

![abd291039eca977a148b06078feab0c4d09ee7f6793afa296e7dee1159bdd6cc.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/abd291039eca977a148b06078feab0c4d09ee7f6793afa296e7dee1159bdd6cc.jpg)

![b6f2c1e36ee6301a6585feea87c8ba6a44e836a59cd995b16e66c2a7eba0cbec.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/b6f2c1e36ee6301a6585feea87c8ba6a44e836a59cd995b16e66c2a7eba0cbec.jpg)

![bae6c4ad4ed385525117ee54289641e9dc740d7424249d0e58ad63f9d81d492b.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/bae6c4ad4ed385525117ee54289641e9dc740d7424249d0e58ad63f9d81d492b.jpg)

![bbc1744d7d0661d5f83dd480bb0c59bff9b5d6c744b585739eb7efbba3d4f533.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/bbc1744d7d0661d5f83dd480bb0c59bff9b5d6c744b585739eb7efbba3d4f533.jpg)

![c15c41651893210164ff9fc96a41056bae3db41b4e94617f1d2de0ec5123fce2.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/images/c15c41651893210164ff9fc96a41056bae3db41b4e94617f1d2de0ec5123fce2.jpg)

### Tables

![017b8b24701f2c2ac2af6fc2e110b1d53c7f0a394f25fcc6298783b41db96c7d.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/tables/017b8b24701f2c2ac2af6fc2e110b1d53c7f0a394f25fcc6298783b41db96c7d.jpg)

![03707320e7b05ac812f6d0138e4ad7c5cfcf2e07a15b8cf8400fc6d54f1d7eb5.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/tables/03707320e7b05ac812f6d0138e4ad7c5cfcf2e07a15b8cf8400fc6d54f1d7eb5.jpg)

![3668f5b69625e68a614f51cb0785aa3dd3085943051a43ea57cba496c5be69c7.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/tables/3668f5b69625e68a614f51cb0785aa3dd3085943051a43ea57cba496c5be69c7.jpg)

![bdc51b9fccf1d94d05e93a5c70501f8c3d76f0d74a223659fdc69c1f8b7a9d1e.jpg](../icml_results/4773_WET_%20Mitigating%20World-Conditioned%20Knowledge%20Conflicts%20via%20World%20Entropy%20Tethering/tables/bdc51b9fccf1d94d05e93a5c70501f8c3d76f0d74a223659fdc69c1f8b7a9d1e.jpg)

<a id="paper-4726"></a>

## Towards A Generative Protein Evolution Machine with DPLM-Evo


### Images

![0c6e098d5a913fee9952534845b8bea7004c8233fb39cdd16df689820e4b5dc3.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/0c6e098d5a913fee9952534845b8bea7004c8233fb39cdd16df689820e4b5dc3.jpg)

![4e155d818f35e959de384c704b8468e8a47393e8204e9b57ddb29421ab79e194.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/4e155d818f35e959de384c704b8468e8a47393e8204e9b57ddb29421ab79e194.jpg)

![4e462a498e19a2a76e9abb261c53641a0f0800f2c840a15aadfa299ace4a68a7.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/4e462a498e19a2a76e9abb261c53641a0f0800f2c840a15aadfa299ace4a68a7.jpg)

![5f813d68b89c76102b03f02c95a9fe93c8d683167210e07ad79ded4cfe0e916f.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/5f813d68b89c76102b03f02c95a9fe93c8d683167210e07ad79ded4cfe0e916f.jpg)

![612e2680dc1ba0261872d2d58601b4a93fc90b8fcc123e23a42f9d6ce29dcb00.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/612e2680dc1ba0261872d2d58601b4a93fc90b8fcc123e23a42f9d6ce29dcb00.jpg)

![8ef5197eb3b2bb0edf79f7340ff640efe0e14e9b4a2888093d42b8084cd61568.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/8ef5197eb3b2bb0edf79f7340ff640efe0e14e9b4a2888093d42b8084cd61568.jpg)

![e55540257695235b1bfbb3d84e83c5107dba7c79faa2a2ed45c077e7d0740554.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/e55540257695235b1bfbb3d84e83c5107dba7c79faa2a2ed45c077e7d0740554.jpg)

![f02d8ab32f80ff074834f9093aa72769ce907257ff8934ba55684c2bfea7181b.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/images/f02d8ab32f80ff074834f9093aa72769ce907257ff8934ba55684c2bfea7181b.jpg)

### Tables

![3474c2f79aea8350e2366f97f614d71441d916398b5d8877c906e8c0da35329e.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/tables/3474c2f79aea8350e2366f97f614d71441d916398b5d8877c906e8c0da35329e.jpg)

![951af3d8003dccf8a73ed10c789a940c30d1304b076eedc3b7838ca885a25e2f.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/tables/951af3d8003dccf8a73ed10c789a940c30d1304b076eedc3b7838ca885a25e2f.jpg)

![e260a748b4fd4fc3b4249336a91390bdc911062d0245fdf59af0e19f06eaafa9.jpg](../icml_results/4774_Towards%20A%20Generative%20Protein%20Evolution%20Machine%20with%20DPLM-Evo/tables/e260a748b4fd4fc3b4249336a91390bdc911062d0245fdf59af0e19f06eaafa9.jpg)

<a id="paper-4727"></a>

## Intervene When It Doubts: Conjunction-Guided Interactive Reasoning


### Images

![5bfd26710221f0cb06cb831ff175ffe97d5724ef23e4fea3fd42fca46df3fde4.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/5bfd26710221f0cb06cb831ff175ffe97d5724ef23e4fea3fd42fca46df3fde4.jpg)

![805874f9d1edddc23bfdcb5bffcfdc3a85cb446b56329c6e40ce9d2699d9c1ec.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/805874f9d1edddc23bfdcb5bffcfdc3a85cb446b56329c6e40ce9d2699d9c1ec.jpg)

![85217d799dc946bdc65355f3a0e044dea04b5a01a77b50ee270cc128929ffa28.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/85217d799dc946bdc65355f3a0e044dea04b5a01a77b50ee270cc128929ffa28.jpg)

![8c7ab5a7cc36df30444b973c9e9b111f7a15922280475f8e5136df1155609185.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/8c7ab5a7cc36df30444b973c9e9b111f7a15922280475f8e5136df1155609185.jpg)

![954d38f87580e241093c5e4efc458802806045a918962fd10816f5a0eee90866.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/954d38f87580e241093c5e4efc458802806045a918962fd10816f5a0eee90866.jpg)

![9a576daa74d4b38612a59c3d92fc652c7afa4b2ecd8bcf628b48f32a2035f3bc.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/9a576daa74d4b38612a59c3d92fc652c7afa4b2ecd8bcf628b48f32a2035f3bc.jpg)

![b2500fcbfa12a21741b5ce388205ed8295d3f5b1c16344e8557f7e3613907e81.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/b2500fcbfa12a21741b5ce388205ed8295d3f5b1c16344e8557f7e3613907e81.jpg)

![e26f6c169870691ca35c29da6c9c6829aa869c0c950b53431549dd2108c7ae92.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/e26f6c169870691ca35c29da6c9c6829aa869c0c950b53431549dd2108c7ae92.jpg)

![e2b9a314fd292b831464c63f59d8307a5a1dcb42c82d47395a88b98d9445195b.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/e2b9a314fd292b831464c63f59d8307a5a1dcb42c82d47395a88b98d9445195b.jpg)

![e2bbbc494fdf902b0cb55d63c9aea0ff74b6faa03e66d0b117ffac50aecb1dc5.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/e2bbbc494fdf902b0cb55d63c9aea0ff74b6faa03e66d0b117ffac50aecb1dc5.jpg)

![f0d96831b8d609848d0932dc2c6ebaf6c3b3334a843597fc936a485f2d02986b.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/images/f0d96831b8d609848d0932dc2c6ebaf6c3b3334a843597fc936a485f2d02986b.jpg)

### Tables

![07ec78cb4bf94023138505dd0b63549a64b6f8d62492150b02d867327d2c624c.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/07ec78cb4bf94023138505dd0b63549a64b6f8d62492150b02d867327d2c624c.jpg)

![368bd4b852a25952c3c159c3833ccb05668359cac417afda4122505a248af007.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/368bd4b852a25952c3c159c3833ccb05668359cac417afda4122505a248af007.jpg)

![3bfe08d0ab3e9996f953c41a73356d3398ba1ec736cb6bc13c328cd09ada8ab0.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/3bfe08d0ab3e9996f953c41a73356d3398ba1ec736cb6bc13c328cd09ada8ab0.jpg)

![3d6a8e99d546bca4663d6db4528091841770df9253e639b2e592d9e1a7d0f63d.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/3d6a8e99d546bca4663d6db4528091841770df9253e639b2e592d9e1a7d0f63d.jpg)

![6c18e175c0ab0ddf87a33b5654e92c71d704a63b3002c4f53594d1bb47dc64f2.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/6c18e175c0ab0ddf87a33b5654e92c71d704a63b3002c4f53594d1bb47dc64f2.jpg)

![9ec61551226cdbb634ed2f5709c751db57532f4124cb08facb1ecc88bd8a07de.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/9ec61551226cdbb634ed2f5709c751db57532f4124cb08facb1ecc88bd8a07de.jpg)

![aa8d01b39cbe1da578d15b5b998aaf70a15ee889b0e2596446fc4d053e47fe22.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/aa8d01b39cbe1da578d15b5b998aaf70a15ee889b0e2596446fc4d053e47fe22.jpg)

![d5b58dcdfcc5aa59e6a3e56ecf9aeda601e04a1fc32944caa9096cb0ab8abb73.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/d5b58dcdfcc5aa59e6a3e56ecf9aeda601e04a1fc32944caa9096cb0ab8abb73.jpg)

![e9caa627ace0eb8eaa8d7cf36990db3cf3011e43bf0b18bdb4a7215760c2a399.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/e9caa627ace0eb8eaa8d7cf36990db3cf3011e43bf0b18bdb4a7215760c2a399.jpg)

![f0229f88f70d33c469c4e4948a31ad03ce0c19e2746cd55eb34550fa37f885f1.jpg](../icml_results/4775_Intervene%20When%20It%20Doubts_%20Conjunction-Guided%20Interactive%20Reasoning/tables/f0229f88f70d33c469c4e4948a31ad03ce0c19e2746cd55eb34550fa37f885f1.jpg)

<a id="paper-4728"></a>

## GeoEvo: Identity-Aware Potential Game with Geometric Evolution for Personalized Multimodal Federated Learning


### Images

![69e2242e3be50d54f1a1d7e70bac4f13038553558ec1d7b2bc06dab81ac7a117.jpg](../icml_results/4776_GeoEvo_%20Identity-Aware%20Potential%20Game%20with%20Geometric%20Evolution%20for%20Personalized%20Multimodal%20Federated/images/69e2242e3be50d54f1a1d7e70bac4f13038553558ec1d7b2bc06dab81ac7a117.jpg)

![b01e57233c757c854d260828f8e5730ad73893adba7b1e55a5875ead265c00cf.jpg](../icml_results/4776_GeoEvo_%20Identity-Aware%20Potential%20Game%20with%20Geometric%20Evolution%20for%20Personalized%20Multimodal%20Federated/images/b01e57233c757c854d260828f8e5730ad73893adba7b1e55a5875ead265c00cf.jpg)

![ea1d511711f04366deab6a63983f8bfe0e0ac9361f62f0a9a7a2cefbd69cd3a9.jpg](../icml_results/4776_GeoEvo_%20Identity-Aware%20Potential%20Game%20with%20Geometric%20Evolution%20for%20Personalized%20Multimodal%20Federated/images/ea1d511711f04366deab6a63983f8bfe0e0ac9361f62f0a9a7a2cefbd69cd3a9.jpg)

### Tables

![006190ecab8f0576caf0e556f6359655b7abb8fa057aeb42fc18f3a9054622af.jpg](../icml_results/4776_GeoEvo_%20Identity-Aware%20Potential%20Game%20with%20Geometric%20Evolution%20for%20Personalized%20Multimodal%20Federated/tables/006190ecab8f0576caf0e556f6359655b7abb8fa057aeb42fc18f3a9054622af.jpg)

![c5e8fec17db1f3f335cdf2170a921e000e455b60a5080e292f236e597446ca46.jpg](../icml_results/4776_GeoEvo_%20Identity-Aware%20Potential%20Game%20with%20Geometric%20Evolution%20for%20Personalized%20Multimodal%20Federated/tables/c5e8fec17db1f3f335cdf2170a921e000e455b60a5080e292f236e597446ca46.jpg)

<a id="paper-4729"></a>

## Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence


### Images

![0ea0d8fc1f605d651d4b6bc5aad97b458e7d58c95904af3de24adfcc42c3d749.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/0ea0d8fc1f605d651d4b6bc5aad97b458e7d58c95904af3de24adfcc42c3d749.jpg)

![199e96bb10320926413b04006b6b79effa48393dff35dcf873907b69a30df677.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/199e96bb10320926413b04006b6b79effa48393dff35dcf873907b69a30df677.jpg)

![48c0ef11ce0e238b1fef853b5b1124d752d7d9161e79ed492b749b34a5fdc3ac.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/48c0ef11ce0e238b1fef853b5b1124d752d7d9161e79ed492b749b34a5fdc3ac.jpg)

![65804f5df693edd9f800fd01ba46a0f68a794ecb656dda02a86e2a4d1dc3a6b2.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/65804f5df693edd9f800fd01ba46a0f68a794ecb656dda02a86e2a4d1dc3a6b2.jpg)

![666e302c209a33a3f5813c746cee62d8041064a77da77750a2b985e2f8359e07.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/666e302c209a33a3f5813c746cee62d8041064a77da77750a2b985e2f8359e07.jpg)

![809104ce02d897c3e13b42ec75160c6bc35a31c45eb56f0e60d400b4a6d64b0c.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/809104ce02d897c3e13b42ec75160c6bc35a31c45eb56f0e60d400b4a6d64b0c.jpg)

![9118c992c331dfd5b86ff3aaf62ab719c33b5eee7dab2684d737ca585346dc6a.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/9118c992c331dfd5b86ff3aaf62ab719c33b5eee7dab2684d737ca585346dc6a.jpg)

![a86366b4f61e43e25b71ee5d42cf4414febe23eee4f696f5fa31804b39650d2d.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/a86366b4f61e43e25b71ee5d42cf4414febe23eee4f696f5fa31804b39650d2d.jpg)

![cf456cb211c67802a5e538ee6840d2d5bcd391e2b45c69a9d55c6e7e62d00c75.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/cf456cb211c67802a5e538ee6840d2d5bcd391e2b45c69a9d55c6e7e62d00c75.jpg)

![d7ddbbabb1c8518bd5ef352e5d22cc0ab7aa9b115559e510501bae73fef08dcc.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/d7ddbbabb1c8518bd5ef352e5d22cc0ab7aa9b115559e510501bae73fef08dcc.jpg)

![dc3f6c5430680c513728cfaedbacb9b8c830e8e8253d23f1ed4b9141fedde0bf.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/dc3f6c5430680c513728cfaedbacb9b8c830e8e8253d23f1ed4b9141fedde0bf.jpg)

![eb8835d039cfc4ce7d5484b7801669bebaa69b01e3db3746e6b38f2da6839455.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/eb8835d039cfc4ce7d5484b7801669bebaa69b01e3db3746e6b38f2da6839455.jpg)

![f36f69bf79eb32b8e5ad945d41ab1de2f78d8a24e6c13a550a34ebbc709d6e42.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/f36f69bf79eb32b8e5ad945d41ab1de2f78d8a24e6c13a550a34ebbc709d6e42.jpg)

![f92abba09e5db9f910947c7fa0655f3390477a3fe9c836bba0eb50976b8261b4.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/f92abba09e5db9f910947c7fa0655f3390477a3fe9c836bba0eb50976b8261b4.jpg)

![fb3a89483f70ae1b9628090fbb679fff7087781ed0581fe174ec5b7538b883a5.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/images/fb3a89483f70ae1b9628090fbb679fff7087781ed0581fe174ec5b7538b883a5.jpg)

### Tables

![0ceed225212f3991b8b8ce09488cfb0f26e9596898d52c20d7672e5fa1bbaa95.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/0ceed225212f3991b8b8ce09488cfb0f26e9596898d52c20d7672e5fa1bbaa95.jpg)

![24191bb82051e0406d335587323a91968628022eb1e82060a6bf41c7bf73c52b.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/24191bb82051e0406d335587323a91968628022eb1e82060a6bf41c7bf73c52b.jpg)

![27f71354fe4506960a97ff252e60005fa18d352559f0b67839a78f71ec64634e.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/27f71354fe4506960a97ff252e60005fa18d352559f0b67839a78f71ec64634e.jpg)

![4f592b04de8f058c9807675c82a764e102a9b1fa2100995b74e7f6b23ca01a94.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/4f592b04de8f058c9807675c82a764e102a9b1fa2100995b74e7f6b23ca01a94.jpg)

![66d1f80e7f054740d0ca670d7def5721deb6ecaf5aa8423f58b62d67a7132add.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/66d1f80e7f054740d0ca670d7def5721deb6ecaf5aa8423f58b62d67a7132add.jpg)

![6ad994415b0434c1b0cd85bae811ab1bb2bb8a1df9d3ab1aa28e8c17c1af6ab1.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/6ad994415b0434c1b0cd85bae811ab1bb2bb8a1df9d3ab1aa28e8c17c1af6ab1.jpg)

![81065918412a346629921a0003760b449a7b732341fb3c85073bcfd91cf2d3cb.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/81065918412a346629921a0003760b449a7b732341fb3c85073bcfd91cf2d3cb.jpg)

![ad2b70f0927c7b1ba634b30a86d2ab9404a0732e19ff610a6ba67f1d17d42083.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/ad2b70f0927c7b1ba634b30a86d2ab9404a0732e19ff610a6ba67f1d17d42083.jpg)

![b1e08e443574e327ec2aa294deab117d65122697112e7697c0d72e4ff42635b0.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/b1e08e443574e327ec2aa294deab117d65122697112e7697c0d72e4ff42635b0.jpg)

![b5429f5fc0b5ae5dfea16fa1f3aa0a030146bf1756a00f8186f9945e930eefe2.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/b5429f5fc0b5ae5dfea16fa1f3aa0a030146bf1756a00f8186f9945e930eefe2.jpg)

![bd855a8d6f59e2271c546c8245b9c593dcefbc2f92d97516c971c92c4bf94671.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/bd855a8d6f59e2271c546c8245b9c593dcefbc2f92d97516c971c92c4bf94671.jpg)

![c7b2d5434ea642248d4479a75596769579fa324d8caea1a347bd32f49fd791f2.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/c7b2d5434ea642248d4479a75596769579fa324d8caea1a347bd32f49fd791f2.jpg)

![c8577606886f7cf29ecba6bc78201cde0000c44063a3899e51a4d4e84e9589fb.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/c8577606886f7cf29ecba6bc78201cde0000c44063a3899e51a4d4e84e9589fb.jpg)

![c973543d75fbc654d1eef8906d3a20b1585e90184fd16bd4704e5dbf116058f7.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/c973543d75fbc654d1eef8906d3a20b1585e90184fd16bd4704e5dbf116058f7.jpg)

![e78ad8da95c0fcfe2187c5bff2b339202b60cace816d7c329a9b51292053983e.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/e78ad8da95c0fcfe2187c5bff2b339202b60cace816d7c329a9b51292053983e.jpg)

![ebb2020e9f12a79ddde8b08775fce3e522e72fc1a32d26cbcbbea71a52e05e9c.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/ebb2020e9f12a79ddde8b08775fce3e522e72fc1a32d26cbcbbea71a52e05e9c.jpg)

![ef6a7c580cea40a33cdcc53926c7469c3be047c57ac5476a05b78e8ea46747f4.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/ef6a7c580cea40a33cdcc53926c7469c3be047c57ac5476a05b78e8ea46747f4.jpg)

![f109bd2423a220f7e4bf57c09d54c2693f88792bd3e14fdab6ad3a1b60caf170.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/f109bd2423a220f7e4bf57c09d54c2693f88792bd3e14fdab6ad3a1b60caf170.jpg)

![f1a826b2fdbf055a3c6c308892c7ba328284534459d7cf96cef4f6283824e4b0.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/f1a826b2fdbf055a3c6c308892c7ba328284534459d7cf96cef4f6283824e4b0.jpg)

![fdfcf3b01948202c664f7bea6113b10dedbbda7efc4413550f0aeb95d877a003.jpg](../icml_results/4777_Why%20Specialist%20Models%20Still%20Matter_%20A%20Heterogeneous%20Multi-Agent%20Paradigm%20for%20Medical%20Artificial%20Inte/tables/fdfcf3b01948202c664f7bea6113b10dedbbda7efc4413550f0aeb95d877a003.jpg)

<a id="paper-4730"></a>

## Unveiling Multi-regime Patterns in SciML: Distinct Failure Modes and Regime-specific Optimization


### Images

![11f85bb9b0de2fe3644dcb6a16bb14a005d7aed5c57cc70867c27355aabe1091.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/11f85bb9b0de2fe3644dcb6a16bb14a005d7aed5c57cc70867c27355aabe1091.jpg)

![1dca8619c9f2cf124a9433be7442818817eb1e0666b69d484c7b52ee290408cc.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/1dca8619c9f2cf124a9433be7442818817eb1e0666b69d484c7b52ee290408cc.jpg)

![2b95fbd0c4802581ca0b4d23634bbcadff4b959e51d5f947a0d80b459ef16552.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/2b95fbd0c4802581ca0b4d23634bbcadff4b959e51d5f947a0d80b459ef16552.jpg)

![3a8a2a5cfa53a9f12251f9991d875316cf203db80f46d53464e34f280708e631.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/3a8a2a5cfa53a9f12251f9991d875316cf203db80f46d53464e34f280708e631.jpg)

![4ad9ee8ef84a6405a4110265104850076a61f8a3b58bf7320a44fafb83f857db.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/4ad9ee8ef84a6405a4110265104850076a61f8a3b58bf7320a44fafb83f857db.jpg)

![637c9ad2b50b00879c881e8cbaf6f7bd4d4b0f6c47bbf8b52dd586fbe9890640.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/637c9ad2b50b00879c881e8cbaf6f7bd4d4b0f6c47bbf8b52dd586fbe9890640.jpg)

![694d2c0912338e1a974da72d5d9172eb47483e494c4137d8968e5045158be635.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/694d2c0912338e1a974da72d5d9172eb47483e494c4137d8968e5045158be635.jpg)

![6ed4a46b05b7a7d53001335d1e2658df833a318b187046709d4f114130c6a897.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/6ed4a46b05b7a7d53001335d1e2658df833a318b187046709d4f114130c6a897.jpg)

![7ae9f6029606a0c77dc19fe577379dc0b8680c54e70b966713614716b0411f7f.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/7ae9f6029606a0c77dc19fe577379dc0b8680c54e70b966713614716b0411f7f.jpg)

![7b30aa8d3b19ebc92246204f6097022275bd70cf4a994d84dd47e2de52c6fc49.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/7b30aa8d3b19ebc92246204f6097022275bd70cf4a994d84dd47e2de52c6fc49.jpg)

![8d9844d5a593acee0bc901e38e2b1d3ce7fbe04543868f9ec8cad111748171dc.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/8d9844d5a593acee0bc901e38e2b1d3ce7fbe04543868f9ec8cad111748171dc.jpg)

![8dd00aea11d8fb071e2824c65683844c178c4cdac7b7ec9a3f13dd72ca79539f.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/8dd00aea11d8fb071e2824c65683844c178c4cdac7b7ec9a3f13dd72ca79539f.jpg)

![a62e905c076bcb0de34290002dcc236466615702051eee65281b6484326ad145.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/a62e905c076bcb0de34290002dcc236466615702051eee65281b6484326ad145.jpg)

![bc24fe68561e02aba140fca5596829d873ed5146618da0f1702a71a741fa7520.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/bc24fe68561e02aba140fca5596829d873ed5146618da0f1702a71a741fa7520.jpg)

![d47ede969713f8cdbd721bfac1d77eb6050d6a23d91cc6cc79ddf3006df38358.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/d47ede969713f8cdbd721bfac1d77eb6050d6a23d91cc6cc79ddf3006df38358.jpg)

![f84598112849c682278d0fbfb490f2d2b1bf7bd254f5f297b9ea6c7e15b7b456.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/images/f84598112849c682278d0fbfb490f2d2b1bf7bd254f5f297b9ea6c7e15b7b456.jpg)

### Tables

![1d50f9fedb6a30ed3a13189b68efbb00378aa07e9d3881bc3bf33bc67beb5916.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/tables/1d50f9fedb6a30ed3a13189b68efbb00378aa07e9d3881bc3bf33bc67beb5916.jpg)

![1f0bd08ed8cec4af80634c7d13919578ab62955aed1c90c72510877417b1c983.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/tables/1f0bd08ed8cec4af80634c7d13919578ab62955aed1c90c72510877417b1c983.jpg)

![825599dd8dcdd9db18132c29cb2f370dbafdfd58677e0452f0e44fe72b2e6e89.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/tables/825599dd8dcdd9db18132c29cb2f370dbafdfd58677e0452f0e44fe72b2e6e89.jpg)

![d1171f686b559094f3375414eec7d921c6bc4babf096adf2c555252e38f3e7e3.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/tables/d1171f686b559094f3375414eec7d921c6bc4babf096adf2c555252e38f3e7e3.jpg)

![e10248e1ff691a9e04fe900c2a73a7fea05158ff68ab30bb28959094caf22785.jpg](../icml_results/4778_Unveiling%20Multi-regime%20Patterns%20in%20SciML_%20Distinct%20Failure%20Modes%20and%20Regime-specific%20Optimization/tables/e10248e1ff691a9e04fe900c2a73a7fea05158ff68ab30bb28959094caf22785.jpg)

<a id="paper-4731"></a>

## Dual-Calibration Multi-View Clustering via Compact Anchor Learning


### Images

![2fb51293d3975ab546824b3e304e6864fafc6546065eb9082588bd24c44a1511.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/2fb51293d3975ab546824b3e304e6864fafc6546065eb9082588bd24c44a1511.jpg)

![3577925902edcf397b775b475c38467e62f495bee4f51923eb3a1b9b500a888b.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/3577925902edcf397b775b475c38467e62f495bee4f51923eb3a1b9b500a888b.jpg)

![374720451f0a7afcb4cf8a75d1ad02709c5178a7e1c380314cf5de8ee94bca46.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/374720451f0a7afcb4cf8a75d1ad02709c5178a7e1c380314cf5de8ee94bca46.jpg)

![4eb65fffc130038fec8a810fe8c522d81281355c3fa35f3b7f0f27352eaf6bc4.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/4eb65fffc130038fec8a810fe8c522d81281355c3fa35f3b7f0f27352eaf6bc4.jpg)

![8dbf70c76993d4734825f0382b581fe7c230143b4a4e5b1881e54dab5c7e8480.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/8dbf70c76993d4734825f0382b581fe7c230143b4a4e5b1881e54dab5c7e8480.jpg)

![9815f246f78998d56cc8f36552ec38d957733a3130e8763ccc6cabda92d89586.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/9815f246f78998d56cc8f36552ec38d957733a3130e8763ccc6cabda92d89586.jpg)

![b4eff139184243e8da51aacca6376fef83d1dcddc7a0a49b69f6ad0380dbae4d.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/images/b4eff139184243e8da51aacca6376fef83d1dcddc7a0a49b69f6ad0380dbae4d.jpg)

### Tables

![c34fe74ab50a2f20510a93bf352fb711479fa768c9d230714ecc29430ad02b07.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/tables/c34fe74ab50a2f20510a93bf352fb711479fa768c9d230714ecc29430ad02b07.jpg)

![e2a0f71ffc0d06441b08c9d523b33a34d376f12a21c074e162b9a79406b81655.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/tables/e2a0f71ffc0d06441b08c9d523b33a34d376f12a21c074e162b9a79406b81655.jpg)

![fdf509c103b0e5b7b6c78e2f44d82338a84c4da36cdbf9998d5adc2b36bafcc5.jpg](../icml_results/4779_Dual-Calibration%20Multi-View%20Clustering%20via%20Compact%20Anchor%20Learning/tables/fdf509c103b0e5b7b6c78e2f44d82338a84c4da36cdbf9998d5adc2b36bafcc5.jpg)

<a id="paper-4732"></a>

## Spherical Procrustes Alignment for Reliable Medical Audio Diagnosis


### Images

![03450f0d44e9b04010f26eeb3f4c1c85b0858721cd21e785405665572b5a9690.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/03450f0d44e9b04010f26eeb3f4c1c85b0858721cd21e785405665572b5a9690.jpg)

![158ce2202e6d47ea482155f09d0d508512000a29c41901de8d157d65891e3e86.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/158ce2202e6d47ea482155f09d0d508512000a29c41901de8d157d65891e3e86.jpg)

![1ab47ab11f826cee9c8de6c93d8efa49212ef16f0e47de74d1b6f7579c8ddc14.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/1ab47ab11f826cee9c8de6c93d8efa49212ef16f0e47de74d1b6f7579c8ddc14.jpg)

![4c319ece1710da48eac8a5560821d0ccc317e3fca351b96d87dbd08a419cda35.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/4c319ece1710da48eac8a5560821d0ccc317e3fca351b96d87dbd08a419cda35.jpg)

![683d7f5276f6ea578ee46d827665792cdaf9d8d3d711577b8de65c66671ac140.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/683d7f5276f6ea578ee46d827665792cdaf9d8d3d711577b8de65c66671ac140.jpg)

![eb7511a8addb0a70ff20cd209c4f0fe5e91346603aa4f2280778eccf925d5879.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/images/eb7511a8addb0a70ff20cd209c4f0fe5e91346603aa4f2280778eccf925d5879.jpg)

### Tables

![0d1b389add7fc32e0dbb78571721a30fe48a2a4d636ba3b804b73a8fe2e1c897.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/tables/0d1b389add7fc32e0dbb78571721a30fe48a2a4d636ba3b804b73a8fe2e1c897.jpg)

![12cb1b3462b604f308412feb8cf6f6b8237b28515a19f127869007c099ee0e5c.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/tables/12cb1b3462b604f308412feb8cf6f6b8237b28515a19f127869007c099ee0e5c.jpg)

![61f561a754a7b9b0744340cef205bd07a2a8e50e5a836328ae91574a92fb2784.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/tables/61f561a754a7b9b0744340cef205bd07a2a8e50e5a836328ae91574a92fb2784.jpg)

![7b45ec6500525a6824e10b0442d568faffbaaa36e657eb990f34ec09bc93e466.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/tables/7b45ec6500525a6824e10b0442d568faffbaaa36e657eb990f34ec09bc93e466.jpg)

![89ede2e6b9582d9b78c2b712a46f9448b2419be7cb84fcce55dae4e8fdfa598d.jpg](../icml_results/4780_Spherical%20Procrustes%20Alignment%20for%20Reliable%20Medical%20Audio%20Diagnosis/tables/89ede2e6b9582d9b78c2b712a46f9448b2419be7cb84fcce55dae4e8fdfa598d.jpg)

<a id="paper-4733"></a>

## Acoustic Interference: A New Paradigm Weaponizing Acoustic Latent Semantic for Universal Jailbreak against Large Audio Language Models


### Images

![34b869bb737d15c75c1016683f08494f6554b09e8b1c760a40c7e35112c83f14.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/34b869bb737d15c75c1016683f08494f6554b09e8b1c760a40c7e35112c83f14.jpg)

![6da553f77be39da2b9a27491b545d37ece7d46bbc61b004328835023235e37fc.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/6da553f77be39da2b9a27491b545d37ece7d46bbc61b004328835023235e37fc.jpg)

![8054d9f1cc40d86785a6a99c01a0646208fec320d6706e1a025668e433a279c5.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/8054d9f1cc40d86785a6a99c01a0646208fec320d6706e1a025668e433a279c5.jpg)

![9e81148c9b62fba5ac4043ff63efeb1aa77bca24da35b42fbd2ef873adfd82af.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/9e81148c9b62fba5ac4043ff63efeb1aa77bca24da35b42fbd2ef873adfd82af.jpg)

![ba0feb67b5644ddcf589258bfc8c40bfdfc6fa5aa117899598a6d1f783f1f9bc.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/ba0feb67b5644ddcf589258bfc8c40bfdfc6fa5aa117899598a6d1f783f1f9bc.jpg)

![ce04a4b0b922cac219ed8501c55fa213dea43abb1d260d9423adf44a52fc92dd.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/ce04a4b0b922cac219ed8501c55fa213dea43abb1d260d9423adf44a52fc92dd.jpg)

![f292b13943b60e50ee6bf2828c094a49008a30cfed8c5a07882ec71d895e7cbc.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/images/f292b13943b60e50ee6bf2828c094a49008a30cfed8c5a07882ec71d895e7cbc.jpg)

### Tables

![3217a1a06aaca1e00011ac358215941cb6f01bfac8af450bf4e3e99982b1f443.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/3217a1a06aaca1e00011ac358215941cb6f01bfac8af450bf4e3e99982b1f443.jpg)

![4395a8e588daa60bda084638c5df6f6d0ea845c412be7ab84c450279788ebbe3.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/4395a8e588daa60bda084638c5df6f6d0ea845c412be7ab84c450279788ebbe3.jpg)

![5502ff650bc3b7f3806e3daae14365760ee944169e6d94d4f053e89980aea86e.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/5502ff650bc3b7f3806e3daae14365760ee944169e6d94d4f053e89980aea86e.jpg)

![8587a6bd6ed98301ea5ba28eaf2443b43f93695ea37693a8dcc6e255bcd218bd.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/8587a6bd6ed98301ea5ba28eaf2443b43f93695ea37693a8dcc6e255bcd218bd.jpg)

![987999185d848d367013b15eaab6c4a562ae194465d03cf16d8edee432786d0a.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/987999185d848d367013b15eaab6c4a562ae194465d03cf16d8edee432786d0a.jpg)

![9f40f6d4352cb05de2d4b63279d4821425317a87ce2c88a583e56d66dfc9df0f.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/9f40f6d4352cb05de2d4b63279d4821425317a87ce2c88a583e56d66dfc9df0f.jpg)

![bd271a88bcdfe7c30af4a3a1093a42fb0b28f926d4293cd61de063d304de9e87.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/bd271a88bcdfe7c30af4a3a1093a42fb0b28f926d4293cd61de063d304de9e87.jpg)

![eac6f52b6449456fa841ddd6ce217aa7267f56b0f559386cf10dbfe702130209.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/eac6f52b6449456fa841ddd6ce217aa7267f56b0f559386cf10dbfe702130209.jpg)

![fc9808b332aa7bdbe88ab0d823c3922f4a382401063610e18ab23037a99cf549.jpg](../icml_results/4781_Acoustic%20Interference_%20A%20New%20Paradigm%20Weaponizing%20Acoustic%20Latent%20Semantic%20for%20Universal%20Jailbreak%20a/tables/fc9808b332aa7bdbe88ab0d823c3922f4a382401063610e18ab23037a99cf549.jpg)

<a id="paper-4734"></a>

## FormalRx: Rectify and eXamine Semantic Failures in Autoformalization


### Images

![189d980dfb5593baf8272a96924f8d1d2c3eda95e8fa98f7480a27be96f6a242.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/189d980dfb5593baf8272a96924f8d1d2c3eda95e8fa98f7480a27be96f6a242.jpg)

![2ce70657862f90598a91ca2eefe841e4948081348283cc4c432f761a27ed4a1c.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/2ce70657862f90598a91ca2eefe841e4948081348283cc4c432f761a27ed4a1c.jpg)

![4086ea222043d5636118873b4114c0df288156c990bcaa4b5a9554654aa42d30.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/4086ea222043d5636118873b4114c0df288156c990bcaa4b5a9554654aa42d30.jpg)

![5893568d77772d3c37f3cb79c461e3ee46bca1136b0a05f9829e32de5fb09766.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/5893568d77772d3c37f3cb79c461e3ee46bca1136b0a05f9829e32de5fb09766.jpg)

![8e368c2d571148bd6ed6f0e498efee1fbac6cbe6f5dc88fba4eb716fd7ab78ee.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/8e368c2d571148bd6ed6f0e498efee1fbac6cbe6f5dc88fba4eb716fd7ab78ee.jpg)

![9c3e66b587008a40ea7745735fbdbcaa2d006ebddcd45fd23dc6aeda6c129411.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/9c3e66b587008a40ea7745735fbdbcaa2d006ebddcd45fd23dc6aeda6c129411.jpg)

![f67b2fe1ff104732fb73aeab4afb1bb478c32fc99fdd733d7d0247591e6cdd1d.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/images/f67b2fe1ff104732fb73aeab4afb1bb478c32fc99fdd733d7d0247591e6cdd1d.jpg)

### Tables

![02dd5bcc1ec064f6049701693fbc1e39d837df12f26606aaf4f412afa1db3728.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/02dd5bcc1ec064f6049701693fbc1e39d837df12f26606aaf4f412afa1db3728.jpg)

![1cf3a79ad06a7327a982fd09bffa33ad2dc4077f08942378cedc4074df758396.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/1cf3a79ad06a7327a982fd09bffa33ad2dc4077f08942378cedc4074df758396.jpg)

![251207c0c131c239a079959e8bfaf8e4922dd470edee34d6f76e81bce993c35a.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/251207c0c131c239a079959e8bfaf8e4922dd470edee34d6f76e81bce993c35a.jpg)

![45a07816f849621fe2f54e03516f4d929c83430bfd01466093c69120c2183a28.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/45a07816f849621fe2f54e03516f4d929c83430bfd01466093c69120c2183a28.jpg)

![45e3fc3f8ac89e9c1dd68d05c0a6082443fd441b2b5cc79cdffc4430d8282e3d.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/45e3fc3f8ac89e9c1dd68d05c0a6082443fd441b2b5cc79cdffc4430d8282e3d.jpg)

![53d8aae535d6730cecbc801705784169071de84a10651aa48fadeb048e78e5a1.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/53d8aae535d6730cecbc801705784169071de84a10651aa48fadeb048e78e5a1.jpg)

![6603ae9069dea1de8bdec17cacc68771f729d757de836f36423f86bf11321d01.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/6603ae9069dea1de8bdec17cacc68771f729d757de836f36423f86bf11321d01.jpg)

![7b0d8ba9bb28116f7b70ae9526c2fbe28a6c677d60c83bd6fdca21268c1395b9.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/7b0d8ba9bb28116f7b70ae9526c2fbe28a6c677d60c83bd6fdca21268c1395b9.jpg)

![811585894142fe94f47439b8c037b8368e29f0a5e8a61d4d29ca5204f564551d.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/811585894142fe94f47439b8c037b8368e29f0a5e8a61d4d29ca5204f564551d.jpg)

![816ee53d5267d935dcce987dc87c68f7d19d26cc14af525801e648e23a0d1d42.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/816ee53d5267d935dcce987dc87c68f7d19d26cc14af525801e648e23a0d1d42.jpg)

![8fc35328d527adf8ac24a9e5faff398bb1c3d74571a4a231f410bf9df8babe30.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/8fc35328d527adf8ac24a9e5faff398bb1c3d74571a4a231f410bf9df8babe30.jpg)

![9bfaa6fb053bd85169e6ba9b796b5bfb5e211b5488a763219eb4d2ad7f4dfe1f.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/9bfaa6fb053bd85169e6ba9b796b5bfb5e211b5488a763219eb4d2ad7f4dfe1f.jpg)

![c9bd505b1d6c6f9a7a037d9cd70e12383c71d9c5b152922c24fc9d23b054e623.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/c9bd505b1d6c6f9a7a037d9cd70e12383c71d9c5b152922c24fc9d23b054e623.jpg)

![d273fbb21b0769e86b1fd94500ab63aec3086c671f76d3d2e048d19c6847482d.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/d273fbb21b0769e86b1fd94500ab63aec3086c671f76d3d2e048d19c6847482d.jpg)

![e1a66f240fa7bb512f363ed50a75a518de6d467d7cb0eed15ccf85e5b5d71f3b.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/e1a66f240fa7bb512f363ed50a75a518de6d467d7cb0eed15ccf85e5b5d71f3b.jpg)

![fe3b0a721c39536f0cf511715c79adfdbf98418e41d56bb50ee33c0801c88516.jpg](../icml_results/4782_FormalRx_%20Rectify%20and%20eXamine%20Semantic%20Failures%20in%20Autoformalization/tables/fe3b0a721c39536f0cf511715c79adfdbf98418e41d56bb50ee33c0801c88516.jpg)

<a id="paper-4735"></a>

## Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution


### Images

![1045f017c2292cbbcf860506d2507cc4e2413cf6f3b8d2facd4b8fbe5b3f8c4b.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/1045f017c2292cbbcf860506d2507cc4e2413cf6f3b8d2facd4b8fbe5b3f8c4b.jpg)

![1abe8ebf22c0f43724f139fa31317e7ff3a7d4881b735d6a92f12ef6d964d4d1.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/1abe8ebf22c0f43724f139fa31317e7ff3a7d4881b735d6a92f12ef6d964d4d1.jpg)

![1fb6f1629852485b590ac0a662f21c378be9cc89ab5fd4395fddd29d24f1bb05.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/1fb6f1629852485b590ac0a662f21c378be9cc89ab5fd4395fddd29d24f1bb05.jpg)

![228dac99b174779b0c9b458885c661f978300acb21ef82d1bafd3880193624c5.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/228dac99b174779b0c9b458885c661f978300acb21ef82d1bafd3880193624c5.jpg)

![231754b5483fe1bbc90f3279549a01c2d4df60984db197cf8fd030b176f285a2.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/231754b5483fe1bbc90f3279549a01c2d4df60984db197cf8fd030b176f285a2.jpg)

![640dbb35757e3b869247c2ec35c6db9bdd845b27a6fe4a412d9b3359c819a960.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/640dbb35757e3b869247c2ec35c6db9bdd845b27a6fe4a412d9b3359c819a960.jpg)

![67f5171f90f07eb49cdefe674e1eaa96ac24a452ab6f03eb7493959348a4a43f.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/67f5171f90f07eb49cdefe674e1eaa96ac24a452ab6f03eb7493959348a4a43f.jpg)

![96e552788ed77468a19339c13019afeade7e00741c0d2fd68c609dbc61596a7f.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/96e552788ed77468a19339c13019afeade7e00741c0d2fd68c609dbc61596a7f.jpg)

![cce4068624f40832016a696ed102cb667788bd8d1d1657607ec270a2ecf5a75d.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/cce4068624f40832016a696ed102cb667788bd8d1d1657607ec270a2ecf5a75d.jpg)

![d4ccf51099c9320a60b2bf891df60a3a09a56b4016c23319a67aca7cac9175aa.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/d4ccf51099c9320a60b2bf891df60a3a09a56b4016c23319a67aca7cac9175aa.jpg)

![ef2be69b9435974987201ea81fc3bb49ccb5b992c0ae97a882ad2d97a368655c.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/ef2be69b9435974987201ea81fc3bb49ccb5b992c0ae97a882ad2d97a368655c.jpg)

![ef472f93218ed31e0dec667b690b4c9bc98d2807752f55c04926a5bd58057e89.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/ef472f93218ed31e0dec667b690b4c9bc98d2807752f55c04926a5bd58057e89.jpg)

![f7a892590422a8a676dd81156ef2af4f2d40dd9ac25b8fca46d5f1ffa6e8417f.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/images/f7a892590422a8a676dd81156ef2af4f2d40dd9ac25b8fca46d5f1ffa6e8417f.jpg)

### Tables

![0819fb5a336a5429d84d23df5ec0a096233a90dd955493f0882b02af7e42ec44.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/0819fb5a336a5429d84d23df5ec0a096233a90dd955493f0882b02af7e42ec44.jpg)

![31497b3729c76641c847587cdc38f2b0883ec1e21544008c9db5e3333c4078cb.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/31497b3729c76641c847587cdc38f2b0883ec1e21544008c9db5e3333c4078cb.jpg)

![3b237a6e4e661a3968d0cb2c927bb37df65d25959451e78044c1501c09e593b7.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/3b237a6e4e661a3968d0cb2c927bb37df65d25959451e78044c1501c09e593b7.jpg)

![87712e3a2a70c62538312bfd7d7d72b137cf3dae6773ee2411a7615b30b37e28.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/87712e3a2a70c62538312bfd7d7d72b137cf3dae6773ee2411a7615b30b37e28.jpg)

![a53a40582f46782eba7679e30f3d143c532b105c0bd7ba2e7f34d7ac1fa24431.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/a53a40582f46782eba7679e30f3d143c532b105c0bd7ba2e7f34d7ac1fa24431.jpg)

![b9de1169dca61820d0a415b238ad62b7de9baadaf9b72e9d23f06bdb52b98315.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/b9de1169dca61820d0a415b238ad62b7de9baadaf9b72e9d23f06bdb52b98315.jpg)

![caa00dbb5bba9d00ef69de74fd3af6effd54f9c06cb1c6db7506087403263ed0.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/caa00dbb5bba9d00ef69de74fd3af6effd54f9c06cb1c6db7506087403263ed0.jpg)

![d6f41f36220aad97ea51212b1ff425cc8a105a5fd73f1c34a31013f21dbe5da5.jpg](../icml_results/4783_Coloring%20the%20Noise_%20Adversarial%20Sobolev%20Alignment%20for%20Faithful%20Image%20Super%20Resolution/tables/d6f41f36220aad97ea51212b1ff425cc8a105a5fd73f1c34a31013f21dbe5da5.jpg)

<a id="paper-4736"></a>

## Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO


### Images

![0998447cc753d3385f52b605c3730eedd5b1d2b2354e1a529452f6fced264ffc.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/0998447cc753d3385f52b605c3730eedd5b1d2b2354e1a529452f6fced264ffc.jpg)

![0d99011bd436b729e98458ea00e98c6f05dfca9b0d24ef1593de65e9e8f65fe8.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/0d99011bd436b729e98458ea00e98c6f05dfca9b0d24ef1593de65e9e8f65fe8.jpg)

![2927f973d8a95c763ee85d7272c99b9c3ee3eed8dbc889f18a194ee1ee01ba5d.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/2927f973d8a95c763ee85d7272c99b9c3ee3eed8dbc889f18a194ee1ee01ba5d.jpg)

![324312f7b3ce8a7157f89941039b4bad2391ce66423e551df9609cb47f2a595f.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/324312f7b3ce8a7157f89941039b4bad2391ce66423e551df9609cb47f2a595f.jpg)

![4c93e224b49991bcdf1164642e2f9dbddb848233b085915b99bd22a453ef7bff.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/4c93e224b49991bcdf1164642e2f9dbddb848233b085915b99bd22a453ef7bff.jpg)

![5244a4390fb533ba9543e630ae1aa0bceef9dbee4a9ef99b5a5939ee7c4d8bea.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/5244a4390fb533ba9543e630ae1aa0bceef9dbee4a9ef99b5a5939ee7c4d8bea.jpg)

![6e30959ea75511a3984eae65b77d0aa8c9f0b9816e8831bba1e52c60d357eebe.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/6e30959ea75511a3984eae65b77d0aa8c9f0b9816e8831bba1e52c60d357eebe.jpg)

![6e525732a4427db348ca9010d51f9b66c248df8ee88cc5ae0b379969aeec4f9e.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/6e525732a4427db348ca9010d51f9b66c248df8ee88cc5ae0b379969aeec4f9e.jpg)

![720fbfa196b3eb23a03bf138b4318f954cad1eece958d40884dab4629c57d9c0.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/720fbfa196b3eb23a03bf138b4318f954cad1eece958d40884dab4629c57d9c0.jpg)

![740e0759aca3b5ca8338c0f5b5fd64a96aec52c46e5dd4292d11ec1854686dad.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/740e0759aca3b5ca8338c0f5b5fd64a96aec52c46e5dd4292d11ec1854686dad.jpg)

![75abb4b6ed6a8368917aeb039f3dfcd87a295ac2de2b53f6fe2dc04766e15b5f.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/75abb4b6ed6a8368917aeb039f3dfcd87a295ac2de2b53f6fe2dc04766e15b5f.jpg)

![933af514de8671415e7e35be755e5746e98d57a49b88b19f7096bcc2ac602c1d.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/933af514de8671415e7e35be755e5746e98d57a49b88b19f7096bcc2ac602c1d.jpg)

![af3497ce954e32ec52e9ef511580e3f53f0fd3bc3e8b8cdaac5b0cbedf6f726b.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/af3497ce954e32ec52e9ef511580e3f53f0fd3bc3e8b8cdaac5b0cbedf6f726b.jpg)

![d31d750f52ef592f42cd235ebf961a618d501e595cbf57aac8420ff0a7e79580.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/d31d750f52ef592f42cd235ebf961a618d501e595cbf57aac8420ff0a7e79580.jpg)

![d3b53d6b7377ee6dab794aa15c0ead547b452bed384222ac6625963706cbeca9.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/d3b53d6b7377ee6dab794aa15c0ead547b452bed384222ac6625963706cbeca9.jpg)

![d3c21fe047c2a0ba2b88d906a46a087972a125e932720d5d9952ad37bd8d99a7.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/d3c21fe047c2a0ba2b88d906a46a087972a125e932720d5d9952ad37bd8d99a7.jpg)

![f27e99a9dfb695e332d413a53bba3803d65df078f0612826174117026908a2a3.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/f27e99a9dfb695e332d413a53bba3803d65df078f0612826174117026908a2a3.jpg)

![f5b2af8f2e1f1e54723c490bd4fa8c73b771aac920605d28342d6c45c5986385.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/images/f5b2af8f2e1f1e54723c490bd4fa8c73b771aac920605d28342d6c45c5986385.jpg)

### Tables

![0ef4e892380c2ac05cc8ba8ec9ce9d4cf6a20af80e57d723e68735ef9d063af0.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/0ef4e892380c2ac05cc8ba8ec9ce9d4cf6a20af80e57d723e68735ef9d063af0.jpg)

![26721b6c6238d452a9f10ec93391ab8db3ac29e94efa6fa264dd60024f2a5951.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/26721b6c6238d452a9f10ec93391ab8db3ac29e94efa6fa264dd60024f2a5951.jpg)

![3d23d3ae5217bee530e43413996d4b97c8b63a93a64c26f97d962d4d08b5a23a.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/3d23d3ae5217bee530e43413996d4b97c8b63a93a64c26f97d962d4d08b5a23a.jpg)

![5f0a970d6df8c6edf5c955552cebcaa973b919794983ddd2497cf67744a61645.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/5f0a970d6df8c6edf5c955552cebcaa973b919794983ddd2497cf67744a61645.jpg)

![669dcb4ed80064830c8c2245ae8281889d59dc3b753e482854622c432f225484.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/669dcb4ed80064830c8c2245ae8281889d59dc3b753e482854622c432f225484.jpg)

![8850ec3f65045eb9634f24cdfb5418801bda8839a13decfd555bb9dddc1e5c50.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/8850ec3f65045eb9634f24cdfb5418801bda8839a13decfd555bb9dddc1e5c50.jpg)

![90f0067c5345280fea8cc335fdec86a9d08a8806eabfc5fafbdce80d2bd8ee18.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/90f0067c5345280fea8cc335fdec86a9d08a8806eabfc5fafbdce80d2bd8ee18.jpg)

![9a4e7962b41ad75464a1e459da7f447f97c4ead0ee1b10063b6e82c93ab9b314.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/9a4e7962b41ad75464a1e459da7f447f97c4ead0ee1b10063b6e82c93ab9b314.jpg)

![9f05d4f168227b10bc18cc7aa56d884d4b25aae474be913de85512859abe3028.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/9f05d4f168227b10bc18cc7aa56d884d4b25aae474be913de85512859abe3028.jpg)

![ad60f2abc932bc24a4defb7463f1f3aa977e8bd5000ece81b9bb084731976f91.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/ad60f2abc932bc24a4defb7463f1f3aa977e8bd5000ece81b9bb084731976f91.jpg)

![b002afe5dadfbffafc572cac6e9812e086bd063fd2b95c511bfe5561e046f4a8.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/b002afe5dadfbffafc572cac6e9812e086bd063fd2b95c511bfe5561e046f4a8.jpg)

![e9c44ee9e0d61eb06bc1e2bf16ce2480156fc43077ac0c7d2670a67393888ff4.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/e9c44ee9e0d61eb06bc1e2bf16ce2480156fc43077ac0c7d2670a67393888ff4.jpg)

![f8f2c0c5f3ef865a99c0bdc2fc27bd349276ac81743897094491854ab3f8c539.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/f8f2c0c5f3ef865a99c0bdc2fc27bd349276ac81743897094491854ab3f8c539.jpg)

![fe64f4ea66c193644cde8685017c8c209b0c763c4adc26f2fc544a13634cf3af.jpg](../icml_results/4784_Why%20Tree-Style%20Branching%20Matters%20for%20Thought%20Advantage%20Estimation%20in%20GRPO/tables/fe64f4ea66c193644cde8685017c8c209b0c763c4adc26f2fc544a13634cf3af.jpg)

<a id="paper-4737"></a>

## TranX-Adapter: Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection


### Images

![00cd76bbbf2d6fcdaa2e886174ccf0b7eb4ab08638b0d2bcf12e2c849762bfcf.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/00cd76bbbf2d6fcdaa2e886174ccf0b7eb4ab08638b0d2bcf12e2c849762bfcf.jpg)

![2507eb437fe480db83e98481fead484bdcf88a1b2f2f2bee5f5f28bee967837e.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/2507eb437fe480db83e98481fead484bdcf88a1b2f2f2bee5f5f28bee967837e.jpg)

![4d4be768d103293b2948beaffd79de8bc08b803c3d7f43a2ca2d7ff0196a5dec.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/4d4be768d103293b2948beaffd79de8bc08b803c3d7f43a2ca2d7ff0196a5dec.jpg)

![625c201df93f69d2ae94e48e80f581c1e88ddef2535a5b3f02a02763458516e5.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/625c201df93f69d2ae94e48e80f581c1e88ddef2535a5b3f02a02763458516e5.jpg)

![8789770bb292e1ee14a4b5c3917d4a07c4a35bea241714160bec64ebde035592.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/8789770bb292e1ee14a4b5c3917d4a07c4a35bea241714160bec64ebde035592.jpg)

![a1c538f7798aae51ca2a75c82ce1b1f7fdcf44cefefc4317981f1e06cab66141.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/a1c538f7798aae51ca2a75c82ce1b1f7fdcf44cefefc4317981f1e06cab66141.jpg)

![a2b93fefe4d3150d618a693e255846e5f660478250344fd6b4480c248efb1463.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/a2b93fefe4d3150d618a693e255846e5f660478250344fd6b4480c248efb1463.jpg)

![a957a13ae70efc9a33b83a6ee9ffcdf6117daf40dc1f14aacc5483724fd4a353.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/a957a13ae70efc9a33b83a6ee9ffcdf6117daf40dc1f14aacc5483724fd4a353.jpg)

![c9988bebfe8938b1e9d6cd629d2f0f720a8c78d93ec782075cb9a3936aa6ab5e.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/images/c9988bebfe8938b1e9d6cd629d2f0f720a8c78d93ec782075cb9a3936aa6ab5e.jpg)

### Tables

![024b11591fe70322d1846561cc2258300ae0ba090a74f22828c7b75b2c6dcce3.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/024b11591fe70322d1846561cc2258300ae0ba090a74f22828c7b75b2c6dcce3.jpg)

![0a44aa30154e11fe9c9878d10bf0e136c02928a3442eb69c64be488804fb4960.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/0a44aa30154e11fe9c9878d10bf0e136c02928a3442eb69c64be488804fb4960.jpg)

![2063138632307d766132eadf1222a819743c69581c74d68739e8004ac9157b37.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/2063138632307d766132eadf1222a819743c69581c74d68739e8004ac9157b37.jpg)

![2e6a5d6fef6d8969e0cd87d2545cd8258c8d7cb7909ba4b5079617eb3be01bd6.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/2e6a5d6fef6d8969e0cd87d2545cd8258c8d7cb7909ba4b5079617eb3be01bd6.jpg)

![310f32391f45cea693172a3884da257704b1ddd5cddd9c8fe27fdccfa43d9280.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/310f32391f45cea693172a3884da257704b1ddd5cddd9c8fe27fdccfa43d9280.jpg)

![764d2e23674726a16fb4fdc3ceb5b152ebb8a3290abaa2630318cd16037739cb.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/764d2e23674726a16fb4fdc3ceb5b152ebb8a3290abaa2630318cd16037739cb.jpg)

![c8b34fa9b471e9635d4bc87b0395442e613a42dd259c6c76a0dd8bc25c0e87ac.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/c8b34fa9b471e9635d4bc87b0395442e613a42dd259c6c76a0dd8bc25c0e87ac.jpg)

![ce92687b916be555542a4434cfd72285910d368f1f7aedec6431f6b32f193c87.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/ce92687b916be555542a4434cfd72285910d368f1f7aedec6431f6b32f193c87.jpg)

![dd3eaf79f3d04e7f90628362ca21e6af2319feb2de7e8a46c0af12ff23c20cb8.jpg](../icml_results/4785_TranX-Adapter_%20Bridging%20Artifacts%20and%20Semantics%20within%20MLLMs%20for%20Robust%20AI-generated%20Image%20Detection/tables/dd3eaf79f3d04e7f90628362ca21e6af2319feb2de7e8a46c0af12ff23c20cb8.jpg)

<a id="paper-4738"></a>

## Large-scale Uncertainty Quantification for Latent Variable Models Using Subsampling Markov Chain Monte Carlo


### Images

![366a8982eb213c82e6a1df930525498b5f11295000edbc47c5faaaf80c80d326.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/366a8982eb213c82e6a1df930525498b5f11295000edbc47c5faaaf80c80d326.jpg)

![56cba45c347017ae0e4709fd13f08f7b1f4ec854265bbe738df15db8bf6633b4.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/56cba45c347017ae0e4709fd13f08f7b1f4ec854265bbe738df15db8bf6633b4.jpg)

![68903355fc7778662852b1115e416f9c693b98f6737a29761b3153cc53479347.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/68903355fc7778662852b1115e416f9c693b98f6737a29761b3153cc53479347.jpg)

![afb401e04025b6aa0ae50b865e557ad1eec6c793e9e53196ae241579900dfeff.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/afb401e04025b6aa0ae50b865e557ad1eec6c793e9e53196ae241579900dfeff.jpg)

![f49d8a5a11830e9095d27085b977ce75d1f55a8d84f4ce1440f7b5ca96640b41.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/f49d8a5a11830e9095d27085b977ce75d1f55a8d84f4ce1440f7b5ca96640b41.jpg)

![fe6436ff3cc938bad55583765222821d005307daadcc13a7aafcd500d16eed01.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/images/fe6436ff3cc938bad55583765222821d005307daadcc13a7aafcd500d16eed01.jpg)

### Tables

![2c4c25e462a8aa25a39c9c39d420be1b10dd705662a4dbfaecb6b2581006ff52.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/tables/2c4c25e462a8aa25a39c9c39d420be1b10dd705662a4dbfaecb6b2581006ff52.jpg)

![5d2fcfc4065545e81b591a3c0d61427a5bc985340b000330e4dc7dc3cdd5bdef.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/tables/5d2fcfc4065545e81b591a3c0d61427a5bc985340b000330e4dc7dc3cdd5bdef.jpg)

![ebc453fca67827e726d08b089f7669acaf95400c77d91ed5d9f48bc8603f44ad.jpg](../icml_results/4786_Large-scale%20Uncertainty%20Quantification%20for%20Latent%20Variable%20Models%20Using%20Subsampling%20Markov%20Chain%20Mon/tables/ebc453fca67827e726d08b089f7669acaf95400c77d91ed5d9f48bc8603f44ad.jpg)

<a id="paper-4739"></a>

## Learning $U$-Statistics with Active Inference


### Images

![15071767956ae3285737afa867bb5533a16ad1161c5d2e2af446fcfaa892e9f9.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/15071767956ae3285737afa867bb5533a16ad1161c5d2e2af446fcfaa892e9f9.jpg)

![1a8e3c161f1b3bd57f40239e32850dd5be470e70821bf1d739a059b34d63c2d3.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/1a8e3c161f1b3bd57f40239e32850dd5be470e70821bf1d739a059b34d63c2d3.jpg)

![577c4911c6977594c39cab97a5acefd404867423b7f18b955a5798472c80805a.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/577c4911c6977594c39cab97a5acefd404867423b7f18b955a5798472c80805a.jpg)

![95e1935fb8e31cdcd30a02a188f79baaabdc7e9e6a670e9a969876272701aca7.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/95e1935fb8e31cdcd30a02a188f79baaabdc7e9e6a670e9a969876272701aca7.jpg)

![9b69de412e9fd3533f9647b19101e8b55fb8ccb456ded6c27217f8532bb47e35.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/9b69de412e9fd3533f9647b19101e8b55fb8ccb456ded6c27217f8532bb47e35.jpg)

![de0a4ef458c2af02c68e7acc635a4fdbcb1b324ef7828952a85aaa71bb5af862.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/de0a4ef458c2af02c68e7acc635a4fdbcb1b324ef7828952a85aaa71bb5af862.jpg)

![ef8ecc0bc28a7da01d734af44aae4e2f09c0b58e8fd48b5fd54e5466e81fc1ee.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/images/ef8ecc0bc28a7da01d734af44aae4e2f09c0b58e8fd48b5fd54e5466e81fc1ee.jpg)

### Tables

![3f443d81cd51413e5032240a75fcfb3d390b65d4de5a7962cd8196899ba93703.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/tables/3f443d81cd51413e5032240a75fcfb3d390b65d4de5a7962cd8196899ba93703.jpg)

![98a39ff8579ad712b3b26907ec7ef56d6af20d47cc14cd849104fd941dfd4e83.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/tables/98a39ff8579ad712b3b26907ec7ef56d6af20d47cc14cd849104fd941dfd4e83.jpg)

![c8caaa7b6c26d892b9a09ebc502338d8e289ed4aa57438a3c48a21439893cb25.jpg](../icml_results/4787_Learning%20%24U%24-Statistics%20with%20Active%20Inference/tables/c8caaa7b6c26d892b9a09ebc502338d8e289ed4aa57438a3c48a21439893cb25.jpg)

<a id="paper-4740"></a>

## The Truth Lies Somewhere in the Middle (of the Generated Tokens)


### Images

![05369c562e6674a473c48079a0d58e2113925819acbf370bc758709a38b5c663.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/05369c562e6674a473c48079a0d58e2113925819acbf370bc758709a38b5c663.jpg)

![110d1fa73727d1d60f172a00a9c070c8429b227b1f3f8d0b7df22ab02662cbe8.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/110d1fa73727d1d60f172a00a9c070c8429b227b1f3f8d0b7df22ab02662cbe8.jpg)

![205140ce1c93c7a3f76d78de2190d9dab67d64031fce4be6574b1e36c063acc7.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/205140ce1c93c7a3f76d78de2190d9dab67d64031fce4be6574b1e36c063acc7.jpg)

![2074e145cf961a4702e45d2cb2b8b8afa5fb8099e2d69db3e9bd5958a516db87.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/2074e145cf961a4702e45d2cb2b8b8afa5fb8099e2d69db3e9bd5958a516db87.jpg)

![21c903b80abb260d52ed1d5a8385d33eb92742e7d40b0e42d0aae931772ebfc3.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/21c903b80abb260d52ed1d5a8385d33eb92742e7d40b0e42d0aae931772ebfc3.jpg)

![22495a9d1b0684b6612ae75fa0ecf27c4bcf7395b9c5c2d9c16009878b033bd6.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/22495a9d1b0684b6612ae75fa0ecf27c4bcf7395b9c5c2d9c16009878b033bd6.jpg)

![439fe87c7b2943430beffecd9c038f760b8bb93ffbf992b7d628f8c0ee0f3e02.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/439fe87c7b2943430beffecd9c038f760b8bb93ffbf992b7d628f8c0ee0f3e02.jpg)

![460bb4a9c28762ad3602c54052361efbbda0b8054e6220a1a2ed8507b36fec0d.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/460bb4a9c28762ad3602c54052361efbbda0b8054e6220a1a2ed8507b36fec0d.jpg)

![501a8397ec717e8bd32b9a02ea5bc170250604148b346568b3758817729dec7b.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/501a8397ec717e8bd32b9a02ea5bc170250604148b346568b3758817729dec7b.jpg)

![55f4e144c72a3a458bdb104d74de51ffe89448a898cfbf075e1899cb19347733.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/55f4e144c72a3a458bdb104d74de51ffe89448a898cfbf075e1899cb19347733.jpg)

![658d03674e8536ef924cc72a14be1ee8c5c750f161b47791faf043b4371f1f88.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/658d03674e8536ef924cc72a14be1ee8c5c750f161b47791faf043b4371f1f88.jpg)

![68678348e831ccdc7ae8529f0a8cfd8dab721c9288cf661d5598a651caecc2cd.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/68678348e831ccdc7ae8529f0a8cfd8dab721c9288cf661d5598a651caecc2cd.jpg)

![7099305c44b98abb371db50d2b24a3a42c76c71ad66561248d7f51360944f2b1.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/7099305c44b98abb371db50d2b24a3a42c76c71ad66561248d7f51360944f2b1.jpg)

![76003bd9c7195b13f656d28160bdf69e10a8a73ea1b889caaf7d52bf6ec5e4ea.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/76003bd9c7195b13f656d28160bdf69e10a8a73ea1b889caaf7d52bf6ec5e4ea.jpg)

![883100385c1e653bfd131238e878a5f0271d519a767c08100d1f1449a7f6bfa9.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/883100385c1e653bfd131238e878a5f0271d519a767c08100d1f1449a7f6bfa9.jpg)

![a6ef7e2a868bd2222c549955243666fd18e93a9179b192861254732472266f9b.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/a6ef7e2a868bd2222c549955243666fd18e93a9179b192861254732472266f9b.jpg)

![b50a692b34b5a34abdcf9f41cebbd4ddb3899863cd47455e24473f8594640ac0.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/b50a692b34b5a34abdcf9f41cebbd4ddb3899863cd47455e24473f8594640ac0.jpg)

![ba4722264f62928108b066277daa15e52e94ba7b947acd6232716a15d973ddf2.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/ba4722264f62928108b066277daa15e52e94ba7b947acd6232716a15d973ddf2.jpg)

![c068784dc645ea23e31965ecfa65cc76d9bb13ba31df363839334244343d6a06.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/c068784dc645ea23e31965ecfa65cc76d9bb13ba31df363839334244343d6a06.jpg)

![c794d521ae528a06afec007a81d5a577999d61d65e878c872abdb37f18cdf9aa.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/c794d521ae528a06afec007a81d5a577999d61d65e878c872abdb37f18cdf9aa.jpg)

![dfc0a96f8753fc4ee1c16086fee067056fd1412563ffa934b464fc189e5e677b.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/dfc0a96f8753fc4ee1c16086fee067056fd1412563ffa934b464fc189e5e677b.jpg)

![e268203cce2a2ded21793f8bd1129e476908b0b9895f7280350a92a8f1cfee53.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/e268203cce2a2ded21793f8bd1129e476908b0b9895f7280350a92a8f1cfee53.jpg)

![eef5a87910caa54df22ad3ca829310c22943253a83019f29d053d087db08eedd.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/eef5a87910caa54df22ad3ca829310c22943253a83019f29d053d087db08eedd.jpg)

![f9e645f87346fa6c4cf23ab418ef559216962fa5e40dbb9e228d6b28785ece51.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/f9e645f87346fa6c4cf23ab418ef559216962fa5e40dbb9e228d6b28785ece51.jpg)

![fbe32ce30c8156e85846240c32c4ae42f6addd202e1cc487a28c7012bdfaef91.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/fbe32ce30c8156e85846240c32c4ae42f6addd202e1cc487a28c7012bdfaef91.jpg)

![ff1b148eff36e98a720cfd8e27954d9881563c0fad513df4d2f60c46e086c68c.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/images/ff1b148eff36e98a720cfd8e27954d9881563c0fad513df4d2f60c46e086c68c.jpg)

### Tables

![14d2618d3bf7ab49cdf302c721d1c125ba3910e029a2daa72a203c74573ae23b.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/tables/14d2618d3bf7ab49cdf302c721d1c125ba3910e029a2daa72a203c74573ae23b.jpg)

![9185000fdb634e81dd6c3f60795f85d4f5d2a25b426747dbdf74d141e60bebdc.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/tables/9185000fdb634e81dd6c3f60795f85d4f5d2a25b426747dbdf74d141e60bebdc.jpg)

![b787fdb821a591a1af28fc1c92351803e4c13095ae04888b4a3642e8eb008efe.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/tables/b787fdb821a591a1af28fc1c92351803e4c13095ae04888b4a3642e8eb008efe.jpg)

![eba9c3cee6e69bd0052e4fcee2018b4985c7a608ab93601dbae0104c4fcce826.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/tables/eba9c3cee6e69bd0052e4fcee2018b4985c7a608ab93601dbae0104c4fcce826.jpg)

![f9430a53d97dd3f894ebd8f13a132accdb3f53cec88de355a60df5437f067111.jpg](../icml_results/4788_The%20Truth%20Lies%20Somewhere%20in%20the%20Middle%20%28of%20the%20Generated%20Tokens%29/tables/f9430a53d97dd3f894ebd8f13a132accdb3f53cec88de355a60df5437f067111.jpg)

<a id="paper-4741"></a>

## PhaseAlign: Complex Phase Alignment for Stable Open-Vocabulary Semantic Segmentation


### Images

![24201fe570337d19a88efa7151fed959a76cdc6584f3c4fbe5106545fcb3684d.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/24201fe570337d19a88efa7151fed959a76cdc6584f3c4fbe5106545fcb3684d.jpg)

![4d76f0edd2d9906f28a3501d0cc55343ff7d0728447a05864dfa3dd6112a3052.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/4d76f0edd2d9906f28a3501d0cc55343ff7d0728447a05864dfa3dd6112a3052.jpg)

![54c6c0974c9f7fb98b70c767798bd76bea80d29edb58ca043270ce7948b79a79.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/54c6c0974c9f7fb98b70c767798bd76bea80d29edb58ca043270ce7948b79a79.jpg)

![66bbe9b313bbcd0bf3775c8b351821fade395c5361a6f0bedfc2eba4346612cd.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/66bbe9b313bbcd0bf3775c8b351821fade395c5361a6f0bedfc2eba4346612cd.jpg)

![80ba2713d5bc3475c2d7dec998b3fa0d737679a6aa15917fd85250383cb51216.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/80ba2713d5bc3475c2d7dec998b3fa0d737679a6aa15917fd85250383cb51216.jpg)

![f7e40629f0c028c7b5779614227a219067361fea8d9ac4eb6ac735b1d3a45a1f.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/images/f7e40629f0c028c7b5779614227a219067361fea8d9ac4eb6ac735b1d3a45a1f.jpg)

### Tables

![0cec2d355597c9b34d391d7cb4edfd7bcf797b61cba72bdb22e86011430a07c8.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/0cec2d355597c9b34d391d7cb4edfd7bcf797b61cba72bdb22e86011430a07c8.jpg)

![1b1c2a493d7851d49b0bb754aa210197356351721e5cf1add22999dcea663440.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/1b1c2a493d7851d49b0bb754aa210197356351721e5cf1add22999dcea663440.jpg)

![3cdc5e7b10c5b01d2409ae3edb2e42ba3ebedcee3848180e666ce1b63f5fab82.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/3cdc5e7b10c5b01d2409ae3edb2e42ba3ebedcee3848180e666ce1b63f5fab82.jpg)

![4c02d2cd414b76e8de1f0b9ed822997658eafbb71764d1509c8d31740647809c.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/4c02d2cd414b76e8de1f0b9ed822997658eafbb71764d1509c8d31740647809c.jpg)

![6b901f95c2a0789267d3f9af01a6e7c985770e75ec65bb9e832d52daa4502c06.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/6b901f95c2a0789267d3f9af01a6e7c985770e75ec65bb9e832d52daa4502c06.jpg)

![6e3a028aefe103c9278050525a42558e9b41e9023275c3ce787b55377e87df5e.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/6e3a028aefe103c9278050525a42558e9b41e9023275c3ce787b55377e87df5e.jpg)

![7920c25e095ab5404bb8048343b57259907513ef5d02b4854cd29a26be630ad1.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/7920c25e095ab5404bb8048343b57259907513ef5d02b4854cd29a26be630ad1.jpg)

![bdc96a06ac8eca1a7337001ffc557e3565703c1f08881687df73cdc417f4adbf.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/bdc96a06ac8eca1a7337001ffc557e3565703c1f08881687df73cdc417f4adbf.jpg)

![f347e2403368836162addfc90b1430a9ee649f445ba8836e8c683b7c0be35336.jpg](../icml_results/4789_PhaseAlign_%20Complex%20Phase%20Alignment%20for%20Stable%20Open-Vocabulary%20Semantic%20Segmentation/tables/f347e2403368836162addfc90b1430a9ee649f445ba8836e8c683b7c0be35336.jpg)

<a id="paper-4742"></a>

## ProtoVAR: Efficient Dataset Distillation via Prototype-Guided Visual Autoregressive Modeling


### Images

![0c11c7eb5fcd367e9e84d3f051a03a46dff055863946ce01446e6278befad4c5.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/0c11c7eb5fcd367e9e84d3f051a03a46dff055863946ce01446e6278befad4c5.jpg)

![1611a2024b675049f0430e76b381be56424e36b687e038f1007fc3ab8ad8e7f0.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/1611a2024b675049f0430e76b381be56424e36b687e038f1007fc3ab8ad8e7f0.jpg)

![1f803721cdf0444d0c3925b0f052659b2dc622f0a1c6801fdb511cdab380ff02.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/1f803721cdf0444d0c3925b0f052659b2dc622f0a1c6801fdb511cdab380ff02.jpg)

![23d2aeb1bb74ef78a7bde308852475d5a9eba0d1f12e9abd43102c748360a102.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/23d2aeb1bb74ef78a7bde308852475d5a9eba0d1f12e9abd43102c748360a102.jpg)

![3b363459c2d62b3a2874e65d82122a111314cc8170022db0235ea1822018f022.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/3b363459c2d62b3a2874e65d82122a111314cc8170022db0235ea1822018f022.jpg)

![50a156019084f926eafab8e919d3ff4779cbbd83da9a6e5fa7a2bbd914376197.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/50a156019084f926eafab8e919d3ff4779cbbd83da9a6e5fa7a2bbd914376197.jpg)

![62c17a4b3742c29c7793c0516eeba358bb612016a9afc8414c9c419a2a3c2050.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/62c17a4b3742c29c7793c0516eeba358bb612016a9afc8414c9c419a2a3c2050.jpg)

![6750a9bccc8adbd638015fbb94530beebc631668f2e8637a17edd2655f33b81d.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/6750a9bccc8adbd638015fbb94530beebc631668f2e8637a17edd2655f33b81d.jpg)

![6bb828a7c3778d2f031f00566cc2d02609850803c5715d75d27112999179f304.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/6bb828a7c3778d2f031f00566cc2d02609850803c5715d75d27112999179f304.jpg)

![7474842f6d1e30813377749a417365b572e565fea0d68d931f90193d0ff57cec.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/7474842f6d1e30813377749a417365b572e565fea0d68d931f90193d0ff57cec.jpg)

![789ecd5e856f31a6def369a5cfca9bca7d509b96bd5d8e4ebbda5ac2012d119a.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/789ecd5e856f31a6def369a5cfca9bca7d509b96bd5d8e4ebbda5ac2012d119a.jpg)

![7f83c67ebefe7af5712d1e440369f7a573021049cf8269905625e48ce435c29a.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/7f83c67ebefe7af5712d1e440369f7a573021049cf8269905625e48ce435c29a.jpg)

![88a80bf7bd06fc2e81a56447bfd636d9101917bf56934751ab91218acc386ed4.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/88a80bf7bd06fc2e81a56447bfd636d9101917bf56934751ab91218acc386ed4.jpg)

![a204e910a4c2937bc173bbe124a9d3b308fc5a1b748c0cdf920782a311af463e.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/a204e910a4c2937bc173bbe124a9d3b308fc5a1b748c0cdf920782a311af463e.jpg)

![c2b3dcc13de0dffdfa9ff2b99074fe89e956527cbaaaa283b473c8c973c5d759.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/c2b3dcc13de0dffdfa9ff2b99074fe89e956527cbaaaa283b473c8c973c5d759.jpg)

![ced6fc4c9a6b2acfdf385563b89147d569daab89c5b21950b208e741e670f7be.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/ced6fc4c9a6b2acfdf385563b89147d569daab89c5b21950b208e741e670f7be.jpg)

![d3fbafb8fee06defe83a489973609d3a46b4abbe7d912f959a5deed34b70f4d1.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/d3fbafb8fee06defe83a489973609d3a46b4abbe7d912f959a5deed34b70f4d1.jpg)

![d8077a23e71ba168dcc496f3dd7a8e8d5d30606f3bf5363685eb5a2a4e748e56.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/d8077a23e71ba168dcc496f3dd7a8e8d5d30606f3bf5363685eb5a2a4e748e56.jpg)

![d8f84948adf815a00d8660b1eeb410ce12dd99dcb570e7b59a71ce95981c7266.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/d8f84948adf815a00d8660b1eeb410ce12dd99dcb570e7b59a71ce95981c7266.jpg)

![e169fde7d93c60f6fb013b6df9c4f65f0a12824bd0b46b74b35fcb0efa0caf5c.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/e169fde7d93c60f6fb013b6df9c4f65f0a12824bd0b46b74b35fcb0efa0caf5c.jpg)

![e5650bd35254c381fa800662644966e1f267d2c757e119e274d7f126a9c1f469.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/images/e5650bd35254c381fa800662644966e1f267d2c757e119e274d7f126a9c1f469.jpg)

### Tables

![05f4a62bfb8aa960854e3cf4a5fc07bb2071bf69104ee347b26497505d1a6690.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/05f4a62bfb8aa960854e3cf4a5fc07bb2071bf69104ee347b26497505d1a6690.jpg)

![0c41a0115e49e71ad3384e33181d592c94f8ffea13d71461c4189f71d4f10763.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/0c41a0115e49e71ad3384e33181d592c94f8ffea13d71461c4189f71d4f10763.jpg)

![11ff78d486c2059dc27a16fe2a16dbfa160267c151fc0dff85e90a75541a1b2e.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/11ff78d486c2059dc27a16fe2a16dbfa160267c151fc0dff85e90a75541a1b2e.jpg)

![255b19b0490862654f6500fd6e0b6102f431c3c2e2eede474f1680bfda7957f9.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/255b19b0490862654f6500fd6e0b6102f431c3c2e2eede474f1680bfda7957f9.jpg)

![2829bcb38f26dfadf8cebb4e1594e3e932b09f47d69ddc897b478ed095c8d3af.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/2829bcb38f26dfadf8cebb4e1594e3e932b09f47d69ddc897b478ed095c8d3af.jpg)

![35ecf7f9962c36a400b5516dafc8a5aa5db76ea1b9fe228f80cb5f3b28cfd387.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/35ecf7f9962c36a400b5516dafc8a5aa5db76ea1b9fe228f80cb5f3b28cfd387.jpg)

![809f47de66bdc53a402a2f42a78255a6fe04bb061ed8d33317eee2321901b3db.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/809f47de66bdc53a402a2f42a78255a6fe04bb061ed8d33317eee2321901b3db.jpg)

![81269092c1aeb80bd9cec3c33efbb4ee68c56f95248cea5c27f58fe2b2d984e2.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/81269092c1aeb80bd9cec3c33efbb4ee68c56f95248cea5c27f58fe2b2d984e2.jpg)

![929b62c1bef1347a874ff39457e88fa8870cc82d897cebdaf09d50c31ba09e8e.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/929b62c1bef1347a874ff39457e88fa8870cc82d897cebdaf09d50c31ba09e8e.jpg)

![a092b39b1e2fa65bc8e4b4de214287088e44ec7c2886f0811a4d30226b8926db.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/a092b39b1e2fa65bc8e4b4de214287088e44ec7c2886f0811a4d30226b8926db.jpg)

![b36330253d5596c35b3b7d075b847814f86fb29b3a5be5500d3064d044d56b3e.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/b36330253d5596c35b3b7d075b847814f86fb29b3a5be5500d3064d044d56b3e.jpg)

![bb77e005da91cca86c3543fdfeb9f054473ef51ad81bd2eb81d6139fbda1903b.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/bb77e005da91cca86c3543fdfeb9f054473ef51ad81bd2eb81d6139fbda1903b.jpg)

![bf95c98b19d3eae265f9fb26e1f0cdf26d515530eb9f3beb200eb2286df23fae.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/bf95c98b19d3eae265f9fb26e1f0cdf26d515530eb9f3beb200eb2286df23fae.jpg)

![d54cca83cf146bdd774c043885a6ef1947688027c56fb2fe08bf9db9617c1aba.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/d54cca83cf146bdd774c043885a6ef1947688027c56fb2fe08bf9db9617c1aba.jpg)

![db64e8e5ef7d6625ab4b6b874d1880462e4d600a1c4b506d41ac118879243b23.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/db64e8e5ef7d6625ab4b6b874d1880462e4d600a1c4b506d41ac118879243b23.jpg)

![e12d06d1c9e9368802bbab6ee8425debe3635ad6b93c669f80d904b66ede71ce.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/e12d06d1c9e9368802bbab6ee8425debe3635ad6b93c669f80d904b66ede71ce.jpg)

![f105b9a905a95425c5cf8114c85e5a4809788218cbdaff8d07c71175e11620b9.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/f105b9a905a95425c5cf8114c85e5a4809788218cbdaff8d07c71175e11620b9.jpg)

![ff7e0a9f0df2781bdd3c309d8902134d74f2a0e27586d5313367d0951a40d9b6.jpg](../icml_results/4790_ProtoVAR_%20Efficient%20Dataset%20Distillation%20via%20Prototype-Guided%20Visual%20Autoregressive%20Modeling/tables/ff7e0a9f0df2781bdd3c309d8902134d74f2a0e27586d5313367d0951a40d9b6.jpg)

<a id="paper-4743"></a>

## ScDiVa: Masked Discrete Diffusion for Joint Modeling of Single-Cell Identity and Expression


### Images

![12ad727463268511e3b5232cc0d83abc6fb309e7a58c9e1830a71d5917c68378.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/12ad727463268511e3b5232cc0d83abc6fb309e7a58c9e1830a71d5917c68378.jpg)

![2df0d6c2a21c513a4f4c5558526ab331f03e091b60413c067b9329ce5bac6ac6.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/2df0d6c2a21c513a4f4c5558526ab331f03e091b60413c067b9329ce5bac6ac6.jpg)

![4c2378ceffcd94056f258350f2c238ae3db4b896c682953ff7cc4aaf7015bc03.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/4c2378ceffcd94056f258350f2c238ae3db4b896c682953ff7cc4aaf7015bc03.jpg)

![6f60ecc407c535481811c4ae8b8ed580da5b97442f93567fcccd4bd7728e1873.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/6f60ecc407c535481811c4ae8b8ed580da5b97442f93567fcccd4bd7728e1873.jpg)

![6fe373b8e203034158fce6ef875e7e7814c3631dcab39ab7099b52ae04f7627e.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/6fe373b8e203034158fce6ef875e7e7814c3631dcab39ab7099b52ae04f7627e.jpg)

![71cd119e55b16c4229330cfb77a9d49c303becdb11f43746fca8943d0eabedfa.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/71cd119e55b16c4229330cfb77a9d49c303becdb11f43746fca8943d0eabedfa.jpg)

![b69393f2a3decc4ea4635876ac1a73af18aeadf66d64a4f64a76baacb8157ad0.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/b69393f2a3decc4ea4635876ac1a73af18aeadf66d64a4f64a76baacb8157ad0.jpg)

![c46652afc94a12220707c029c4d7167cb0a4562b445121e963594c3a2ecdd903.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/c46652afc94a12220707c029c4d7167cb0a4562b445121e963594c3a2ecdd903.jpg)

![d4a4351ee4a9953f7054e0c7229cb8e9a89203b07049acad7e178bf2b751c00a.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/d4a4351ee4a9953f7054e0c7229cb8e9a89203b07049acad7e178bf2b751c00a.jpg)

![dbbc3eb9e14160975d7793d6518902835aabac85f2d09ce82ab3a2cde42c8db8.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/dbbc3eb9e14160975d7793d6518902835aabac85f2d09ce82ab3a2cde42c8db8.jpg)

![f33bf0aa4d683035fc5cb6a90f34fb193052fb7d7bcab59bfa113b351636cb8f.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/images/f33bf0aa4d683035fc5cb6a90f34fb193052fb7d7bcab59bfa113b351636cb8f.jpg)

### Tables

![01c3e0fcf2a9c07133321b716f480ed883148ad1e94aa77be57969f9a6f157c7.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/01c3e0fcf2a9c07133321b716f480ed883148ad1e94aa77be57969f9a6f157c7.jpg)

![21a00254580f083daf3fa1812175e56d547ffd23c9044d93c7f85eefe9d9a6cc.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/21a00254580f083daf3fa1812175e56d547ffd23c9044d93c7f85eefe9d9a6cc.jpg)

![2610565b777344956ddac7aae0e931e61bc0ad14107cfc4a13ee55f41c42c59d.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/2610565b777344956ddac7aae0e931e61bc0ad14107cfc4a13ee55f41c42c59d.jpg)

![28d5091c6276bf1d0917364d9ef046cb74093afc91640f0d542791306a1720d6.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/28d5091c6276bf1d0917364d9ef046cb74093afc91640f0d542791306a1720d6.jpg)

![2e84bba5e0b8e7bb51366ef2a9819cacac36e9bc2a823807b656757e4e193988.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/2e84bba5e0b8e7bb51366ef2a9819cacac36e9bc2a823807b656757e4e193988.jpg)

![3b35e5e503fad7aef3c5725e1b079f3c702cb3150a05aec4a29cf1315f1df1cf.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/3b35e5e503fad7aef3c5725e1b079f3c702cb3150a05aec4a29cf1315f1df1cf.jpg)

![617ec72f72edd9b99a19d737861950a428ad577d039ec3aed03b8058a6fe977e.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/617ec72f72edd9b99a19d737861950a428ad577d039ec3aed03b8058a6fe977e.jpg)

![7dc3a9549e20ca99ef3881afd459608590542b50261d16aaeaa4af51657ba229.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/7dc3a9549e20ca99ef3881afd459608590542b50261d16aaeaa4af51657ba229.jpg)

![7fa1ba8f89923a6b36474642ef084a5c21b56175179b3891bc7b59eb0c060d97.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/7fa1ba8f89923a6b36474642ef084a5c21b56175179b3891bc7b59eb0c060d97.jpg)

![871bc940153e4e2e23df52075b33afa073779317b30d4661e9090c8a019db53a.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/871bc940153e4e2e23df52075b33afa073779317b30d4661e9090c8a019db53a.jpg)

![b5e1716266e4b6143effbd19096cadad191985db7993a077e46216aafaaa16b3.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/b5e1716266e4b6143effbd19096cadad191985db7993a077e46216aafaaa16b3.jpg)

![b9367157929c4847f3c5da92cce1d06883762cb4579d66071faea06dc0b84e07.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/b9367157929c4847f3c5da92cce1d06883762cb4579d66071faea06dc0b84e07.jpg)

![d2e0bd422584b1add1c90889e426ad1373e4a9302a76f9b4a4bf5ef1000052de.jpg](../icml_results/4791_ScDiVa_%20Masked%20Discrete%20Diffusion%20for%20Joint%20Modeling%20of%20Single-Cell%20Identity%20and%20Expression/tables/d2e0bd422584b1add1c90889e426ad1373e4a9302a76f9b4a4bf5ef1000052de.jpg)

<a id="paper-4744"></a>

## GAM-RAG: Gain-Adaptive Memory for Evolving Retrieval in Retrieval-Augmented Generation


### Images

![1f39846138da27ef738743ed20f7aa6850be60b3e5659966ca57a80617a9c40f.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/1f39846138da27ef738743ed20f7aa6850be60b3e5659966ca57a80617a9c40f.jpg)

![985fc66058cb7a3e83488e451aa1c20a633812ab279a513b41da054b4808b0af.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/985fc66058cb7a3e83488e451aa1c20a633812ab279a513b41da054b4808b0af.jpg)

![b26b38c168c1c9445fde02fcbdae76e0231b7081854aee09bd034bffec064f3e.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/b26b38c168c1c9445fde02fcbdae76e0231b7081854aee09bd034bffec064f3e.jpg)

![c7729b5d4b4b9c6596b132d166cc2f1714e2c1612a292934e943ca33777fc793.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/c7729b5d4b4b9c6596b132d166cc2f1714e2c1612a292934e943ca33777fc793.jpg)

![d068934e653f1e786b1b3483cfb55d865ad7d3febe1e2ae3fe967cb6ea980a0d.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/d068934e653f1e786b1b3483cfb55d865ad7d3febe1e2ae3fe967cb6ea980a0d.jpg)

![dd16c13a444460b3a120816ff8569d8556712d9cf78dd4077ca503244df5ea21.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/images/dd16c13a444460b3a120816ff8569d8556712d9cf78dd4077ca503244df5ea21.jpg)

### Tables

![318ff7f395407482bfb3404838fa42355652da3012cd2378c832d0f3ad679f04.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/318ff7f395407482bfb3404838fa42355652da3012cd2378c832d0f3ad679f04.jpg)

![51b138c2ed76def201bb5dfc9c5727f8c353ba15c75c40e965861f47dcaf11d6.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/51b138c2ed76def201bb5dfc9c5727f8c353ba15c75c40e965861f47dcaf11d6.jpg)

![6454c652ff9b87ee2da0c74cf69b7121f159a45a3c82938975fb44a15a2e9e9a.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/6454c652ff9b87ee2da0c74cf69b7121f159a45a3c82938975fb44a15a2e9e9a.jpg)

![c0725ecc28143b8f9ddfdfd3163d0486f4378079140b2ec95a964c675f06aab6.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/c0725ecc28143b8f9ddfdfd3163d0486f4378079140b2ec95a964c675f06aab6.jpg)

![c443225124717659b32d5687835a36f108d7dec3d961ded0f216af2151426d4b.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/c443225124717659b32d5687835a36f108d7dec3d961ded0f216af2151426d4b.jpg)

![d8fa61f58d3987dc3d321500392ce71086cedb0c828852314c1b1724b6fdfa7e.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/d8fa61f58d3987dc3d321500392ce71086cedb0c828852314c1b1724b6fdfa7e.jpg)

![dc78916cd4ab291b1ef6e65a6a5c6aeea6cb750b3d112b61ddfdd8ba7ab07124.jpg](../icml_results/4792_GAM-RAG_%20Gain-Adaptive%20Memory%20for%20Evolving%20Retrieval%20in%20Retrieval-Augmented%20Generation/tables/dc78916cd4ab291b1ef6e65a6a5c6aeea6cb750b3d112b61ddfdd8ba7ab07124.jpg)

<a id="paper-4745"></a>

## DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders


### Images

![17aeb5e7779086eafb44e43ce967b4cd7162000729749f8f2f6a184d3e545564.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/17aeb5e7779086eafb44e43ce967b4cd7162000729749f8f2f6a184d3e545564.jpg)

![1bc8fdedaf67a329fe994b6d624efc49e1460b791f2f95dd7e2183088d30f2a4.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/1bc8fdedaf67a329fe994b6d624efc49e1460b791f2f95dd7e2183088d30f2a4.jpg)

![335f0723991e56e9854ec50556755e7a6a97c1fbf02734a927b6a232ec90b5b7.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/335f0723991e56e9854ec50556755e7a6a97c1fbf02734a927b6a232ec90b5b7.jpg)

![4411c41b110c03bace5b65d5fe28c90465ad12032f2c96e171b86c3a21cb66fe.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/4411c41b110c03bace5b65d5fe28c90465ad12032f2c96e171b86c3a21cb66fe.jpg)

![4935204cd3d030ba3877b05bdb964cbd184df516af0bca5cbed7b4ebb513bd45.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/4935204cd3d030ba3877b05bdb964cbd184df516af0bca5cbed7b4ebb513bd45.jpg)

![49fb1277bb30c7d6caed68924f42015458b370e136a4169aeab620f7cfa7b48e.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/49fb1277bb30c7d6caed68924f42015458b370e136a4169aeab620f7cfa7b48e.jpg)

![72837598eb9fe88dfcefdf55a508b83af7c14291b5f5e6855362794a4533018a.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/72837598eb9fe88dfcefdf55a508b83af7c14291b5f5e6855362794a4533018a.jpg)

![86d8232fccd2c6302da1fab21d7d3072527501bc68d0ee671081d9b60cbc90e3.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/86d8232fccd2c6302da1fab21d7d3072527501bc68d0ee671081d9b60cbc90e3.jpg)

![bb3c34974a57d16eea234b096888f0ea51f62ef23a7b7ba1625e24726f15dd6c.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/bb3c34974a57d16eea234b096888f0ea51f62ef23a7b7ba1625e24726f15dd6c.jpg)

![d89227162b0fb05f01bf916667430a0174164a1a9d205f0032d8d8724166b3c1.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/d89227162b0fb05f01bf916667430a0174164a1a9d205f0032d8d8724166b3c1.jpg)

![f06ada0a552caa5f34b0618a1070b2712c7021a32f67609b33839214946d5bba.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/f06ada0a552caa5f34b0618a1070b2712c7021a32f67609b33839214946d5bba.jpg)

![f2df26341af6345bf76f5dd4eef89fd046c1d19203f0615b2a270995b3a5fb37.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/images/f2df26341af6345bf76f5dd4eef89fd046c1d19203f0615b2a270995b3a5fb37.jpg)

### Tables

![726bbf976c2284795e238e359610e2a15a04633c96e144349c87b6e47d5b39e4.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/tables/726bbf976c2284795e238e359610e2a15a04633c96e144349c87b6e47d5b39e4.jpg)

![81c338c58d9445cf509d3588689a7b76d366a93048c8533b86038f32d910acee.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/tables/81c338c58d9445cf509d3588689a7b76d366a93048c8533b86038f32d910acee.jpg)

![a25b034818333e934645cebc9147989b6ae6664198fa6e20951d48775018fc0d.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/tables/a25b034818333e934645cebc9147989b6ae6664198fa6e20951d48775018fc0d.jpg)

![be50796fb8812d4de1c0bb6afcd6fceda992599aeb9b00ab5d799f22f958a947.jpg](../icml_results/4793_DLM-Scope_%20Mechanistic%20Interpretability%20of%20Diffusion%20Language%20Models%20via%20Sparse%20Autoencoders/tables/be50796fb8812d4de1c0bb6afcd6fceda992599aeb9b00ab5d799f22f958a947.jpg)

<a id="paper-4746"></a>

## PESD-TSF: A Period-Aware and Explicit Structured Decomposition Framework for Long-Term Time Series Forecasting


### Images

![1ffafa127753b7664c7f73c073bcd56aed1e09ed33cb80cdd9f3956572b884bd.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/1ffafa127753b7664c7f73c073bcd56aed1e09ed33cb80cdd9f3956572b884bd.jpg)

![31841bf7c2eb5aa6b9399d010029c672f52504c9c36230fc82fb052546407cba.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/31841bf7c2eb5aa6b9399d010029c672f52504c9c36230fc82fb052546407cba.jpg)

![473476287cbe3d02d231d7bd947fbff95bb60490951cb9986d959e8042ec848b.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/473476287cbe3d02d231d7bd947fbff95bb60490951cb9986d959e8042ec848b.jpg)

![4a19f884c779d66fa460e0e5ffa92497bfe77e062da50b08aa167bcf13919c63.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/4a19f884c779d66fa460e0e5ffa92497bfe77e062da50b08aa167bcf13919c63.jpg)

![5631867f39f91428d5b717d244442ddc76f7484a640ea279265c9bef9049963a.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/5631867f39f91428d5b717d244442ddc76f7484a640ea279265c9bef9049963a.jpg)

![920b46602358a122ca5407074e02d3c228aa635a958c2a0bef10bfafd5d76377.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/920b46602358a122ca5407074e02d3c228aa635a958c2a0bef10bfafd5d76377.jpg)

![ac22f54676d03cbdfe206955efdf9e1bfe4651ed0f5cc4297f2307b86de0b08e.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/ac22f54676d03cbdfe206955efdf9e1bfe4651ed0f5cc4297f2307b86de0b08e.jpg)

![c673662b4b6fb0b87152b75acbf01a22dd774f762efa0f3fc55e486c541490af.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/c673662b4b6fb0b87152b75acbf01a22dd774f762efa0f3fc55e486c541490af.jpg)

![d36794984e38fd1d10976020eefc4adee76641cb38c79072c6f1d4528d5670df.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/images/d36794984e38fd1d10976020eefc4adee76641cb38c79072c6f1d4528d5670df.jpg)

### Tables

![1dd8976a3a4e3e929ca90d9963585d7c438a05d90457a51957e2df48bd3484c8.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/1dd8976a3a4e3e929ca90d9963585d7c438a05d90457a51957e2df48bd3484c8.jpg)

![23d8941e9a0c3f6b3955e9924f8da10fcd35cf9288f831985564c010e8dc4f4b.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/23d8941e9a0c3f6b3955e9924f8da10fcd35cf9288f831985564c010e8dc4f4b.jpg)

![42c73d35abe38a61b5c2ef3f73d50b9e9004ef0c0393265fd15da1071071e82c.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/42c73d35abe38a61b5c2ef3f73d50b9e9004ef0c0393265fd15da1071071e82c.jpg)

![5068fa40551f10543760e0f651d934ede9de9f6418679922de891d0dcdda664d.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/5068fa40551f10543760e0f651d934ede9de9f6418679922de891d0dcdda664d.jpg)

![53c0ce70b6549c9a2b7396d55e0cac1336198af3183596dd9013209423eea668.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/53c0ce70b6549c9a2b7396d55e0cac1336198af3183596dd9013209423eea668.jpg)

![554f6a10e88e54ef58f85e47aa5b77b5527db2cc64dbb63d44ae5a78d34f0eae.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/554f6a10e88e54ef58f85e47aa5b77b5527db2cc64dbb63d44ae5a78d34f0eae.jpg)

![68fafa622a32684fc095d30986651d2395b8533dcdb9842a70d59f30d42f9d02.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/68fafa622a32684fc095d30986651d2395b8533dcdb9842a70d59f30d42f9d02.jpg)

![6d781c0b76ea5c9784ca0f6955d2dcabba65499265113ae7dae2689144ba9a71.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/6d781c0b76ea5c9784ca0f6955d2dcabba65499265113ae7dae2689144ba9a71.jpg)

![6e50cd43624a0b5f762307cb98014ccf2add80fe5e22f9b9927f985d9c44380c.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/6e50cd43624a0b5f762307cb98014ccf2add80fe5e22f9b9927f985d9c44380c.jpg)

![a24842710b1c102e7452d1e41b994fa7a9ba3f31262f0bf0efdd82d06201ea7c.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/a24842710b1c102e7452d1e41b994fa7a9ba3f31262f0bf0efdd82d06201ea7c.jpg)

![a3990cd346ec92b6e484df03b53686232c73ae7537d060eb496ce3e7a2180f62.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/a3990cd346ec92b6e484df03b53686232c73ae7537d060eb496ce3e7a2180f62.jpg)

![a900a12f1dfa22dcc250ca929fdf87fa21d1f2cb4251ed449d8d064eec6f9ebe.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/a900a12f1dfa22dcc250ca929fdf87fa21d1f2cb4251ed449d8d064eec6f9ebe.jpg)

![b49e70b6bde4bbef8c45ba073669bbfe7a42c4214d064cf3cb8ca23b81fc2457.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/b49e70b6bde4bbef8c45ba073669bbfe7a42c4214d064cf3cb8ca23b81fc2457.jpg)

![cac830dce3d4e99cfeff5ae5cc117ecff583a64e423db8052ac8eaa1c416f188.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/cac830dce3d4e99cfeff5ae5cc117ecff583a64e423db8052ac8eaa1c416f188.jpg)

![ebe49992aa052263c1e7417c16cd6d4e6aef4224d7705095981fd96f97bfee58.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/ebe49992aa052263c1e7417c16cd6d4e6aef4224d7705095981fd96f97bfee58.jpg)

![f656d1e3933af487fd2d393f115e827c29e53422b8c3aa12ab901052f1bde21a.jpg](../icml_results/4794_PESD-TSF_%20A%20Period-Aware%20and%20Explicit%20Structured%20Decomposition%20Framework%20for%20Long-Term%20Time%20Series%20F/tables/f656d1e3933af487fd2d393f115e827c29e53422b8c3aa12ab901052f1bde21a.jpg)

<a id="paper-4747"></a>

## When Planning Fails Despite Correct Execution: On Epistemic Calibration for LLM-Based Multi-Agent Systems


### Images

![3aa3c8089864a4a3d1f26eeafddd069a700511d0beed28241f96a60a73144e2a.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/images/3aa3c8089864a4a3d1f26eeafddd069a700511d0beed28241f96a60a73144e2a.jpg)

![b3a85ae3aef6af881c415eace97fb48fdfc58656b1c28f7fce48b59bae9e9c3e.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/images/b3a85ae3aef6af881c415eace97fb48fdfc58656b1c28f7fce48b59bae9e9c3e.jpg)

### Tables

![26aca56b1bdb88bee8cd77aa25b1d82a07646067bf27c5df407b47e9fb4d9544.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/26aca56b1bdb88bee8cd77aa25b1d82a07646067bf27c5df407b47e9fb4d9544.jpg)

![360085ec2b33b39cf9ed78e59eb3de9fee156513751c7062c1a784e3aa4eba13.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/360085ec2b33b39cf9ed78e59eb3de9fee156513751c7062c1a784e3aa4eba13.jpg)

![680ac107b84b52f8a6566a0a673f681674106e83c929307f9200d75cc89eb38c.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/680ac107b84b52f8a6566a0a673f681674106e83c929307f9200d75cc89eb38c.jpg)

![6d1933d3f81b04dd7c74cbc71ad268bee48f623365a6a2b493e8c8d24f6e7bc0.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/6d1933d3f81b04dd7c74cbc71ad268bee48f623365a6a2b493e8c8d24f6e7bc0.jpg)

![76b3ef56e2d8df4da35ec7bf099e7ec5da23e503e41014aeb4ac5b3c198955cd.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/76b3ef56e2d8df4da35ec7bf099e7ec5da23e503e41014aeb4ac5b3c198955cd.jpg)

![a4cbeaf9291417a2b76456da6db34468d87b3361c3deeeff3a874027cbb14cbe.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/a4cbeaf9291417a2b76456da6db34468d87b3361c3deeeff3a874027cbb14cbe.jpg)

![d646029e3d28d051aad814b018608366f11e2dfec95eeba671440c595fd6ab97.jpg](../icml_results/4795_When%20Planning%20Fails%20Despite%20Correct%20Execution_%20On%20Epistemic%20Calibration%20for%20LLM-Based%20Multi-Agent%20Sy/tables/d646029e3d28d051aad814b018608366f11e2dfec95eeba671440c595fd6ab97.jpg)

<a id="paper-4748"></a>

## AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving


### Images

![039823b86422e095a8cf98a4054be9f3eb26bb725bd5825c4e0ccad8d4faa8b5.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/039823b86422e095a8cf98a4054be9f3eb26bb725bd5825c4e0ccad8d4faa8b5.jpg)

![0ec30d1788a5bbf96646d5f26d1ccd4000a304248f3308cf201e95d5217ead8f.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/0ec30d1788a5bbf96646d5f26d1ccd4000a304248f3308cf201e95d5217ead8f.jpg)

![1276fd08e17d75cc1895ef9b97af397c230c5ee0bb37d07046335eb38b761755.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/1276fd08e17d75cc1895ef9b97af397c230c5ee0bb37d07046335eb38b761755.jpg)

![1ffdd6972afe318392eaaf61a78a60c393cfb82f2f638435a0d08def4accc94a.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/1ffdd6972afe318392eaaf61a78a60c393cfb82f2f638435a0d08def4accc94a.jpg)

![2156aa03e64fc194033b946fa73bdc388ac8e9663025a158a3e1b69062b1a08d.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/2156aa03e64fc194033b946fa73bdc388ac8e9663025a158a3e1b69062b1a08d.jpg)

![2ed912b8ea43b9e7f3ded1b970d9dc5af4718a92b26f5cf853aa9e238a8ecfab.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/2ed912b8ea43b9e7f3ded1b970d9dc5af4718a92b26f5cf853aa9e238a8ecfab.jpg)

![611892ae59eada0d46b8853b4ae7bb7d2ae2da3723963df4eb65beafc52fa0a8.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/611892ae59eada0d46b8853b4ae7bb7d2ae2da3723963df4eb65beafc52fa0a8.jpg)

![6779f36e8ad780db3c50c1890747f937f13b13d28d779fe27f928914bce3a514.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/6779f36e8ad780db3c50c1890747f937f13b13d28d779fe27f928914bce3a514.jpg)

![72d87bb70faafc7b6ec817101306451a823d20edb1508809730b66eed867a390.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/72d87bb70faafc7b6ec817101306451a823d20edb1508809730b66eed867a390.jpg)

![791f088a97d3394945c439d356bc17e796cb280b59bd7e2ccd23550971bb8292.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/791f088a97d3394945c439d356bc17e796cb280b59bd7e2ccd23550971bb8292.jpg)

![797e48b0f027a42db76a2568bb9b47d3f32bbe2b1bd93967ff51a9e735cbdce2.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/797e48b0f027a42db76a2568bb9b47d3f32bbe2b1bd93967ff51a9e735cbdce2.jpg)

![9bef1889f151e0c748729c0fb8b63edf399277aafa76e48920584c6eab745733.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/9bef1889f151e0c748729c0fb8b63edf399277aafa76e48920584c6eab745733.jpg)

![bc3253bec91c6911783ca3628705caa802b2cc3647080008601a8cc579e2cf73.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/bc3253bec91c6911783ca3628705caa802b2cc3647080008601a8cc579e2cf73.jpg)

![c33561ef4229e5c05fdfbe0c3dc98a87ececb53340db77336363dec80c6b8a1c.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/c33561ef4229e5c05fdfbe0c3dc98a87ececb53340db77336363dec80c6b8a1c.jpg)

![c721fa22133a8fe70959efbca99dedb87423a78fff4e28ba37e05dee88ddf69d.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/c721fa22133a8fe70959efbca99dedb87423a78fff4e28ba37e05dee88ddf69d.jpg)

![e1ebfaae98d4ef50bb54cf143b3e48b1a4b9ec870d21529df470d3d94c362d9a.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/e1ebfaae98d4ef50bb54cf143b3e48b1a4b9ec870d21529df470d3d94c362d9a.jpg)

![e9cb213925b8a40c933e19240e4bc1e632bac6d1a2640e9431140a1e5429a602.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/e9cb213925b8a40c933e19240e4bc1e632bac6d1a2640e9431140a1e5429a602.jpg)

![eb997856c5233042950a88962cb049f581dcd5fe42a3fb77473e712dcb700b3d.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/images/eb997856c5233042950a88962cb049f581dcd5fe42a3fb77473e712dcb700b3d.jpg)

### Tables

![289c043b600226ce2cabbcf1026e33e4b84241c88ae679fb9a177637ed4f8b18.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/289c043b600226ce2cabbcf1026e33e4b84241c88ae679fb9a177637ed4f8b18.jpg)

![2906c28cf04ab7b0712c17c8c804b8591971f4690b6ff989afd1ee9fbc999bdc.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/2906c28cf04ab7b0712c17c8c804b8591971f4690b6ff989afd1ee9fbc999bdc.jpg)

![7edc8b0e42c1603cd3f7a7f4dbe47bfc0592e0092a877c6ca183c71e6eb27582.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/7edc8b0e42c1603cd3f7a7f4dbe47bfc0592e0092a877c6ca183c71e6eb27582.jpg)

![db5c53203cc15c03d0a189f906646304d19ab1495ce1c1e1c2389a5dda974bc6.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/db5c53203cc15c03d0a189f906646304d19ab1495ce1c1e1c2389a5dda974bc6.jpg)

![df154f798ca59584ddedad268a7c409686b5892eff25cbe8605c9a86496c6b12.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/df154f798ca59584ddedad268a7c409686b5892eff25cbe8605c9a86496c6b12.jpg)

![eef788900e37f396b5d44834ce4be2e3f8fd83d3cae0f88d005501592ce253b5.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/eef788900e37f396b5d44834ce4be2e3f8fd83d3cae0f88d005501592ce253b5.jpg)

![f3a8dedd40ab07ae301f9e0511251e729ff030167f67ab7cc4078c7081d4f0df.jpg](../icml_results/4796_AugServe_%20Adaptive%20Request%20Scheduling%20for%20Augmented%20Large%20Language%20Model%20Inference%20Serving/tables/f3a8dedd40ab07ae301f9e0511251e729ff030167f67ab7cc4078c7081d4f0df.jpg)

<a id="paper-4749"></a>

## LightWM: Training-Free Hierarchical Working Memory for Small Language Model Agents


### Images

![22281ee4d4ad5c7307276bed8cd53fd7cc289f5abd8ed566f453669a5201c4d2.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/images/22281ee4d4ad5c7307276bed8cd53fd7cc289f5abd8ed566f453669a5201c4d2.jpg)

![322d645cc24bf9743bbcd8d4bc13f5b9f634f1358d22a60e27978829ab4c7e2c.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/images/322d645cc24bf9743bbcd8d4bc13f5b9f634f1358d22a60e27978829ab4c7e2c.jpg)

![d529a085f9419d0c6358e188123e1d1722903bb0452e53b2df00c4f2c225f964.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/images/d529a085f9419d0c6358e188123e1d1722903bb0452e53b2df00c4f2c225f964.jpg)

### Tables

![10e9a5e59e7a3c3df6a70e71a028410c67fc694b8fa20daf19adaba08f7c3eae.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/10e9a5e59e7a3c3df6a70e71a028410c67fc694b8fa20daf19adaba08f7c3eae.jpg)

![5258d43b62a696ee28d7f1f4f50b615415316045626858f58a496c6f9c107468.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/5258d43b62a696ee28d7f1f4f50b615415316045626858f58a496c6f9c107468.jpg)

![701c1aa8929ee1b6e2fc1e5afaaf27604207337e1c0fd0529e5fdcc77242c5b5.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/701c1aa8929ee1b6e2fc1e5afaaf27604207337e1c0fd0529e5fdcc77242c5b5.jpg)

![779379faf4d60e7a97addf1c0bd8c115146bf6a4f44bc419888208e90d36ab60.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/779379faf4d60e7a97addf1c0bd8c115146bf6a4f44bc419888208e90d36ab60.jpg)

![816853190481a4cf4be1d60cf407de19250e192e75354a2bb0cca4cf7a282f74.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/816853190481a4cf4be1d60cf407de19250e192e75354a2bb0cca4cf7a282f74.jpg)

![9608ae4b857e95246abc4c7f84d5ac5e6797fd53e4994cf803bdf5ca1562d69f.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/9608ae4b857e95246abc4c7f84d5ac5e6797fd53e4994cf803bdf5ca1562d69f.jpg)

![98478bb019cfc67eee2c418e8f9dd27e49d68a49cdfe1e93c71be0e200c3fff3.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/98478bb019cfc67eee2c418e8f9dd27e49d68a49cdfe1e93c71be0e200c3fff3.jpg)

![9e3b1d3d92f352fd0df569d39de4d03d1107485742031ae3d8d41d6dc0e88ff8.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/9e3b1d3d92f352fd0df569d39de4d03d1107485742031ae3d8d41d6dc0e88ff8.jpg)

![aab52a936ec7b75ec726d8e3c7e4e7515695c9e5ceb8eaec3f05cec6fb8e27a1.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/aab52a936ec7b75ec726d8e3c7e4e7515695c9e5ceb8eaec3f05cec6fb8e27a1.jpg)

![bb0a8a9d5a19b04a1b0838700bdc6e57d0545fa975594ef0fab2f2011f741c31.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/bb0a8a9d5a19b04a1b0838700bdc6e57d0545fa975594ef0fab2f2011f741c31.jpg)

![dfb0fa41e8977e3c88b14801e379f11bff88f099ca051dba779086fd68ba9b9f.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/dfb0fa41e8977e3c88b14801e379f11bff88f099ca051dba779086fd68ba9b9f.jpg)

![f0c74dc160db4770f8bc57d4e9b32d6ece47087ba9ceb79375480b3166d35bab.jpg](../icml_results/4797_LightWM_%20Training-Free%20Hierarchical%20Working%20Memory%20for%20Small%20Language%20Model%20Agents/tables/f0c74dc160db4770f8bc57d4e9b32d6ece47087ba9ceb79375480b3166d35bab.jpg)

<a id="paper-4750"></a>

## The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works


### Images

![092680a83cb089c469bab79e2611a4f155ec61cab8db2529aec5017b7b988caa.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/092680a83cb089c469bab79e2611a4f155ec61cab8db2529aec5017b7b988caa.jpg)

![3e60bd180fa20bb7ce829b180ccf626263f37a268d905f7a1432c758628b3b60.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/3e60bd180fa20bb7ce829b180ccf626263f37a268d905f7a1432c758628b3b60.jpg)

![3f51bca3c7710df35ad381bd1faba9397fcebe9f35cfff7cdb21c92bb71fbd76.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/3f51bca3c7710df35ad381bd1faba9397fcebe9f35cfff7cdb21c92bb71fbd76.jpg)

![4951aa5042fe151638f4bff55b509f226fd10c52c6c20b9a6446d3837c37f43d.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/4951aa5042fe151638f4bff55b509f226fd10c52c6c20b9a6446d3837c37f43d.jpg)

![5dcde5fc3e13a3fd9e470040598d4be55afcebc024e92200235f9fe785ca752c.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/5dcde5fc3e13a3fd9e470040598d4be55afcebc024e92200235f9fe785ca752c.jpg)

![600faffcf2c2f762b6f43fff8bdb555cd394a42acece1776ae70c3546e2ceaf7.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/600faffcf2c2f762b6f43fff8bdb555cd394a42acece1776ae70c3546e2ceaf7.jpg)

![a45e5d4287ce64ac9af551ca1fe89f9eee31647de942bca89e458dabdd72bb00.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/a45e5d4287ce64ac9af551ca1fe89f9eee31647de942bca89e458dabdd72bb00.jpg)

![ac87ae34750fe95bd54f74ca7fa9017466ec3b50ac3f4e41d66e6726b98c4232.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/ac87ae34750fe95bd54f74ca7fa9017466ec3b50ac3f4e41d66e6726b98c4232.jpg)

![b8979869a717227e8ec1d6ee9e95911d61427d7d99c835ba01366e14fa5c4361.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/b8979869a717227e8ec1d6ee9e95911d61427d7d99c835ba01366e14fa5c4361.jpg)

![ce63f6141a50dff9196562f94f7ad4e6b5b062083a6224a63aed681f23c0be6d.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/ce63f6141a50dff9196562f94f7ad4e6b5b062083a6224a63aed681f23c0be6d.jpg)

![fbc46587aa77235bfe84c3e8e6c89974cb885ba1c31d90650832b74f1dab7a9a.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/fbc46587aa77235bfe84c3e8e6c89974cb885ba1c31d90650832b74f1dab7a9a.jpg)

![fde4c94635b22c1b388f6f37650f09ed605170e91d306919a99fc400a17e3b51.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/images/fde4c94635b22c1b388f6f37650f09ed605170e91d306919a99fc400a17e3b51.jpg)

### Tables

![027d18c61529fc784226bd6d9b491e1dea3b627cdb70d24e0de4f089e9e2c494.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/027d18c61529fc784226bd6d9b491e1dea3b627cdb70d24e0de4f089e9e2c494.jpg)

![2759bab7ac87943a86e94a3f41e8c401b479ffdb591055fd9d7bfbf26af15e1c.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/2759bab7ac87943a86e94a3f41e8c401b479ffdb591055fd9d7bfbf26af15e1c.jpg)

![3021bacc3fdc2cf5b7e45c394779f90705a9d019059582d8cb842f0dcf5c335c.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/3021bacc3fdc2cf5b7e45c394779f90705a9d019059582d8cb842f0dcf5c335c.jpg)

![4605bf87faf0d5e78984db1dcb3041d194cac44f6e94245a16b29b6a69898519.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/4605bf87faf0d5e78984db1dcb3041d194cac44f6e94245a16b29b6a69898519.jpg)

![4e125e271373800d9a4fbcacc4c21a9f78bc61c5727e3acd184b28c85a7844e6.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/4e125e271373800d9a4fbcacc4c21a9f78bc61c5727e3acd184b28c85a7844e6.jpg)

![71905aa7d0c7c94e48d1d65d501b91435a06c77b12368c4db3bd849b6397ff4f.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/71905aa7d0c7c94e48d1d65d501b91435a06c77b12368c4db3bd849b6397ff4f.jpg)

![84e8eba0d876ac8277eb8dddb528e5b825e58e847a17447237e4c68452190ba5.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/84e8eba0d876ac8277eb8dddb528e5b825e58e847a17447237e4c68452190ba5.jpg)

![88e5150d989be85e926cf6fbdfa76f556c5694c7b7a04ffc3979b6fade2ab3bd.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/88e5150d989be85e926cf6fbdfa76f556c5694c7b7a04ffc3979b6fade2ab3bd.jpg)

![98fc6d0c8eccf5891ab8ba232c7bad2beccd4c2e97519d83580e89387a55f1f0.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/98fc6d0c8eccf5891ab8ba232c7bad2beccd4c2e97519d83580e89387a55f1f0.jpg)

![9ca70f97cf41b5382ac47c9a23b00aacfa1a038e1b493ff8a6f582868bfdd013.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/9ca70f97cf41b5382ac47c9a23b00aacfa1a038e1b493ff8a6f582868bfdd013.jpg)

![a08b80f36d8711d662e36b09346273514808b99a7efe9b3d22c14a59fb32c408.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/a08b80f36d8711d662e36b09346273514808b99a7efe9b3d22c14a59fb32c408.jpg)

![a469fad72797d47da2a097cd4b2ff91b79ef059728562f55c2412a2da6c82c88.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/a469fad72797d47da2a097cd4b2ff91b79ef059728562f55c2412a2da6c82c88.jpg)

![ae3c57c81365d212b3649dcb03607191bb4f1e5486d32f6949500b38057753d6.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/ae3c57c81365d212b3649dcb03607191bb4f1e5486d32f6949500b38057753d6.jpg)

![ae91aeee39125ff2c55844d386c7d7c4da4ec7b2b57e06d342f9798f019570e4.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/ae91aeee39125ff2c55844d386c7d7c4da4ec7b2b57e06d342f9798f019570e4.jpg)

![b1d1ec191ef45eff5218ca63eeb1603f530518d85b5ef89c9e65f0c179c07647.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/b1d1ec191ef45eff5218ca63eeb1603f530518d85b5ef89c9e65f0c179c07647.jpg)

![b398ef44a2ebf2988a2b19244f5c9c231f6d1e93c91018579951e95d4930ad00.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/b398ef44a2ebf2988a2b19244f5c9c231f6d1e93c91018579951e95d4930ad00.jpg)

![b43b1a1f2ee870a92a72b00f07903c02e29d298249c12efc2b8944ea650e4836.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/b43b1a1f2ee870a92a72b00f07903c02e29d298249c12efc2b8944ea650e4836.jpg)

![b4fd5cd2cad57cad825917a70fa10b5b8f3f8939ee48222ee85a04d3bbb237a3.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/b4fd5cd2cad57cad825917a70fa10b5b8f3f8939ee48222ee85a04d3bbb237a3.jpg)

![be3d59bf26d92881a1f6dde55de993a8db7fbc76fa6a4b3c0ed6a18f41fc9fba.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/be3d59bf26d92881a1f6dde55de993a8db7fbc76fa6a4b3c0ed6a18f41fc9fba.jpg)

![c6d21bf89e84e5f18a1e891e6be54419a9055177d822b54fa499241d7fd0f60e.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/c6d21bf89e84e5f18a1e891e6be54419a9055177d822b54fa499241d7fd0f60e.jpg)

![dab7a89011ae4fdae3db547905e910b8757fa0836738fe6978fc77e23c169e23.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/dab7a89011ae4fdae3db547905e910b8757fa0836738fe6978fc77e23c169e23.jpg)

![e5dd89c8da06fcdfd5941becd4d712722665f059ee9d0a41cca9bd3956b9374b.jpg](../icml_results/4798_The%20Bridge-Garden%20Dilemma%20in%20LLM%20Distillation_%20Why%20Mixing%20Hard%20and%20Soft%20Labels%20Works/tables/e5dd89c8da06fcdfd5941becd4d712722665f059ee9d0a41cca9bd3956b9374b.jpg)

<a id="paper-4751"></a>

## EvReflection: Event-Driven Micro-Dynamics for Reflection Removal


### Images

![122673d5a143b8c74308c51c1481e696ddc59e7644242738e178a50b4a7ea457.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/122673d5a143b8c74308c51c1481e696ddc59e7644242738e178a50b4a7ea457.jpg)

![1d6d121d2b91981fd2fa0938f153dfb9bc8f19027ff7e4d1c21911c6dc0eff12.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/1d6d121d2b91981fd2fa0938f153dfb9bc8f19027ff7e4d1c21911c6dc0eff12.jpg)

![43b1d6b79284aad44aeaa034f48e5adf6364601fa5c2f7153d53b81431c31bcb.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/43b1d6b79284aad44aeaa034f48e5adf6364601fa5c2f7153d53b81431c31bcb.jpg)

![5f93c4fe15612fe2c322257a173b5d9e5dd90d698dfd80cd62c11b04cc14c824.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/5f93c4fe15612fe2c322257a173b5d9e5dd90d698dfd80cd62c11b04cc14c824.jpg)

![7acb83c30d81173d3eca0bfa70387d6754e54aa9961d44757e24e6feebd18640.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/7acb83c30d81173d3eca0bfa70387d6754e54aa9961d44757e24e6feebd18640.jpg)

![8fc3f1e63540893fb5a3ec6a597b1c21a833b91e69377d994dfd378cdc620650.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/8fc3f1e63540893fb5a3ec6a597b1c21a833b91e69377d994dfd378cdc620650.jpg)

![a4ee8bbe36a3b94c3d6c030edf20581aee459e149f8121d143dcb7e11ae9a614.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/a4ee8bbe36a3b94c3d6c030edf20581aee459e149f8121d143dcb7e11ae9a614.jpg)

![c9c4ef7fe87d44071cc1587d6915f9608391ec6d8fdcfbe7fed2c07c768e4a02.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/c9c4ef7fe87d44071cc1587d6915f9608391ec6d8fdcfbe7fed2c07c768e4a02.jpg)

![e74a926e9ecda1c3bf134c123b4a98c02e1db0f280de07a99e07cac636037b71.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/e74a926e9ecda1c3bf134c123b4a98c02e1db0f280de07a99e07cac636037b71.jpg)

![ec779c41cd2df3cb5848e2b8da0f83d391b9b8af1d3d30e9ac48754fb70a6954.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/ec779c41cd2df3cb5848e2b8da0f83d391b9b8af1d3d30e9ac48754fb70a6954.jpg)

![faad8476a44b5acf631fc7b36a4a104d10f1cc31a95c8c6624b904672f24d06f.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/images/faad8476a44b5acf631fc7b36a4a104d10f1cc31a95c8c6624b904672f24d06f.jpg)

### Tables

![0e966640a9d68978955f19834354d569d44a7b0bf924b2fb003384814306f1a2.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/0e966640a9d68978955f19834354d569d44a7b0bf924b2fb003384814306f1a2.jpg)

![247092b9c6fa12290909feb2424cc63958ef24aefa5c6d756199c926dd051f16.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/247092b9c6fa12290909feb2424cc63958ef24aefa5c6d756199c926dd051f16.jpg)

![29115805c0a65fc532e79b0b0ff3ca3ab966118cac0c6b2d2dbd132af28c77dc.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/29115805c0a65fc532e79b0b0ff3ca3ab966118cac0c6b2d2dbd132af28c77dc.jpg)

![3fa61e0bcc115879651fb53866606f61c0cacf12d3de0144942384d4f55a25a5.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/3fa61e0bcc115879651fb53866606f61c0cacf12d3de0144942384d4f55a25a5.jpg)

![94714ec7b0c1061f3895ed77090408873a34a4c8f92c598c025046b4953f276f.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/94714ec7b0c1061f3895ed77090408873a34a4c8f92c598c025046b4953f276f.jpg)

![a287dbed9883374ff2ed6f7faafb8caa134fac85be01dcbeaf21b540474b0db3.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/a287dbed9883374ff2ed6f7faafb8caa134fac85be01dcbeaf21b540474b0db3.jpg)

![c5c0d1106469e0874cd166e844833b4067fdfa55ddd469c85d2b74cd5564d9e9.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/c5c0d1106469e0874cd166e844833b4067fdfa55ddd469c85d2b74cd5564d9e9.jpg)

![d1427f4d03a98f28e1022f5bcf5436c6ccc956f41836607f17e0ba2556fa576b.jpg](../icml_results/4799_EvReflection_%20Event-Driven%20Micro-Dynamics%20for%20Reflection%20Removal/tables/d1427f4d03a98f28e1022f5bcf5436c6ccc956f41836607f17e0ba2556fa576b.jpg)

<a id="paper-4752"></a>

## Discovering Differences in Strategic Behavior between Humans and LLMs

### Images

![13f6125bd0f4d10081419aa1b37f9864fc45722433907616efd238ba6ba114c7.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/13f6125bd0f4d10081419aa1b37f9864fc45722433907616efd238ba6ba114c7.jpg)

![15e7c3363b031be4f679a8f83d18e606e4c5b7e2829f11b01a4301b5ad8bf165.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/15e7c3363b031be4f679a8f83d18e606e4c5b7e2829f11b01a4301b5ad8bf165.jpg)

![3a89c2ddf53ed7c1ff4696dfe19675239f3855a54889857b4408109a24e520d4.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3a89c2ddf53ed7c1ff4696dfe19675239f3855a54889857b4408109a24e520d4.jpg)

![3c0ddf9ee3017cc0de53ba725b52caba25745a4c596fc8181efff11b7771fd6b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3c0ddf9ee3017cc0de53ba725b52caba25745a4c596fc8181efff11b7771fd6b.jpg)

![3c9ff92c67ccfdbd37c952855848cabae949785410839d91a264a423517f210b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3c9ff92c67ccfdbd37c952855848cabae949785410839d91a264a423517f210b.jpg)

![3d9096193950663a5559136120b1bec988542b166fc7698ca303080e03cee13b.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/3d9096193950663a5559136120b1bec988542b166fc7698ca303080e03cee13b.jpg)

![4ef856ca4f6a6be75ad3d716a3fe2bb8aeb4a3187a10ebf1607e104a18464349.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/4ef856ca4f6a6be75ad3d716a3fe2bb8aeb4a3187a10ebf1607e104a18464349.jpg)

![5ce30542dfad26de530f945c59cd9813e575e3f69c0cc5099d923c79ece6a748.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/5ce30542dfad26de530f945c59cd9813e575e3f69c0cc5099d923c79ece6a748.jpg)

![7acddc3c0f582bbe087a6b74f2ad56302a3c15ea55b6d1d21a25bbc98eae1a4e.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/7acddc3c0f582bbe087a6b74f2ad56302a3c15ea55b6d1d21a25bbc98eae1a4e.jpg)

![8581c21864af0f8fac0d087ca50a5b2afe0529bfa6da8ea8aada07ea5bfe5c74.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/8581c21864af0f8fac0d087ca50a5b2afe0529bfa6da8ea8aada07ea5bfe5c74.jpg)

![959080bdd69fbdf3275c6e4b469e3f30128d6be5bce634ab8c871f5099aeea66.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/959080bdd69fbdf3275c6e4b469e3f30128d6be5bce634ab8c871f5099aeea66.jpg)

![a79de15b1911ff87920f6388e9ba26899ddccf236f7a1cdb12dab1b35f5b04de.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/a79de15b1911ff87920f6388e9ba26899ddccf236f7a1cdb12dab1b35f5b04de.jpg)

![fc0596a4a6a878a20ca12705da702883848993853c7e46d8666fba7d407dda2a.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/images/fc0596a4a6a878a20ca12705da702883848993853c7e46d8666fba7d407dda2a.jpg)

### Tables

![281f0c8b439a38eed16f02801b9a3bb93aed8b2cf0b96c94a0d23c3996f9376a.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/281f0c8b439a38eed16f02801b9a3bb93aed8b2cf0b96c94a0d23c3996f9376a.jpg)

![518fdb96b99c45856e66b233f33adde8c8b7616685a25f27197eced3bf9f8539.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/518fdb96b99c45856e66b233f33adde8c8b7616685a25f27197eced3bf9f8539.jpg)

![5ec3275ece175b727eaf0240b610d31f9de58b6986da2f91ed471286879af599.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/5ec3275ece175b727eaf0240b610d31f9de58b6986da2f91ed471286879af599.jpg)

![da8d9b797c5ff756fe38ede140e3c0da57fb718ab00e9ae801c654b917101b45.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/da8d9b797c5ff756fe38ede140e3c0da57fb718ab00e9ae801c654b917101b45.jpg)

![f6535214da12d182c00c58e3ad4a362c04f0d686a744adae03ccccb1184b39b0.jpg](../icml_results/4800_Discovering%20Differences%20in%20Strategic%20Behavior%20between%20Humans%20and%20LLMs/tables/f6535214da12d182c00c58e3ad4a362c04f0d686a744adae03ccccb1184b39b0.jpg)
