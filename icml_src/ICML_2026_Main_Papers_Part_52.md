# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 52 of 197

## 目录 (Table of Contents)

1. [LLM-Guided Loop Bound Generation for Program Termination Verification](#paper-1684)
2. [Direct 3D-Aware Object Insertion via Decomposed Visual Proxies](#paper-1685)
3. [EchoingPixels: Aliasing-Resistant Joint Token Reduction for Audio-Visual LLMs](#paper-1686)
4. [D-Judge: Disrupting Multi-Turn Jailbreaks using Semantics-Preserving Output Rewriting](#paper-1687)
5. [The Cost of Information: Phase Transitions in Contextual Bandits with Paid Observations](#paper-1688)
6. [Light Up Your Face: A Physically Consistent Dataset and Diffusion Model for Face Fill-Light Enhancement](#paper-1689)
7. [QEDBench: Quantifying the Alignment Gap in Automated Evaluation of University-Level Mathematical Proofs](#paper-1690)
8. [The Differences Between Direct Alignment Algorithms are a Blur](#paper-1691)
9. [Towards Spectroscopy: Susceptibility Clusters in Language Models](#paper-1692)
10. [LATMiX: Learnable Affine Transformations for Microscaling Quantization of LLMs](#paper-1693)
11. [TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning](#paper-1694)
12. [The Illusion of Generalization in Tabular Language Models](#paper-1695)
13. [Certifying Graph Neural Networks Against Label and Structure Poisoning](#paper-1696)
14. [Byte Pair Encoding for Efficient Time Series Forecasting](#paper-1697)
15. [Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning](#paper-1698)
16. [Categorical Reparameterization with Denoising Diffusion Models](#paper-1699)
17. [Contextual Slate GLM Bandits with Limited Adaptivity](#paper-1700)
18. [FlowState: Sampling-Rate-Equivariant Time-Series Forecasting](#paper-1701)
19. [Safety-Efficacy Trade Off: Robustness against Data-Poisoning](#paper-1702)
20. [Multi-marginal temporal Schrödinger Bridge Matching from unpaired data](#paper-1703)
21. [Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits](#paper-1704)
22. [Approximating Drift-Diffusion Models for User Decisions under Nudging and External Information](#paper-1705)
23. [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](#paper-1706)
24. [The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](#paper-1707)
25. [Feedback Control for Multi-Objective Graph Self-Supervision](#paper-1708)
26. [Text Has Curvature](#paper-1709)
27. [Tensor Train Diffusion: Leveraging Low-Rank Structures for High-Dimensional Score-Based Sampling](#paper-1710)
28. [SARL: Structure-Aligned Reinforcement Learning for Bridging the Perception-Action Gap in Airspace](#paper-1711)
29. [CSG: Cognitive Structure Generation for Intelligent Education](#paper-1712)
30. [TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling](#paper-1713)
31. [Temporal Weighted Encoding: Towards Maximal-Capacity Spike Coding for ANN–SNN Conversion](#paper-1714)
32. [Mirror Mean-Field Langevin Dynamics](#paper-1715)
33. [Mitigating Noise-Induced Layout Priors for Object Counting in Diffusion Models](#paper-1716)

---


<a id="paper-1684"></a>

## LLM-Guided Loop Bound Generation for Program Termination Verification

### Images

![1d4f4a594fe1bb953d586daacaaa32c93e0a4887f693e474c197f17957f5537f.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/1d4f4a594fe1bb953d586daacaaa32c93e0a4887f693e474c197f17957f5537f.jpg)

![3776035cbde491442ab75c90b028a4d330f5f405affa0878494aa4517b416688.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/3776035cbde491442ab75c90b028a4d330f5f405affa0878494aa4517b416688.jpg)

![48c595e6e17b00c17cf3ddedca3359205e6a075736c4e5efa887b8a3ff87d403.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/48c595e6e17b00c17cf3ddedca3359205e6a075736c4e5efa887b8a3ff87d403.jpg)

![9c772f771efd0a49370476e29393812dfd921def038dee428170dcbbbcf871d8.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/9c772f771efd0a49370476e29393812dfd921def038dee428170dcbbbcf871d8.jpg)

![b8449632ef698771c651fd5d9737ed8df254ef2896bf48b7708d90e0b213d53e.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/b8449632ef698771c651fd5d9737ed8df254ef2896bf48b7708d90e0b213d53e.jpg)

![cbeb1815d18a2c620d52812d83c883094e761c3894dc2c9169a47273892b33ee.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/cbeb1815d18a2c620d52812d83c883094e761c3894dc2c9169a47273892b33ee.jpg)

![f4c252fcde37f68f79f2cc0a0ccac146d5c3e206e0b775f1921df7a9430253d7.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/f4c252fcde37f68f79f2cc0a0ccac146d5c3e206e0b775f1921df7a9430253d7.jpg)

![f54560979e8882df0f38ea55e3dd8eee4af0aa1e7f289793ef0217b85afda8cd.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/images/f54560979e8882df0f38ea55e3dd8eee4af0aa1e7f289793ef0217b85afda8cd.jpg)

### Tables

![06891d331f49375f72d9a301bcf8fbbb5898a7b9b35df2a063d198218c9233bb.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/06891d331f49375f72d9a301bcf8fbbb5898a7b9b35df2a063d198218c9233bb.jpg)

![125a344f96ef2911ba61310e4239e0f09acc2ccf1351cee5307ca1c4062c2944.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/125a344f96ef2911ba61310e4239e0f09acc2ccf1351cee5307ca1c4062c2944.jpg)

![2ac3112fd9eea0bfae24acce13bfdd547ab1278396c374552e7b4d675af4aba4.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/2ac3112fd9eea0bfae24acce13bfdd547ab1278396c374552e7b4d675af4aba4.jpg)

![3b33e94953d97e9d447de59e430d89cf32b1b0c3994793f53c661cb60f5da72f.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/3b33e94953d97e9d447de59e430d89cf32b1b0c3994793f53c661cb60f5da72f.jpg)

![3b9ffb9df70ff85dd3579c9004b4d21c4316536c901b8041d88df7ebd42fe6a4.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/3b9ffb9df70ff85dd3579c9004b4d21c4316536c901b8041d88df7ebd42fe6a4.jpg)

![43458c7f5cda0c981fd7e9980a75d508722da63ed66cc2a18ed1a087b8de26c5.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/43458c7f5cda0c981fd7e9980a75d508722da63ed66cc2a18ed1a087b8de26c5.jpg)

![7a10843efb9e8e370ff9ea74b92c97f10c9e96d6d956cf068135b3a66c2fbffd.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/7a10843efb9e8e370ff9ea74b92c97f10c9e96d6d956cf068135b3a66c2fbffd.jpg)

![7c47541c3682752d264c9f1411f8cb0074ca5aa3506fe8b56748571d8bc092f1.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/7c47541c3682752d264c9f1411f8cb0074ca5aa3506fe8b56748571d8bc092f1.jpg)

![851ff7409dd7d3fe9ef84cd6d4188407133bf76a6be089268b3cef00c662465c.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/851ff7409dd7d3fe9ef84cd6d4188407133bf76a6be089268b3cef00c662465c.jpg)

![87278c02d57bdc9f965364dd7c107c4b037afbf2deef14b2c4a6f295e56399db.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/87278c02d57bdc9f965364dd7c107c4b037afbf2deef14b2c4a6f295e56399db.jpg)

![9023f4f901277cd397c1d313fa90bab9d6a6f9516439537ed8e969ca29b2f6a9.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/9023f4f901277cd397c1d313fa90bab9d6a6f9516439537ed8e969ca29b2f6a9.jpg)

![9947468b0ea14a8163f6f218742528ec28d5c0f435f53c83395013481151f70f.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/9947468b0ea14a8163f6f218742528ec28d5c0f435f53c83395013481151f70f.jpg)

![a447337417f54d9b1d3be37ef8f3a55ddef3ec71fb0c8a06c72031847ec28e1f.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/a447337417f54d9b1d3be37ef8f3a55ddef3ec71fb0c8a06c72031847ec28e1f.jpg)

![a5e3ecc2b4047d7b17337a710116fa9cab1fd9642f6575fa3f109b48633fe881.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/a5e3ecc2b4047d7b17337a710116fa9cab1fd9642f6575fa3f109b48633fe881.jpg)

![bb4d286e0f99bf3e949a72986e45fe17fe2b9a3e85b7de8bc95a85e5f3fa141b.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/bb4d286e0f99bf3e949a72986e45fe17fe2b9a3e85b7de8bc95a85e5f3fa141b.jpg)

![cadc4de99e21a990bf2cb48c1326c28f36bce9f149211e45d36683056e05529f.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/cadc4de99e21a990bf2cb48c1326c28f36bce9f149211e45d36683056e05529f.jpg)

![cef80294fae865402320aeb0d5ee25ba425baa4e6c350dbb31879206e21df2a4.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/cef80294fae865402320aeb0d5ee25ba425baa4e6c350dbb31879206e21df2a4.jpg)

![cf3aa13f1874ed66ae941d165944e2e98b09383bbf5b01b3b6f72780ad2e7df5.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/cf3aa13f1874ed66ae941d165944e2e98b09383bbf5b01b3b6f72780ad2e7df5.jpg)

![e284c2b5f122b4edb0707c61861cddfbdddd47adc756da288e47a4e85d3dccfe.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/e284c2b5f122b4edb0707c61861cddfbdddd47adc756da288e47a4e85d3dccfe.jpg)

![e87d9d1c84b28fcdf983af34606c8f613496ba8dcba6a6c10cb37c0ca2959444.jpg](../icml_results/1705_Curriculum%20Reinforcement%20Learning%20for%20Black-Box%20Prompt%20Tuning%20via%20Large%20Language%20Models/tables/e87d9d1c84b28fcdf983af34606c8f613496ba8dcba6a6c10cb37c0ca2959444.jpg)

## LLM-Guided Loop Bound Generation for Program Termination Verification


### Images

![12617eb863604a9930fd15b354bf163a33094b76f08ccc47fd2419ed7539fc19.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/images/12617eb863604a9930fd15b354bf163a33094b76f08ccc47fd2419ed7539fc19.jpg)

![e335b9dcf09f64849f6cb268a63145926c4305f0c16fd3baa16bb4088e74d1ed.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/images/e335b9dcf09f64849f6cb268a63145926c4305f0c16fd3baa16bb4088e74d1ed.jpg)

### Tables

![0678b109b800ba22d6b10a3eb5c8d8d4b1d4c92e51558af2c75cc12f7735592c.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/0678b109b800ba22d6b10a3eb5c8d8d4b1d4c92e51558af2c75cc12f7735592c.jpg)

![14a42dc5183db590f1ffecc8a8c1e2b5b39ed0c7b9c33dfd9309124bcbbbb083.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/14a42dc5183db590f1ffecc8a8c1e2b5b39ed0c7b9c33dfd9309124bcbbbb083.jpg)

![6f1a417b1dedebccf9e306aa844e90d7aed807d3564cf9ad4f8088400cd7e1d7.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/6f1a417b1dedebccf9e306aa844e90d7aed807d3564cf9ad4f8088400cd7e1d7.jpg)

![d697ace670883704127a5a031c3d96bd46c5c31275d70c570a802b984e6894fa.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/d697ace670883704127a5a031c3d96bd46c5c31275d70c570a802b984e6894fa.jpg)

![e2d293d8190d726659893611f60484e56ff80d6f53281d64b473c72a1d6fb114.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/e2d293d8190d726659893611f60484e56ff80d6f53281d64b473c72a1d6fb114.jpg)

![ec0043dea542c4eaba9d742641bf7068055421d56586f0691a3e86c6256f4f20.jpg](../icml_results/1706_LLM-Guided%20Loop%20Bound%20Generation%20for%20Program%20Termination%20Verification/tables/ec0043dea542c4eaba9d742641bf7068055421d56586f0691a3e86c6256f4f20.jpg)

<a id="paper-1685"></a>

## Direct 3D-Aware Object Insertion via Decomposed Visual Proxies


### Images

![2cd85a8335909d2448ab8f53b5ed8cf28748d0309e425f3e9dbc590d84701649.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/2cd85a8335909d2448ab8f53b5ed8cf28748d0309e425f3e9dbc590d84701649.jpg)

![3593c35a652e8232a4aea24f4925bd8a23fe0f8108c9b401c55429514db25a7c.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/3593c35a652e8232a4aea24f4925bd8a23fe0f8108c9b401c55429514db25a7c.jpg)

![41a4eadfee7ce18812216ba0b00a1bd1f6e7b0c44d6bcdc1b29d5a4fec1f55d0.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/41a4eadfee7ce18812216ba0b00a1bd1f6e7b0c44d6bcdc1b29d5a4fec1f55d0.jpg)

![48dc82684f49346705dcaa62a10d366f3222c8c53171d8a2419a9032cfc4cbb0.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/48dc82684f49346705dcaa62a10d366f3222c8c53171d8a2419a9032cfc4cbb0.jpg)

![61c2b7f9b37f8980bd35e0ad6a01409633c29fbb50bc458970f17198fb9cc99c.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/61c2b7f9b37f8980bd35e0ad6a01409633c29fbb50bc458970f17198fb9cc99c.jpg)

![658bd76cdb9fd4fe5685610fb6e56084dceb480e989672ee0a95872822e4d912.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/658bd76cdb9fd4fe5685610fb6e56084dceb480e989672ee0a95872822e4d912.jpg)

![675aec933593d2b6d5dcec3c1d07941ed8d16fec0b0623cb13e6e8b56de90699.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/675aec933593d2b6d5dcec3c1d07941ed8d16fec0b0623cb13e6e8b56de90699.jpg)

![8d89956050b0747d4fa6fdf5484b56d00c3c30be6cb71d96ef1b424bc04fdd75.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/8d89956050b0747d4fa6fdf5484b56d00c3c30be6cb71d96ef1b424bc04fdd75.jpg)

![a27d8fc313a40bdb75c26bd2af6eea7edcd0b0a8633821f586a643b36b3c1731.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/a27d8fc313a40bdb75c26bd2af6eea7edcd0b0a8633821f586a643b36b3c1731.jpg)

![b549e7f25b1e8c2aafdd5162baf3770f76f40aeb9be68b0aab84fa3fa645f312.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/b549e7f25b1e8c2aafdd5162baf3770f76f40aeb9be68b0aab84fa3fa645f312.jpg)

![b9b33423233bfcb6f2e07fbe4a6559b56254c905a234dfba9ff7e25d40bc8575.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/b9b33423233bfcb6f2e07fbe4a6559b56254c905a234dfba9ff7e25d40bc8575.jpg)

![c573c1edb0ac40ee7bccdea1332c05f9261c613ef1943d10ce4a199090bab06e.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/c573c1edb0ac40ee7bccdea1332c05f9261c613ef1943d10ce4a199090bab06e.jpg)

![d54a36491565235519dd7b47d99f33ac8db9bf96a7fc0c4766b08d60d467abff.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/d54a36491565235519dd7b47d99f33ac8db9bf96a7fc0c4766b08d60d467abff.jpg)

![d73bd376fc3289bf5106e0853d96509e2979df1936e44094839501c8e9327055.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/d73bd376fc3289bf5106e0853d96509e2979df1936e44094839501c8e9327055.jpg)

![dc01ceb0042e0459cc74fd9ec78803efbd1f6be536416855770eca52367dc55e.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/dc01ceb0042e0459cc74fd9ec78803efbd1f6be536416855770eca52367dc55e.jpg)

![efbc1ce8c7711679db0642df525c59c6d74c57d14f10e6b437a7b3454ea3d0e0.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/images/efbc1ce8c7711679db0642df525c59c6d74c57d14f10e6b437a7b3454ea3d0e0.jpg)

### Tables

![2ebcb5d5a63ab9400320eb333c12bcc0026334514da22bde363a0603792352cd.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/tables/2ebcb5d5a63ab9400320eb333c12bcc0026334514da22bde363a0603792352cd.jpg)

![5b8e1aa443200c8e4b240330a10b89f2f328aa3d569d3af6d48eef72181366a2.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/tables/5b8e1aa443200c8e4b240330a10b89f2f328aa3d569d3af6d48eef72181366a2.jpg)

![607b0edaeb2bb3ccd03f56b9cf95627a6847927f51ce7f82302ad6225c4174ae.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/tables/607b0edaeb2bb3ccd03f56b9cf95627a6847927f51ce7f82302ad6225c4174ae.jpg)

![d37cde761ce80a984bc7efe6c6010627a3c9b90e637c45843fd35b4064a9f785.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/tables/d37cde761ce80a984bc7efe6c6010627a3c9b90e637c45843fd35b4064a9f785.jpg)

![f1d273adb5194e07be76bcd0d8a710c8f49966d7189d54dfe287e60645be0c4d.jpg](../icml_results/1707_Direct%203D-Aware%20Object%20Insertion%20via%20Decomposed%20Visual%20Proxies/tables/f1d273adb5194e07be76bcd0d8a710c8f49966d7189d54dfe287e60645be0c4d.jpg)

<a id="paper-1686"></a>

## EchoingPixels: Aliasing-Resistant Joint Token Reduction for Audio-Visual LLMs


### Images

![105b2d7bd1cf575e1af365d1cff937a4e69d63d104c567a0dda9e0ce9cd5fc95.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/105b2d7bd1cf575e1af365d1cff937a4e69d63d104c567a0dda9e0ce9cd5fc95.jpg)

![25972fac60c3222ec543870424deb16e2d866d4b0f58d73da7e4803ee6b0f028.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/25972fac60c3222ec543870424deb16e2d866d4b0f58d73da7e4803ee6b0f028.jpg)

![440d73b9cf653a4008789e71833b24f9c729e19c396a6e0cfc8f70d22b1d441d.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/440d73b9cf653a4008789e71833b24f9c729e19c396a6e0cfc8f70d22b1d441d.jpg)

![466b872654b22487b37f83be25355d4131ef752ae7d441d818265b6f755dcff5.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/466b872654b22487b37f83be25355d4131ef752ae7d441d818265b6f755dcff5.jpg)

![6d2a1a567d5dca41a6948ce71ef1e71286a4b26d6fa8b4e814558d89de646a84.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/6d2a1a567d5dca41a6948ce71ef1e71286a4b26d6fa8b4e814558d89de646a84.jpg)

![8a03130c95679695e9047fbf891995031c7253323bff6df2425053ee07102a27.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/8a03130c95679695e9047fbf891995031c7253323bff6df2425053ee07102a27.jpg)

![917caa6f5d1317c536771c2efe4ad0fe6b0005ca0673ee561fd2a2a57c0a0bb2.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/917caa6f5d1317c536771c2efe4ad0fe6b0005ca0673ee561fd2a2a57c0a0bb2.jpg)

![a3bc8d83b80b8aca0491caa17cc7b9fed6329aca84a992015052eb38695aef08.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/a3bc8d83b80b8aca0491caa17cc7b9fed6329aca84a992015052eb38695aef08.jpg)

![c47e7e25a48498335b7d61a902beb7dfadbef6b5f22349efaed2c59ce9e865d3.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/c47e7e25a48498335b7d61a902beb7dfadbef6b5f22349efaed2c59ce9e865d3.jpg)

![c8e3cad597f9bf38a26e00bcc4eaaf2722043eaf56af88765e9b3a182b4553a7.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/c8e3cad597f9bf38a26e00bcc4eaaf2722043eaf56af88765e9b3a182b4553a7.jpg)

![d289436e420324d50a5612f31201f4c7ea231c89e50d94eef282b32ab72d4476.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/images/d289436e420324d50a5612f31201f4c7ea231c89e50d94eef282b32ab72d4476.jpg)

### Tables

![089e6eed4c5c2a40920286ade6a64dad1f2dea1c962e2135bf43a14e45dfb9ef.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/089e6eed4c5c2a40920286ade6a64dad1f2dea1c962e2135bf43a14e45dfb9ef.jpg)

![0d08773ba205dff3306d2887a0bbad5b1792216a65143f6d964b21794a4070e8.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/0d08773ba205dff3306d2887a0bbad5b1792216a65143f6d964b21794a4070e8.jpg)

![4c3fc833712456e8a06ec472e5156123431b350697cafab3b777b9f3c2be577a.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/4c3fc833712456e8a06ec472e5156123431b350697cafab3b777b9f3c2be577a.jpg)

![5a025f0554aa2aca30b80cd49ccf237407d6ecc17be97044e775269f58a28a07.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/5a025f0554aa2aca30b80cd49ccf237407d6ecc17be97044e775269f58a28a07.jpg)

![66099876125b6f84b91027a263a04c708ee9174c24fbd8625a22fe986a8dab43.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/66099876125b6f84b91027a263a04c708ee9174c24fbd8625a22fe986a8dab43.jpg)

![69d6d92535e29b1d0afab6a52e17b47491c781f53dc16103c82b1bb4b0d1d40d.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/69d6d92535e29b1d0afab6a52e17b47491c781f53dc16103c82b1bb4b0d1d40d.jpg)

![760a1e358b1864bbbe7ecfc99b911f0f9f3c32e5a6baecccb33fe71d74b2d5ed.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/760a1e358b1864bbbe7ecfc99b911f0f9f3c32e5a6baecccb33fe71d74b2d5ed.jpg)

![c6f8f404d520ab7a43d0554bd6db9d857d52cd9ceed0a7bab1c0bded968b7e9a.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/c6f8f404d520ab7a43d0554bd6db9d857d52cd9ceed0a7bab1c0bded968b7e9a.jpg)

![c813676f86a677dbfbd01c0a63162a7be361e1b4b33060d334c9990d6d827b0d.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/c813676f86a677dbfbd01c0a63162a7be361e1b4b33060d334c9990d6d827b0d.jpg)

![ee309318a071808bb38df59d51a29ae75c42ed00608b0050d6af8212786400b4.jpg](../icml_results/1708_EchoingPixels_%20Aliasing-Resistant%20Joint%20Token%20Reduction%20for%20Audio-Visual%20LLMs/tables/ee309318a071808bb38df59d51a29ae75c42ed00608b0050d6af8212786400b4.jpg)

<a id="paper-1687"></a>

## D-Judge: Disrupting Multi-Turn Jailbreaks using Semantics-Preserving Output Rewriting


### Images

![068c924def14157660bfe4d5797aed03349f924fd98148337f11de518971a2ad.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/images/068c924def14157660bfe4d5797aed03349f924fd98148337f11de518971a2ad.jpg)

![affafa0b8ce608cefa7211867243d5f0bbbcbe01017dc4d643676d7ef597e092.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/images/affafa0b8ce608cefa7211867243d5f0bbbcbe01017dc4d643676d7ef597e092.jpg)

![dbcd3f555524bc663cb14131619bdf590cf567459021315c3265cd1a1008706a.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/images/dbcd3f555524bc663cb14131619bdf590cf567459021315c3265cd1a1008706a.jpg)

### Tables

![26d1fca8e869df84cff8356671245cb12fcccb72a90dc62555d9befd7f479987.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/26d1fca8e869df84cff8356671245cb12fcccb72a90dc62555d9befd7f479987.jpg)

![323378b51a9fe14766c0e6ff9dafeed2e1551be990970e1443cbb8b1ffc3c58c.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/323378b51a9fe14766c0e6ff9dafeed2e1551be990970e1443cbb8b1ffc3c58c.jpg)

![454245505887b43b18fb5c745ea473400a8c0b51f54abee2c1b0d09d9051f53f.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/454245505887b43b18fb5c745ea473400a8c0b51f54abee2c1b0d09d9051f53f.jpg)

![536ffa82c615cb05c2a25ea8fa78ad14d7876d2c5e57de688be1609e13340c50.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/536ffa82c615cb05c2a25ea8fa78ad14d7876d2c5e57de688be1609e13340c50.jpg)

![75a260326e9d3750bf1f8109366a43dfc4288ecf32db92f4cbb1a9f6b0bee500.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/75a260326e9d3750bf1f8109366a43dfc4288ecf32db92f4cbb1a9f6b0bee500.jpg)

![b1dd6ad2604315130f1bcff6eb2613d8d15c0fc5c4670c3f9613500286111299.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/b1dd6ad2604315130f1bcff6eb2613d8d15c0fc5c4670c3f9613500286111299.jpg)

![b46e703124694c3a86d3ad59ee048f355eedcb8654067927a93b524d02205b78.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/b46e703124694c3a86d3ad59ee048f355eedcb8654067927a93b524d02205b78.jpg)

![ce6b4f8db478dd80d9101ff881d549926bb8f1943ff986515c455413d6f97d6c.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/ce6b4f8db478dd80d9101ff881d549926bb8f1943ff986515c455413d6f97d6c.jpg)

![d4adb66d462eca9e9515cd54bbcd5588c0a60e4befcad55dbb2720363c301661.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/d4adb66d462eca9e9515cd54bbcd5588c0a60e4befcad55dbb2720363c301661.jpg)

![ebf3cf1182d4690ccaedb0c7ae800e704bc87e2e4779997bf210837277f0755d.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/ebf3cf1182d4690ccaedb0c7ae800e704bc87e2e4779997bf210837277f0755d.jpg)

![f6437da0bd49bfe71404f9ecc148b3791305023da969bb8c72ded39efa82f4d4.jpg](../icml_results/1709_D-Judge_%20Disrupting%20Multi-Turn%20Jailbreaks%20using%20Semantics-Preserving%20Output%20Rewriting/tables/f6437da0bd49bfe71404f9ecc148b3791305023da969bb8c72ded39efa82f4d4.jpg)

<a id="paper-1688"></a>

## The Cost of Information: Phase Transitions in Contextual Bandits with Paid Observations


### Images

![3fe2e3b74aef2ad08ebc3c756b75ad0c9c869f8f62ee4263132990a1ee8f9400.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/images/3fe2e3b74aef2ad08ebc3c756b75ad0c9c869f8f62ee4263132990a1ee8f9400.jpg)

![c2cd5e76327ccdabc4e6106e47df256d8357f472ed5567e8f157819f381276ad.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/images/c2cd5e76327ccdabc4e6106e47df256d8357f472ed5567e8f157819f381276ad.jpg)

![cbbfa9edfbfe4a8d9a090efaf33626f2a4489dfd8733f36b03002cabf4781374.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/images/cbbfa9edfbfe4a8d9a090efaf33626f2a4489dfd8733f36b03002cabf4781374.jpg)

### Tables

![6e582c9a56a664ca00a4df7c1085bb8e75cb5a7d7712a29625e46c859ed80c11.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/tables/6e582c9a56a664ca00a4df7c1085bb8e75cb5a7d7712a29625e46c859ed80c11.jpg)

![83f0ec4a92f0cd05104af49848ec6935365e06293e8a83f1a3060f95b938193e.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/tables/83f0ec4a92f0cd05104af49848ec6935365e06293e8a83f1a3060f95b938193e.jpg)

![a0ac11d8e93dc0fef5014a5158144100fa2181a7784929d069fdb1f5886d84b8.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/tables/a0ac11d8e93dc0fef5014a5158144100fa2181a7784929d069fdb1f5886d84b8.jpg)

![b6cf3d0453271d3f6e43d931db9acd49364d5840e28926ff74f9289ae3cf084a.jpg](../icml_results/1710_The%20Cost%20of%20Information_%20Phase%20Transitions%20in%20Contextual%20Bandits%20with%20Paid%20Observations/tables/b6cf3d0453271d3f6e43d931db9acd49364d5840e28926ff74f9289ae3cf084a.jpg)

<a id="paper-1689"></a>

## Light Up Your Face: A Physically Consistent Dataset and Diffusion Model for Face Fill-Light Enhancement


### Images

![007f6d509bdbd158184d4de61eac48f5f9d460b603720ba8527e60b98d2fa87e.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/007f6d509bdbd158184d4de61eac48f5f9d460b603720ba8527e60b98d2fa87e.jpg)

![0b92607f98629203dc4e724217bb7db716ead46ac76b15dd90e65333acacc0fc.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/0b92607f98629203dc4e724217bb7db716ead46ac76b15dd90e65333acacc0fc.jpg)

![208220b5b68f444de514c62565954bff696be83556852a0692fa3a8db517fbee.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/208220b5b68f444de514c62565954bff696be83556852a0692fa3a8db517fbee.jpg)

![3982ab444aadfb35255f3aaf02edf69dbbf0edda7f57e28261911a8c693a8030.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/3982ab444aadfb35255f3aaf02edf69dbbf0edda7f57e28261911a8c693a8030.jpg)

![41c2bb88359d8ec8f73a88afe861f50a16f2052d55c2e3d1fe9ff153109282a1.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/41c2bb88359d8ec8f73a88afe861f50a16f2052d55c2e3d1fe9ff153109282a1.jpg)

![85e4c4ff534ec6e3474af7d72f0d1edd7d66a5dd5e19ee969be955277d49e4d5.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/85e4c4ff534ec6e3474af7d72f0d1edd7d66a5dd5e19ee969be955277d49e4d5.jpg)

![862ef86d661deabf44d422ddd3f35626c7565e2d50c48fedfd92dae5e8a7f185.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/images/862ef86d661deabf44d422ddd3f35626c7565e2d50c48fedfd92dae5e8a7f185.jpg)

### Tables

![01b58f5af9501aa60285dbf4e1ba900ccde86be57d4e83cd8ce37402dca18444.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/tables/01b58f5af9501aa60285dbf4e1ba900ccde86be57d4e83cd8ce37402dca18444.jpg)

![c1ad76823fbf4b57ef0d370c45b250eca76c5cda2766fa9410d468c0b86c7fad.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/tables/c1ad76823fbf4b57ef0d370c45b250eca76c5cda2766fa9410d468c0b86c7fad.jpg)

![efcb73c52d375b869d61f1d74a4e5cceb7d6c0ad22351f82611f02e339fdcd78.jpg](../icml_results/1711_Light%20Up%20Your%20Face_%20A%20Physically%20Consistent%20Dataset%20and%20Diffusion%20Model%20for%20Face%20Fill-Light%20Enhancem/tables/efcb73c52d375b869d61f1d74a4e5cceb7d6c0ad22351f82611f02e339fdcd78.jpg)

<a id="paper-1690"></a>

## QEDBench: Quantifying the Alignment Gap in Automated Evaluation of University-Level Mathematical Proofs


### Images

![00238aed9e8cc4b6e509e78092dcf66443902bf04819f21c65edd003abedb2d0.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/00238aed9e8cc4b6e509e78092dcf66443902bf04819f21c65edd003abedb2d0.jpg)

![06f38bacc1e72b1d515e3bedb64d4912397933d20c3cc3835d832e45a303c8fe.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/06f38bacc1e72b1d515e3bedb64d4912397933d20c3cc3835d832e45a303c8fe.jpg)

![141a710090b54bfd4e3dac2f7a69809f13e02652ebcb921722e71f1a55885ee7.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/141a710090b54bfd4e3dac2f7a69809f13e02652ebcb921722e71f1a55885ee7.jpg)

![1c61ea3650a08d7c0debb84875449b35bcd38a5224b8b08ac8d068e65c0b5ec7.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/1c61ea3650a08d7c0debb84875449b35bcd38a5224b8b08ac8d068e65c0b5ec7.jpg)

![22112cd9a0d43710ee6ad44b3946dc5d060f8a2f63393f2d7bff00a0ffb3793e.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/22112cd9a0d43710ee6ad44b3946dc5d060f8a2f63393f2d7bff00a0ffb3793e.jpg)

![2bf4d07f6f328704a13f41a9f666374df2b1b2ba993ed04914744b7e54065b7d.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/2bf4d07f6f328704a13f41a9f666374df2b1b2ba993ed04914744b7e54065b7d.jpg)

![34ca81d3d00d18df6e4409d2d3e987c4268ef4cf9a7761207037959d95fbbfce.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/34ca81d3d00d18df6e4409d2d3e987c4268ef4cf9a7761207037959d95fbbfce.jpg)

![3ae3a5e956c841ca58657645b52dfb17c06e6fd5376f2ada370746c4fd4a71cd.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/3ae3a5e956c841ca58657645b52dfb17c06e6fd5376f2ada370746c4fd4a71cd.jpg)

![46fbf868cb63a59e3070f238e953328471fedeb95a73025b6a6e23d7e3f14b0f.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/46fbf868cb63a59e3070f238e953328471fedeb95a73025b6a6e23d7e3f14b0f.jpg)

![628980b3810e94c47eb7bfc269383d19da23ffbbbeb9d4d9a0766e32aea0a18d.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/628980b3810e94c47eb7bfc269383d19da23ffbbbeb9d4d9a0766e32aea0a18d.jpg)

![6991d0cebe94e54b5a1a8712b0ba7998b91483c143a3c514f602fd1ca4bcb499.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/6991d0cebe94e54b5a1a8712b0ba7998b91483c143a3c514f602fd1ca4bcb499.jpg)

![6e7f944f915076e7bc7721c1c3743ccf9ec7f2665e5faf5b5ecdcebb21650bfe.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/6e7f944f915076e7bc7721c1c3743ccf9ec7f2665e5faf5b5ecdcebb21650bfe.jpg)

![84f4ac01a887f300391d55736cbe4161aac7bd3e1304beebdf9050da96d19816.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/84f4ac01a887f300391d55736cbe4161aac7bd3e1304beebdf9050da96d19816.jpg)

![a71d729dd2acd6c554335a2b7cf7e07d97bc9065dc38968a0a666c68cfb33569.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/a71d729dd2acd6c554335a2b7cf7e07d97bc9065dc38968a0a666c68cfb33569.jpg)

![a970986490918db0374d05647a1edabbcb310c20e96e56ee1b72fce91a47964c.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/a970986490918db0374d05647a1edabbcb310c20e96e56ee1b72fce91a47964c.jpg)

![b7e697c6fb30656a5665ebc22a63cbc0579b8ff19b59d52e31a47c3490f975ba.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/b7e697c6fb30656a5665ebc22a63cbc0579b8ff19b59d52e31a47c3490f975ba.jpg)

![c0732e2a97157f57b0e3ed4a715d8e53415c8552a9757eaf8f0dd5c74f5601a6.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/c0732e2a97157f57b0e3ed4a715d8e53415c8552a9757eaf8f0dd5c74f5601a6.jpg)

![c626a906cb7ecb9c56efd4ff9ba24f34fc3a0c770d3b66810e955af0c7e9add2.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/c626a906cb7ecb9c56efd4ff9ba24f34fc3a0c770d3b66810e955af0c7e9add2.jpg)

![c70f04dded635fcea19f0a10aafca1400f209e9de2b91fdd71c6e67722f2f987.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/c70f04dded635fcea19f0a10aafca1400f209e9de2b91fdd71c6e67722f2f987.jpg)

![c830821f0f8e460f24be95c8a7cea8d57db47baeae3cdf45cdf473ebce1db1df.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/c830821f0f8e460f24be95c8a7cea8d57db47baeae3cdf45cdf473ebce1db1df.jpg)

![d20b4f7636a128634f0730fe60d6f30fc0fc7c9ad04d09ddfef4420d53049cb9.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/d20b4f7636a128634f0730fe60d6f30fc0fc7c9ad04d09ddfef4420d53049cb9.jpg)

![f088a113aef52456d2c7be66dee86e1c79b286b4f62473e15af33e6b1e3e5b73.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/f088a113aef52456d2c7be66dee86e1c79b286b4f62473e15af33e6b1e3e5b73.jpg)

![fce6bd035121491cc17aa8277f0e6cc86637fd53bbe9ac96a9280df1c965b987.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/images/fce6bd035121491cc17aa8277f0e6cc86637fd53bbe9ac96a9280df1c965b987.jpg)

### Tables

![4fa80a5cb2dc059bfe4d50620570c0d9a8221f94d65931f60df74cc8e477f20f.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/4fa80a5cb2dc059bfe4d50620570c0d9a8221f94d65931f60df74cc8e477f20f.jpg)

![89109eec1ec62237e9daca11945a5d434e56ceea3b4f7d9fe4cd7388a5580938.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/89109eec1ec62237e9daca11945a5d434e56ceea3b4f7d9fe4cd7388a5580938.jpg)

![abae356523ab1407b2f2c4314a11f4283bc9575f5af14a6a831e64bc1f353245.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/abae356523ab1407b2f2c4314a11f4283bc9575f5af14a6a831e64bc1f353245.jpg)

![ad23d40bca4465228e283563feddf418cdd914740ff3a1cbb022cd2a3bfa47f9.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/ad23d40bca4465228e283563feddf418cdd914740ff3a1cbb022cd2a3bfa47f9.jpg)

![cdd80c1909c89d38703f22c319478d935a99ff84bcfa180fa7533ce0fde39f27.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/cdd80c1909c89d38703f22c319478d935a99ff84bcfa180fa7533ce0fde39f27.jpg)

![e96527f75dfd51e7326ace4a23d60f13322f365c3e8d0a6966c8182280c7102d.jpg](../icml_results/1712_QEDBench_%20Quantifying%20the%20Alignment%20Gap%20in%20Automated%20Evaluation%20of%20University-Level%20Mathematical%20Pro/tables/e96527f75dfd51e7326ace4a23d60f13322f365c3e8d0a6966c8182280c7102d.jpg)

<a id="paper-1691"></a>

## The Differences Between Direct Alignment Algorithms are a Blur


### Images

![06b5d9837183af1a06487529c2b66c444e994590f176870fa6cc7e303ab22124.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/06b5d9837183af1a06487529c2b66c444e994590f176870fa6cc7e303ab22124.jpg)

![0b803731f6c5ff67a6833d3b412544e60d53fac22e5ec1e8eda71f1fce9ce632.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/0b803731f6c5ff67a6833d3b412544e60d53fac22e5ec1e8eda71f1fce9ce632.jpg)

![293e5697bb5b4c83871e4af1b500dba8ac131ab885a63722d17505b0bef6e845.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/293e5697bb5b4c83871e4af1b500dba8ac131ab885a63722d17505b0bef6e845.jpg)

![6451d1085223fe40207bb8dbcb2b0e4779324ecbe25b072692f8522620aae0cd.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/6451d1085223fe40207bb8dbcb2b0e4779324ecbe25b072692f8522620aae0cd.jpg)

![676f0085d71a837e1c48d3bf1a5d976d0645c3d43da20dc5d7db1ddf412d0650.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/676f0085d71a837e1c48d3bf1a5d976d0645c3d43da20dc5d7db1ddf412d0650.jpg)

![ac5882d52d8d2430d22d53599495b1f460bc320a934f2e4a3e46dbbb32a7f5b7.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/ac5882d52d8d2430d22d53599495b1f460bc320a934f2e4a3e46dbbb32a7f5b7.jpg)

![afca252ae64fb4f884fb8792a25d57e8c468d5f5960e223adde69491ef261e7c.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/afca252ae64fb4f884fb8792a25d57e8c468d5f5960e223adde69491ef261e7c.jpg)

![b600798a72a3fcac980341919243ea0e8e6999f23bcf37c8242ef2b05950832e.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/b600798a72a3fcac980341919243ea0e8e6999f23bcf37c8242ef2b05950832e.jpg)

![cdd210c96f578de17a642f649bd49aca03ef9e68ef1f7bf49022c705835780db.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/cdd210c96f578de17a642f649bd49aca03ef9e68ef1f7bf49022c705835780db.jpg)

![e2ea048f077bf14f9087b3f763b293c58fc58def32e4c45a18ddb2bf55f8c322.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/e2ea048f077bf14f9087b3f763b293c58fc58def32e4c45a18ddb2bf55f8c322.jpg)

![e6f8a25e5ac50481dbaff14add925748f674f082fce738d8d53b12b5c979dd46.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/e6f8a25e5ac50481dbaff14add925748f674f082fce738d8d53b12b5c979dd46.jpg)

![f0e67f3b453415e9e3b1deb23efec55578377c81943c643f0e0e2d2254989a1f.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/images/f0e67f3b453415e9e3b1deb23efec55578377c81943c643f0e0e2d2254989a1f.jpg)

### Tables

![09ed5305ba31e1565afff1cdae58f4c327dd2c81578eb85387387d3ea1f5166c.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/09ed5305ba31e1565afff1cdae58f4c327dd2c81578eb85387387d3ea1f5166c.jpg)

![0a54df88b4588d6f556df8aab418d19d6f706b6be0b8855e3e0f55d16ab7a696.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/0a54df88b4588d6f556df8aab418d19d6f706b6be0b8855e3e0f55d16ab7a696.jpg)

![24dad0704db7ea95d0b400fb7ff920fd469f7e2145b0cde3bbfd559e9c898945.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/24dad0704db7ea95d0b400fb7ff920fd469f7e2145b0cde3bbfd559e9c898945.jpg)

![29148ba2bf9b59bfad1a3ee03b044f39f221e6b9d1b62e138f56b2470d9e6c20.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/29148ba2bf9b59bfad1a3ee03b044f39f221e6b9d1b62e138f56b2470d9e6c20.jpg)

![34ed4c4f51efc05df4d78f31b53496ff769d3aa506570576733b88ceced0ce89.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/34ed4c4f51efc05df4d78f31b53496ff769d3aa506570576733b88ceced0ce89.jpg)

![594f214f1a83495c079e9acc27ba7f85ea9976dab0870f55d5263e70b8d15eb8.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/594f214f1a83495c079e9acc27ba7f85ea9976dab0870f55d5263e70b8d15eb8.jpg)

![697d28ae0ab3b8bb4d84ebec5370444f395460e13ec9101abe517e16b9dd05c5.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/697d28ae0ab3b8bb4d84ebec5370444f395460e13ec9101abe517e16b9dd05c5.jpg)

![7f4aa5fc8c99ace72a2879a2ef0b5ded814489f268388507fae4cc437b4c35b8.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/7f4aa5fc8c99ace72a2879a2ef0b5ded814489f268388507fae4cc437b4c35b8.jpg)

![8550830de82041d428f95099a4562a3c856f72ca50e0469ea6aeff68e59c58b9.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/8550830de82041d428f95099a4562a3c856f72ca50e0469ea6aeff68e59c58b9.jpg)

![873811a2d84fea5acd340250219992b052ae73aeaf5accecd3b644c73f230abb.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/873811a2d84fea5acd340250219992b052ae73aeaf5accecd3b644c73f230abb.jpg)

![87f1ac1fb7a713645bb985f813131bcb5e985dd3f6906d57f4fbd7f325eb024d.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/87f1ac1fb7a713645bb985f813131bcb5e985dd3f6906d57f4fbd7f325eb024d.jpg)

![e16eb0c0079fd40f5b3b86173cdd91bd980d731e404bb08a6c1ce2eb0e36e64a.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/e16eb0c0079fd40f5b3b86173cdd91bd980d731e404bb08a6c1ce2eb0e36e64a.jpg)

![e66885ebaac345fcfded5059cf938564b40ee13e63ca7e813f3fa167f1558ae7.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/e66885ebaac345fcfded5059cf938564b40ee13e63ca7e813f3fa167f1558ae7.jpg)

![e7f72ace52a457d5df740178d58a2d4faa876a4d9b94709b0e0f04c4c798dbc0.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/e7f72ace52a457d5df740178d58a2d4faa876a4d9b94709b0e0f04c4c798dbc0.jpg)

![e8e9b5c5054dc34808009d80d450c9cedcfe1728b09d647ab19d8eecefd78b96.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/e8e9b5c5054dc34808009d80d450c9cedcfe1728b09d647ab19d8eecefd78b96.jpg)

![f17f077377e3a1e087445deecc77f1338038b51c8b2cbbb3c7d1780b9b89bc8b.jpg](../icml_results/1713_The%20Differences%20Between%20Direct%20Alignment%20Algorithms%20are%20a%20Blur/tables/f17f077377e3a1e087445deecc77f1338038b51c8b2cbbb3c7d1780b9b89bc8b.jpg)

<a id="paper-1692"></a>

## Towards Spectroscopy: Susceptibility Clusters in Language Models


### Images

![05b4f16f1e16214f75a529b15170eab2134a8a13eb29050200bf43c093b85cbc.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/05b4f16f1e16214f75a529b15170eab2134a8a13eb29050200bf43c093b85cbc.jpg)

![1114ecbf4e9702ccd745f8f032b583390fa063d5b0b80756355958f4c12151d2.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/1114ecbf4e9702ccd745f8f032b583390fa063d5b0b80756355958f4c12151d2.jpg)

![20ced3aedeaff8d3f13da0a6ed85d61ae94a0853ed29d0b4e28a880f58684cc2.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/20ced3aedeaff8d3f13da0a6ed85d61ae94a0853ed29d0b4e28a880f58684cc2.jpg)

![2ec6319d14150b8833c8e06db0824d5f1b32830f986999e85c2a25e949418916.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/2ec6319d14150b8833c8e06db0824d5f1b32830f986999e85c2a25e949418916.jpg)

![3ae7ebd1eb7fb85e2ff1ebe1d5e6a11b4c185daee71da8633196afc4eb27d0b1.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/3ae7ebd1eb7fb85e2ff1ebe1d5e6a11b4c185daee71da8633196afc4eb27d0b1.jpg)

![408add3b79ce7abeffb2f010fd8c839c241e97b8800e3866a1cc96a2ba89402b.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/408add3b79ce7abeffb2f010fd8c839c241e97b8800e3866a1cc96a2ba89402b.jpg)

![44c62284d6babc7f1470e12bf150414b282d7ebbde99ab9d26647e43d936b607.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/44c62284d6babc7f1470e12bf150414b282d7ebbde99ab9d26647e43d936b607.jpg)

![45fc7a646ce158937177d2ffd9c76f10c04e33c65a8b16d4122251f48318d3cc.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/45fc7a646ce158937177d2ffd9c76f10c04e33c65a8b16d4122251f48318d3cc.jpg)

![48391d743c150c409fefb302847f719da96353071264719c59885d23ccb5749d.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/48391d743c150c409fefb302847f719da96353071264719c59885d23ccb5749d.jpg)

![53e58767ea78b6c127e1e8c84600b6dbea20e00ac82389dbede3ab11aef9da9f.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/53e58767ea78b6c127e1e8c84600b6dbea20e00ac82389dbede3ab11aef9da9f.jpg)

![58fc6a4b5960a1f1219d03120301bbc03d9ee2faa3d5542cbd046dd8bb624545.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/58fc6a4b5960a1f1219d03120301bbc03d9ee2faa3d5542cbd046dd8bb624545.jpg)

![6b6ac78cddfa31138b248dd9dcceda6f11a2aac902b145fbecbe41fdebcbec5a.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/6b6ac78cddfa31138b248dd9dcceda6f11a2aac902b145fbecbe41fdebcbec5a.jpg)

![73ecf5ac293d8b76e1b9cfa205bd40cedfe46b0a8d69bff1887b7c376eb672cd.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/73ecf5ac293d8b76e1b9cfa205bd40cedfe46b0a8d69bff1887b7c376eb672cd.jpg)

![766de04458d8caa75f97ab92f44dc9c585104f968ba55c8f857d75ab6bff704d.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/766de04458d8caa75f97ab92f44dc9c585104f968ba55c8f857d75ab6bff704d.jpg)

![785bb94d51b84d5df0b8b9829ba7b9e5ea32e90182382c695d9dd9281058ca2b.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/785bb94d51b84d5df0b8b9829ba7b9e5ea32e90182382c695d9dd9281058ca2b.jpg)

![7b2d3ffad71afa67a3b3c3511e1d7fa9208b7b8df92488a5c8352e5e9f00e7ff.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/7b2d3ffad71afa67a3b3c3511e1d7fa9208b7b8df92488a5c8352e5e9f00e7ff.jpg)

![7b49311c924231d9c147134f6ec54ec34ceebcc39fee4a5c00deb5cb55b56574.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/7b49311c924231d9c147134f6ec54ec34ceebcc39fee4a5c00deb5cb55b56574.jpg)

![7e486d91c5c697701093d182781ae06e18f9f14930590953cf1983151e9f8800.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/7e486d91c5c697701093d182781ae06e18f9f14930590953cf1983151e9f8800.jpg)

![86674a7ba09566cd2eac399174b0aea0188f3975cafdf7e094fd3d8d37f4357e.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/86674a7ba09566cd2eac399174b0aea0188f3975cafdf7e094fd3d8d37f4357e.jpg)

![9425b43446422235a847397d17856d0b616b2afca9a343bd13868c3a9a1ae760.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/9425b43446422235a847397d17856d0b616b2afca9a343bd13868c3a9a1ae760.jpg)

![94f3c14f6bf67d1564af8becea2c1ddb27863861e83fb0a112c63c930f104df4.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/94f3c14f6bf67d1564af8becea2c1ddb27863861e83fb0a112c63c930f104df4.jpg)

![97c10798172cea568825ef08b1e47c601e6d58546588ce3940eb7e0fb06b6136.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/97c10798172cea568825ef08b1e47c601e6d58546588ce3940eb7e0fb06b6136.jpg)

![98d3ec3940e588e8b835bcf163a98a333640c455d607c4e621a1b6631c7f3d3d.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/98d3ec3940e588e8b835bcf163a98a333640c455d607c4e621a1b6631c7f3d3d.jpg)

![9aa06c3ba06d07c794619c264421d5f0be71ebaea7ea1236bfec81dec92100ad.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/9aa06c3ba06d07c794619c264421d5f0be71ebaea7ea1236bfec81dec92100ad.jpg)

![9c84a5bee250fa2b36459aca1ad3433903f93f5adf73aaf3858cb354a7f0396b.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/9c84a5bee250fa2b36459aca1ad3433903f93f5adf73aaf3858cb354a7f0396b.jpg)

![a99c3aab30d10c59604f0c09475133ff9a6a344785f34ed5aed280bad8d33cef.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/a99c3aab30d10c59604f0c09475133ff9a6a344785f34ed5aed280bad8d33cef.jpg)

![b29e8cca3c693913d7df2b820d9277fcbbc14878e5aa4a6809e8e760c248dd41.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/b29e8cca3c693913d7df2b820d9277fcbbc14878e5aa4a6809e8e760c248dd41.jpg)

![b2a8eb385343cbed79e79b0cd92d21d4a810aba70be3660ea17819a09fb1889f.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/b2a8eb385343cbed79e79b0cd92d21d4a810aba70be3660ea17819a09fb1889f.jpg)

![b4a88f602f07eddc0850fb9cb1e857ea12b27f3100009602526d876b9a125fae.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/b4a88f602f07eddc0850fb9cb1e857ea12b27f3100009602526d876b9a125fae.jpg)

![bb5725e0e12c205463df09ff779760f1e0e4f11ad798965c48c716022fe97f38.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/bb5725e0e12c205463df09ff779760f1e0e4f11ad798965c48c716022fe97f38.jpg)

![c04656f22048a170a6131de73680c53033b5ec229e27edb77e4f03e100d8fa62.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/c04656f22048a170a6131de73680c53033b5ec229e27edb77e4f03e100d8fa62.jpg)

![cc0119ee71fe5661c49e361cbb86ab4c25f8e39ed9609bb2f993a1901fe52c76.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/cc0119ee71fe5661c49e361cbb86ab4c25f8e39ed9609bb2f993a1901fe52c76.jpg)

![d0c3a1b042e37be090a1f1fa55104204f74489115d2c4b3b2b6d46998ddd65bb.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/d0c3a1b042e37be090a1f1fa55104204f74489115d2c4b3b2b6d46998ddd65bb.jpg)

![d33125aa9e3b9559f60057399db315585fc48e1687618b810d9ba55d75cf29b8.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/d33125aa9e3b9559f60057399db315585fc48e1687618b810d9ba55d75cf29b8.jpg)

![ea35f68da80eba04f8ccf3602a1c04486c25c4e1abf3b35825e3e618f47d441c.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/images/ea35f68da80eba04f8ccf3602a1c04486c25c4e1abf3b35825e3e618f47d441c.jpg)

### Tables

![18f3d32fc70ef7e3de2970f2f1f4f9a62dc915a6a08d5ba813c02e44dff562ee.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/18f3d32fc70ef7e3de2970f2f1f4f9a62dc915a6a08d5ba813c02e44dff562ee.jpg)

![1b4a8b56447ba12ae6dd94d0b90355e3605bc7f97330086adbc34dcf04734c9b.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/1b4a8b56447ba12ae6dd94d0b90355e3605bc7f97330086adbc34dcf04734c9b.jpg)

![414fca91dfa19d2a11334404a1977381f683d108a4099572a809fa369e9762df.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/414fca91dfa19d2a11334404a1977381f683d108a4099572a809fa369e9762df.jpg)

![769b84a7539945ac54d3bb02bfa42eadbed1368c2569a345e45f040e7eed5f81.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/769b84a7539945ac54d3bb02bfa42eadbed1368c2569a345e45f040e7eed5f81.jpg)

![81b46b7e9240b22494bec3a91af8c110c3ca32b9baf67fb64f3ad7ce404df6b8.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/81b46b7e9240b22494bec3a91af8c110c3ca32b9baf67fb64f3ad7ce404df6b8.jpg)

![e3411b19f5dc6ce38184cc0573f909083287613eda378d6e8ca018a3e6e7bb63.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/e3411b19f5dc6ce38184cc0573f909083287613eda378d6e8ca018a3e6e7bb63.jpg)

![e38977b6928f5dbfb4a1d8567acf7994797214f7ce3e41a1beebef9fe2e51bcc.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/e38977b6928f5dbfb4a1d8567acf7994797214f7ce3e41a1beebef9fe2e51bcc.jpg)

![e3ef361ceaa1b79ad06e45269afcdb7ffcc96b3b4cc6a72cedd8c68369978856.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/e3ef361ceaa1b79ad06e45269afcdb7ffcc96b3b4cc6a72cedd8c68369978856.jpg)

![efdf842e19295ab3c955208c6faadf99b6c80a90d8b3eaa087ba7c3984bb576b.jpg](../icml_results/1714_Towards%20Spectroscopy_%20Susceptibility%20Clusters%20in%20Language%20Models/tables/efdf842e19295ab3c955208c6faadf99b6c80a90d8b3eaa087ba7c3984bb576b.jpg)

<a id="paper-1693"></a>

## LATMiX: Learnable Affine Transformations for Microscaling Quantization of LLMs


### Images

![1247c6d457d46881ae52f53887be498d6c03e12704e53afea24591c66f40c27b.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/1247c6d457d46881ae52f53887be498d6c03e12704e53afea24591c66f40c27b.jpg)

![3d64c2af5817aed8806812d40a0c958d58e60180f1228cfbc06bb87e5baa6dce.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/3d64c2af5817aed8806812d40a0c958d58e60180f1228cfbc06bb87e5baa6dce.jpg)

![89c096811bf0e124be27d207d58d9e870cef6ec3b9c1cc1f81912b9817aecaaa.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/89c096811bf0e124be27d207d58d9e870cef6ec3b9c1cc1f81912b9817aecaaa.jpg)

![8ec6b5b9cb63c35a3e03ec4821a23721faa09377a0f987f5b4e1e26bb26996f4.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/8ec6b5b9cb63c35a3e03ec4821a23721faa09377a0f987f5b4e1e26bb26996f4.jpg)

![9a6111555012360e569f8d2f1393e262f18336c6ed59853f6811c1fc7b2a9208.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/9a6111555012360e569f8d2f1393e262f18336c6ed59853f6811c1fc7b2a9208.jpg)

![b11e6f36b81aa6d09f32f89dc68716c5186ddfffe8553ecaca1191b6c03cc2e3.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/b11e6f36b81aa6d09f32f89dc68716c5186ddfffe8553ecaca1191b6c03cc2e3.jpg)

![c8a35761c189bae5c4b38e3594ab3544130d22f6f85f1419ca8fc1cfa211a1c2.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/c8a35761c189bae5c4b38e3594ab3544130d22f6f85f1419ca8fc1cfa211a1c2.jpg)

![cdd7d4b7361a62dff3a66f49b46bb5836e24553d43a74e21654d730617fdcfa0.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/images/cdd7d4b7361a62dff3a66f49b46bb5836e24553d43a74e21654d730617fdcfa0.jpg)

### Tables

![0a783330d10a0a4ec8901449b481438fd05158cafe9aff203b0764ce73e7bb93.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/0a783330d10a0a4ec8901449b481438fd05158cafe9aff203b0764ce73e7bb93.jpg)

![120c75e9e54a2c5d38eb2c257916400b5163ec8fe8b8176cf8391a1b8f2a15c4.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/120c75e9e54a2c5d38eb2c257916400b5163ec8fe8b8176cf8391a1b8f2a15c4.jpg)

![3e8498f04d8a97eab095b4b8d0f6bcc1d1950150fb7d92b8e9f7029a99c2667b.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/3e8498f04d8a97eab095b4b8d0f6bcc1d1950150fb7d92b8e9f7029a99c2667b.jpg)

![4c7c043dfc4fd921e23b8b2afea20cfb535831e3e7b391daf92ad3963567c361.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/4c7c043dfc4fd921e23b8b2afea20cfb535831e3e7b391daf92ad3963567c361.jpg)

![5015bc5744f6ab95ee510b107f31c0f686c5cd00c96ed1813e328afc23b1b8e9.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/5015bc5744f6ab95ee510b107f31c0f686c5cd00c96ed1813e328afc23b1b8e9.jpg)

![5f51599abffaeca981cc841f01b6c520f4a1f4f8c6368bc9c3add05475ee057f.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/5f51599abffaeca981cc841f01b6c520f4a1f4f8c6368bc9c3add05475ee057f.jpg)

![60c46fa928c62c2ef76394101e9a6aa55d560386c5272dc67e4a2fe80c000e0b.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/60c46fa928c62c2ef76394101e9a6aa55d560386c5272dc67e4a2fe80c000e0b.jpg)

![63f2a91b48204410964885685ff3dd25822c6b12ec4b81dc7c1b8d7431e02511.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/63f2a91b48204410964885685ff3dd25822c6b12ec4b81dc7c1b8d7431e02511.jpg)

![644cd0266b4213efe4bebb91fc165def9c3cbeff75ac56451b60e5703e149164.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/644cd0266b4213efe4bebb91fc165def9c3cbeff75ac56451b60e5703e149164.jpg)

![716370300c22267083878d72fd357b01f89ed8e91e61682f6f8373e6a602f800.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/716370300c22267083878d72fd357b01f89ed8e91e61682f6f8373e6a602f800.jpg)

![7994f10e421a3a0c624b37fa22a72312a239170162026207c0cbb8bf4cef27e3.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/7994f10e421a3a0c624b37fa22a72312a239170162026207c0cbb8bf4cef27e3.jpg)

![7a604e9c6f9ba70ca886b48b12fab8b65abf3073139697bfc423415574766a15.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/7a604e9c6f9ba70ca886b48b12fab8b65abf3073139697bfc423415574766a15.jpg)

![7bbcbc20f5765928d078d554d09c45e03d0c909e159d7f5f0530bb908d1cdd79.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/7bbcbc20f5765928d078d554d09c45e03d0c909e159d7f5f0530bb908d1cdd79.jpg)

![7da5c23b0f6a0d56cc90695bb7c835b923c27ed492b180e526777e88c75b2408.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/7da5c23b0f6a0d56cc90695bb7c835b923c27ed492b180e526777e88c75b2408.jpg)

![8dbfbc00c24b7703ce7cd3161611f36af8832e6c18dc493e2cadc6bd9d9e2d13.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/8dbfbc00c24b7703ce7cd3161611f36af8832e6c18dc493e2cadc6bd9d9e2d13.jpg)

![a95a4a0ad594ed6cad06a12ce48553bb6ffa873daad56349bbcd2e000d536838.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/a95a4a0ad594ed6cad06a12ce48553bb6ffa873daad56349bbcd2e000d536838.jpg)

![b063adb4c5793a0854ec1f636205028fb527bfe3f6ea2b84795f08e24cefc646.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/b063adb4c5793a0854ec1f636205028fb527bfe3f6ea2b84795f08e24cefc646.jpg)

![b7d53eb9fd3dda720ead25fc1159c21609d88f0e5c0117624a19f4fb5ce3392c.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/b7d53eb9fd3dda720ead25fc1159c21609d88f0e5c0117624a19f4fb5ce3392c.jpg)

![cded00b8dbba841be26c6e4c4c9d55e646c9020337b8a119c0ef7153fd043eb9.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/cded00b8dbba841be26c6e4c4c9d55e646c9020337b8a119c0ef7153fd043eb9.jpg)

![da8afb4aaedb3fb6d3189c4d3af77b00676420fe0fc65be3c59f72e15aaedf66.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/da8afb4aaedb3fb6d3189c4d3af77b00676420fe0fc65be3c59f72e15aaedf66.jpg)

![dab4b353f4757bc64dd950de3c6960b7089fd34fc8f824cccd431c8b29e99ba7.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/dab4b353f4757bc64dd950de3c6960b7089fd34fc8f824cccd431c8b29e99ba7.jpg)

![e86fdbd4e44ea7946c20e2752ef6a8d28a9a4fbf7664d51f19c598a0bc95ed4d.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/e86fdbd4e44ea7946c20e2752ef6a8d28a9a4fbf7664d51f19c598a0bc95ed4d.jpg)

![f426bbc181168c59419c1782543c6f23d2d0a8e5eabb278c009351589b97147d.jpg](../icml_results/1715_LATMiX_%20Learnable%20Affine%20Transformations%20for%20Microscaling%20Quantization%20of%20LLMs/tables/f426bbc181168c59419c1782543c6f23d2d0a8e5eabb278c009351589b97147d.jpg)

<a id="paper-1694"></a>

## TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning


### Images

![1f98dd5007e5cc1227c9a303f8eaad94a52b2ac57d5a8b23bd34529af07d2451.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/1f98dd5007e5cc1227c9a303f8eaad94a52b2ac57d5a8b23bd34529af07d2451.jpg)

![4e071e5fed7c4a7ee9102ce4e55ad414aa7f4dcbb9842c45a5bdcaa8dcf54d28.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/4e071e5fed7c4a7ee9102ce4e55ad414aa7f4dcbb9842c45a5bdcaa8dcf54d28.jpg)

![7423415ebed27ec721812026c03ebc61f88656a4f6de8a837bd076d1deeaaaca.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/7423415ebed27ec721812026c03ebc61f88656a4f6de8a837bd076d1deeaaaca.jpg)

![8242b46fe4273c94a6b8fb6bd7722fa0a9c3e570f1370ecd2db276f82da47d09.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/8242b46fe4273c94a6b8fb6bd7722fa0a9c3e570f1370ecd2db276f82da47d09.jpg)

![84448bcbfa7d97ed5063d17d7ef0a302433db93d5b3c1596f029802e0ae20490.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/84448bcbfa7d97ed5063d17d7ef0a302433db93d5b3c1596f029802e0ae20490.jpg)

![a076139d9e6154d94c4dc1b0bfbe7a26b6c67618c77f3cb48ea2402f932d4fd9.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/a076139d9e6154d94c4dc1b0bfbe7a26b6c67618c77f3cb48ea2402f932d4fd9.jpg)

![b2bc4e6dccb6b652e88024ca2b097ec91252f5f2b287294ef0bd4bf0ae137494.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/b2bc4e6dccb6b652e88024ca2b097ec91252f5f2b287294ef0bd4bf0ae137494.jpg)

![c5592175f3365c499123e124d0e71f8583d8da3dd47e26355d131033aa176e2d.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/c5592175f3365c499123e124d0e71f8583d8da3dd47e26355d131033aa176e2d.jpg)

![e2991c036d0779f250c165a186423a010b6fd4fbf88f643743f823562b6ace48.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/e2991c036d0779f250c165a186423a010b6fd4fbf88f643743f823562b6ace48.jpg)

![ec65f75976a7e312091338225647ae6e97380125335bd46d8a8b7d15a372127a.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/ec65f75976a7e312091338225647ae6e97380125335bd46d8a8b7d15a372127a.jpg)

![f436d5faf302b963126644a21ddc2801fb9d8926c27cfac1ec57c0054fb7f17f.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/images/f436d5faf302b963126644a21ddc2801fb9d8926c27cfac1ec57c0054fb7f17f.jpg)

### Tables

![05cd84f6300c9d107b018f2fab21bd214b6e562fbf1da71b2cc34f3fab41e926.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/05cd84f6300c9d107b018f2fab21bd214b6e562fbf1da71b2cc34f3fab41e926.jpg)

![1353f2d847787a6c2d245a8a70b8ae411d5304ea82ca71d09906c82c341882ae.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/1353f2d847787a6c2d245a8a70b8ae411d5304ea82ca71d09906c82c341882ae.jpg)

![2485fe3cdd02ded3d0b680f7503f29ab1b84b0f1f96829592c70ebf527d29296.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/2485fe3cdd02ded3d0b680f7503f29ab1b84b0f1f96829592c70ebf527d29296.jpg)

![29bec15c8506f18705479cf9577425a4688a4b62ebb78e162e2828b9fdd68514.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/29bec15c8506f18705479cf9577425a4688a4b62ebb78e162e2828b9fdd68514.jpg)

![2b843eded27c2f41247fa2fb9867e28e3d3eea0945190ad98e7c9c788c639e1e.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/2b843eded27c2f41247fa2fb9867e28e3d3eea0945190ad98e7c9c788c639e1e.jpg)

![2e9edbf3152c3790333184de85f9b305117cfafd49baebf58026b9486b5f7a40.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/2e9edbf3152c3790333184de85f9b305117cfafd49baebf58026b9486b5f7a40.jpg)

![46a7c7d799a104246cade945b33314efd376ac8c7e6ae80d0e3101bed03c1267.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/46a7c7d799a104246cade945b33314efd376ac8c7e6ae80d0e3101bed03c1267.jpg)

![639417674e8c29db442203e34baf78af363a6c84250ed930422131c206a5ea7c.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/639417674e8c29db442203e34baf78af363a6c84250ed930422131c206a5ea7c.jpg)

![66fc8be42343770c64b6a28547cbc782dad92d7d50abdc157c99080b7935cc42.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/66fc8be42343770c64b6a28547cbc782dad92d7d50abdc157c99080b7935cc42.jpg)

![682931c2bb93b8ea7cf566090997009151c5dcdbc33d52a53eac3904d49e514f.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/682931c2bb93b8ea7cf566090997009151c5dcdbc33d52a53eac3904d49e514f.jpg)

![6e087d2bed0a492c63e8620599b8ce9cc071c5d0b441c907f32d9e1d0a30d91a.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/6e087d2bed0a492c63e8620599b8ce9cc071c5d0b441c907f32d9e1d0a30d91a.jpg)

![751bc9f98eab53fe64c130ee2545eabe49d9f16315aa40f68c912ccc29c88bd9.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/751bc9f98eab53fe64c130ee2545eabe49d9f16315aa40f68c912ccc29c88bd9.jpg)

![80c1e87227cc728e7a3fb6edd1179b8ac093d6ddb9829a84d95ab3e99dd65edf.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/80c1e87227cc728e7a3fb6edd1179b8ac093d6ddb9829a84d95ab3e99dd65edf.jpg)

![82ee9339a6c2cbe2b076105ed5900ed615507c0c3393016279ae9ef35acf4b6c.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/82ee9339a6c2cbe2b076105ed5900ed615507c0c3393016279ae9ef35acf4b6c.jpg)

![8e5f2a0b2ae85f7a00d9d8c1ddfef886aed3dc9fc62a29ee3b25445eb7fc8149.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/8e5f2a0b2ae85f7a00d9d8c1ddfef886aed3dc9fc62a29ee3b25445eb7fc8149.jpg)

![9701aaf6e88a4a4fae154aea6d4c1afd9fe153332e04f6b323b4707eb107f7c4.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/9701aaf6e88a4a4fae154aea6d4c1afd9fe153332e04f6b323b4707eb107f7c4.jpg)

![a925cc1ffe8c17da70b5034e1710a358e12af1b25f7dd60a771be015752e9943.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/a925cc1ffe8c17da70b5034e1710a358e12af1b25f7dd60a771be015752e9943.jpg)

![b0b4ed52f496e0217af5a322b553211492d87763347bc6e3e7759d5f3fa3843b.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/b0b4ed52f496e0217af5a322b553211492d87763347bc6e3e7759d5f3fa3843b.jpg)

![b583ebadbfff756437e985ae8abf7c6464be17e9220cf4a0d5bf1d1e3bc3570a.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/b583ebadbfff756437e985ae8abf7c6464be17e9220cf4a0d5bf1d1e3bc3570a.jpg)

![c0ab70e664206df591391e5605de422389ee75efd61e5a36dc31089ae2a4b863.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/c0ab70e664206df591391e5605de422389ee75efd61e5a36dc31089ae2a4b863.jpg)

![c5a5a0002f9b752535407941cf71a82b3f53c742e67b91a7e42c8baf7acb5dfd.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/c5a5a0002f9b752535407941cf71a82b3f53c742e67b91a7e42c8baf7acb5dfd.jpg)

![cc7735709a33f73545969fcd3029046e40f7e1b1ae30f8598f689a2fb0a1faa7.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/cc7735709a33f73545969fcd3029046e40f7e1b1ae30f8598f689a2fb0a1faa7.jpg)

![ce7523ae306419bb3d1242770866647774ccf8a4483b8de94aabd7c94036c96a.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/ce7523ae306419bb3d1242770866647774ccf8a4483b8de94aabd7c94036c96a.jpg)

![d973bc411342b26796adcb1e86060f16f6019d215ffb960eb6d51e3050d1a37d.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/d973bc411342b26796adcb1e86060f16f6019d215ffb960eb6d51e3050d1a37d.jpg)

![f5de03415a08673a9a6950b10574337d42f7d6ff5c4ddac4489133ad68e5abc7.jpg](../icml_results/1716_TabPack_%20Efficient%20Hyperparameter%20Ensembles%20for%20Tabular%20Deep%20Learning/tables/f5de03415a08673a9a6950b10574337d42f7d6ff5c4ddac4489133ad68e5abc7.jpg)

<a id="paper-1695"></a>

## The Illusion of Generalization in Tabular Language Models


### Images

![3499569f661f4dbaa9d7292a6e00f04623cbe42ad61bfb15f81539aca4f4d005.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/images/3499569f661f4dbaa9d7292a6e00f04623cbe42ad61bfb15f81539aca4f4d005.jpg)

![5ad0161beb21c370e023c4cfca8a684a5f2c419ed5f64ecbec0bbcb7155e1599.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/images/5ad0161beb21c370e023c4cfca8a684a5f2c419ed5f64ecbec0bbcb7155e1599.jpg)

![a31a33155c1983725949728573213f20eb26d218fbc386005c77fc99a3d2d1ab.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/images/a31a33155c1983725949728573213f20eb26d218fbc386005c77fc99a3d2d1ab.jpg)

![b242e4b0b15a34751b3d725abe4d6d23d85c021c4205515504a9c97dfa322dd7.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/images/b242e4b0b15a34751b3d725abe4d6d23d85c021c4205515504a9c97dfa322dd7.jpg)

![bdba8948fa4f9c8f6216de8f76b8508e1f6d3e893b6f10b58925bfd5fc586e6f.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/images/bdba8948fa4f9c8f6216de8f76b8508e1f6d3e893b6f10b58925bfd5fc586e6f.jpg)

### Tables

![1c649ca0c821e7b239370eb4c7809167a6e9211c785f81171a4a3fa8879a805b.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/1c649ca0c821e7b239370eb4c7809167a6e9211c785f81171a4a3fa8879a805b.jpg)

![2204c9b6e92611cfc986d82729bb77563c3f07545c8498572d79bfd542792a22.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/2204c9b6e92611cfc986d82729bb77563c3f07545c8498572d79bfd542792a22.jpg)

![2985c0298291ee885720f49a5c5abbd66ecc175e066560737aa70a6a1a2664c3.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/2985c0298291ee885720f49a5c5abbd66ecc175e066560737aa70a6a1a2664c3.jpg)

![2cc73ce83caa6dea875dbf7751f67f6ec4d6fdd660220186b60150b547dd7584.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/2cc73ce83caa6dea875dbf7751f67f6ec4d6fdd660220186b60150b547dd7584.jpg)

![442917922f43f6fe591fee6d4a16e7556e0deb5a3ed46543521cd8bc69958f5c.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/442917922f43f6fe591fee6d4a16e7556e0deb5a3ed46543521cd8bc69958f5c.jpg)

![4ec8b5c730ab6903950ce8656b11a371598c9816a0fcf18e999ae9fea6d2ad00.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/4ec8b5c730ab6903950ce8656b11a371598c9816a0fcf18e999ae9fea6d2ad00.jpg)

![5f49da306dd18c94a9a856fa06a749b42873b05621c0883d32d885c32761c572.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/5f49da306dd18c94a9a856fa06a749b42873b05621c0883d32d885c32761c572.jpg)

![8a3f03702f0931ddaf0e126987c594ae1bf104aaa1535acd3dbb7b0be30e5414.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/8a3f03702f0931ddaf0e126987c594ae1bf104aaa1535acd3dbb7b0be30e5414.jpg)

![994209da9de7e7dead13f0375e40450537fc12431f1ed53aad8b78f16fa2ae1c.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/994209da9de7e7dead13f0375e40450537fc12431f1ed53aad8b78f16fa2ae1c.jpg)

![acbfa22451fa16b134b168b90fa2e78c1018b89b08045e3fbdb873b0579cc17f.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/acbfa22451fa16b134b168b90fa2e78c1018b89b08045e3fbdb873b0579cc17f.jpg)

![b276d02e7dc0b24424e95779cb4a7cac50ed6da03366248a5c55db63bc7a7fdd.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/b276d02e7dc0b24424e95779cb4a7cac50ed6da03366248a5c55db63bc7a7fdd.jpg)

![b6bf292d26175ff435e5769c591aee273275af775577dca55f98278ea027a3fa.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/b6bf292d26175ff435e5769c591aee273275af775577dca55f98278ea027a3fa.jpg)

![d7e495c5f7239f33773e7624a0069430845f73d0d2541189f3adba9c33e07440.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/d7e495c5f7239f33773e7624a0069430845f73d0d2541189f3adba9c33e07440.jpg)

![d9a5e9390c4a9d64a9ccec5a41f979486d8cc8cdf37997beaa30cb32daf46837.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/d9a5e9390c4a9d64a9ccec5a41f979486d8cc8cdf37997beaa30cb32daf46837.jpg)

![e051b953d9a6fe37f7fbc125003b74e496f98991e809fe52090fca7ee46765c9.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/e051b953d9a6fe37f7fbc125003b74e496f98991e809fe52090fca7ee46765c9.jpg)

![f713b74cd886cfe1a123ca4211577b328f412161781dbaac0ba2a51e03c66077.jpg](../icml_results/1717_The%20Illusion%20of%20Generalization%20in%20Tabular%20Language%20Models/tables/f713b74cd886cfe1a123ca4211577b328f412161781dbaac0ba2a51e03c66077.jpg)

<a id="paper-1696"></a>

## Certifying Graph Neural Networks Against Label and Structure Poisoning


### Images

![14beef8193aa046bc5f55c0fb9fe3751ed978308543f3d0c6d8a5e7c3d3987d6.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/14beef8193aa046bc5f55c0fb9fe3751ed978308543f3d0c6d8a5e7c3d3987d6.jpg)

![24838bfc913a2170f8cc5769f0510dfcba94fc29c55b6c678f547c242c4ebad7.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/24838bfc913a2170f8cc5769f0510dfcba94fc29c55b6c678f547c242c4ebad7.jpg)

![24ba5fe3b267f508df948e67001dd12b1eff2f242fd49b9238f8f6fecbc4c5e2.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/24ba5fe3b267f508df948e67001dd12b1eff2f242fd49b9238f8f6fecbc4c5e2.jpg)

![328a6215424384936ad2a2dc88d053db442d4619b7c8b3702582fb52284e4475.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/328a6215424384936ad2a2dc88d053db442d4619b7c8b3702582fb52284e4475.jpg)

![3c3493b3c30c1242dbec24796bf13d892e2e06ee9485144c0b0a5619c12791f8.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/3c3493b3c30c1242dbec24796bf13d892e2e06ee9485144c0b0a5619c12791f8.jpg)

![5b6b3f4862ea7fd0f191a629225036d128369b09d226c4ed86d6449de264ff2d.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/5b6b3f4862ea7fd0f191a629225036d128369b09d226c4ed86d6449de264ff2d.jpg)

![653e79c64819b5e6aea558987ae774bfd82a0781b1c8f61fea82afc4a4336700.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/653e79c64819b5e6aea558987ae774bfd82a0781b1c8f61fea82afc4a4336700.jpg)

![6d8d488875bc7efdc97a688c5ecad136f81aa9c91357e60946ff150d4d2fa49f.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/6d8d488875bc7efdc97a688c5ecad136f81aa9c91357e60946ff150d4d2fa49f.jpg)

![6e1ca521bdade53e7813337b34729415e199b011ae4bd8d3598e95d4a8bbab8d.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/6e1ca521bdade53e7813337b34729415e199b011ae4bd8d3598e95d4a8bbab8d.jpg)

![970b2bb5c22e70ab52894d3daf08e388d3b2cffef1963c3c754205aa4fe7ac93.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/970b2bb5c22e70ab52894d3daf08e388d3b2cffef1963c3c754205aa4fe7ac93.jpg)

![a586e4a05866080ecf3dd9e1221c316539792c83e169a4fb16d91f9947261b01.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/a586e4a05866080ecf3dd9e1221c316539792c83e169a4fb16d91f9947261b01.jpg)

![a7a3dac9333848681fadad623ab7aebd10124a992097765015cad3acc5e2d098.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/a7a3dac9333848681fadad623ab7aebd10124a992097765015cad3acc5e2d098.jpg)

![b334dacdb9d74aa63a40f5a41373190b1e39e60f6eb8d2c942044ac7f6f0bbae.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/b334dacdb9d74aa63a40f5a41373190b1e39e60f6eb8d2c942044ac7f6f0bbae.jpg)

![bec42017dd2c7e1c408eeb357466ed8edd291e3f73360e54d3024008d8af9507.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/bec42017dd2c7e1c408eeb357466ed8edd291e3f73360e54d3024008d8af9507.jpg)

![d5914496a48d056aa705ea49e0f6f17ffd738cd4ee725eb17de11f467eda2afa.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/d5914496a48d056aa705ea49e0f6f17ffd738cd4ee725eb17de11f467eda2afa.jpg)

![e156923b78926271a158e5aa8f3f9a4dce3c06075455c44309f592aded2c73c9.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/e156923b78926271a158e5aa8f3f9a4dce3c06075455c44309f592aded2c73c9.jpg)

![e3d521fc4263275981a6a5bfbe6789551aa5c97efc1e10d4a15760f30cfac43f.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/e3d521fc4263275981a6a5bfbe6789551aa5c97efc1e10d4a15760f30cfac43f.jpg)

![e51ca9fe484571fea434ac5c29adc40c2884d9ff80a8f1b2c35bd22c11761400.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/e51ca9fe484571fea434ac5c29adc40c2884d9ff80a8f1b2c35bd22c11761400.jpg)

![eab2fc8f4b7515fd058d42859cd00507a750d8601b909843f5eea722cc6755b6.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/eab2fc8f4b7515fd058d42859cd00507a750d8601b909843f5eea722cc6755b6.jpg)

![eb71d7c677391066249737b68230ea13edd2a3daf8311a142fa9f3e14668d6e7.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/images/eb71d7c677391066249737b68230ea13edd2a3daf8311a142fa9f3e14668d6e7.jpg)

### Tables

![008bf2e91adb4a8044e15344a914ea082b9f2dae6306e9cfb9c68c90b7d96b4f.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/008bf2e91adb4a8044e15344a914ea082b9f2dae6306e9cfb9c68c90b7d96b4f.jpg)

![07166d8fb47cbcbbef706bdeea962458dc1f7f743131cf80d884d4dc8bcf8a87.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/07166d8fb47cbcbbef706bdeea962458dc1f7f743131cf80d884d4dc8bcf8a87.jpg)

![42d26e39fe27b67398bea562f73c703a9e551a23d71109e8e6d93e9692abf3ea.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/42d26e39fe27b67398bea562f73c703a9e551a23d71109e8e6d93e9692abf3ea.jpg)

![b7d165e095d811e81452ba1dbca5e091cc2a1125ebed72e708c0c34254590e24.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/b7d165e095d811e81452ba1dbca5e091cc2a1125ebed72e708c0c34254590e24.jpg)

![f874717c202b6e41f647e4a1c846695846c42f57faa6db49a60e4bed1866088b.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/f874717c202b6e41f647e4a1c846695846c42f57faa6db49a60e4bed1866088b.jpg)

![fbf5ac7e84fa867197d848b3cc02d8559377b2e3e6bd84663a7e96ca9b3c9ed3.jpg](../icml_results/1718_Certifying%20Graph%20Neural%20Networks%20Against%20Label%20and%20Structure%20Poisoning/tables/fbf5ac7e84fa867197d848b3cc02d8559377b2e3e6bd84663a7e96ca9b3c9ed3.jpg)

<a id="paper-1697"></a>

## Byte Pair Encoding for Efficient Time Series Forecasting


### Images

![047ab307c76fc931826717f38c8b752a8a96a9eb44b8449137365dbaf3201946.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/047ab307c76fc931826717f38c8b752a8a96a9eb44b8449137365dbaf3201946.jpg)

![1470dc56d47ab547dc054247509bfa5b782e7f79dea072e76c47a2b12734996b.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/1470dc56d47ab547dc054247509bfa5b782e7f79dea072e76c47a2b12734996b.jpg)

![15756ad86312568bd806b836204e30ba5d4cc16f3b0507613270f68589a06484.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/15756ad86312568bd806b836204e30ba5d4cc16f3b0507613270f68589a06484.jpg)

![1bc40b8cccabf13041479c5a6234c77bc32c8395c9c116e0316bdc1da815bb22.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/1bc40b8cccabf13041479c5a6234c77bc32c8395c9c116e0316bdc1da815bb22.jpg)

![325ddb1269a8a9aecb4935829d06f484a94147f9a6ca4fd4d020b6ec4c459ab8.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/325ddb1269a8a9aecb4935829d06f484a94147f9a6ca4fd4d020b6ec4c459ab8.jpg)

![340d749da1db887990d9039a2becbb1574de608ba7c66b98637e91cf27a47745.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/340d749da1db887990d9039a2becbb1574de608ba7c66b98637e91cf27a47745.jpg)

![384af17511826e294b8f9fd779cc20ef853a799846821aece8a006c6617e17d9.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/384af17511826e294b8f9fd779cc20ef853a799846821aece8a006c6617e17d9.jpg)

![4f6081d2c2e9f004ad1ec2b3380a95658842d528c63370b56640543e5cfff3bd.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/4f6081d2c2e9f004ad1ec2b3380a95658842d528c63370b56640543e5cfff3bd.jpg)

![4fc344c60f356c9ac56fb80459255339aa85ad5df29e046943bce4a2230457bf.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/4fc344c60f356c9ac56fb80459255339aa85ad5df29e046943bce4a2230457bf.jpg)

![52c5b6c24e4b627c87ec036293a71e1d3d809ca82a2f1cb9df07c33f97e38d4f.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/52c5b6c24e4b627c87ec036293a71e1d3d809ca82a2f1cb9df07c33f97e38d4f.jpg)

![5758a7afcbf231234b8afe4f29c8fc35733afa31f1688d4ca122a68e02c5f603.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/5758a7afcbf231234b8afe4f29c8fc35733afa31f1688d4ca122a68e02c5f603.jpg)

![761cfeb7cee912c587bc216fb5305e0d93aea464372484f16a6d54582fb262c2.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/761cfeb7cee912c587bc216fb5305e0d93aea464372484f16a6d54582fb262c2.jpg)

![8f21ac125015fd90635b54509561a5ef36f3e00a9d371ea495ea09c8226502c9.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/8f21ac125015fd90635b54509561a5ef36f3e00a9d371ea495ea09c8226502c9.jpg)

![9199fc19521c3932112f671779e23ec2ef9cb47be65fbc6f5b35f85f560346d6.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/9199fc19521c3932112f671779e23ec2ef9cb47be65fbc6f5b35f85f560346d6.jpg)

![9833652d11b6c1ff06748e9797214ea8e9692deb5b9a2fdb662597d4ec38aed7.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/9833652d11b6c1ff06748e9797214ea8e9692deb5b9a2fdb662597d4ec38aed7.jpg)

![b17822671c22598d26a6395ee91fc89fecf6b12d2a09aaca8d3f05db41e49f1c.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/b17822671c22598d26a6395ee91fc89fecf6b12d2a09aaca8d3f05db41e49f1c.jpg)

![b4bf8fb7ffb0649a5cb8bf19ba0cf74c4f20725bd7065ebbac88c124eae393b2.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/b4bf8fb7ffb0649a5cb8bf19ba0cf74c4f20725bd7065ebbac88c124eae393b2.jpg)

![c2e318a4054bb91fadd7e2ab4a1224be51693d21e5aaa09018c0a70673933225.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/c2e318a4054bb91fadd7e2ab4a1224be51693d21e5aaa09018c0a70673933225.jpg)

![c6f63656f02845b84ad59f9cff2552777d6c5d11629640a0492caa78abb04cea.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/c6f63656f02845b84ad59f9cff2552777d6c5d11629640a0492caa78abb04cea.jpg)

![cd1814a06b0347876418f0158e7eef8d67be30a5c6a0b46af4a79af28366c052.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/cd1814a06b0347876418f0158e7eef8d67be30a5c6a0b46af4a79af28366c052.jpg)

![d363e4814dcebccf9e3e1b156924f1a66950fd4fdb5b9bd13c8f97c8e6c4b742.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/d363e4814dcebccf9e3e1b156924f1a66950fd4fdb5b9bd13c8f97c8e6c4b742.jpg)

![f2cfde6003fe930a571ddb5f490019e1e4c941f19e9627e4b8121114eb9b05f0.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/f2cfde6003fe930a571ddb5f490019e1e4c941f19e9627e4b8121114eb9b05f0.jpg)

![f74370dc41131069598b5b14eeca0900ab162b26ec6ff7ef3cf9ee658bb48691.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/f74370dc41131069598b5b14eeca0900ab162b26ec6ff7ef3cf9ee658bb48691.jpg)

![ff04c16b037fb4b2a2dfda5268ae8d0cbab3a7c880fa490dff7a52d9fc85f298.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/images/ff04c16b037fb4b2a2dfda5268ae8d0cbab3a7c880fa490dff7a52d9fc85f298.jpg)

### Tables

![1be43abe754e4a8d2593eecb42d5f6db40caa60f0dd8e7f197c53ae14cf3d402.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/1be43abe754e4a8d2593eecb42d5f6db40caa60f0dd8e7f197c53ae14cf3d402.jpg)

![240f3a2530b3b9114560ade1f3c45400b9e762e34c40b1f6026f0421659289c3.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/240f3a2530b3b9114560ade1f3c45400b9e762e34c40b1f6026f0421659289c3.jpg)

![29d8a095bcb3e3d1dfc56d8c500fefd3a49bb9c1af4040589417b7845025dcc3.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/29d8a095bcb3e3d1dfc56d8c500fefd3a49bb9c1af4040589417b7845025dcc3.jpg)

![2b3963e7e8ea61fe0bcee4183272ef01862a9ce7b5a829b73d4f2f4aa9bc5f4a.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/2b3963e7e8ea61fe0bcee4183272ef01862a9ce7b5a829b73d4f2f4aa9bc5f4a.jpg)

![3e012e996b48631cb1685d7dfacc6cda7f33988789f3c77fc2c55fa9da1743c9.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/3e012e996b48631cb1685d7dfacc6cda7f33988789f3c77fc2c55fa9da1743c9.jpg)

![45bc716e0dffbc6839a1fb79857bfce0500d4eaa8e2cb1d95609ddd752787a74.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/45bc716e0dffbc6839a1fb79857bfce0500d4eaa8e2cb1d95609ddd752787a74.jpg)

![4a513f44101dc568b4d3e0237cc0487660c664421189b213ba171e48e20fbcdd.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/4a513f44101dc568b4d3e0237cc0487660c664421189b213ba171e48e20fbcdd.jpg)

![509dcded6cd629b6cf294febce67e39e73c6be6bc7dfcb47f36d5e74870d6170.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/509dcded6cd629b6cf294febce67e39e73c6be6bc7dfcb47f36d5e74870d6170.jpg)

![67545bf76cc9892485060ca7b13844cca811309d7f3073818b6b4f11ba7cf8f0.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/67545bf76cc9892485060ca7b13844cca811309d7f3073818b6b4f11ba7cf8f0.jpg)

![6f954ff9353501a5dbef3a0e4a1f58f10855f12b23c1d006ce557d1f445ce1df.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/6f954ff9353501a5dbef3a0e4a1f58f10855f12b23c1d006ce557d1f445ce1df.jpg)

![951d91b6b6986c4638571d5c1107ce43ce2826a3752aaaf396bc4133463d1b2b.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/951d91b6b6986c4638571d5c1107ce43ce2826a3752aaaf396bc4133463d1b2b.jpg)

![9b045382311f634f866d604469707a7dbda77cb435124d04d4472a91ee9fda5d.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/9b045382311f634f866d604469707a7dbda77cb435124d04d4472a91ee9fda5d.jpg)

![9f520f0e1ab658ce22b17fa8cd42e92a37e00884afbbfd4cc47ed9a30460367d.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/9f520f0e1ab658ce22b17fa8cd42e92a37e00884afbbfd4cc47ed9a30460367d.jpg)

![a948dac37e042050b4e03aa700ab7ee439bb7e55bdc3427e1b5640320d9faeda.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/a948dac37e042050b4e03aa700ab7ee439bb7e55bdc3427e1b5640320d9faeda.jpg)

![b24be8ceebc21a1653806b0cd6aefc75760a8aad7595a9891b9f9f220c74cd0d.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/b24be8ceebc21a1653806b0cd6aefc75760a8aad7595a9891b9f9f220c74cd0d.jpg)

![b48ae7a0a736d38fdd4cbbce0075d1d16d8c0da7fcd9cd21d7dff605323907d2.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/b48ae7a0a736d38fdd4cbbce0075d1d16d8c0da7fcd9cd21d7dff605323907d2.jpg)

![b4aef742b3185bc5f9a4a93e174971a945f8a5c24d7fe8e94bf25cd953748871.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/b4aef742b3185bc5f9a4a93e174971a945f8a5c24d7fe8e94bf25cd953748871.jpg)

![c9cfaa90ee5d643bb326bb4c7525e6177c38242baba26a485c485e15d3fa5bae.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/c9cfaa90ee5d643bb326bb4c7525e6177c38242baba26a485c485e15d3fa5bae.jpg)

![ccf19e9a1dd102624708f4d25f6bcbcf2b49dce367e9822ff60d60cbc90f5da0.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/ccf19e9a1dd102624708f4d25f6bcbcf2b49dce367e9822ff60d60cbc90f5da0.jpg)

![d10fb637505857e37db705adf875e8515fe2149ee0eb82463e6d7456e8e9b971.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/d10fb637505857e37db705adf875e8515fe2149ee0eb82463e6d7456e8e9b971.jpg)

![d2936dcf5ece1c9a937e94509d8377a31ad2da2598d5d0c864b24814b390ae89.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/d2936dcf5ece1c9a937e94509d8377a31ad2da2598d5d0c864b24814b390ae89.jpg)

![d6f7440723254b67707d4bb6cc88f05ac9ec70b0f759b4642a61de45e793b423.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/d6f7440723254b67707d4bb6cc88f05ac9ec70b0f759b4642a61de45e793b423.jpg)

![dabd345978e0d4dabbd68ecf739d2c4844631f50cd35b3c25d6ee3c34bc4173a.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/dabd345978e0d4dabbd68ecf739d2c4844631f50cd35b3c25d6ee3c34bc4173a.jpg)

![f04756574a715cd07c0ccc1c0e3ed43b7d07c4527259115c9fcb993785ad80d5.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/f04756574a715cd07c0ccc1c0e3ed43b7d07c4527259115c9fcb993785ad80d5.jpg)

![f3ef22213103cd83fcd3be113f3a57badc78b436dfcaafb7fb4e8acd34759ab9.jpg](../icml_results/1719_Byte%20Pair%20Encoding%20for%20Efficient%20Time%20Series%20Forecasting/tables/f3ef22213103cd83fcd3be113f3a57badc78b436dfcaafb7fb4e8acd34759ab9.jpg)

<a id="paper-1698"></a>

## Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning


### Images

![06ddb717b3af8a92be290a849a3dc29a9c81d0e3b2c98370f205967d95007dc0.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/06ddb717b3af8a92be290a849a3dc29a9c81d0e3b2c98370f205967d95007dc0.jpg)

![0b71118035f0238938c767eb5709b4ac905d98cf8790dbbb8f4c2f15378d22a9.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/0b71118035f0238938c767eb5709b4ac905d98cf8790dbbb8f4c2f15378d22a9.jpg)

![13b2b929af064152004b8569106015695f918663d2f20fec216fd8783f4db2e4.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/13b2b929af064152004b8569106015695f918663d2f20fec216fd8783f4db2e4.jpg)

![36abfbf8a1f339d48d18044959a058bd5fc432dcd6460ecd1996aa58bc1d888c.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/36abfbf8a1f339d48d18044959a058bd5fc432dcd6460ecd1996aa58bc1d888c.jpg)

![3d054f12e27841377bc9e05b375c78bc14f21da6bc70a359410840a8894daffe.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/3d054f12e27841377bc9e05b375c78bc14f21da6bc70a359410840a8894daffe.jpg)

![3d06925f020b1ceb7efe3d8c83b1ca26735e896ed13910b2180decdf7ba02c76.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/3d06925f020b1ceb7efe3d8c83b1ca26735e896ed13910b2180decdf7ba02c76.jpg)

![3eca2fd1614484475e0c9602ac68bba4a54d9dd0ffe671b72db7553cf26f8ce9.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/3eca2fd1614484475e0c9602ac68bba4a54d9dd0ffe671b72db7553cf26f8ce9.jpg)

![427fb36bb5c87d2dc23cf824b7d49461c3f59e4800952f9111558607fe941781.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/427fb36bb5c87d2dc23cf824b7d49461c3f59e4800952f9111558607fe941781.jpg)

![45edc8ff95a19e2e211432d8a5753dc53408886b68247f1ffe87b8a2df75dfc3.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/45edc8ff95a19e2e211432d8a5753dc53408886b68247f1ffe87b8a2df75dfc3.jpg)

![4d2a2a7aa837ffb722532489838898656217751b9755236c7edc83869a533440.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/4d2a2a7aa837ffb722532489838898656217751b9755236c7edc83869a533440.jpg)

![50a8902922c9d9001a3b15aaa53307549b6f39e58f019e63d1cb2e90303415d7.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/50a8902922c9d9001a3b15aaa53307549b6f39e58f019e63d1cb2e90303415d7.jpg)

![50f81f12e740903743b9350190c10a2d8db98f64ed1b7719135572d595c5156b.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/50f81f12e740903743b9350190c10a2d8db98f64ed1b7719135572d595c5156b.jpg)

![70eee6e38d68fc661156e30355b53185ecb5574a5e3b49796d96367a9171d4d5.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/70eee6e38d68fc661156e30355b53185ecb5574a5e3b49796d96367a9171d4d5.jpg)

![7aa3374234049769d5e4030815b7fc026410f22c8878bfc1f35745115dba6855.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/7aa3374234049769d5e4030815b7fc026410f22c8878bfc1f35745115dba6855.jpg)

![8dec2ddf89ffff1abd8ef3105143efcd853ec60114177a65d848eda0543a788c.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/8dec2ddf89ffff1abd8ef3105143efcd853ec60114177a65d848eda0543a788c.jpg)

![962a9140932aa87c1e6493a031e9c13f8a1769b16cd80b3e7eb9b74fa6ee103b.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/962a9140932aa87c1e6493a031e9c13f8a1769b16cd80b3e7eb9b74fa6ee103b.jpg)

![97b8cde590eec917c73f584683712056e43a1cad8567a69ac57a2533d5b300f9.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/97b8cde590eec917c73f584683712056e43a1cad8567a69ac57a2533d5b300f9.jpg)

![9a019606f56bba0353bb13530319ef35ab64c04e3412481278f410824983d760.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/9a019606f56bba0353bb13530319ef35ab64c04e3412481278f410824983d760.jpg)

![9cb6d360ccbf919011d17860444deb9219ad178e9234df1602015c6314a85e1d.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/9cb6d360ccbf919011d17860444deb9219ad178e9234df1602015c6314a85e1d.jpg)

![a33107285a08b83fd96489b2cf21b4e46d52c61d27d682966651b347b1809709.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/a33107285a08b83fd96489b2cf21b4e46d52c61d27d682966651b347b1809709.jpg)

![b014784b358fb3ce0c95cb1c6a7463a7cc760aefb483203fbe803320088d395c.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/b014784b358fb3ce0c95cb1c6a7463a7cc760aefb483203fbe803320088d395c.jpg)

![cd1f77af829449d84dce5f01cbd143cd265eccfbbd4d07bf564eac0369887fc2.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/cd1f77af829449d84dce5f01cbd143cd265eccfbbd4d07bf564eac0369887fc2.jpg)

![d22492da446790bec9b4dc5b646f73abd66f97749eeb2d3e3ff61ca47d929d40.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/d22492da446790bec9b4dc5b646f73abd66f97749eeb2d3e3ff61ca47d929d40.jpg)

![dbd97ade7cf54634703a6d7a654641447017e9f46d1fc25b74d5810788581419.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/dbd97ade7cf54634703a6d7a654641447017e9f46d1fc25b74d5810788581419.jpg)

![e68708515775e441fbd78d3f529f08207e4d1d94c41ac1fc1ecc1421e11ca746.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/e68708515775e441fbd78d3f529f08207e4d1d94c41ac1fc1ecc1421e11ca746.jpg)

![ea65785964eaede6ea161fcf14cf1ee02d8e5b44742ea92a745e1683bd0adefa.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/ea65785964eaede6ea161fcf14cf1ee02d8e5b44742ea92a745e1683bd0adefa.jpg)

![ec029f3b0d2834bc9ecd29b891ec95dd03722fe7d60d1d4bcaa4817f5737c282.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/ec029f3b0d2834bc9ecd29b891ec95dd03722fe7d60d1d4bcaa4817f5737c282.jpg)

![f1c6b7775613a42906ec4846d6a5447ba0859b46377abdeb76d819a3e5eef403.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/images/f1c6b7775613a42906ec4846d6a5447ba0859b46377abdeb76d819a3e5eef403.jpg)

### Tables

![207c0d13f72b634b317a05b4f3e035ff7a2c60aba7e949eecfafd421d0fc87df.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/207c0d13f72b634b317a05b4f3e035ff7a2c60aba7e949eecfafd421d0fc87df.jpg)

![20c290e430a7d38d58ba611569ad7c744ef6fa872f4762134d89e043f2d14811.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/20c290e430a7d38d58ba611569ad7c744ef6fa872f4762134d89e043f2d14811.jpg)

![2242e325096cfd48e16f68ce7a316cd80dcd8b2e9d4d60fa4a9cab5e7fb01eed.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/2242e325096cfd48e16f68ce7a316cd80dcd8b2e9d4d60fa4a9cab5e7fb01eed.jpg)

![4a8ecde88e03bea4894965745728c75bbe3e9b4e00aa9f276fbcd219ebf7d8c0.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/4a8ecde88e03bea4894965745728c75bbe3e9b4e00aa9f276fbcd219ebf7d8c0.jpg)

![6d0be98c7a4ad3069a4806a16dd88b6b686cba97effed5ed3f358935b2cb8155.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/6d0be98c7a4ad3069a4806a16dd88b6b686cba97effed5ed3f358935b2cb8155.jpg)

![7ecfe272ef7755605c8957929180244e6112f23c7555a951a8e77c4d454f26eb.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/7ecfe272ef7755605c8957929180244e6112f23c7555a951a8e77c4d454f26eb.jpg)

![9d0e8430e995845fe668eead9d86fead938add569be72f5642b0dc8f6a4851ba.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/9d0e8430e995845fe668eead9d86fead938add569be72f5642b0dc8f6a4851ba.jpg)

![a614ead4c5e314e5caa213e8a5f5137a6dcd0fe69436561c24eb8f91701f0630.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/a614ead4c5e314e5caa213e8a5f5137a6dcd0fe69436561c24eb8f91701f0630.jpg)

![f7524c025cf481e930d600648104578e5f5170adb5cef93c42fe7669ee9577ce.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/f7524c025cf481e930d600648104578e5f5170adb5cef93c42fe7669ee9577ce.jpg)

![f9d451d0cb6e846452bae3de325f52ff316d6d4326409ba1d8859a4f0e8bc90e.jpg](../icml_results/1720_Black-Box%20Combinatorial%20Optimization%20with%20Order-Invariant%20Reinforcement%20Learning/tables/f9d451d0cb6e846452bae3de325f52ff316d6d4326409ba1d8859a4f0e8bc90e.jpg)

<a id="paper-1699"></a>

## Categorical Reparameterization with Denoising Diffusion Models


### Images

![049f80ebef5ce53ca41597c3699da6902c171fd192502412d497c6bebcf429fe.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/049f80ebef5ce53ca41597c3699da6902c171fd192502412d497c6bebcf429fe.jpg)

![1cd14b178c7719db242f2eb86351abffa52b83e23718c2a148fa13abda3503c4.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/1cd14b178c7719db242f2eb86351abffa52b83e23718c2a148fa13abda3503c4.jpg)

![1f84c451074b58e3c2170c9972ec0099734a12c87878c7b8a56f3824b9158d6a.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/1f84c451074b58e3c2170c9972ec0099734a12c87878c7b8a56f3824b9158d6a.jpg)

![480ee9b980e992a247101262e89fe359f46a302640a2cd829264677e9bcc121b.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/480ee9b980e992a247101262e89fe359f46a302640a2cd829264677e9bcc121b.jpg)

![4aeebd28d0e45cf64a5a207b3474000c8ad27526a9239b6cf38210b75ef1c7e2.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/4aeebd28d0e45cf64a5a207b3474000c8ad27526a9239b6cf38210b75ef1c7e2.jpg)

![51ea5e4a5a18c69966c115c0151beea5e9a17653ce05f1fb461ce29e44ecaf25.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/51ea5e4a5a18c69966c115c0151beea5e9a17653ce05f1fb461ce29e44ecaf25.jpg)

![63025ac0d6fd2c96ad5cb20f3b17e9cff6c3c08b048124a117d390b3a0fbb0d9.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/63025ac0d6fd2c96ad5cb20f3b17e9cff6c3c08b048124a117d390b3a0fbb0d9.jpg)

![74f385b7cd8a2dbbec222a9b2a431d31fb5f8b50a1c79a02bbacf760fb62ba5b.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/74f385b7cd8a2dbbec222a9b2a431d31fb5f8b50a1c79a02bbacf760fb62ba5b.jpg)

![9abbd505e3b2913f475e18a6551caf84651dd870ce76fc7db83b9448ed94d2f4.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/9abbd505e3b2913f475e18a6551caf84651dd870ce76fc7db83b9448ed94d2f4.jpg)

![bf876b411a7b47b5c254cd657a94f72844315ac3e87586489db86d4ba9425fb6.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/bf876b411a7b47b5c254cd657a94f72844315ac3e87586489db86d4ba9425fb6.jpg)

![f44d6690d0e40bc0a1912261b668f7a85066a6b5e8d6aae973efa2b275187cb8.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/images/f44d6690d0e40bc0a1912261b668f7a85066a6b5e8d6aae973efa2b275187cb8.jpg)

### Tables

![19d4699d70501014533b76afe6a4514c042ecd6725e2ddd176484ba5680d5b6c.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/19d4699d70501014533b76afe6a4514c042ecd6725e2ddd176484ba5680d5b6c.jpg)

![445ed238dbb07b67728e31ed3a8880c1ef954723f120b5a7e9717d07aa129e2b.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/445ed238dbb07b67728e31ed3a8880c1ef954723f120b5a7e9717d07aa129e2b.jpg)

![4541848abe7b476a6262021f69f36b7dc7d53e6a308658941d07341304ef9067.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/4541848abe7b476a6262021f69f36b7dc7d53e6a308658941d07341304ef9067.jpg)

![4b8c3207017737dc72b8ce6ab8cc13bba1fdd8217032d541b7aac86494dd57df.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/4b8c3207017737dc72b8ce6ab8cc13bba1fdd8217032d541b7aac86494dd57df.jpg)

![6b141e2da8b9ae88632b059988d0e0afddc5a40ebc8c9a6371f244f90b5c0d40.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/6b141e2da8b9ae88632b059988d0e0afddc5a40ebc8c9a6371f244f90b5c0d40.jpg)

![93035bb5041c5f37dc209bba57ce67f4dc6b56491dcaf0c5ea70cdb5df133a59.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/93035bb5041c5f37dc209bba57ce67f4dc6b56491dcaf0c5ea70cdb5df133a59.jpg)

![9645c6dcb2192ff1fffea3f9e2509c79d46e9f9beafae54ce03eb552b32c8c67.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/9645c6dcb2192ff1fffea3f9e2509c79d46e9f9beafae54ce03eb552b32c8c67.jpg)

![ed2c429445ed214c7c3871e43dad926c8449c36901aa88105d4ca8567db04cc2.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/ed2c429445ed214c7c3871e43dad926c8449c36901aa88105d4ca8567db04cc2.jpg)

![f6b1a05a14920f6705f8fc70c4bcfc694249389abdadbfb199f37b771c465813.jpg](../icml_results/1721_Categorical%20Reparameterization%20with%20Denoising%20Diffusion%20Models/tables/f6b1a05a14920f6705f8fc70c4bcfc694249389abdadbfb199f37b771c465813.jpg)

<a id="paper-1700"></a>

## Contextual Slate GLM Bandits with Limited Adaptivity


### Images

![2c497fd6bc8b8fb3c8de68a69ff87da3db92b32280d356f3855a0893c83c68e9.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/2c497fd6bc8b8fb3c8de68a69ff87da3db92b32280d356f3855a0893c83c68e9.jpg)

![3cb06972342a6dc966206622b3bb71d7391d29e67e47cb3a70219ae12ce6bd12.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/3cb06972342a6dc966206622b3bb71d7391d29e67e47cb3a70219ae12ce6bd12.jpg)

![7d67dfdb743552c524576f3b5354ab8900e60c271e937d817471008b29fc6e78.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/7d67dfdb743552c524576f3b5354ab8900e60c271e937d817471008b29fc6e78.jpg)

![8c3bbd2f5cfe2e83fd86fcd9dd36ec2b8b2d20f9c27c24411a3b4b648a12dc7a.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/8c3bbd2f5cfe2e83fd86fcd9dd36ec2b8b2d20f9c27c24411a3b4b648a12dc7a.jpg)

![d2052510b62e01b1fef24f7d4943c67cd7ce3c273189a001eeb3842a1f622532.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/d2052510b62e01b1fef24f7d4943c67cd7ce3c273189a001eeb3842a1f622532.jpg)

![de421045964d48101f90d89309b2b6ff91ebee13c8f58b766acce3e675c1119a.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/de421045964d48101f90d89309b2b6ff91ebee13c8f58b766acce3e675c1119a.jpg)

![f019494192918f36efae5841f9cd80fc738acd403c51f0cc9057b34a87ae3a11.jpg](../icml_results/1722_Contextual%20Slate%20GLM%20Bandits%20with%20Limited%20Adaptivity/images/f019494192918f36efae5841f9cd80fc738acd403c51f0cc9057b34a87ae3a11.jpg)

<a id="paper-1701"></a>

## FlowState: Sampling-Rate-Equivariant Time-Series Forecasting


### Images

![078ae9a40320a4214d522471853321bf89062a4b18d889e32d4a888772e8507e.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/078ae9a40320a4214d522471853321bf89062a4b18d889e32d4a888772e8507e.jpg)

![11152b1501a6e3513dc94e1ffd858f34e1de26a535e8bef4d27b26d2162e3823.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/11152b1501a6e3513dc94e1ffd858f34e1de26a535e8bef4d27b26d2162e3823.jpg)

![19b514e2aa40e8b4228e14d30dd73b49a2d0d08b82598e56c70ab48296cddba4.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/19b514e2aa40e8b4228e14d30dd73b49a2d0d08b82598e56c70ab48296cddba4.jpg)

![248df77e272377ef929056cf38d6e434076ffc14d529cba796ff33d12e702f65.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/248df77e272377ef929056cf38d6e434076ffc14d529cba796ff33d12e702f65.jpg)

![316682dda99496d7a274d32f8aa59dfd3c17020979f7dc7b757b4c968beda175.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/316682dda99496d7a274d32f8aa59dfd3c17020979f7dc7b757b4c968beda175.jpg)

![4b9a7d53d9cb6fbe6e259c551c9dfc019c3c2f30c397bc823345fdb3ea7138eb.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/4b9a7d53d9cb6fbe6e259c551c9dfc019c3c2f30c397bc823345fdb3ea7138eb.jpg)

![be314962f449a9fa06328e0f4609de8a42a98856e9af423a950c9d7bd9109104.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/be314962f449a9fa06328e0f4609de8a42a98856e9af423a950c9d7bd9109104.jpg)

![c1e70a33462b8d14bf93d6acdd18c366962ea4ee1274f22e85719e8f69f20a93.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/c1e70a33462b8d14bf93d6acdd18c366962ea4ee1274f22e85719e8f69f20a93.jpg)

![c90b538e6ec57d6f44caf680cffd9b17e5fb5f3e2b5a904dab3ba3e8f9580a3b.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/images/c90b538e6ec57d6f44caf680cffd9b17e5fb5f3e2b5a904dab3ba3e8f9580a3b.jpg)

### Tables

![212987f7c3cef2cf6d33f656dfb01d7c3063804b1d65e46e6957152febc286bf.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/212987f7c3cef2cf6d33f656dfb01d7c3063804b1d65e46e6957152febc286bf.jpg)

![5032ee7038cdb5841dd31db842c6ae29eeecfc1956b1b289ef5922b95a0481c6.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/5032ee7038cdb5841dd31db842c6ae29eeecfc1956b1b289ef5922b95a0481c6.jpg)

![5b5b27d8d363e69fe551271deebf10cbfe3fa3de3fe76a05824db3f98758883e.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/5b5b27d8d363e69fe551271deebf10cbfe3fa3de3fe76a05824db3f98758883e.jpg)

![68f9e1c5d8426a8f7e97c426ff6ba7068ae60e5667e005d6df2223f316fb07e1.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/68f9e1c5d8426a8f7e97c426ff6ba7068ae60e5667e005d6df2223f316fb07e1.jpg)

![9edf3553920da902c5e66b0bca6e252038dfb25d824dcf3ab478a0d30775063c.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/9edf3553920da902c5e66b0bca6e252038dfb25d824dcf3ab478a0d30775063c.jpg)

![bcd92bf43865100192db0dae5bd22f540ffa4d8ac75237ed7e78e22e8d7406ce.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/bcd92bf43865100192db0dae5bd22f540ffa4d8ac75237ed7e78e22e8d7406ce.jpg)

![c4c22f1ef33994803f8595b9bf2044905a4dec4538ba82fa36a027b1ce96ffb6.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/c4c22f1ef33994803f8595b9bf2044905a4dec4538ba82fa36a027b1ce96ffb6.jpg)

![f80f96abfdf49d574503cd8d46c0fa5cb176e3ceee7d8a599b19bce46d65cfb7.jpg](../icml_results/1723_FlowState_%20Sampling-Rate-Equivariant%20Time-Series%20Forecasting/tables/f80f96abfdf49d574503cd8d46c0fa5cb176e3ceee7d8a599b19bce46d65cfb7.jpg)

<a id="paper-1702"></a>

## Safety-Efficacy Trade Off: Robustness against Data-Poisoning


### Images

![224b060bc951e459b393328accd2adea4b90fe0e66a66483bcce2ce8be86a30f.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/224b060bc951e459b393328accd2adea4b90fe0e66a66483bcce2ce8be86a30f.jpg)

![3b4891b02d46f3c03464ffc694f62f15139ee498c013aeda406aa4e86bcbf713.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/3b4891b02d46f3c03464ffc694f62f15139ee498c013aeda406aa4e86bcbf713.jpg)

![3d6e4ccc405c158e070b80cc9e863bd0a80dc9297e6fa5f6117d2d6994b7bff3.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/3d6e4ccc405c158e070b80cc9e863bd0a80dc9297e6fa5f6117d2d6994b7bff3.jpg)

![437691dec01b1cfdba4d640e172fbc10e24478c2ee837443a32a2c9bb1c927a5.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/437691dec01b1cfdba4d640e172fbc10e24478c2ee837443a32a2c9bb1c927a5.jpg)

![5873b2cb37d8651a9ec08884e323fa72f6ab63f0a0d209795fb3dd82512209c0.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/5873b2cb37d8651a9ec08884e323fa72f6ab63f0a0d209795fb3dd82512209c0.jpg)

![7985597ec6ac5a39530409e3db3504de33722fd1406dbcf26f46f2fc4a180c1c.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/7985597ec6ac5a39530409e3db3504de33722fd1406dbcf26f46f2fc4a180c1c.jpg)

![9892e856347967b7b38fb2a5a8c1b9230db423f8edac77fe5622c85ce643366a.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/9892e856347967b7b38fb2a5a8c1b9230db423f8edac77fe5622c85ce643366a.jpg)

![a0e3477551acec48d8b955257c2f199fa01dfc43c8911425ca1307b7fc63d9d4.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/a0e3477551acec48d8b955257c2f199fa01dfc43c8911425ca1307b7fc63d9d4.jpg)

![aa0e9cab9926ac2c129c100ef7aa822bc75ac8e8f23df6836a12f071a43f631a.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/aa0e9cab9926ac2c129c100ef7aa822bc75ac8e8f23df6836a12f071a43f631a.jpg)

![da09029c193165667f5fef4efa49db1f21e9670e2bf69bf92a3a5325c7d75d50.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/da09029c193165667f5fef4efa49db1f21e9670e2bf69bf92a3a5325c7d75d50.jpg)

![e74a83ac17970ff87d46d7c64b9641c0cc3c34870cc9756fe8d4c02dcf33ba01.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/e74a83ac17970ff87d46d7c64b9641c0cc3c34870cc9756fe8d4c02dcf33ba01.jpg)

![e94be022e247b019f6e610aa4f6258b179f94bdd5199d36ab0ff5aad34e7b684.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/e94be022e247b019f6e610aa4f6258b179f94bdd5199d36ab0ff5aad34e7b684.jpg)

![f2ef4628ed38cc737f96b547e7620b7f14e145034a31dbe9ac9490640af0abd0.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/f2ef4628ed38cc737f96b547e7620b7f14e145034a31dbe9ac9490640af0abd0.jpg)

![f690f39ac32cc834fce56feb7214b02a15a4db51deabb46ceb20c4e4a765b561.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/f690f39ac32cc834fce56feb7214b02a15a4db51deabb46ceb20c4e4a765b561.jpg)

![fb661a04fa478a3f2d9ac3543c1e55139f7ce851013de53d090cf7a44a6ec6e2.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/images/fb661a04fa478a3f2d9ac3543c1e55139f7ce851013de53d090cf7a44a6ec6e2.jpg)

### Tables

![0be141307416c9f3772af81068b5a5b5de11954baa400ab082a572c085c047b4.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/tables/0be141307416c9f3772af81068b5a5b5de11954baa400ab082a572c085c047b4.jpg)

![64a4a4a175483fd6f2661f9c3a61b5b350449df1674b06819636a4b9399875d8.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/tables/64a4a4a175483fd6f2661f9c3a61b5b350449df1674b06819636a4b9399875d8.jpg)

![ae34d0e81b7de5f1f4ac202fc56ed008fc6a3c7005de330725ced009526dfabf.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/tables/ae34d0e81b7de5f1f4ac202fc56ed008fc6a3c7005de330725ced009526dfabf.jpg)

![d61749b2fc42f25f7aa4d1bbab84e1334aa8f8607b373f44c5171ab8f199371c.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/tables/d61749b2fc42f25f7aa4d1bbab84e1334aa8f8607b373f44c5171ab8f199371c.jpg)

![f8b443e291a87248cbb2e47a0a34999d5982ad0d7e7db02502d08a8e040c433c.jpg](../icml_results/1724_Safety-Efficacy%20Trade%20Off_%20Robustness%20against%20Data-Poisoning/tables/f8b443e291a87248cbb2e47a0a34999d5982ad0d7e7db02502d08a8e040c433c.jpg)

<a id="paper-1703"></a>

## Multi-marginal temporal Schrödinger Bridge Matching from unpaired data


### Images

![16f7d2c1dbb66b9c28c327606604f61c1c7287fbd037eaa2878e309ca6e89bf9.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/16f7d2c1dbb66b9c28c327606604f61c1c7287fbd037eaa2878e309ca6e89bf9.jpg)

![1e23d6a59053922b783c8c8ff5c0ea6fd56a4daeb3dea5b624c45fce14d8386a.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/1e23d6a59053922b783c8c8ff5c0ea6fd56a4daeb3dea5b624c45fce14d8386a.jpg)

![3e2b1cda6f9841acea07c454d7f37ce48549c69f29edb052fd6c32cfd54e805b.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/3e2b1cda6f9841acea07c454d7f37ce48549c69f29edb052fd6c32cfd54e805b.jpg)

![4beaf84478f2d2a4cbca81290d5cbf0b250c2504614506c52599ac334bab60d5.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/4beaf84478f2d2a4cbca81290d5cbf0b250c2504614506c52599ac334bab60d5.jpg)

![ab1a1cf4d98efa3771dc9e8891ac50aa16d5592b0dcdc408ad89de56e4d6ebdd.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/ab1a1cf4d98efa3771dc9e8891ac50aa16d5592b0dcdc408ad89de56e4d6ebdd.jpg)

![afc6582ebf1e0fc7ccf3b1f246e8eb69bc288884fd5551ef9e4696095883bc5d.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/afc6582ebf1e0fc7ccf3b1f246e8eb69bc288884fd5551ef9e4696095883bc5d.jpg)

![bdebb905be25ce9ae7db06835d01f7246e6a9e4a9c30c010e7e2f847532659c7.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/bdebb905be25ce9ae7db06835d01f7246e6a9e4a9c30c010e7e2f847532659c7.jpg)

![e98e49b9dd5a9aaec358f3af4af12c511e209f9df734587571980307c25f6689.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/images/e98e49b9dd5a9aaec358f3af4af12c511e209f9df734587571980307c25f6689.jpg)

### Tables

![2cbc7ebba39c36e8fa8a2945ba034804d8ff74deda00c1a5823a76968ce34b17.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/2cbc7ebba39c36e8fa8a2945ba034804d8ff74deda00c1a5823a76968ce34b17.jpg)

![50dc202e7a48c89de5c2c5b00718bae4111f66f92b6446558e568db0c5911938.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/50dc202e7a48c89de5c2c5b00718bae4111f66f92b6446558e568db0c5911938.jpg)

![51f9de1095a21257ef1980f0835d45a4403f2f6070775fa954f0dc5bfbe4cefb.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/51f9de1095a21257ef1980f0835d45a4403f2f6070775fa954f0dc5bfbe4cefb.jpg)

![9ea9812eb3197ac505a64fc6ce0d2c258758159e7bc5fab4d12bf8f8a3536dd1.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/9ea9812eb3197ac505a64fc6ce0d2c258758159e7bc5fab4d12bf8f8a3536dd1.jpg)

![b52727a56f76548d8f8ae737fe18ebac1a90ab7de80e6ce9f526c50d4acc1a29.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/b52727a56f76548d8f8ae737fe18ebac1a90ab7de80e6ce9f526c50d4acc1a29.jpg)

![bc85063aa50e9b9792fc822bdae6b485049f1d5f28eb2fbe06e6491dfea7f357.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/bc85063aa50e9b9792fc822bdae6b485049f1d5f28eb2fbe06e6491dfea7f357.jpg)

![c36c860496abff082bc9a7b9b7b8fd6e700907d34d322a8788b3ae8f78168a46.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/c36c860496abff082bc9a7b9b7b8fd6e700907d34d322a8788b3ae8f78168a46.jpg)

![cf4c5917942eb90d33b6fe8310946eed5dad1f474105be2e20f4812bb480acca.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/cf4c5917942eb90d33b6fe8310946eed5dad1f474105be2e20f4812bb480acca.jpg)

![ecb1b92c24461957418f5baa699ed6b105e7f8804010d218355aa3b047101033.jpg](../icml_results/1725_Multi-marginal%20temporal%20Schr%C3%B6dinger%20Bridge%20Matching%20from%20unpaired%20data/tables/ecb1b92c24461957418f5baa699ed6b105e7f8804010d218355aa3b047101033.jpg)

<a id="paper-1704"></a>

## Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits


### Images

![1213b2619e3e0a6bb6426bbd64753d0eae4fbd410074948b9eb4c48d5a590d60.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/1213b2619e3e0a6bb6426bbd64753d0eae4fbd410074948b9eb4c48d5a590d60.jpg)

![1722333ec6a93c4c91677ae6e0c79e8484373a7b85499154e3dfb42554ed0f78.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/1722333ec6a93c4c91677ae6e0c79e8484373a7b85499154e3dfb42554ed0f78.jpg)

![560eb0671fdca94d5e8b9f25ea7c9d9018b2316293852be04c1181adaba33ac0.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/560eb0671fdca94d5e8b9f25ea7c9d9018b2316293852be04c1181adaba33ac0.jpg)

![87d1e2766a66e9071216e2a286cd7717413a384fb94c43e01e3ab93fae2aa1b6.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/87d1e2766a66e9071216e2a286cd7717413a384fb94c43e01e3ab93fae2aa1b6.jpg)

![da4be4ed195820719be7bb0c2cd79ed4e003f3fd99def7ecddb68abf1f3e8963.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/da4be4ed195820719be7bb0c2cd79ed4e003f3fd99def7ecddb68abf1f3e8963.jpg)

![dd45484d7da376d634f27717b56535d36c702af412ae6e2dce0ec4103ce88e2c.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/dd45484d7da376d634f27717b56535d36c702af412ae6e2dce0ec4103ce88e2c.jpg)

![f0106689ab743b009caa48262eb75ab380c5931675f85d23c0a4e38280bf4e1c.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/images/f0106689ab743b009caa48262eb75ab380c5931675f85d23c0a4e38280bf4e1c.jpg)

### Tables

![138eef8a3278a360fbea058fcb14fbc2fbc3ef2453aee0e9214a0bb27f6f273b.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/138eef8a3278a360fbea058fcb14fbc2fbc3ef2453aee0e9214a0bb27f6f273b.jpg)

![180060c4bbb980c1b9befdb981a1c38cdfe4e822bc1c84154e31629782efc6cb.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/180060c4bbb980c1b9befdb981a1c38cdfe4e822bc1c84154e31629782efc6cb.jpg)

![1b5c6b705e52f435e4da5dae49a6b74ca0968728ec07f46b24a9a2303eb9d0a2.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/1b5c6b705e52f435e4da5dae49a6b74ca0968728ec07f46b24a9a2303eb9d0a2.jpg)

![1fc4420e53620d35f2a3e5606003b145dfe8ebf89b59f9295a9e399baddb8dbb.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/1fc4420e53620d35f2a3e5606003b145dfe8ebf89b59f9295a9e399baddb8dbb.jpg)

![22c91b26e89b87cb602c385beee9fc1ece8dbdb9e3e7be345b167393f8f32679.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/22c91b26e89b87cb602c385beee9fc1ece8dbdb9e3e7be345b167393f8f32679.jpg)

![25ddba7f655e5db450d1cc5bfd970af89b55d0bb275ea356deffab736dc9579b.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/25ddba7f655e5db450d1cc5bfd970af89b55d0bb275ea356deffab736dc9579b.jpg)

![265d8256e3bf41c34f07691d4f421457e1cc71ab6da3cd8e52feabd97178f8b5.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/265d8256e3bf41c34f07691d4f421457e1cc71ab6da3cd8e52feabd97178f8b5.jpg)

![2ce76587d52f987fb45dfabcd9145ac080a51932e76b4d5e82c029b3271addfd.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/2ce76587d52f987fb45dfabcd9145ac080a51932e76b4d5e82c029b3271addfd.jpg)

![39674d8a5d550f7b56e3ec236c8b9c9198d28241cfde36e4bce599370c521504.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/39674d8a5d550f7b56e3ec236c8b9c9198d28241cfde36e4bce599370c521504.jpg)

![3d0fbed62eed621017efaba7be1e5b94d58b5066e423c145d37f3098b718d9b4.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/3d0fbed62eed621017efaba7be1e5b94d58b5066e423c145d37f3098b718d9b4.jpg)

![50fff4363fdcc945cf9d670ddb2bf8fd4db5a31bbf865a363c490ab087401780.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/50fff4363fdcc945cf9d670ddb2bf8fd4db5a31bbf865a363c490ab087401780.jpg)

![5324c0842d12527306aa87e5e1edbe5ba7e3419ce3c51cf34d616eec2bd292d5.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/5324c0842d12527306aa87e5e1edbe5ba7e3419ce3c51cf34d616eec2bd292d5.jpg)

![5c168b38f0f98fb139c5c0f779e3a58e6ae0f4f02e5fcc2e45046d4d8165ec5a.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/5c168b38f0f98fb139c5c0f779e3a58e6ae0f4f02e5fcc2e45046d4d8165ec5a.jpg)

![5df4504e4b4c7d0d3db4d470f9b8637092f6ff38ba7f176555c98423cf8a4f97.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/5df4504e4b4c7d0d3db4d470f9b8637092f6ff38ba7f176555c98423cf8a4f97.jpg)

![6dba737243f63aa3f185f1f00589e62857d690dab2a401c15893b75bc4a887c1.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/6dba737243f63aa3f185f1f00589e62857d690dab2a401c15893b75bc4a887c1.jpg)

![7a65a54e4766d1110c0cebe705b8e7eaeb83257a65ac8e37fcfeeea602b23222.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/7a65a54e4766d1110c0cebe705b8e7eaeb83257a65ac8e37fcfeeea602b23222.jpg)

![7f938ff4f808e949ee31c35cc044362f06561cbbf3a8ee3db732b4308c501e36.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/7f938ff4f808e949ee31c35cc044362f06561cbbf3a8ee3db732b4308c501e36.jpg)

![9b2676d08c82bf4fbba0539303790eb763b56c40ea0cf1058ed2e2a763747b4f.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/9b2676d08c82bf4fbba0539303790eb763b56c40ea0cf1058ed2e2a763747b4f.jpg)

![9edca8b7ba81502e2f19266bc5acf136eccff90f90b8134ce6a3f2df651ad5c8.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/9edca8b7ba81502e2f19266bc5acf136eccff90f90b8134ce6a3f2df651ad5c8.jpg)

![b1c9659fb1260321daf003095460748d951c4b37bb820cdfd2583b45d80b4643.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/b1c9659fb1260321daf003095460748d951c4b37bb820cdfd2583b45d80b4643.jpg)

![b68eacdf1e01d7f1f0699049a189ca2e471a32ebcf36feae97b19c895dd35b20.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/b68eacdf1e01d7f1f0699049a189ca2e471a32ebcf36feae97b19c895dd35b20.jpg)

![b8de919d4820257a4e3451dfea44013be29d05c456c56016c12eb9681cd51730.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/b8de919d4820257a4e3451dfea44013be29d05c456c56016c12eb9681cd51730.jpg)

![c81d64e6d20516f8080b29a86fc86663b5dfa0259a9a58d529a7a5342b32f084.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/c81d64e6d20516f8080b29a86fc86663b5dfa0259a9a58d529a7a5342b32f084.jpg)

![cee2240ed2fb09fb4aa56de9cafd734f015fc351c9bd983ae673428897881fe3.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/cee2240ed2fb09fb4aa56de9cafd734f015fc351c9bd983ae673428897881fe3.jpg)

![d0cf78fde680e1b2ec4f0ab8b35c17a9441b660e0b7b5874d9fe84c017278897.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/d0cf78fde680e1b2ec4f0ab8b35c17a9441b660e0b7b5874d9fe84c017278897.jpg)

![d1ab966120545d6850aee31d876a005331a56cd68177694ecb503159ebc2d9e8.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/d1ab966120545d6850aee31d876a005331a56cd68177694ecb503159ebc2d9e8.jpg)

![d95de4c6ca256bb541ae60b492aa8f473a387c0e687f0ef11cdad8a2e72bf011.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/d95de4c6ca256bb541ae60b492aa8f473a387c0e687f0ef11cdad8a2e72bf011.jpg)

![dcea9ed85d115e1f0681bcaf02cdc9e122ae13f63c7ca8efb29d7d3e96b81c62.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/dcea9ed85d115e1f0681bcaf02cdc9e122ae13f63c7ca8efb29d7d3e96b81c62.jpg)

![e98a325b458055b295a8eb5e2fa75933604cc50a5e9bd69008b417aec41fe8a0.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/e98a325b458055b295a8eb5e2fa75933604cc50a5e9bd69008b417aec41fe8a0.jpg)

![ea11d436cbabe625f0870a4573f742388241c4f4d57fc63dbe9d8d16dea96796.jpg](../icml_results/1726_Fast%20and%20Expressive%20Multi-Byte%20Prediction%20with%20Probabilistic%20Circuits/tables/ea11d436cbabe625f0870a4573f742388241c4f4d57fc63dbe9d8d16dea96796.jpg)

<a id="paper-1705"></a>

## Approximating Drift-Diffusion Models for User Decisions under Nudging and External Information


### Images

![363ff9c93edcd73b5ebcc6460492a09c8ed07634965417a57c7216f183a5f7ca.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/images/363ff9c93edcd73b5ebcc6460492a09c8ed07634965417a57c7216f183a5f7ca.jpg)

### Tables

![10cba4d74012cd57953eae5493e8caa0348da89dc6d54681693395ab1fa1c000.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/10cba4d74012cd57953eae5493e8caa0348da89dc6d54681693395ab1fa1c000.jpg)

![26710004dd215be7b912c6328aeda518d7e96dee82560882a1050debf7727d3c.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/26710004dd215be7b912c6328aeda518d7e96dee82560882a1050debf7727d3c.jpg)

![299457e6a80c6928de75b934d56115557df9477177534166d6ae8a10694e59b2.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/299457e6a80c6928de75b934d56115557df9477177534166d6ae8a10694e59b2.jpg)

![40f8acea5f165882a42113619ed3699b9f0db9b6ab22c7efad50031e8b8fb509.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/40f8acea5f165882a42113619ed3699b9f0db9b6ab22c7efad50031e8b8fb509.jpg)

![5bce7d697aedc9dedbec45a859f5c49563fcdbb68b9dc59c21c7ea381168bad1.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/5bce7d697aedc9dedbec45a859f5c49563fcdbb68b9dc59c21c7ea381168bad1.jpg)

![62cbec39ddae70420b9504f0899b9146463bbdeb4317b65b13aa28473ff56a99.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/62cbec39ddae70420b9504f0899b9146463bbdeb4317b65b13aa28473ff56a99.jpg)

![84c5fb67b6dc4c5d72c5b716f37e4f729df1cdb55b5ae7980864ba48158db042.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/84c5fb67b6dc4c5d72c5b716f37e4f729df1cdb55b5ae7980864ba48158db042.jpg)

![86744bb211281deaa44053d4949c69802467caa400eff74b2d3e3108a6b0d13a.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/86744bb211281deaa44053d4949c69802467caa400eff74b2d3e3108a6b0d13a.jpg)

![b0f3dda19b077590d715dd5ae66b8d2c9fa96c5f774506b9bd0acd39a9592e4a.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/b0f3dda19b077590d715dd5ae66b8d2c9fa96c5f774506b9bd0acd39a9592e4a.jpg)

![b5a34d2a8f0c1bcf00aaa2448d0d171748ddf5cc81c7c44ee0905cfe715d9638.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/b5a34d2a8f0c1bcf00aaa2448d0d171748ddf5cc81c7c44ee0905cfe715d9638.jpg)

![cac4d7267c66259cc3410f0025ccce3e327ec89102720cf48c46dd760cbf0276.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/cac4d7267c66259cc3410f0025ccce3e327ec89102720cf48c46dd760cbf0276.jpg)

![d51b15f9841e5dec92be5540ca44f3922c6332f1c7ab54f136b8efb128f884f1.jpg](../icml_results/1727_Approximating%20Drift-Diffusion%20Models%20for%20User%20Decisions%20under%20Nudging%20and%20External%20Information/tables/d51b15f9841e5dec92be5540ca44f3922c6332f1c7ab54f136b8efb128f884f1.jpg)

<a id="paper-1706"></a>

## Revisiting the Platonic Representation Hypothesis: An Aristotelian View


### Images

![0127649ac6507e7f1f9382ad22a8c732c421a5990adea29e404f6b051087a648.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/0127649ac6507e7f1f9382ad22a8c732c421a5990adea29e404f6b051087a648.jpg)

![0f5d682f892c1e2d56429e3f5b1c39f1153e3d2a4623294161053b68e6d66983.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/0f5d682f892c1e2d56429e3f5b1c39f1153e3d2a4623294161053b68e6d66983.jpg)

![3513f7491925849292d0ca2c77e96e7d15abb6fc84a69eb530dde77d18b3e250.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/3513f7491925849292d0ca2c77e96e7d15abb6fc84a69eb530dde77d18b3e250.jpg)

![36eced2ee73a7065096696d0e3181237c3c78fdf9b5ff1e9b2e18f52ab5abac9.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/36eced2ee73a7065096696d0e3181237c3c78fdf9b5ff1e9b2e18f52ab5abac9.jpg)

![48c844ac2cf799e7190b23088330b891f28cd377a1cd9ae0b326670d2fe9ae82.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/48c844ac2cf799e7190b23088330b891f28cd377a1cd9ae0b326670d2fe9ae82.jpg)

![4d99ab654492a85486c0de66f8573721bf165407200354a1288686d44c593fb7.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/4d99ab654492a85486c0de66f8573721bf165407200354a1288686d44c593fb7.jpg)

![4f9769e8bb5440f055e3531c342250ac859d3b70cf8d95cdd02138db0d6c4278.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/4f9769e8bb5440f055e3531c342250ac859d3b70cf8d95cdd02138db0d6c4278.jpg)

![53f21df65a740a666fcd0ddad924b06e57242c36e9de38401909389991d20fc8.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/53f21df65a740a666fcd0ddad924b06e57242c36e9de38401909389991d20fc8.jpg)

![576d356f514fbd1cc71672e79f2098229180d83761ba895698502b3a17a9f835.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/576d356f514fbd1cc71672e79f2098229180d83761ba895698502b3a17a9f835.jpg)

![5a7f31c951299d23e41e388fe79435df19cbab4c531247574b86e0ac1624a5b9.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/5a7f31c951299d23e41e388fe79435df19cbab4c531247574b86e0ac1624a5b9.jpg)

![5bf4f072788974c2e06aa4d58a52765a53fbf45263a5fc942659c15f2d161a50.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/5bf4f072788974c2e06aa4d58a52765a53fbf45263a5fc942659c15f2d161a50.jpg)

![6168f3f0a076ad47b31783cc1ba382521b0a477da1b72ef02a720313bfb0c846.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/6168f3f0a076ad47b31783cc1ba382521b0a477da1b72ef02a720313bfb0c846.jpg)

![6bdee2f0fb7ac750e2250c2cb59f5e211a75fd75d7b095667a2a03838b48b6d7.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/6bdee2f0fb7ac750e2250c2cb59f5e211a75fd75d7b095667a2a03838b48b6d7.jpg)

![8a71cd889653f79e0fb631f8dc1867c022d4db4adc8a61294cc138956c6ffe3e.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/8a71cd889653f79e0fb631f8dc1867c022d4db4adc8a61294cc138956c6ffe3e.jpg)

![91c53a4cb2469d4e72e38807297359472ff25ae5b15be34b2f97173932f2baa1.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/91c53a4cb2469d4e72e38807297359472ff25ae5b15be34b2f97173932f2baa1.jpg)

![a545e26a3c31c689f712f62b17007fc0d9a7ea2b6af94e6ab426d8a037a74b78.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/a545e26a3c31c689f712f62b17007fc0d9a7ea2b6af94e6ab426d8a037a74b78.jpg)

![b03306b5645733750851bc417a42f8cccae8fd167a6b9cf27a7ad00213954f3c.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/b03306b5645733750851bc417a42f8cccae8fd167a6b9cf27a7ad00213954f3c.jpg)

![b68d6bb5dcc944d4094f90f10596e3721834f15d4217c073a4255332867882c4.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/b68d6bb5dcc944d4094f90f10596e3721834f15d4217c073a4255332867882c4.jpg)

![b82b47a6f565cacb959cb1241a6dc016bbc96e4c586af48b5eb063265acccc7c.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/b82b47a6f565cacb959cb1241a6dc016bbc96e4c586af48b5eb063265acccc7c.jpg)

![bef949ad01dc66d1e5608e73c233f89daa7da334708b8801bada1ae0eff64c8d.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/bef949ad01dc66d1e5608e73c233f89daa7da334708b8801bada1ae0eff64c8d.jpg)

![c9956520840de272d11be2c9e208c2e4bd2b0eec8f320980c3b73fd2ce46e7fe.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/c9956520840de272d11be2c9e208c2e4bd2b0eec8f320980c3b73fd2ce46e7fe.jpg)

![ca1c30b1f86b0c2936517de50f2eef2266a1987e261b9bdde6912e134b606c57.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/ca1c30b1f86b0c2936517de50f2eef2266a1987e261b9bdde6912e134b606c57.jpg)

![cc31eb416d73e4b8a01b8ce5e8a7903ea898457b2a9831fe506a5c7db666d49c.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/cc31eb416d73e4b8a01b8ce5e8a7903ea898457b2a9831fe506a5c7db666d49c.jpg)

![d0cad2282c657c2bb93236958ecb0a3f93537747f68bfada56f6b9cbb735a73d.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/d0cad2282c657c2bb93236958ecb0a3f93537747f68bfada56f6b9cbb735a73d.jpg)

![dd783c712fe4f40c4e1e36a3bd6a9ddf59dde5f5762411445f4c2bf36bf74509.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/dd783c712fe4f40c4e1e36a3bd6a9ddf59dde5f5762411445f4c2bf36bf74509.jpg)

![e3f61f5c55429c705bfd3d3451f3916c8d2a4eee068afff99c2486a33e493e93.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/e3f61f5c55429c705bfd3d3451f3916c8d2a4eee068afff99c2486a33e493e93.jpg)

![e43e250a1706707e83fc620f18314e2c63c933b88f31ee4dce2ad165a98fd086.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/e43e250a1706707e83fc620f18314e2c63c933b88f31ee4dce2ad165a98fd086.jpg)

![ebf05749b529ac8995ad814cd8402810e8b70131e5767ea8e110728f4dea9015.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/ebf05749b529ac8995ad814cd8402810e8b70131e5767ea8e110728f4dea9015.jpg)

![efc51a2161ce2c13db280ae71b325a69bf45ee465be951e2a56c6a0fac81d54f.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/efc51a2161ce2c13db280ae71b325a69bf45ee465be951e2a56c6a0fac81d54f.jpg)

![ff065aa87e3e0d070c4d4271e2ebd0eca2d2e826a1532e18ed6a4d5dbf1d9206.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/ff065aa87e3e0d070c4d4271e2ebd0eca2d2e826a1532e18ed6a4d5dbf1d9206.jpg)

![ff648967759a006d4ae834b11a79c8dde597e074ef6aca05f7f1ec3486966838.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/images/ff648967759a006d4ae834b11a79c8dde597e074ef6aca05f7f1ec3486966838.jpg)

### Tables

![94ef9bdc3e7f46c64c0f63ad70742677f57b0866e15cebca12380a7bd2a2e7e8.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/tables/94ef9bdc3e7f46c64c0f63ad70742677f57b0866e15cebca12380a7bd2a2e7e8.jpg)

![afea7a6b5122a10953ee2bab4aab457300ab4017141e4b005be7dce83ef28b75.jpg](../icml_results/1728_Revisiting%20the%20Platonic%20Representation%20Hypothesis_%20An%20Aristotelian%20View/tables/afea7a6b5122a10953ee2bab4aab457300ab4017141e4b005be7dce83ef28b75.jpg)

<a id="paper-1707"></a>

## The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks


### Images

![3a09aea7326175ffd28f69638214a89ae40384228219b57fdfd2be97d0015e6b.jpg](../icml_results/1729_The%20Implicit%20Bias%20of%20Adam%20and%20Muon%20on%20Smooth%20Homogeneous%20Neural%20Networks/images/3a09aea7326175ffd28f69638214a89ae40384228219b57fdfd2be97d0015e6b.jpg)

![6558438bfbe4e7bf6548529cadf9b0f447ab3d8aa3a23948201a0cb9a5cd8f3e.jpg](../icml_results/1729_The%20Implicit%20Bias%20of%20Adam%20and%20Muon%20on%20Smooth%20Homogeneous%20Neural%20Networks/images/6558438bfbe4e7bf6548529cadf9b0f447ab3d8aa3a23948201a0cb9a5cd8f3e.jpg)

![951e419553cb520261687ce342dc74e9116ca2b24bab86bab40833ec7712f1e8.jpg](../icml_results/1729_The%20Implicit%20Bias%20of%20Adam%20and%20Muon%20on%20Smooth%20Homogeneous%20Neural%20Networks/images/951e419553cb520261687ce342dc74e9116ca2b24bab86bab40833ec7712f1e8.jpg)

### Tables

![89e138ca1a70e1ef5cf55cf8a4eb3c6ec40dcf2c79ae861340e744d25a1e8846.jpg](../icml_results/1729_The%20Implicit%20Bias%20of%20Adam%20and%20Muon%20on%20Smooth%20Homogeneous%20Neural%20Networks/tables/89e138ca1a70e1ef5cf55cf8a4eb3c6ec40dcf2c79ae861340e744d25a1e8846.jpg)

![f7578da57466bebd5034be615a4af3ad625f367f860dea78d60fec0dcf0d8095.jpg](../icml_results/1729_The%20Implicit%20Bias%20of%20Adam%20and%20Muon%20on%20Smooth%20Homogeneous%20Neural%20Networks/tables/f7578da57466bebd5034be615a4af3ad625f367f860dea78d60fec0dcf0d8095.jpg)

<a id="paper-1708"></a>

## Feedback Control for Multi-Objective Graph Self-Supervision


### Images

![169809ea0951f11d23d832490dd8721e4fcf5d67ffcff951633e3f0b36f764b9.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/169809ea0951f11d23d832490dd8721e4fcf5d67ffcff951633e3f0b36f764b9.jpg)

![225b6ad7feb31a3809a86ef7ae8e8077060a8b8e633bd970267e647125a511e1.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/225b6ad7feb31a3809a86ef7ae8e8077060a8b8e633bd970267e647125a511e1.jpg)

![27f088ae118427930f844484341037761d28a391f04657d8fcc3c2297271846d.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/27f088ae118427930f844484341037761d28a391f04657d8fcc3c2297271846d.jpg)

![53c930b881935247f67daa5a68a2955157fe00accde9fa8583c5656ae4fc1e72.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/53c930b881935247f67daa5a68a2955157fe00accde9fa8583c5656ae4fc1e72.jpg)

![54eed4a8fd3ce98f4287bc26411e0e0c9c917a6446136a0b4d0b85be5785ae2e.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/54eed4a8fd3ce98f4287bc26411e0e0c9c917a6446136a0b4d0b85be5785ae2e.jpg)

![5cdfc35403f55f9ebf2c55c2329da69c3e220995ebbcfede15c4720c3d8d88a8.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/5cdfc35403f55f9ebf2c55c2329da69c3e220995ebbcfede15c4720c3d8d88a8.jpg)

![7f50c8ab96e447c72b6debf48a7c5ed4f48e5a44fdc03141b1537de2f833c02c.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/7f50c8ab96e447c72b6debf48a7c5ed4f48e5a44fdc03141b1537de2f833c02c.jpg)

![cac24363384113fddb4ea070d9fe93ccd808f29dd2184a5c040740f7f2f3835e.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/cac24363384113fddb4ea070d9fe93ccd808f29dd2184a5c040740f7f2f3835e.jpg)

![d1ddc0f8799c9c90c62bca83203bdcd2aa4b96e0d17cc6e5673da077f20eb926.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/d1ddc0f8799c9c90c62bca83203bdcd2aa4b96e0d17cc6e5673da077f20eb926.jpg)

![e60f1fa4582b66ad36562bd2486e3cb0c8ee85f61325142ab63eb12f2d7a3dd4.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/images/e60f1fa4582b66ad36562bd2486e3cb0c8ee85f61325142ab63eb12f2d7a3dd4.jpg)

### Tables

![0899e68478d0eb85e2c7399a66b63e295c5a7d3b011cd468bec87da718aa1ca6.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/0899e68478d0eb85e2c7399a66b63e295c5a7d3b011cd468bec87da718aa1ca6.jpg)

![2b18abc78a7b32fdb86c8190c8219d72b695b7aed485f29eb79a5bc8d3fbbe46.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/2b18abc78a7b32fdb86c8190c8219d72b695b7aed485f29eb79a5bc8d3fbbe46.jpg)

![35ac3dbabb6332cb9f8ba137d91ae95e227070ab20b47e0fa01ca75e74eb245f.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/35ac3dbabb6332cb9f8ba137d91ae95e227070ab20b47e0fa01ca75e74eb245f.jpg)

![7d56eedfa8d45b8c6f16fb18b9c7f7b8e4c861df88c88bbdebfe02840fc859d1.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/7d56eedfa8d45b8c6f16fb18b9c7f7b8e4c861df88c88bbdebfe02840fc859d1.jpg)

![945afff0188da85b1ebf776c046b0a3186b08647af8ae90d8571dc015458c9fa.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/945afff0188da85b1ebf776c046b0a3186b08647af8ae90d8571dc015458c9fa.jpg)

![9f4796da5670fad20bd073ae7cef0741e7059985e503f4527c541890d5c3d2c9.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/9f4796da5670fad20bd073ae7cef0741e7059985e503f4527c541890d5c3d2c9.jpg)

![a67a9881138e32db854eecf8450556bc80bae775a91621ffd6cf58f40b1010db.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/a67a9881138e32db854eecf8450556bc80bae775a91621ffd6cf58f40b1010db.jpg)

![c92de288e284cb9b5bbdebc1ffd0caf3d6b902709ea7f43c906c5a011035e859.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/c92de288e284cb9b5bbdebc1ffd0caf3d6b902709ea7f43c906c5a011035e859.jpg)

![cc50a53cbfaa8e8907614122d5406c415aa65c382509af018d5acf5310fb8dfb.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/cc50a53cbfaa8e8907614122d5406c415aa65c382509af018d5acf5310fb8dfb.jpg)

![cc62f3ae3a6b0471cb5fe9025600993f91fb1fbd980e03a28adfb4b1f2634093.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/cc62f3ae3a6b0471cb5fe9025600993f91fb1fbd980e03a28adfb4b1f2634093.jpg)

![d1ecd6aea7c912fd569ba2213de1ef5a12c3237bbb879f79da7da25dfb7d99f0.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/d1ecd6aea7c912fd569ba2213de1ef5a12c3237bbb879f79da7da25dfb7d99f0.jpg)

![f47b3f3039877346557c89721a64adaa045b6cac368352ff32359de490686e63.jpg](../icml_results/1731_Feedback%20Control%20for%20Multi-Objective%20Graph%20Self-Supervision/tables/f47b3f3039877346557c89721a64adaa045b6cac368352ff32359de490686e63.jpg)

<a id="paper-1709"></a>

## Text Has Curvature


### Images

![0e2be9da7827e28f8f19528999950234ec4bd82bd876dd8296688f38d19dc28f.jpg](../icml_results/1732_Text%20Has%20Curvature/images/0e2be9da7827e28f8f19528999950234ec4bd82bd876dd8296688f38d19dc28f.jpg)

![14750c5d183aeea5507b5c27b086fca3736e0a2e450fee9d392ffeff4a96e91f.jpg](../icml_results/1732_Text%20Has%20Curvature/images/14750c5d183aeea5507b5c27b086fca3736e0a2e450fee9d392ffeff4a96e91f.jpg)

![2be131c485a387079fd16acb1af7bc49ab7475ae36e3977ab8e1384af6c2c588.jpg](../icml_results/1732_Text%20Has%20Curvature/images/2be131c485a387079fd16acb1af7bc49ab7475ae36e3977ab8e1384af6c2c588.jpg)

![49cb0cef518e5df42245c11f6ea44c0376d92d1682a84e020a2e6bbdbc3f3f59.jpg](../icml_results/1732_Text%20Has%20Curvature/images/49cb0cef518e5df42245c11f6ea44c0376d92d1682a84e020a2e6bbdbc3f3f59.jpg)

![5357611be48eeab6a7295106eea90be9f82838d8433c634f950cd082cd21b2ac.jpg](../icml_results/1732_Text%20Has%20Curvature/images/5357611be48eeab6a7295106eea90be9f82838d8433c634f950cd082cd21b2ac.jpg)

![77c6954617d4540f6559debd55dcb555e1859596c5e4a04ba2c29ebb54e69d42.jpg](../icml_results/1732_Text%20Has%20Curvature/images/77c6954617d4540f6559debd55dcb555e1859596c5e4a04ba2c29ebb54e69d42.jpg)

![8b77a776792ec77772414a81055810f2bc43bd1c8f088521de7008bb02a538c7.jpg](../icml_results/1732_Text%20Has%20Curvature/images/8b77a776792ec77772414a81055810f2bc43bd1c8f088521de7008bb02a538c7.jpg)

![91f8a0acbec25d9a0d3134a50a3538229bb7bacb28aa4445b6e65590e4b293a5.jpg](../icml_results/1732_Text%20Has%20Curvature/images/91f8a0acbec25d9a0d3134a50a3538229bb7bacb28aa4445b6e65590e4b293a5.jpg)

![98682ad18b989b055ad77b96ff9decca1022c5ff584f87129cc79d22748d538c.jpg](../icml_results/1732_Text%20Has%20Curvature/images/98682ad18b989b055ad77b96ff9decca1022c5ff584f87129cc79d22748d538c.jpg)

![a990d00e0ee4ac93196753a880625d2e7b1ab1e0a46c0d513f706502784f94b0.jpg](../icml_results/1732_Text%20Has%20Curvature/images/a990d00e0ee4ac93196753a880625d2e7b1ab1e0a46c0d513f706502784f94b0.jpg)

![b9294ed863ea6f706084b731375a560ed94e4958d59bb70e4f0885d40ef9a9fa.jpg](../icml_results/1732_Text%20Has%20Curvature/images/b9294ed863ea6f706084b731375a560ed94e4958d59bb70e4f0885d40ef9a9fa.jpg)

![ce3617b3933916eaae2cb215af1ac2ee9a424c29022c195391eb285c8d32bd7a.jpg](../icml_results/1732_Text%20Has%20Curvature/images/ce3617b3933916eaae2cb215af1ac2ee9a424c29022c195391eb285c8d32bd7a.jpg)

![db37b5441e8c1d26bf76d743d4257c7158abc5a1f9c5e049faeccc1969ee1306.jpg](../icml_results/1732_Text%20Has%20Curvature/images/db37b5441e8c1d26bf76d743d4257c7158abc5a1f9c5e049faeccc1969ee1306.jpg)

![db3e5fe0dd18297ebf9cd97e4eb3035db583a5e7110259d10d9be9a1c634b6c1.jpg](../icml_results/1732_Text%20Has%20Curvature/images/db3e5fe0dd18297ebf9cd97e4eb3035db583a5e7110259d10d9be9a1c634b6c1.jpg)

![dbd3efc3b72f1b340c88aae52e3f9e3644717ab30c9f1d06cfacdf8e29422aa4.jpg](../icml_results/1732_Text%20Has%20Curvature/images/dbd3efc3b72f1b340c88aae52e3f9e3644717ab30c9f1d06cfacdf8e29422aa4.jpg)

![e49670eb71f14027b3eacd9613e625e8f01ea44bb449f3d7336afd85da1bb5f0.jpg](../icml_results/1732_Text%20Has%20Curvature/images/e49670eb71f14027b3eacd9613e625e8f01ea44bb449f3d7336afd85da1bb5f0.jpg)

![f33b7340e96b2a0089aa475224cb939f35d3dd7df71c5aee22b7f30e213c1922.jpg](../icml_results/1732_Text%20Has%20Curvature/images/f33b7340e96b2a0089aa475224cb939f35d3dd7df71c5aee22b7f30e213c1922.jpg)

![f59e694397ef17c3f729e1d963073730b34948bf4b6e78c399a5ffff8ffe88bf.jpg](../icml_results/1732_Text%20Has%20Curvature/images/f59e694397ef17c3f729e1d963073730b34948bf4b6e78c399a5ffff8ffe88bf.jpg)

### Tables

![0a9d5df9e367cac6124d7c56056ce7feda8687884fd53eb0824aa5b2328c1481.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/0a9d5df9e367cac6124d7c56056ce7feda8687884fd53eb0824aa5b2328c1481.jpg)

![15a120ff9422fe38ce9cc10a1640772564234c4019cf803bcf78658e74096df3.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/15a120ff9422fe38ce9cc10a1640772564234c4019cf803bcf78658e74096df3.jpg)

![2739334b5fcc340d56e0e595e14e56fa2295be24a4029c360b6aa44d7ef74eba.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/2739334b5fcc340d56e0e595e14e56fa2295be24a4029c360b6aa44d7ef74eba.jpg)

![33044cfabde47864113af4095c6e92833e71f52ebbe6de8f71a750f41bdd15ca.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/33044cfabde47864113af4095c6e92833e71f52ebbe6de8f71a750f41bdd15ca.jpg)

![4294dc76fffa947db10da50a1e10b65f2fe053aac3c99162c5cf733111e302fa.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/4294dc76fffa947db10da50a1e10b65f2fe053aac3c99162c5cf733111e302fa.jpg)

![500ae7be15babc245c09bf2f51b912e40d2b28bdb1b8c3e042a80b1afd548a80.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/500ae7be15babc245c09bf2f51b912e40d2b28bdb1b8c3e042a80b1afd548a80.jpg)

![74d2714a45438bc97f23e4109c356bece859c9f48aa80c015ba3b1677d09f7eb.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/74d2714a45438bc97f23e4109c356bece859c9f48aa80c015ba3b1677d09f7eb.jpg)

![750a2baa1258d93935daef11f3e5fdba221f3db684ad59a09b988750d53dee56.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/750a2baa1258d93935daef11f3e5fdba221f3db684ad59a09b988750d53dee56.jpg)

![836b7cc1aa7caa12bf6e63fdd2736840e23577bf1d5522231fe1832092a5601a.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/836b7cc1aa7caa12bf6e63fdd2736840e23577bf1d5522231fe1832092a5601a.jpg)

![cded03a71bc535b87b3ccd719f7b43158911ddf288abda2c6ae608c26c2ddc10.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/cded03a71bc535b87b3ccd719f7b43158911ddf288abda2c6ae608c26c2ddc10.jpg)

![ecf202c56b5bef377c6e085003c0c4059c97c47e45a1c7c38476d0fe634060ff.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/ecf202c56b5bef377c6e085003c0c4059c97c47e45a1c7c38476d0fe634060ff.jpg)

![fd8a37acf8f5d54a2e678ecbf03dc676955614338bc603794ec9118112940195.jpg](../icml_results/1732_Text%20Has%20Curvature/tables/fd8a37acf8f5d54a2e678ecbf03dc676955614338bc603794ec9118112940195.jpg)

<a id="paper-1710"></a>

## Tensor Train Diffusion: Leveraging Low-Rank Structures for High-Dimensional Score-Based Sampling


### Images

![1200b9f6910f6f6292b1c4a1cbf81a89fa423a774fb56733c6d853f0f54d4975.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/1200b9f6910f6f6292b1c4a1cbf81a89fa423a774fb56733c6d853f0f54d4975.jpg)

![3b4a2043b29174a5657bab1e17c31fc60b6352dd4cbfe74b5065b961b9459526.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/3b4a2043b29174a5657bab1e17c31fc60b6352dd4cbfe74b5065b961b9459526.jpg)

![454f0bf02605e715994bc0662ccc5f9656007dad36183d90a0e873ab79b21177.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/454f0bf02605e715994bc0662ccc5f9656007dad36183d90a0e873ab79b21177.jpg)

![4a88313c3e1de2cd1b9cde7528f5e83408ae09e38750ff89cb71c6db37ebe606.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/4a88313c3e1de2cd1b9cde7528f5e83408ae09e38750ff89cb71c6db37ebe606.jpg)

![6571b1b45a0bdba5160dc35d088a6d9a45ff02a31e89016d9274286286ec31cd.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/6571b1b45a0bdba5160dc35d088a6d9a45ff02a31e89016d9274286286ec31cd.jpg)

![798bca48ee6de4de7be7273462c204a9d6423d03d581e376ce0e813cb5706fb6.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/798bca48ee6de4de7be7273462c204a9d6423d03d581e376ce0e813cb5706fb6.jpg)

![85ea74ff70e9912165a04c00c5fcea5860fd45c1841192e5b86a8c6919c7546f.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/85ea74ff70e9912165a04c00c5fcea5860fd45c1841192e5b86a8c6919c7546f.jpg)

![a1709f0e25cc455e7209c1cf21bc9305f2c23e4c009d26455a25c70c7fcc2861.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/a1709f0e25cc455e7209c1cf21bc9305f2c23e4c009d26455a25c70c7fcc2861.jpg)

![a56ee7c59b5bb878fe99019fc26d322e194a2ca8c42e520ccb2451c40f46de1c.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/a56ee7c59b5bb878fe99019fc26d322e194a2ca8c42e520ccb2451c40f46de1c.jpg)

![a78a3c8add3a97f7333182ff9420e2f46495ee59c4b869c393734adedae0afdc.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/a78a3c8add3a97f7333182ff9420e2f46495ee59c4b869c393734adedae0afdc.jpg)

![ae178e53d9d858bc8402d1a9c38acc667641923814d2f3fa3a5a34beedc3ee14.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/ae178e53d9d858bc8402d1a9c38acc667641923814d2f3fa3a5a34beedc3ee14.jpg)

![b56ca603e690d4ec4d859b0837182ebff4ea435931eb3d1f4d9a872c0fba76fc.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/b56ca603e690d4ec4d859b0837182ebff4ea435931eb3d1f4d9a872c0fba76fc.jpg)

![bf1d570fbaf1b0d45bf176f3a0641d1a04894717898804f1a2a833d7201c3c67.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/bf1d570fbaf1b0d45bf176f3a0641d1a04894717898804f1a2a833d7201c3c67.jpg)

![dc29028830b9d2a90fd36b12243154200d6e22bd6a9e0ad4ff5c10a65d286f0c.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/dc29028830b9d2a90fd36b12243154200d6e22bd6a9e0ad4ff5c10a65d286f0c.jpg)

![fc7df41157ddfd7af8cea61d397fe0c4d6f9109c594cc0cba6ccc19c1bc565e8.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/images/fc7df41157ddfd7af8cea61d397fe0c4d6f9109c594cc0cba6ccc19c1bc565e8.jpg)

### Tables

![4553bfe07a97792c1de37af7da02f856e6b8a91dd344d9286dd4dd2368616ad7.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/tables/4553bfe07a97792c1de37af7da02f856e6b8a91dd344d9286dd4dd2368616ad7.jpg)

![b9e113d1ecaa06af7cde0a4638d656bd3ba0e42b08114ea50f1473cd39e13040.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/tables/b9e113d1ecaa06af7cde0a4638d656bd3ba0e42b08114ea50f1473cd39e13040.jpg)

![bebba0886615262285fa529770b1b68118e40f32d6d8f862b4ecd451af9b70ad.jpg](../icml_results/1733_Tensor%20Train%20Diffusion_%20Leveraging%20Low-Rank%20Structures%20for%20High-Dimensional%20Score-Based%20Sampling/tables/bebba0886615262285fa529770b1b68118e40f32d6d8f862b4ecd451af9b70ad.jpg)

<a id="paper-1711"></a>

## SARL: Structure-Aligned Reinforcement Learning for Bridging the Perception-Action Gap in Airspace


### Images

![079b17711de9c3bcb80c71d7513b353f140d4b2576dfd68f0627bea3cdbacc8c.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/079b17711de9c3bcb80c71d7513b353f140d4b2576dfd68f0627bea3cdbacc8c.jpg)

![25b032330995496612380fae042ced5e8e1db6f406ebfbd6b26c20825c520bf2.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/25b032330995496612380fae042ced5e8e1db6f406ebfbd6b26c20825c520bf2.jpg)

![3e1c90898b7f134b39ef933d2403c7e8185edb52670dae7f99f40a36f380595c.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/3e1c90898b7f134b39ef933d2403c7e8185edb52670dae7f99f40a36f380595c.jpg)

![4b09b04da49613ee4a2dbbeac2ce06daa37ba3b63418e184e5e6382f12d28e2b.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/4b09b04da49613ee4a2dbbeac2ce06daa37ba3b63418e184e5e6382f12d28e2b.jpg)

![5bf0f75926b168fd7565f7a0e666f3adc84f5e804b99d558e72db183e35958e6.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/5bf0f75926b168fd7565f7a0e666f3adc84f5e804b99d558e72db183e35958e6.jpg)

![86f376c3238f405137c48b9d996cabe4c8d2aeb89fa9bf4e142db63aa36ecfd4.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/86f376c3238f405137c48b9d996cabe4c8d2aeb89fa9bf4e142db63aa36ecfd4.jpg)

![a39fcf74361765a6bfa4c86a3d532e3bc06acc166fb879bdbda3475d1b34d1a8.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/a39fcf74361765a6bfa4c86a3d532e3bc06acc166fb879bdbda3475d1b34d1a8.jpg)

![ae6ca96c36c555267ca3ab8ede54802cb2003d3ed4871324881fbe7ced610880.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/ae6ca96c36c555267ca3ab8ede54802cb2003d3ed4871324881fbe7ced610880.jpg)

![d096c4016ade8c7792f750b4764e930fbe21322168b1e0cf79de97e282b5316b.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/d096c4016ade8c7792f750b4764e930fbe21322168b1e0cf79de97e282b5316b.jpg)

![fc6ad68d64f2414c1f2a683894fa7c2c629c7a928ba989550af7e7d35f3a735f.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/images/fc6ad68d64f2414c1f2a683894fa7c2c629c7a928ba989550af7e7d35f3a735f.jpg)

### Tables

![290b5b61b7ee0e0aab17f41f2afb2608691b9c4caaa8aa1f8688c51c872607dc.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/290b5b61b7ee0e0aab17f41f2afb2608691b9c4caaa8aa1f8688c51c872607dc.jpg)

![3d6e76de87e7b38e9fb52b9ca16cb26e9d44963d791a23b26319ee1444083790.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/3d6e76de87e7b38e9fb52b9ca16cb26e9d44963d791a23b26319ee1444083790.jpg)

![540915eaab39cee350d482d09be14ff0251c1e4fdfac463adf6324d54103d384.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/540915eaab39cee350d482d09be14ff0251c1e4fdfac463adf6324d54103d384.jpg)

![8513939eb3f923c63b5295683779c508ce96078df123f8ed13a8cbc44ab69d3a.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/8513939eb3f923c63b5295683779c508ce96078df123f8ed13a8cbc44ab69d3a.jpg)

![c2abfdd048d2a4985af3e046062e067345a436e2bdb7fa45b2e1b7b91ddea1d0.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/c2abfdd048d2a4985af3e046062e067345a436e2bdb7fa45b2e1b7b91ddea1d0.jpg)

![c9beccab5ce30e94ea1ab14c4fc32aa691d6cf746e0cc65c57a211dbee90165f.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/c9beccab5ce30e94ea1ab14c4fc32aa691d6cf746e0cc65c57a211dbee90165f.jpg)

![dbd24d911030ff0aba6fa6e130f35e35a5ca382d9e6c9e1b8a9c07b46249a8ea.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/dbd24d911030ff0aba6fa6e130f35e35a5ca382d9e6c9e1b8a9c07b46249a8ea.jpg)

![e20aa83eaa800206e3964dfaf395abef42e0048af27d3b7b3fc50b64643acf58.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/e20aa83eaa800206e3964dfaf395abef42e0048af27d3b7b3fc50b64643acf58.jpg)

![e653198aff137219c233dd08a66ebf7c999b9ca31e3c11498bd3e143d350aec8.jpg](../icml_results/1734_SARL_%20Structure-Aligned%20Reinforcement%20Learning%20for%20Bridging%20the%20Perception-Action%20Gap%20in%20Airspace/tables/e653198aff137219c233dd08a66ebf7c999b9ca31e3c11498bd3e143d350aec8.jpg)

<a id="paper-1712"></a>

## CSG: Cognitive Structure Generation for Intelligent Education


### Images

![26f58fddc98ddf6639d60d4fd8bb42697cd1d833b2ea2f235d2822d0c9498b81.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/images/26f58fddc98ddf6639d60d4fd8bb42697cd1d833b2ea2f235d2822d0c9498b81.jpg)

![7a89bf6472fa0579a96440d4916a51174e13de7ab107a28f38f512a6d18bf37e.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/images/7a89bf6472fa0579a96440d4916a51174e13de7ab107a28f38f512a6d18bf37e.jpg)

![c52c8e91d0af70b361eba60b3e818831bc40c2f597ba500c63ffc230c7266ed1.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/images/c52c8e91d0af70b361eba60b3e818831bc40c2f597ba500c63ffc230c7266ed1.jpg)

### Tables

![2274799cd1b2638dfed66510cc3095fd9dcd84eababdad6e1e9ae37ec4870dc1.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/2274799cd1b2638dfed66510cc3095fd9dcd84eababdad6e1e9ae37ec4870dc1.jpg)

![4db406c76d0c3e552d1e40fcf1dd99f3beb33a2ac90531b20be50a25cdcb5783.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/4db406c76d0c3e552d1e40fcf1dd99f3beb33a2ac90531b20be50a25cdcb5783.jpg)

![5515103042bac5463853bf64c54cc98ea138634254cae270b112d3f5cf8a35fa.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/5515103042bac5463853bf64c54cc98ea138634254cae270b112d3f5cf8a35fa.jpg)

![6bf80e83ea25ff6d420aa54df158d9b0c6d8099f3927093d90c9d910a91b81b6.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/6bf80e83ea25ff6d420aa54df158d9b0c6d8099f3927093d90c9d910a91b81b6.jpg)

![7d90e72cefbec1633fbc0c4fdfd57cad27635b38395d99af19b1fb641dc9b4a4.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/7d90e72cefbec1633fbc0c4fdfd57cad27635b38395d99af19b1fb641dc9b4a4.jpg)

![932bec95c98d965edee713dbab30fab8980ba0aae55d5e24a4ee8f3e396360d0.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/932bec95c98d965edee713dbab30fab8980ba0aae55d5e24a4ee8f3e396360d0.jpg)

![a3cdc725b88acb3681fba8ba34c7af6b10abfef4c986d5687431aa8442022c6d.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/a3cdc725b88acb3681fba8ba34c7af6b10abfef4c986d5687431aa8442022c6d.jpg)

![b24f74215f2c0db8c5edb66e63a36f44a8090c2158362aa50716a45a62c2e56c.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/b24f74215f2c0db8c5edb66e63a36f44a8090c2158362aa50716a45a62c2e56c.jpg)

![bf3baccfe473af4b042bfa051c4d357fd90e8f46265381820be8aa097434018f.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/bf3baccfe473af4b042bfa051c4d357fd90e8f46265381820be8aa097434018f.jpg)

![d785df0ddb252f12f46ffe2bdd7c3c7269f14f15cb6905b1b6b54e223f2e18d0.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/d785df0ddb252f12f46ffe2bdd7c3c7269f14f15cb6905b1b6b54e223f2e18d0.jpg)

![e0a3bf20287a64b609749865caa0413871bb821deb38dbf2e30d488d9ab7d127.jpg](../icml_results/1735_CSG_%20Cognitive%20Structure%20Generation%20for%20Intelligent%20Education/tables/e0a3bf20287a64b609749865caa0413871bb821deb38dbf2e30d488d9ab7d127.jpg)

<a id="paper-1713"></a>

## TileQ: Efficient Low-Rank Quantization of Mixture-of-Experts with 2D Tiling


### Images

![279f7a028624664fed35aadef207c4c84c4cb3250de9d4cb4d514c8687135909.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/279f7a028624664fed35aadef207c4c84c4cb3250de9d4cb4d514c8687135909.jpg)

![385ba162f4bd79139a39b93a214c2c6859161e2979d4fc3f098c7b0b62716afa.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/385ba162f4bd79139a39b93a214c2c6859161e2979d4fc3f098c7b0b62716afa.jpg)

![55535d574c0e143f9a4de737921b1c53b968027baa99ca4274d6bbd435104b1f.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/55535d574c0e143f9a4de737921b1c53b968027baa99ca4274d6bbd435104b1f.jpg)

![8b3af019cacd55639cb658f50dec03486811ac444e7a17432b15c5a87feffaec.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/8b3af019cacd55639cb658f50dec03486811ac444e7a17432b15c5a87feffaec.jpg)

![8d1bc3c4273420d42bd3d8da8b63d84b0692ce3af3609a1c808906e3ee105160.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/8d1bc3c4273420d42bd3d8da8b63d84b0692ce3af3609a1c808906e3ee105160.jpg)

![b4b3d81bc0c6a74f064ed2ab8ccd1b97361f91434db30f52b1f17bd3ec290b59.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/b4b3d81bc0c6a74f064ed2ab8ccd1b97361f91434db30f52b1f17bd3ec290b59.jpg)

![cd7778136a503826de98823848c9a91c9541f044d60df2c9484229ed5716f117.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/images/cd7778136a503826de98823848c9a91c9541f044d60df2c9484229ed5716f117.jpg)

### Tables

![20cfecb046fd960691a72bc30bead4c34ce5ccea908171133db6a9e7115627de.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/20cfecb046fd960691a72bc30bead4c34ce5ccea908171133db6a9e7115627de.jpg)

![2d507db5cafbaf9aa04a72b758d9250261c7eef1e2fee7792e2e0dfd062dc0bb.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/2d507db5cafbaf9aa04a72b758d9250261c7eef1e2fee7792e2e0dfd062dc0bb.jpg)

![349c1cfec305e24d94bbaccade8a40083e9ea529f4244f70a2ff873a88fd364d.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/349c1cfec305e24d94bbaccade8a40083e9ea529f4244f70a2ff873a88fd364d.jpg)

![6f0c4e5adcd5e5db894e92fa6296e40f6e7bf685651e81bb1c6e2dd581fd612f.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/6f0c4e5adcd5e5db894e92fa6296e40f6e7bf685651e81bb1c6e2dd581fd612f.jpg)

![73d153ee6abdcb58a91150c3088c0887bffea25c6e62495a969dbf93d11ab2d9.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/73d153ee6abdcb58a91150c3088c0887bffea25c6e62495a969dbf93d11ab2d9.jpg)

![9f0c3ed26da4368d7f2f37dc4f6e441164e8fbba10bd67b23e9fc3894c3cc2eb.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/9f0c3ed26da4368d7f2f37dc4f6e441164e8fbba10bd67b23e9fc3894c3cc2eb.jpg)

![c38b8f8f60d21478c0662ed6a8db7d7cd2aadc2a8ac9eb07cd30c1b71be23082.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/c38b8f8f60d21478c0662ed6a8db7d7cd2aadc2a8ac9eb07cd30c1b71be23082.jpg)

![d42cc30d16468d92350993940b47729991b99afbcee07ab4d31f684d521f1518.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/d42cc30d16468d92350993940b47729991b99afbcee07ab4d31f684d521f1518.jpg)

![d736023ce542191c46657eab9f620798645b113968d3468c800dd4f3dd2be144.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/d736023ce542191c46657eab9f620798645b113968d3468c800dd4f3dd2be144.jpg)

![e7ea2030140e8cc0ad0fcaaa8409f18212ebbe168c3d9e3d625b9e4201919df5.jpg](../icml_results/1736_TileQ_%20Efficient%20Low-Rank%20Quantization%20of%20Mixture-of-Experts%20with%202D%20Tiling/tables/e7ea2030140e8cc0ad0fcaaa8409f18212ebbe168c3d9e3d625b9e4201919df5.jpg)

<a id="paper-1714"></a>

## Temporal Weighted Encoding: Towards Maximal-Capacity Spike Coding for ANN–SNN Conversion


### Images

![0e2212c908faec1954674c0ca7f23b68eaad51896ccd13a06ecefffad7f3842a.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/images/0e2212c908faec1954674c0ca7f23b68eaad51896ccd13a06ecefffad7f3842a.jpg)

![1c07775b8b3bdf957fb4a0cb1d5176b412a367368998e495a7f83dd5578799c5.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/images/1c07775b8b3bdf957fb4a0cb1d5176b412a367368998e495a7f83dd5578799c5.jpg)

![a4257d6cd8d95d9a62d9bf934ab08b6fac9d28bea736e3518aeaea1985ef434e.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/images/a4257d6cd8d95d9a62d9bf934ab08b6fac9d28bea736e3518aeaea1985ef434e.jpg)

![ae4d52744cca3fc980744a3fc2560e5c789459115c4ae3de38b0fa37ec2d0252.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/images/ae4d52744cca3fc980744a3fc2560e5c789459115c4ae3de38b0fa37ec2d0252.jpg)

### Tables

![00ce1260dfcad5192cb3a4c4b945ccf79f39a33457cf4606d09e80f461f2c4a9.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/tables/00ce1260dfcad5192cb3a4c4b945ccf79f39a33457cf4606d09e80f461f2c4a9.jpg)

![6f94b312700d076fc3d30292ba87666966666fdb553f25a31f4e8a6028d65673.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/tables/6f94b312700d076fc3d30292ba87666966666fdb553f25a31f4e8a6028d65673.jpg)

![cbeaf8ced49932c09c6a37576a53b53aaed7ebc4361c88b7f00d3f6272baeb0f.jpg](../icml_results/1737_Temporal%20Weighted%20Encoding_%20Towards%20Maximal-Capacity%20Spike%20Coding%20for%20ANN%E2%80%93SNN%20Conversion/tables/cbeaf8ced49932c09c6a37576a53b53aaed7ebc4361c88b7f00d3f6272baeb0f.jpg)

<a id="paper-1715"></a>

## Mirror Mean-Field Langevin Dynamics


### Images

![29dc31ef3b7555a294a5f6a0e65278cbbc2b7189ef33d900dda8c1cba81041c8.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/29dc31ef3b7555a294a5f6a0e65278cbbc2b7189ef33d900dda8c1cba81041c8.jpg)

![706ded7eda7e4922ce0ae521ad7bd9d8423e87698ce86625af3998cdcf61b705.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/706ded7eda7e4922ce0ae521ad7bd9d8423e87698ce86625af3998cdcf61b705.jpg)

![74ea1da96a22309a253a6f339c11b519ccdaeda42673d504615d795bbeb6638d.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/74ea1da96a22309a253a6f339c11b519ccdaeda42673d504615d795bbeb6638d.jpg)

![91bb53f37f63ef529cec22c2521906ec08f205a5fb5a45ca47d1e55e06d5155d.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/91bb53f37f63ef529cec22c2521906ec08f205a5fb5a45ca47d1e55e06d5155d.jpg)

![c1b2f36319a2126b21119d6adc25e259ec7a3b53828b3718ba92e8d9e900eeb7.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/c1b2f36319a2126b21119d6adc25e259ec7a3b53828b3718ba92e8d9e900eeb7.jpg)

![cdbc28a07c75cf05962b6cce7a8797d0851c497722ea0f068ec92357640c6131.jpg](../icml_results/1738_Mirror%20Mean-Field%20Langevin%20Dynamics/images/cdbc28a07c75cf05962b6cce7a8797d0851c497722ea0f068ec92357640c6131.jpg)

<a id="paper-1716"></a>

## Mitigating Noise-Induced Layout Priors for Object Counting in Diffusion Models

### Images

![0e7148cba1b9fb2c2162cccf4aa0f73f93bf483542a377166f7761c8afbb48f9.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/0e7148cba1b9fb2c2162cccf4aa0f73f93bf483542a377166f7761c8afbb48f9.jpg)

![138bcb699deac7896f0b81632e4a69cc311f4f3e5becce9d40ca107a72501f07.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/138bcb699deac7896f0b81632e4a69cc311f4f3e5becce9d40ca107a72501f07.jpg)

![21cf5e13fa4cd466182c8fca9cecb1af793d4f80b7aa10b5bd6805dcfdf23d48.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/21cf5e13fa4cd466182c8fca9cecb1af793d4f80b7aa10b5bd6805dcfdf23d48.jpg)

![27b9803c7420eaafe7529ce9edcf31693dcf8dfb6037302c7a080066eff27491.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/27b9803c7420eaafe7529ce9edcf31693dcf8dfb6037302c7a080066eff27491.jpg)

![2873525bd7bd2269a8e2c494952ed20b9bc33a178b0a052d12e4278e39c4e3d9.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/2873525bd7bd2269a8e2c494952ed20b9bc33a178b0a052d12e4278e39c4e3d9.jpg)

![3b693cb7c8c428c840b4fd4cadc9e527cfe0cb5937d44a3986a56db301c7f635.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3b693cb7c8c428c840b4fd4cadc9e527cfe0cb5937d44a3986a56db301c7f635.jpg)

![3bda2920eb1b057d2ecd9290d3d00cbb0fa9af4cbce699df94277eafa8fbecfd.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3bda2920eb1b057d2ecd9290d3d00cbb0fa9af4cbce699df94277eafa8fbecfd.jpg)

![3f97e3cd77075d5cc98f7c67be7afdccfff98cbea3c551af27f650fd1a584ff4.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/3f97e3cd77075d5cc98f7c67be7afdccfff98cbea3c551af27f650fd1a584ff4.jpg)

![4561c3256f84092370873ef9ca890cd2b2ad439462d8f9e7cabc36c2a014f86f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/4561c3256f84092370873ef9ca890cd2b2ad439462d8f9e7cabc36c2a014f86f.jpg)

![46103f2b18cc03ec3cfb5399116dc44cc27fd188524e7e19a98f573c7e6a1463.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/46103f2b18cc03ec3cfb5399116dc44cc27fd188524e7e19a98f573c7e6a1463.jpg)

![46b382c0382ad9092f8365274e97ba481eaee17b3ff4fd95edb5b48cb1c65f43.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/46b382c0382ad9092f8365274e97ba481eaee17b3ff4fd95edb5b48cb1c65f43.jpg)

![4a26e8ba41ce2b8a94422d74c64b2e733c1082d8442ebd2851bed3fbc850698a.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/4a26e8ba41ce2b8a94422d74c64b2e733c1082d8442ebd2851bed3fbc850698a.jpg)

![5ee11afe7f8ea56cc532c01c4e89b3b2edd004745c59b10770d48654e639dfe6.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/5ee11afe7f8ea56cc532c01c4e89b3b2edd004745c59b10770d48654e639dfe6.jpg)

![6aa7f49d23d36f39d7b5238c2b8c6621cc85a64473a47e89d951f67d99c5965c.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/6aa7f49d23d36f39d7b5238c2b8c6621cc85a64473a47e89d951f67d99c5965c.jpg)

![6d4f1065b924cc4cbd2a2a8409d308c790fa5f7eaef3bb5c9aff59e2300a5ad3.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/6d4f1065b924cc4cbd2a2a8409d308c790fa5f7eaef3bb5c9aff59e2300a5ad3.jpg)

![7033f3e5a33549ded085222de0ee69c51c77dcea640b324e29e57895aceb509f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/7033f3e5a33549ded085222de0ee69c51c77dcea640b324e29e57895aceb509f.jpg)

![74329c67abcbe51b40bbd28519f43a645b1a153050312806ef290d33a906b925.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/74329c67abcbe51b40bbd28519f43a645b1a153050312806ef290d33a906b925.jpg)

![86166e645d317f1af27338aedb13691d0e3481598e3eaff474a02094b3874798.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/86166e645d317f1af27338aedb13691d0e3481598e3eaff474a02094b3874798.jpg)

![8de738c3c9c4548fdffa7a8ccd1389c49b26f8794cae6d9a5a6b178f3e594903.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/8de738c3c9c4548fdffa7a8ccd1389c49b26f8794cae6d9a5a6b178f3e594903.jpg)

![9ac2297cd9404cc8bc65d0a583fb132246cfd829887e9fe899f222eb694725b2.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/9ac2297cd9404cc8bc65d0a583fb132246cfd829887e9fe899f222eb694725b2.jpg)

![d99bc5178d867a9a61f4872f87ec9fb8e81b2ced2cfa74330cd9c28f582ad619.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/d99bc5178d867a9a61f4872f87ec9fb8e81b2ced2cfa74330cd9c28f582ad619.jpg)

![da47a2df3188bafa0f088b8e5e748f27daecfeff36cb546aa57a0305d3942578.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/da47a2df3188bafa0f088b8e5e748f27daecfeff36cb546aa57a0305d3942578.jpg)

![e825c89627f28a065e521efe4e4d89822fb90d445dcf29f31c6b7ac31a0d727c.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/e825c89627f28a065e521efe4e4d89822fb90d445dcf29f31c6b7ac31a0d727c.jpg)

![e9d3a1dd31c8dc304508ae5b9fa02723d066e7c482b11a6bc478b2be830b9535.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/e9d3a1dd31c8dc304508ae5b9fa02723d066e7c482b11a6bc478b2be830b9535.jpg)

![ecd0bda487e7bacab67a619252418bb0282ff169c29fa1ded0fe45c7c1380870.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/ecd0bda487e7bacab67a619252418bb0282ff169c29fa1ded0fe45c7c1380870.jpg)

![f138ce2c54f45a6f1025630648324c56d95057b47324072e22a669ba4743a2e5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/f138ce2c54f45a6f1025630648324c56d95057b47324072e22a669ba4743a2e5.jpg)

![f9fc6e9f685d3cc36ea2b7217d44d8f91c56c70bc0396020770398c0818cc9a8.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/images/f9fc6e9f685d3cc36ea2b7217d44d8f91c56c70bc0396020770398c0818cc9a8.jpg)

### Tables

![11c9958d7b860e8175e1e72d8fb17dad7f77c80df5a2fe508c584e8304a6faf8.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/11c9958d7b860e8175e1e72d8fb17dad7f77c80df5a2fe508c584e8304a6faf8.jpg)

![223e6dbaac59547bd47688420a115d176f3723eefa5631f49ff4d7340c279240.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/223e6dbaac59547bd47688420a115d176f3723eefa5631f49ff4d7340c279240.jpg)

![2c95c72f2688f99a4363b730169fc5578535663aa60543e02abcb2a7770df73f.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/2c95c72f2688f99a4363b730169fc5578535663aa60543e02abcb2a7770df73f.jpg)

![6f9eb91ead2c6b55cabf27677fb126c5473628665a51491e5efb39bf05118ce5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/6f9eb91ead2c6b55cabf27677fb126c5473628665a51491e5efb39bf05118ce5.jpg)

![7dd4501cc0a172db1ed23f18a5714426a5b3a30aee11601a34f9a6a9fae62e70.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/7dd4501cc0a172db1ed23f18a5714426a5b3a30aee11601a34f9a6a9fae62e70.jpg)

![8b0e17e50d186264328389e9d33c19fa02d3e7cf26e706b1eef1af65461156fe.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/8b0e17e50d186264328389e9d33c19fa02d3e7cf26e706b1eef1af65461156fe.jpg)

![a7c12ff14136830d4ab182fa1de75b805543a3d89fda00171a7155c8bcad0071.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/a7c12ff14136830d4ab182fa1de75b805543a3d89fda00171a7155c8bcad0071.jpg)

![c4fd1468606ae21e9e6630d408064f8e2a72a371f608a327f6ea4b7c977964c5.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/c4fd1468606ae21e9e6630d408064f8e2a72a371f608a327f6ea4b7c977964c5.jpg)

![df66909decbe2162881246950edbf22cf43c05888843294b1aec769e820a00c0.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/df66909decbe2162881246950edbf22cf43c05888843294b1aec769e820a00c0.jpg)

![e907786c802a0ad929d28354a2a608ea6d9313f7057e27fb343999b66be8aee6.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/e907786c802a0ad929d28354a2a608ea6d9313f7057e27fb343999b66be8aee6.jpg)

![eae5ec9b74107f9aebe21afaf543a81567d4c0277ff0182d757f0b73c214f5f1.jpg](../icml_results/1739_Mitigating%20Noise-Induced%20Layout%20Priors%20for%20Object%20Counting%20in%20Diffusion%20Models/tables/eae5ec9b74107f9aebe21afaf543a81567d4c0277ff0182d757f0b73c214f5f1.jpg)
