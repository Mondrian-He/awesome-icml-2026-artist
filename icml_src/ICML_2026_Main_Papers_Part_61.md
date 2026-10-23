# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 61 of 197

## 目录 (Table of Contents)

1. [Hierarchical Anchor Graph Learning for Multi-View Clustering](#paper-1981)
2. [LightningRL: Breaking the Accuracy–Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning](#paper-1982)
3. [AURA: Visually Interpretable Affective Understanding via Robust Archetypes](#paper-1983)
4. [Fast and Accurate Causal Parallel Decoding using Jacobi Forcing](#paper-1984)
5. [EAKV: An Entropy-Driven Adaptive KV Compression Framework for Long Video Understanding](#paper-1985)
6. [RAG without Forgetting: Continual Query-Infused Key Memory](#paper-1986)
7. [Censoring with Plausible Deniability: Asymmetric Local Privacy for Multi-Category CDF Estimation](#paper-1987)
8. [Universal Approximation with Softmax Attention](#paper-1988)
9. [A Consensus Anchor-guided Hypergraph Framework for Incomplete Multi-view Clustering](#paper-1989)
10. [BroRL: Scaling Reinforcement Learning via Broadened Exploration](#paper-1990)
11. [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution](#paper-1991)
12. [Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference](#paper-1992)
13. [Not All Answers Are Contextually Persuadable: Inference Dynamics in Large Language Models under Contextual Influence](#paper-1993)
14. [CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning](#paper-1994)
15. [Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning](#paper-1995)
16. [ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](#paper-1996)
17. [FedReLa: Imbalanced Federated Learning via Re-Labeling](#paper-1997)
18. [An In-Depth Study on Deep Learning Model Cloning](#paper-1998)
19. [Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models](#paper-1999)
20. [Residual Context Diffusion Language Models](#paper-2000)
21. [MoLF: Mixture-of-Latent-Flow for Pan-Cancer Spatial Gene Expression Prediction from Histology](#paper-2001)
22. [Interactive Segmentation with Elaborate Focus Prior](#paper-2002)
23. [Steering Large Language Models through the DMTA Cycle: Structure-Based Drug Design via Knowledge-Driven Bi-Level Thompson Sampling](#paper-2003)
24. [DART: Distribution-Aware Adaptive Relational Transfer for Adversarial Attacks against Closed-Source MLLMs](#paper-2004)
25. [Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](#paper-2005)
26. [EntRAG: Entity-Centric Retrieval-Augmented Generation for Knowledge-based Visual Question Answering](#paper-2006)
27. [Parameter Manifold Purification](#paper-2007)
28. [TVI-CoT: Text-Visual Interleaved Chain-of-Thought Reasoning for Multimodal Understanding](#paper-2008)
29. [Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models](#paper-2009)
30. [Near-Minimax Multi-Objective RL under Predictable Adversarial Preferences and Preference-Free Exploration in Linear MDPs](#paper-2010)
31. [Time-Consistent Robust Multi-Objective Reinforcement Learning via a Bellman–Isaacs Weight-Adversary Recursion](#paper-2011)
32. [On Structured State-Space Duality](#paper-2012)
33. [TeamWork: Multivariate Time Series Anomaly Detection via Asymmetric Role-aware Channel Modeling](#paper-2013)

---


<a id="paper-1981"></a>

## Hierarchical Anchor Graph Learning for Multi-View Clustering

### Images

![14307c0f16d4793f9cbc9df77a6636a3ddd1398ebf755427d1af97dd79d2eded.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/14307c0f16d4793f9cbc9df77a6636a3ddd1398ebf755427d1af97dd79d2eded.jpg)

![1b2bcc4eb6f4d9f6f0325f2fbcef55bc33f94275f3a30252751133863c196b95.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/1b2bcc4eb6f4d9f6f0325f2fbcef55bc33f94275f3a30252751133863c196b95.jpg)

![20a08367732f80a96f314c24df6805305cbb4dbbed609088ee572d8c3fbe5d32.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/20a08367732f80a96f314c24df6805305cbb4dbbed609088ee572d8c3fbe5d32.jpg)

![c74996980f4c69ea3ab570140a2ef3fb6a3c7581adc306d090e886d6730c93ae.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/c74996980f4c69ea3ab570140a2ef3fb6a3c7581adc306d090e886d6730c93ae.jpg)

![d599cf979125f3b55581af1441fa818691bf0fd90492b379ef2f7a9e846a9edb.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/d599cf979125f3b55581af1441fa818691bf0fd90492b379ef2f7a9e846a9edb.jpg)

![f9fff97e0415574cd0a2cb6162668a5776714ca47d061bb707f6b71246adb4f7.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/images/f9fff97e0415574cd0a2cb6162668a5776714ca47d061bb707f6b71246adb4f7.jpg)

### Tables

![116dd3c468044ff20f0cbee2f34e87d38c8075d74090a33c1c13c798f148e7bd.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/116dd3c468044ff20f0cbee2f34e87d38c8075d74090a33c1c13c798f148e7bd.jpg)

![4ab2ad87e9574a7b1d7a10c759e8842af99d084bc687eb26e2ecca10bb09c764.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/4ab2ad87e9574a7b1d7a10c759e8842af99d084bc687eb26e2ecca10bb09c764.jpg)

![4b6f744b06d584c72b858007f7d08c12856a6c56b7762a1d6e894a9d0e341459.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/4b6f744b06d584c72b858007f7d08c12856a6c56b7762a1d6e894a9d0e341459.jpg)

![725b56f6e61c76d2da66569b2cdbbef7f3dd09424e51d9f12640107e36803bb2.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/725b56f6e61c76d2da66569b2cdbbef7f3dd09424e51d9f12640107e36803bb2.jpg)

![7cb89a101c6b77530cff9381a1c60e4630bbb8e3ec49f6686b3ff793c72d968d.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/7cb89a101c6b77530cff9381a1c60e4630bbb8e3ec49f6686b3ff793c72d968d.jpg)

![aaef16bf006b41275a6240d0ec196b020befc72e94c3571d5a47d635cacfa237.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/aaef16bf006b41275a6240d0ec196b020befc72e94c3571d5a47d635cacfa237.jpg)

![e00eb32340006137366429770ba139c8331758bbb63f94060aceaa5ab8b4c623.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/e00eb32340006137366429770ba139c8331758bbb63f94060aceaa5ab8b4c623.jpg)

![f2f0ee6c1a08940500879712c54004ee9e7daea7108fda915ff7263e58230ee3.jpg](../icml_results/2008_Turning%20Adaptation%20into%20Assets_%20Cross-Domain%20Bridging%20for%20Online%20Vision-Language%20Navigation/tables/f2f0ee6c1a08940500879712c54004ee9e7daea7108fda915ff7263e58230ee3.jpg)

## Hierarchical Anchor Graph Learning for Multi-View Clustering


### Images

![061ff56a89a5008d92fcfb849e67afe285e9151256c6987da889037289b8b5db.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/061ff56a89a5008d92fcfb849e67afe285e9151256c6987da889037289b8b5db.jpg)

![58f2cd12d32a58fd465d975a1128e2fa74d3fe45ca118eec14ae70966e7839f3.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/58f2cd12d32a58fd465d975a1128e2fa74d3fe45ca118eec14ae70966e7839f3.jpg)

![5b3705d9a7f699c30dd5db7406675a018602e1c4561ec9fdae0d108fa34d62c5.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/5b3705d9a7f699c30dd5db7406675a018602e1c4561ec9fdae0d108fa34d62c5.jpg)

![68c0cb820798d80e6b7d7c7964d9ee07351e2f9edadef25a9e746959eca5c8c4.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/68c0cb820798d80e6b7d7c7964d9ee07351e2f9edadef25a9e746959eca5c8c4.jpg)

![72ae9a906c6607250b4548ff1245cf90e4e7799ad13202a1285c76d61e63daa8.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/72ae9a906c6607250b4548ff1245cf90e4e7799ad13202a1285c76d61e63daa8.jpg)

![adf706a1cef456a40abbe982e01f939ae2c6cd54f8d203b1f58a82ff6bf7c98a.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/adf706a1cef456a40abbe982e01f939ae2c6cd54f8d203b1f58a82ff6bf7c98a.jpg)

![ae6897038d088587057cb2b9f4e3f9e2dd7231067a854d3d1a2f33e933e937b0.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/ae6897038d088587057cb2b9f4e3f9e2dd7231067a854d3d1a2f33e933e937b0.jpg)

![aef0f79a129b7547f97d40d4d9949f7c4533a837caa05fd41a5b245fd5c36761.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/aef0f79a129b7547f97d40d4d9949f7c4533a837caa05fd41a5b245fd5c36761.jpg)

![db10e8b8583c2f5a9d30e12e9c0099c4cbbf59eef7a60d3c55085db5799560b7.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/images/db10e8b8583c2f5a9d30e12e9c0099c4cbbf59eef7a60d3c55085db5799560b7.jpg)

### Tables

![14bc56fd94658897f6d64e9818c24bb63e70b4e0ddeb2bc7fa4ee2b469dc2766.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/tables/14bc56fd94658897f6d64e9818c24bb63e70b4e0ddeb2bc7fa4ee2b469dc2766.jpg)

![1d19dc57f08ae434f67086bbe276e717abc2f8e77d7f9b8c45e0018b0e9b1a78.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/tables/1d19dc57f08ae434f67086bbe276e717abc2f8e77d7f9b8c45e0018b0e9b1a78.jpg)

![32889c2cbb1b09c406ed21f3d73a9db583e463a16c0a9bd1eb13aeba7f1252f1.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/tables/32889c2cbb1b09c406ed21f3d73a9db583e463a16c0a9bd1eb13aeba7f1252f1.jpg)

![d40ec1dc681dc600b9b681813e014eb4bce622b5ab7cb4d4d5c67a0c55ffeddb.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/tables/d40ec1dc681dc600b9b681813e014eb4bce622b5ab7cb4d4d5c67a0c55ffeddb.jpg)

![ee6d5a4e3dfc15cdd2900ebd7a42942b0a8d53d2f4d81fce03df9dcbab2142ea.jpg](../icml_results/2009_Hierarchical%20Anchor%20Graph%20Learning%20for%20Multi-View%20Clustering/tables/ee6d5a4e3dfc15cdd2900ebd7a42942b0a8d53d2f4d81fce03df9dcbab2142ea.jpg)

<a id="paper-1982"></a>

## LightningRL: Breaking the Accuracy–Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning


### Images

![0a6c545ca01abbeedbabbfdcc43328747302d19e9e9e88ca750a7fe91480dbdd.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/0a6c545ca01abbeedbabbfdcc43328747302d19e9e9e88ca750a7fe91480dbdd.jpg)

![1196dc504d881fe49f3ffc4b84559f4d7d92a82d2a1d4784b03f62a46601c1b5.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/1196dc504d881fe49f3ffc4b84559f4d7d92a82d2a1d4784b03f62a46601c1b5.jpg)

![4229ab0db36670172720c98ddc0f457e53317997c8fcdf42b047d9fe38e6afcf.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/4229ab0db36670172720c98ddc0f457e53317997c8fcdf42b047d9fe38e6afcf.jpg)

![698b61f430bb332489e37cc04b73a604eeead3c56858d78991b55ced6d57c979.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/698b61f430bb332489e37cc04b73a604eeead3c56858d78991b55ced6d57c979.jpg)

![6ab18377a103db5e475ef599f5c0432b2aca79ae8aaa2fc1a41c837ef12b6f90.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/6ab18377a103db5e475ef599f5c0432b2aca79ae8aaa2fc1a41c837ef12b6f90.jpg)

![7f5ec848adf7c41039359379ac336e39963466f2a92df9e4e63f5c82c65e070a.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/7f5ec848adf7c41039359379ac336e39963466f2a92df9e4e63f5c82c65e070a.jpg)

![c6d9a99e4e5a37abd6c7970791aee02aff81dcee5a77b7693635bc08d4b23385.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/c6d9a99e4e5a37abd6c7970791aee02aff81dcee5a77b7693635bc08d4b23385.jpg)

![ceff72e7eb692f947b151ca61a572f7c9e1c65bc22d027ffabba4ac2f3493a47.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/images/ceff72e7eb692f947b151ca61a572f7c9e1c65bc22d027ffabba4ac2f3493a47.jpg)

### Tables

![0700f7ce46cb66104f7831a71bc7b1cd5c131b1bc51e6bd8b9f589db4e7144df.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/0700f7ce46cb66104f7831a71bc7b1cd5c131b1bc51e6bd8b9f589db4e7144df.jpg)

![1881c87672301f4e1c9a3c27c41b1fd50f2dd90f8ee1a85c5ab000799c84b536.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/1881c87672301f4e1c9a3c27c41b1fd50f2dd90f8ee1a85c5ab000799c84b536.jpg)

![27a112b49d2059b75128bb971dd0a4b3f4f1ed34976ab9f36bc921bb9fe721e3.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/27a112b49d2059b75128bb971dd0a4b3f4f1ed34976ab9f36bc921bb9fe721e3.jpg)

![3b74beb7ae0d7ed76209a92d7d47fbb546d541e96fdbfbc2955b763145f7cc2c.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/3b74beb7ae0d7ed76209a92d7d47fbb546d541e96fdbfbc2955b763145f7cc2c.jpg)

![557735bb106f308cfaf147dccf4948f4818e32a03b9ae041b1f05d54684bd670.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/557735bb106f308cfaf147dccf4948f4818e32a03b9ae041b1f05d54684bd670.jpg)

![8c19a5c2cc422f9f4c0e80c2a9a3ada42d3ec8e2071b33b5410ae9294beed18c.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/8c19a5c2cc422f9f4c0e80c2a9a3ada42d3ec8e2071b33b5410ae9294beed18c.jpg)

![91f44d8426f88f2e02e64711e2b61842ca8b26348a7a080fd39bcdc701367a68.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/91f44d8426f88f2e02e64711e2b61842ca8b26348a7a080fd39bcdc701367a68.jpg)

![96a04417174b2574d76af23803bae6e66dca3929cec941cc3c420b0f6dbdd24f.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/96a04417174b2574d76af23803bae6e66dca3929cec941cc3c420b0f6dbdd24f.jpg)

![b36a7b6395fa0392e7c7a3ac2fc76e7d1527f019caa7cce2838a74590cffd808.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/b36a7b6395fa0392e7c7a3ac2fc76e7d1527f019caa7cce2838a74590cffd808.jpg)

![b6bb8fd2ffd188b897ee72bbc34e36b339f308f3fcf3e338b027ef5a6eb85a06.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/b6bb8fd2ffd188b897ee72bbc34e36b339f308f3fcf3e338b027ef5a6eb85a06.jpg)

![eca74319591800fe7d430405065629c6cb7a558dac09a446d05dbe348c63ef7e.jpg](../icml_results/2010_LightningRL_%20Breaking%20the%20Accuracy%E2%80%93Parallelism%20Trade-off%20of%20Block-wise%20dLLMs%20via%20Reinforcement%20Learn/tables/eca74319591800fe7d430405065629c6cb7a558dac09a446d05dbe348c63ef7e.jpg)

<a id="paper-1983"></a>

## AURA: Visually Interpretable Affective Understanding via Robust Archetypes


### Images

![00116ee258982a3a025f6d131685c00935e0fa69e4680749824f8a8ac64b4598.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/00116ee258982a3a025f6d131685c00935e0fa69e4680749824f8a8ac64b4598.jpg)

![09ade85a3600210c29f90bf92a08d5d7c2627ef8a821947848cbf6072429a3b7.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/09ade85a3600210c29f90bf92a08d5d7c2627ef8a821947848cbf6072429a3b7.jpg)

![12c9bcd371a3dbca028258ab0e579d85ff4b477c46b6ce619f502c7a5041d6da.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/12c9bcd371a3dbca028258ab0e579d85ff4b477c46b6ce619f502c7a5041d6da.jpg)

![24dd759bb319cd068f6c990c1d6476b6abd6bfb0084fd2ca526086d906a6ee70.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/24dd759bb319cd068f6c990c1d6476b6abd6bfb0084fd2ca526086d906a6ee70.jpg)

![7e1fa221ef55cab657232c8d71a2a4d0cbe83c314f81c874b37c990141c804d2.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/7e1fa221ef55cab657232c8d71a2a4d0cbe83c314f81c874b37c990141c804d2.jpg)

![8ab1d3d67fe2c6a58aef4680259a3c19e7a54ef445fa35f6bcdeadd80aac6176.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/8ab1d3d67fe2c6a58aef4680259a3c19e7a54ef445fa35f6bcdeadd80aac6176.jpg)

![9107e7059567602889395cf3a1f7ab929451cc658b785386124547e63bd9e641.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/9107e7059567602889395cf3a1f7ab929451cc658b785386124547e63bd9e641.jpg)

![f4d48bda80d16c948025aeb96051d2cd0ed70325b460b9acb3b0249ae3bafa01.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/images/f4d48bda80d16c948025aeb96051d2cd0ed70325b460b9acb3b0249ae3bafa01.jpg)

### Tables

![1f32f1003134736f5473f3633b23f849c66ea72b4adf726bb7b33520fb5e5646.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/1f32f1003134736f5473f3633b23f849c66ea72b4adf726bb7b33520fb5e5646.jpg)

![293b47824fc4762beb6eedb9b3bfb8b980baf3e9b88ae200ee42fe7fad548698.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/293b47824fc4762beb6eedb9b3bfb8b980baf3e9b88ae200ee42fe7fad548698.jpg)

![3d7054816eaa57402ecb33468720bddd0d58da4ae940a37c3d33cf064992c0c1.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/3d7054816eaa57402ecb33468720bddd0d58da4ae940a37c3d33cf064992c0c1.jpg)

![54a0786da0b4245305ee09e4190633d4eea9c39e6202d1f8477c62cb2736807d.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/54a0786da0b4245305ee09e4190633d4eea9c39e6202d1f8477c62cb2736807d.jpg)

![67c3125add3cb0628a073505b58c17b87380f1b1c51650eed53beb44aaee3c5e.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/67c3125add3cb0628a073505b58c17b87380f1b1c51650eed53beb44aaee3c5e.jpg)

![868f27ab0480609097ee114e2cf73cedd344b6bc27dfe6847f9f3b443071afb4.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/868f27ab0480609097ee114e2cf73cedd344b6bc27dfe6847f9f3b443071afb4.jpg)

![bc49fb9ec32b9888148732996f0c93d607a8a392aae876acd6b1502b74d4c22d.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/bc49fb9ec32b9888148732996f0c93d607a8a392aae876acd6b1502b74d4c22d.jpg)

![cbe7590a506962fb7ad33200f3eb94981e6c24e26f945485ed86d41f8fd423e8.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/cbe7590a506962fb7ad33200f3eb94981e6c24e26f945485ed86d41f8fd423e8.jpg)

![d1008ef7ed6ec0f7bfc76b95c3de0a9ec823d7a54e697a88955eee2da7351794.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/d1008ef7ed6ec0f7bfc76b95c3de0a9ec823d7a54e697a88955eee2da7351794.jpg)

![d7d6aeb1173f3617dd37c6c28d250def7f8088c626fbb7d43d4c070aa71cf24f.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/d7d6aeb1173f3617dd37c6c28d250def7f8088c626fbb7d43d4c070aa71cf24f.jpg)

![dce071b58c2bdb6867fa1c55538c605d2edfa88cfe75ec1e685bf4d90fa0836c.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/dce071b58c2bdb6867fa1c55538c605d2edfa88cfe75ec1e685bf4d90fa0836c.jpg)

![df79ad68b13c814e5afed3348a61f360c0c6cab7da2e2e730af38e745cee1542.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/df79ad68b13c814e5afed3348a61f360c0c6cab7da2e2e730af38e745cee1542.jpg)

![e83d2c35215d43716d6590da69b4838a92000db6cf86f89975e11e037cbaa50b.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/e83d2c35215d43716d6590da69b4838a92000db6cf86f89975e11e037cbaa50b.jpg)

![f639f0d7979620857a044ea6cf129eee1118b69814f1f614f2f0812b5e3b38ec.jpg](../icml_results/2011_AURA_%20Visually%20Interpretable%20Affective%20Understanding%20via%20Robust%20Archetypes/tables/f639f0d7979620857a044ea6cf129eee1118b69814f1f614f2f0812b5e3b38ec.jpg)

<a id="paper-1984"></a>

## Fast and Accurate Causal Parallel Decoding using Jacobi Forcing


### Images

![1971b921e646aaef65bf1b9509d4cad615da996dd5773dcac44d2035bbecf58a.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/1971b921e646aaef65bf1b9509d4cad615da996dd5773dcac44d2035bbecf58a.jpg)

![26632db5dae6286832041de175e4279bbbfbd08eb784c363aa633ce6d4b0ffc6.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/26632db5dae6286832041de175e4279bbbfbd08eb784c363aa633ce6d4b0ffc6.jpg)

![61efb2e78613c3d91ebb416b5ef8708abee29fa825fc835f52c14ca28e0e9f24.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/61efb2e78613c3d91ebb416b5ef8708abee29fa825fc835f52c14ca28e0e9f24.jpg)

![90c12ebd8f0095e5a9e3f798830e89474a8d58f40d4e14054a4e586a3c89b86a.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/90c12ebd8f0095e5a9e3f798830e89474a8d58f40d4e14054a4e586a3c89b86a.jpg)

![9cf5474d0b934bada3e816fd0fda5b0dc39ca7b7ed814dcf9d7cbb54bc96983f.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/9cf5474d0b934bada3e816fd0fda5b0dc39ca7b7ed814dcf9d7cbb54bc96983f.jpg)

![a4acb0d983ebe15849c2b3ac7c68ec80d237407baaefbce9b92bddac66782aa2.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/a4acb0d983ebe15849c2b3ac7c68ec80d237407baaefbce9b92bddac66782aa2.jpg)

![a5b0d9368a442b0de1a22f9fde9938f0de5bb75be3b1149558d87899370143ba.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/a5b0d9368a442b0de1a22f9fde9938f0de5bb75be3b1149558d87899370143ba.jpg)

![b7aafd36fdd15e0294b7d69bdcbce8adddecefe6aa2c3b430c220ce431075e0f.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/b7aafd36fdd15e0294b7d69bdcbce8adddecefe6aa2c3b430c220ce431075e0f.jpg)

![dcbf57dea83b27627b789896c057ab927ebbf605488a018fa5aa5571bfab53b6.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/images/dcbf57dea83b27627b789896c057ab927ebbf605488a018fa5aa5571bfab53b6.jpg)

### Tables

![0771a085e152e9b8ca46f1f953c92db504d50a8d0c9c869d2529f331e8543f10.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/0771a085e152e9b8ca46f1f953c92db504d50a8d0c9c869d2529f331e8543f10.jpg)

![09e7e3f44f8db8dc9e819c877e3fed8417f3fecc644363990b71d0e3108a4f73.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/09e7e3f44f8db8dc9e819c877e3fed8417f3fecc644363990b71d0e3108a4f73.jpg)

![5be11a6ae6fa0bc7e5b01d746016afa8cc4964b3f94bc9fba04abcf7b9534ed6.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/5be11a6ae6fa0bc7e5b01d746016afa8cc4964b3f94bc9fba04abcf7b9534ed6.jpg)

![8588369f72fdfddc6acd0c5374c58de806c34947283faed2508cf0627b716e50.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/8588369f72fdfddc6acd0c5374c58de806c34947283faed2508cf0627b716e50.jpg)

![dddd0144401e9bf3b4ee489a9d65f77c33a10e7c053be980468dfa891485a502.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/dddd0144401e9bf3b4ee489a9d65f77c33a10e7c053be980468dfa891485a502.jpg)

![e5aee040cd5276efe62e4c39a39f5077b7d8eb0a943d37485bbf7b9805b06920.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/e5aee040cd5276efe62e4c39a39f5077b7d8eb0a943d37485bbf7b9805b06920.jpg)

![eaf7f9cb1d58f0e5313f86afa0e2c58ebb8028ec9c0b158b8d0b3b271e00c3e3.jpg](../icml_results/2012_Fast%20and%20Accurate%20Causal%20Parallel%20Decoding%20using%20Jacobi%20Forcing/tables/eaf7f9cb1d58f0e5313f86afa0e2c58ebb8028ec9c0b158b8d0b3b271e00c3e3.jpg)

<a id="paper-1985"></a>

## EAKV: An Entropy-Driven Adaptive KV Compression Framework for Long Video Understanding


### Images

![2b7296cb9adb060b4ccd7f41525dfa0e69a8a092dfc8118e01b6cd7ebfddc2d8.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/images/2b7296cb9adb060b4ccd7f41525dfa0e69a8a092dfc8118e01b6cd7ebfddc2d8.jpg)

![5e1ddde4786406fbec32f00adf97bbdcffe997c369c721d22578eae60c308fd3.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/images/5e1ddde4786406fbec32f00adf97bbdcffe997c369c721d22578eae60c308fd3.jpg)

![a9fb0bfe17f65f62dd2cde798b8c44ed1df03d2ff31291d6083fdd7c3d7366c5.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/images/a9fb0bfe17f65f62dd2cde798b8c44ed1df03d2ff31291d6083fdd7c3d7366c5.jpg)

![cc5ac02a19ac0e4538c8c8fb31fa75dec17b89093bfee676c99d1916a792e3b6.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/images/cc5ac02a19ac0e4538c8c8fb31fa75dec17b89093bfee676c99d1916a792e3b6.jpg)

![e38d90188f4d28c390bb67bafa09b8622fe8bc921550aeddaac24205f579b629.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/images/e38d90188f4d28c390bb67bafa09b8622fe8bc921550aeddaac24205f579b629.jpg)

### Tables

![1057123ad5263198fb0f163c248012ed9fd829fc0ccb1c9748e35566d1ec2430.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/1057123ad5263198fb0f163c248012ed9fd829fc0ccb1c9748e35566d1ec2430.jpg)

![4b9edb9df3b1cefaf1eb45e9cce5804f6b7ce77688eb9f443f6fdc66ae15d242.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/4b9edb9df3b1cefaf1eb45e9cce5804f6b7ce77688eb9f443f6fdc66ae15d242.jpg)

![5865cb6835db353271b5e6a36042bad40bfa6e2e75a14be3f8571612065cdc8e.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/5865cb6835db353271b5e6a36042bad40bfa6e2e75a14be3f8571612065cdc8e.jpg)

![5e3d733838ddc0cd88a45406f9e457c0c075ea405e5971f2e20dc73281c720a3.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/5e3d733838ddc0cd88a45406f9e457c0c075ea405e5971f2e20dc73281c720a3.jpg)

![a9639e0889e29da31823f3656e77ba4f841c3c694c20b4badfd71ef82a81d577.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/a9639e0889e29da31823f3656e77ba4f841c3c694c20b4badfd71ef82a81d577.jpg)

![b1810edbedfa7b710ce9471ae01be6b0c69bbdfdf0e47631c4691fd22de9ec6d.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/b1810edbedfa7b710ce9471ae01be6b0c69bbdfdf0e47631c4691fd22de9ec6d.jpg)

![b9a81acd10bcf1b391b841f74c2b3811b8cf1712ab1f60ef85556fea6e146478.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/b9a81acd10bcf1b391b841f74c2b3811b8cf1712ab1f60ef85556fea6e146478.jpg)

![bfe3cdc73812118fd518d289175b73db3e3c993bf520b83484ab2952f2623bb8.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/bfe3cdc73812118fd518d289175b73db3e3c993bf520b83484ab2952f2623bb8.jpg)

![d6b1766f71231042a81a84d2aeadbca25dfa4d874e2e46c9f7315249589243c2.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/d6b1766f71231042a81a84d2aeadbca25dfa4d874e2e46c9f7315249589243c2.jpg)

![f3bf38d6f30a9b7ce473f806187d278f0053f6fed2b20464a16a2f299416744c.jpg](../icml_results/2013_EAKV_%20An%20Entropy-Driven%20Adaptive%20KV%20Compression%20Framework%20for%20Long%20Video%20Understanding/tables/f3bf38d6f30a9b7ce473f806187d278f0053f6fed2b20464a16a2f299416744c.jpg)

<a id="paper-1986"></a>

## RAG without Forgetting: Continual Query-Infused Key Memory


### Images

![26f2a55b860f8def5c0af57540eb56f8de9629e02840fe0c1ad06e9ee0e41be4.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/26f2a55b860f8def5c0af57540eb56f8de9629e02840fe0c1ad06e9ee0e41be4.jpg)

![5c990d998a05120e74b5c4823ff4e1bc1051b4043229e891ef1e5909200de7c5.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/5c990d998a05120e74b5c4823ff4e1bc1051b4043229e891ef1e5909200de7c5.jpg)

![656b963dcc3b1b152414fc3fbdd0314ca6ca602e0deaac9f45445acd7737b5b1.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/656b963dcc3b1b152414fc3fbdd0314ca6ca602e0deaac9f45445acd7737b5b1.jpg)

![8da2d58ff86f896d9d8291b9ef97dfc267a75355ac947f93c0e88900db9061a3.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/8da2d58ff86f896d9d8291b9ef97dfc267a75355ac947f93c0e88900db9061a3.jpg)

![d69b17ba183042d6b6ac4e11e84578799abdafc47de5fe2602d7c05a33a744f5.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/d69b17ba183042d6b6ac4e11e84578799abdafc47de5fe2602d7c05a33a744f5.jpg)

![e3eea7d96a578b8330fca110af92fc537333db311ad7f19a6b6c56f3fa64db9f.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/e3eea7d96a578b8330fca110af92fc537333db311ad7f19a6b6c56f3fa64db9f.jpg)

![e85f017ff04fcc232bc994dbbb17a07ee7f774f0c975adddb2154ed2ba899055.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/images/e85f017ff04fcc232bc994dbbb17a07ee7f774f0c975adddb2154ed2ba899055.jpg)

### Tables

![2b10d3e997f751b553a2f8fb6bcf4c8f3c5582bde186c1439466c1996f69a066.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/2b10d3e997f751b553a2f8fb6bcf4c8f3c5582bde186c1439466c1996f69a066.jpg)

![33820dca9a9b48c28e952a4063b9c595c968a3175b611575c3ce7ebe1a0f401d.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/33820dca9a9b48c28e952a4063b9c595c968a3175b611575c3ce7ebe1a0f401d.jpg)

![9403abda7629a885d7914ce69039f8bdab398ceeb5825a7a1150aea88a2a9439.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/9403abda7629a885d7914ce69039f8bdab398ceeb5825a7a1150aea88a2a9439.jpg)

![e9118d57709f20e0ed9250071ab3863634d3b9e8b1a853acf933feb82a00ea71.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/e9118d57709f20e0ed9250071ab3863634d3b9e8b1a853acf933feb82a00ea71.jpg)

![f87ef8dfc6f752ba481a8d257217877b8243262419fbb6b4468595d0211e6c99.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/f87ef8dfc6f752ba481a8d257217877b8243262419fbb6b4468595d0211e6c99.jpg)

![f93eb3e23c840328507f607502f6efe411a8e7a0c63f702e35fa7a98c9c47635.jpg](../icml_results/2014_RAG%20without%20Forgetting_%20Continual%20Query-Infused%20Key%20Memory/tables/f93eb3e23c840328507f607502f6efe411a8e7a0c63f702e35fa7a98c9c47635.jpg)

<a id="paper-1987"></a>

## Censoring with Plausible Deniability: Asymmetric Local Privacy for Multi-Category CDF Estimation


### Images

![32c4dc9d8c89a9415c8c2e715cbe4fa8ae7f3a33131e3ecc86e13009c93ddfae.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/images/32c4dc9d8c89a9415c8c2e715cbe4fa8ae7f3a33131e3ecc86e13009c93ddfae.jpg)

![5e023f6669cca573cfbe008f0bc90f4a8be31d3fe8e9efa6765339b3adb1e2b1.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/images/5e023f6669cca573cfbe008f0bc90f4a8be31d3fe8e9efa6765339b3adb1e2b1.jpg)

![cc249c912c0e7919be5716f5bc469bcd1a10fcb1259ec613f7764bdd7e74ef64.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/images/cc249c912c0e7919be5716f5bc469bcd1a10fcb1259ec613f7764bdd7e74ef64.jpg)

![d7ca39ceb13bd9d188126b721c69a2d5658aa635268451a1533322d709d277d0.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/images/d7ca39ceb13bd9d188126b721c69a2d5658aa635268451a1533322d709d277d0.jpg)

### Tables

![14c23a3eab084f4b96ed3c4eb1673a2fde5a596b49fff702df1960c94c32974b.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/14c23a3eab084f4b96ed3c4eb1673a2fde5a596b49fff702df1960c94c32974b.jpg)

![6c13d4881a33d32b508c645f4613386d0c463f81ce08c10c2c8b3aee0d4dc978.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/6c13d4881a33d32b508c645f4613386d0c463f81ce08c10c2c8b3aee0d4dc978.jpg)

![957f3e0f01d1c47d9c006044a227680fa3c11fc3aa18d0e96b514866d50be70c.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/957f3e0f01d1c47d9c006044a227680fa3c11fc3aa18d0e96b514866d50be70c.jpg)

![972e05592b8c7497b0dae110586a09bed244d076ff3d57954a666f990dbf8b84.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/972e05592b8c7497b0dae110586a09bed244d076ff3d57954a666f990dbf8b84.jpg)

![9c173074ffc2045e4a717e52f07d5b56e5c8ee48fdb5774ee385a2b6cde4f1d8.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/9c173074ffc2045e4a717e52f07d5b56e5c8ee48fdb5774ee385a2b6cde4f1d8.jpg)

![b367eefde10828e25226d9e473060841d9dd9c896e867c31d7f50ce72f507a5d.jpg](../icml_results/2015_Censoring%20with%20Plausible%20Deniability_%20Asymmetric%20Local%20Privacy%20for%20Multi-Category%20CDF%20Estimation/tables/b367eefde10828e25226d9e473060841d9dd9c896e867c31d7f50ce72f507a5d.jpg)

<a id="paper-1988"></a>

## Universal Approximation with Softmax Attention


### Images

![48ad486a7da7ef637c01d0aea721ff1273d300b6d1ad7717b0085b0c0d29664b.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/images/48ad486a7da7ef637c01d0aea721ff1273d300b6d1ad7717b0085b0c0d29664b.jpg)

![93c9e0edf249194f2d0e8aca539edf0101f7d9168cde7b0e645498d2740ede66.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/images/93c9e0edf249194f2d0e8aca539edf0101f7d9168cde7b0e645498d2740ede66.jpg)

![aac61ca59944d024214635330157fe5edf928b245c461c0106df045ceae5cd97.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/images/aac61ca59944d024214635330157fe5edf928b245c461c0106df045ceae5cd97.jpg)

![e63bc72b69c69eca39c8a6123043a1126f81e34ff3c955cc840e13265118ee69.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/images/e63bc72b69c69eca39c8a6123043a1126f81e34ff3c955cc840e13265118ee69.jpg)

![f058742b201426e315d5801724db8781ae1e456ce26d7f9e9b5ee33675451449.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/images/f058742b201426e315d5801724db8781ae1e456ce26d7f9e9b5ee33675451449.jpg)

### Tables

![d7b7f408f014a7697cadcd1fdf218d13dc5648e8511ade11dce1e002914abca6.jpg](../icml_results/2016_Universal%20Approximation%20with%20Softmax%20Attention/tables/d7b7f408f014a7697cadcd1fdf218d13dc5648e8511ade11dce1e002914abca6.jpg)

<a id="paper-1989"></a>

## A Consensus Anchor-guided Hypergraph Framework for Incomplete Multi-view Clustering


### Images

![072cac4300d903eb337c70d901d04647ae619a4ee46d2e067ab9fb6ff410d0e5.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/images/072cac4300d903eb337c70d901d04647ae619a4ee46d2e067ab9fb6ff410d0e5.jpg)

![5cff55f05801b5988b524a77d3a16a6ff2fb8fbc819ee6a2cc861126e0ddda09.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/images/5cff55f05801b5988b524a77d3a16a6ff2fb8fbc819ee6a2cc861126e0ddda09.jpg)

![656d7fe623a4a9b18b00b937ae0c91a5c814dd96f1a5d84bf4ec1a06b859c3e8.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/images/656d7fe623a4a9b18b00b937ae0c91a5c814dd96f1a5d84bf4ec1a06b859c3e8.jpg)

![da867652a5e50bc4f7072c8f37b46d585966f779196c989cf737fa1ae91512ef.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/images/da867652a5e50bc4f7072c8f37b46d585966f779196c989cf737fa1ae91512ef.jpg)

### Tables

![0c20d1991b95444cbf990356f6bc4228a934c499666127f232e7307ba571a237.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/tables/0c20d1991b95444cbf990356f6bc4228a934c499666127f232e7307ba571a237.jpg)

![5a963b2f67936bf9297dc8573369351a16ddea5ff932baf2a6d7d798049f8068.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/tables/5a963b2f67936bf9297dc8573369351a16ddea5ff932baf2a6d7d798049f8068.jpg)

![5f38f5632d97fd8f17af1341d8f71da11402a2390ef1ec11b667b1139083981e.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/tables/5f38f5632d97fd8f17af1341d8f71da11402a2390ef1ec11b667b1139083981e.jpg)

![913c9bb58ff33dad84fca1ecdea85c953c969c6cb2a9ce5c16d73454825c8368.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/tables/913c9bb58ff33dad84fca1ecdea85c953c969c6cb2a9ce5c16d73454825c8368.jpg)

![c5622478e4dcacd660ee502c225973069a0fca3fa1a33c369d43f871b3057cb9.jpg](../icml_results/2017_A%20Consensus%20Anchor-guided%20Hypergraph%20Framework%20for%20Incomplete%20Multi-view%20Clustering/tables/c5622478e4dcacd660ee502c225973069a0fca3fa1a33c369d43f871b3057cb9.jpg)

<a id="paper-1990"></a>

## BroRL: Scaling Reinforcement Learning via Broadened Exploration


### Images

![0198e27c4b17bfd9d8c3ee9330938be0d4f4b81b9ebc20056aa6fa6475158b42.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/images/0198e27c4b17bfd9d8c3ee9330938be0d4f4b81b9ebc20056aa6fa6475158b42.jpg)

![59ab620498778f9d6fa9a4f6a184525f3a39a3bbeca93bf6801b9fa1e64f8ad8.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/images/59ab620498778f9d6fa9a4f6a184525f3a39a3bbeca93bf6801b9fa1e64f8ad8.jpg)

![76041221b6f3cf149050a994b874b16c1278f19cbd4f1aeff3d682e865363562.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/images/76041221b6f3cf149050a994b874b16c1278f19cbd4f1aeff3d682e865363562.jpg)

![b43f07af9a64372f5af9e9ff373b68bb92b37ec5dfa1ce53af918487f8afffb6.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/images/b43f07af9a64372f5af9e9ff373b68bb92b37ec5dfa1ce53af918487f8afffb6.jpg)

### Tables

![0f0c0f6987a86255bd1d5324491c72ebbe2afd8475cd6e63f8536e9d1e4d7dca.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/0f0c0f6987a86255bd1d5324491c72ebbe2afd8475cd6e63f8536e9d1e4d7dca.jpg)

![248cd78b26b690f3d8893f642bc46cd3026aea5214563fe65e9bdea34d63f4ac.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/248cd78b26b690f3d8893f642bc46cd3026aea5214563fe65e9bdea34d63f4ac.jpg)

![397ae72a936fa2d27dfe993e22fd453975d9972b4035d8891889a9d9bf8969a0.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/397ae72a936fa2d27dfe993e22fd453975d9972b4035d8891889a9d9bf8969a0.jpg)

![816f8ac9eefbd8a3ad34f8995325b8f8deab7f9225b1c0b9f6098f1b023bdce8.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/816f8ac9eefbd8a3ad34f8995325b8f8deab7f9225b1c0b9f6098f1b023bdce8.jpg)

![8cd24473b893c678248a4987de0c884887f9989a2aa374fb2e553aa49bde676e.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/8cd24473b893c678248a4987de0c884887f9989a2aa374fb2e553aa49bde676e.jpg)

![c11ef0ce1b431d332d4330b8e6d6bd87f227d61773cbb65e26068c7adc33a62d.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/c11ef0ce1b431d332d4330b8e6d6bd87f227d61773cbb65e26068c7adc33a62d.jpg)

![cbb12e06674c75360f794f8dc1064316ab488cbd267186064ca43491c18807ac.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/cbb12e06674c75360f794f8dc1064316ab488cbd267186064ca43491c18807ac.jpg)

![d5e75ecb6485a8d5795e2e7a95f3d93639c25c03407ecd572c09ae0e8dfeb9f6.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/d5e75ecb6485a8d5795e2e7a95f3d93639c25c03407ecd572c09ae0e8dfeb9f6.jpg)

![f168b5b8461619197f96866afe0876a1d784695e8e368f85b5f16d7f25c384f2.jpg](../icml_results/2018_BroRL_%20Scaling%20Reinforcement%20Learning%20via%20Broadened%20Exploration/tables/f168b5b8461619197f96866afe0876a1d784695e8e368f85b5f16d7f25c384f2.jpg)

<a id="paper-1991"></a>

## DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution


### Images

![705ee71a9d73c4cb86d31de1b9a2806580be6abd7725715a0d3ed69e6f07c727.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/705ee71a9d73c4cb86d31de1b9a2806580be6abd7725715a0d3ed69e6f07c727.jpg)

![7a1838b7d64994786650f694ea575433e9dab837050f4dd03ec1667e984099b6.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/7a1838b7d64994786650f694ea575433e9dab837050f4dd03ec1667e984099b6.jpg)

![9881e0f2238a4ca47b6588be851c858e42af576df77ca57df575b89f6443d220.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/9881e0f2238a4ca47b6588be851c858e42af576df77ca57df575b89f6443d220.jpg)

![9c504ebf9e001e78447ecbae39e3fc8e9914f89a3ce2746c4ceb382827201621.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/9c504ebf9e001e78447ecbae39e3fc8e9914f89a3ce2746c4ceb382827201621.jpg)

![c44796c098de20dceb2bcfcf4a6b095fba18dec6039f36c08e65e57b70073cb9.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/c44796c098de20dceb2bcfcf4a6b095fba18dec6039f36c08e65e57b70073cb9.jpg)

![d3320f69322fe73884ba4ed4654e2292c833b46cd7449c7f1e132629d6819d8a.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/d3320f69322fe73884ba4ed4654e2292c833b46cd7449c7f1e132629d6819d8a.jpg)

![fb272614e013053ee1abc0627d11de22969c8d2c99f85160eea58805040c8b51.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/images/fb272614e013053ee1abc0627d11de22969c8d2c99f85160eea58805040c8b51.jpg)

### Tables

![7d3d83b5d7084ea7598b70b2709ff3dd090b3f00928c84bd066bdb5353d5ed33.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/tables/7d3d83b5d7084ea7598b70b2709ff3dd090b3f00928c84bd066bdb5353d5ed33.jpg)

![9fdcdc67a8dd1baf2497a495dc692b2936ec3366396118c34704a0f5ec68c616.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/tables/9fdcdc67a8dd1baf2497a495dc692b2936ec3366396118c34704a0f5ec68c616.jpg)

![c75fb17c83ee74bbafd03843968d10924ed59dbae32785de5bd6a2b8e7d03051.jpg](../icml_results/2019_DREAM-R_%20Multimodal%20Speculative%20Reasoning%20with%20RL-Based%20Refined%20Drafting%2C%20Precise%20Verification%2C%20and%20/tables/c75fb17c83ee74bbafd03843968d10924ed59dbae32785de5bd6a2b8e7d03051.jpg)

<a id="paper-1992"></a>

## Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference


### Images

![41fdcbea9dd8f698e735325e3c283c77da27acb91a216996cdc750ed312eade9.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/images/41fdcbea9dd8f698e735325e3c283c77da27acb91a216996cdc750ed312eade9.jpg)

![54d78547636e4fd60dc66abed0a9e78cc9e26a75fbe743cb30238745287ad2c9.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/images/54d78547636e4fd60dc66abed0a9e78cc9e26a75fbe743cb30238745287ad2c9.jpg)

![8df7c3037f62c91425fdd69de3dd2fd417bfb72f0400d464aae7266867c1bc0e.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/images/8df7c3037f62c91425fdd69de3dd2fd417bfb72f0400d464aae7266867c1bc0e.jpg)

![a8a9e3786448ec1b8211d3000e417cb0ecd4fd81c47fd82af2977939c9ffc1b6.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/images/a8a9e3786448ec1b8211d3000e417cb0ecd4fd81c47fd82af2977939c9ffc1b6.jpg)

![d81d544b9312ed20386c20bde4126461a4f15db429bf47759f406484209aa353.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/images/d81d544b9312ed20386c20bde4126461a4f15db429bf47759f406484209aa353.jpg)

### Tables

![072556551677853c414b60e2be7b5c83285bc0cfede75536465d9de3243330c6.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/072556551677853c414b60e2be7b5c83285bc0cfede75536465d9de3243330c6.jpg)

![09e3962328357eb154a034518638d88d17414d0e73ff0d98cf264cd48e87aeeb.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/09e3962328357eb154a034518638d88d17414d0e73ff0d98cf264cd48e87aeeb.jpg)

![0c2677174c39b1ce1cc656db9e64af6647fb57289b6f45704345d189b67a184d.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/0c2677174c39b1ce1cc656db9e64af6647fb57289b6f45704345d189b67a184d.jpg)

![13a2828d344450ab52b991d04f8ac876d6caa4e1e5e56226b515d4663cc81fda.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/13a2828d344450ab52b991d04f8ac876d6caa4e1e5e56226b515d4663cc81fda.jpg)

![66825b7a74e67bb0a30046dac4aff04cd08cb2f8abd264757d32848252a4d7db.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/66825b7a74e67bb0a30046dac4aff04cd08cb2f8abd264757d32848252a4d7db.jpg)

![8e93b027b53ded104b650bc42a94460f32913a4b42b8367b85f24501c4f386bb.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/8e93b027b53ded104b650bc42a94460f32913a4b42b8367b85f24501c4f386bb.jpg)

![a1ac2ce0d5e893c5262a8816363e19ab0aaf878e6499133fa448645184d2e21b.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/a1ac2ce0d5e893c5262a8816363e19ab0aaf878e6499133fa448645184d2e21b.jpg)

![bd7b98dbd3ce2763fd1bf7c99fc4dc760b7c0459cf41a3195ef4ea662d34613a.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/bd7b98dbd3ce2763fd1bf7c99fc4dc760b7c0459cf41a3195ef4ea662d34613a.jpg)

![bd99b9fe1290a60cdf3dd3d5ad395fa170dd5d0e0d41d77c7b73922632a41466.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/bd99b9fe1290a60cdf3dd3d5ad395fa170dd5d0e0d41d77c7b73922632a41466.jpg)

![c28eb2ab27c73b17960b4449e2295b3e866115978bbf858439a83a444a10402c.jpg](../icml_results/2020_Exploring%20Accurate%20and%20Transparent%20Domain%20Adaptation%20in%20Predictive%20Healthcare%20via%20Concept-Grounded%20O/tables/c28eb2ab27c73b17960b4449e2295b3e866115978bbf858439a83a444a10402c.jpg)

<a id="paper-1993"></a>

## Not All Answers Are Contextually Persuadable: Inference Dynamics in Large Language Models under Contextual Influence


### Images

![1eebacb41f52fb8a6fd124de3a3eb12bf1c965983f3ce145ec938cd5fbf23928.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/images/1eebacb41f52fb8a6fd124de3a3eb12bf1c965983f3ce145ec938cd5fbf23928.jpg)

![48565e5871cc01c817c383e90f7078dfc9d7a21d399fb1a67a2082d9db081cf3.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/images/48565e5871cc01c817c383e90f7078dfc9d7a21d399fb1a67a2082d9db081cf3.jpg)

![4ff896f204a8e8108e6f44bf9c897429b5da5b8ec32360bffa77eb19f1d5f09d.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/images/4ff896f204a8e8108e6f44bf9c897429b5da5b8ec32360bffa77eb19f1d5f09d.jpg)

![9fea81eb7c1738ed1143e77fe9590aeca9558c3202051ee2ba32f7d6da38331b.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/images/9fea81eb7c1738ed1143e77fe9590aeca9558c3202051ee2ba32f7d6da38331b.jpg)

### Tables

![5918125001eac3e91fb6e1517367cdbb9544f1a4fb24076253d381c030ed561d.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/tables/5918125001eac3e91fb6e1517367cdbb9544f1a4fb24076253d381c030ed561d.jpg)

![5fd64be862eb57a4a8fd67602f162da55def1a8338d48d66df63028f8f066cd7.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/tables/5fd64be862eb57a4a8fd67602f162da55def1a8338d48d66df63028f8f066cd7.jpg)

![ba08680b666bc4c06f4ab179a335d68b74f8802cec0d0619b225d8015cf0f88a.jpg](../icml_results/2021_Not%20All%20Answers%20Are%20Contextually%20Persuadable_%20Inference%20Dynamics%20in%20Large%20Language%20Models%20under%20Cont/tables/ba08680b666bc4c06f4ab179a335d68b74f8802cec0d0619b225d8015cf0f88a.jpg)

<a id="paper-1994"></a>

## CocoRNA: Collective RNA Design with Cooperative Multi-agent Reinforcement Learning


### Images

![2cbd5bab1de9d1c5dc1b4b110b962b045bc68de188825155e212e91607861183.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/2cbd5bab1de9d1c5dc1b4b110b962b045bc68de188825155e212e91607861183.jpg)

![2e26d769fc5fb02fe63bd7afbcb0661dd8428b52c090cea90607945d9cfcfa77.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/2e26d769fc5fb02fe63bd7afbcb0661dd8428b52c090cea90607945d9cfcfa77.jpg)

![3690df7d8b69edcf91c2d4cc4ac644b706ef3440b8dd975cd79c4229f4e79585.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/3690df7d8b69edcf91c2d4cc4ac644b706ef3440b8dd975cd79c4229f4e79585.jpg)

![5277a61906b30065ced1b61bd12f71874bce47aa9c332678ba38a401d756797e.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/5277a61906b30065ced1b61bd12f71874bce47aa9c332678ba38a401d756797e.jpg)

![65e000e04fb2f106218db955f565def9e25bea96b9e0cf1a89cc34a17309e6c1.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/65e000e04fb2f106218db955f565def9e25bea96b9e0cf1a89cc34a17309e6c1.jpg)

![7b4baca52f3250a0a14255bfa95715f8deff4a443e99afa528924afe19eb72c0.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/7b4baca52f3250a0a14255bfa95715f8deff4a443e99afa528924afe19eb72c0.jpg)

![7d72d0f55d5fe5c43b1b60e93bdc75c8987abd06b9c295a1a0b50e0351312a63.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/7d72d0f55d5fe5c43b1b60e93bdc75c8987abd06b9c295a1a0b50e0351312a63.jpg)

![981955241330cf9cca3eecc3f5da34629eb0109d273ad037836ab99de311934b.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/981955241330cf9cca3eecc3f5da34629eb0109d273ad037836ab99de311934b.jpg)

![a34adc606da446417edc3b7cc9c1715f552fb94e99cf806d1807ee3fb87e87d9.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/a34adc606da446417edc3b7cc9c1715f552fb94e99cf806d1807ee3fb87e87d9.jpg)

![c80e9d7b1fd386f22b25717f8d57d5b07238edec9be6951173e9623c9628ac7b.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/c80e9d7b1fd386f22b25717f8d57d5b07238edec9be6951173e9623c9628ac7b.jpg)

![e68d1c97cb57cc2aed124087c2565eaf6577aa656c73e90ddf3a311bdc450b5e.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/images/e68d1c97cb57cc2aed124087c2565eaf6577aa656c73e90ddf3a311bdc450b5e.jpg)

### Tables

![195b679fe209acb338f52626229e20069582aee6b66c9df3494d79af2f4b2dcb.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/195b679fe209acb338f52626229e20069582aee6b66c9df3494d79af2f4b2dcb.jpg)

![442f1b13d24937cca99f7f03b254ad2bf0f23e8a22fb1e379a797c384ed378e9.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/442f1b13d24937cca99f7f03b254ad2bf0f23e8a22fb1e379a797c384ed378e9.jpg)

![64234c7aca5df9dc501d7b397e62975fe7694765e3b853d1411a822666eb5cbe.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/64234c7aca5df9dc501d7b397e62975fe7694765e3b853d1411a822666eb5cbe.jpg)

![6cc0b6c9e635faf6e3c8bf1d6125f64a0ea735ff23355e8b3dc8de775f3f75a7.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/6cc0b6c9e635faf6e3c8bf1d6125f64a0ea735ff23355e8b3dc8de775f3f75a7.jpg)

![6ea4b5ea787ca194205057fd3634d78c3df67f76423ea926c5b2285e20eb43ca.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/6ea4b5ea787ca194205057fd3634d78c3df67f76423ea926c5b2285e20eb43ca.jpg)

![7f5a7e3544e1ce151af4d6ba10fd5afa5b840a11bdea19ea1f7638f4eb20d8f9.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/7f5a7e3544e1ce151af4d6ba10fd5afa5b840a11bdea19ea1f7638f4eb20d8f9.jpg)

![8059a228b50b1f065699b907a53a38091c26afdbb5690d83ace10b9727babdb0.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/8059a228b50b1f065699b907a53a38091c26afdbb5690d83ace10b9727babdb0.jpg)

![8bddb2d797c5b7c044b431215930bca341c2954747e077b7cb19a8f637011a82.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/8bddb2d797c5b7c044b431215930bca341c2954747e077b7cb19a8f637011a82.jpg)

![9776b4a19e1b63ebdecc6b2f7223a34026cab9a8baa3c03a76ebf2f8082efb77.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/9776b4a19e1b63ebdecc6b2f7223a34026cab9a8baa3c03a76ebf2f8082efb77.jpg)

![b1345cbb0a0a31adb0e66f9fd12113213ab53e466cba7947d17f736a0087904f.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/b1345cbb0a0a31adb0e66f9fd12113213ab53e466cba7947d17f736a0087904f.jpg)

![d0f96303927607ecd8cc695ccd5a4ba58a35129dca67e7ce822332cb5079f467.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/d0f96303927607ecd8cc695ccd5a4ba58a35129dca67e7ce822332cb5079f467.jpg)

![e610b3e1309fe29d635eecfb4881e0f786dd420cd0ea26db075eb8a013c17f7f.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/e610b3e1309fe29d635eecfb4881e0f786dd420cd0ea26db075eb8a013c17f7f.jpg)

![f2ff6cce2ba9343d5251785e8ac93a587cff86cc36e79d7023651b07c36361b6.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/f2ff6cce2ba9343d5251785e8ac93a587cff86cc36e79d7023651b07c36361b6.jpg)

![ff46ca956b5925ee9daa2472ee4beea811f4f85d1904048dd44a61ba3a0400c5.jpg](../icml_results/2022_CocoRNA_%20Collective%20RNA%20Design%20with%20Cooperative%20Multi-agent%20Reinforcement%20Learning/tables/ff46ca956b5925ee9daa2472ee4beea811f4f85d1904048dd44a61ba3a0400c5.jpg)

<a id="paper-1995"></a>

## Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning


### Images

![04b261a108f5cb88ceb755fe5a873d4b1a715b0dfe88d592c1dd26842b149963.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/04b261a108f5cb88ceb755fe5a873d4b1a715b0dfe88d592c1dd26842b149963.jpg)

![233bff014efbd84b44dedc49c37a8ac585733b39afc720df00589d422481d338.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/233bff014efbd84b44dedc49c37a8ac585733b39afc720df00589d422481d338.jpg)

![2547ec8472f2b128d8ecff64203299d9bc626a78f243dc1b8c6ef37797890102.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/2547ec8472f2b128d8ecff64203299d9bc626a78f243dc1b8c6ef37797890102.jpg)

![2ea59f8d9870675b63a9447f713b1db49e3730b02167c4fca493eaa97405326a.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/2ea59f8d9870675b63a9447f713b1db49e3730b02167c4fca493eaa97405326a.jpg)

![48799e2b028d1fd0fb2e44491610b5e80ba9ea4780287dc946848ad240ad317d.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/48799e2b028d1fd0fb2e44491610b5e80ba9ea4780287dc946848ad240ad317d.jpg)

![5b7ae3b2b6657567e4bb322a3057bacfccd810c905c0cbe4ced269df144cd4de.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/5b7ae3b2b6657567e4bb322a3057bacfccd810c905c0cbe4ced269df144cd4de.jpg)

![5dab25a2315989d897ff71d15c45ca090f2ada5c0cdd7f2d475c48b479dea124.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/5dab25a2315989d897ff71d15c45ca090f2ada5c0cdd7f2d475c48b479dea124.jpg)

![6e7f5526b19678bac28d24caf0c3c38ed0d68ef724e66c4713384d8c4a90b5aa.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/6e7f5526b19678bac28d24caf0c3c38ed0d68ef724e66c4713384d8c4a90b5aa.jpg)

![9e0f4efaf35be6fea7e0ad38c7b2c7bc123fa16c2f99962fb9cea2b6f3539e32.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/9e0f4efaf35be6fea7e0ad38c7b2c7bc123fa16c2f99962fb9cea2b6f3539e32.jpg)

![bf3e67b96b84b30ac481ef57c7663a8e6ed530625b2103e9061240c168fe393f.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/bf3e67b96b84b30ac481ef57c7663a8e6ed530625b2103e9061240c168fe393f.jpg)

![c9a9114029f1df2fea246604c6064e3c6f7c4728f3a4bb7d7e7f1f9e8fe2ad7e.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/c9a9114029f1df2fea246604c6064e3c6f7c4728f3a4bb7d7e7f1f9e8fe2ad7e.jpg)

![d540222d4ca9f5916abd691bc144673e742af945d262f44626f0d370a97a21a5.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/d540222d4ca9f5916abd691bc144673e742af945d262f44626f0d370a97a21a5.jpg)

![f5be374a4ccf2b28912e427b18bdd509c02722fed3167c7f8bd30ab2e733ce49.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/images/f5be374a4ccf2b28912e427b18bdd509c02722fed3167c7f8bd30ab2e733ce49.jpg)

### Tables

![081ed594b237853f11d7351fe86885b5001ef65f9b12e56e2da03b104e26ea40.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/081ed594b237853f11d7351fe86885b5001ef65f9b12e56e2da03b104e26ea40.jpg)

![147674ca09c25dd5033f32007567cf009986005575b1cd2d9c9c4501892614d2.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/147674ca09c25dd5033f32007567cf009986005575b1cd2d9c9c4501892614d2.jpg)

![163ae90d0b08ec95acda5b2d91df142f20f7efdad226e8db5bd8397e8644c9dc.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/163ae90d0b08ec95acda5b2d91df142f20f7efdad226e8db5bd8397e8644c9dc.jpg)

![39573cd2d7f41479c7a49568a0c7c5f7c7b76ea7fba0236a87027ae74db8064d.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/39573cd2d7f41479c7a49568a0c7c5f7c7b76ea7fba0236a87027ae74db8064d.jpg)

![3eed10cef17f314588db576dd1d24b209b8b6691183f5fadcf261640674d0dbd.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/3eed10cef17f314588db576dd1d24b209b8b6691183f5fadcf261640674d0dbd.jpg)

![50053ec90c9178f11b64cd3d756d95afb9ac063f9f6f455271813cd667c5489e.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/50053ec90c9178f11b64cd3d756d95afb9ac063f9f6f455271813cd667c5489e.jpg)

![7b26e794594de174ddf864691651b9694fb979e974acdcb3f03feb99b64e5a9e.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/7b26e794594de174ddf864691651b9694fb979e974acdcb3f03feb99b64e5a9e.jpg)

![80044e6b4a13540e17a6f7bb4a5782106f5e58071a52331712d9a80057b29743.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/80044e6b4a13540e17a6f7bb4a5782106f5e58071a52331712d9a80057b29743.jpg)

![9b969d7298d5eafde37c819a7fca15946bf050c52ac9d05bcc01d558b5ac8103.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/9b969d7298d5eafde37c819a7fca15946bf050c52ac9d05bcc01d558b5ac8103.jpg)

![acde04e23130c3895738186f8787222790d14db61a06a40f587bd12856acbb6f.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/acde04e23130c3895738186f8787222790d14db61a06a40f587bd12856acbb6f.jpg)

![b40c30071f5f6c1c8b8726d702df69a6765b211f1e1362cf4d298ba18fb21e17.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/b40c30071f5f6c1c8b8726d702df69a6765b211f1e1362cf4d298ba18fb21e17.jpg)

![c4f0641d34831c9b3195d4a05439d0045f821f57eae9cdf06ccada495580c1f5.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/c4f0641d34831c9b3195d4a05439d0045f821f57eae9cdf06ccada495580c1f5.jpg)

![e6f5c096db3d1612c69b1a4c7c397428a245856ae748dead4e9ed2304ab4aabc.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/e6f5c096db3d1612c69b1a4c7c397428a245856ae748dead4e9ed2304ab4aabc.jpg)

![f2f9061c1236cc3ce9d4fa5f13963210d8af0956c310e9bd26125cf790909ee7.jpg](../icml_results/2023_Human-in-the-Loop%20Policy%20Optimization%20for%20Preference-Based%20Multi-Objective%20Reinforcement%20Learning/tables/f2f9061c1236cc3ce9d4fa5f13963210d8af0956c310e9bd26125cf790909ee7.jpg)

<a id="paper-1996"></a>

## ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control


### Images

![0c6b07c1d15862434f203d3a0030c924515c650c433a83db37ecc8c166cd4631.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/0c6b07c1d15862434f203d3a0030c924515c650c433a83db37ecc8c166cd4631.jpg)

![13d0e05b65066c6e56a77c19ceba26d611584b18010bb7e90f5f51765e43c392.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/13d0e05b65066c6e56a77c19ceba26d611584b18010bb7e90f5f51765e43c392.jpg)

![229c5b7505b4323b278f821292ca27baf9d233781f724dc6a920644df49063a4.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/229c5b7505b4323b278f821292ca27baf9d233781f724dc6a920644df49063a4.jpg)

![24701d33a73033b5dce9ae6b378e959d37495c0e0ce78b7fa70b4702abe76b51.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/24701d33a73033b5dce9ae6b378e959d37495c0e0ce78b7fa70b4702abe76b51.jpg)

![297bfbfea907a5db6b8b1f6aaaf115c65aafda239a32eb755fe2f4d6ee2be138.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/297bfbfea907a5db6b8b1f6aaaf115c65aafda239a32eb755fe2f4d6ee2be138.jpg)

![2d5d9e1be36e17328d3986f03a1d417c12977a36a7e89260a116335857662e30.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/2d5d9e1be36e17328d3986f03a1d417c12977a36a7e89260a116335857662e30.jpg)

![373c27dba872b339c3b411dd99c19f7a74ad2f068ec43f98736f6d02d42aaa8e.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/373c27dba872b339c3b411dd99c19f7a74ad2f068ec43f98736f6d02d42aaa8e.jpg)

![657a6e05778b6ed6301386dc0bb45dd758778eec61ee586165ddcde3b421f1a6.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/657a6e05778b6ed6301386dc0bb45dd758778eec61ee586165ddcde3b421f1a6.jpg)

![6c5238779128e6b9c96043e9e0e2688d22a01cab45ed1b876943d8506d670379.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/6c5238779128e6b9c96043e9e0e2688d22a01cab45ed1b876943d8506d670379.jpg)

![7240aafdfd51ca4dafcfb2aaa154e51e522b2ce5c0f485bd3e3c16fe2ee3cd0b.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/7240aafdfd51ca4dafcfb2aaa154e51e522b2ce5c0f485bd3e3c16fe2ee3cd0b.jpg)

![7d48d1d64ea4bb8d426e12849b2920255fa5a3d11ba328d845e9e65beec6af4d.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/7d48d1d64ea4bb8d426e12849b2920255fa5a3d11ba328d845e9e65beec6af4d.jpg)

![7f4d6479256ddd282ab8eea308cd3ad667b47a89c21991fee9dc0ea721d2ddbd.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/7f4d6479256ddd282ab8eea308cd3ad667b47a89c21991fee9dc0ea721d2ddbd.jpg)

![90a5a6a152af1e34fe3971df6d756eb8914913e0c08fddf6ec3c21ffeed305f0.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/90a5a6a152af1e34fe3971df6d756eb8914913e0c08fddf6ec3c21ffeed305f0.jpg)

![9979dcf48da29ff187bbcc9baa869660bacaed187d5e7f703e90947909a6e6a2.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/9979dcf48da29ff187bbcc9baa869660bacaed187d5e7f703e90947909a6e6a2.jpg)

![9a4b4cf583e1715c045e03cea074eee6859d6023e699b3acebe1f3ab4f635ae3.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/9a4b4cf583e1715c045e03cea074eee6859d6023e699b3acebe1f3ab4f635ae3.jpg)

![b50f80eee8cbe09f0636f13673ca856fbf945a7b9971ae05baf4dadfbefcb1d3.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/b50f80eee8cbe09f0636f13673ca856fbf945a7b9971ae05baf4dadfbefcb1d3.jpg)

![c94714b230abaed3904970929c91224f18f952e968f1c0e543efece839bb3ee9.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/c94714b230abaed3904970929c91224f18f952e968f1c0e543efece839bb3ee9.jpg)

![d2495624955a79b5df3ed2750704ca7fe9ba86647d8374782a1e5c03f9f60d32.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/d2495624955a79b5df3ed2750704ca7fe9ba86647d8374782a1e5c03f9f60d32.jpg)

![d9c67ee171af8b8891b72f871dc4e845ea2d804e1d9e105fff41bf2089df250c.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/d9c67ee171af8b8891b72f871dc4e845ea2d804e1d9e105fff41bf2089df250c.jpg)

![db283dd9df629b7ec713a1a037106b423dda476ec4899e125ef7cce823e171dd.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/db283dd9df629b7ec713a1a037106b423dda476ec4899e125ef7cce823e171dd.jpg)

![f48362097862a2b6a2c714045d0225b7db8b645ab0559d5c51e44a068be61507.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/images/f48362097862a2b6a2c714045d0225b7db8b645ab0559d5c51e44a068be61507.jpg)

### Tables

![abbc9fec62a5af9d992862773c38acf4eac9a538c960036fdeda8079fb033ec5.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/tables/abbc9fec62a5af9d992862773c38acf4eac9a538c960036fdeda8079fb033ec5.jpg)

![d4b2cb3499b774d33d30d46e9d49cbc07adf4491fa671ef0d71047d106aacc59.jpg](../icml_results/2024_ACTG-ARL_%20Differentially%20Private%20Conditional%20Text%20Generation%20with%20RL-Boosted%20Control/tables/d4b2cb3499b774d33d30d46e9d49cbc07adf4491fa671ef0d71047d106aacc59.jpg)

<a id="paper-1997"></a>

## FedReLa: Imbalanced Federated Learning via Re-Labeling


### Images

![7e1a8f084a444fd15440ae25f492638b9b45e8320cb76f52e95739532ec1bdab.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/images/7e1a8f084a444fd15440ae25f492638b9b45e8320cb76f52e95739532ec1bdab.jpg)

![931475c07e0d2d9d269239d235eff5ba5245dc0869f1099705e4ad2ab0032f8a.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/images/931475c07e0d2d9d269239d235eff5ba5245dc0869f1099705e4ad2ab0032f8a.jpg)

![c3568d7950c9980c172d015875c2fd539e44d0063df51248c67ad58f4a489ce3.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/images/c3568d7950c9980c172d015875c2fd539e44d0063df51248c67ad58f4a489ce3.jpg)

### Tables

![29af113d69dc39ab2de020672ddcdf157a172e2777e7c42e80feddc5d6657a4c.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/29af113d69dc39ab2de020672ddcdf157a172e2777e7c42e80feddc5d6657a4c.jpg)

![4ecb9485e21a74c096d010ed8fc3ed9d8ae4793fcb37f08e3600965be858eee1.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/4ecb9485e21a74c096d010ed8fc3ed9d8ae4793fcb37f08e3600965be858eee1.jpg)

![59e11d15dd959fb95dbf560266c3694e6a2aa630db43f5f47177a79156af3533.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/59e11d15dd959fb95dbf560266c3694e6a2aa630db43f5f47177a79156af3533.jpg)

![627b40323d1a63757c7539ca5f274bd3b928eee59f836220314908da2ce9b035.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/627b40323d1a63757c7539ca5f274bd3b928eee59f836220314908da2ce9b035.jpg)

![68642b7e6d59f7c3de0ab92128f4cee7296c10e6ccfead42545a39d448abcf91.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/68642b7e6d59f7c3de0ab92128f4cee7296c10e6ccfead42545a39d448abcf91.jpg)

![7392baa5b6f7159e88ac0a5780f108f67d9f6eaa7d4eb4a4bae85dead3563e43.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/7392baa5b6f7159e88ac0a5780f108f67d9f6eaa7d4eb4a4bae85dead3563e43.jpg)

![a0940de2ee20f7f641d28325bd0f97b688700a53fa7ee4f7c8b54ef1194e6192.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/a0940de2ee20f7f641d28325bd0f97b688700a53fa7ee4f7c8b54ef1194e6192.jpg)

![a6415d269c12f9fb316aa749178349688e644bf65a9c94c2877f48284f969c00.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/a6415d269c12f9fb316aa749178349688e644bf65a9c94c2877f48284f969c00.jpg)

![a9ea084812d6175d85f0b13a483bc9364434cd0f2501e816316f0111233a4d29.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/a9ea084812d6175d85f0b13a483bc9364434cd0f2501e816316f0111233a4d29.jpg)

![e2808052ee24360368e503b5f412949889b4741bd7d3aa0c968c18c1d7487286.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/e2808052ee24360368e503b5f412949889b4741bd7d3aa0c968c18c1d7487286.jpg)

![ebe4464b0c7cc50ebf71342873222bdf91f33cb1286ae2beb1cf75020c93fb5a.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/ebe4464b0c7cc50ebf71342873222bdf91f33cb1286ae2beb1cf75020c93fb5a.jpg)

![f08ef1115b540aa2e2e9aa6bd9c07fe2212dc008716c8ccbbf63d2ee886c4647.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/f08ef1115b540aa2e2e9aa6bd9c07fe2212dc008716c8ccbbf63d2ee886c4647.jpg)

![f98f18bb8a0c30e6fc629abd360db312fe288746f6ff5ce8fb9e1e33c783cb41.jpg](../icml_results/2025_FedReLa_%20Imbalanced%20Federated%20Learning%20via%20Re-Labeling/tables/f98f18bb8a0c30e6fc629abd360db312fe288746f6ff5ce8fb9e1e33c783cb41.jpg)

<a id="paper-1998"></a>

## An In-Depth Study on Deep Learning Model Cloning


### Images

![07f943934b6466f02d433637fbe97b8cbe7e7d23059fa38ca7e477131eefcc1f.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/07f943934b6466f02d433637fbe97b8cbe7e7d23059fa38ca7e477131eefcc1f.jpg)

![0d12116846a08d8b8d354066b493caf477c9513ccc2ddfb25a80b6778ba076c6.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/0d12116846a08d8b8d354066b493caf477c9513ccc2ddfb25a80b6778ba076c6.jpg)

![8a99f7b3c40289be421087e6bcbc460fa07d14508a88f3bd3db15b7cd3cc173a.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/8a99f7b3c40289be421087e6bcbc460fa07d14508a88f3bd3db15b7cd3cc173a.jpg)

![ba73dd98d083c75f72480257e8cc8e5fd6d16c0a64048eabd797eecb064dcf92.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/ba73dd98d083c75f72480257e8cc8e5fd6d16c0a64048eabd797eecb064dcf92.jpg)

![c67db6824af340a26b4101656c14c4b26d67a30d83c25e2fcf5c3c8c76679f28.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/c67db6824af340a26b4101656c14c4b26d67a30d83c25e2fcf5c3c8c76679f28.jpg)

![d48c00194e28c0aea0bf96a550895f15fc711a4911ecfa2dadb579a3e6a014e5.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/d48c00194e28c0aea0bf96a550895f15fc711a4911ecfa2dadb579a3e6a014e5.jpg)

![d6ff2b2f006a9a378665fe59fd0bed6c642a10c44a974c0e93f69b55325d6593.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/d6ff2b2f006a9a378665fe59fd0bed6c642a10c44a974c0e93f69b55325d6593.jpg)

![f55e775b81d08d5eec8a2eaaedc16858b0985e4733cc6ad3763747f840f2fff2.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/images/f55e775b81d08d5eec8a2eaaedc16858b0985e4733cc6ad3763747f840f2fff2.jpg)

### Tables

![065cd5016f4ed0bd813647c0e508e6dec05eb22ebb16831a4b853f9358079f90.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/tables/065cd5016f4ed0bd813647c0e508e6dec05eb22ebb16831a4b853f9358079f90.jpg)

![3394e564e823a9e032f8339e5295b82caaf9ba592bc7f2c1183ae2d390ba0385.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/tables/3394e564e823a9e032f8339e5295b82caaf9ba592bc7f2c1183ae2d390ba0385.jpg)

![3de36ee757ee6b1593156affe8e717bfe9e4ab678543c792a71c83f0fa9ae767.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/tables/3de36ee757ee6b1593156affe8e717bfe9e4ab678543c792a71c83f0fa9ae767.jpg)

![6543a0d500bc5163b2ae31d46f8585ebcfa41771f07e059391a8af02f1fbc004.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/tables/6543a0d500bc5163b2ae31d46f8585ebcfa41771f07e059391a8af02f1fbc004.jpg)

![8edca7edaf0e47dc8ac3adb416f14fb9adaebf26fb1153d9ba5313cb1d984be3.jpg](../icml_results/2026_An%20In-Depth%20Study%20on%20Deep%20Learning%20Model%20Cloning/tables/8edca7edaf0e47dc8ac3adb416f14fb9adaebf26fb1153d9ba5313cb1d984be3.jpg)

<a id="paper-1999"></a>

## Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models


### Images

![2d52b4aaca5586844f2a092d56856760af4780be36d3097ed3bcf17cc7cabf26.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/images/2d52b4aaca5586844f2a092d56856760af4780be36d3097ed3bcf17cc7cabf26.jpg)

### Tables

![033496177edb924d08e947629893cae7a1be6d127235f0e243e391da1ef91a82.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/033496177edb924d08e947629893cae7a1be6d127235f0e243e391da1ef91a82.jpg)

![071718e043bf9b79075ff4b9376b4a570782d8a548bff5dc0b2aa47170659cfc.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/071718e043bf9b79075ff4b9376b4a570782d8a548bff5dc0b2aa47170659cfc.jpg)

![0a5ade6a0ade979d1e5cd81f3a09c0fcb6f5ab04224bc10a96031fd52a6d74d5.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/0a5ade6a0ade979d1e5cd81f3a09c0fcb6f5ab04224bc10a96031fd52a6d74d5.jpg)

![1608a7388c4252ab6681bc6209814dff1ae2cc550853e2e5cac9c3fb8ee8f272.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/1608a7388c4252ab6681bc6209814dff1ae2cc550853e2e5cac9c3fb8ee8f272.jpg)

![36e8f3da5865991a63211668c34b0cfd146eded4a4d099b7a6a3804972722b01.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/36e8f3da5865991a63211668c34b0cfd146eded4a4d099b7a6a3804972722b01.jpg)

![50ed16e48e204e271919e17cde6ad9f6ccf7b792f977bc67d7569c52307fc022.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/50ed16e48e204e271919e17cde6ad9f6ccf7b792f977bc67d7569c52307fc022.jpg)

![7b933726d70d9228c0d534d646ae62897640b23949d9ffe358191e3f412bf0b9.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/7b933726d70d9228c0d534d646ae62897640b23949d9ffe358191e3f412bf0b9.jpg)

![95b7d30cb3b8ade3a00cb5be4204791a6811997cd12fb9c921f0c83e9241dec1.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/95b7d30cb3b8ade3a00cb5be4204791a6811997cd12fb9c921f0c83e9241dec1.jpg)

![9cc10772dd930d05a8d6d02aa5062cc9ad2c743b5fa05c3f8a4cad93eecd8666.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/9cc10772dd930d05a8d6d02aa5062cc9ad2c743b5fa05c3f8a4cad93eecd8666.jpg)

![acb8071ce651a91292e0bad235f4185cdb83264b36a059f8a008dfa6d624d3be.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/acb8071ce651a91292e0bad235f4185cdb83264b36a059f8a008dfa6d624d3be.jpg)

![adea787d8b3fa562c3e84079a4a08708fadc7cf76bfb5f5c54941e762e24637a.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/adea787d8b3fa562c3e84079a4a08708fadc7cf76bfb5f5c54941e762e24637a.jpg)

![c0d8cdac111f5ce8740957d92175f14ea01d8a4969f4e0bbabc98e316d6c774e.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/c0d8cdac111f5ce8740957d92175f14ea01d8a4969f4e0bbabc98e316d6c774e.jpg)

![ce7cfa8c5c17c774562c8a1b7f588fbf6ebe67975d8e21393eb507bd0ab77ebc.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/ce7cfa8c5c17c774562c8a1b7f588fbf6ebe67975d8e21393eb507bd0ab77ebc.jpg)

![cec417109b5473bab11e0b77f9ead931598ececd356258a366406ea117b979b8.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/cec417109b5473bab11e0b77f9ead931598ececd356258a366406ea117b979b8.jpg)

![f53d77b5902ef20eff2fc489ae4e28fcb4f1a5b02e2f9a512fdbf82c69b7e6f2.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/f53d77b5902ef20eff2fc489ae4e28fcb4f1a5b02e2f9a512fdbf82c69b7e6f2.jpg)

![f8b727cc25c8a5b95b6893653a49e04881f526c00c07621fcbb279e4cb93acfd.jpg](../icml_results/2027_Respecting%20Modality%20Gap%20in%20Post-hoc%20Out-of-distribution%20Detection%20with%20Pre-trained%20Vision-Language%20M/tables/f8b727cc25c8a5b95b6893653a49e04881f526c00c07621fcbb279e4cb93acfd.jpg)

<a id="paper-2000"></a>

## Residual Context Diffusion Language Models


### Images

![262587c658717a83ef425567ca55cef7569a16f0e1a29ec172a5145acecfb963.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/images/262587c658717a83ef425567ca55cef7569a16f0e1a29ec172a5145acecfb963.jpg)

![57dcb377f5982af6e5ab6e0462cac8c4fa04b554862bceffe8b17bd36b07b807.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/images/57dcb377f5982af6e5ab6e0462cac8c4fa04b554862bceffe8b17bd36b07b807.jpg)

![857b2e26420532e2c9c0d03179b3c923da482040755216410fb7e9b6e8c579f3.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/images/857b2e26420532e2c9c0d03179b3c923da482040755216410fb7e9b6e8c579f3.jpg)

![cd19810d46b3f61e41825fe2eee5d13e2634cd4367dab24d1280c9f23d9e1aaf.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/images/cd19810d46b3f61e41825fe2eee5d13e2634cd4367dab24d1280c9f23d9e1aaf.jpg)

![f39195de886996780b3a99b792830e345b42d2f76e42d9b924374f34e2632b11.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/images/f39195de886996780b3a99b792830e345b42d2f76e42d9b924374f34e2632b11.jpg)

### Tables

![230647114eb508bc93074923714d574a86b7a9db93089dfa0af7ba04cb76c055.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/230647114eb508bc93074923714d574a86b7a9db93089dfa0af7ba04cb76c055.jpg)

![2ea59d65b8f3c7254ab2f7b0f36bf87450dfe43701433e821398c35136c32133.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/2ea59d65b8f3c7254ab2f7b0f36bf87450dfe43701433e821398c35136c32133.jpg)

![3d261eff4187223a4beffb4c24e2b0187d538910b6def5bb2d991a59bcbb3d71.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/3d261eff4187223a4beffb4c24e2b0187d538910b6def5bb2d991a59bcbb3d71.jpg)

![6bc61d7f3b95f1a00f12559852f266695e3a5e32ebd13ffc8b6b7b7b967d633b.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/6bc61d7f3b95f1a00f12559852f266695e3a5e32ebd13ffc8b6b7b7b967d633b.jpg)

![718550231a591e0496b2385009dd479da8b7d3f83aba4731d54721130acb939f.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/718550231a591e0496b2385009dd479da8b7d3f83aba4731d54721130acb939f.jpg)

![8a0791f754c443a50042a749caa4e7618228cea25f73d12b2ce6eee7968a5b4b.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/8a0791f754c443a50042a749caa4e7618228cea25f73d12b2ce6eee7968a5b4b.jpg)

![9ea8f81960cd5d66f3153dceb496fe4f8c34c5913627b65420568930d21dd0f3.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/9ea8f81960cd5d66f3153dceb496fe4f8c34c5913627b65420568930d21dd0f3.jpg)

![a49f42c6f8b956e52c6bb631d8e38ff167c97af36cda0d896076c27668b27508.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/a49f42c6f8b956e52c6bb631d8e38ff167c97af36cda0d896076c27668b27508.jpg)

![dd41b7b3837813f5aeb01a650cb6e5756bc719032fef8fcee4a579aa06ff3f45.jpg](../icml_results/2028_Residual%20Context%20Diffusion%20Language%20Models/tables/dd41b7b3837813f5aeb01a650cb6e5756bc719032fef8fcee4a579aa06ff3f45.jpg)

<a id="paper-2001"></a>

## MoLF: Mixture-of-Latent-Flow for Pan-Cancer Spatial Gene Expression Prediction from Histology


### Images

![24901f49c549ee938ec74336e22f8d6b37ee513000c07d54c3851983a0f96e92.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/24901f49c549ee938ec74336e22f8d6b37ee513000c07d54c3851983a0f96e92.jpg)

![4804a64d9e2a40a1fbf47cd65a4083a537702781b59cc2ec6289a3cccc586c8e.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/4804a64d9e2a40a1fbf47cd65a4083a537702781b59cc2ec6289a3cccc586c8e.jpg)

![4c7444d4562149465a549096159cad1a62d36e7cc0dea83dfc9ec8d197bd376c.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/4c7444d4562149465a549096159cad1a62d36e7cc0dea83dfc9ec8d197bd376c.jpg)

![63362a247189a28c15aa1625ce1e378f8cf47783c5887ef5b1ca626454096cb2.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/63362a247189a28c15aa1625ce1e378f8cf47783c5887ef5b1ca626454096cb2.jpg)

![63ea315957d04152c5912229da8f16a5f6a6e07cfe5732aab4f2420472691c70.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/63ea315957d04152c5912229da8f16a5f6a6e07cfe5732aab4f2420472691c70.jpg)

![6d163d7e49c802e488737a8fc2a151b8d45155b05683211536ca31bc73616d70.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/6d163d7e49c802e488737a8fc2a151b8d45155b05683211536ca31bc73616d70.jpg)

![89f35b1776ee82052e017206665e3cc72587ead3e91edfc26a13ec07b58c18ea.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/89f35b1776ee82052e017206665e3cc72587ead3e91edfc26a13ec07b58c18ea.jpg)

![8f6c71e65759c6b22e924a312364090fa67ec8c888b54775c32264419e022219.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/8f6c71e65759c6b22e924a312364090fa67ec8c888b54775c32264419e022219.jpg)

![a44a520f05f3aececd91b2c7b8364269eaff3787e80f15b73b53e57f8f1e70d5.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/a44a520f05f3aececd91b2c7b8364269eaff3787e80f15b73b53e57f8f1e70d5.jpg)

![c8bfc7095d44dad9a7972923e7541bd4c6899dd0cf983ba393d851d8764a927f.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/c8bfc7095d44dad9a7972923e7541bd4c6899dd0cf983ba393d851d8764a927f.jpg)

![d916a8b11bdc32fe41d884b4539ef672e31879b2c12695b20562734e2360a744.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/d916a8b11bdc32fe41d884b4539ef672e31879b2c12695b20562734e2360a744.jpg)

![dfd05a246bb0785cea9df375f15d39e6b6f895e481c0af0cac4ce60717bb60c3.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/dfd05a246bb0785cea9df375f15d39e6b6f895e481c0af0cac4ce60717bb60c3.jpg)

![e698d5e348c3329e43f865ecbdabfbc7903f7536535e870a442b5774347b8c06.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/e698d5e348c3329e43f865ecbdabfbc7903f7536535e870a442b5774347b8c06.jpg)

![fe52b8117b488a6ee6a4dc5fa016a280c771d9b520705e84157914f8f62bc44e.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/images/fe52b8117b488a6ee6a4dc5fa016a280c771d9b520705e84157914f8f62bc44e.jpg)

### Tables

![0940f9d27cc6da16911189e34a642de2f18fe2c5fd68e10d7900994ec78219f1.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/0940f9d27cc6da16911189e34a642de2f18fe2c5fd68e10d7900994ec78219f1.jpg)

![2cfc88e65869f4e308a8efc01fa09bb32821dac52fccb24c4e940b91edd4e1de.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/2cfc88e65869f4e308a8efc01fa09bb32821dac52fccb24c4e940b91edd4e1de.jpg)

![2d0b15f6f7e481cc75b277e2694bdcbcb15d783ed611203a8b0448d77d6efa2d.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/2d0b15f6f7e481cc75b277e2694bdcbcb15d783ed611203a8b0448d77d6efa2d.jpg)

![4c831321c96eeef12d31b6fc0509f33e7b203d1e42a91aa2ce1c9a15d3217ff7.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/4c831321c96eeef12d31b6fc0509f33e7b203d1e42a91aa2ce1c9a15d3217ff7.jpg)

![61c487da17c8f86e6dbb85e4ec0b777a253dd2fc3d101e87d6f325ee1adc7a29.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/61c487da17c8f86e6dbb85e4ec0b777a253dd2fc3d101e87d6f325ee1adc7a29.jpg)

![6792a1f7f8be8443b4e3a813d12d5cc924f985840de0b9d21d9a6a8af45ea9ce.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/6792a1f7f8be8443b4e3a813d12d5cc924f985840de0b9d21d9a6a8af45ea9ce.jpg)

![6afc2d5546c472da03f4097d311b46bf392e9f29567e34985a11a9492d07d683.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/6afc2d5546c472da03f4097d311b46bf392e9f29567e34985a11a9492d07d683.jpg)

![7fa979f7283fb61ff902592f86f8b3314034b006308991283bb6dc809f81e8d5.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/7fa979f7283fb61ff902592f86f8b3314034b006308991283bb6dc809f81e8d5.jpg)

![8892f38c861304260fb14c182ac03f3741f0b0bac9884c54852662626655634d.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/8892f38c861304260fb14c182ac03f3741f0b0bac9884c54852662626655634d.jpg)

![8a5e17b7d215b7f02f28e87ea75a25547e4a08610d0df205f021ca3d1eaff568.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/8a5e17b7d215b7f02f28e87ea75a25547e4a08610d0df205f021ca3d1eaff568.jpg)

![8b70d4f321133535690b34a94ca208ee32f37b5f1b64cfca213fa65c57b37943.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/8b70d4f321133535690b34a94ca208ee32f37b5f1b64cfca213fa65c57b37943.jpg)

![8d1f887f8a8f201419f4e3c9a5dd07fb457463b83b6d7248add9087716895563.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/8d1f887f8a8f201419f4e3c9a5dd07fb457463b83b6d7248add9087716895563.jpg)

![c53c1ae1869f34749d9c75b80e1ab5670489b25fcc8a69f6c84e843f45ada22b.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/c53c1ae1869f34749d9c75b80e1ab5670489b25fcc8a69f6c84e843f45ada22b.jpg)

![cb5e7dfc552dd02ab37e92548d9266022a2854f3ebce486d694bca5e1844c2f5.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/cb5e7dfc552dd02ab37e92548d9266022a2854f3ebce486d694bca5e1844c2f5.jpg)

![d216a962b20f08164a4dba13922b52d9d8633397d03366b368e38bd20dbca8bc.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/d216a962b20f08164a4dba13922b52d9d8633397d03366b368e38bd20dbca8bc.jpg)

![dd9eebdf0569ea9ebebd6744226fc1f8b106354db9d2e66d9ce672ad79162b66.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/dd9eebdf0569ea9ebebd6744226fc1f8b106354db9d2e66d9ce672ad79162b66.jpg)

![e6a2266a32fcfa107017d321b9759e225b2e6245e5c12cd4a554f969e3920939.jpg](../icml_results/2029_MoLF_%20Mixture-of-Latent-Flow%20for%20Pan-Cancer%20Spatial%20Gene%20Expression%20Prediction%20from%20Histology/tables/e6a2266a32fcfa107017d321b9759e225b2e6245e5c12cd4a554f969e3920939.jpg)

<a id="paper-2002"></a>

## Interactive Segmentation with Elaborate Focus Prior


### Images

![19996198e9b882edfb6ca5037532326b5dcb542f7a3e5ce98356ed49f154a851.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/19996198e9b882edfb6ca5037532326b5dcb542f7a3e5ce98356ed49f154a851.jpg)

![20496851d012c1b29de093206fcfa308ca3408e1559aa91e8f6a36d73a4b5509.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/20496851d012c1b29de093206fcfa308ca3408e1559aa91e8f6a36d73a4b5509.jpg)

![39c364b91c41e943465fa445335fdd6b1850bba20075a6de3f8132415b0bdf8d.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/39c364b91c41e943465fa445335fdd6b1850bba20075a6de3f8132415b0bdf8d.jpg)

![40d338372ab6073749ada8572cdee10e072c792254fede3407b61e056cb71197.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/40d338372ab6073749ada8572cdee10e072c792254fede3407b61e056cb71197.jpg)

![8768953d19dc30107bb7d4463c8e8b15d74232b3fdbfda95b28a68b027cff5ea.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/8768953d19dc30107bb7d4463c8e8b15d74232b3fdbfda95b28a68b027cff5ea.jpg)

![c2993745fa02ef707038a0d88be8d37ec919c695faac697275c639ccef0e27b2.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/c2993745fa02ef707038a0d88be8d37ec919c695faac697275c639ccef0e27b2.jpg)

![d875683f40e27f1c481839d4a7d5f5f1ab465be6e7dd1f95466aa09864d2e6ce.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/d875683f40e27f1c481839d4a7d5f5f1ab465be6e7dd1f95466aa09864d2e6ce.jpg)

![ff62fe15c1cd24b225aabaca467e4679c84320ada559fa574b6438ecf6c210c8.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/images/ff62fe15c1cd24b225aabaca467e4679c84320ada559fa574b6438ecf6c210c8.jpg)

### Tables

![0970c06995afebcad219860d543c1e3857e4e2a5ed8d0d33779272b09131c7db.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/0970c06995afebcad219860d543c1e3857e4e2a5ed8d0d33779272b09131c7db.jpg)

![0f33c25364b188ce55a35e99ca2449f372f3a631cfb835dbcc166acb8a7b6e5d.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/0f33c25364b188ce55a35e99ca2449f372f3a631cfb835dbcc166acb8a7b6e5d.jpg)

![2f8b93ef1b88f9bca0d2e80a9fbbd50457f555627eb1d9b42351c81b8a18a8f1.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/2f8b93ef1b88f9bca0d2e80a9fbbd50457f555627eb1d9b42351c81b8a18a8f1.jpg)

![375167322385b56037750e0edcc8cb39c97090a4f0739fdefd21a58a13822cdc.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/375167322385b56037750e0edcc8cb39c97090a4f0739fdefd21a58a13822cdc.jpg)

![583c7888342d6ac8f8cf5c48b619b601b49d5f2f27d4565a321129d1d3d7bf9d.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/583c7888342d6ac8f8cf5c48b619b601b49d5f2f27d4565a321129d1d3d7bf9d.jpg)

![731ea44baf275375f8e7d24a44c0707aa2e18d97e8c2d477709b43e20898c1a0.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/731ea44baf275375f8e7d24a44c0707aa2e18d97e8c2d477709b43e20898c1a0.jpg)

![ac2975d170de5ea768c91e9b92be6e226902fde75a3a1e7719f6ceeb2e456ae4.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/ac2975d170de5ea768c91e9b92be6e226902fde75a3a1e7719f6ceeb2e456ae4.jpg)

![c987bef620ed49a72fda8485f801e39dbcf796c42eda2fd29f2a0e7db5b70d95.jpg](../icml_results/2030_Interactive%20Segmentation%20with%20Elaborate%20Focus%20Prior/tables/c987bef620ed49a72fda8485f801e39dbcf796c42eda2fd29f2a0e7db5b70d95.jpg)

<a id="paper-2003"></a>

## Steering Large Language Models through the DMTA Cycle: Structure-Based Drug Design via Knowledge-Driven Bi-Level Thompson Sampling


### Images

![1fc3a0f81d55627846ff6fb30b4d457244346f717f768240a1ee5c671253485a.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/1fc3a0f81d55627846ff6fb30b4d457244346f717f768240a1ee5c671253485a.jpg)

![23d08e3a00d71a500d9fca1ab76831db439ea4024cc1694f24595a75f69f2cb8.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/23d08e3a00d71a500d9fca1ab76831db439ea4024cc1694f24595a75f69f2cb8.jpg)

![262437fd2b3210e47eb2e34cd4773cfe6db57a69ee49c4cd87266904d026043c.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/262437fd2b3210e47eb2e34cd4773cfe6db57a69ee49c4cd87266904d026043c.jpg)

![2d9bf5bdaf33f3ef4032bf4389143a9e98dfa104d713c471e34736f6eb0c9110.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/2d9bf5bdaf33f3ef4032bf4389143a9e98dfa104d713c471e34736f6eb0c9110.jpg)

![324874feb4130257fa973500a3f40cc7acf2385e848b9e2ba9242d40b3637bd4.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/324874feb4130257fa973500a3f40cc7acf2385e848b9e2ba9242d40b3637bd4.jpg)

![35d8f96a75529cb187e27b05ab94d171df9ff10cd22f3f66e38397844affb3cf.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/35d8f96a75529cb187e27b05ab94d171df9ff10cd22f3f66e38397844affb3cf.jpg)

![35eaa22d788a104c5185b13fe335f5b33127b03d740eb69fffec70e77921342b.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/35eaa22d788a104c5185b13fe335f5b33127b03d740eb69fffec70e77921342b.jpg)

![451599cf6642d9920ce03bce5e74987c23ad73f93b758837fe140b304a664b61.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/451599cf6642d9920ce03bce5e74987c23ad73f93b758837fe140b304a664b61.jpg)

![bb17e9b4f0c2971601e819aa8cf38b0a6d61c05f9b0a83c34e729e59e9b11517.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/bb17e9b4f0c2971601e819aa8cf38b0a6d61c05f9b0a83c34e729e59e9b11517.jpg)

![c634bbc48c3066bb984085fc6d58609b12e396041e120092ee3110a9d579c87f.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/c634bbc48c3066bb984085fc6d58609b12e396041e120092ee3110a9d579c87f.jpg)

![e3460904237d0196682125d5afb59c505a4406ebe291f1ac72ee6e1f276649db.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/e3460904237d0196682125d5afb59c505a4406ebe291f1ac72ee6e1f276649db.jpg)

![f0edf980195f76c36c7165c1bb8ab9fe630b0cb96660653541acdf2035bc2ccb.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/f0edf980195f76c36c7165c1bb8ab9fe630b0cb96660653541acdf2035bc2ccb.jpg)

![f8f42193764a14e82ef43500c5b278547a8d0f71c855b585a69368cd603517ff.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/images/f8f42193764a14e82ef43500c5b278547a8d0f71c855b585a69368cd603517ff.jpg)

### Tables

![0cbf22b9528ad56fbb7b8b4f2518955fb3e9f26e7e7ff11c34157fd53c7f10a9.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/tables/0cbf22b9528ad56fbb7b8b4f2518955fb3e9f26e7e7ff11c34157fd53c7f10a9.jpg)

![4825dfa906fd3d312eddf6fa2f60c676595ca55f9a1a8cc7a6bdd447bd463504.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/tables/4825dfa906fd3d312eddf6fa2f60c676595ca55f9a1a8cc7a6bdd447bd463504.jpg)

![84807ddaf680c4294bc06793390d5c6e17c06742f04b0819b4fec07d3135343b.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/tables/84807ddaf680c4294bc06793390d5c6e17c06742f04b0819b4fec07d3135343b.jpg)

![dd6ebf1eb4e15eb2ea4f5cf1edd37f6edd6b3092808d4e1e135f4c40468a1760.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/tables/dd6ebf1eb4e15eb2ea4f5cf1edd37f6edd6b3092808d4e1e135f4c40468a1760.jpg)

![ec96b7163a7c75802991af70551086179f5e628e9982d30f5fa5472c970157a6.jpg](../icml_results/2031_Steering%20Large%20Language%20Models%20through%20the%20DMTA%20Cycle_%20Structure-Based%20Drug%20Design%20via%20Knowledge-Dri/tables/ec96b7163a7c75802991af70551086179f5e628e9982d30f5fa5472c970157a6.jpg)

<a id="paper-2004"></a>

## DART: Distribution-Aware Adaptive Relational Transfer for Adversarial Attacks against Closed-Source MLLMs


### Images

![12cf7c7decd61c32ee4a298ef7dbeecd3b0bbd33788a368f04192316a7e128aa.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/12cf7c7decd61c32ee4a298ef7dbeecd3b0bbd33788a368f04192316a7e128aa.jpg)

![169e9fa3810b435cf5817d9afa4ae26533a544894bea97358bf582e77478a6f2.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/169e9fa3810b435cf5817d9afa4ae26533a544894bea97358bf582e77478a6f2.jpg)

![1ee8cf6137615cfd9cb1d4a0d19453086f8ed3c8e54c9249c414d2db1da9acf5.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/1ee8cf6137615cfd9cb1d4a0d19453086f8ed3c8e54c9249c414d2db1da9acf5.jpg)

![4be2041c617afeba34df5c2c6b3472bb867010223971ba76a0096e343e8d27b9.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/4be2041c617afeba34df5c2c6b3472bb867010223971ba76a0096e343e8d27b9.jpg)

![aa19aeb1e89f74c5f817f1b290965230821c76a88e69c663bf4c9ae204d7e189.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/aa19aeb1e89f74c5f817f1b290965230821c76a88e69c663bf4c9ae204d7e189.jpg)

![aca96e1d1a69afec16ed55cb69dd6474f27e7ceb28211c78bb603c576aca9c54.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/aca96e1d1a69afec16ed55cb69dd6474f27e7ceb28211c78bb603c576aca9c54.jpg)

![e9dcc1fb67df5ee334084fa994774191059bd84ceeaf82132392d6ce058e7bfb.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/e9dcc1fb67df5ee334084fa994774191059bd84ceeaf82132392d6ce058e7bfb.jpg)

![f457bf41144b254dd22cd9fb05b61033142285263e447df85d84099c3fc693e5.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/f457bf41144b254dd22cd9fb05b61033142285263e447df85d84099c3fc693e5.jpg)

![fee41a36698bfab050637d4c0d64cb9a5d2b558a2a82470f5084b12aefac3047.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/images/fee41a36698bfab050637d4c0d64cb9a5d2b558a2a82470f5084b12aefac3047.jpg)

### Tables

![0576544c7f3f2163a9a4cb3a7b4ef5470a7bdbe6383b19dcd963a384ca530048.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/0576544c7f3f2163a9a4cb3a7b4ef5470a7bdbe6383b19dcd963a384ca530048.jpg)

![09aab6ec51c76772564c353e0b2c8101b45fe178b8bbc66cb7db17ab40d89275.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/09aab6ec51c76772564c353e0b2c8101b45fe178b8bbc66cb7db17ab40d89275.jpg)

![5dc29cf771cd822d183cbfc5a4c44c435fc3dac06dcb1324c64640603f1fe093.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/5dc29cf771cd822d183cbfc5a4c44c435fc3dac06dcb1324c64640603f1fe093.jpg)

![6a75ef356aa4e348221982a3033bc110ba2dd3afe9ae86ab65996254fbc436fa.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/6a75ef356aa4e348221982a3033bc110ba2dd3afe9ae86ab65996254fbc436fa.jpg)

![6cd79122a33b4f9745c37b60c1a7e3368e7c84b4b7bfc37b8fd7f2e97a80afb3.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/6cd79122a33b4f9745c37b60c1a7e3368e7c84b4b7bfc37b8fd7f2e97a80afb3.jpg)

![6dc951e9b64f496b3a1b6884da2f7415ac2e3a8f90b7b63a0bf330d230e49180.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/6dc951e9b64f496b3a1b6884da2f7415ac2e3a8f90b7b63a0bf330d230e49180.jpg)

![846b7c095ea1356db6e5635b35a91112fd6c6bed1868a21a03a30b9b3fe027a0.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/846b7c095ea1356db6e5635b35a91112fd6c6bed1868a21a03a30b9b3fe027a0.jpg)

![aec1c89136fdfe0b67056d9ccece85110c105e00aab69b7caa7127bb51417507.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/aec1c89136fdfe0b67056d9ccece85110c105e00aab69b7caa7127bb51417507.jpg)

![c3a77eb5841c9111495574cf4ac6ccd37d8615f3144466079d451f07dad4e509.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/c3a77eb5841c9111495574cf4ac6ccd37d8615f3144466079d451f07dad4e509.jpg)

![d4688022d346060acbefbe82b5e92af7e6a825fb7d458d3bdcf4c8806b1c511f.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/d4688022d346060acbefbe82b5e92af7e6a825fb7d458d3bdcf4c8806b1c511f.jpg)

![e6ef62af024a3c73c563e0c8690af09938fa659e50179dc9e50bd843e006337c.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/e6ef62af024a3c73c563e0c8690af09938fa659e50179dc9e50bd843e006337c.jpg)

![ef9c9626794ec14ae361cc56bc32b121d29a79826b141f247cddb31fe101b945.jpg](../icml_results/2032_DART_%20Distribution-Aware%20Adaptive%20Relational%20Transfer%20for%20Adversarial%20Attacks%20against%20Closed-Source%20/tables/ef9c9626794ec14ae361cc56bc32b121d29a79826b141f247cddb31fe101b945.jpg)

<a id="paper-2005"></a>

## Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection


### Images

![1b4eeb07fa03671479679bc48e56bf92fd518633ee1795b645112aadb703e607.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/1b4eeb07fa03671479679bc48e56bf92fd518633ee1795b645112aadb703e607.jpg)

![2a211a340361915ca3613b122296b0b3f6b953062e59879cd323c6b1aee2c560.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/2a211a340361915ca3613b122296b0b3f6b953062e59879cd323c6b1aee2c560.jpg)

![2eb31819b750080c9ee6b8c294860caf98008a31acf9cc396a6f698a96445f34.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/2eb31819b750080c9ee6b8c294860caf98008a31acf9cc396a6f698a96445f34.jpg)

![3eb9d6355d3145294406d3149b92cb282c50ef9ce1cecf702f6ee6db2cb608c1.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/3eb9d6355d3145294406d3149b92cb282c50ef9ce1cecf702f6ee6db2cb608c1.jpg)

![51c6bf6e8a645880bc12a67260380db29be9764d48dab87fb2a531182a4f7e17.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/51c6bf6e8a645880bc12a67260380db29be9764d48dab87fb2a531182a4f7e17.jpg)

![5264cf36ab4c4a378cfa0d04bc962148235d2997ea755249a79cc38e88621fe4.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/5264cf36ab4c4a378cfa0d04bc962148235d2997ea755249a79cc38e88621fe4.jpg)

![93b699a91594a839f65add4551b22c49ab487986ffb078308db45246939c896a.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/93b699a91594a839f65add4551b22c49ab487986ffb078308db45246939c896a.jpg)

![bc32b4cb56938903ff0039a5a22358bf2db3f0b4965a3d115c3b996e1d314dc8.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/images/bc32b4cb56938903ff0039a5a22358bf2db3f0b4965a3d115c3b996e1d314dc8.jpg)

### Tables

![224a335c76ca285d8e7cdb6f116e589556d4cb31fe967b033947395f8bc5d1e4.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/224a335c76ca285d8e7cdb6f116e589556d4cb31fe967b033947395f8bc5d1e4.jpg)

![34cece9797e97623f2ec86c1c8a226d66f4b140e1e24fa4bce5385700b15b1e2.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/34cece9797e97623f2ec86c1c8a226d66f4b140e1e24fa4bce5385700b15b1e2.jpg)

![3576d1a8fef3fae538ecb3aade18bdf534ccf1c9c62ef3efa776299a147b7ad1.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/3576d1a8fef3fae538ecb3aade18bdf534ccf1c9c62ef3efa776299a147b7ad1.jpg)

![3cb7f56c48164e6a3f0e5d28f5d545d486626e40c2dac16b144841d0b86d5afa.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/3cb7f56c48164e6a3f0e5d28f5d545d486626e40c2dac16b144841d0b86d5afa.jpg)

![55beaa57fd71be52e39a156088b565725a73b18a959c3fc8cda86ab185350372.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/55beaa57fd71be52e39a156088b565725a73b18a959c3fc8cda86ab185350372.jpg)

![5bc320ddb34b4fea2bab7840048b550d45c9e44e539924fd790c5d7ec8c2e819.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/5bc320ddb34b4fea2bab7840048b550d45c9e44e539924fd790c5d7ec8c2e819.jpg)

![aeda5eb2292223b9f930f06c5d837e3b8ad26704b48234c03d4e9fa25de9dab2.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/aeda5eb2292223b9f930f06c5d837e3b8ad26704b48234c03d4e9fa25de9dab2.jpg)

![b0b80cfb379d05aaea456af8efedba8172ab16de2e859042903fdce2aca437e8.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/b0b80cfb379d05aaea456af8efedba8172ab16de2e859042903fdce2aca437e8.jpg)

![e7ef9dbbb262f538990b8331e8b5ebe8817e56b427b64eb32bc74a1dca196686.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/e7ef9dbbb262f538990b8331e8b5ebe8817e56b427b64eb32bc74a1dca196686.jpg)

![fcd230e4349b5a9820281aa092666b34274ec31ab9c2cb82d01101b2945d4e64.jpg](../icml_results/2033_Tracing%20the%20Dynamics%20of%20Refusal_%20Exploiting%20Latent%20Refusal%20Trajectories%20for%20Robust%20Jailbreak%20Detecti/tables/fcd230e4349b5a9820281aa092666b34274ec31ab9c2cb82d01101b2945d4e64.jpg)

<a id="paper-2006"></a>

## EntRAG: Entity-Centric Retrieval-Augmented Generation for Knowledge-based Visual Question Answering


### Images

![0ad830d67a29404e99994e6274f04aa445b8742b1316940e6d6c8cb61146ce1e.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/0ad830d67a29404e99994e6274f04aa445b8742b1316940e6d6c8cb61146ce1e.jpg)

![26a60ea0cca0e23ce341265df57148753d20bfdfeb8109b6c6699c76c51171cf.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/26a60ea0cca0e23ce341265df57148753d20bfdfeb8109b6c6699c76c51171cf.jpg)

![57128ca5041eee1140784f64c667a5ccf870eeb49025b8797726b12d03fec511.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/57128ca5041eee1140784f64c667a5ccf870eeb49025b8797726b12d03fec511.jpg)

![87a60e121fef0bc178f4a4fe48b2989dc3d9b77f84436037d072fc535764d3c5.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/87a60e121fef0bc178f4a4fe48b2989dc3d9b77f84436037d072fc535764d3c5.jpg)

![c4129625677abae377a1322bb1128966e93b07c5976d4e631f50c943b0eff192.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/c4129625677abae377a1322bb1128966e93b07c5976d4e631f50c943b0eff192.jpg)

![c47297ed3074c1c259f2c45380e2846c7fb4017efca28ff617c4a51cea37fcc7.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/c47297ed3074c1c259f2c45380e2846c7fb4017efca28ff617c4a51cea37fcc7.jpg)

![c51994c5ff3dceb7cdec95c519deb6e17acd85ba2cccd5311353d91ec3ea5de8.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/c51994c5ff3dceb7cdec95c519deb6e17acd85ba2cccd5311353d91ec3ea5de8.jpg)

![ca2c1e714f897b146f716d4271524bc75faeb2dc2cebdd7aa16502e862ab712b.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/ca2c1e714f897b146f716d4271524bc75faeb2dc2cebdd7aa16502e862ab712b.jpg)

![dfc0e1ae0cf63f198344b992625667bc24847a48f2f011ee620893f5685367f7.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/dfc0e1ae0cf63f198344b992625667bc24847a48f2f011ee620893f5685367f7.jpg)

![e01a9d1887517f3549c0b527760eec5b54b30bb98a648efda3b93f2454ea4bcf.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/e01a9d1887517f3549c0b527760eec5b54b30bb98a648efda3b93f2454ea4bcf.jpg)

![fda5d784f5dfc7202338418c3c7b24aa00e4de98c77e5af14ad61004f9fd7860.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/images/fda5d784f5dfc7202338418c3c7b24aa00e4de98c77e5af14ad61004f9fd7860.jpg)

### Tables

![0641f1eac2b20c9c2e77dc271f10ad9f3080c289a25ab27f9b61cfe4fb9e6f1b.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/0641f1eac2b20c9c2e77dc271f10ad9f3080c289a25ab27f9b61cfe4fb9e6f1b.jpg)

![0a44df47e2a161087a9aaf9f191ce638983b07ecab101854651c9cb8a8239187.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/0a44df47e2a161087a9aaf9f191ce638983b07ecab101854651c9cb8a8239187.jpg)

![10dacf2d2926e6ca7675fb55bb78174e09c7ca4543063da4907c8c7e70ec6634.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/10dacf2d2926e6ca7675fb55bb78174e09c7ca4543063da4907c8c7e70ec6634.jpg)

![2ab7397888733a456bc019e29a901af4a39fc1056f949dca68dc47217488912b.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/2ab7397888733a456bc019e29a901af4a39fc1056f949dca68dc47217488912b.jpg)

![577c4020608a392e65e64592aba4cf775dc995b91990946f6835c58401b96a64.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/577c4020608a392e65e64592aba4cf775dc995b91990946f6835c58401b96a64.jpg)

![7a4f99bf6faf19fd4b7f5961326a76a4368cb4b11baff6ac076c5c4d542c2196.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/7a4f99bf6faf19fd4b7f5961326a76a4368cb4b11baff6ac076c5c4d542c2196.jpg)

![82fb3c0b2cf573e399cd5099a69de63e770bb26ca0ac4e926399cb6dc10a405f.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/82fb3c0b2cf573e399cd5099a69de63e770bb26ca0ac4e926399cb6dc10a405f.jpg)

![8504cd5e5eef39dc6f19862cbd4cf9523d06e66a4a450fb5ffae514113678e02.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/8504cd5e5eef39dc6f19862cbd4cf9523d06e66a4a450fb5ffae514113678e02.jpg)

![abc2519604263fbfaff6d8097b70ee1fbcfea0d21eafd5f8e6dc7a20aa6b55c7.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/abc2519604263fbfaff6d8097b70ee1fbcfea0d21eafd5f8e6dc7a20aa6b55c7.jpg)

![ae0474c7cfacf9091adfeebecd41e2bf8529c54eac897f4daf8b2e07bf362012.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/ae0474c7cfacf9091adfeebecd41e2bf8529c54eac897f4daf8b2e07bf362012.jpg)

![d962f75ec5322ab31ba2fa1477e3e4c4363db98bdfb83c602836e139fa5e1473.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/d962f75ec5322ab31ba2fa1477e3e4c4363db98bdfb83c602836e139fa5e1473.jpg)

![f66c7942937ed1e8535df8aa93b07da88c12bec62775a7e07789ddee7e2af6e9.jpg](../icml_results/2034_EntRAG_%20Entity-Centric%20Retrieval-Augmented%20Generation%20for%20Knowledge-based%20Visual%20Question%20Answering/tables/f66c7942937ed1e8535df8aa93b07da88c12bec62775a7e07789ddee7e2af6e9.jpg)

<a id="paper-2007"></a>

## Parameter Manifold Purification


### Images

![493ba1b5b5d4250df607882bc04568e19bb0528bc9dac65943c099efa121284a.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/493ba1b5b5d4250df607882bc04568e19bb0528bc9dac65943c099efa121284a.jpg)

![54599fe764c67d724e5d4281575ca1d4601fe25a9628742c32d3cb0968feb39a.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/54599fe764c67d724e5d4281575ca1d4601fe25a9628742c32d3cb0968feb39a.jpg)

![704e4b3824b02b51c059a3c215d817c2e9386774b325dfab910d13ae0b46d530.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/704e4b3824b02b51c059a3c215d817c2e9386774b325dfab910d13ae0b46d530.jpg)

![8cef7e131dab4ac6b9a25b920118bb71c619cfec59042d5555e9d6183a3c26c9.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/8cef7e131dab4ac6b9a25b920118bb71c619cfec59042d5555e9d6183a3c26c9.jpg)

![9011bd53339084ed26fd0cfeecd7642896e9434e3d7db829922ffcc54c4128e0.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/9011bd53339084ed26fd0cfeecd7642896e9434e3d7db829922ffcc54c4128e0.jpg)

![95600956568d95e08123c560570676d8c4f15c5ad096bc186f17fee521703cec.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/95600956568d95e08123c560570676d8c4f15c5ad096bc186f17fee521703cec.jpg)

![9ad1f800ac23b6112f3c1054066e4a15ff64af7a89838a25d6917b8db82c83b8.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/images/9ad1f800ac23b6112f3c1054066e4a15ff64af7a89838a25d6917b8db82c83b8.jpg)

### Tables

![210566ab519623f3bdb747f8096d22e214d8e862f333456a7eb25558c5881230.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/210566ab519623f3bdb747f8096d22e214d8e862f333456a7eb25558c5881230.jpg)

![246ed0058d00d0107669e3ae24d2e181e525db3ccc2e2869f177cdb3b6eae215.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/246ed0058d00d0107669e3ae24d2e181e525db3ccc2e2869f177cdb3b6eae215.jpg)

![277cde5727591b5037c1ad93e9b10e1b6c038def98984cf077bb9ca31d687ba5.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/277cde5727591b5037c1ad93e9b10e1b6c038def98984cf077bb9ca31d687ba5.jpg)

![4cfe831d88a8c89aa106899c221de8f3d8e6a7fbfd45c148f14ffb092ab2e22d.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/4cfe831d88a8c89aa106899c221de8f3d8e6a7fbfd45c148f14ffb092ab2e22d.jpg)

![685064a4c7b8f7a1e5b54621205fff56369d906a570d31469c40b4eba4ff429a.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/685064a4c7b8f7a1e5b54621205fff56369d906a570d31469c40b4eba4ff429a.jpg)

![75df135543fdabd4494d03ae4879e0f27ce5c4443dcdcb5caa32da2396768d3e.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/75df135543fdabd4494d03ae4879e0f27ce5c4443dcdcb5caa32da2396768d3e.jpg)

![761d59694b05632f230f7beef0483444208aad19206b28eb5ac8e864301cdb28.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/761d59694b05632f230f7beef0483444208aad19206b28eb5ac8e864301cdb28.jpg)

![8a25247e8c165937752d4434b44334c7e8e0d386a70ae22da2cbc7af9321cfcd.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/8a25247e8c165937752d4434b44334c7e8e0d386a70ae22da2cbc7af9321cfcd.jpg)

![8a9cf6e62e6b0b3ae4405ce1ebfb2b4822a528287e0b9faad2b06add9edcebc0.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/8a9cf6e62e6b0b3ae4405ce1ebfb2b4822a528287e0b9faad2b06add9edcebc0.jpg)

![9627df7d49596bd6427306a9d0c3b108ee0cdd093b0cb8ef4d2918f2cf668a78.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/9627df7d49596bd6427306a9d0c3b108ee0cdd093b0cb8ef4d2918f2cf668a78.jpg)

![b6f273fffff40b8a73a995669bb1412186af3e044c90e8fdb6d62474829693df.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/b6f273fffff40b8a73a995669bb1412186af3e044c90e8fdb6d62474829693df.jpg)

![e7b2d0530a679c6cd1f42a8697b9287ed45c3e0ff408fe21884556fe83a44f10.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/e7b2d0530a679c6cd1f42a8697b9287ed45c3e0ff408fe21884556fe83a44f10.jpg)

![ebaec544edfce0eeebb35f8e4ab8a8ae381647bcef511eb7750d64498b42ce65.jpg](../icml_results/2035_Parameter%20Manifold%20Purification/tables/ebaec544edfce0eeebb35f8e4ab8a8ae381647bcef511eb7750d64498b42ce65.jpg)

<a id="paper-2008"></a>

## TVI-CoT: Text-Visual Interleaved Chain-of-Thought Reasoning for Multimodal Understanding


### Images

![69b11c88913d5d8bba6b436e27b9c5174c55c50fe06ca159d98e62e66ce66d17.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/images/69b11c88913d5d8bba6b436e27b9c5174c55c50fe06ca159d98e62e66ce66d17.jpg)

![e6ba2c5663608636e0da73dfe858d9405c223ed2c46d549c6c96137cc36551d8.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/images/e6ba2c5663608636e0da73dfe858d9405c223ed2c46d549c6c96137cc36551d8.jpg)

![ec76d8805371341564ebb94b6d91d29129a79ad6f24871f0b0709143286562a7.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/images/ec76d8805371341564ebb94b6d91d29129a79ad6f24871f0b0709143286562a7.jpg)

### Tables

![0bffc3529adf2c61d3636b744025519a906935f576b29720d26682ab89db1b70.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/0bffc3529adf2c61d3636b744025519a906935f576b29720d26682ab89db1b70.jpg)

![1d707c9c928db67dcb9dd336d307d671ae8fc05415c9594490f63a37d7abd9ad.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/1d707c9c928db67dcb9dd336d307d671ae8fc05415c9594490f63a37d7abd9ad.jpg)

![3d552becad197b0a8ec67541565a5c9221ca5f709ef1715731fa6137d8e093da.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/3d552becad197b0a8ec67541565a5c9221ca5f709ef1715731fa6137d8e093da.jpg)

![93644f2a26980e718f025b98e016cdd8fd1a181f44b64555c7cbe227d71c2022.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/93644f2a26980e718f025b98e016cdd8fd1a181f44b64555c7cbe227d71c2022.jpg)

![a81032d0ec8e8a49fa46d3731c02502d5be481faf7fd86e1f564bd136b0743e2.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/a81032d0ec8e8a49fa46d3731c02502d5be481faf7fd86e1f564bd136b0743e2.jpg)

![b2be46b3320c02d2d97f920f95054fc1ca413cd388e8ce7feebd0f636c1eaab7.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/b2be46b3320c02d2d97f920f95054fc1ca413cd388e8ce7feebd0f636c1eaab7.jpg)

![cb14b746fa20d2b8812a9364795904264b045f64b1b0a28f6a5dfd31606f2f00.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/cb14b746fa20d2b8812a9364795904264b045f64b1b0a28f6a5dfd31606f2f00.jpg)

![ec01479bc64b2add9bfdad65703ce129b9207bb665eb15a65bf72baaec1d5b15.jpg](../icml_results/2036_TVI-CoT_%20Text-Visual%20Interleaved%20Chain-of-Thought%20Reasoning%20for%20Multimodal%20Understanding/tables/ec01479bc64b2add9bfdad65703ce129b9207bb665eb15a65bf72baaec1d5b15.jpg)

<a id="paper-2009"></a>

## Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models


### Images

![091793cd0aa98b557d7a3c3fed075e74f068c85406931453973863b625f9f024.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/091793cd0aa98b557d7a3c3fed075e74f068c85406931453973863b625f9f024.jpg)

![152e8cc51bb953ce37a7e07a61ffc814a3c1dd205476519684bf6f61e3850680.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/152e8cc51bb953ce37a7e07a61ffc814a3c1dd205476519684bf6f61e3850680.jpg)

![3bb2ae62898e98888eeb28504ce34b1d7cee838a6bad4379a8ed0f460b543475.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/3bb2ae62898e98888eeb28504ce34b1d7cee838a6bad4379a8ed0f460b543475.jpg)

![65c6e0e13caf78274ac9ab2c05031a9c3c6b9f84f7fcb82e454cc40aa0527dc2.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/65c6e0e13caf78274ac9ab2c05031a9c3c6b9f84f7fcb82e454cc40aa0527dc2.jpg)

![86e6d2268dd9b8c803afa5ac1945c1a2388b05cf639c36e43b1d6626b8f2660a.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/86e6d2268dd9b8c803afa5ac1945c1a2388b05cf639c36e43b1d6626b8f2660a.jpg)

![8a10e13e297c3d467179b29e7d65a356eb169b6cfe69a4efad4e4984eafdba20.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/8a10e13e297c3d467179b29e7d65a356eb169b6cfe69a4efad4e4984eafdba20.jpg)

![92d07ef03afe0b4d07861c941a05e5d529ba88eee4fd75f0827567583c11fd79.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/92d07ef03afe0b4d07861c941a05e5d529ba88eee4fd75f0827567583c11fd79.jpg)

![9f191ad153d85eb12acfb087a2521108fcd94f42b6c1b47cc0d58ad46d4be5fd.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/9f191ad153d85eb12acfb087a2521108fcd94f42b6c1b47cc0d58ad46d4be5fd.jpg)

![d3d290a8e5c64caf13029ee90fd290c1bf781cd2ce826bc858ad21d97bae7887.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/d3d290a8e5c64caf13029ee90fd290c1bf781cd2ce826bc858ad21d97bae7887.jpg)

![dadad5c5b31583cd0638992598f12483c490899ffd3c1c6e9ddcb7cc58f2e837.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/dadad5c5b31583cd0638992598f12483c490899ffd3c1c6e9ddcb7cc58f2e837.jpg)

![df4ce9960b750da0a7bbc7cb2289edef5e710cfbc1873b7e3ac2ab4e90c766ed.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/df4ce9960b750da0a7bbc7cb2289edef5e710cfbc1873b7e3ac2ab4e90c766ed.jpg)

![f6df768353bccf20c0239d61b75c264bc021a02a9ed092bcc51f3981382bfeea.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/images/f6df768353bccf20c0239d61b75c264bc021a02a9ed092bcc51f3981382bfeea.jpg)

### Tables

![0e43bbfd351c08c2eca23160816638412dafee157cfed9afe2dde7ec1d5684e1.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/0e43bbfd351c08c2eca23160816638412dafee157cfed9afe2dde7ec1d5684e1.jpg)

![2a8c4287873434e12a946676861805e3d4c199800e559219f84f01bade6d971b.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/2a8c4287873434e12a946676861805e3d4c199800e559219f84f01bade6d971b.jpg)

![2e60351ae5407c879c6e8d6a9f13c11881de2222a8a4b3b670ba9eae7e1b9a23.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/2e60351ae5407c879c6e8d6a9f13c11881de2222a8a4b3b670ba9eae7e1b9a23.jpg)

![2fa97a30ed2f0a8aaa6d2bb1864b8420bd3a9d419514c74ceb42c624a10163fb.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/2fa97a30ed2f0a8aaa6d2bb1864b8420bd3a9d419514c74ceb42c624a10163fb.jpg)

![3ba30d2ccf419587e3cacad81b375a80055fba72577470fb07b1e3ac60591068.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/3ba30d2ccf419587e3cacad81b375a80055fba72577470fb07b1e3ac60591068.jpg)

![43bb4739511525574e2331888a70a75bda0141b3fd8f1ff17cddf9f78ab3a721.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/43bb4739511525574e2331888a70a75bda0141b3fd8f1ff17cddf9f78ab3a721.jpg)

![54aa305912bf23472937138581e2890068d851d3b7bc93dc8640ccf10970ae11.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/54aa305912bf23472937138581e2890068d851d3b7bc93dc8640ccf10970ae11.jpg)

![58ef4a3d3550fcdc14b798184a262e2d3550c484e925988fb4f77949f9ede672.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/58ef4a3d3550fcdc14b798184a262e2d3550c484e925988fb4f77949f9ede672.jpg)

![5f6b44a819d710778d0da64a1532cce2d173333ab3e9f5d8ca5232554aeea008.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/5f6b44a819d710778d0da64a1532cce2d173333ab3e9f5d8ca5232554aeea008.jpg)

![706514dec60328e0bec78d5aee6893dcc9de4f2e9f448c31624542be04c4339e.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/706514dec60328e0bec78d5aee6893dcc9de4f2e9f448c31624542be04c4339e.jpg)

![8c6c36f6dcc41b81efe170dbf8a9cc31715943ece4bff5786317d392e8af907e.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/8c6c36f6dcc41b81efe170dbf8a9cc31715943ece4bff5786317d392e8af907e.jpg)

![c261b43cb0ed176c101efe677723f2252e8d9bc86626fcfcd02ae183eceb8ba8.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/c261b43cb0ed176c101efe677723f2252e8d9bc86626fcfcd02ae183eceb8ba8.jpg)

![cf90cbba34b8b20dfd41e8009fa61803b42a05a487a913fc759c6d90c82201de.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/cf90cbba34b8b20dfd41e8009fa61803b42a05a487a913fc759c6d90c82201de.jpg)

![cfbd669c92fe1e8f9ec6160aae41b0836bcc17f4dc4bb0744cc72418c09c5f86.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/cfbd669c92fe1e8f9ec6160aae41b0836bcc17f4dc4bb0744cc72418c09c5f86.jpg)

![d854c543261fdb74fff8c62b2d087d9b34bf0707f043f90b172ecb793a40bc28.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/d854c543261fdb74fff8c62b2d087d9b34bf0707f043f90b172ecb793a40bc28.jpg)

![dc4ab0fb082a552f746a79562c1a4b9336471b915a4b6a3d4792beb6cbb919c5.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/dc4ab0fb082a552f746a79562c1a4b9336471b915a4b6a3d4792beb6cbb919c5.jpg)

![e2ea481eaa88955c112e37f05fad36998e019f6403599ae23932b51d2792b4b5.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/e2ea481eaa88955c112e37f05fad36998e019f6403599ae23932b51d2792b4b5.jpg)

![ed4c204a5e8dca330c337bb350e0b687d27eb4b84b10e43ec600ffcbd638168b.jpg](../icml_results/2037_Real-Time%20Monitoring%20and%20Calibration%20of%20Chain-of-Thought%20Sycophancy%20in%20Large%20Reasoning%20Models/tables/ed4c204a5e8dca330c337bb350e0b687d27eb4b84b10e43ec600ffcbd638168b.jpg)

<a id="paper-2010"></a>

## Near-Minimax Multi-Objective RL under Predictable Adversarial Preferences and Preference-Free Exploration in Linear MDPs


### Images

![17247abd1d0f800f12b33481a39f1fcc508120f84c807a5495d9cc3857dec108.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/17247abd1d0f800f12b33481a39f1fcc508120f84c807a5495d9cc3857dec108.jpg)

![1e02d9857c20e1a0847d37797f2028a45a576c7eb04b56c4608fadedda66c968.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/1e02d9857c20e1a0847d37797f2028a45a576c7eb04b56c4608fadedda66c968.jpg)

![1fb7a6634174e0e5ae8f26d4e6d8a3a3d7cd8e177846f19216b1af4ca90a30d3.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/1fb7a6634174e0e5ae8f26d4e6d8a3a3d7cd8e177846f19216b1af4ca90a30d3.jpg)

![22192098ed2597080dba2b9791a501d59c7db8079ffdf1e5919197921f63f1a9.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/22192098ed2597080dba2b9791a501d59c7db8079ffdf1e5919197921f63f1a9.jpg)

![2863c0691f73294422dee7d84a72748192e64be582fa6f11fd69b4fdeb68bf47.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/2863c0691f73294422dee7d84a72748192e64be582fa6f11fd69b4fdeb68bf47.jpg)

![34159e41ae9deb783d14a238bb374e598bae1160371c48a7574d2753ea58b263.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/34159e41ae9deb783d14a238bb374e598bae1160371c48a7574d2753ea58b263.jpg)

![4a6b35fef87e2fa079041ac557d70dd4adf41577a762c01b1fefe40dd0ee1b94.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/4a6b35fef87e2fa079041ac557d70dd4adf41577a762c01b1fefe40dd0ee1b94.jpg)

![4c83fb82a2d7f5625f6c10ec000c6af8c93e6cc20678740e012224ccae146daf.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/4c83fb82a2d7f5625f6c10ec000c6af8c93e6cc20678740e012224ccae146daf.jpg)

![5863430bbc8f231b957dda98ccfe275ccdc8867be8b3322b2fd7f78c7ffc2448.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/5863430bbc8f231b957dda98ccfe275ccdc8867be8b3322b2fd7f78c7ffc2448.jpg)

![59c47df3624f66e333fbff54913bc91f7b0338bd8bb7fcc025a52256435b0c7d.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/59c47df3624f66e333fbff54913bc91f7b0338bd8bb7fcc025a52256435b0c7d.jpg)

![621c8a67c2f2ab4ab39f768d42c90f0db619061edf2470075df64610ea20eed7.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/621c8a67c2f2ab4ab39f768d42c90f0db619061edf2470075df64610ea20eed7.jpg)

![6d78f7a9d0fd560b14f6c58d377ffaa4456e25c869ba5002415fa7a9ddb0fd1e.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/6d78f7a9d0fd560b14f6c58d377ffaa4456e25c869ba5002415fa7a9ddb0fd1e.jpg)

![6de65a130018236fed1cabcf2dc8762931e965098ea5b500e428b43511d35923.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/6de65a130018236fed1cabcf2dc8762931e965098ea5b500e428b43511d35923.jpg)

![a5995b958f3780f3bc438099d59ed40cf47c78f4db9ae731ab2a89f19d1d02a4.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/a5995b958f3780f3bc438099d59ed40cf47c78f4db9ae731ab2a89f19d1d02a4.jpg)

![b31d923988f8444097234ea81bb2dc293d677d1c934f4b099a0f88a7f23256d3.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/b31d923988f8444097234ea81bb2dc293d677d1c934f4b099a0f88a7f23256d3.jpg)

![b3553518281c7641ef9fa1c10eb6d79420c1555efc4957c432a5f7ad9a780333.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/b3553518281c7641ef9fa1c10eb6d79420c1555efc4957c432a5f7ad9a780333.jpg)

![dc13c106e4d44ebaa0a0d9bd46afb662e3a03d623a181bc7c8fc6d82ea454bbc.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/images/dc13c106e4d44ebaa0a0d9bd46afb662e3a03d623a181bc7c8fc6d82ea454bbc.jpg)

### Tables

![0c90a6c5be4a5935c3ceb4725efff62ee646522e8629dea0acb632cc65883d72.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/tables/0c90a6c5be4a5935c3ceb4725efff62ee646522e8629dea0acb632cc65883d72.jpg)

![0e4944682441c7764466ed4315d2f9dc537bc0cd34fc990f23107c2c07fd781c.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/tables/0e4944682441c7764466ed4315d2f9dc537bc0cd34fc990f23107c2c07fd781c.jpg)

![cdd75c6ee9b2b58f8a7b5db7bbf01fb8d21c060246c217f8362eb8a126ff6089.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/tables/cdd75c6ee9b2b58f8a7b5db7bbf01fb8d21c060246c217f8362eb8a126ff6089.jpg)

![e15631ee47ea0a3c502a9ed800f8a1cc72ad0e230b2a86c85a6d3c62f2d89e98.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/tables/e15631ee47ea0a3c502a9ed800f8a1cc72ad0e230b2a86c85a6d3c62f2d89e98.jpg)

![f5fe69d7ec6bd8f2ac9ac99db437c607b9e883e042bd3226bded5a0d851369cc.jpg](../icml_results/2038_Near-Minimax%20Multi-Objective%20RL%20under%20Predictable%20Adversarial%20Preferences%20and%20Preference-Free%20Explor/tables/f5fe69d7ec6bd8f2ac9ac99db437c607b9e883e042bd3226bded5a0d851369cc.jpg)

<a id="paper-2011"></a>

## Time-Consistent Robust Multi-Objective Reinforcement Learning via a Bellman–Isaacs Weight-Adversary Recursion


### Images

![6ee314f6c95df74630d465660a8c639fd28124a03fcb79e03c5685adc2539300.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/images/6ee314f6c95df74630d465660a8c639fd28124a03fcb79e03c5685adc2539300.jpg)

![72594c5fa8f9974810bb5ff95209d2a39bbde1a27b71b404184f362ef34aa1da.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/images/72594c5fa8f9974810bb5ff95209d2a39bbde1a27b71b404184f362ef34aa1da.jpg)

![83edca1395f3424645a816f49ecabbacdb1b0141c1155d88f03481b7045e6745.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/images/83edca1395f3424645a816f49ecabbacdb1b0141c1155d88f03481b7045e6745.jpg)

![d2a691ede5402caffb4820e9bf2e3529ca21c94a6783aa6105c3483084ec573f.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/images/d2a691ede5402caffb4820e9bf2e3529ca21c94a6783aa6105c3483084ec573f.jpg)

### Tables

![23be9f5c803b58931f0261ef0ade6c45f755bcbc54fd88bd0b34e05f1abb029d.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/23be9f5c803b58931f0261ef0ade6c45f755bcbc54fd88bd0b34e05f1abb029d.jpg)

![2be04a66f63d878867d2379ec4a11c1861262af251b683800ee2467a9adde4da.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/2be04a66f63d878867d2379ec4a11c1861262af251b683800ee2467a9adde4da.jpg)

![4140944196d95a0e35f64385693ce1a94c64370b68212dcabeaafbbd5c87675f.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/4140944196d95a0e35f64385693ce1a94c64370b68212dcabeaafbbd5c87675f.jpg)

![db2b3d09a0494436350cfe38c693d6287b5c2dfc31cd8cd5a0fbe0c3b7067dbf.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/db2b3d09a0494436350cfe38c693d6287b5c2dfc31cd8cd5a0fbe0c3b7067dbf.jpg)

![e6afbac34eb7569cbb27d7d5e129ae1e93ecb4aa56481fa5befc45a9ba1e1618.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/e6afbac34eb7569cbb27d7d5e129ae1e93ecb4aa56481fa5befc45a9ba1e1618.jpg)

![e8255731854d56bb689bd7e165a4a5e89fe209d4f978efb8e0d4f0b7cc96798f.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/e8255731854d56bb689bd7e165a4a5e89fe209d4f978efb8e0d4f0b7cc96798f.jpg)

![ead23d01fdb4208a1895c04b24205db8b19f3fff0d8ad0795cca8dc1be30c712.jpg](../icml_results/2039_Time-Consistent%20Robust%20Multi-Objective%20Reinforcement%20Learning%20via%20a%20Bellman%E2%80%93Isaacs%20Weight-Adversary%20/tables/ead23d01fdb4208a1895c04b24205db8b19f3fff0d8ad0795cca8dc1be30c712.jpg)

<a id="paper-2012"></a>

## On Structured State-Space Duality


### Images

![1e7f0d3860b2041851eef69531872f1b3f09b2744ddfa5d0f916f951f0eeb76c.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/1e7f0d3860b2041851eef69531872f1b3f09b2744ddfa5d0f916f951f0eeb76c.jpg)

![6f3fcf2718ab20d5ff8a1354295e7d72a15497f6bb96b1d1e25638f406ad34f7.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/6f3fcf2718ab20d5ff8a1354295e7d72a15497f6bb96b1d1e25638f406ad34f7.jpg)

![70aad1e22d3eeb06f6fd948158828d815688190afda4edd3b5c3948d9ddab17d.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/70aad1e22d3eeb06f6fd948158828d815688190afda4edd3b5c3948d9ddab17d.jpg)

![8811e490977d068c38b663861cf0141df1db413c534ff8d395a80ae7ac0b6092.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/8811e490977d068c38b663861cf0141df1db413c534ff8d395a80ae7ac0b6092.jpg)

![a467e5486bf4c71abacebec60d4448bff0022d3f700f3d577ac84e1f2be94289.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/a467e5486bf4c71abacebec60d4448bff0022d3f700f3d577ac84e1f2be94289.jpg)

![a81b70da3ec7aa8f254293d9af4d3df5ce2387081be04d089dbba2d952f0b01b.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/a81b70da3ec7aa8f254293d9af4d3df5ce2387081be04d089dbba2d952f0b01b.jpg)

![b1eda5c833d73a584253af7514319639f41ea2ccbaf1650d1de1f1a2e4736d47.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/b1eda5c833d73a584253af7514319639f41ea2ccbaf1650d1de1f1a2e4736d47.jpg)

![bdc86911e06d9035298d8ac68a9fe68a9ff36a67d429e7eb07f124f182fb85ba.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/bdc86911e06d9035298d8ac68a9fe68a9ff36a67d429e7eb07f124f182fb85ba.jpg)

![c4161a1b547c38da5a664e8110ecb4b72fe9e3bed3d3aff53c2044bc1f25752b.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/c4161a1b547c38da5a664e8110ecb4b72fe9e3bed3d3aff53c2044bc1f25752b.jpg)

![d398b324090c89b2bdc57e35e788c57a79082061026db9638c5201b97eda971a.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/d398b324090c89b2bdc57e35e788c57a79082061026db9638c5201b97eda971a.jpg)

![d57ec88ce0a5920c4e0461511618a1eafcd85f1a7313022343e95fed9913e040.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/d57ec88ce0a5920c4e0461511618a1eafcd85f1a7313022343e95fed9913e040.jpg)

![e8431812187dc709609e51c04848fe567fd9fc3bba5a9eb929d14b9a512e0f76.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/e8431812187dc709609e51c04848fe567fd9fc3bba5a9eb929d14b9a512e0f76.jpg)

![eac7988df1376e8f6aef7db4790e3254d851908d3f11d4c3f142be5a6544a528.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/eac7988df1376e8f6aef7db4790e3254d851908d3f11d4c3f142be5a6544a528.jpg)

![ef43660e5a3880cc116c50f410cc350b53c492fcaa5923fb8a26d8f95f4c4503.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/ef43660e5a3880cc116c50f410cc350b53c492fcaa5923fb8a26d8f95f4c4503.jpg)

![febf5fd760563793c5f491c236ab8b0c1d7c00b9f6f876b347e802ef8058d6f0.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/images/febf5fd760563793c5f491c236ab8b0c1d7c00b9f6f876b347e802ef8058d6f0.jpg)

### Tables

![5a975b5de6eb0cffb07b42a62fe13fc386dd232306eb5b309402201b21fe2670.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/tables/5a975b5de6eb0cffb07b42a62fe13fc386dd232306eb5b309402201b21fe2670.jpg)

![d669097beeab9277f4fa7af1ad148c766455a36520f3eedc946b34c5afbb0b44.jpg](../icml_results/2040_On%20Structured%20State-Space%20Duality/tables/d669097beeab9277f4fa7af1ad148c766455a36520f3eedc946b34c5afbb0b44.jpg)

<a id="paper-2013"></a>

## TeamWork: Multivariate Time Series Anomaly Detection via Asymmetric Role-aware Channel Modeling

### Images

![0f737fa5c19363197cedc1198b893eed62f684c66e26d2a9e1b2897aff1a1cf3.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/0f737fa5c19363197cedc1198b893eed62f684c66e26d2a9e1b2897aff1a1cf3.jpg)

![367129916a25878f0f79fc7c4dae459380c0667d6c99fdf975311fc003a1de8b.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/367129916a25878f0f79fc7c4dae459380c0667d6c99fdf975311fc003a1de8b.jpg)

![579925a2cc43bb38d68a9bf1ec45b64ccb6c3b8f2f8df1605d633bdc71dccc7d.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/579925a2cc43bb38d68a9bf1ec45b64ccb6c3b8f2f8df1605d633bdc71dccc7d.jpg)

![8006ecade2a0a1af6307114226711e8476ad469a881ae1d14ddcbe681ddd7937.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/8006ecade2a0a1af6307114226711e8476ad469a881ae1d14ddcbe681ddd7937.jpg)

![e87de3a183b834179138fbd4ddd2e86dfae2d0c5c4191e34076da4f637459d80.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/e87de3a183b834179138fbd4ddd2e86dfae2d0c5c4191e34076da4f637459d80.jpg)

![eb0a8806ea0c35ccfea7ad89e8495ad7ebdc17a7e17451e97f12170564391075.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/images/eb0a8806ea0c35ccfea7ad89e8495ad7ebdc17a7e17451e97f12170564391075.jpg)

### Tables

![062ca96d25884bc53ce1003647bc23bbf3893ace8f0043ce416877e91003fc96.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/062ca96d25884bc53ce1003647bc23bbf3893ace8f0043ce416877e91003fc96.jpg)

![2696deab22ea99c55b7bb463df9120956a14acf5ad03c31529d6c022b6c32fec.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/2696deab22ea99c55b7bb463df9120956a14acf5ad03c31529d6c022b6c32fec.jpg)

![3fc8ba9708767e9c46fa193367337470e8a0d9748f87bf1ba375733cc25ac267.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/3fc8ba9708767e9c46fa193367337470e8a0d9748f87bf1ba375733cc25ac267.jpg)

![500d17ffcc9dfdb36f09f5f0b04b0816b76369c877b6d6c0cd202610f34b2686.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/500d17ffcc9dfdb36f09f5f0b04b0816b76369c877b6d6c0cd202610f34b2686.jpg)

![523b027835e2725e909fe37405298eb4ebc9c8f6497125c360a4074ecb08c8bf.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/523b027835e2725e909fe37405298eb4ebc9c8f6497125c360a4074ecb08c8bf.jpg)

![5687dd532d0cda8364235375b27604d5572f04a5e70d349683f63adf36c6587d.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/5687dd532d0cda8364235375b27604d5572f04a5e70d349683f63adf36c6587d.jpg)

![5c2e15938fefd567c15436366256dce3e227acbc87a867073a6a5be2c24c821b.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/5c2e15938fefd567c15436366256dce3e227acbc87a867073a6a5be2c24c821b.jpg)

![5c70fd1bfda16de5c5c807ad430b76dcdddb53e5c2f87cba6fc32ffd979ab656.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/5c70fd1bfda16de5c5c807ad430b76dcdddb53e5c2f87cba6fc32ffd979ab656.jpg)

![bb50972c52166d3d82eb91e1b3080c09ab646769c7f1e1446b10eb68c34324d2.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/bb50972c52166d3d82eb91e1b3080c09ab646769c7f1e1446b10eb68c34324d2.jpg)

![c24a2ce55c2319dabc2ec909bc158e17d1a265413623c41d0e13680356fefe8c.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/c24a2ce55c2319dabc2ec909bc158e17d1a265413623c41d0e13680356fefe8c.jpg)

![c462f3f44f03a99ae3b9b8fb0787d7dc0980b1a6037dd41040fb0227efcf1c22.jpg](../icml_results/2041_TeamWork_%20Multivariate%20Time%20Series%20Anomaly%20Detection%20via%20Asymmetric%20Role-aware%20Channel%20Modeling/tables/c462f3f44f03a99ae3b9b8fb0787d7dc0980b1a6037dd41040fb0227efcf1c22.jpg)
