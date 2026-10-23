# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 183 of 197

## 目录 (Table of Contents)

1. [Strategy-Aware Optimization Modeling with Reasoning LLMs](#paper-6007)
2. [Preference Goal Tuning: Post-Training as Latent Control for Frozen Policies](#paper-6008)
3. [PyVision-RL: Forging Open Agentic Vision Models via RL](#paper-6009)
4. [Benchmarking the Scientific Mind: A Pathology-Derived Biomedical VQA Benchmark for Complex Scientific Reasoning](#paper-6010)
5. [TexEditor: Structure-Preserving Text-Driven Texture Editing](#paper-6011)
6. [Navigating the Energy Landscape of Collaboration: Multi-Agent Communication Graph Generation via Score-Based Diffusion](#paper-6012)
7. [SPAR: Support-Preserving Action Rectification](#paper-6013)
8. [Structured Expert Routing with Multi-View Task Priors for Offline Meta-Reinforcement Learning](#paper-6014)
9. [MeshTok: Efficient Multi-Scale Tokenization for Scalable PDE Transformers](#paper-6015)
10. [Random Scaling of Emergent Capabilities](#paper-6016)
11. [On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](#paper-6017)
12. [Does AI Reviewer See the Full Picture? Attacking and Defending Multimodal Peer Review](#paper-6018)
13. [Exploring 3D Dataset Pruning](#paper-6019)
14. [ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](#paper-6020)
15. [CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation](#paper-6021)
16. [Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding](#paper-6022)
17. [You Don’t Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models](#paper-6023)
18. [SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning](#paper-6024)
19. [A Unified Framework for Deep Hypergraph Clustering Beyond Homophily](#paper-6025)
20. [Gradient Flow Dynamics and Implicit Bias of Diagonal Linear Networks under Infinitesimal Initialization](#paper-6026)
21. [Large Language Model Agents Are Not Always Faithful Self-Evolvers](#paper-6027)
22. [Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks](#paper-6028)
23. [PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Fine-Grained Expert Merging and Bit-packed Inference](#paper-6029)
24. [Contractive Anchor Resolvent Diffusion for Incomplete Multi-View Clustering](#paper-6030)
25. [Anti-Backdoor Coreset Selection via Cumulative Entropy](#paper-6031)
26. [Scaling Prompt Synthesis for Large Language Model Reasoning](#paper-6032)
27. [Attend to Anything: Foundation Model for Unified Human Attention Modeling](#paper-6033)
28. [Weaving in the Clouds: Achieving Synergistic Collaboration among LLM Agents via Federated Learning](#paper-6034)
29. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](#paper-6035)
30. [Broadening the Backdoor Basin: Understanding LLM Backdoors Collapse and Making Backdoors Persistent](#paper-6036)
31. [DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise](#paper-6037)
32. [AutoMS: Multi-Agent Evolutionary Search for Cross-Physics Inverse Microstructure Design](#paper-6038)
33. [TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization](#paper-6039)

---


<a id="paper-6007"></a>

## Strategy-Aware Optimization Modeling with Reasoning LLMs

### Images

![006b30250d471e619220066770cc6caa9154b33ef924a837acadae73bd2281c7.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/006b30250d471e619220066770cc6caa9154b33ef924a837acadae73bd2281c7.jpg)

![692cfa6b9b72c547e1854b4fb95d252c6a1d47ead11151fec4c22e5f1f7dfa5b.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/692cfa6b9b72c547e1854b4fb95d252c6a1d47ead11151fec4c22e5f1f7dfa5b.jpg)

![6cab311595c90a3b16e4402f9eb275fde9ab51b479dffdfeb1b7287a897a2691.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/6cab311595c90a3b16e4402f9eb275fde9ab51b479dffdfeb1b7287a897a2691.jpg)

![7483fc45873f268b30b2663ed9b865fc9ffbf7bd4fb7377b2bea29e01fa6d87f.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/7483fc45873f268b30b2663ed9b865fc9ffbf7bd4fb7377b2bea29e01fa6d87f.jpg)

![7e7d7938ec6058798e335ed5f789e1e7086e64c218bd0c63930a3b3f6e81cbbf.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/7e7d7938ec6058798e335ed5f789e1e7086e64c218bd0c63930a3b3f6e81cbbf.jpg)

![8702742e6bef249824c0bf83d627067640570a464e3f199f51a35d78935e51a4.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/8702742e6bef249824c0bf83d627067640570a464e3f199f51a35d78935e51a4.jpg)

![9f375310f8a38dbd74195de3f37a51f958a58e4ed7d8ece4fb5c7a1cccba5800.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/9f375310f8a38dbd74195de3f37a51f958a58e4ed7d8ece4fb5c7a1cccba5800.jpg)

![ae17599ce8fef4b458925a7caeda85129e5609a33256e1dff211efa2da349469.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/ae17599ce8fef4b458925a7caeda85129e5609a33256e1dff211efa2da349469.jpg)

![c62f2a4588fcf28052547b91ded9cb4774e75bdafb3684229a4bb49df735f9bb.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/c62f2a4588fcf28052547b91ded9cb4774e75bdafb3684229a4bb49df735f9bb.jpg)

![f906df4d115182c1ee095667d68e1d8be73b9cab7ed01ea18958c20ff83e3e00.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/images/f906df4d115182c1ee095667d68e1d8be73b9cab7ed01ea18958c20ff83e3e00.jpg)

### Tables

![021076733f25ad9854bbd97b46ae406867da0b25729c3e0acb50e70eb0331ead.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/021076733f25ad9854bbd97b46ae406867da0b25729c3e0acb50e70eb0331ead.jpg)

![4ac3b54dea157090da0b689151ad02ad08c8b139d07522992f41557700897717.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/4ac3b54dea157090da0b689151ad02ad08c8b139d07522992f41557700897717.jpg)

![5b83b04101bb372fa47f69603d351e9de43b526234f5f3f8d0aba5d18e5b0ea8.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/5b83b04101bb372fa47f69603d351e9de43b526234f5f3f8d0aba5d18e5b0ea8.jpg)

![8ad00627ebfb01f2326cdd07734647b1f5a3b0df56e8d7e2bb1d8666d3b9976f.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/8ad00627ebfb01f2326cdd07734647b1f5a3b0df56e8d7e2bb1d8666d3b9976f.jpg)

![91d25c9ffaa085af5a2c1ce108d5442248a2c13454e54fccb76f2b255a57112d.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/91d25c9ffaa085af5a2c1ce108d5442248a2c13454e54fccb76f2b255a57112d.jpg)

![961cec77729ca3aff2a47f2aefb8fd43d44cd4c9af71a7776a4560e35d9f436c.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/961cec77729ca3aff2a47f2aefb8fd43d44cd4c9af71a7776a4560e35d9f436c.jpg)

![c2d1268b5d6e6ab81e1b30fc102bc650acf717546225dd2ee1b361638496c1c0.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/c2d1268b5d6e6ab81e1b30fc102bc650acf717546225dd2ee1b361638496c1c0.jpg)

![e02452acc7195615d4dbdd4c19def0ceb8355283f73ee0c26cfde8f6509dae1b.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/e02452acc7195615d4dbdd4c19def0ceb8355283f73ee0c26cfde8f6509dae1b.jpg)

![e40ea90c4106c38b85269c6cb4303f163ce1b89451b2532e8134914fd45c5c29.jpg](../icml_results/6055_Rethinking%20Low-Confidence%20Pseudo%20Labels_%20Influence-Aware%20Semi-Supervised%20Fine-Tuning%20for%20Hyperspectr/tables/e40ea90c4106c38b85269c6cb4303f163ce1b89451b2532e8134914fd45c5c29.jpg)

## Strategy-Aware Optimization Modeling with Reasoning LLMs


### Images

![424c0e0cc906ab9c2adafc115d25c18e8ba1e8242b9a3a5158ed9737bd3f9fd9.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/images/424c0e0cc906ab9c2adafc115d25c18e8ba1e8242b9a3a5158ed9737bd3f9fd9.jpg)

![43b6ab46cebce33f736327f4ad15be1927da13476c014234ff441818e8f848ba.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/images/43b6ab46cebce33f736327f4ad15be1927da13476c014234ff441818e8f848ba.jpg)

![f8b0c31c7cf7636b9d46f7b2fc2b4a60cf5cc64f260f954609d202d654b6c563.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/images/f8b0c31c7cf7636b9d46f7b2fc2b4a60cf5cc64f260f954609d202d654b6c563.jpg)

### Tables

![28bb253bd52c8546ffdce36c805a84b96681db2a278662a8ca5d9ed8003bf787.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/28bb253bd52c8546ffdce36c805a84b96681db2a278662a8ca5d9ed8003bf787.jpg)

![4be2b61488da7baa8659d1f65d23d0a1d3755505d7b8d876e8fcb5541a96fcdc.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/4be2b61488da7baa8659d1f65d23d0a1d3755505d7b8d876e8fcb5541a96fcdc.jpg)

![6bf22c16018b171f3a1fb68020e721f395b98b5358dbd4c2e869847787d7e843.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/6bf22c16018b171f3a1fb68020e721f395b98b5358dbd4c2e869847787d7e843.jpg)

![707ffa1c6be64f02011645e3718b4e14c875c9b0fe62e70bb2b996773a5a091a.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/707ffa1c6be64f02011645e3718b4e14c875c9b0fe62e70bb2b996773a5a091a.jpg)

![769a446d071b0e98ea2c613ecc1b7ba2a73bcedeed5a81a7b78f1ba900acc82f.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/769a446d071b0e98ea2c613ecc1b7ba2a73bcedeed5a81a7b78f1ba900acc82f.jpg)

![7f07ff1916064483401ea2884824a86cabe2c9dcd30dbf9a4e32c011a54ea859.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/7f07ff1916064483401ea2884824a86cabe2c9dcd30dbf9a4e32c011a54ea859.jpg)

![90f36f04c917f5f02201ce8b6a3801e428ac75888791fafa0f9c3a54499cac55.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/90f36f04c917f5f02201ce8b6a3801e428ac75888791fafa0f9c3a54499cac55.jpg)

![a3b908c740e2c162b1bd76c76868e49395a5d4fa7a2ad528eb6488b5c743e272.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/a3b908c740e2c162b1bd76c76868e49395a5d4fa7a2ad528eb6488b5c743e272.jpg)

![c2112834e3379ee34121f791f2f7ce1168029b08818326662474d22a9bfe4468.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/c2112834e3379ee34121f791f2f7ce1168029b08818326662474d22a9bfe4468.jpg)

![d080eb6a1292be27730cdc6eb21a42b787f5b08c8fc9ffc4c33eb83c5d2b32d8.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/d080eb6a1292be27730cdc6eb21a42b787f5b08c8fc9ffc4c33eb83c5d2b32d8.jpg)

![e11b6b8136af8cdacd63a252b1af9905659c9dd1279257c3ad0bc93adce5f108.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/e11b6b8136af8cdacd63a252b1af9905659c9dd1279257c3ad0bc93adce5f108.jpg)

![e27d18a6089787e6982a7e5e8bcfe076f03eca2baacfdc17d3eba899b6928c5b.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/e27d18a6089787e6982a7e5e8bcfe076f03eca2baacfdc17d3eba899b6928c5b.jpg)

![e9b1db8bb22a858e25da02084d8b6450c4edc84350e2561c65a1a6eef9ba442a.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/e9b1db8bb22a858e25da02084d8b6450c4edc84350e2561c65a1a6eef9ba442a.jpg)

![ec9a5055e715bcd361629294da532632a367e9913914394fa574d18806332710.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/ec9a5055e715bcd361629294da532632a367e9913914394fa574d18806332710.jpg)

![fefcdefc59c228e44f286aca2950251694bffb4e9117121cd8ea78074bdd1594.jpg](../icml_results/6056_Strategy-Aware%20Optimization%20Modeling%20with%20Reasoning%20LLMs/tables/fefcdefc59c228e44f286aca2950251694bffb4e9117121cd8ea78074bdd1594.jpg)

<a id="paper-6008"></a>

## Preference Goal Tuning: Post-Training as Latent Control for Frozen Policies


### Images

![01849cb0939988bb433040621e5d843f12e7f5c82eb6e6c2b1aadf9ed517961b.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/01849cb0939988bb433040621e5d843f12e7f5c82eb6e6c2b1aadf9ed517961b.jpg)

![03904f004d5fe09458838a1122d4d259cc94b81cffabb922c72b52fbe9c7e1e6.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/03904f004d5fe09458838a1122d4d259cc94b81cffabb922c72b52fbe9c7e1e6.jpg)

![1701f39ad7af2f0847bc5b8e61b424c8c5d423bf7c11dffaea972711d36f3d69.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/1701f39ad7af2f0847bc5b8e61b424c8c5d423bf7c11dffaea972711d36f3d69.jpg)

![1911b70f38c07a78d774a9bd990b6f1ecc3469fc8aefe8beb06953f424432341.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/1911b70f38c07a78d774a9bd990b6f1ecc3469fc8aefe8beb06953f424432341.jpg)

![28dd65190a5a8ee08e3d2d830c812efd9fb2351e7f67c3be465e7fe85fcb2628.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/28dd65190a5a8ee08e3d2d830c812efd9fb2351e7f67c3be465e7fe85fcb2628.jpg)

![3394ab519886bc03596c3b38a9d444bf562d3e73e7697e4a4656617f62b0cb7f.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/3394ab519886bc03596c3b38a9d444bf562d3e73e7697e4a4656617f62b0cb7f.jpg)

![6c9d05d5e3abd9a8b3604d3de780def2b7dfd6611b60ccba6dcf54df0d87c946.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/6c9d05d5e3abd9a8b3604d3de780def2b7dfd6611b60ccba6dcf54df0d87c946.jpg)

![8f73abf57125b1681df38d45d72780018a3adb29d85b8175774d07cdcb51b9b8.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/8f73abf57125b1681df38d45d72780018a3adb29d85b8175774d07cdcb51b9b8.jpg)

![acda95db0e99861160acf9b71b9d3008a854aa154d1fe2a6f8b93a210e3b4cfb.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/acda95db0e99861160acf9b71b9d3008a854aa154d1fe2a6f8b93a210e3b4cfb.jpg)

![ecb94ba74e71c9315e168a37bd9358820ff98f9a72f95ec48c86a9ef744bc727.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/ecb94ba74e71c9315e168a37bd9358820ff98f9a72f95ec48c86a9ef744bc727.jpg)

![efd0ef11b8f4e5fc0731d347ac863ec6981b6896e489cc9655e729d0aebbd4ce.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/efd0ef11b8f4e5fc0731d347ac863ec6981b6896e489cc9655e729d0aebbd4ce.jpg)

![fc8c914e8084cc6dc949ac6a3c2c0bef01472569c7f25daf518cd3e1f67579f2.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/images/fc8c914e8084cc6dc949ac6a3c2c0bef01472569c7f25daf518cd3e1f67579f2.jpg)

### Tables

![2beb44cf4866808965e75aaab07d924643bfaa3b4c1f2de8126a42e591c38189.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/2beb44cf4866808965e75aaab07d924643bfaa3b4c1f2de8126a42e591c38189.jpg)

![396ec9b275bbd9772a2966e0513ef3167460d2f8d9032b213672e5ccfdbcfb5d.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/396ec9b275bbd9772a2966e0513ef3167460d2f8d9032b213672e5ccfdbcfb5d.jpg)

![3def695e3c3d3c2c1bb0ab823e3cfdc5a49561ad6d7cd37c626c00b6a294a87f.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/3def695e3c3d3c2c1bb0ab823e3cfdc5a49561ad6d7cd37c626c00b6a294a87f.jpg)

![4275c03797cca74ff45b26cbff54318c046e3f843998eab31b80c84be18419ca.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/4275c03797cca74ff45b26cbff54318c046e3f843998eab31b80c84be18419ca.jpg)

![4febc8c1715d29197f3b14c13e58fe46fab9673ac18d8cfb92999fe95104136b.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/4febc8c1715d29197f3b14c13e58fe46fab9673ac18d8cfb92999fe95104136b.jpg)

![53f404cb64ddffaf2eb888c695bdba1e2640985aae81f94dfeb00436872b14a6.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/53f404cb64ddffaf2eb888c695bdba1e2640985aae81f94dfeb00436872b14a6.jpg)

![6c0b4981faafc16e3dc43e2f3c120e39897aeabc746ea66666da137f9c44fafe.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/6c0b4981faafc16e3dc43e2f3c120e39897aeabc746ea66666da137f9c44fafe.jpg)

![6dd09471b1cd94429fd40759d6b2a518e522b59120367757658e9fb691ccbf62.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/6dd09471b1cd94429fd40759d6b2a518e522b59120367757658e9fb691ccbf62.jpg)

![d0b558cc74536391b6d4ec569f41bc933b7ade19e9e24efdbc9ba571e18e823e.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/d0b558cc74536391b6d4ec569f41bc933b7ade19e9e24efdbc9ba571e18e823e.jpg)

![d20cbca9435ed1b9553739a5f571e8eb4987e6e51cd2f21fbc5ffc45fe94b1ec.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/d20cbca9435ed1b9553739a5f571e8eb4987e6e51cd2f21fbc5ffc45fe94b1ec.jpg)

![d3a80e2499a70e709561874139853855738a02cb8c893a873016120a8c443e82.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/d3a80e2499a70e709561874139853855738a02cb8c893a873016120a8c443e82.jpg)

![e1815dca509670eb0996a7b7617a09e8238c5a56f13e504b74cc67fadca8b20e.jpg](../icml_results/6057_Preference%20Goal%20Tuning_%20Post-Training%20as%20Latent%20Control%20for%20Frozen%20Policies/tables/e1815dca509670eb0996a7b7617a09e8238c5a56f13e504b74cc67fadca8b20e.jpg)

<a id="paper-6009"></a>

## PyVision-RL: Forging Open Agentic Vision Models via RL


### Images

![03bad4bbf3d8c30757a8cc69b52bb4d48d79a03caad12818c843beb9fbce39c9.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/03bad4bbf3d8c30757a8cc69b52bb4d48d79a03caad12818c843beb9fbce39c9.jpg)

![05daf8fdeb7585dcce0597da2deddb7bcc0a8b4948bc0b0e22e9ce30ef3f4187.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/05daf8fdeb7585dcce0597da2deddb7bcc0a8b4948bc0b0e22e9ce30ef3f4187.jpg)

![08bf37b4cf211f8b2ca0acd804b3287eb76f7671a0c47f201ad3931de8316df5.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/08bf37b4cf211f8b2ca0acd804b3287eb76f7671a0c47f201ad3931de8316df5.jpg)

![0b84924d5a92236e8c1026249d369b7be4e8454d57a1d34518e4989b3f3c808a.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/0b84924d5a92236e8c1026249d369b7be4e8454d57a1d34518e4989b3f3c808a.jpg)

![16b6aae8233c780a5523a5c510e8b2839e279247a1858e632d86a744a11bd00e.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/16b6aae8233c780a5523a5c510e8b2839e279247a1858e632d86a744a11bd00e.jpg)

![18279bb999b0eeb63f245a803cd790aae4594222c355925c7fa8b47c3fbcadea.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/18279bb999b0eeb63f245a803cd790aae4594222c355925c7fa8b47c3fbcadea.jpg)

![20eb1f7329bdcfadc9fded08a15f4e1939ea561b1922668358133eb5240a3801.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/20eb1f7329bdcfadc9fded08a15f4e1939ea561b1922668358133eb5240a3801.jpg)

![2423bb3acf2c6a7e90494fc58876f1947c835b1e469e5dc3469e3add7baf8c7e.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/2423bb3acf2c6a7e90494fc58876f1947c835b1e469e5dc3469e3add7baf8c7e.jpg)

![303d14d85a4be602c4734e64754c868b44048835094494a14e204ea076e4c396.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/303d14d85a4be602c4734e64754c868b44048835094494a14e204ea076e4c396.jpg)

![38c04838157d721fa2561fa41a15a175ea523b22cb29d74f602ed01d45b29d50.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/38c04838157d721fa2561fa41a15a175ea523b22cb29d74f602ed01d45b29d50.jpg)

![3b1c5aea01489161e29eaed8f329cc26d92d830f2a0c7c73e655b373aac38846.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/3b1c5aea01489161e29eaed8f329cc26d92d830f2a0c7c73e655b373aac38846.jpg)

![4c5e2606466a12fa5375dd5fe1914a86f49aa21f501e64a4c38bd6b084e34414.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/4c5e2606466a12fa5375dd5fe1914a86f49aa21f501e64a4c38bd6b084e34414.jpg)

![4d964f5e0db08c64fd0538dd103ad236982c431aa0bcd34daed689e6505b8ace.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/4d964f5e0db08c64fd0538dd103ad236982c431aa0bcd34daed689e6505b8ace.jpg)

![508a985b50b843fbca20e76730732a5ecc2ec929b2542996a180dccdac7151d5.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/508a985b50b843fbca20e76730732a5ecc2ec929b2542996a180dccdac7151d5.jpg)

![5a2a4fb006e636d5fb82c2c47b929cdaefdc0196050ce54d754af00a4702563a.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/5a2a4fb006e636d5fb82c2c47b929cdaefdc0196050ce54d754af00a4702563a.jpg)

![622fa6d20ff791cde27bde0dbf5e8311b6cf7e22599473d57f4eb18e75c52953.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/622fa6d20ff791cde27bde0dbf5e8311b6cf7e22599473d57f4eb18e75c52953.jpg)

![632d66c52a5d63c27a85cb5049f7c529e155496508ea722c8ee2589a9c555d69.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/632d66c52a5d63c27a85cb5049f7c529e155496508ea722c8ee2589a9c555d69.jpg)

![65d106b1937ad5934e35a1d01e7e639eeb12115251c316a37d399a445f94ee73.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/65d106b1937ad5934e35a1d01e7e639eeb12115251c316a37d399a445f94ee73.jpg)

![7002c3c57f50693f863faa98349c7ddcb155f836097213fdff5f98562bf7d17b.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/7002c3c57f50693f863faa98349c7ddcb155f836097213fdff5f98562bf7d17b.jpg)

![7f3d89da8e4be280bfe335573bb66d09fa04bd5e43850380cc753e926946d0d5.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/7f3d89da8e4be280bfe335573bb66d09fa04bd5e43850380cc753e926946d0d5.jpg)

![860829afbc43382adf9728ab31d81b2b90a1f2c2debb56eacdd24f81589b176e.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/860829afbc43382adf9728ab31d81b2b90a1f2c2debb56eacdd24f81589b176e.jpg)

![8afee6489331a34fbb9c423c665e88d1f0bb0ee920dacdeb19ccc6c5dc159498.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/8afee6489331a34fbb9c423c665e88d1f0bb0ee920dacdeb19ccc6c5dc159498.jpg)

![948d9c0c9db28592ebdfe9523d068c72ceaa5822266e67eac16f18d150a584f7.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/948d9c0c9db28592ebdfe9523d068c72ceaa5822266e67eac16f18d150a584f7.jpg)

![96bca3426acb132a527d5190beaa2fcda7d4e4681ddcf50534c77b65f2df1fd9.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/96bca3426acb132a527d5190beaa2fcda7d4e4681ddcf50534c77b65f2df1fd9.jpg)

![9930a60d695bb9e3bbbf5ef698780685adaf0c42e758fbd69f7768a5bdaa26d3.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/9930a60d695bb9e3bbbf5ef698780685adaf0c42e758fbd69f7768a5bdaa26d3.jpg)

![a9dbec8337eeac9f396ec736318981acf58f11f17921b24feef95cae5e6bedc4.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/a9dbec8337eeac9f396ec736318981acf58f11f17921b24feef95cae5e6bedc4.jpg)

![b71f58e398af64f0fb2c82344bdd994827118c5d68e4fc71c26ae84a12569fa4.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/b71f58e398af64f0fb2c82344bdd994827118c5d68e4fc71c26ae84a12569fa4.jpg)

![de7d331ebb167dfdfbe40f74f7c80b93f0e3db20e1130a560864aa44b49e2cf9.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/de7d331ebb167dfdfbe40f74f7c80b93f0e3db20e1130a560864aa44b49e2cf9.jpg)

![e56fd707face7845aa169918c558a6bed7caa5aa7b7241eefcfb0403bab727e3.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/e56fd707face7845aa169918c558a6bed7caa5aa7b7241eefcfb0403bab727e3.jpg)

![ea5f14c1695453f2202f16c2ae0f440feb8f81a44fce7f9299e85f1bde108469.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/ea5f14c1695453f2202f16c2ae0f440feb8f81a44fce7f9299e85f1bde108469.jpg)

![ee6d913bed47ee2e7ba0f68dfc557cd516431238713a908b405aaec9177408d0.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/ee6d913bed47ee2e7ba0f68dfc557cd516431238713a908b405aaec9177408d0.jpg)

![fe0d24a2e4563978e2da206e17a9dbd7da315e56a74bafcad13edc25489795f0.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/images/fe0d24a2e4563978e2da206e17a9dbd7da315e56a74bafcad13edc25489795f0.jpg)

### Tables

![093c0cbfec6d8b42b5516694d544ca2de32f329c471700a4b699a513cc793486.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/tables/093c0cbfec6d8b42b5516694d544ca2de32f329c471700a4b699a513cc793486.jpg)

![b15d08ab9d5e0b559d1e286993acd6201462f832e137b06106cfd9d470daf44e.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/tables/b15d08ab9d5e0b559d1e286993acd6201462f832e137b06106cfd9d470daf44e.jpg)

![bd45371df4fe7a89980baf8f6ddb14350c433af330db49720a570c69cbc550b6.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/tables/bd45371df4fe7a89980baf8f6ddb14350c433af330db49720a570c69cbc550b6.jpg)

![d9a50d873e2469ea20c539a741a626d606a41f27504af36332a59ef442a3e38f.jpg](../icml_results/6058_PyVision-RL_%20Forging%20Open%20Agentic%20Vision%20Models%20via%20RL/tables/d9a50d873e2469ea20c539a741a626d606a41f27504af36332a59ef442a3e38f.jpg)

<a id="paper-6010"></a>

## Benchmarking the Scientific Mind: A Pathology-Derived Biomedical VQA Benchmark for Complex Scientific Reasoning


### Images

![2668e9b4db00263c937f156d410e36257a38169dbde15806796cb6512ee0ab66.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/2668e9b4db00263c937f156d410e36257a38169dbde15806796cb6512ee0ab66.jpg)

![3b5c047903c487688e9fd293c9970a37dbae062d04b00243a3f90f8237f6f45a.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/3b5c047903c487688e9fd293c9970a37dbae062d04b00243a3f90f8237f6f45a.jpg)

![3c8c79974127f7d3d5222cfcb2e3d445bd8fd7ccbd80414ee7ab5181457b3d31.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/3c8c79974127f7d3d5222cfcb2e3d445bd8fd7ccbd80414ee7ab5181457b3d31.jpg)

![64055ff2936df4c76a0c89fc80007026efc33c29b951ab259c846f95e13cf228.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/64055ff2936df4c76a0c89fc80007026efc33c29b951ab259c846f95e13cf228.jpg)

![a3d4c4a3707dec30ec2eaf8740b8eb790ae275f690b201c1070e98ccda6297b3.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/a3d4c4a3707dec30ec2eaf8740b8eb790ae275f690b201c1070e98ccda6297b3.jpg)

![adba909684f5ed141f1718d6fb184680a7fdc0a62f4535a082fa03e42247039a.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/adba909684f5ed141f1718d6fb184680a7fdc0a62f4535a082fa03e42247039a.jpg)

![bd2e492889671b65d25ef692760d8c530d16ebdf17a7a1079dbeb30725215b80.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/bd2e492889671b65d25ef692760d8c530d16ebdf17a7a1079dbeb30725215b80.jpg)

![ec31ef521e7560dbe6e83405001307e1d3926895786c8c637724552d7005e838.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/ec31ef521e7560dbe6e83405001307e1d3926895786c8c637724552d7005e838.jpg)

![f7ad9702fd4e00bc2f380bd3d2912e3fce2b2350169f0c99f6d96329cf5c1c95.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/images/f7ad9702fd4e00bc2f380bd3d2912e3fce2b2350169f0c99f6d96329cf5c1c95.jpg)

### Tables

![135d055349d53a8a916853ec4629d7d761d101fb81bb0ec9df1e74a93d25795f.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/135d055349d53a8a916853ec4629d7d761d101fb81bb0ec9df1e74a93d25795f.jpg)

![305597e768a2d04e67e8799decbd8c4d2469de312a65ddb4dfcadfd4ff57d632.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/305597e768a2d04e67e8799decbd8c4d2469de312a65ddb4dfcadfd4ff57d632.jpg)

![82195ab11ab2dafe7097a00c9dbdd7f3b1c75ff690c2b94b41a2500283d66ae6.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/82195ab11ab2dafe7097a00c9dbdd7f3b1c75ff690c2b94b41a2500283d66ae6.jpg)

![89707b39d86a37a1102be3b2902d114b38a6c226920b3b08d10ea306b295983f.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/89707b39d86a37a1102be3b2902d114b38a6c226920b3b08d10ea306b295983f.jpg)

![b45e2473ebecf0f8c4c5770204946b6b295ec9bf99756606f6e11d35d5b8eda9.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/b45e2473ebecf0f8c4c5770204946b6b295ec9bf99756606f6e11d35d5b8eda9.jpg)

![cc3f63df12a4b3178572ff1c609979cee6682cb6981a1322a20decb80fb263d4.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/cc3f63df12a4b3178572ff1c609979cee6682cb6981a1322a20decb80fb263d4.jpg)

![db06aa89cf8a817e0b985e37620665202fda98f3c088524bcd89c8f7ac4a404a.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/db06aa89cf8a817e0b985e37620665202fda98f3c088524bcd89c8f7ac4a404a.jpg)

![e0d33df2c09712b8c893810b19b8da049a62436c97c7031886968e04ffddbd83.jpg](../icml_results/6059_Benchmarking%20the%20Scientific%20Mind_%20A%20Pathology-Derived%20Biomedical%20VQA%20Benchmark%20for%20Complex%20Scientifi/tables/e0d33df2c09712b8c893810b19b8da049a62436c97c7031886968e04ffddbd83.jpg)

<a id="paper-6011"></a>

## TexEditor: Structure-Preserving Text-Driven Texture Editing


### Images

![192089189e67d439328737287c293d47163105eee5bfe3ad7e038a8cbe08fa65.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/192089189e67d439328737287c293d47163105eee5bfe3ad7e038a8cbe08fa65.jpg)

![3869b729d1ba0b125ecef5462cab6ffee8d92c091e5869d6ffcbbe60279d389f.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/3869b729d1ba0b125ecef5462cab6ffee8d92c091e5869d6ffcbbe60279d389f.jpg)

![403df67535597e36a319c4ae2ddde00a4fcc17627dc04001adfab9aed8e720b5.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/403df67535597e36a319c4ae2ddde00a4fcc17627dc04001adfab9aed8e720b5.jpg)

![41dc47966bb2d5a2e50ac20c523232d42946f5c4e7a8e1869ad2be83b3d5d4ab.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/41dc47966bb2d5a2e50ac20c523232d42946f5c4e7a8e1869ad2be83b3d5d4ab.jpg)

![4a431f0d0a378e173d4a6c563aa18fafa9b329b519e42fddd9280785ab51e2d0.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/4a431f0d0a378e173d4a6c563aa18fafa9b329b519e42fddd9280785ab51e2d0.jpg)

![5b4f67c7dd1d66c5974e635e0c28a72ef9811653b46e4d73ddda2ba8c7058357.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/5b4f67c7dd1d66c5974e635e0c28a72ef9811653b46e4d73ddda2ba8c7058357.jpg)

![672f2548b1c165d84f4388bc9464dcc8a86f62a44d3339315c9c6f9bfce35550.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/672f2548b1c165d84f4388bc9464dcc8a86f62a44d3339315c9c6f9bfce35550.jpg)

![7869ae4494a8c8c633fd360d34100df613282734c6cc45375d3d71dea25e8aca.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/7869ae4494a8c8c633fd360d34100df613282734c6cc45375d3d71dea25e8aca.jpg)

![7a41208fa95b2600f3691e3a950ef394df759ac29bc779d2057ab50d1912effa.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/7a41208fa95b2600f3691e3a950ef394df759ac29bc779d2057ab50d1912effa.jpg)

![b76e9044e2b59f6f986845b0a636e29bdce8e4a85bc4c58e4a58ba04b826fc73.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/b76e9044e2b59f6f986845b0a636e29bdce8e4a85bc4c58e4a58ba04b826fc73.jpg)

![b85217d5cc75634b4c8f64619747c569f040e8c628f2d17543bd68f03680d112.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/b85217d5cc75634b4c8f64619747c569f040e8c628f2d17543bd68f03680d112.jpg)

![e0c800614af6d8ee64eb2826bf0922a900a40ce79337d7c6430bb99ee297b493.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/e0c800614af6d8ee64eb2826bf0922a900a40ce79337d7c6430bb99ee297b493.jpg)

![e329aad381d1b5f7137426bc416558121e1a996a0be0a4f224eb4b93498164d4.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/e329aad381d1b5f7137426bc416558121e1a996a0be0a4f224eb4b93498164d4.jpg)

![e6d1c4e41fde8b71f0b24844aaed7645c2f67727329cf07aa7f6bc41d115e738.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/images/e6d1c4e41fde8b71f0b24844aaed7645c2f67727329cf07aa7f6bc41d115e738.jpg)

### Tables

![11058d72dd28d0dce2643798738c98340ca093d960b797ce5bafd67cb033f301.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/11058d72dd28d0dce2643798738c98340ca093d960b797ce5bafd67cb033f301.jpg)

![2fb60b8b3ceb9ac76335b0a6ae1c2e3ca6a6b82cca6f0dee6c8ebb03ce4c94ad.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/2fb60b8b3ceb9ac76335b0a6ae1c2e3ca6a6b82cca6f0dee6c8ebb03ce4c94ad.jpg)

![7f4bf6a15350598ed6c58133faa1d81681c32dce419a193b0f61a3ef079e006f.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/7f4bf6a15350598ed6c58133faa1d81681c32dce419a193b0f61a3ef079e006f.jpg)

![91f606f52302deac2f22a165ef27a94a9ab314a4ccb11197022581e2392a84d1.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/91f606f52302deac2f22a165ef27a94a9ab314a4ccb11197022581e2392a84d1.jpg)

![c3eeaba4054a816a99ffc8b6a272f0e8d3da57e686b6ccc70abce816cd3fbf3c.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/c3eeaba4054a816a99ffc8b6a272f0e8d3da57e686b6ccc70abce816cd3fbf3c.jpg)

![d4b9c6b9cc57988f5d2f4ff7bfd3cdc8bf07017b31fdf872ff3359a60a0b6b31.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/d4b9c6b9cc57988f5d2f4ff7bfd3cdc8bf07017b31fdf872ff3359a60a0b6b31.jpg)

![f79c111d9778385cf2c85fe15b76d1d88989146013e30fa3b1282aad394d9253.jpg](../icml_results/6060_TexEditor_%20Structure-Preserving%20Text-Driven%20Texture%20Editing/tables/f79c111d9778385cf2c85fe15b76d1d88989146013e30fa3b1282aad394d9253.jpg)

<a id="paper-6012"></a>

## Navigating the Energy Landscape of Collaboration: Multi-Agent Communication Graph Generation via Score-Based Diffusion


### Images

![6cc3a17011b4530132b14215982eede9d731ede25977dcb2cb9247ea2416b68c.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/images/6cc3a17011b4530132b14215982eede9d731ede25977dcb2cb9247ea2416b68c.jpg)

![7411316fffeb61e4e85ea231cefcb265c2b213929118d157c8caf4edfcdc7a22.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/images/7411316fffeb61e4e85ea231cefcb265c2b213929118d157c8caf4edfcdc7a22.jpg)

![9e4eec64dc828b8f30c3590987b98787da4ac4b7b8f729c8551e38e9afb2c7aa.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/images/9e4eec64dc828b8f30c3590987b98787da4ac4b7b8f729c8551e38e9afb2c7aa.jpg)

![fe5d9c025137e339e84d0b0d7a0bb49c80c05db153336e4346c7285ede9cf7fa.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/images/fe5d9c025137e339e84d0b0d7a0bb49c80c05db153336e4346c7285ede9cf7fa.jpg)

### Tables

![31c8c52bb6adc532a2915ed11279f14bca2b1299f514d30863afc2eea8aa95ae.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/31c8c52bb6adc532a2915ed11279f14bca2b1299f514d30863afc2eea8aa95ae.jpg)

![56acf8d639c408e90093b02fb32795fc9987ee1d8ab94a5b482e32ecceb0f3c2.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/56acf8d639c408e90093b02fb32795fc9987ee1d8ab94a5b482e32ecceb0f3c2.jpg)

![58567d8de1955785ab4f57ec3fb32c5a92bb048edb6681d8da867022b9ff8d6c.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/58567d8de1955785ab4f57ec3fb32c5a92bb048edb6681d8da867022b9ff8d6c.jpg)

![62b890798234a359398a36d335bb165f3147e832f32d0a4525f57585d1785cb3.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/62b890798234a359398a36d335bb165f3147e832f32d0a4525f57585d1785cb3.jpg)

![9cca80fff6dd1a02633c799efb67c2fdbf72d6689d0a810e4b4607bbff7c7172.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/9cca80fff6dd1a02633c799efb67c2fdbf72d6689d0a810e4b4607bbff7c7172.jpg)

![a91f2c3324a1450d55228ad8710b98256bd5ad4e6347d1094557c004bda99285.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/a91f2c3324a1450d55228ad8710b98256bd5ad4e6347d1094557c004bda99285.jpg)

![b698ec86a492e56e97f8e329da14d159628d974b17c4a8d88738ca66b379c3b3.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/b698ec86a492e56e97f8e329da14d159628d974b17c4a8d88738ca66b379c3b3.jpg)

![e101e743ddb9f0117b9046314781eab106d8cbfdd3fab668c6cedad2591b4f34.jpg](../icml_results/6061_Navigating%20the%20Energy%20Landscape%20of%20Collaboration_%20Multi-Agent%20Communication%20Graph%20Generation%20via%20Sco/tables/e101e743ddb9f0117b9046314781eab106d8cbfdd3fab668c6cedad2591b4f34.jpg)

<a id="paper-6013"></a>

## SPAR: Support-Preserving Action Rectification


### Images

![274be995e1c71a566d1604b3031d0d8beb1292192e26200536ee4f8983768c7e.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/274be995e1c71a566d1604b3031d0d8beb1292192e26200536ee4f8983768c7e.jpg)

![4c980c9ef0b40f8be149d508353df5402caf4b130e22b73e76e38d3f7fc53f7c.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/4c980c9ef0b40f8be149d508353df5402caf4b130e22b73e76e38d3f7fc53f7c.jpg)

![5a9d85e149bc3069e22b3e87cd8e1a295112acbdc7044e85095b8f14a523b860.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/5a9d85e149bc3069e22b3e87cd8e1a295112acbdc7044e85095b8f14a523b860.jpg)

![7e934fd4f24656bbcee5b522a4cd757aa5ac093db343d959eaf349dd8f2c68b8.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/7e934fd4f24656bbcee5b522a4cd757aa5ac093db343d959eaf349dd8f2c68b8.jpg)

![934183431b05c294947a321ad72ad24f7794cd77ffbe13243245258474c998b5.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/934183431b05c294947a321ad72ad24f7794cd77ffbe13243245258474c998b5.jpg)

![9bb7c42266a91a81227ce4480d15472d0413c76a1d4acdf8af6ed75f86f2df4d.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/images/9bb7c42266a91a81227ce4480d15472d0413c76a1d4acdf8af6ed75f86f2df4d.jpg)

### Tables

![043fa5c480d36c04cb02c261e4abcf4dd08444a561fb56d5364e25c4a2daafb7.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/043fa5c480d36c04cb02c261e4abcf4dd08444a561fb56d5364e25c4a2daafb7.jpg)

![2adcb17c38259fe184523d412561532ed38448443fb876e3b9fc4fe2d41e8985.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/2adcb17c38259fe184523d412561532ed38448443fb876e3b9fc4fe2d41e8985.jpg)

![34445cdf28879392f04bb863e01f73d57d906b2d821fe2adafc4a49886a07b0a.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/34445cdf28879392f04bb863e01f73d57d906b2d821fe2adafc4a49886a07b0a.jpg)

![622965a6e1b141f019111a144600fb23bf313be7760d2547c0ed944768b24328.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/622965a6e1b141f019111a144600fb23bf313be7760d2547c0ed944768b24328.jpg)

![639a1625534cd8ceb2f5b666db9d3b0779d946a7dd04737e3668988feca278c8.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/639a1625534cd8ceb2f5b666db9d3b0779d946a7dd04737e3668988feca278c8.jpg)

![9410e43822f313376690791a421bd31000e68661daa2065e9f00d2161e34065b.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/9410e43822f313376690791a421bd31000e68661daa2065e9f00d2161e34065b.jpg)

![a1345b6836990d42e880bb1e295200ed79b29c696bdff3c1082d4d5798ac31cb.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/a1345b6836990d42e880bb1e295200ed79b29c696bdff3c1082d4d5798ac31cb.jpg)

![a1d222302adfb88b2760316b5ce273787a72c6b02301fd340cb1cad5ec0965db.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/a1d222302adfb88b2760316b5ce273787a72c6b02301fd340cb1cad5ec0965db.jpg)

![b0669d22a683f1c7777a47b04db568530b2f9a531e82c3456e09021e25ba0b40.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/b0669d22a683f1c7777a47b04db568530b2f9a531e82c3456e09021e25ba0b40.jpg)

![b3aa3f25d96b4aaf16a50e0f52adfd01352980bef3f47ae577b273ca8983140d.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/b3aa3f25d96b4aaf16a50e0f52adfd01352980bef3f47ae577b273ca8983140d.jpg)

![c90c97235e1bc015c9c5a28d63668a544138234a45c1628184a19343ee737c34.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/c90c97235e1bc015c9c5a28d63668a544138234a45c1628184a19343ee737c34.jpg)

![cbca0ad7449cd22ca84635ad0f8785fd593caf76c55fcaf78a8e0b83587bba2a.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/cbca0ad7449cd22ca84635ad0f8785fd593caf76c55fcaf78a8e0b83587bba2a.jpg)

![d380d6f7040fa455cc430f51c3768cdff50733f9f644a5693d1d994285c4d555.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/d380d6f7040fa455cc430f51c3768cdff50733f9f644a5693d1d994285c4d555.jpg)

![eb562f9a01b1ecdb78a044f1aff6945d6494d800ed8ce125e2a5bcab1a5af3cb.jpg](../icml_results/6063_SPAR_%20Support-Preserving%20Action%20Rectification/tables/eb562f9a01b1ecdb78a044f1aff6945d6494d800ed8ce125e2a5bcab1a5af3cb.jpg)

<a id="paper-6014"></a>

## Structured Expert Routing with Multi-View Task Priors for Offline Meta-Reinforcement Learning


### Images

![afb55049e4b0a23bf8a5ab925981830351e40b1b455e9a25487eded58bbd4fe2.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/images/afb55049e4b0a23bf8a5ab925981830351e40b1b455e9a25487eded58bbd4fe2.jpg)

![e2a6a1e2ad0a9b0f7f5c02a3921550589e37fe4b3387aefd11f2eac5b1e44538.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/images/e2a6a1e2ad0a9b0f7f5c02a3921550589e37fe4b3387aefd11f2eac5b1e44538.jpg)

### Tables

![06703eda0dd01512957d37dc05e19d175de195573e659e28ab308a64c228bf7c.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/06703eda0dd01512957d37dc05e19d175de195573e659e28ab308a64c228bf7c.jpg)

![1115397542e6fb7e2b9c543d42514db7d6d550739be6db6fa390449af9bfaaa3.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/1115397542e6fb7e2b9c543d42514db7d6d550739be6db6fa390449af9bfaaa3.jpg)

![17308b86f18929795b9636172de939f8af776b54cbd95e7e1c841283de00209a.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/17308b86f18929795b9636172de939f8af776b54cbd95e7e1c841283de00209a.jpg)

![4a463f6f423f2d9e125eae45f800c0a391600dabde1cc4f129974058fdb8e93a.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/4a463f6f423f2d9e125eae45f800c0a391600dabde1cc4f129974058fdb8e93a.jpg)

![4b4e26879a6b6db7f3d5f7c0736078f6e8e0aa35c668fbe131be466056694e15.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/4b4e26879a6b6db7f3d5f7c0736078f6e8e0aa35c668fbe131be466056694e15.jpg)

![6115d7d752778411860c66a0107fc884dfa7ab72d69039776fe308c013328191.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/6115d7d752778411860c66a0107fc884dfa7ab72d69039776fe308c013328191.jpg)

![75b1a491f7296e3bfd691759a9c92301f3f06808c472ea7539c7dde9d5108b39.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/75b1a491f7296e3bfd691759a9c92301f3f06808c472ea7539c7dde9d5108b39.jpg)

![88b82b12aa4929827c399c1461a82bf49e694d3f5324c8f4faa3e7208a394f58.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/88b82b12aa4929827c399c1461a82bf49e694d3f5324c8f4faa3e7208a394f58.jpg)

![931ddd9ed0e1c7d03b1258d32b4100ea98afcb6ad2523234de65dffa832461a4.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/931ddd9ed0e1c7d03b1258d32b4100ea98afcb6ad2523234de65dffa832461a4.jpg)

![c47823819bf6232256c0316d5b5c074d9581874c6d6a948a806c6e29a5c91de8.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/c47823819bf6232256c0316d5b5c074d9581874c6d6a948a806c6e29a5c91de8.jpg)

![cbd3f27ef7739d7c088d73a3978ecc327d65d68e915980671cec9d7f3ffcf7df.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/cbd3f27ef7739d7c088d73a3978ecc327d65d68e915980671cec9d7f3ffcf7df.jpg)

![fa47cf8d2af9e2b1988c1c07b0132a3b28f94b5a69177653fb904004b102d0d3.jpg](../icml_results/6064_Structured%20Expert%20Routing%20with%20Multi-View%20Task%20Priors%20for%20Offline%20Meta-Reinforcement%20Learning/tables/fa47cf8d2af9e2b1988c1c07b0132a3b28f94b5a69177653fb904004b102d0d3.jpg)

<a id="paper-6015"></a>

## MeshTok: Efficient Multi-Scale Tokenization for Scalable PDE Transformers


### Images

![4a3fdbdb23a8b102832d50aa8108aa13f48e0d0296d0016c731e578243c2f79b.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/4a3fdbdb23a8b102832d50aa8108aa13f48e0d0296d0016c731e578243c2f79b.jpg)

![4b168379cc64bfce4398318eafae8286b57604791fdc286c5c9ffa6b6b17d1b8.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/4b168379cc64bfce4398318eafae8286b57604791fdc286c5c9ffa6b6b17d1b8.jpg)

![65a420b8621e0480dcc0ab33d0bc8728697aad3fc0dcffe11b1690146650027b.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/65a420b8621e0480dcc0ab33d0bc8728697aad3fc0dcffe11b1690146650027b.jpg)

![ba76a4cb7ecb8ef64072bcdd6ecab18a72914c84cde57d3f8a5da864a0b0dad5.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/ba76a4cb7ecb8ef64072bcdd6ecab18a72914c84cde57d3f8a5da864a0b0dad5.jpg)

![c619ec8885216a1f037b308bec58509777ca1c0f021db333cb67e4825da0de1c.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/c619ec8885216a1f037b308bec58509777ca1c0f021db333cb67e4825da0de1c.jpg)

![e571884da92dfa0b3afb6af078c9fe375b7dac4f6b52cf37c36ad4bf55a9f7ff.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/images/e571884da92dfa0b3afb6af078c9fe375b7dac4f6b52cf37c36ad4bf55a9f7ff.jpg)

### Tables

![1800896247a53eaeeec55fbd2df9e9601b7991cc4d6de799092bcb3da337f0da.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/1800896247a53eaeeec55fbd2df9e9601b7991cc4d6de799092bcb3da337f0da.jpg)

![207e5a3e4e878ac6275e3fede57408b4548e1b9b622d1857b0c2947ceb39e0b5.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/207e5a3e4e878ac6275e3fede57408b4548e1b9b622d1857b0c2947ceb39e0b5.jpg)

![2574280c2aa0a5d46b27b87e806a7bafba722bc38e71659ca56574d2d3a26c54.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/2574280c2aa0a5d46b27b87e806a7bafba722bc38e71659ca56574d2d3a26c54.jpg)

![26baf353c86850ba2842a5a6d7c39c175511b2a64719be05f79ba90016436762.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/26baf353c86850ba2842a5a6d7c39c175511b2a64719be05f79ba90016436762.jpg)

![3294b00bbea480d5dfc638506fdde994d425caf36de863231868e4b5760b12f4.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/3294b00bbea480d5dfc638506fdde994d425caf36de863231868e4b5760b12f4.jpg)

![4a443bc18a98bbe59d5e6c50bc265aa1288166313b324ab04dafea972870a608.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/4a443bc18a98bbe59d5e6c50bc265aa1288166313b324ab04dafea972870a608.jpg)

![50961f92e0c403f68a528757b183374b661451ca70dfb5517e0ebd986d8daaf7.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/50961f92e0c403f68a528757b183374b661451ca70dfb5517e0ebd986d8daaf7.jpg)

![679a7eac7cd4df613db03ef9cf5cee6a807a1cbbf026d9e11832f942da3a4853.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/679a7eac7cd4df613db03ef9cf5cee6a807a1cbbf026d9e11832f942da3a4853.jpg)

![680293fcf15c8ce3073bd727d74d7353f58c4b07f8c7496d262140d9906714eb.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/680293fcf15c8ce3073bd727d74d7353f58c4b07f8c7496d262140d9906714eb.jpg)

![6e91a85d03ec1aa8276ac3de1210e74eb86286b9ecbeaeefed0bd0e65a930d0f.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/6e91a85d03ec1aa8276ac3de1210e74eb86286b9ecbeaeefed0bd0e65a930d0f.jpg)

![7024c851d444970d4d73f6fec668ea3c041f80cf24902f0852fe5f8de586d6f8.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/7024c851d444970d4d73f6fec668ea3c041f80cf24902f0852fe5f8de586d6f8.jpg)

![80ab5b65cfdc4af7b0dbc2fadfb144e4e819f33fe86f34a95867d16058cd0454.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/80ab5b65cfdc4af7b0dbc2fadfb144e4e819f33fe86f34a95867d16058cd0454.jpg)

![a122a43bb37edec027ed3636faeb7bb0c60c8d11206edda4476eac024acfbc07.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/a122a43bb37edec027ed3636faeb7bb0c60c8d11206edda4476eac024acfbc07.jpg)

![b209fba9c3cd324dcae0c717aa6ccf146dbe4e80b299ef7feaf52fec263a91d5.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/b209fba9c3cd324dcae0c717aa6ccf146dbe4e80b299ef7feaf52fec263a91d5.jpg)

![b3de2e1ba14c6e6adf2a4609b8829a3e6889103e1962dd24737b6b183e8f4310.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/b3de2e1ba14c6e6adf2a4609b8829a3e6889103e1962dd24737b6b183e8f4310.jpg)

![d7b5a793bf3290a02bfacfa94ea6e147367e376e53cbf0af4328df17c8bd21e2.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/d7b5a793bf3290a02bfacfa94ea6e147367e376e53cbf0af4328df17c8bd21e2.jpg)

![d92ce6a3dee861bc26c90d9f3a5ca6f5bc3eb28344e29136450852c4045f50ac.jpg](../icml_results/6065_MeshTok_%20Efficient%20Multi-Scale%20Tokenization%20for%20Scalable%20PDE%20Transformers/tables/d92ce6a3dee861bc26c90d9f3a5ca6f5bc3eb28344e29136450852c4045f50ac.jpg)

<a id="paper-6016"></a>

## Random Scaling of Emergent Capabilities


### Images

![2a5ca5f8b4968fe3084774065dbace0c90b5872a61ef728c845d1b36a70edc0f.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/2a5ca5f8b4968fe3084774065dbace0c90b5872a61ef728c845d1b36a70edc0f.jpg)

![2d9acb8d556e81e91fc970f7ccc9ecb43a74c9d267414c2bea0ac7d665971592.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/2d9acb8d556e81e91fc970f7ccc9ecb43a74c9d267414c2bea0ac7d665971592.jpg)

![2e2e789b30d83723a40dc02bc0e50efff78a582a1cb20e250d633f07452ecb5b.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/2e2e789b30d83723a40dc02bc0e50efff78a582a1cb20e250d633f07452ecb5b.jpg)

![30361f17dcb9aa71cde6adb68d1cb5abd9f33cf04a57aff9167bf2ee1d3f2615.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/30361f17dcb9aa71cde6adb68d1cb5abd9f33cf04a57aff9167bf2ee1d3f2615.jpg)

![3b0f3dcbc98476aac9be69c6ca64bd6e8d8ebbccacea296a1bce2f465394b900.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/3b0f3dcbc98476aac9be69c6ca64bd6e8d8ebbccacea296a1bce2f465394b900.jpg)

![3ba9fffd63ffa74ff47c307d5d6b55406d4e9d793379bad95633fc32dc5997ad.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/3ba9fffd63ffa74ff47c307d5d6b55406d4e9d793379bad95633fc32dc5997ad.jpg)

![42f9201f70dd387d763ae1fc662775beb3205e5705394e1627cdc7dc0177830f.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/42f9201f70dd387d763ae1fc662775beb3205e5705394e1627cdc7dc0177830f.jpg)

![883422a4a6eed46fdfd80c1a11d35dd99be6d29905df25b2191f75420216ff1c.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/883422a4a6eed46fdfd80c1a11d35dd99be6d29905df25b2191f75420216ff1c.jpg)

![89aa5f2413285bd7149dcfbaccdb6ce80fdcb2b0993c14708bdf2dbea2d1e233.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/89aa5f2413285bd7149dcfbaccdb6ce80fdcb2b0993c14708bdf2dbea2d1e233.jpg)

![8ce2200b50edbdd3a2c20c3f4cd8b3231e2bd5428f3efd0d1bf81ae8bc3308e2.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/8ce2200b50edbdd3a2c20c3f4cd8b3231e2bd5428f3efd0d1bf81ae8bc3308e2.jpg)

![985203f23b70755105304f26c3fe160899cc631a7e180cdcf313a7a83360b898.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/985203f23b70755105304f26c3fe160899cc631a7e180cdcf313a7a83360b898.jpg)

![9f08272ebc16372c7d6f54717d437b7951d1e6cd5d5c22662d93da6940cbe734.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/9f08272ebc16372c7d6f54717d437b7951d1e6cd5d5c22662d93da6940cbe734.jpg)

![a77ae00f8a096521d1f8dafac69eab06884b8a76c42c9a929a3a78868ef1a345.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/a77ae00f8a096521d1f8dafac69eab06884b8a76c42c9a929a3a78868ef1a345.jpg)

![b8b206f142c02d95c03ecd34a5a92e0db26ae9902059ca029967d0d2c53cf20b.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/b8b206f142c02d95c03ecd34a5a92e0db26ae9902059ca029967d0d2c53cf20b.jpg)

![bf6639ad425a2b2574b7b36ecca65a65d002eebff8f228e4c3a3841f3a0da414.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/bf6639ad425a2b2574b7b36ecca65a65d002eebff8f228e4c3a3841f3a0da414.jpg)

![c843f6684d9e853673ffa478f108d5be07af849922c4596dbf4a9afed7f144c8.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/c843f6684d9e853673ffa478f108d5be07af849922c4596dbf4a9afed7f144c8.jpg)

![e4aa78724a0edfeefe692f9d6d56abbe42fd951588cdfe1f3065b4658616d039.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/e4aa78724a0edfeefe692f9d6d56abbe42fd951588cdfe1f3065b4658616d039.jpg)

![e5ef1f7785e8a14de21088677427c21d0e7605ed1ce85f6e7b22da1f50121b32.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/e5ef1f7785e8a14de21088677427c21d0e7605ed1ce85f6e7b22da1f50121b32.jpg)

![f1e5cfafcd0bec1fe1d87f0dabe7a37f57f801f1ba145873345fdae5316820b1.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/f1e5cfafcd0bec1fe1d87f0dabe7a37f57f801f1ba145873345fdae5316820b1.jpg)

![fcd54fb97c1de207c9bed58ac1f151d9c06bba3425f7fc30aa511c5fe5a5a629.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/images/fcd54fb97c1de207c9bed58ac1f151d9c06bba3425f7fc30aa511c5fe5a5a629.jpg)

### Tables

![12b13de89f312a40d02bb3bde5f678ab5b1981bc0f615405817bb22c4652d491.jpg](../icml_results/6066_Random%20Scaling%20of%20Emergent%20Capabilities/tables/12b13de89f312a40d02bb3bde5f678ab5b1981bc0f615405817bb22c4652d491.jpg)

<a id="paper-6017"></a>

## On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs


### Images

![080d02383c3fa98cf07b36a9e19a76add20d8bfdce6d1d7faac98fedfddfb27a.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/080d02383c3fa98cf07b36a9e19a76add20d8bfdce6d1d7faac98fedfddfb27a.jpg)

![12ae7382ceb1560b8741b9a093ce6e4fbdcf9291ac5fe6211724c876b1fd667d.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/12ae7382ceb1560b8741b9a093ce6e4fbdcf9291ac5fe6211724c876b1fd667d.jpg)

![158f01562db93690fad3406e57e716b8c5b761af3716da369079b823b30a24dc.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/158f01562db93690fad3406e57e716b8c5b761af3716da369079b823b30a24dc.jpg)

![1bc90ac8f991c0b3bbd4b0c8b5c78801490ab3dcad07d86d702f3f9c660ece46.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/1bc90ac8f991c0b3bbd4b0c8b5c78801490ab3dcad07d86d702f3f9c660ece46.jpg)

![1de7ee0bcce7683b6292849efc6d6c9bba3a84ee80c3f9e5f6eb02db6be9048d.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/1de7ee0bcce7683b6292849efc6d6c9bba3a84ee80c3f9e5f6eb02db6be9048d.jpg)

![3894d6cdd21f4666e547a9a26c2962111c346662a0750dab5be3194d970bc7c0.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/3894d6cdd21f4666e547a9a26c2962111c346662a0750dab5be3194d970bc7c0.jpg)

![3b92b468e3feb9f3117144363663ddc88ca6b23e0feea6a9a28ef328f78862d2.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/3b92b468e3feb9f3117144363663ddc88ca6b23e0feea6a9a28ef328f78862d2.jpg)

![4784def5b7ab75fcf833549c5a7e0a051704f0cf99d2261d1efcbff423ffda0a.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/4784def5b7ab75fcf833549c5a7e0a051704f0cf99d2261d1efcbff423ffda0a.jpg)

![47d7db5963d5c3f1c7172bd68f0b6d8ee12b1eab017ae612ce8825689cbdf732.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/47d7db5963d5c3f1c7172bd68f0b6d8ee12b1eab017ae612ce8825689cbdf732.jpg)

![483446de3a06ec7344aed9b0e46e0fe1fed1c06a992788411be38a8f47f26a26.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/483446de3a06ec7344aed9b0e46e0fe1fed1c06a992788411be38a8f47f26a26.jpg)

![4e2207d0f58f2f4fbc232194ec63422d4779b155c9dcba8a0cf5c02cfff2d235.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/4e2207d0f58f2f4fbc232194ec63422d4779b155c9dcba8a0cf5c02cfff2d235.jpg)

![4eed594628354bbc834b5b0e2abb22ec89ffbb059216f66412c6ddb478acd239.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/4eed594628354bbc834b5b0e2abb22ec89ffbb059216f66412c6ddb478acd239.jpg)

![5a76264375f6283513dc97479e8c8bab584ea9786e61450202b7d0dc071864a6.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/5a76264375f6283513dc97479e8c8bab584ea9786e61450202b7d0dc071864a6.jpg)

![5e3ca8b4c09be022aafa8e10cf7799577db7460e92398260fb4a8eed35ef10ba.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/5e3ca8b4c09be022aafa8e10cf7799577db7460e92398260fb4a8eed35ef10ba.jpg)

![61feba4fff1a9691704674cbba9a9ed69562dbf9106a4f2177acf78617d0df4a.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/61feba4fff1a9691704674cbba9a9ed69562dbf9106a4f2177acf78617d0df4a.jpg)

![664e08c595ec81d21d7c6d4f579118e7bd53f48e5416a3d5b7ea467ec8165101.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/664e08c595ec81d21d7c6d4f579118e7bd53f48e5416a3d5b7ea467ec8165101.jpg)

![67bc42696b47d2dabf3e1d03359d84950611d0ca3b1d20c54febe8d709105cba.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/67bc42696b47d2dabf3e1d03359d84950611d0ca3b1d20c54febe8d709105cba.jpg)

![75012f3b816bfcece3722aa43490d78bcb25f84daa53bfd8f5638e7cbf9849cb.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/75012f3b816bfcece3722aa43490d78bcb25f84daa53bfd8f5638e7cbf9849cb.jpg)

![755c6d1e3fa8f511c0097a8432d64e94f5c66f5ae01e25cd10b5ddb4d81d0560.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/755c6d1e3fa8f511c0097a8432d64e94f5c66f5ae01e25cd10b5ddb4d81d0560.jpg)

![7b7188b69a8ba8a2400344a3b0408190c9c8edc97915e18256a0b8490b37f078.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/7b7188b69a8ba8a2400344a3b0408190c9c8edc97915e18256a0b8490b37f078.jpg)

![8dcf037792dd89a24ffd53711c11581db45cb37a318e83d66294cab52df034bc.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/8dcf037792dd89a24ffd53711c11581db45cb37a318e83d66294cab52df034bc.jpg)

![923ccd63c9101e48385219df713a6c2a24474cb277e64db128c532f8ef775bd6.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/923ccd63c9101e48385219df713a6c2a24474cb277e64db128c532f8ef775bd6.jpg)

![b753da59fd1b98dca598fea7ad8911dc6fc130ece834771341db55cd777ad9a6.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/b753da59fd1b98dca598fea7ad8911dc6fc130ece834771341db55cd777ad9a6.jpg)

![bdc70f2260fe580d5562b71c449cba5af743644fc20354110c0cab8cf6f0a9d1.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/bdc70f2260fe580d5562b71c449cba5af743644fc20354110c0cab8cf6f0a9d1.jpg)

![bfacd71322329b7d32d66acc9b2057d588092e2467f6253a5c5f559ae7b405e6.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/bfacd71322329b7d32d66acc9b2057d588092e2467f6253a5c5f559ae7b405e6.jpg)

![c8ef433d13b3ac33449093b338917f1ea0634c037d92bd4a07ef760d6a6dfa8f.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/c8ef433d13b3ac33449093b338917f1ea0634c037d92bd4a07ef760d6a6dfa8f.jpg)

![c97f8a351a5d6d6dfe3350c445db0dba52ea21ff9ed4ff7bbc105a5e10c8ca2c.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/c97f8a351a5d6d6dfe3350c445db0dba52ea21ff9ed4ff7bbc105a5e10c8ca2c.jpg)

![cb8c5a633325b2c8f0fcb670c7126b06fac83cceb822ae98181d50c4d77e567a.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/cb8c5a633325b2c8f0fcb670c7126b06fac83cceb822ae98181d50c4d77e567a.jpg)

![d7203dda97ed7c8299959d3269349f6a1064d8fb8f2f8c5d7c48c6e6c0ca3ae6.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/d7203dda97ed7c8299959d3269349f6a1064d8fb8f2f8c5d7c48c6e6c0ca3ae6.jpg)

![e3a015fc27eaae338f432fae0fd7268339e1d59af308b4a7ecfc02920b475056.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/e3a015fc27eaae338f432fae0fd7268339e1d59af308b4a7ecfc02920b475056.jpg)

![f3335f078f28ef62550506305cc36befc2deb12b2f5c03f7480339cb8152f13b.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/f3335f078f28ef62550506305cc36befc2deb12b2f5c03f7480339cb8152f13b.jpg)

![faf627a3a0423478c7b44c326371cd519df4e9a5ede310396381b5ec5c26791b.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/faf627a3a0423478c7b44c326371cd519df4e9a5ede310396381b5ec5c26791b.jpg)

![fb337f1bc7a26829237e200e658114306279b16c169bdb922f2fcdbd3a56bb05.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/images/fb337f1bc7a26829237e200e658114306279b16c169bdb922f2fcdbd3a56bb05.jpg)

### Tables

![3e6af7c391611932a5a1e1878edfc124e79bc49f6e141e3f82c038d63d169a92.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/3e6af7c391611932a5a1e1878edfc124e79bc49f6e141e3f82c038d63d169a92.jpg)

![4f328da84fecea4c3d555380ca18e008ec5d099556718dd480e294af6d711bdb.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/4f328da84fecea4c3d555380ca18e008ec5d099556718dd480e294af6d711bdb.jpg)

![60f007303645c794e208f3496a99bb16a178230b8b4c68de0ec0b26f2d0785b4.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/60f007303645c794e208f3496a99bb16a178230b8b4c68de0ec0b26f2d0785b4.jpg)

![6225377c445794ebab19b05413044b022b5cf827e151912eeeaade2c25fda137.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/6225377c445794ebab19b05413044b022b5cf827e151912eeeaade2c25fda137.jpg)

![629754f0f6bd6244a932647034e307c03b57abbe73a00af3cb833e754b153cf9.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/629754f0f6bd6244a932647034e307c03b57abbe73a00af3cb833e754b153cf9.jpg)

![686850daafb04b0e75c724e8390c401d4b1819c935959b75258cf67907849a3a.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/686850daafb04b0e75c724e8390c401d4b1819c935959b75258cf67907849a3a.jpg)

![69670d1f66c59dca4e8065c8091a6d24084c5596b2aaf3da9a936e9b33d06bf0.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/69670d1f66c59dca4e8065c8091a6d24084c5596b2aaf3da9a936e9b33d06bf0.jpg)

![81135da48257bad74c2ff6adb904bdf93e717530e76f4996e934718df69155a1.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/81135da48257bad74c2ff6adb904bdf93e717530e76f4996e934718df69155a1.jpg)

![8d08b76d34ad56204249d9f3bb9ad308a9f4ccea3137e151f99b1b65251c8dfc.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/8d08b76d34ad56204249d9f3bb9ad308a9f4ccea3137e151f99b1b65251c8dfc.jpg)

![cc9ac0d535c4dc362563969ef15e6564cafb401c2cd1c05de82533adb78bd401.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/cc9ac0d535c4dc362563969ef15e6564cafb401c2cd1c05de82533adb78bd401.jpg)

![d4610f8fb036eba0f9470310723bc60fd30820e374dcdf66660374f90136fd4b.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/d4610f8fb036eba0f9470310723bc60fd30820e374dcdf66660374f90136fd4b.jpg)

![d74b77bdfcd93e931a19fc80a8e96737356976d021d4984bd9b35fe32ac90b4e.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/d74b77bdfcd93e931a19fc80a8e96737356976d021d4984bd9b35fe32ac90b4e.jpg)

![dbf24133f89fe0220d966e979fd8bb9787fb1d189fc50b13e877d3f69c60adaf.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/dbf24133f89fe0220d966e979fd8bb9787fb1d189fc50b13e877d3f69c60adaf.jpg)

![f838ec207ab667b35c23dc4fd85cc04df0d531fd18159868bd4e92e4ef3ebb15.jpg](../icml_results/6067_On%20Robustness%20and%20Chain-of-Thought%20Consistency%20of%20RL-Finetuned%20VLMs/tables/f838ec207ab667b35c23dc4fd85cc04df0d531fd18159868bd4e92e4ef3ebb15.jpg)

<a id="paper-6018"></a>

## Does AI Reviewer See the Full Picture? Attacking and Defending Multimodal Peer Review


### Images

![8584100ca67d633820206d2165ff6d977374ce685237bec8db00a658ad39b4f4.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/images/8584100ca67d633820206d2165ff6d977374ce685237bec8db00a658ad39b4f4.jpg)

![a07fcaf2a168cf99eb545a0e22cda55cb4cebf4cbf1c7b72c98f09064e411937.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/images/a07fcaf2a168cf99eb545a0e22cda55cb4cebf4cbf1c7b72c98f09064e411937.jpg)

![cee12d996b24ee5deebd6c71c980f9c536e58e7b8b3fc9a8741b4e30215e9d95.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/images/cee12d996b24ee5deebd6c71c980f9c536e58e7b8b3fc9a8741b4e30215e9d95.jpg)

### Tables

![092e9dbc4d23941cdb025dab7599753d430744eadc1f3be22f8cf2b0196153dd.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/092e9dbc4d23941cdb025dab7599753d430744eadc1f3be22f8cf2b0196153dd.jpg)

![0fe9c1dddb01b44bedf796b3abc8f31aef5675556691f207d487c81b81eeb507.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/0fe9c1dddb01b44bedf796b3abc8f31aef5675556691f207d487c81b81eeb507.jpg)

![2a98c4be29273700ae1c8e4973ad02b65a1cbd83a888f179376fcc8275d9b106.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/2a98c4be29273700ae1c8e4973ad02b65a1cbd83a888f179376fcc8275d9b106.jpg)

![4044e1f6ab435734be6aac60fbe1f7cec8e532eb9047dfa9792313450408bd41.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/4044e1f6ab435734be6aac60fbe1f7cec8e532eb9047dfa9792313450408bd41.jpg)

![5b86ed8d2ae8b3b8db72eafa14eef8ecbbe6f123dfbf3f7edcc06e1ca64d7000.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/5b86ed8d2ae8b3b8db72eafa14eef8ecbbe6f123dfbf3f7edcc06e1ca64d7000.jpg)

![766395fc1e4868606afd4029560e7e93ae3f4a447615f501198395838c9d095f.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/766395fc1e4868606afd4029560e7e93ae3f4a447615f501198395838c9d095f.jpg)

![816b550ec18b733c43e8176bdf64c5e3e33baa1fa2f5a5b928103d6c36633fd5.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/816b550ec18b733c43e8176bdf64c5e3e33baa1fa2f5a5b928103d6c36633fd5.jpg)

![87ad71c703de17561f74a0068d73acb69afb0a77c8969019d5508b38be237b7d.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/87ad71c703de17561f74a0068d73acb69afb0a77c8969019d5508b38be237b7d.jpg)

![b2ecd31d08a5ae05c5912f579ea1583634aa1bbe1a17932ce299d37d8cd4f5be.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/b2ecd31d08a5ae05c5912f579ea1583634aa1bbe1a17932ce299d37d8cd4f5be.jpg)

![c06d6c3516c4d6f474c8106797bd4733c7353371d3a4dfd8affba4410e167a07.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/c06d6c3516c4d6f474c8106797bd4733c7353371d3a4dfd8affba4410e167a07.jpg)

![f0b2f339e296ad8c675e16f8d6d3fee0a1441886d39bc4acb2903d1331ba2b3b.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/f0b2f339e296ad8c675e16f8d6d3fee0a1441886d39bc4acb2903d1331ba2b3b.jpg)

![f4bff8ecdf03f11d9edbe015c97ff23bbac01d3341564f8fe7f76ad571671c09.jpg](../icml_results/6068_Does%20AI%20Reviewer%20See%20the%20Full%20Picture_%20Attacking%20and%20Defending%20Multimodal%20Peer%20Review/tables/f4bff8ecdf03f11d9edbe015c97ff23bbac01d3341564f8fe7f76ad571671c09.jpg)

<a id="paper-6019"></a>

## Exploring 3D Dataset Pruning


### Images

![25f98a127008692ad5d34719391ae47c8df8f6cc06cd0d604bcdb4702ed5e661.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/25f98a127008692ad5d34719391ae47c8df8f6cc06cd0d604bcdb4702ed5e661.jpg)

![597adf6130280ced84543af1c2bff73695b4b03139c6b1e49f94087f2b20e9ce.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/597adf6130280ced84543af1c2bff73695b4b03139c6b1e49f94087f2b20e9ce.jpg)

![8bd36aea55ed34e1dc642c4936091bfdda5aca85c16ce63d7c558c0c5d3fef49.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/8bd36aea55ed34e1dc642c4936091bfdda5aca85c16ce63d7c558c0c5d3fef49.jpg)

![c820199292072cbeb7e600ec01dca74753818d64f744439b59b4f9c543115f3b.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/c820199292072cbeb7e600ec01dca74753818d64f744439b59b4f9c543115f3b.jpg)

![cb62ffeba7ed30a24978c96ea5d5eddf3909028e1fba0479496f49800cd3080e.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/cb62ffeba7ed30a24978c96ea5d5eddf3909028e1fba0479496f49800cd3080e.jpg)

![cdd597e297f9dd00bb0d86b9dc0801192954e16d0a5be16f5ecbb00ac2235a50.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/cdd597e297f9dd00bb0d86b9dc0801192954e16d0a5be16f5ecbb00ac2235a50.jpg)

![cfebb95e29fda84f12f4e78c413c116c3b46c25abf317de3126d1a678628b3a4.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/cfebb95e29fda84f12f4e78c413c116c3b46c25abf317de3126d1a678628b3a4.jpg)

![d7530b2b2a961b9413103f56e30dcecf45f32a8bfd9c393c1de424535254ff61.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/d7530b2b2a961b9413103f56e30dcecf45f32a8bfd9c393c1de424535254ff61.jpg)

![dd18954155cdd637b3a32642ec4742dd7f51a5f1016ad9ef7cbd06a667116d44.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/images/dd18954155cdd637b3a32642ec4742dd7f51a5f1016ad9ef7cbd06a667116d44.jpg)

### Tables

![045e0d81efb1619fdc6c584cd2f3ef476fe1a4721971e4403948cf5ff6987e49.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/045e0d81efb1619fdc6c584cd2f3ef476fe1a4721971e4403948cf5ff6987e49.jpg)

![17247458d2d15247b6f8bdd81fa11763d58a56321ca1c9bed39dcaefcfd3a9f8.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/17247458d2d15247b6f8bdd81fa11763d58a56321ca1c9bed39dcaefcfd3a9f8.jpg)

![44d782517fc1b0e70bd1925cb7f0964c0e165fa90cb3d462624d46193d90cb44.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/44d782517fc1b0e70bd1925cb7f0964c0e165fa90cb3d462624d46193d90cb44.jpg)

![7aede57ce0c40053a6f4aa48cdfbbaa6845097f9a8999cd8d130c91a93c9dcb1.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/7aede57ce0c40053a6f4aa48cdfbbaa6845097f9a8999cd8d130c91a93c9dcb1.jpg)

![94cdc1d26768e4f1eb9dbe478fdb7e5edac72ef91fc7e03f6dbfb612dc5acd12.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/94cdc1d26768e4f1eb9dbe478fdb7e5edac72ef91fc7e03f6dbfb612dc5acd12.jpg)

![c096903756e275cc16157798c0e276f1c527f649126e2b4ffa68d2252ece8f8c.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/c096903756e275cc16157798c0e276f1c527f649126e2b4ffa68d2252ece8f8c.jpg)

![c9576bcb55e0ccb8655fb4fb5ad01d91512d7491d14f6590de65fde41616cc21.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/c9576bcb55e0ccb8655fb4fb5ad01d91512d7491d14f6590de65fde41616cc21.jpg)

![ed691aa4b9d7fc9254378bf42b1265a6836fda7e5f520e9ab7547460a0e666d8.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/ed691aa4b9d7fc9254378bf42b1265a6836fda7e5f520e9ab7547460a0e666d8.jpg)

![f4397d73d07c8d91158b7af9d06a896e88ce3e13837b8a031654b02ea0f6443a.jpg](../icml_results/6069_Exploring%203D%20Dataset%20Pruning/tables/f4397d73d07c8d91158b7af9d06a896e88ce3e13837b8a031654b02ea0f6443a.jpg)

<a id="paper-6020"></a>

## ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts


### Images

![0ff317c90917d64644878dfe669728403452b995e1a57adba48b8a97588f6333.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/0ff317c90917d64644878dfe669728403452b995e1a57adba48b8a97588f6333.jpg)

![312843e06a5990eb8b70b759c3d05e510e573b9f4dfd7bad35fb53cd2e79db18.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/312843e06a5990eb8b70b759c3d05e510e573b9f4dfd7bad35fb53cd2e79db18.jpg)

![465e297c3f536519aaf6fc22f65e6a23560550816260fcf5ff2b0c823fd3b59e.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/465e297c3f536519aaf6fc22f65e6a23560550816260fcf5ff2b0c823fd3b59e.jpg)

![9389c0160e83ed0befe96aa9ab3cf7639f9101d84c82abc9dbe2d0b3057f6edd.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/9389c0160e83ed0befe96aa9ab3cf7639f9101d84c82abc9dbe2d0b3057f6edd.jpg)

![98ef62a65e85f8b648d0211e10dfeb9d7a7972886fd344408f12d03bc5eaf8f9.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/98ef62a65e85f8b648d0211e10dfeb9d7a7972886fd344408f12d03bc5eaf8f9.jpg)

![c14149ebc535165e96f44612421a3dc41af6dcd7729a2a0fd72eda844439d04f.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/c14149ebc535165e96f44612421a3dc41af6dcd7729a2a0fd72eda844439d04f.jpg)

![e34765537143363368b37f61b173006c8fb2924e6f7d1551994f334e553603f2.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/images/e34765537143363368b37f61b173006c8fb2924e6f7d1551994f334e553603f2.jpg)

### Tables

![2185ff32b749d657883246ec4cee1418b22cc0805e703db52109264ac9c4476c.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/2185ff32b749d657883246ec4cee1418b22cc0805e703db52109264ac9c4476c.jpg)

![3997893012d9f9703bec68949479ace93766cd568fb434c939b094c8434a9266.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/3997893012d9f9703bec68949479ace93766cd568fb434c939b094c8434a9266.jpg)

![540f3a498a797a1e48d218f4adcbd68f99117a6ffcdf9cd9e1be9dfb91452da8.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/540f3a498a797a1e48d218f4adcbd68f99117a6ffcdf9cd9e1be9dfb91452da8.jpg)

![589d22abd621d54233e839a60e2e7820602ef0ce6c108bb0f154faffdff2853a.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/589d22abd621d54233e839a60e2e7820602ef0ce6c108bb0f154faffdff2853a.jpg)

![77c4bc68255d075cbab849a3d5c2c9ab93a9f3617dfdcd609e7c5dc754a333de.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/77c4bc68255d075cbab849a3d5c2c9ab93a9f3617dfdcd609e7c5dc754a333de.jpg)

![7b5d6cce5ae02b6f2034b739556e3f0531ac29ac42dbb0520230828c2f668217.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/7b5d6cce5ae02b6f2034b739556e3f0531ac29ac42dbb0520230828c2f668217.jpg)

![9dd3b7c224161b5270b115ee945d2a82b2e8f4cf4460a3605891b74f640c927c.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/9dd3b7c224161b5270b115ee945d2a82b2e8f4cf4460a3605891b74f640c927c.jpg)

![b13dcf18c1b6b497f72efa62790015c8948b1860776e8fc54d9eff0fce5f67c1.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/b13dcf18c1b6b497f72efa62790015c8948b1860776e8fc54d9eff0fce5f67c1.jpg)

![c121cc62784093dfd5eae6f0fe594033d3d7939b513edab3928051eb842077ca.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/c121cc62784093dfd5eae6f0fe594033d3d7939b513edab3928051eb842077ca.jpg)

![f8006dc43ac132a65be00800035a62184b4f28ea7f14af621d023aaf19ea39b5.jpg](../icml_results/6070_ProbMoE_%20Differentiable%20Probabilistic%20Routing%20for%20Mixture-of-Experts/tables/f8006dc43ac132a65be00800035a62184b4f28ea7f14af621d023aaf19ea39b5.jpg)

<a id="paper-6021"></a>

## CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation


### Images

![0a3af6a5c16a068db6ffb8f5c35f3c92447627a4f86768d99fb941fbdb161430.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/0a3af6a5c16a068db6ffb8f5c35f3c92447627a4f86768d99fb941fbdb161430.jpg)

![35286f020294a0925b65658101837f66c5a73948f9f3888edfa58bda96ed41da.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/35286f020294a0925b65658101837f66c5a73948f9f3888edfa58bda96ed41da.jpg)

![689bd3d090e8588523b110d14be05875344e08edfc6930f1e33dc177a0ecb008.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/689bd3d090e8588523b110d14be05875344e08edfc6930f1e33dc177a0ecb008.jpg)

![870a2519e270a1f845f81c34e7338b4310e46118d813d4a25ef61d393c22aa2d.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/870a2519e270a1f845f81c34e7338b4310e46118d813d4a25ef61d393c22aa2d.jpg)

![b2287a90bfd49e496e4d7217b076d5212e3dcbe72e1d9c3e859f1c928da5e09d.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/b2287a90bfd49e496e4d7217b076d5212e3dcbe72e1d9c3e859f1c928da5e09d.jpg)

![d9c41e35a6ca9b2278f36653e9ad8c42e97a4feed4de0c9bd943637eccf01095.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/d9c41e35a6ca9b2278f36653e9ad8c42e97a4feed4de0c9bd943637eccf01095.jpg)

![de6f8d8278355cedd626461e1aee5c298fe5b786353f817e504708ba9b413c18.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/de6f8d8278355cedd626461e1aee5c298fe5b786353f817e504708ba9b413c18.jpg)

![e2a6513c0f80cf50345999f14beb58f374553ae069a7d72edc3faef37d6881fd.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/e2a6513c0f80cf50345999f14beb58f374553ae069a7d72edc3faef37d6881fd.jpg)

![fd3f072ba08f1598e53c4dbff1773963cd3d37ba183dab87fff5cc43c496b5a3.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/images/fd3f072ba08f1598e53c4dbff1773963cd3d37ba183dab87fff5cc43c496b5a3.jpg)

### Tables

![1fc8ed179d3fcb2c70d608ca40753211064769a868210db6a2825c01a1a63451.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/1fc8ed179d3fcb2c70d608ca40753211064769a868210db6a2825c01a1a63451.jpg)

![4453fe20292b597be8185767a9d018afe801593faf24ca182120abe9d7c50e47.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/4453fe20292b597be8185767a9d018afe801593faf24ca182120abe9d7c50e47.jpg)

![7a4e8ce0bcfc147d4da24ac49091c4c58fbaa42238fa81c926fb18b4815e306d.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/7a4e8ce0bcfc147d4da24ac49091c4c58fbaa42238fa81c926fb18b4815e306d.jpg)

![9244d5cdfe1aa1e89cb3574270ff99f5e3efae6e0c90052c22ab927f598b6993.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/9244d5cdfe1aa1e89cb3574270ff99f5e3efae6e0c90052c22ab927f598b6993.jpg)

![9ffa3e82071408e1747e17f4518f3d75c068f12975878e6bd114886be8270aaa.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/9ffa3e82071408e1747e17f4518f3d75c068f12975878e6bd114886be8270aaa.jpg)

![e07524338ed9f33753ba2a92bd6cede4905cb83e952696f218623efd8fe73e79.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/e07524338ed9f33753ba2a92bd6cede4905cb83e952696f218623efd8fe73e79.jpg)

![e6a94beb5743fb6906e2d68db1f6d415efc066cc8ab0a50a70ed508d1b5e8f45.jpg](../icml_results/6071_CARE_%20Confounder-Aware%20Aggregation%20for%20Reliable%20LLM%20Evaluation/tables/e6a94beb5743fb6906e2d68db1f6d415efc066cc8ab0a50a70ed508d1b5e8f45.jpg)

<a id="paper-6022"></a>

## Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding


### Images

![57217645268a5021f126d025f68e37399be95d8ee14e1e5378ccebf12e64677d.jpg](../icml_results/6072_Regret%20Pre-training_%20Bridging%20Prior%20and%20Posterior%20Views%20for%20Enhanced%20Knowledge%20Grounding/images/57217645268a5021f126d025f68e37399be95d8ee14e1e5378ccebf12e64677d.jpg)

![8bd1f2ba48543c27269dda323cc12095853460d830a34d99017e142bb727163c.jpg](../icml_results/6072_Regret%20Pre-training_%20Bridging%20Prior%20and%20Posterior%20Views%20for%20Enhanced%20Knowledge%20Grounding/images/8bd1f2ba48543c27269dda323cc12095853460d830a34d99017e142bb727163c.jpg)

![f273795b931406724a88f6bba65da1e57e5c27865fc747785c5071e791c49996.jpg](../icml_results/6072_Regret%20Pre-training_%20Bridging%20Prior%20and%20Posterior%20Views%20for%20Enhanced%20Knowledge%20Grounding/images/f273795b931406724a88f6bba65da1e57e5c27865fc747785c5071e791c49996.jpg)

### Tables

![d4a9af6524e81de07be942d8c160e4b4a3351f716b650533811a215a0c458842.jpg](../icml_results/6072_Regret%20Pre-training_%20Bridging%20Prior%20and%20Posterior%20Views%20for%20Enhanced%20Knowledge%20Grounding/tables/d4a9af6524e81de07be942d8c160e4b4a3351f716b650533811a215a0c458842.jpg)

![f2f3713a5e45543c49609dfd85b5b24467fd9b29f511d91f903d41541234e289.jpg](../icml_results/6072_Regret%20Pre-training_%20Bridging%20Prior%20and%20Posterior%20Views%20for%20Enhanced%20Knowledge%20Grounding/tables/f2f3713a5e45543c49609dfd85b5b24467fd9b29f511d91f903d41541234e289.jpg)

<a id="paper-6023"></a>

## You Don’t Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models


### Images

![17f0b2db0f603bb0531a822ec376aa6a8a80b3a1fda44a0154d5f3086edf0631.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/17f0b2db0f603bb0531a822ec376aa6a8a80b3a1fda44a0154d5f3086edf0631.jpg)

![1b29ca51caec04658f2a01521c35b38b024e9ee949eae097753c0d0657a941d5.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/1b29ca51caec04658f2a01521c35b38b024e9ee949eae097753c0d0657a941d5.jpg)

![22a346cc272c8bd6d719fc2165f85e638b33e7f60a6d636636d33b3f566463b2.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/22a346cc272c8bd6d719fc2165f85e638b33e7f60a6d636636d33b3f566463b2.jpg)

![2444e600d5f581e4b470b21192e6dc35acc9aeec29821f7939b4223acf3c94e6.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/2444e600d5f581e4b470b21192e6dc35acc9aeec29821f7939b4223acf3c94e6.jpg)

![49334f78adc836df8b33acfa7256e2d94c8dcc2f09fd0ecb41df5fa1a836faa5.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/49334f78adc836df8b33acfa7256e2d94c8dcc2f09fd0ecb41df5fa1a836faa5.jpg)

![4d7d110207cf20bec2f6c5a13ca0d744b920af1106876fc0f60a698715a7bdda.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/4d7d110207cf20bec2f6c5a13ca0d744b920af1106876fc0f60a698715a7bdda.jpg)

![5f7d74333342bdd20bb27132efade711f29ac212d99fbaee1dd13a212f39d2a0.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/5f7d74333342bdd20bb27132efade711f29ac212d99fbaee1dd13a212f39d2a0.jpg)

![5fbc2c290ebedaa3ca5f68816682ebaed34a8fbf2a7e381be7e23da3eb24f17e.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/5fbc2c290ebedaa3ca5f68816682ebaed34a8fbf2a7e381be7e23da3eb24f17e.jpg)

![7b35c5cc7a41a189881af1a90d4eb4249541dbab8d7c39d617d8f9cc27bbfbae.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/7b35c5cc7a41a189881af1a90d4eb4249541dbab8d7c39d617d8f9cc27bbfbae.jpg)

![8bf5b3e389e5879fc37b466345796ca564fdcd614a4c2c00c85689598756dea7.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/8bf5b3e389e5879fc37b466345796ca564fdcd614a4c2c00c85689598756dea7.jpg)

![a698ba43dc32c5739ea5e0050f28f4bef375def2859361ce7e758e97f2ecd7d2.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/a698ba43dc32c5739ea5e0050f28f4bef375def2859361ce7e758e97f2ecd7d2.jpg)

![aa18920dfac0f3ed7a64fd794c6dfddb5dbb5bd1d2c2a72a6c5a49085cd85d0d.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/aa18920dfac0f3ed7a64fd794c6dfddb5dbb5bd1d2c2a72a6c5a49085cd85d0d.jpg)

![d19d2df5915b5189589864e56e866a21c7044a981f6dbc0749113b9b2f018705.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/d19d2df5915b5189589864e56e866a21c7044a981f6dbc0749113b9b2f018705.jpg)

![d32276725f4be0b010165182a5d70526833df622956f9b04016999bf571b2f05.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/images/d32276725f4be0b010165182a5d70526833df622956f9b04016999bf571b2f05.jpg)

### Tables

![0311cde418541815a5d45e06938ea3dba8c53f339a90d96ced1ec602460af1fd.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/0311cde418541815a5d45e06938ea3dba8c53f339a90d96ced1ec602460af1fd.jpg)

![04eb3f1a2d3bcf1dddafdc57fae0262cf2cb5c1f6a1edf74fd6b9539d798e1ae.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/04eb3f1a2d3bcf1dddafdc57fae0262cf2cb5c1f6a1edf74fd6b9539d798e1ae.jpg)

![13dc95ff216302a02c9e1746d5cc06ed2060e700186574faa23b35ac76a92ac3.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/13dc95ff216302a02c9e1746d5cc06ed2060e700186574faa23b35ac76a92ac3.jpg)

![7cf6e844e46e1f53c694aac6e10783943065f8a76ed3b9c5eba221633fdd469e.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/7cf6e844e46e1f53c694aac6e10783943065f8a76ed3b9c5eba221633fdd469e.jpg)

![8cb7f6ae670e4887d875d1422e6778290006b37acb9d4af684af98ee80677547.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/8cb7f6ae670e4887d875d1422e6778290006b37acb9d4af684af98ee80677547.jpg)

![97448d8d8af7289f6edbdeb4581a08db3d27f0c761eb3882909c6713727df295.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/97448d8d8af7289f6edbdeb4581a08db3d27f0c761eb3882909c6713727df295.jpg)

![9e80c72ba08a33f28a14f94f0600ac0d5c9950473599628fa6b9d9957b2fe28d.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/9e80c72ba08a33f28a14f94f0600ac0d5c9950473599628fa6b9d9957b2fe28d.jpg)

![aa14b5925500a957e5b5bc84f3e9b829f2beaa5ce5c2c81a5a325676d45d8bcf.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/aa14b5925500a957e5b5bc84f3e9b829f2beaa5ce5c2c81a5a325676d45d8bcf.jpg)

![c63ea805415e309aff64b4fe16e38e7edf196e12e7483dc250cf55c5e69b0195.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/c63ea805415e309aff64b4fe16e38e7edf196e12e7483dc250cf55c5e69b0195.jpg)

![d022e798622165f680371d6a245aaabdf55b5a1ab80f69235f90f458549c96f5.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/d022e798622165f680371d6a245aaabdf55b5a1ab80f69235f90f458549c96f5.jpg)

![fb5f916a25ea593735dc9b1f3adb87d6a4a76a10a272ffc846b6257095897905.jpg](../icml_results/6073_You%20Don%E2%80%99t%20Need%20All%20That%20Attention_%20Surgical%20Memorization%20Mitigation%20in%20Text-to-Image%20Diffusion%20Model/tables/fb5f916a25ea593735dc9b1f3adb87d6a4a76a10a272ffc846b6257095897905.jpg)

<a id="paper-6024"></a>

## SIKA-GP: Accelerating Gaussian Process Inference with Sparse Inducing Kernel Approximations for Bayesian Deep Learning


### Images

![03add32b0a8aa69428ed5132fb9a3becb81ad92a1e3d88db44ed5b0912ae67f3.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/03add32b0a8aa69428ed5132fb9a3becb81ad92a1e3d88db44ed5b0912ae67f3.jpg)

![0bc1d95264d2753e9def7a8738398bf14432b61b6300bf88f0c38ef324d27a67.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/0bc1d95264d2753e9def7a8738398bf14432b61b6300bf88f0c38ef324d27a67.jpg)

![3035fe1ba39225743a426dfcb1f0e9ccba80e7a22d45433c135624823fcde3a6.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/3035fe1ba39225743a426dfcb1f0e9ccba80e7a22d45433c135624823fcde3a6.jpg)

![37bf99a80a1f213c1db2c5efd299913e228e97beba4ddb8ed26fdc91370b5046.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/37bf99a80a1f213c1db2c5efd299913e228e97beba4ddb8ed26fdc91370b5046.jpg)

![6a49499a31f078147c60774d1ee8d6911258d5ab45f5c5081e84e4f8a82a4430.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/6a49499a31f078147c60774d1ee8d6911258d5ab45f5c5081e84e4f8a82a4430.jpg)

![70e64291ea502bc6363f248e2c603d1652c845dd53836bcee8681bd35016df75.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/70e64291ea502bc6363f248e2c603d1652c845dd53836bcee8681bd35016df75.jpg)

![b2f835ff12507f385201783c7b51cac64472137be3220990e642f990b1b1be33.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/b2f835ff12507f385201783c7b51cac64472137be3220990e642f990b1b1be33.jpg)

![f6bab58700dccd60aa8efa29911c2b233051e3fb792f6fb65bed540b2400a869.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/images/f6bab58700dccd60aa8efa29911c2b233051e3fb792f6fb65bed540b2400a869.jpg)

### Tables

![266ed7e23b1224fba8df1828398b73fa58fa7c2e91294612b2461eeea1e7c5f5.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/266ed7e23b1224fba8df1828398b73fa58fa7c2e91294612b2461eeea1e7c5f5.jpg)

![32722cc4d48d0c8e5f21831ad4e98673c680576f6733282ceca2b97947bcd9f2.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/32722cc4d48d0c8e5f21831ad4e98673c680576f6733282ceca2b97947bcd9f2.jpg)

![50c00df745fe14f0c5bfb8c0f95e704c8963ea6485e64952c4f785d95a09e78a.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/50c00df745fe14f0c5bfb8c0f95e704c8963ea6485e64952c4f785d95a09e78a.jpg)

![a2e4845ccd30bd63a64e85f78d2a2e5353476d2cacda8277ca8a3a2641985ea5.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/a2e4845ccd30bd63a64e85f78d2a2e5353476d2cacda8277ca8a3a2641985ea5.jpg)

![a522091c52982b0862c5ef4037ebe259952271008bfe7f736a9ee06e4ecac867.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/a522091c52982b0862c5ef4037ebe259952271008bfe7f736a9ee06e4ecac867.jpg)

![ba6a03f1491b86e9bbdc1aa6ab10b02088de8460476393e9053448c75668a855.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/ba6a03f1491b86e9bbdc1aa6ab10b02088de8460476393e9053448c75668a855.jpg)

![c6ede3aad3118678a0465e6ffceb3eb7e4df32b6522e80429794e541e9656089.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/c6ede3aad3118678a0465e6ffceb3eb7e4df32b6522e80429794e541e9656089.jpg)

![ce624d721d46a960a4771c6e94dd68db4df7abd7b334d2ee70e93856f9949ca6.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/ce624d721d46a960a4771c6e94dd68db4df7abd7b334d2ee70e93856f9949ca6.jpg)

![d670a083be0bb1155b563abe31c56a25f7b4cc9381f416d70d9afb7fe8b20a0e.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/d670a083be0bb1155b563abe31c56a25f7b4cc9381f416d70d9afb7fe8b20a0e.jpg)

![d8c4fafc306093fe641d0ca3c7018d116e729b747eb5fce6565aa306031b7fa5.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/d8c4fafc306093fe641d0ca3c7018d116e729b747eb5fce6565aa306031b7fa5.jpg)

![e4d7443ad5fbc73a756c4eb7ee0f87253fb220bcbae716d0c93c47889eb6afdc.jpg](../icml_results/6074_SIKA-GP_%20Accelerating%20Gaussian%20Process%20Inference%20with%20Sparse%20Inducing%20Kernel%20Approximations%20for%20Baye/tables/e4d7443ad5fbc73a756c4eb7ee0f87253fb220bcbae716d0c93c47889eb6afdc.jpg)

<a id="paper-6025"></a>

## A Unified Framework for Deep Hypergraph Clustering Beyond Homophily


### Images

![1d9d587e70c358c142f255198313b901a62e59474993b15510f49deeb7e8801e.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/1d9d587e70c358c142f255198313b901a62e59474993b15510f49deeb7e8801e.jpg)

![4049920b415cba8652481c59290877e966697865d093a19be51dc234c65252f9.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/4049920b415cba8652481c59290877e966697865d093a19be51dc234c65252f9.jpg)

![6b9dd4e9dfba60f0c0cd2298def14db7e493f433e02e21ebd5d3156acfb949e6.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/6b9dd4e9dfba60f0c0cd2298def14db7e493f433e02e21ebd5d3156acfb949e6.jpg)

![701a033d776f29d0be90ae63f8da9cba043619f2943545b9d3f4b88399687f51.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/701a033d776f29d0be90ae63f8da9cba043619f2943545b9d3f4b88399687f51.jpg)

![7cb4d708d93c144c357a0daa9eea501c0a8808d06d140ec1333a62d46a16d409.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/7cb4d708d93c144c357a0daa9eea501c0a8808d06d140ec1333a62d46a16d409.jpg)

![85a740a9626f616e4dc6628a1206e9a958041e93a65a433f355828d4007b10cf.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/85a740a9626f616e4dc6628a1206e9a958041e93a65a433f355828d4007b10cf.jpg)

![a3fbe825f46ee5c3b6b65f3e2d65c44d7eda7f8b22ce62ba4ac5a3df473c66c1.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/images/a3fbe825f46ee5c3b6b65f3e2d65c44d7eda7f8b22ce62ba4ac5a3df473c66c1.jpg)

### Tables

![04467e4738f6454131bfa79eba90ec984deceb244483c2671b9293d81fb5a908.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/tables/04467e4738f6454131bfa79eba90ec984deceb244483c2671b9293d81fb5a908.jpg)

![0a11700f27438df627d9ea5b6477113aec6567bda0335d71dc0ab727a137f45e.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/tables/0a11700f27438df627d9ea5b6477113aec6567bda0335d71dc0ab727a137f45e.jpg)

![18af646f2c2033151362d876334ea1931d1856f4848746a8b02a9bd0da8b1833.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/tables/18af646f2c2033151362d876334ea1931d1856f4848746a8b02a9bd0da8b1833.jpg)

![87e805fa27b357d0476888582ad13027df52ee95caa151fb3b77b5aabd0a0da0.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/tables/87e805fa27b357d0476888582ad13027df52ee95caa151fb3b77b5aabd0a0da0.jpg)

![a4101b456dd7b3d80a51bb13d4841731a55da355618bfb0649bd26c4cd3eb213.jpg](../icml_results/6075_A%20Unified%20Framework%20for%20Deep%20Hypergraph%20Clustering%20Beyond%20Homophily/tables/a4101b456dd7b3d80a51bb13d4841731a55da355618bfb0649bd26c4cd3eb213.jpg)

<a id="paper-6026"></a>

## Gradient Flow Dynamics and Implicit Bias of Diagonal Linear Networks under Infinitesimal Initialization


### Images

![47f3ff75005ff8438d62aeb5bd985ae30fe8cde401bbef9b0549172745adffda.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/47f3ff75005ff8438d62aeb5bd985ae30fe8cde401bbef9b0549172745adffda.jpg)

![718eb84e415c48b6092711c317cdabc207832716f0236ae6be84ad25ff72f6be.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/718eb84e415c48b6092711c317cdabc207832716f0236ae6be84ad25ff72f6be.jpg)

![813169ae700dcfc985eb39f7d60ba048a1f164e61e0d320047c950548f18f0d1.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/813169ae700dcfc985eb39f7d60ba048a1f164e61e0d320047c950548f18f0d1.jpg)

![831fe5da92d35f5744f2a0fe5592753e640ccd6c585509ae94855db53a5355d2.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/831fe5da92d35f5744f2a0fe5592753e640ccd6c585509ae94855db53a5355d2.jpg)

![beae7f4f9e556b4bff66caf011fa76b1df9681a037852fe1011a5139fc1fb584.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/beae7f4f9e556b4bff66caf011fa76b1df9681a037852fe1011a5139fc1fb584.jpg)

![c488c1a8e228221cf14303fee00b9dd58e7108c8fb77eac18a93f56b123f6730.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/c488c1a8e228221cf14303fee00b9dd58e7108c8fb77eac18a93f56b123f6730.jpg)

![c54040a0575a53e18d6f72caedd38d097f04dd9c0d1789b2d269089f0f115309.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/c54040a0575a53e18d6f72caedd38d097f04dd9c0d1789b2d269089f0f115309.jpg)

![c643d31e88ace1cade7d70f1c856b744b5b2b6de98c58a8e0b9c48bf22696031.jpg](../icml_results/6076_Gradient%20Flow%20Dynamics%20and%20Implicit%20Bias%20of%20Diagonal%20Linear%20Networks%20under%20Infinitesimal%20Initializat/images/c643d31e88ace1cade7d70f1c856b744b5b2b6de98c58a8e0b9c48bf22696031.jpg)

<a id="paper-6027"></a>

## Large Language Model Agents Are Not Always Faithful Self-Evolvers


### Images

![0fc86f4efa9dc53db9f6a4e0a16fb4ac34cbee18baf0fb468a26f28b1776962a.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/0fc86f4efa9dc53db9f6a4e0a16fb4ac34cbee18baf0fb468a26f28b1776962a.jpg)

![18fbcfd4fecc55864174ebe783e63201118835b7493854810bcc888b06f34afe.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/18fbcfd4fecc55864174ebe783e63201118835b7493854810bcc888b06f34afe.jpg)

![1d82fc074ce0c4307f15c835021e04f36ffcdb5ce70164bcc8b9e86f2e9f6229.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/1d82fc074ce0c4307f15c835021e04f36ffcdb5ce70164bcc8b9e86f2e9f6229.jpg)

![1e838f42cabea7d5b3c644d68f47597cd09426639a56212441512eff7840e158.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/1e838f42cabea7d5b3c644d68f47597cd09426639a56212441512eff7840e158.jpg)

![1fb49d3c151fe8b79e599b4567675f34ed55b58432ce703d54194116d6e73ce2.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/1fb49d3c151fe8b79e599b4567675f34ed55b58432ce703d54194116d6e73ce2.jpg)

![2ed2c0086036f6920ca3b5239b7f18407e6f51a1091a34a76a1b32e378deccdd.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/2ed2c0086036f6920ca3b5239b7f18407e6f51a1091a34a76a1b32e378deccdd.jpg)

![30806d364a19a5ee0d0b08cd3b36dbdbae09bb9c1571067eca4f3b46b5a17d8a.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/30806d364a19a5ee0d0b08cd3b36dbdbae09bb9c1571067eca4f3b46b5a17d8a.jpg)

![4112033eeaf1d0dbf7b6ae9b370a01fc491d0558b0132c488ae50d3317409cfe.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/4112033eeaf1d0dbf7b6ae9b370a01fc491d0558b0132c488ae50d3317409cfe.jpg)

![5ca47f442bb158582e877aa10204d4943afa1f29dd27ec7100f9ff566dfae349.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/5ca47f442bb158582e877aa10204d4943afa1f29dd27ec7100f9ff566dfae349.jpg)

![6fbf7b5edb7dd9167007245a85aefe4c76b05b206bf26175be972c9d8e988d54.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/6fbf7b5edb7dd9167007245a85aefe4c76b05b206bf26175be972c9d8e988d54.jpg)

![9292657b5dda448720253fc73cc4b9864712e36fa29c4d31992cdff0793e7b8a.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/9292657b5dda448720253fc73cc4b9864712e36fa29c4d31992cdff0793e7b8a.jpg)

![96163157243b485cbb5aa7f567d9fa59697b7944f6f3c2f2782f024fb7e1656d.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/96163157243b485cbb5aa7f567d9fa59697b7944f6f3c2f2782f024fb7e1656d.jpg)

![a35d9155a065d740d747a02a54840169bb1ced145f1eaed1e5b779ff522e09f6.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/a35d9155a065d740d747a02a54840169bb1ced145f1eaed1e5b779ff522e09f6.jpg)

![a767b425b0cdfae9d07aa80d0a6966b78d48401d48eca6d3642f160ac156fc3c.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/a767b425b0cdfae9d07aa80d0a6966b78d48401d48eca6d3642f160ac156fc3c.jpg)

![c63d4b8d318d0d92d89694671bc4a77134115ddbeb087108132d6c22b7fb9b90.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/c63d4b8d318d0d92d89694671bc4a77134115ddbeb087108132d6c22b7fb9b90.jpg)

![d47af2b350d0a0fd3861c90e4f47a59c58384cabf55b492205a34a859b213507.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/d47af2b350d0a0fd3861c90e4f47a59c58384cabf55b492205a34a859b213507.jpg)

![d69349e559e3e5058f90b087735ae42784741155b0e67e3eba08af1173117b53.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/d69349e559e3e5058f90b087735ae42784741155b0e67e3eba08af1173117b53.jpg)

![dd4ce75cca35579a41460132ad4c48bd9a69446a111636b97255fd53e17ffa92.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/dd4ce75cca35579a41460132ad4c48bd9a69446a111636b97255fd53e17ffa92.jpg)

![ea30deecd21a882a22b86955583a8dbd32b58db6c7eef424f9dce8ff875c09ba.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/images/ea30deecd21a882a22b86955583a8dbd32b58db6c7eef424f9dce8ff875c09ba.jpg)

### Tables

![005ffe39d51fab662d8e5a6345d0bbeba48e8251d24b09ff2d039f26adca8b55.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/005ffe39d51fab662d8e5a6345d0bbeba48e8251d24b09ff2d039f26adca8b55.jpg)

![02f8f3acc90cf4cff82bb6ee58eee1b98ed33e4d7bbe06e473d70ac91b855231.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/02f8f3acc90cf4cff82bb6ee58eee1b98ed33e4d7bbe06e473d70ac91b855231.jpg)

![536071c8c9cf7cfad644b38b74c2d0e862f41a86c9d28edc6976c2051a980e22.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/536071c8c9cf7cfad644b38b74c2d0e862f41a86c9d28edc6976c2051a980e22.jpg)

![a9d1bed9f39d1fb65c026ea85b160bfdbef7e4ab9a21d6e13c3ce0a91f181b47.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/a9d1bed9f39d1fb65c026ea85b160bfdbef7e4ab9a21d6e13c3ce0a91f181b47.jpg)

![add419162572cfd4332adb962472981497bfc9fb11c1743baf63fd4aa20dde71.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/add419162572cfd4332adb962472981497bfc9fb11c1743baf63fd4aa20dde71.jpg)

![df0d1e46505c3a652469a16bf167c554934433185c0942f39fadaba1698f3912.jpg](../icml_results/6077_Large%20Language%20Model%20Agents%20Are%20Not%20Always%20Faithful%20Self-Evolvers/tables/df0d1e46505c3a652469a16bf167c554934433185c0942f39fadaba1698f3912.jpg)

<a id="paper-6028"></a>

## Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks


### Images

![057a03a37271d36e1fcad191e7a35ab6ef01eb3b07438b941b6a4e59e895f764.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/057a03a37271d36e1fcad191e7a35ab6ef01eb3b07438b941b6a4e59e895f764.jpg)

![081eabf8d4196ee2136c0c09002c508c09cee3a40247752302dbdf07bb7ee7f3.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/081eabf8d4196ee2136c0c09002c508c09cee3a40247752302dbdf07bb7ee7f3.jpg)

![09f1e92941f597d147e1c5d7effc623b5c0c8109dfafe0d9bf42bb21e7fdc05e.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/09f1e92941f597d147e1c5d7effc623b5c0c8109dfafe0d9bf42bb21e7fdc05e.jpg)

![3761edbd14e494b82414deed03e5e364fb32ecde79075f5b48d664a9563ef062.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/3761edbd14e494b82414deed03e5e364fb32ecde79075f5b48d664a9563ef062.jpg)

![3f1061def4dc447faf87de52be9f39d2c0a9e1a1a8c4a8828a9cd7d376a5985f.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/3f1061def4dc447faf87de52be9f39d2c0a9e1a1a8c4a8828a9cd7d376a5985f.jpg)

![437af50f576c2ef6a4fd1c9e0d994ab37c77148dd506e9fb303495cf8ac5c974.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/437af50f576c2ef6a4fd1c9e0d994ab37c77148dd506e9fb303495cf8ac5c974.jpg)

![5c95f84792d63b0a4b2b62c492b27b458876c1c5fd03ff0b408f7bbe8e99f7c5.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/5c95f84792d63b0a4b2b62c492b27b458876c1c5fd03ff0b408f7bbe8e99f7c5.jpg)

![5e973d3891fc9154b31e1e0e8f67f98be646ca5238e4cd031eaea29db2a5c5f8.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/5e973d3891fc9154b31e1e0e8f67f98be646ca5238e4cd031eaea29db2a5c5f8.jpg)

![a5eb8eba9eac5736adfb365161ac19ff00478b006b0f9017781b8f1c5ef92c7d.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/a5eb8eba9eac5736adfb365161ac19ff00478b006b0f9017781b8f1c5ef92c7d.jpg)

![e61bb4ed16981d26a51eaa443d6b4f7eefebade513c85d95feb8f664d39a2310.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/images/e61bb4ed16981d26a51eaa443d6b4f7eefebade513c85d95feb8f664d39a2310.jpg)

### Tables

![249fd0812a0e8d25af93f8f0e634e624cb5e7a2e3a3f2757c0c3604afef619fe.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/249fd0812a0e8d25af93f8f0e634e624cb5e7a2e3a3f2757c0c3604afef619fe.jpg)

![3c4d55b8a9a0d40428941f77a241728e1fe91e0eb02576278973cfd9d22d6ede.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/3c4d55b8a9a0d40428941f77a241728e1fe91e0eb02576278973cfd9d22d6ede.jpg)

![3cc6b9860de21bc3427ce59bbf251e34e894eeb18dc56f920a606933981c5bce.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/3cc6b9860de21bc3427ce59bbf251e34e894eeb18dc56f920a606933981c5bce.jpg)

![6bd860401fd1459fadb7ff8e5240f8c402427268b1ac1ec1e815107793f5fe51.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/6bd860401fd1459fadb7ff8e5240f8c402427268b1ac1ec1e815107793f5fe51.jpg)

![80323d3f18fd412ef13809922e549a9bcb39fa002b4dc677273857b4d8c996fa.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/80323d3f18fd412ef13809922e549a9bcb39fa002b4dc677273857b4d8c996fa.jpg)

![814727d3896339d87472d58d2a5ca3f88209d9584a4294884c53e8807702f3fa.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/814727d3896339d87472d58d2a5ca3f88209d9584a4294884c53e8807702f3fa.jpg)

![84fa820c70e82d0c697a127220ce3f0af7e207ff2b9df634ffb78c03b330e94d.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/84fa820c70e82d0c697a127220ce3f0af7e207ff2b9df634ffb78c03b330e94d.jpg)

![a7da0ca846936bf4891f83fda465facbe2ff324dcd297064a5c502bc4799cc11.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/a7da0ca846936bf4891f83fda465facbe2ff324dcd297064a5c502bc4799cc11.jpg)

![c80185cf422c4928646ba1b04384fa2554c4269d10d22c4dffcac4e92547aa79.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/c80185cf422c4928646ba1b04384fa2554c4269d10d22c4dffcac4e92547aa79.jpg)

![dfe33a698530e157f55da9e6a111ea6fc98519629f12168b015d77c8b3fa1223.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/dfe33a698530e157f55da9e6a111ea6fc98519629f12168b015d77c8b3fa1223.jpg)

![eb4e2603ffc9456a1e55fdcb93da5b40746fe05c44ba3677f45ca15fdb3aa3a3.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/eb4e2603ffc9456a1e55fdcb93da5b40746fe05c44ba3677f45ca15fdb3aa3a3.jpg)

![f6748c2d12f45d7dfdfa1c5aac5f945e1684953f431b43b6d60c4ce88c6a8763.jpg](../icml_results/6078_Is%20Vibe%20Coding%20Safe_%20Benchmarking%20Vulnerability%20of%20Agent-Generated%20Code%20in%20Real-World%20Tasks/tables/f6748c2d12f45d7dfdfa1c5aac5f945e1684953f431b43b6d60c4ce88c6a8763.jpg)

<a id="paper-6029"></a>

## PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Fine-Grained Expert Merging and Bit-packed Inference


### Images

![0572019a50f45e87d06d00576228572bccdc6a27935bbd495fbf3c775e24ac4b.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/0572019a50f45e87d06d00576228572bccdc6a27935bbd495fbf3c775e24ac4b.jpg)

![09429bb4521a2767e2c904fc320c0c06831c4ab803769c02959cc6021d67051e.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/09429bb4521a2767e2c904fc320c0c06831c4ab803769c02959cc6021d67051e.jpg)

![0bf48f68003e5414b1d03569ea54d9bd4129d57c6dd568927fc4e24466028fbb.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/0bf48f68003e5414b1d03569ea54d9bd4129d57c6dd568927fc4e24466028fbb.jpg)

![199d25769c4955e79db72ef46a3a0338a1ffbe6ac72344b55b1ac4e90b915c5c.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/199d25769c4955e79db72ef46a3a0338a1ffbe6ac72344b55b1ac4e90b915c5c.jpg)

![4ff9df84a5f92366e0f1300c806f85cd18f7f99e895607cc8abfa5577f86edae.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/4ff9df84a5f92366e0f1300c806f85cd18f7f99e895607cc8abfa5577f86edae.jpg)

![52842cb73fefedd32120c50496a2e5f8a686ee29b72b1443f9b835f91631a19a.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/52842cb73fefedd32120c50496a2e5f8a686ee29b72b1443f9b835f91631a19a.jpg)

![5e1f3834e686acb864fba166271bc5080d7326f0cb97af23ecc883b6cd1b7f00.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/5e1f3834e686acb864fba166271bc5080d7326f0cb97af23ecc883b6cd1b7f00.jpg)

![5eec22062e6fa5e2fb0bdaf34f6af80c1fbdab4b285fe604ed4f7ac260c528e0.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/5eec22062e6fa5e2fb0bdaf34f6af80c1fbdab4b285fe604ed4f7ac260c528e0.jpg)

![733fabc657269ded37101e4ad122d7699ea086d3eb64f34e7f85145fa0544b44.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/733fabc657269ded37101e4ad122d7699ea086d3eb64f34e7f85145fa0544b44.jpg)

![a7f18240c7a5497d2d77055022407b66c25dc7b08002c3805a03c2a3a5b0ccf3.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/a7f18240c7a5497d2d77055022407b66c25dc7b08002c3805a03c2a3a5b0ccf3.jpg)

![b3972535294842736bb0b98e07f36ee329f026a25a71b8d2d07ee1eae6366187.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/b3972535294842736bb0b98e07f36ee329f026a25a71b8d2d07ee1eae6366187.jpg)

![d59706c7cec01849227fed0b0db71525e26c714477b7ec1d2f6190a1a927d692.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/d59706c7cec01849227fed0b0db71525e26c714477b7ec1d2f6190a1a927d692.jpg)

![d6db97664a2e0a98395d863abcf8cbb3e000a93463a969a8e3892a4537017a43.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/d6db97664a2e0a98395d863abcf8cbb3e000a93463a969a8e3892a4537017a43.jpg)

![d92acc9c70a24960378d19d5cb856403b63dd0e98df8c5841cc48a4a8c234d16.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/d92acc9c70a24960378d19d5cb856403b63dd0e98df8c5841cc48a4a8c234d16.jpg)

![ec5408355023c991c85fdcc03cff14bbf04b9329677f710af6ed472f8b389af0.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/ec5408355023c991c85fdcc03cff14bbf04b9329677f710af6ed472f8b389af0.jpg)

![f0e2a7d61ed8482bd7bd3a850cc173cc88f88e94759eeae09f505cd5d812bb60.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/images/f0e2a7d61ed8482bd7bd3a850cc173cc88f88e94759eeae09f505cd5d812bb60.jpg)

### Tables

![22cf408e712f33d1c85bfa121d536f4b9ec092588d84c78d1976f78d360ed0a5.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/22cf408e712f33d1c85bfa121d536f4b9ec092588d84c78d1976f78d360ed0a5.jpg)

![2470f2a6a711cd7c9d4657d0f643ed4c41cf0bc8f48401efa44b4438c8dd597b.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/2470f2a6a711cd7c9d4657d0f643ed4c41cf0bc8f48401efa44b4438c8dd597b.jpg)

![33c3be7a54fbbc6080c41b2f6728f3efa296eeaebc3e8204b6afdfa9564da95d.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/33c3be7a54fbbc6080c41b2f6728f3efa296eeaebc3e8204b6afdfa9564da95d.jpg)

![37bd470a449a6cc4cbecfd36cb0fb6b64d120237ec62315a1511bcb26a8d35b1.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/37bd470a449a6cc4cbecfd36cb0fb6b64d120237ec62315a1511bcb26a8d35b1.jpg)

![3e4ed87bc1a2172c046b21d06a2a19fb56226bd22b729b2c40259a194d844eaf.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/3e4ed87bc1a2172c046b21d06a2a19fb56226bd22b729b2c40259a194d844eaf.jpg)

![4db018633e3a0af2da7b5187de15d04ccf6c31e46ebeabd9684f86f8b2155cef.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/4db018633e3a0af2da7b5187de15d04ccf6c31e46ebeabd9684f86f8b2155cef.jpg)

![5944ef13c925e11eadadd411f3484a6f7c4187a834657195abd2994c4541e157.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/5944ef13c925e11eadadd411f3484a6f7c4187a834657195abd2994c4541e157.jpg)

![5aa94ae97cb0b8fdab511a9e203f82ad236d81725c54d914d8c74ee64e5e5d49.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/5aa94ae97cb0b8fdab511a9e203f82ad236d81725c54d914d8c74ee64e5e5d49.jpg)

![5c2ec6bdc72bc8dd3dcb0390381214bfc269b1cf97b98b64fdbd9fdd4984afab.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/5c2ec6bdc72bc8dd3dcb0390381214bfc269b1cf97b98b64fdbd9fdd4984afab.jpg)

![6c5dba485d8b1f811fd835106abf7356d8b46893b4f303999991ab4d6638628e.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/6c5dba485d8b1f811fd835106abf7356d8b46893b4f303999991ab4d6638628e.jpg)

![808700191a15871170ad4091c5687bd968045a66909d77dcf5836014018a6cde.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/808700191a15871170ad4091c5687bd968045a66909d77dcf5836014018a6cde.jpg)

![82deb125d77815112ca3d6c80bd56a675c6c5ad89592a8514a5ca212dc3511e0.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/82deb125d77815112ca3d6c80bd56a675c6c5ad89592a8514a5ca212dc3511e0.jpg)

![9a2dec6ffec6fbf6eff9509075ba2053113266e2f0aff4ef8aafba73a4b02fac.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/9a2dec6ffec6fbf6eff9509075ba2053113266e2f0aff4ef8aafba73a4b02fac.jpg)

![9f17579181e76af10ba3e0ad1eb3fa40cec7e139a6e6be9028348c55c4101e6f.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/9f17579181e76af10ba3e0ad1eb3fa40cec7e139a6e6be9028348c55c4101e6f.jpg)

![b581cfd839822500102cfd46d993664ff6629bcb97287d98546d6b7e351b7199.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/b581cfd839822500102cfd46d993664ff6629bcb97287d98546d6b7e351b7199.jpg)

![bcee7e557ea78c9774822adaf86cd4cf1fa9d40c2394e1ee55a5f8ac95998bc9.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/bcee7e557ea78c9774822adaf86cd4cf1fa9d40c2394e1ee55a5f8ac95998bc9.jpg)

![f13f03c99edfebd066c4b2e0cfcbe0f9573c65ba6dc92eb0b65a13ca658e9f43.jpg](../icml_results/6079_PuzzleMoE_%20Efficient%20Compression%20of%20Large%20Mixture-of-Experts%20Models%20via%20Fine-Grained%20Expert%20Merging%20/tables/f13f03c99edfebd066c4b2e0cfcbe0f9573c65ba6dc92eb0b65a13ca658e9f43.jpg)

<a id="paper-6030"></a>

## Contractive Anchor Resolvent Diffusion for Incomplete Multi-View Clustering


### Images

![22eb459adacb977746ac04a7bde0a3cc6a66cca7692df780850d739c93696875.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/22eb459adacb977746ac04a7bde0a3cc6a66cca7692df780850d739c93696875.jpg)

![2b4662fd8929fba0c1574c2e686d51e26984fe9e4a1363c6ba0f3e13f8794d23.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/2b4662fd8929fba0c1574c2e686d51e26984fe9e4a1363c6ba0f3e13f8794d23.jpg)

![70e1d10ed0fa5e180e2bcc9bd899e27a0eedb152598abde90c5bede7f85ce60b.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/70e1d10ed0fa5e180e2bcc9bd899e27a0eedb152598abde90c5bede7f85ce60b.jpg)

![74d37efe68b624c1cdc9a3586b08d98a0de7ae9c1db5e8ee8c06c9d49bcf252a.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/74d37efe68b624c1cdc9a3586b08d98a0de7ae9c1db5e8ee8c06c9d49bcf252a.jpg)

![c1dd5e39ec19ad27b44952e9faa370abd0720e72759b1e1132432d12e30b472b.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/c1dd5e39ec19ad27b44952e9faa370abd0720e72759b1e1132432d12e30b472b.jpg)

![cd6b2cee6954c572af37c654e2f356eba4aae57047bdafe15c49fdb4a250508b.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/images/cd6b2cee6954c572af37c654e2f356eba4aae57047bdafe15c49fdb4a250508b.jpg)

### Tables

![4a699297b136200f2ffe056bfed1da20dd8c7ea456b40ddda11fc7a7dc1fe181.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/tables/4a699297b136200f2ffe056bfed1da20dd8c7ea456b40ddda11fc7a7dc1fe181.jpg)

![a0c2475c7b12936b5a0f29ea9aaf9785022486a1f8beebff6f5255cd0dd5a71c.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/tables/a0c2475c7b12936b5a0f29ea9aaf9785022486a1f8beebff6f5255cd0dd5a71c.jpg)

![a58a44a524dee96916020626f48a5418588e78ad2cb7314605ab4827cbcbf04a.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/tables/a58a44a524dee96916020626f48a5418588e78ad2cb7314605ab4827cbcbf04a.jpg)

![bffdf23ea9a8e88516a9baf462feea72cfa0ecab096a633bfe266f1ea041078d.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/tables/bffdf23ea9a8e88516a9baf462feea72cfa0ecab096a633bfe266f1ea041078d.jpg)

![db288bec18bb0164607ca9b504466a5d9dccd1d0025c1c9ea9cf06b6b48cf5b8.jpg](../icml_results/6080_Contractive%20Anchor%20Resolvent%20Diffusion%20for%20Incomplete%20Multi-View%20Clustering/tables/db288bec18bb0164607ca9b504466a5d9dccd1d0025c1c9ea9cf06b6b48cf5b8.jpg)

<a id="paper-6031"></a>

## Anti-Backdoor Coreset Selection via Cumulative Entropy


### Images

![12f4bb9baacb05de13857ad17b5e3ca7de4d8e7d1d0157e56fb6680583c8edd0.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/12f4bb9baacb05de13857ad17b5e3ca7de4d8e7d1d0157e56fb6680583c8edd0.jpg)

![136a82f7a8d21404c4dce04089a8859f5c12e5aa1a0f9e7742a83ebdb1707440.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/136a82f7a8d21404c4dce04089a8859f5c12e5aa1a0f9e7742a83ebdb1707440.jpg)

![1564730f6343546672cf0af42ef46f8d617fb586e41641146fdbf734c708dffa.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/1564730f6343546672cf0af42ef46f8d617fb586e41641146fdbf734c708dffa.jpg)

![39898c7821fe743f6428540a7a234b080593e0bca0d7fe8e378176f36e1f61b5.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/39898c7821fe743f6428540a7a234b080593e0bca0d7fe8e378176f36e1f61b5.jpg)

![527d7eb136a389ca7be32689d3adad4b0f4b2fcefd47686a3ea9098da8452c98.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/527d7eb136a389ca7be32689d3adad4b0f4b2fcefd47686a3ea9098da8452c98.jpg)

![5e61f84d92b2945e4ab8da5d1e7620d41cce303be50138148052c09d4239ed2d.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/5e61f84d92b2945e4ab8da5d1e7620d41cce303be50138148052c09d4239ed2d.jpg)

![630be30c3d40d3874d18ee4b5f73f2c62aa3afdce34fd72dde543353d260b536.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/630be30c3d40d3874d18ee4b5f73f2c62aa3afdce34fd72dde543353d260b536.jpg)

![77614007a396db4a7d4037158284916139cda11093916d7af87bf895ee78d708.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/77614007a396db4a7d4037158284916139cda11093916d7af87bf895ee78d708.jpg)

![7df4c04aec1257986565040fb39862b87cb85db2f2cd2e97a0cdc86375fc3b8a.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/7df4c04aec1257986565040fb39862b87cb85db2f2cd2e97a0cdc86375fc3b8a.jpg)

![829a6729d433ec25823981405a0c3f2975eda971c616fd80445ed2d85ad8440b.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/829a6729d433ec25823981405a0c3f2975eda971c616fd80445ed2d85ad8440b.jpg)

![9c967e6094c46a0fa180c2544ccb823b8acb11feae2346553ca3c20c631f39d2.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/9c967e6094c46a0fa180c2544ccb823b8acb11feae2346553ca3c20c631f39d2.jpg)

![9f29da46c5d608718538a72f7aa21599753c2055bce0d707dba9204317a0a6ad.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/9f29da46c5d608718538a72f7aa21599753c2055bce0d707dba9204317a0a6ad.jpg)

![ad8bd2031909ab3fadf7c20ace0c395a1c6fc73c9e7493b9111b358e572cc314.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/ad8bd2031909ab3fadf7c20ace0c395a1c6fc73c9e7493b9111b358e572cc314.jpg)

![ae9e5e4399807530ec29d0a34dfe6090a705fee23ff9a0f092466454e39c4742.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/ae9e5e4399807530ec29d0a34dfe6090a705fee23ff9a0f092466454e39c4742.jpg)

![c87dd62902817bcbcfcf1fd27a2c88ce29babe30b9355fb53d277de3716f4b12.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/c87dd62902817bcbcfcf1fd27a2c88ce29babe30b9355fb53d277de3716f4b12.jpg)

![d953a67c467d937ea855e3dae76458557aa9ed19a38c321571b770181ee78438.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/d953a67c467d937ea855e3dae76458557aa9ed19a38c321571b770181ee78438.jpg)

![d9ca03f7603f8b6b05ba7552e36e70e88ba10f36bdc266a79eae4e54920c96f6.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/d9ca03f7603f8b6b05ba7552e36e70e88ba10f36bdc266a79eae4e54920c96f6.jpg)

![ede06cd6d91143fcaef0705cc03be81c27e0fde4368ed9d16718b4a231540432.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/ede06cd6d91143fcaef0705cc03be81c27e0fde4368ed9d16718b4a231540432.jpg)

![f252203b6cc50c3c23276a8594bb84238156047e678864cd361632141ddecc50.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/images/f252203b6cc50c3c23276a8594bb84238156047e678864cd361632141ddecc50.jpg)

### Tables

![06c3e8d8c8c72ece58a8e29bd4cc23f39cb4e239e322a0eff45b66bb6865d7ce.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/06c3e8d8c8c72ece58a8e29bd4cc23f39cb4e239e322a0eff45b66bb6865d7ce.jpg)

![117fcbae2586283ccd7ba5b548362cc5b41332afeee69cce1da079c2fccea642.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/117fcbae2586283ccd7ba5b548362cc5b41332afeee69cce1da079c2fccea642.jpg)

![19fefead31b2d27846f2acebeea925f681a2057bb33e0cfbab129282222f3f6d.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/19fefead31b2d27846f2acebeea925f681a2057bb33e0cfbab129282222f3f6d.jpg)

![1d7a43a149f7e131142e7b3764ef17fa20aa61508bf6f99815191fad92d13cbb.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/1d7a43a149f7e131142e7b3764ef17fa20aa61508bf6f99815191fad92d13cbb.jpg)

![38caa18c47e510b668a97b47481abe31c704df48994818ad126eb4efd6aca246.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/38caa18c47e510b668a97b47481abe31c704df48994818ad126eb4efd6aca246.jpg)

![4c1b91401e762ccfd3659635e906c8ec61997c1650d4c4902e6c5fadfe41ab01.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/4c1b91401e762ccfd3659635e906c8ec61997c1650d4c4902e6c5fadfe41ab01.jpg)

![5f9f38694ce1f98c0038c3bc027a3437d38211d33a22cb549243293b1d91b4e8.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/5f9f38694ce1f98c0038c3bc027a3437d38211d33a22cb549243293b1d91b4e8.jpg)

![6387c5bc2359a6aa0b1685637dd8ea39a8a9a168ebb71c37eebdb0e9554fe42a.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/6387c5bc2359a6aa0b1685637dd8ea39a8a9a168ebb71c37eebdb0e9554fe42a.jpg)

![7570a3b8cb1c6a2810e1a2696265ed3857b51ba5b04428ed0eea072bb486d2f8.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/7570a3b8cb1c6a2810e1a2696265ed3857b51ba5b04428ed0eea072bb486d2f8.jpg)

![8f67d070c96f82770cf17b27ed9c6add1c2dc54c6878c12a87e4f5f525e939ee.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/8f67d070c96f82770cf17b27ed9c6add1c2dc54c6878c12a87e4f5f525e939ee.jpg)

![90695bbb1758cb09cd7fb093e486754895b5fcc3e36dac2cd7a23b32d546de1a.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/90695bbb1758cb09cd7fb093e486754895b5fcc3e36dac2cd7a23b32d546de1a.jpg)

![9a942b2d98f95a1fa48a12c23612b043820f81bc9cbb2f924c37f3cdb131f4fe.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/9a942b2d98f95a1fa48a12c23612b043820f81bc9cbb2f924c37f3cdb131f4fe.jpg)

![9aa0107ab48368fcb63015aad510a9b6814d7cd0953b6cec8afa4eb65698f1e2.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/9aa0107ab48368fcb63015aad510a9b6814d7cd0953b6cec8afa4eb65698f1e2.jpg)

![9fb75fea53677c07474f35c8494fdb07ee49ea93b401aea698d7d3db6cfc4f0f.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/9fb75fea53677c07474f35c8494fdb07ee49ea93b401aea698d7d3db6cfc4f0f.jpg)

![b04f150e1ce0055978e8fd4f32fba3733daf7fc2258c195de2b3532afe079325.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/b04f150e1ce0055978e8fd4f32fba3733daf7fc2258c195de2b3532afe079325.jpg)

![c92a4f4528c5e414426ab7134ccbd002401a599c089d9e48cc1eab33a64f3c79.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/c92a4f4528c5e414426ab7134ccbd002401a599c089d9e48cc1eab33a64f3c79.jpg)

![d29af53b2a4a6bcbe759d642ba1feb942a7ff34e9708384a6ca205d1c0f05c51.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/d29af53b2a4a6bcbe759d642ba1feb942a7ff34e9708384a6ca205d1c0f05c51.jpg)

![e724b752c0bfdfe19c143897af896b4f419600dc1590e4995ea74a1f7339e2f2.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/e724b752c0bfdfe19c143897af896b4f419600dc1590e4995ea74a1f7339e2f2.jpg)

![fbc351e14aa59f44f26c9a9a4b38b68a9f5d8d7b81732f10eca8bbbc53cddba7.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/fbc351e14aa59f44f26c9a9a4b38b68a9f5d8d7b81732f10eca8bbbc53cddba7.jpg)

![fd5d73873dd719a9f1fa131af0a24ee692cdd00024502ad266ac8ffecc712796.jpg](../icml_results/6081_Anti-Backdoor%20Coreset%20Selection%20via%20Cumulative%20Entropy/tables/fd5d73873dd719a9f1fa131af0a24ee692cdd00024502ad266ac8ffecc712796.jpg)

<a id="paper-6032"></a>

## Scaling Prompt Synthesis for Large Language Model Reasoning


### Images

![0c69aaa0c7eddb75ff74a897b0e8afeb76f2917aefcd8c0aba32c3adfc2431d5.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/0c69aaa0c7eddb75ff74a897b0e8afeb76f2917aefcd8c0aba32c3adfc2431d5.jpg)

![20dc9b8be48e85127ad506827ad88927df67f4780299e56d958655fd86676c91.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/20dc9b8be48e85127ad506827ad88927df67f4780299e56d958655fd86676c91.jpg)

![280c817a84ab1a3096b6848b75fbc875911dac9d65e85e132a23937342ffe414.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/280c817a84ab1a3096b6848b75fbc875911dac9d65e85e132a23937342ffe414.jpg)

![572b75a8f502b3c20e07c95e4e5668f8603e993932df413b573611d7e33beb47.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/572b75a8f502b3c20e07c95e4e5668f8603e993932df413b573611d7e33beb47.jpg)

![577a1960a86c06cd9a4ce5f12e2b5544875f6b0a6e038de3b6bf05fa9b745fa1.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/577a1960a86c06cd9a4ce5f12e2b5544875f6b0a6e038de3b6bf05fa9b745fa1.jpg)

![eca87e13d9788d32275160bf03660e4373074a90528070afe74b2803584c6304.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/images/eca87e13d9788d32275160bf03660e4373074a90528070afe74b2803584c6304.jpg)

### Tables

![06d9e985003e3beda08a967a6f29afd55b329006b4296309f3115455f1d8862b.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/06d9e985003e3beda08a967a6f29afd55b329006b4296309f3115455f1d8862b.jpg)

![254384feda4c82291f82b47c420464b5cec1912676a25d4197d7d6405b467192.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/254384feda4c82291f82b47c420464b5cec1912676a25d4197d7d6405b467192.jpg)

![6d201556bd0a06264225a0b4354995ff0503025fe845de6ede1fb8e3c03f16f9.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/6d201556bd0a06264225a0b4354995ff0503025fe845de6ede1fb8e3c03f16f9.jpg)

![88ccbcce047e97d614d849ee39ef720e006c63f8939084f4518fe0f5f4ba80e0.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/88ccbcce047e97d614d849ee39ef720e006c63f8939084f4518fe0f5f4ba80e0.jpg)

![8d821e0067cf679fce1ca5a7fe0e29d3de302713cfdccd6bb36500d0d05146fb.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/8d821e0067cf679fce1ca5a7fe0e29d3de302713cfdccd6bb36500d0d05146fb.jpg)

![a3939b4c80b584d376bae72179d9fa33b83633d7e89da942a4aebc26c89686ca.jpg](../icml_results/6082_Scaling%20Prompt%20Synthesis%20for%20Large%20Language%20Model%20Reasoning/tables/a3939b4c80b584d376bae72179d9fa33b83633d7e89da942a4aebc26c89686ca.jpg)

<a id="paper-6033"></a>

## Attend to Anything: Foundation Model for Unified Human Attention Modeling


### Images

![1179828cc1ee2059e789cc88ec88ca5cf67217bcd6fc56e5922db3755953c2ad.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/1179828cc1ee2059e789cc88ec88ca5cf67217bcd6fc56e5922db3755953c2ad.jpg)

![1da019b70312e8450e338f3b6c26bb1135e7887470790579d0dada7f4aee2c61.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/1da019b70312e8450e338f3b6c26bb1135e7887470790579d0dada7f4aee2c61.jpg)

![222cc7f7042c4fcb2d6c49a7d8773950c967ed7b0c01292a9366b9c3aec78bd4.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/222cc7f7042c4fcb2d6c49a7d8773950c967ed7b0c01292a9366b9c3aec78bd4.jpg)

![29837c2099fe4953e78d097df1499ed7145c8e5c95f3bad4926ab33ecfa5d9b7.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/29837c2099fe4953e78d097df1499ed7145c8e5c95f3bad4926ab33ecfa5d9b7.jpg)

![502b23b15dbc2fb2e9945603ac352db23cc04d4b832037d50238e19abd22a8f3.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/502b23b15dbc2fb2e9945603ac352db23cc04d4b832037d50238e19abd22a8f3.jpg)

![57835d470b72a3ab074a16b782c0efc4c6670a98dc0871bbedd6f2e75bd0f55d.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/57835d470b72a3ab074a16b782c0efc4c6670a98dc0871bbedd6f2e75bd0f55d.jpg)

![692aac7d30a9011b9ddc267a94fcf8d7565a4f92828a88f1af0244e00781a988.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/692aac7d30a9011b9ddc267a94fcf8d7565a4f92828a88f1af0244e00781a988.jpg)

![8491c739b9a663d8f94d764b7066133f2b2949d6f36ad9fed44f76a0d4c07c95.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/8491c739b9a663d8f94d764b7066133f2b2949d6f36ad9fed44f76a0d4c07c95.jpg)

![a42679a77d1486f484702237c2d069f979addadb9ec6965c10a6d2fec2611f1f.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/a42679a77d1486f484702237c2d069f979addadb9ec6965c10a6d2fec2611f1f.jpg)

![b7ce2c2c59d6c73c15d6a6a5bca7aab285bc53df13b1046868f40d02d52135b4.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/b7ce2c2c59d6c73c15d6a6a5bca7aab285bc53df13b1046868f40d02d52135b4.jpg)

![da3d94c69967cc01e7e87dc927d09835fcda93955012ac48bafc764cbafdea45.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/images/da3d94c69967cc01e7e87dc927d09835fcda93955012ac48bafc764cbafdea45.jpg)

### Tables

![0b2d879c3ee6f3e62290890cb82e4bb30df96ec36ba18b71bc95fcd4e8b32a4b.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/0b2d879c3ee6f3e62290890cb82e4bb30df96ec36ba18b71bc95fcd4e8b32a4b.jpg)

![16e77aea4c49388d666351c2d1e7784024294ee0b8048aba72442340bf65bce0.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/16e77aea4c49388d666351c2d1e7784024294ee0b8048aba72442340bf65bce0.jpg)

![1ae23f791a3cb7bc81eaa3d50af0b19f6090822fd4e2d5c7a8cbb062529e5cc5.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/1ae23f791a3cb7bc81eaa3d50af0b19f6090822fd4e2d5c7a8cbb062529e5cc5.jpg)

![336c401664d92356fed5632576ef247905d5ef7b708b47aba766d149497e43f4.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/336c401664d92356fed5632576ef247905d5ef7b708b47aba766d149497e43f4.jpg)

![356211c8e3e0b0c3b189fd80920a7a133f2a5a9a9d655f43f346f207f76c9e4b.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/356211c8e3e0b0c3b189fd80920a7a133f2a5a9a9d655f43f346f207f76c9e4b.jpg)

![450ce632f982d3892cb362be4aab929b2ea0ea955c88a3047ec19cf4781ea26e.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/450ce632f982d3892cb362be4aab929b2ea0ea955c88a3047ec19cf4781ea26e.jpg)

![45b45b074efce1c6e27c0a6b24da7f390fe17d33661351c0eb89bed1d1b56c84.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/45b45b074efce1c6e27c0a6b24da7f390fe17d33661351c0eb89bed1d1b56c84.jpg)

![4c8116a5c3ddd16700e58bc6ddf34c758cb3164bc868c5d0272b9b1aa7c7d128.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/4c8116a5c3ddd16700e58bc6ddf34c758cb3164bc868c5d0272b9b1aa7c7d128.jpg)

![555d290960f157e578e7a776448ac2f1ba7293abfe1197b62f25c5ed776c712f.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/555d290960f157e578e7a776448ac2f1ba7293abfe1197b62f25c5ed776c712f.jpg)

![5ecdda66bb1629416df5eb24ac5868b3d357c390d1370caa5219cae3aa5c405a.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/5ecdda66bb1629416df5eb24ac5868b3d357c390d1370caa5219cae3aa5c405a.jpg)

![63057f69557572b4c03e9e790f16c570d97c94a8041187a6d5aaafdd796e99ea.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/63057f69557572b4c03e9e790f16c570d97c94a8041187a6d5aaafdd796e99ea.jpg)

![651105fdf90c65d30f6731856a26708025110d8b20ab212cbf2f58b9e3518cc8.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/651105fdf90c65d30f6731856a26708025110d8b20ab212cbf2f58b9e3518cc8.jpg)

![7e6b85ac819af0fb61dd3cab4feedb40367402fff3d9c321ea885971675031c2.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/7e6b85ac819af0fb61dd3cab4feedb40367402fff3d9c321ea885971675031c2.jpg)

![8b2bed71730b1c0b7677abae1236db088f3fc37ef98d3b2ed751ef1cc795d5a7.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/8b2bed71730b1c0b7677abae1236db088f3fc37ef98d3b2ed751ef1cc795d5a7.jpg)

![8faea914d59ce7aaa45324c302350c679dc1102184064195bb0ee9b41c92eeed.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/8faea914d59ce7aaa45324c302350c679dc1102184064195bb0ee9b41c92eeed.jpg)

![988f42887ce781c937ddd7b5a6fb2b3c600587d8d8c37b66983d0ad80a9c77a3.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/988f42887ce781c937ddd7b5a6fb2b3c600587d8d8c37b66983d0ad80a9c77a3.jpg)

![c2cb05a5cd3ecaedca0d4c16030d2bff113382cc4e6b9a9e45e0ed3465e6e48c.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/c2cb05a5cd3ecaedca0d4c16030d2bff113382cc4e6b9a9e45e0ed3465e6e48c.jpg)

![c2cf4b66de2b988ce3d93da5085754c783a795ba4289972f58cedf7d3eadf2c9.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/c2cf4b66de2b988ce3d93da5085754c783a795ba4289972f58cedf7d3eadf2c9.jpg)

![d88ba81104b84c4709ff47242a9d744bb70d7171828582fd2593e9f5bcce4d07.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/d88ba81104b84c4709ff47242a9d744bb70d7171828582fd2593e9f5bcce4d07.jpg)

![da9d2c51d586937be09313ac38d41f3fd59602c5df351f43595fade3a8b3d04e.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/da9d2c51d586937be09313ac38d41f3fd59602c5df351f43595fade3a8b3d04e.jpg)

![e0f0afa92ec994f1b39dce5078f0f1c0f73262f8efded867c0517dbd6619f54b.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/e0f0afa92ec994f1b39dce5078f0f1c0f73262f8efded867c0517dbd6619f54b.jpg)

![e555bfc30ab3b1898562133f0a2e7f050e1d65061ed0e8214980d8c4eb706354.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/e555bfc30ab3b1898562133f0a2e7f050e1d65061ed0e8214980d8c4eb706354.jpg)

![e83d0da23419f256443c8e2d2d922a238f5cc5af0673463f3275aa689231ff81.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/e83d0da23419f256443c8e2d2d922a238f5cc5af0673463f3275aa689231ff81.jpg)

![ec1fd9af84602e89de9d3d4881694473eedf92aec4ce01b3b74ede5cf32b2789.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/ec1fd9af84602e89de9d3d4881694473eedf92aec4ce01b3b74ede5cf32b2789.jpg)

![ee6b372eff31a0ed5172880e3d0b19211847ea6af30473bc2f63e66212b80f02.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/ee6b372eff31a0ed5172880e3d0b19211847ea6af30473bc2f63e66212b80f02.jpg)

![ef2f4be537834321cd21992afef8a8680e7588e04f85aef0cbb8f22ca6ff0e89.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/ef2f4be537834321cd21992afef8a8680e7588e04f85aef0cbb8f22ca6ff0e89.jpg)

![f3e6ba283bdbb75a5793a32ddfd33a0386dfef7fdd4641eb0eb7b0c73bb940c5.jpg](../icml_results/6083_Attend%20to%20Anything_%20Foundation%20Model%20for%20Unified%20Human%20Attention%20Modeling/tables/f3e6ba283bdbb75a5793a32ddfd33a0386dfef7fdd4641eb0eb7b0c73bb940c5.jpg)

<a id="paper-6034"></a>

## Weaving in the Clouds: Achieving Synergistic Collaboration among LLM Agents via Federated Learning


### Images

![2a43d48f4898e714d1028ee0bcf17ff4560fcd0bbd16886a398abc5f739363a1.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/images/2a43d48f4898e714d1028ee0bcf17ff4560fcd0bbd16886a398abc5f739363a1.jpg)

![3a094d49c056c3a8ae8298d3a962b6bfcb9222a23a6ccd18c6dbe80d8785ed3f.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/images/3a094d49c056c3a8ae8298d3a962b6bfcb9222a23a6ccd18c6dbe80d8785ed3f.jpg)

![5e3aa92f5f98d86b35892b3f01ad0166aa027f5187f47e07174d57b530486604.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/images/5e3aa92f5f98d86b35892b3f01ad0166aa027f5187f47e07174d57b530486604.jpg)

### Tables

![1906ea9006423433c4b6dab16767fe7ade54723e6b2ca235bb64b0a41298dbcc.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/1906ea9006423433c4b6dab16767fe7ade54723e6b2ca235bb64b0a41298dbcc.jpg)

![28ab139455f80746a790a76fca0bb0553d47c12341e955d64cb65e536fa8f60d.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/28ab139455f80746a790a76fca0bb0553d47c12341e955d64cb65e536fa8f60d.jpg)

![2a5ce3c4fc200c62c27b3494435e5a1f5abc7d5f5b02ceb1e748248617ee4f78.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/2a5ce3c4fc200c62c27b3494435e5a1f5abc7d5f5b02ceb1e748248617ee4f78.jpg)

![3ce4e02bb4952da258a705b91f23b56b9bfba9e01cab692e4648f28eb9630db0.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/3ce4e02bb4952da258a705b91f23b56b9bfba9e01cab692e4648f28eb9630db0.jpg)

![41d3ee27a55999f8355c59df7534fd83e14034954466ffb9f8f49f63125e0891.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/41d3ee27a55999f8355c59df7534fd83e14034954466ffb9f8f49f63125e0891.jpg)

![6f6925d7aa3ccbe8fffa8ff73f65f5e33632c774ea948f5348aea15f03cf7ccd.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/6f6925d7aa3ccbe8fffa8ff73f65f5e33632c774ea948f5348aea15f03cf7ccd.jpg)

![710c4104bdde73062f7cd52f480ebb0b63b12acba7441d1cad4d62204f3a918c.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/710c4104bdde73062f7cd52f480ebb0b63b12acba7441d1cad4d62204f3a918c.jpg)

![7e1ff4c22431adae6d074ca76c0b5fdea112f3e256b230d89c83700e0010ea49.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/7e1ff4c22431adae6d074ca76c0b5fdea112f3e256b230d89c83700e0010ea49.jpg)

![7fe9b1c3ae40ae956fd78541b6224c7781908766436537dd493c614f776a9f38.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/7fe9b1c3ae40ae956fd78541b6224c7781908766436537dd493c614f776a9f38.jpg)

![8d6331b8df35c0c75128c7f92b4e2c5827575b695a667aca7de4b1487f0c90ea.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/8d6331b8df35c0c75128c7f92b4e2c5827575b695a667aca7de4b1487f0c90ea.jpg)

![9ccb1d4132a89b9b234fd7611363de2c7da3d9524596fabc690fadc1e3913c68.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/9ccb1d4132a89b9b234fd7611363de2c7da3d9524596fabc690fadc1e3913c68.jpg)

![a74f5fe7ea88c6fde0c12ea6277f20341f3304f98e6647df0ff74351042a7c6b.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/a74f5fe7ea88c6fde0c12ea6277f20341f3304f98e6647df0ff74351042a7c6b.jpg)

![a7c817d65912a1307d021529c9e06996fecda38d4f71d1afe9478e5bbbcb8f85.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/a7c817d65912a1307d021529c9e06996fecda38d4f71d1afe9478e5bbbcb8f85.jpg)

![a80c2cad4b5da716152f217a0590c152a52d60754f3894830b30556936ccd4ae.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/a80c2cad4b5da716152f217a0590c152a52d60754f3894830b30556936ccd4ae.jpg)

![afddfca185f8889ca6730c49edd9ba9fb6447006f9fd71e0d1ebea832741f1e3.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/afddfca185f8889ca6730c49edd9ba9fb6447006f9fd71e0d1ebea832741f1e3.jpg)

![b0b4f47150652ab3db3cb75a09fcccf9bf9d89e4b4de1245692b780ffa0124d3.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/b0b4f47150652ab3db3cb75a09fcccf9bf9d89e4b4de1245692b780ffa0124d3.jpg)

![b54e126c4d94344c338953c7f8a6a0e80b7f48421a9195e643fbf1f4a5160c77.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/b54e126c4d94344c338953c7f8a6a0e80b7f48421a9195e643fbf1f4a5160c77.jpg)

![cf77009337b46a8dd92ab03daddfc3cc3302303c49e7f83ab1db2de4d88c1b00.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/cf77009337b46a8dd92ab03daddfc3cc3302303c49e7f83ab1db2de4d88c1b00.jpg)

![e1e23f80a39da089d3dd175f5f75d28fc72018ea8a5bcf523f1a33e0a8665a17.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/e1e23f80a39da089d3dd175f5f75d28fc72018ea8a5bcf523f1a33e0a8665a17.jpg)

![ea9ed95ed2ae2bf4dadf6c76bc40920fb60f959b7070d09e1a86fe426870376f.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/ea9ed95ed2ae2bf4dadf6c76bc40920fb60f959b7070d09e1a86fe426870376f.jpg)

![f5b3f586c9071b5e94161778ab9874ead533962f3537f2829b3f2541a58c05a5.jpg](../icml_results/6084_Weaving%20in%20the%20Clouds_%20Achieving%20Synergistic%20Collaboration%20among%20LLM%20Agents%20via%20Federated%20Learning/tables/f5b3f586c9071b5e94161778ab9874ead533962f3537f2829b3f2541a58c05a5.jpg)

<a id="paper-6035"></a>

## Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


### Images

![198c89895fbcf8453a25c0168cc4c80393bbaa3fe0fbc2fe2c4b815055c9af07.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/images/198c89895fbcf8453a25c0168cc4c80393bbaa3fe0fbc2fe2c4b815055c9af07.jpg)

![39d58de841c188cf8b78ea9b5b3ddb932381662fe7b17ccbc5b08b46661fe11e.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/images/39d58de841c188cf8b78ea9b5b3ddb932381662fe7b17ccbc5b08b46661fe11e.jpg)

![456a0fbc2afd3daaca4dbf0be45a868a78a3f84a2efcaca4fe9c74b62e217b4a.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/images/456a0fbc2afd3daaca4dbf0be45a868a78a3f84a2efcaca4fe9c74b62e217b4a.jpg)

![f0f18570935478f73216e5689e0321cd5e2d3315d4f8d16835eb36a7201fabae.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/images/f0f18570935478f73216e5689e0321cd5e2d3315d4f8d16835eb36a7201fabae.jpg)

### Tables

![0ac2a2de86d1e1b746286fc9c53d71a1f09f8fac4ca37c1e85b1383ba1d6a9e0.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/0ac2a2de86d1e1b746286fc9c53d71a1f09f8fac4ca37c1e85b1383ba1d6a9e0.jpg)

![40bc0987c7eb3e7d1e34723df59a97f4ab52df0cea6215bb1eca2d80e06750a4.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/40bc0987c7eb3e7d1e34723df59a97f4ab52df0cea6215bb1eca2d80e06750a4.jpg)

![52a141cafae2ce6b621303081c97a359e785aa6f6be6aa8a498210be33710bb7.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/52a141cafae2ce6b621303081c97a359e785aa6f6be6aa8a498210be33710bb7.jpg)

![7ca6139a5fda45463bf4631f29fcc44e8d5ba2a2abeb0708ffd0d4908cc021bf.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/7ca6139a5fda45463bf4631f29fcc44e8d5ba2a2abeb0708ffd0d4908cc021bf.jpg)

![98e1a7bebf9dfc1ed0096e205f83c0245b8fa8f3afcd0aa0abdcf381937738f6.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/98e1a7bebf9dfc1ed0096e205f83c0245b8fa8f3afcd0aa0abdcf381937738f6.jpg)

![a9aca7bf7982955112b064c6ea134f6ef0e3729215e8c3244543643c282d8013.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/a9aca7bf7982955112b064c6ea134f6ef0e3729215e8c3244543643c282d8013.jpg)

![ba76450b91a937e3b51db599d9088bf397b0774bb1e49f02da4de0daf3f76e00.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/ba76450b91a937e3b51db599d9088bf397b0774bb1e49f02da4de0daf3f76e00.jpg)

![bb9437e3019e529110beea9b8989c816a04342ae348204555a1cfc62c6bf96f2.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/bb9437e3019e529110beea9b8989c816a04342ae348204555a1cfc62c6bf96f2.jpg)

![c56c294cdff1bd38eba01f52a0a7b6af68f57726647a3775688cb99563a7c066.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/c56c294cdff1bd38eba01f52a0a7b6af68f57726647a3775688cb99563a7c066.jpg)

![fa1641f900d8dce06daae467af83fd4292462ef3fe869d36010271e044094543.jpg](../icml_results/6085_Self-Distilled%20Reasoner_%20On-Policy%20Self-Distillation%20for%20Large%20Language%20Models/tables/fa1641f900d8dce06daae467af83fd4292462ef3fe869d36010271e044094543.jpg)

<a id="paper-6036"></a>

## Broadening the Backdoor Basin: Understanding LLM Backdoors Collapse and Making Backdoors Persistent


### Images

![048ef48fa5b1148d1a70a37a6c57a34d1201bb19ff8d2bfe5a955e7ae55f4aea.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/048ef48fa5b1148d1a70a37a6c57a34d1201bb19ff8d2bfe5a955e7ae55f4aea.jpg)

![065e2eb0c24489494fe5a07488479517b3a5d88151873c20ecf024bc94ec3ef5.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/065e2eb0c24489494fe5a07488479517b3a5d88151873c20ecf024bc94ec3ef5.jpg)

![16c6d0daffdd920c13806310da60f9c0632c02ae92f9769f26a9952944e46cdd.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/16c6d0daffdd920c13806310da60f9c0632c02ae92f9769f26a9952944e46cdd.jpg)

![177cd8f6ab595e5d562a2238bbbd92f81c574d5876e92be81e1697682687a36e.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/177cd8f6ab595e5d562a2238bbbd92f81c574d5876e92be81e1697682687a36e.jpg)

![258cdeba9f2e56cb98c549908964f3aa3bca951df16880a4af40c63b88ee52b7.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/258cdeba9f2e56cb98c549908964f3aa3bca951df16880a4af40c63b88ee52b7.jpg)

![6b81a49d824ba0bfb107f989a23123232e78f692d3f614476256e00d3532fd87.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/6b81a49d824ba0bfb107f989a23123232e78f692d3f614476256e00d3532fd87.jpg)

![7a9d24490978ec7c3d7d9bb47362ac38b51ce99beea43ce088d0788cd5a3845e.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/7a9d24490978ec7c3d7d9bb47362ac38b51ce99beea43ce088d0788cd5a3845e.jpg)

![7af23c3eec679908b11f350870698142e636f8efc6eee3f6250ce277b7e723a1.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/7af23c3eec679908b11f350870698142e636f8efc6eee3f6250ce277b7e723a1.jpg)

![80045de58481494131554ac1834e18a00e01f5dc7c285ade288391d0d619ff75.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/80045de58481494131554ac1834e18a00e01f5dc7c285ade288391d0d619ff75.jpg)

![820eb5613d97b39d9fd94fd789d0e8bf542196281d95fe9c5f7a3b82279e2a8a.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/820eb5613d97b39d9fd94fd789d0e8bf542196281d95fe9c5f7a3b82279e2a8a.jpg)

![8986890839664bbdcd6bc992a22c7e665567ac75d8623f554a3cb4e7ee65a5ce.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/8986890839664bbdcd6bc992a22c7e665567ac75d8623f554a3cb4e7ee65a5ce.jpg)

![9937720434687578aed780f3de5a7fe58d6277b7d2e6e9d55e69b63e60da8cd8.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/9937720434687578aed780f3de5a7fe58d6277b7d2e6e9d55e69b63e60da8cd8.jpg)

![c2be66ee0b24568dd87fa39a90e03862ee3fedb3ad7a2cbfb5bfaffe5a78bab6.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/c2be66ee0b24568dd87fa39a90e03862ee3fedb3ad7a2cbfb5bfaffe5a78bab6.jpg)

![d9bddeddca8971171fb4a879186423625372817b7f55977aa7cd7c60e19141d2.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/d9bddeddca8971171fb4a879186423625372817b7f55977aa7cd7c60e19141d2.jpg)

![f6bcb5577a1642f5b0f5a1d53bb12c2416120bc891f78fe4b20b671d745c7ead.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/images/f6bcb5577a1642f5b0f5a1d53bb12c2416120bc891f78fe4b20b671d745c7ead.jpg)

### Tables

![3c57ad6365d69343e015909c600d4e151a7f7bdc7bb277d4804ac4f8d9e0d6ab.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/tables/3c57ad6365d69343e015909c600d4e151a7f7bdc7bb277d4804ac4f8d9e0d6ab.jpg)

![47f67cfbc66927f174d4f7492c9b7b346f28028c2af8c474d269a42a1279b99b.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/tables/47f67cfbc66927f174d4f7492c9b7b346f28028c2af8c474d269a42a1279b99b.jpg)

![a71d980fbcd34195b2935f7bdd41ccd843f629f0d33ed570af010c4c9b34e7ce.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/tables/a71d980fbcd34195b2935f7bdd41ccd843f629f0d33ed570af010c4c9b34e7ce.jpg)

![fe4e45c72584bf915b092c9be1217b38278c248bd7f400b6f4bd10cea33967d1.jpg](../icml_results/6086_Broadening%20the%20Backdoor%20Basin_%20Understanding%20LLM%20Backdoors%20Collapse%20and%20Making%20Backdoors%20Persistent/tables/fe4e45c72584bf915b092c9be1217b38278c248bd7f400b6f4bd10cea33967d1.jpg)

<a id="paper-6037"></a>

## DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise


### Images

![09f3cb215caa703da67bd425e9a8e771d339f4438f7fe0266a67deec551d8019.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/09f3cb215caa703da67bd425e9a8e771d339f4438f7fe0266a67deec551d8019.jpg)

![0cec1539c765e41ba79346917c106e3a1eaec814228ad30c3c0b4fbea99b8dc7.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/0cec1539c765e41ba79346917c106e3a1eaec814228ad30c3c0b4fbea99b8dc7.jpg)

![1c07002b13d9eb238b3b7b7970925c4268a877d0796a7c5df81c5a72b897cb6d.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/1c07002b13d9eb238b3b7b7970925c4268a877d0796a7c5df81c5a72b897cb6d.jpg)

![20d903529e9cd42c49744b0ba0fd66f777868a9f1a78b5ba13e58dfc2f7b3f26.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/20d903529e9cd42c49744b0ba0fd66f777868a9f1a78b5ba13e58dfc2f7b3f26.jpg)

![56db64569dac9bb5aa2dbc445036738a07e7705c3eb1cce9631befb949f93303.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/56db64569dac9bb5aa2dbc445036738a07e7705c3eb1cce9631befb949f93303.jpg)

![6d366ae4b78d9c29c916ae6cdd8ac3b87fa6a1683485d6411cff9cd8d486f971.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/6d366ae4b78d9c29c916ae6cdd8ac3b87fa6a1683485d6411cff9cd8d486f971.jpg)

![73384be190fae2f89fc4f94c0e156d04bee192fb3633d02e4675a9bdc9d2e412.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/73384be190fae2f89fc4f94c0e156d04bee192fb3633d02e4675a9bdc9d2e412.jpg)

![7b3ddf672466e0837200a5c50b472c329029e925a9ec74d542fefb3d3b622673.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/7b3ddf672466e0837200a5c50b472c329029e925a9ec74d542fefb3d3b622673.jpg)

![d5fb2c4c457a5ddea5653a3050425cd000a211f10fd8fd07de1ec476761823e7.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/d5fb2c4c457a5ddea5653a3050425cd000a211f10fd8fd07de1ec476761823e7.jpg)

![e069678d126032ec2f0db2a90536dd05edf9defe36b0d6ebbbdf411f94ce15bf.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/images/e069678d126032ec2f0db2a90536dd05edf9defe36b0d6ebbbdf411f94ce15bf.jpg)

### Tables

![2f0c3363154b5106518e4b06355a34f65f654df86e05543fa364ae9bb80a2c2f.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/2f0c3363154b5106518e4b06355a34f65f654df86e05543fa364ae9bb80a2c2f.jpg)

![4d94a5a3627f23c95e9f5095e0d5cb94fb3346bd238889144ec8eb1c6d338ec8.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/4d94a5a3627f23c95e9f5095e0d5cb94fb3346bd238889144ec8eb1c6d338ec8.jpg)

![5fbd68b9107c4b6cd9720c4543dd561c4121cae165f14d353eeb9ffea8525ace.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/5fbd68b9107c4b6cd9720c4543dd561c4121cae165f14d353eeb9ffea8525ace.jpg)

![626dd2069dae814a1e8b31a1f12a11c41e20f1d6c7d45eff203545d39ea1d392.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/626dd2069dae814a1e8b31a1f12a11c41e20f1d6c7d45eff203545d39ea1d392.jpg)

![732dc00f8a58cacc783140af93d884e16b25b4cd74a35b2725e2636095194454.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/732dc00f8a58cacc783140af93d884e16b25b4cd74a35b2725e2636095194454.jpg)

![7368eac4cec4d0892ac95b613d8ea31907dc742f00b786aea8ec271eabaf92fb.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/7368eac4cec4d0892ac95b613d8ea31907dc742f00b786aea8ec271eabaf92fb.jpg)

![84b950692beb332a096e4171631d19454670e1ec013b3636fcf62b7d38f16337.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/84b950692beb332a096e4171631d19454670e1ec013b3636fcf62b7d38f16337.jpg)

![875039c10c8fef49e9a0de1ae1f5a6ddc51699050479e3d45d5dce8a81cc7821.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/875039c10c8fef49e9a0de1ae1f5a6ddc51699050479e3d45d5dce8a81cc7821.jpg)

![887c7eda00b43327f13a80541ef4aff147a84dc3bf15e300a1ba715952e8e229.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/887c7eda00b43327f13a80541ef4aff147a84dc3bf15e300a1ba715952e8e229.jpg)

![91c149f911fdd1ec5dac4be05586ce534e13379a28f3e3aff86628596a8aa2ac.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/91c149f911fdd1ec5dac4be05586ce534e13379a28f3e3aff86628596a8aa2ac.jpg)

![9c3cc20c809c85237e527c798521ef4c8161bde6cf6d1ac2049fb7090251405d.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/9c3cc20c809c85237e527c798521ef4c8161bde6cf6d1ac2049fb7090251405d.jpg)

![a33db5141e1cd72ed33e694751886fe31b9971dfc71c7639dc4230ccf8e58956.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/a33db5141e1cd72ed33e694751886fe31b9971dfc71c7639dc4230ccf8e58956.jpg)

![b0c375ea77f7fbc89d26b23893fa0c04b06cd778a153c54272147c0bc5aa3607.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/b0c375ea77f7fbc89d26b23893fa0c04b06cd778a153c54272147c0bc5aa3607.jpg)

![d209d6d0ed0d394f4e3b39a08fb07d05b3524d6f04fe0d94271172ad4fbf8191.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/d209d6d0ed0d394f4e3b39a08fb07d05b3524d6f04fe0d94271172ad4fbf8191.jpg)

![f4be67b8c5736634bfd9b4f64dcbee6efb30e5908d272882dac115d6f6158282.jpg](../icml_results/6087_DREAM_%20Dual-Standard%20Semantic%20Homogeneity%20with%20Dynamic%20Optimization%20for%20Graph%20Learning%20with%20Label%20No/tables/f4be67b8c5736634bfd9b4f64dcbee6efb30e5908d272882dac115d6f6158282.jpg)

<a id="paper-6038"></a>

## AutoMS: Multi-Agent Evolutionary Search for Cross-Physics Inverse Microstructure Design


### Images

![265b4ac669e9e8f16c51731d6a2524e183d6db65fb330a12931e247d4d1639cf.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/265b4ac669e9e8f16c51731d6a2524e183d6db65fb330a12931e247d4d1639cf.jpg)

![2eb18f96f5f08f976530c1cd41b9375681daa0482a3df766808bb83631fe7343.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/2eb18f96f5f08f976530c1cd41b9375681daa0482a3df766808bb83631fe7343.jpg)

![510470849de6d2cc01951d730ecd5685b17af312ff60abca8a52b73634957187.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/510470849de6d2cc01951d730ecd5685b17af312ff60abca8a52b73634957187.jpg)

![643f5651c1d4ae0992ead7aabe0f3b7832cf11f24a1c774ddc7dbd7a2d0174d4.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/643f5651c1d4ae0992ead7aabe0f3b7832cf11f24a1c774ddc7dbd7a2d0174d4.jpg)

![8904c9861b8943015e837a8d497d246432694e9fe246aa5b2190b6400b9e4d43.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/8904c9861b8943015e837a8d497d246432694e9fe246aa5b2190b6400b9e4d43.jpg)

![d1b25c2be0facec7cfb162dde5f12ccd6bcc23f9c3cef375085a38a9c92f2314.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/images/d1b25c2be0facec7cfb162dde5f12ccd6bcc23f9c3cef375085a38a9c92f2314.jpg)

### Tables

![29070b1b3f62dbaf6a2f56a76a4094cb6413de872eb60e79ed091d8180b3fd03.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/29070b1b3f62dbaf6a2f56a76a4094cb6413de872eb60e79ed091d8180b3fd03.jpg)

![2e2f9661a7bf76581bb51ee1b9913c75429dfb67abc51bc7b8ede6089f22fa79.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/2e2f9661a7bf76581bb51ee1b9913c75429dfb67abc51bc7b8ede6089f22fa79.jpg)

![41aabb978473d6ff304f80f857a5964797bab7697ca43a16862f25630f82901c.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/41aabb978473d6ff304f80f857a5964797bab7697ca43a16862f25630f82901c.jpg)

![58416376637f562aa9c7cf6c5556d9bf5f033510f74a8545e716f3554d3ad339.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/58416376637f562aa9c7cf6c5556d9bf5f033510f74a8545e716f3554d3ad339.jpg)

![6ee454a546684c6e588a5cb9a88758b6d41605662585345c356eb09c8ca9f9d2.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/6ee454a546684c6e588a5cb9a88758b6d41605662585345c356eb09c8ca9f9d2.jpg)

![8a8a21e7a75b7cc26738f92e81ae7b0e5814889f354f20d15a7473e5ed11fcc4.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/8a8a21e7a75b7cc26738f92e81ae7b0e5814889f354f20d15a7473e5ed11fcc4.jpg)

![bcd7a021a7e0efe8ca09fdf041822d17e7dcfea1ec1ddff922b99e68b5bca1f8.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/bcd7a021a7e0efe8ca09fdf041822d17e7dcfea1ec1ddff922b99e68b5bca1f8.jpg)

![c9a0e3106589783009731b1896835bf65c8a533a359047b58e60e147c3b1369a.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/c9a0e3106589783009731b1896835bf65c8a533a359047b58e60e147c3b1369a.jpg)

![f94ff484247a271e191b8dc1bfaa5fa52521c1e37f06554c4d57d717212124e7.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/f94ff484247a271e191b8dc1bfaa5fa52521c1e37f06554c4d57d717212124e7.jpg)

![ffc7072b0f061466de83a3d8409cedf01a4b1137755e55d293ee54fcd2f729a0.jpg](../icml_results/6088_AutoMS_%20Multi-Agent%20Evolutionary%20Search%20for%20Cross-Physics%20Inverse%20Microstructure%20Design/tables/ffc7072b0f061466de83a3d8409cedf01a4b1137755e55d293ee54fcd2f729a0.jpg)

<a id="paper-6039"></a>

## TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization

### Images

![09a8d54d3b4caf1b3d94187cf96a5083bf931c0bac647b2b10f590f836990cb1.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/09a8d54d3b4caf1b3d94187cf96a5083bf931c0bac647b2b10f590f836990cb1.jpg)

![15ad892d4513aa2b617a80465f99c3a2f1b41200752b9521fa1a0550539a18e8.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/15ad892d4513aa2b617a80465f99c3a2f1b41200752b9521fa1a0550539a18e8.jpg)

![19cb4499d6658f57301aad44d27be29bfba96088366038a8799f59e29b81212c.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/19cb4499d6658f57301aad44d27be29bfba96088366038a8799f59e29b81212c.jpg)

![61b8385170d8052ce46132498c339c7d68ebaf6d3034b9c7ef26297c20fe0417.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/61b8385170d8052ce46132498c339c7d68ebaf6d3034b9c7ef26297c20fe0417.jpg)

![65c8e76353e3fc0b7411595c51b851effe3a0bd5cbda82cec10800c61a4b50bc.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/65c8e76353e3fc0b7411595c51b851effe3a0bd5cbda82cec10800c61a4b50bc.jpg)

![679b8cd571145d143422f4a101b825cb81ce1187fbaadc634709f6532d1c1dd9.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/679b8cd571145d143422f4a101b825cb81ce1187fbaadc634709f6532d1c1dd9.jpg)

![aa4f0cb0b3bc46a4f3e891ca0623d4e6be8ada24bd8d53f5b768d6d79c157e50.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/aa4f0cb0b3bc46a4f3e891ca0623d4e6be8ada24bd8d53f5b768d6d79c157e50.jpg)

![b337c335f0d004c2db82fc1ff242f0078a4d1c1a3b44b7daab851c62bc3cd0a8.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/b337c335f0d004c2db82fc1ff242f0078a4d1c1a3b44b7daab851c62bc3cd0a8.jpg)

![c643a1022bd6ab568edcf41b86eac5927729cc898f61ff9708c6e807138fabb2.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/c643a1022bd6ab568edcf41b86eac5927729cc898f61ff9708c6e807138fabb2.jpg)

![c9fc2fee683adc6fbda5c66836bd4a8c813c6c802e9f6a1a7e9229e226990b49.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/images/c9fc2fee683adc6fbda5c66836bd4a8c813c6c802e9f6a1a7e9229e226990b49.jpg)

### Tables

![153b399bc93a5dbc7deb97ab4e487f5fe924feeb72797ead2875778c1c0eca93.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/153b399bc93a5dbc7deb97ab4e487f5fe924feeb72797ead2875778c1c0eca93.jpg)

![164b9abb6e66106d5aa02dc71136f66e5a16cf2bbe27bbb96db88aa0706792fd.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/164b9abb6e66106d5aa02dc71136f66e5a16cf2bbe27bbb96db88aa0706792fd.jpg)

![1744bc4c131d85d483889e5f35a96250689847a51c85dd13d4c5272d6e47ef5a.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/1744bc4c131d85d483889e5f35a96250689847a51c85dd13d4c5272d6e47ef5a.jpg)

![2a3d6fac4bd7d9aab1b394e44306ec29d334baf6f20128c36c5f1357464f60d9.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/2a3d6fac4bd7d9aab1b394e44306ec29d334baf6f20128c36c5f1357464f60d9.jpg)

![980b1ae03eb43873bd0ae31a7cf40e77de10bace94155664d2450cedc1ed29f5.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/980b1ae03eb43873bd0ae31a7cf40e77de10bace94155664d2450cedc1ed29f5.jpg)

![b2c2b60bee9c8c5b5632c1900b3e04d8c9afa00c09db8d983decd5a2a3799a58.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/b2c2b60bee9c8c5b5632c1900b3e04d8c9afa00c09db8d983decd5a2a3799a58.jpg)

![c54a16c819e29302d067d837d7d4b16c2ed72463e1e46c8d5db094ce007a0317.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/c54a16c819e29302d067d837d7d4b16c2ed72463e1e46c8d5db094ce007a0317.jpg)

![d49d5b11df18720925288bd21eb91e07102f3f34926b0be941ede022f0d79ac5.jpg](../icml_results/6089_TWLA_%20Achieving%20Ternary%20Weights%20and%20Low-Bit%20Activations%20for%20LLMs%20via%20Post-Training%20Quantization/tables/d49d5b11df18720925288bd21eb91e07102f3f34926b0be941ede022f0d79ac5.jpg)
