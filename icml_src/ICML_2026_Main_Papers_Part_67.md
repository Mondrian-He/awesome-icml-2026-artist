# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 67 of 197

## 目录 (Table of Contents)

1. [An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale](#paper-2179)
2. [Theoretical Perspectives on Data Quality and Synergistic Effects in Pre- and Post-Training Reasoning Models](#paper-2180)
3. [CaliDist: Calibrating Large Language Models via Behavioral Robustness to Distraction](#paper-2181)
4. [Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](#paper-2182)
5. [MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](#paper-2183)
6. [Cerebellar-Inspired Residual Control for Fault Recovery: From Inference-Time Adaptation to Structural Consolidation](#paper-2184)
7. [Generative Large Neighborhood Search: Scalable Set Cover Optimization via Discrete Diffusion](#paper-2185)
8. [Factorized Scheduling Principle: Learning Interpretable and Transferable Policies via Structured Additive Functions](#paper-2186)
9. [Decision-focused Sparse Tangent Portfolio Optimization](#paper-2187)
10. [LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents](#paper-2188)
11. [Vision-aligned Latent Reasoning for Multi-modal Large Language Model](#paper-2189)
12. [Towards Functional Correctness of Code Models with Selective Generation](#paper-2190)
13. [When Do Diffusion Models Learn to Generate Multiple Objects?](#paper-2191)
14. [Context-free Recognition with Transformers](#paper-2192)
15. [MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](#paper-2193)
16. [Tiny Brains, Giant Impact: Uncovering the Keystone Neurons of LLM with Just a Few Prompts](#paper-2194)
17. [ShapCCS: Shapley-Driven Client Coreset Selection in Federated Learning](#paper-2195)
18. [TopBench: A Benchmark for Implicit Predictive Reasoning in Tabular Question Answering](#paper-2196)
19. [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](#paper-2197)
20. [Recovering Hidden Reward in Diffusion-Based Policies](#paper-2198)
21. [Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles](#paper-2199)
22. [DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing](#paper-2200)
23. [Learn-to-learn on Arbitrary Textual Conditioning: A Hypernetwork-Driven Meta-gated LLM](#paper-2201)
24. [Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening](#paper-2202)
25. [HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control](#paper-2203)
26. [Zero-Shot Text-to-Motion Evaluation using Video Language Models](#paper-2204)
27. [SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling](#paper-2205)
28. [Near-Optimal Regret for KL-Regularized Multi-Armed Bandits](#paper-2206)
29. [Sparse ActionGen: Accelerating Diffusion Policy with Real-time Pruning](#paper-2207)
30. [VT-Bench: A Unified Benchmark for Visual-Tabular Multi-Modal Learning](#paper-2208)
31. [Quantitative Estimation of Target Task Performance from Unsupervised Pretext Task in Semi/Self-Supervised Learning](#paper-2209)
32. [From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models](#paper-2210)
33. [EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models](#paper-2211)

---


<a id="paper-2179"></a>

## An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale

### Images

![02879636edf1ca37873931276e4b3e145367a2a2e236a4f4b63f1002716f5f50.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/02879636edf1ca37873931276e4b3e145367a2a2e236a4f4b63f1002716f5f50.jpg)

![0bfb31797be6b5cf58277fc027d832f5a7cf8a1675ffcb90843ada5f63cd9fba.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/0bfb31797be6b5cf58277fc027d832f5a7cf8a1675ffcb90843ada5f63cd9fba.jpg)

![2b64e930cad20b1de3cf3478c2a982203dbe627cbebebefdc6945c6bf337e99b.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/2b64e930cad20b1de3cf3478c2a982203dbe627cbebebefdc6945c6bf337e99b.jpg)

![4582d9adb5106b144a4de4e5370a6282b2a686e20fb7fcbccea34ec8fc85c467.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/4582d9adb5106b144a4de4e5370a6282b2a686e20fb7fcbccea34ec8fc85c467.jpg)

![51867b3289857c26577c9185f1b083c4d301b3fcc3c3dcf09ef62ab802ffeabe.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/51867b3289857c26577c9185f1b083c4d301b3fcc3c3dcf09ef62ab802ffeabe.jpg)

![bd9a082254f9d7878d8d03170394faf334cda74b39d29cf6c69d592a5d47ea35.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/bd9a082254f9d7878d8d03170394faf334cda74b39d29cf6c69d592a5d47ea35.jpg)

![e82893820b3585e39334a0e2ca66898a6b32845c378fdde2ca8dbabf354185f3.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/images/e82893820b3585e39334a0e2ca66898a6b32845c378fdde2ca8dbabf354185f3.jpg)

### Tables

![005021fc97c0b09b672b96102da980d3eb9744973f7e54ad356c85fd4f70a3e8.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/005021fc97c0b09b672b96102da980d3eb9744973f7e54ad356c85fd4f70a3e8.jpg)

![08db5c288ae25638c94db9652db8c73b4a9b8a27817e10d53ae06f7789743a68.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/08db5c288ae25638c94db9652db8c73b4a9b8a27817e10d53ae06f7789743a68.jpg)

![2c09aef9742a149f83c98d44919ad1d5add646c5b6c943d5b1637314de5588df.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/2c09aef9742a149f83c98d44919ad1d5add646c5b6c943d5b1637314de5588df.jpg)

![8fa0decc644c341feacf50819794288d449915fa87c7863069e65ffce1ac7967.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/8fa0decc644c341feacf50819794288d449915fa87c7863069e65ffce1ac7967.jpg)

![bab7dd9a45066243ccf30bb202fe6ead70f3d32a3fd16b3e44f0308b87238c76.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/bab7dd9a45066243ccf30bb202fe6ead70f3d32a3fd16b3e44f0308b87238c76.jpg)

![c98198d0d1b941fd116d3a46f65446d411c386d466728e54fccc7752595461a6.jpg](../icml_results/2207_Towards%20Effective%20Waste%20Segmentation%20for%20Automated%20Waste%20Recycling%20in%20Cluttered%20Background/tables/c98198d0d1b941fd116d3a46f65446d411c386d466728e54fccc7752595461a6.jpg)

## An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale


### Images

![1c5b9e979ebeb14f238890754d3d8b35a49f87494b4a4ac1765eb29bcb061407.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/1c5b9e979ebeb14f238890754d3d8b35a49f87494b4a4ac1765eb29bcb061407.jpg)

![21738b72bc0eb82e59a46045de7127ecaa81b2b06489ab3c823a1f8858ec2f6a.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/21738b72bc0eb82e59a46045de7127ecaa81b2b06489ab3c823a1f8858ec2f6a.jpg)

![57fcfd42629c4310b69d8da6e5cf84927aa0bab5b0edce92d20b5d70754ffd0a.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/57fcfd42629c4310b69d8da6e5cf84927aa0bab5b0edce92d20b5d70754ffd0a.jpg)

![68928a2fecdb25ed523d539d6eae8fd0a8418bd16b0b0402ab51d0ccc9bfd2d6.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/68928a2fecdb25ed523d539d6eae8fd0a8418bd16b0b0402ab51d0ccc9bfd2d6.jpg)

![712af29d0ce9beebd708ec586a0a0aad5f8fe7a1d552968e921234ca9119b0ed.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/712af29d0ce9beebd708ec586a0a0aad5f8fe7a1d552968e921234ca9119b0ed.jpg)

![d498d4573789088d6b27baf78a3b8bb65cd9ad34e50cf60f0b4468e9f769054a.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/d498d4573789088d6b27baf78a3b8bb65cd9ad34e50cf60f0b4468e9f769054a.jpg)

![d59d7804aa887bc4cf55da7829f3f5d1950852ed0d31641bc9483bbd712ab309.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/d59d7804aa887bc4cf55da7829f3f5d1950852ed0d31641bc9483bbd712ab309.jpg)

![d6b7b29b935727abf52719c7271dfd6e93c433b05126b7a33731ca294154cb1c.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/d6b7b29b935727abf52719c7271dfd6e93c433b05126b7a33731ca294154cb1c.jpg)

![df3f4f29fa3b77d4a0e1746dc8c5b7c94c3d6ef289fbc34005cd8018fd76b0da.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/df3f4f29fa3b77d4a0e1746dc8c5b7c94c3d6ef289fbc34005cd8018fd76b0da.jpg)

![e66a12a6f2146f921b54866cff2be80e52799f3f924522d44a14b43fa630fb46.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/e66a12a6f2146f921b54866cff2be80e52799f3f924522d44a14b43fa630fb46.jpg)

![ea0563e826fc34b813bb40adb738dc440777d6023834ddad8fd0789738ed37db.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/images/ea0563e826fc34b813bb40adb738dc440777d6023834ddad8fd0789738ed37db.jpg)

### Tables

![885775495af464bd2314004adf623b83e58d5f9dd8c381bd2779c9744159de2a.jpg](../icml_results/2208_An%20Embarrassingly%20Simple%20Way%20to%20Optimize%20Orthogonal%20Matrices%20at%20Scale/tables/885775495af464bd2314004adf623b83e58d5f9dd8c381bd2779c9744159de2a.jpg)

<a id="paper-2180"></a>

## Theoretical Perspectives on Data Quality and Synergistic Effects in Pre- and Post-Training Reasoning Models


### Images

![05c826b736d684dc775a78bf896561675bfa9440a895f206e8ed5674622e9576.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/05c826b736d684dc775a78bf896561675bfa9440a895f206e8ed5674622e9576.jpg)

![790fe9aedc3866e497814a1d854079b1aee3bd7ce9986b72287c2e07e624af70.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/790fe9aedc3866e497814a1d854079b1aee3bd7ce9986b72287c2e07e624af70.jpg)

![846a19e12196481a09091c251841638f0360634480125c9c3d4ce0cee8ff93a4.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/846a19e12196481a09091c251841638f0360634480125c9c3d4ce0cee8ff93a4.jpg)

![a61b180b7b36afeaee552b2ba214abdee9b413a20641993096113f8d8759aa82.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/a61b180b7b36afeaee552b2ba214abdee9b413a20641993096113f8d8759aa82.jpg)

![b31ea1d8056804fce517f1af13d7aaabc39b0683be6f3e39bfbd16bda074ee7e.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/b31ea1d8056804fce517f1af13d7aaabc39b0683be6f3e39bfbd16bda074ee7e.jpg)

![ce8de795a379f9c5e1ee3ff5184eefe4e6ac7e1c3d98d40d1126f646d75ef507.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/ce8de795a379f9c5e1ee3ff5184eefe4e6ac7e1c3d98d40d1126f646d75ef507.jpg)

![f378b61c5bc275bbf0e241c14d6a75bff55a0790ccec16a2da3d11a3482da251.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/images/f378b61c5bc275bbf0e241c14d6a75bff55a0790ccec16a2da3d11a3482da251.jpg)

### Tables

![ce71e71bbf14d17f14fea486c3f224ba2d5e7c86f7abec9a4ea2af2b7656f006.jpg](../icml_results/2209_Theoretical%20Perspectives%20on%20Data%20Quality%20and%20Synergistic%20Effects%20in%20Pre-%20and%20Post-Training%20Reasoning/tables/ce71e71bbf14d17f14fea486c3f224ba2d5e7c86f7abec9a4ea2af2b7656f006.jpg)

<a id="paper-2181"></a>

## CaliDist: Calibrating Large Language Models via Behavioral Robustness to Distraction


### Images

![3118f4765b6ca3c3c9780bcc9c2b56cffc9dee361a84822074160dea5a2ccd52.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/3118f4765b6ca3c3c9780bcc9c2b56cffc9dee361a84822074160dea5a2ccd52.jpg)

![36edb35f7f13640fb8fec8c5b7f29d4787b2897eef724bae968f81227e0c8e43.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/36edb35f7f13640fb8fec8c5b7f29d4787b2897eef724bae968f81227e0c8e43.jpg)

![a282aa49fe7c8dd81a680b3c066b3f5e5a1b0308d6302dba8f858a9d4500374e.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/a282aa49fe7c8dd81a680b3c066b3f5e5a1b0308d6302dba8f858a9d4500374e.jpg)

![ac9ace4dc7deb355ce31e5b9a5152b40c9e8c150200187a3bf9f5d3a9312ffe3.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/ac9ace4dc7deb355ce31e5b9a5152b40c9e8c150200187a3bf9f5d3a9312ffe3.jpg)

![b89e5303f042806903967c7377ff3b577b0a4ae8a5b4b7df5e0bca06b976818b.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/b89e5303f042806903967c7377ff3b577b0a4ae8a5b4b7df5e0bca06b976818b.jpg)

![cd6ba6e81294049cc59d094ed61099e35543aaaba1347eafcea83d15e9b4dd5f.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/images/cd6ba6e81294049cc59d094ed61099e35543aaaba1347eafcea83d15e9b4dd5f.jpg)

### Tables

![0b4dfd6b6f241dc166c32809b4b26af30f6235cfc7b330e91b1fd01175e76697.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/0b4dfd6b6f241dc166c32809b4b26af30f6235cfc7b330e91b1fd01175e76697.jpg)

![182c6dd6209b9dd7983dd5a8d4cf083bba2d822337686003568db4cb754f99e6.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/182c6dd6209b9dd7983dd5a8d4cf083bba2d822337686003568db4cb754f99e6.jpg)

![19fe8d4616d38cdc691d9038e4449ed479917b7ae003a2fb53c0a458767dda53.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/19fe8d4616d38cdc691d9038e4449ed479917b7ae003a2fb53c0a458767dda53.jpg)

![78fc483a41f700fd95ea55a71d971f2bf280093abcf02cccdadc4eeab61919d7.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/78fc483a41f700fd95ea55a71d971f2bf280093abcf02cccdadc4eeab61919d7.jpg)

![885a9e2a0baf3582863aef33e509e798d011fc6d15ec8bbfdffd0453cd9316e1.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/885a9e2a0baf3582863aef33e509e798d011fc6d15ec8bbfdffd0453cd9316e1.jpg)

![89e29d717adcadd5d91fc9b8da501b852bd7addba90f9718e3b062f07eb32e97.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/89e29d717adcadd5d91fc9b8da501b852bd7addba90f9718e3b062f07eb32e97.jpg)

![93fde5cdf534b6e2479a8139b9b942f70cac6a371b82f2c4d348818c16f6cf05.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/93fde5cdf534b6e2479a8139b9b942f70cac6a371b82f2c4d348818c16f6cf05.jpg)

![cdffd973a4703b20e741044d63e6ce3e086468494b9d36f15ae30dd418f19b3e.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/cdffd973a4703b20e741044d63e6ce3e086468494b9d36f15ae30dd418f19b3e.jpg)

![e4223f7f1f8a5e0b2d9f492504b49d6a9c2d17f4ba86e549658a8a28829d1c7c.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/e4223f7f1f8a5e0b2d9f492504b49d6a9c2d17f4ba86e549658a8a28829d1c7c.jpg)

![ebdbc179a8f338054ab0de275c472db571e20714ef8213bf6d18317064216060.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/ebdbc179a8f338054ab0de275c472db571e20714ef8213bf6d18317064216060.jpg)

![f073388a0cd428574b616a0671521182168c5dd29bb0bcb96bef24ad2e5bcd70.jpg](../icml_results/2210_CaliDist_%20Calibrating%20Large%20Language%20Models%20via%20Behavioral%20Robustness%20to%20Distraction/tables/f073388a0cd428574b616a0671521182168c5dd29bb0bcb96bef24ad2e5bcd70.jpg)

<a id="paper-2182"></a>

## Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision


### Images

![1f52b42537386afc67c04b3536243fc8449c67b31f91742a6882be7b15da503d.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/1f52b42537386afc67c04b3536243fc8449c67b31f91742a6882be7b15da503d.jpg)

![7380f99f97b0893e3f0f410cf30ad6b89a2db1b979fc430f3f59e643004a05a2.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/7380f99f97b0893e3f0f410cf30ad6b89a2db1b979fc430f3f59e643004a05a2.jpg)

![79468ef14f0f75d58186218ab2c1fa2cb7db2dfaff441545ba44b00bf082211a.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/79468ef14f0f75d58186218ab2c1fa2cb7db2dfaff441545ba44b00bf082211a.jpg)

![c7ea8c26e7c8c22468aa305729d8949aa741d6edfd686409fa2a7747dbf057cb.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/c7ea8c26e7c8c22468aa305729d8949aa741d6edfd686409fa2a7747dbf057cb.jpg)

![e577a9a9226d321fd1aa8375ec995ec3b896012ee2cb048d7632ad76404d21b8.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/e577a9a9226d321fd1aa8375ec995ec3b896012ee2cb048d7632ad76404d21b8.jpg)

![fe64b5ca806c61ec8645b3f1fd40759d07f3a402c0e7330fd2f3768311dbf5e1.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/images/fe64b5ca806c61ec8645b3f1fd40759d07f3a402c0e7330fd2f3768311dbf5e1.jpg)

### Tables

![06b4f57606f93e27f88416a111764725c45ed3119356f830b4a1ee12072061d6.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/06b4f57606f93e27f88416a111764725c45ed3119356f830b4a1ee12072061d6.jpg)

![0d22bac1885ba49b342fcceef0028b72ae8e50afc8340eb746c9d5ce12b8c676.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/0d22bac1885ba49b342fcceef0028b72ae8e50afc8340eb746c9d5ce12b8c676.jpg)

![3eecf9560cbe9c7c0e762e2d85c20f275510f644bbae9351c4b5445bf713bc6b.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/3eecf9560cbe9c7c0e762e2d85c20f275510f644bbae9351c4b5445bf713bc6b.jpg)

![5547f9521357b0723dfaa1829ee5008dcb96bdf0a815435fc98a52be2c4613f7.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/5547f9521357b0723dfaa1829ee5008dcb96bdf0a815435fc98a52be2c4613f7.jpg)

![5957f5044b581784060a6a13db529058f1607c119ed9914cb128037c3c6f99cf.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/5957f5044b581784060a6a13db529058f1607c119ed9914cb128037c3c6f99cf.jpg)

![5e5f099ae1bbc075f5efbefc5cec106aee9970abaf78713111db93ac3d4517da.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/5e5f099ae1bbc075f5efbefc5cec106aee9970abaf78713111db93ac3d4517da.jpg)

![633bb052f31ee6b19403adbe638062294d4e9c1d1ddafc72abbc2185d484dca2.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/633bb052f31ee6b19403adbe638062294d4e9c1d1ddafc72abbc2185d484dca2.jpg)

![6c75902680c39600a6c0de2b4df30c3cdda1d664a0b64619dba733f450d1f431.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/6c75902680c39600a6c0de2b4df30c3cdda1d664a0b64619dba733f450d1f431.jpg)

![73ad231a9bf7f735a1b607c6d8f3b3acc077bc0038c7026968defd16be290308.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/73ad231a9bf7f735a1b607c6d8f3b3acc077bc0038c7026968defd16be290308.jpg)

![b39546818249971bbe0fff2e0c09eec487344ecefb8bc42ff9cb35cd4dda764c.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/b39546818249971bbe0fff2e0c09eec487344ecefb8bc42ff9cb35cd4dda764c.jpg)

![bf6280dfdb7d4f7c78d283bd6fda375a3032128c2bb527dc6cffb4f878a239e4.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/bf6280dfdb7d4f7c78d283bd6fda375a3032128c2bb527dc6cffb4f878a239e4.jpg)

![e124b08befe30a4393d569286ee205684d67f042fdd75c89c55c90201fdf7980.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/e124b08befe30a4393d569286ee205684d67f042fdd75c89c55c90201fdf7980.jpg)

![f07130da348eea1f474c748dd42c342196fc0f53b1aadde49cafb23e7ee97f33.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/f07130da348eea1f474c748dd42c342196fc0f53b1aadde49cafb23e7ee97f33.jpg)

![f18ce62302e5847e84739bbd07c52fc1b4f6f03c9dbc8642644d5bd7c97aee67.jpg](../icml_results/2211_Compute%20as%20Teacher_%20Turning%20Inference%20Compute%20Into%20Reference-Free%20Supervision/tables/f18ce62302e5847e84739bbd07c52fc1b4f6f03c9dbc8642644d5bd7c97aee67.jpg)

<a id="paper-2183"></a>

## MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training


### Images

![3585fa7aa70a28561ff3b98cb63608c56b2d54cc4d48e2b4ed34105a7403888d.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/3585fa7aa70a28561ff3b98cb63608c56b2d54cc4d48e2b4ed34105a7403888d.jpg)

![4397b437d5eb544ce2c0dcfd99f49c8020937916c129552579e7e695ae817a92.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/4397b437d5eb544ce2c0dcfd99f49c8020937916c129552579e7e695ae817a92.jpg)

![59640cd4aaff4232d8db0e48e02588671d998c1d1f4915ed455155fe6a086cc6.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/59640cd4aaff4232d8db0e48e02588671d998c1d1f4915ed455155fe6a086cc6.jpg)

![75a8825bf37b0f26f1a8637180e1b1516937907606a5ea5c07a32bc0adfb1426.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/75a8825bf37b0f26f1a8637180e1b1516937907606a5ea5c07a32bc0adfb1426.jpg)

![910b0e74080170e28fd9f243c364977bee6e767bddc5b9b49f2bc0b1d169a056.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/910b0e74080170e28fd9f243c364977bee6e767bddc5b9b49f2bc0b1d169a056.jpg)

![b38304bda343818067971112d230977b3381e3c3d8f243ac7362182f55fe8602.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/b38304bda343818067971112d230977b3381e3c3d8f243ac7362182f55fe8602.jpg)

![cd11605ceba1f3dc524314b838f6f8062d67ec6d447d8839744a94eb1f32f145.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/cd11605ceba1f3dc524314b838f6f8062d67ec6d447d8839744a94eb1f32f145.jpg)

![fcccc8441251d3f1e99d951857f5e91de0a428a9b1bf111abeeac261a13ae623.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/images/fcccc8441251d3f1e99d951857f5e91de0a428a9b1bf111abeeac261a13ae623.jpg)

### Tables

![228cd5772ec04b48b55ea57d17f559274f242dea736dacf8d66f7630185b8fb1.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/tables/228cd5772ec04b48b55ea57d17f559274f242dea736dacf8d66f7630185b8fb1.jpg)

![3d46ecf3cd32e11cec2cb3b5acb0f3a0343222b522eeaae4f167804265f771ed.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/tables/3d46ecf3cd32e11cec2cb3b5acb0f3a0343222b522eeaae4f167804265f771ed.jpg)

![948cd108f35b4eae19bb6c4ef2bbfe40649d11e88455fac297be0aa0e0f45c26.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/tables/948cd108f35b4eae19bb6c4ef2bbfe40649d11e88455fac297be0aa0e0f45c26.jpg)

![bff531e87e9170abbc8b3204b96df2eba703bf38bf22c61c9dbd7b691f4aa221.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/tables/bff531e87e9170abbc8b3204b96df2eba703bf38bf22c61c9dbd7b691f4aa221.jpg)

![fd41263faf8ed1e40d3130764b5626a5ee178df3e559a13f5e0226eda284b603.jpg](../icml_results/2212_MEG-XL_%20Data-Efficient%20Brain-to-Text%20via%20Long-Context%20Pre-Training/tables/fd41263faf8ed1e40d3130764b5626a5ee178df3e559a13f5e0226eda284b603.jpg)

<a id="paper-2184"></a>

## Cerebellar-Inspired Residual Control for Fault Recovery: From Inference-Time Adaptation to Structural Consolidation


### Images

![1a3be40f71578d028c972475c93cbc9beb72dd4a0dace09cf24cd9058b985233.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/1a3be40f71578d028c972475c93cbc9beb72dd4a0dace09cf24cd9058b985233.jpg)

![2abafe2945df8d4e88bf04767c560e4ebd75218682eeb15684ffb6f626f98f12.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/2abafe2945df8d4e88bf04767c560e4ebd75218682eeb15684ffb6f626f98f12.jpg)

![3090e9862eab33941909335a625efa1c472ebdd9acb13802a2ca50fdd3bdff55.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/3090e9862eab33941909335a625efa1c472ebdd9acb13802a2ca50fdd3bdff55.jpg)

![4c797c7425efb2350ad6881a281d416f7dfbe7fec5bd91e8b36e0d02051ab2d2.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/4c797c7425efb2350ad6881a281d416f7dfbe7fec5bd91e8b36e0d02051ab2d2.jpg)

![51f76dce144d722f003f794393af790fc1df619875ee88ec58636db81c31a06e.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/51f76dce144d722f003f794393af790fc1df619875ee88ec58636db81c31a06e.jpg)

![654199821bd4c3cab4f7a3772d8f734bbc364de284cd16164aa0fdd0aeaade00.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/654199821bd4c3cab4f7a3772d8f734bbc364de284cd16164aa0fdd0aeaade00.jpg)

![67bb2edc66823c38f9cf03ef7cfd380e32b666acaef5d955cbb0c982105fbc2d.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/67bb2edc66823c38f9cf03ef7cfd380e32b666acaef5d955cbb0c982105fbc2d.jpg)

![8ae10531af181d31d8426bcb660131cb926fc1a131cc74aff855ffabdad39cb2.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/8ae10531af181d31d8426bcb660131cb926fc1a131cc74aff855ffabdad39cb2.jpg)

![a773a589bb36debfa42ed5a3659c92ea7784f43b18707334737976bbc7a31d34.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/a773a589bb36debfa42ed5a3659c92ea7784f43b18707334737976bbc7a31d34.jpg)

![e16d7fa83aff4bc77d35b45432b318908067c6c92bdfdb9019044de6fe446d20.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/images/e16d7fa83aff4bc77d35b45432b318908067c6c92bdfdb9019044de6fe446d20.jpg)

### Tables

![151abb5effdb2f7b0b98d02efb094be08b96a8b16eaf2b6a5a65889d54e464f5.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/tables/151abb5effdb2f7b0b98d02efb094be08b96a8b16eaf2b6a5a65889d54e464f5.jpg)

![5e01a097e7825004758f1adca1a0c1225c075b24ff15b7b6d71df717c89a12c1.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/tables/5e01a097e7825004758f1adca1a0c1225c075b24ff15b7b6d71df717c89a12c1.jpg)

![6849057f73eceee1797966500a151d0ea3833e4675bd98d4a1d43fb8285f3c3e.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/tables/6849057f73eceee1797966500a151d0ea3833e4675bd98d4a1d43fb8285f3c3e.jpg)

![77db9620977f6e037849309ff024e326b1f7d812606e2efeec3d145cda46b25a.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/tables/77db9620977f6e037849309ff024e326b1f7d812606e2efeec3d145cda46b25a.jpg)

![fda6b433c8246d641b4d151f417d0d6cbcad0dce8508c3f4d5c0a3db410fb23d.jpg](../icml_results/2213_Cerebellar-Inspired%20Residual%20Control%20for%20Fault%20Recovery_%20From%20Inference-Time%20Adaptation%20to%20Structura/tables/fda6b433c8246d641b4d151f417d0d6cbcad0dce8508c3f4d5c0a3db410fb23d.jpg)

<a id="paper-2185"></a>

## Generative Large Neighborhood Search: Scalable Set Cover Optimization via Discrete Diffusion


### Images

![15a6bf4b4d2ccce1821bebe6dcaec715333760f1a6049cbe9a96c0f240f43b8a.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/15a6bf4b4d2ccce1821bebe6dcaec715333760f1a6049cbe9a96c0f240f43b8a.jpg)

![20804fbf7aadaac84d27c987fae497983113dfc9e6dbadb2485727ad6dd98415.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/20804fbf7aadaac84d27c987fae497983113dfc9e6dbadb2485727ad6dd98415.jpg)

![340f369e6fe7503b00401b46ecff32276074ccc813953a64d2c58b30b14125c1.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/340f369e6fe7503b00401b46ecff32276074ccc813953a64d2c58b30b14125c1.jpg)

![4bc87a04e799cc80d64056b5fb23636eb28e36f9a389a33b39c1ac922deb50cf.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/4bc87a04e799cc80d64056b5fb23636eb28e36f9a389a33b39c1ac922deb50cf.jpg)

![710adbd2e31cb17b784ca815e1339acba59f8dfb044a56d07ef102fa7fe13552.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/710adbd2e31cb17b784ca815e1339acba59f8dfb044a56d07ef102fa7fe13552.jpg)

![746125934d5404759a8d26825672db1a6abeb004751f34a12700235bfd80bcb0.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/746125934d5404759a8d26825672db1a6abeb004751f34a12700235bfd80bcb0.jpg)

![91dbf7e112308b68d4e5040e5107ee4c0f0799425c0c26270249beeb6f9225f3.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/91dbf7e112308b68d4e5040e5107ee4c0f0799425c0c26270249beeb6f9225f3.jpg)

![9a3aa3ca508cf79b621ed82f6a561cf348be4d866404c8298ea6a21343ebcb85.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/9a3aa3ca508cf79b621ed82f6a561cf348be4d866404c8298ea6a21343ebcb85.jpg)

![c36456c72f1f9f7f1cd3a9c1ab6482eba9f90878929baf7b947593e5ac919545.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/images/c36456c72f1f9f7f1cd3a9c1ab6482eba9f90878929baf7b947593e5ac919545.jpg)

### Tables

![07db38be7e3b241e2cc401ac63c1662f6a2b7d0ae60f3d79dc0b0b5bd4d04532.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/07db38be7e3b241e2cc401ac63c1662f6a2b7d0ae60f3d79dc0b0b5bd4d04532.jpg)

![19dfdd6ae922522a8fd7d1985047072f9109d83f04d1c5f54bcde30b6d3e3aeb.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/19dfdd6ae922522a8fd7d1985047072f9109d83f04d1c5f54bcde30b6d3e3aeb.jpg)

![2038d4d156474b833ad778b5a111ef970e53d4fce4e1da9a7ab3a01bdbce6c26.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/2038d4d156474b833ad778b5a111ef970e53d4fce4e1da9a7ab3a01bdbce6c26.jpg)

![35634dd1a16a8097bc6805e3c1df754a50aa5266fea4dd242e416733dde188d0.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/35634dd1a16a8097bc6805e3c1df754a50aa5266fea4dd242e416733dde188d0.jpg)

![5b82830f4c46270553658657ffe8c03099c2e0ce87444ba84fe77eee058f2f15.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/5b82830f4c46270553658657ffe8c03099c2e0ce87444ba84fe77eee058f2f15.jpg)

![5e4a184fe98576aa876a9de61ece6d05f5eb034598d024980af4eabbc931c182.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/5e4a184fe98576aa876a9de61ece6d05f5eb034598d024980af4eabbc931c182.jpg)

![62e8e6701f64ca66def1416c6e9f64efa441aac9486acde38fa1c813a914b691.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/62e8e6701f64ca66def1416c6e9f64efa441aac9486acde38fa1c813a914b691.jpg)

![76ade699754d0a4eae0247dad9a89d03ee4cbe3c7155fecf058739e764e1ae33.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/76ade699754d0a4eae0247dad9a89d03ee4cbe3c7155fecf058739e764e1ae33.jpg)

![9630326449c1f5cf6937dec79885f4b376697ae72075e67862fb16d241082abc.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/9630326449c1f5cf6937dec79885f4b376697ae72075e67862fb16d241082abc.jpg)

![a00b4f7844b882081480157c0af896b3759f0d3d9d3a02dc531b528cd0bcb50a.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/a00b4f7844b882081480157c0af896b3759f0d3d9d3a02dc531b528cd0bcb50a.jpg)

![b8c8d66a2dcd4e5c7ac2082f41f0c8cab6ffd7220c3bfbae8afc171e1772dd24.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/b8c8d66a2dcd4e5c7ac2082f41f0c8cab6ffd7220c3bfbae8afc171e1772dd24.jpg)

![d11fdd542f70087126c82911853c9b0e0736a8699c3f9949001da8a63a716de1.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/d11fdd542f70087126c82911853c9b0e0736a8699c3f9949001da8a63a716de1.jpg)

![ddd426b6f159a0dc03b52cbe1be414bb2971fd0fe14aa959e177517ee24429ac.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/ddd426b6f159a0dc03b52cbe1be414bb2971fd0fe14aa959e177517ee24429ac.jpg)

![e388c4fbe1fde474149874771a02d1ccf9fd0b88ccbae970f509652b98850f3d.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/e388c4fbe1fde474149874771a02d1ccf9fd0b88ccbae970f509652b98850f3d.jpg)

![f88faca1f72837e69fe07c15587926f5c39bb48e63683afff44a7ff8e498d1b7.jpg](../icml_results/2214_Generative%20Large%20Neighborhood%20Search_%20Scalable%20Set%20Cover%20Optimization%20via%20Discrete%20Diffusion/tables/f88faca1f72837e69fe07c15587926f5c39bb48e63683afff44a7ff8e498d1b7.jpg)

<a id="paper-2186"></a>

## Factorized Scheduling Principle: Learning Interpretable and Transferable Policies via Structured Additive Functions


### Images

![09865159ba0728067e1ecb23395d91e0df297292b1521a3fda5ecded117528c7.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/09865159ba0728067e1ecb23395d91e0df297292b1521a3fda5ecded117528c7.jpg)

![3548ab911c711168fa394e1dec2634e6a16d973a77c710ab0552bf8938f90cbd.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/3548ab911c711168fa394e1dec2634e6a16d973a77c710ab0552bf8938f90cbd.jpg)

![4c60b77ed4f8139c3bd6b331700b78d7b61861424a025d0a191badea4ae07c22.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/4c60b77ed4f8139c3bd6b331700b78d7b61861424a025d0a191badea4ae07c22.jpg)

![7dc9f72009618af8545cdb398dba6dd37a98e2679e253929ca56b23ac8f2fc78.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/7dc9f72009618af8545cdb398dba6dd37a98e2679e253929ca56b23ac8f2fc78.jpg)

![944097c0b9cd06677546da47baf418236be76de6e1749d198b55945ce6257140.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/944097c0b9cd06677546da47baf418236be76de6e1749d198b55945ce6257140.jpg)

![99d2243f86fdade4eece3dee49a413942c56d363a2bc494b3dd9802f2dbae963.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/99d2243f86fdade4eece3dee49a413942c56d363a2bc494b3dd9802f2dbae963.jpg)

![9df48bd1837632b2925807be44eac4e526520631256ad27f191f703acada79a2.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/9df48bd1837632b2925807be44eac4e526520631256ad27f191f703acada79a2.jpg)

![c1e15acb52f5a2d06307aa048830e9ade4e73f34c7637d93a5ad2f736d32ef4e.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/c1e15acb52f5a2d06307aa048830e9ade4e73f34c7637d93a5ad2f736d32ef4e.jpg)

![c6658d29117da5179f5765b98ef4593e61cc3a94c9f937f70bdc9a06ede487c0.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/c6658d29117da5179f5765b98ef4593e61cc3a94c9f937f70bdc9a06ede487c0.jpg)

![eeacf8b371dc9796a6a057dbca835a0ba466672847c40a7461c4017efa97b9a1.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/eeacf8b371dc9796a6a057dbca835a0ba466672847c40a7461c4017efa97b9a1.jpg)

![f7c6bc06423d2d13bd1420e3f65f8ea03a7ec4864245d50f1f31690c95624523.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/f7c6bc06423d2d13bd1420e3f65f8ea03a7ec4864245d50f1f31690c95624523.jpg)

![fed740ed3ade28d6355e8f420405ccf86fe17d41de82e43836ef3f1591164ad4.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/images/fed740ed3ade28d6355e8f420405ccf86fe17d41de82e43836ef3f1591164ad4.jpg)

### Tables

![2fdb2c0563086491b83741ec83f17c34416ff1c02a8bbad6105c2f51ca58637b.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/2fdb2c0563086491b83741ec83f17c34416ff1c02a8bbad6105c2f51ca58637b.jpg)

![5827f47f11a62b01bb90ccbf8e64bbf20de975eb6c97ed7943906c01a6cbfb22.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/5827f47f11a62b01bb90ccbf8e64bbf20de975eb6c97ed7943906c01a6cbfb22.jpg)

![621f18e22393e4dbac2d7e2a26a738a4becc3534c37e39dce776513c2560e613.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/621f18e22393e4dbac2d7e2a26a738a4becc3534c37e39dce776513c2560e613.jpg)

![8adf04088f842aee3891bbb2dfcea41fd386be060967df1aed0f5b8adf506dd0.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/8adf04088f842aee3891bbb2dfcea41fd386be060967df1aed0f5b8adf506dd0.jpg)

![9242d326538e5b033e522417670c39692a8076d017db7f125f12472b7a905714.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/9242d326538e5b033e522417670c39692a8076d017db7f125f12472b7a905714.jpg)

![af21a307c3bce36fb2a8486bcd93ff8166ae4eb94634e7a74b87b51b12bcbf47.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/af21a307c3bce36fb2a8486bcd93ff8166ae4eb94634e7a74b87b51b12bcbf47.jpg)

![af3316d114d2dd41a939c50b0269580129f110b05f52e4452d9b9186617cbba7.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/af3316d114d2dd41a939c50b0269580129f110b05f52e4452d9b9186617cbba7.jpg)

![c04ed0cc644461d209da048661cdfa183182f6c8c98989430530c0e98d8462d9.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/c04ed0cc644461d209da048661cdfa183182f6c8c98989430530c0e98d8462d9.jpg)

![cf9ced32eedc19469b4db19f4e84531f40ef70783d7f837d326b42c8331b738b.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/cf9ced32eedc19469b4db19f4e84531f40ef70783d7f837d326b42c8331b738b.jpg)

![ffa2958e3ab1fa51db341ca420384e8be8eb54a70a302a0f24492479624a0426.jpg](../icml_results/2215_Factorized%20Scheduling%20Principle_%20Learning%20Interpretable%20and%20Transferable%20Policies%20via%20Structured%20Add/tables/ffa2958e3ab1fa51db341ca420384e8be8eb54a70a302a0f24492479624a0426.jpg)

<a id="paper-2187"></a>

## Decision-focused Sparse Tangent Portfolio Optimization


### Images

![0f4b5accc8465c99a2f6fba2c766af4f335b08631418ab1fbca3a91d49551717.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/images/0f4b5accc8465c99a2f6fba2c766af4f335b08631418ab1fbca3a91d49551717.jpg)

![354c399f9764acb74fc7a52fcf6735b8a74f43b4a12c7e13fcb3b7a7c6f97331.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/images/354c399f9764acb74fc7a52fcf6735b8a74f43b4a12c7e13fcb3b7a7c6f97331.jpg)

![579fd333ed21a620eaa605e67503d51574ff6e4abe9e1632c8d1ce9357a40748.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/images/579fd333ed21a620eaa605e67503d51574ff6e4abe9e1632c8d1ce9357a40748.jpg)

![99a0bfd74749ffc196d63cc298a3c65737039016f2ee69cf668481f13f22db65.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/images/99a0bfd74749ffc196d63cc298a3c65737039016f2ee69cf668481f13f22db65.jpg)

![cd52e15e26e5a7d3b18e3eb079eceab600356650fdef46559cafa253962ffb88.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/images/cd52e15e26e5a7d3b18e3eb079eceab600356650fdef46559cafa253962ffb88.jpg)

### Tables

![33dd59f094a5dc2c5245991a72c56ad0c3a68b892349a80d28b37faacadd50a0.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/tables/33dd59f094a5dc2c5245991a72c56ad0c3a68b892349a80d28b37faacadd50a0.jpg)

![878c8f68911e3d18629eb5559dd36e4a85a4d0d3b1a97d4f8ae96a3c976d40fd.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/tables/878c8f68911e3d18629eb5559dd36e4a85a4d0d3b1a97d4f8ae96a3c976d40fd.jpg)

![e32df81cddebcc333df84f90746ace4cb7332ab10bdc431c9718287ee541a458.jpg](../icml_results/2216_Decision-focused%20Sparse%20Tangent%20Portfolio%20Optimization/tables/e32df81cddebcc333df84f90746ace4cb7332ab10bdc431c9718287ee541a458.jpg)

<a id="paper-2188"></a>

## LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents


### Images

![0e14fe035ae7a22cd1f07869f811aab3c46708df3cedde582eacb0a0d7a335e5.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/0e14fe035ae7a22cd1f07869f811aab3c46708df3cedde582eacb0a0d7a335e5.jpg)

![2dcd755c441446f7bf745da1910046f384895622947aafde840773199873ba65.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/2dcd755c441446f7bf745da1910046f384895622947aafde840773199873ba65.jpg)

![56bf202c7eefbf9a52122a55e3fa5e03691224f1426d29f94d8ba0c4b4787159.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/56bf202c7eefbf9a52122a55e3fa5e03691224f1426d29f94d8ba0c4b4787159.jpg)

![62be91be293cc5a31913d973ddd443861877134a622bff3bec2be9e2db508f10.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/62be91be293cc5a31913d973ddd443861877134a622bff3bec2be9e2db508f10.jpg)

![67b519396070ad97b995700eaeaccbe7aa5825d3a4126743134ac149b6c75ffc.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/67b519396070ad97b995700eaeaccbe7aa5825d3a4126743134ac149b6c75ffc.jpg)

![960f1b916a08d366db54e85d2eb09c1d9cf8740f76a7e0132036d509ff232400.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/960f1b916a08d366db54e85d2eb09c1d9cf8740f76a7e0132036d509ff232400.jpg)

![a03d2088e70b1920c803a6be748d517665589be5bf0f51e8f5ee44b078348dca.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/a03d2088e70b1920c803a6be748d517665589be5bf0f51e8f5ee44b078348dca.jpg)

![cb5c145ce6cbf1936dae7774eba1729f90763cb5149bc573ff66779803616008.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/cb5c145ce6cbf1936dae7774eba1729f90763cb5149bc573ff66779803616008.jpg)

![e0174bd4f06436e66c1a24a6ef794a71b2578d029d6fc48829b5b6c58a8450f5.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/e0174bd4f06436e66c1a24a6ef794a71b2578d029d6fc48829b5b6c58a8450f5.jpg)

![e9a6bacd234ede59e7b1a8c86a78a2314eb7ff4d4853e3ef5a3eab9c36a74086.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/images/e9a6bacd234ede59e7b1a8c86a78a2314eb7ff4d4853e3ef5a3eab9c36a74086.jpg)

### Tables

![1cd2e0c689d0257cf94bf80033c67373fbc6842043a27efdd120a779132936ef.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/1cd2e0c689d0257cf94bf80033c67373fbc6842043a27efdd120a779132936ef.jpg)

![274806925b4aa4b49ec50a46b12d72dc856f928fa46bac51b083eba537e16004.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/274806925b4aa4b49ec50a46b12d72dc856f928fa46bac51b083eba537e16004.jpg)

![367e3214562c71055a6a20ac1e872cc0cd7165e28dfe98f2d0e86d95528d76d2.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/367e3214562c71055a6a20ac1e872cc0cd7165e28dfe98f2d0e86d95528d76d2.jpg)

![394eed4eab819dcf0cdec59a9060f26518ec502d09eb3d5224f737e3cbe179a3.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/394eed4eab819dcf0cdec59a9060f26518ec502d09eb3d5224f737e3cbe179a3.jpg)

![40f3477c401aeb47e27d425728d6d315342a44ea9b4fc03d65e95cc23883ab3f.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/40f3477c401aeb47e27d425728d6d315342a44ea9b4fc03d65e95cc23883ab3f.jpg)

![435286bd1c77e94a5045bff7a7b86766b90ba77b993db4ae725b5559fa88d52c.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/435286bd1c77e94a5045bff7a7b86766b90ba77b993db4ae725b5559fa88d52c.jpg)

![47e0d5ab47967a9c63d07ee7f947e38cc34ea7e94c13436781899ddea35623e2.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/47e0d5ab47967a9c63d07ee7f947e38cc34ea7e94c13436781899ddea35623e2.jpg)

![4850cc093d805cffe7f6c2cf3c27047266c51c4f0389c36762fef6761cd975aa.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/4850cc093d805cffe7f6c2cf3c27047266c51c4f0389c36762fef6761cd975aa.jpg)

![601fdbfb1c166fe5d91d9e34c8576ce6ca3a4fd1c9664c62637a8bf1fe16a145.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/601fdbfb1c166fe5d91d9e34c8576ce6ca3a4fd1c9664c62637a8bf1fe16a145.jpg)

![7c261887ad388f64943e573c95d2ea31d595b48738c32df74a6ddbf81b1b83f5.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/7c261887ad388f64943e573c95d2ea31d595b48738c32df74a6ddbf81b1b83f5.jpg)

![829059719e78a69dbda02271c21a59194f9127943cd5e09be17ff60fd86b1daa.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/829059719e78a69dbda02271c21a59194f9127943cd5e09be17ff60fd86b1daa.jpg)

![8344479b7ac49fe843795acfbca8a0b43d74e2b7cf7356ec4b7c038b6107f36e.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/8344479b7ac49fe843795acfbca8a0b43d74e2b7cf7356ec4b7c038b6107f36e.jpg)

![835244c6ad66a32552dc9dfd90e3653cdd22d573a6ee95a0a0529f6271628d53.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/835244c6ad66a32552dc9dfd90e3653cdd22d573a6ee95a0a0529f6271628d53.jpg)

![8c9cef00fc39db4ae2b1f1c968ad59a314a846acd388f7337941dbf7e858799b.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/8c9cef00fc39db4ae2b1f1c968ad59a314a846acd388f7337941dbf7e858799b.jpg)

![a61ec0913f1eecc5401fd179487c6fb9d93f057a6094234fe5eb7002fab26741.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/a61ec0913f1eecc5401fd179487c6fb9d93f057a6094234fe5eb7002fab26741.jpg)

![a7e96ab20d08010fb076b3e0a82b74162001e75e562c617673eba7aa343f5278.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/a7e96ab20d08010fb076b3e0a82b74162001e75e562c617673eba7aa343f5278.jpg)

![babd2f141bc9f0950a5089ab6e00703041cf9922df2670cc92f218e90d2390bb.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/babd2f141bc9f0950a5089ab6e00703041cf9922df2670cc92f218e90d2390bb.jpg)

![bc2b0d6230f6ed9e327fb6f54e0749e7b4541304c63bab03a41af7be1579d563.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/bc2b0d6230f6ed9e327fb6f54e0749e7b4541304c63bab03a41af7be1579d563.jpg)

![bec03f124663a38e57f26c29b241a60120ebc11ac3fd4988d1e781038599cdac.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/bec03f124663a38e57f26c29b241a60120ebc11ac3fd4988d1e781038599cdac.jpg)

![c072b4855a98b03162d615930f3946b3feaa4a932780079dca2c9546f7b8dd85.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/c072b4855a98b03162d615930f3946b3feaa4a932780079dca2c9546f7b8dd85.jpg)

![c244489dd9bbcf755c8c511da4a150849432709052e836fb2fb9d32ba3204ac7.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/c244489dd9bbcf755c8c511da4a150849432709052e836fb2fb9d32ba3204ac7.jpg)

![c730d825653f1f245e9729ef44515fe8bdbec21cefc46f0b993831bc3d10fe1a.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/c730d825653f1f245e9729ef44515fe8bdbec21cefc46f0b993831bc3d10fe1a.jpg)

![f1a615fa30ada3e6ddcf36d7bf7fe1f98640b173329aed444aaeef3079b64bf1.jpg](../icml_results/2217_LRAgent_%20Efficient%20KV%20Cache%20Sharing%20for%20Multi-LoRA%20LLM%20Agents/tables/f1a615fa30ada3e6ddcf36d7bf7fe1f98640b173329aed444aaeef3079b64bf1.jpg)

<a id="paper-2189"></a>

## Vision-aligned Latent Reasoning for Multi-modal Large Language Model


### Images

![03ff95469a4ace8f802010165fe43bad13acc0fca00cf264f5bcdef37a3c61f6.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/images/03ff95469a4ace8f802010165fe43bad13acc0fca00cf264f5bcdef37a3c61f6.jpg)

![60603bf1242b3108875bc8a40b74c6b2c1cbcef43e0533ad32e48e012fe4aee6.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/images/60603bf1242b3108875bc8a40b74c6b2c1cbcef43e0533ad32e48e012fe4aee6.jpg)

![7cabfae307e8541f80371559aa002a7002e6f41d0516414c958ed3436534dc11.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/images/7cabfae307e8541f80371559aa002a7002e6f41d0516414c958ed3436534dc11.jpg)

![bdee6948087bb807f2ff19b7c559b9ba22425804a3a68e97435c1ec6609b25a1.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/images/bdee6948087bb807f2ff19b7c559b9ba22425804a3a68e97435c1ec6609b25a1.jpg)

![fd0055a1683267def88285852e07626dc236245786c4801d2db088163a05f23b.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/images/fd0055a1683267def88285852e07626dc236245786c4801d2db088163a05f23b.jpg)

### Tables

![07e56210507dcc7a7c91707f09ce43ee969e78ff86560b86ed86e46cf3931b12.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/07e56210507dcc7a7c91707f09ce43ee969e78ff86560b86ed86e46cf3931b12.jpg)

![08722c9dce248e7fde3538a4275a3dcd8a48857a38bb962e40ee0c07289c4fb0.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/08722c9dce248e7fde3538a4275a3dcd8a48857a38bb962e40ee0c07289c4fb0.jpg)

![0a1e044764624fe62779382868a7f53380ae7073afb9b350e17202ac8ba188a0.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/0a1e044764624fe62779382868a7f53380ae7073afb9b350e17202ac8ba188a0.jpg)

![0b37451a80afd07dfe6cecc1f2cd9709c3cd01b1abc799c97f27864811ba35e6.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/0b37451a80afd07dfe6cecc1f2cd9709c3cd01b1abc799c97f27864811ba35e6.jpg)

![5575093691e64d04a4674376248d6539f267ac5ace8ebb6b36edafd23dd3c9e6.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/5575093691e64d04a4674376248d6539f267ac5ace8ebb6b36edafd23dd3c9e6.jpg)

![990fe3e5f22bc78b152adf621234c93c62cd4dc250c14a205613c30a7517f865.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/990fe3e5f22bc78b152adf621234c93c62cd4dc250c14a205613c30a7517f865.jpg)

![b0bbe8135ab4acdb64ea36829d3f7a66af4b16d02cc5b8b2d80386382b49f373.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/b0bbe8135ab4acdb64ea36829d3f7a66af4b16d02cc5b8b2d80386382b49f373.jpg)

![b98bcc0474790927eceb36dd86a98899a4775eb5209adebacee7bf472984a63d.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/b98bcc0474790927eceb36dd86a98899a4775eb5209adebacee7bf472984a63d.jpg)

![d319b6220a3dcccc18577dfa57ced96421cc533940c6e83939e9d71bd6d775e9.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/d319b6220a3dcccc18577dfa57ced96421cc533940c6e83939e9d71bd6d775e9.jpg)

![e413205cd86efa76477529b61b778e2840389376f52e69bf3f828495c05ad258.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/e413205cd86efa76477529b61b778e2840389376f52e69bf3f828495c05ad258.jpg)

![ee02026fa4e9ae06d198524cfe31be99052066e7a13fa8e431688fddaa0fe79d.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/ee02026fa4e9ae06d198524cfe31be99052066e7a13fa8e431688fddaa0fe79d.jpg)

![fbd6e1aab83521dc0422662a5a6db07fa51863b81ead8cf9f7eb3bb711244a90.jpg](../icml_results/2218_Vision-aligned%20Latent%20Reasoning%20for%20Multi-modal%20Large%20Language%20Model/tables/fbd6e1aab83521dc0422662a5a6db07fa51863b81ead8cf9f7eb3bb711244a90.jpg)

<a id="paper-2190"></a>

## Towards Functional Correctness of Code Models with Selective Generation


### Images

![26dd849ef74f2db758ce760aac28f8bf114556b7f0050aefcf1927dd873e5aa8.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/26dd849ef74f2db758ce760aac28f8bf114556b7f0050aefcf1927dd873e5aa8.jpg)

![3ab2d7778353c6bd0c87e710b96242dc0e6ee11bc82fa6cae59df45389b2377d.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/3ab2d7778353c6bd0c87e710b96242dc0e6ee11bc82fa6cae59df45389b2377d.jpg)

![60070d6ab7da249d46444f8a158e775aef41a35b6d15618dee446cf18c89145c.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/60070d6ab7da249d46444f8a158e775aef41a35b6d15618dee446cf18c89145c.jpg)

![65b8d53cd6edf5b8e4bd262274948a80e1555c5e5fddb26f093bf869e10e3613.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/65b8d53cd6edf5b8e4bd262274948a80e1555c5e5fddb26f093bf869e10e3613.jpg)

![a9de928e9310498d7753b5b6f7f0f95ab05d97bf3627cc93a69048ab9c9f7751.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/a9de928e9310498d7753b5b6f7f0f95ab05d97bf3627cc93a69048ab9c9f7751.jpg)

![df5a3d1c20fa27c8ca5c6fd71f7ff6ce548166415a1f0e3dd77e07a1cf2e0b4c.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/images/df5a3d1c20fa27c8ca5c6fd71f7ff6ce548166415a1f0e3dd77e07a1cf2e0b4c.jpg)

### Tables

![0ebcbcb2f4c9a850a173164efd78a9202f1f355fb990629db8de20ed32bad752.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/0ebcbcb2f4c9a850a173164efd78a9202f1f355fb990629db8de20ed32bad752.jpg)

![4a260fc506c5f1fabacd2bc72b1e3396077b2bd330f7462128fadf8dd2efab73.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/4a260fc506c5f1fabacd2bc72b1e3396077b2bd330f7462128fadf8dd2efab73.jpg)

![7a5f82160dff2236b909fe1ef347dcdbbf55fb8eedd9c4924ea653aca9b033f4.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/7a5f82160dff2236b909fe1ef347dcdbbf55fb8eedd9c4924ea653aca9b033f4.jpg)

![887fefa858065d427e4113a186022636c7e98f15f99cd130a7de5a8a4be5ecc9.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/887fefa858065d427e4113a186022636c7e98f15f99cd130a7de5a8a4be5ecc9.jpg)

![9d3e3c8f66ae6940c5def906f468fb4cb52636790f263db83277e77b96d7f85f.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/9d3e3c8f66ae6940c5def906f468fb4cb52636790f263db83277e77b96d7f85f.jpg)

![c75c0e335a1a73c0f241b7a582af4f2b6b9ba8a7fed0b72ecf97824234ac0907.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/c75c0e335a1a73c0f241b7a582af4f2b6b9ba8a7fed0b72ecf97824234ac0907.jpg)

![e601c61bf84dfbb967a0fff26eed398c232f8527d2c9c5aebb7e665d25ad47c5.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/e601c61bf84dfbb967a0fff26eed398c232f8527d2c9c5aebb7e665d25ad47c5.jpg)

![e772d4f22f5a95e97b0a19b478a723c2dc6bae64595352b6d1a710b701ef5be5.jpg](../icml_results/2219_Towards%20Functional%20Correctness%20of%20Code%20Models%20with%20Selective%20Generation/tables/e772d4f22f5a95e97b0a19b478a723c2dc6bae64595352b6d1a710b701ef5be5.jpg)

<a id="paper-2191"></a>

## When Do Diffusion Models Learn to Generate Multiple Objects?


### Images

![0864ebc2ea398e2e9aea1f962245743e940aeb8e5aa262bb287a82d802af6a8e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/0864ebc2ea398e2e9aea1f962245743e940aeb8e5aa262bb287a82d802af6a8e.jpg)

![1118686270a1eb80319da3374ac39d17ec90c6ab6613f3962089b01dc6450af5.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/1118686270a1eb80319da3374ac39d17ec90c6ab6613f3962089b01dc6450af5.jpg)

![20c3b1221b8acd6a68be0412e06fca0c3d9d8a240fe4577e05226c839ac36c39.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/20c3b1221b8acd6a68be0412e06fca0c3d9d8a240fe4577e05226c839ac36c39.jpg)

![26fedbbbb08a6c9af018642678437b6b3eaa7f339b1f2e15f8047ef7820fa413.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/26fedbbbb08a6c9af018642678437b6b3eaa7f339b1f2e15f8047ef7820fa413.jpg)

![2a9b15313117dd315c33adffaba6140834f2cdfc7589b980aaeda1027cd968de.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/2a9b15313117dd315c33adffaba6140834f2cdfc7589b980aaeda1027cd968de.jpg)

![2b73396565043c90a8d2e039a189b06eb3252a0bb202b4da036236a306a5d63e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/2b73396565043c90a8d2e039a189b06eb3252a0bb202b4da036236a306a5d63e.jpg)

![36790ec73ad3c032710b912eea0c7b999593cc43f6d50f56e087617b99f2ae98.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/36790ec73ad3c032710b912eea0c7b999593cc43f6d50f56e087617b99f2ae98.jpg)

![416a0f7f5f4ff53345f6f361a9e1b98e85bc93bdc38712a84ef976041a71dbde.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/416a0f7f5f4ff53345f6f361a9e1b98e85bc93bdc38712a84ef976041a71dbde.jpg)

![41d7509f39d83ab34fc24ffa306825ab1d509cc1166f38220cde80428a00f4a4.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/41d7509f39d83ab34fc24ffa306825ab1d509cc1166f38220cde80428a00f4a4.jpg)

![48ba85cef1096773e35706b631bd9d85c45400401c97bac8e4c85843b7bfad5d.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/48ba85cef1096773e35706b631bd9d85c45400401c97bac8e4c85843b7bfad5d.jpg)

![574dd792cd1256e47ffc47dbd36c6ae2d37854d581024c266cb31a8f3c74b85a.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/574dd792cd1256e47ffc47dbd36c6ae2d37854d581024c266cb31a8f3c74b85a.jpg)

![5dc37f7eba4870ce60a6ede5fd0797e2e7e99933b67fd18867f63eccbca04e18.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/5dc37f7eba4870ce60a6ede5fd0797e2e7e99933b67fd18867f63eccbca04e18.jpg)

![61f6044d491700bb5c25a6fc834420ccd90a7f32906fbc5fdc69855348c843e4.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/61f6044d491700bb5c25a6fc834420ccd90a7f32906fbc5fdc69855348c843e4.jpg)

![6336571b387529f8da1de2ebed219b9ce04b4b3c719f24a6014ac1797452bb2e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/6336571b387529f8da1de2ebed219b9ce04b4b3c719f24a6014ac1797452bb2e.jpg)

![64c3e627a80eac2f9d9b794cbb1e964d453e2b4d6427a20fdaad7c43020d2acd.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/64c3e627a80eac2f9d9b794cbb1e964d453e2b4d6427a20fdaad7c43020d2acd.jpg)

![67ef9c64d944ac7d3da4546817265a76316546f9a361e91bdc91814868e71d9b.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/67ef9c64d944ac7d3da4546817265a76316546f9a361e91bdc91814868e71d9b.jpg)

![6bb2cc16f13e9c26f616f3590e5d1d12e1eb33b924859221a0525bcbef52b946.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/6bb2cc16f13e9c26f616f3590e5d1d12e1eb33b924859221a0525bcbef52b946.jpg)

![6c3e3f755fe1a30a90488970ea57d85bbc49fa223db1447e10e3fe6980f1ed4a.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/6c3e3f755fe1a30a90488970ea57d85bbc49fa223db1447e10e3fe6980f1ed4a.jpg)

![73c3c92bae1ebfcaf484b0d3ef27fb8b1b5326f9679fe449716cf11a9e2b0c72.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/73c3c92bae1ebfcaf484b0d3ef27fb8b1b5326f9679fe449716cf11a9e2b0c72.jpg)

![7739d001366c4b2b0a9a76204e49f38764580cac187238c0599806841ef5b05f.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/7739d001366c4b2b0a9a76204e49f38764580cac187238c0599806841ef5b05f.jpg)

![7a5cee3d0e4746fb9465088022ef7e70335987f54087c363e29130810991ddcf.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/7a5cee3d0e4746fb9465088022ef7e70335987f54087c363e29130810991ddcf.jpg)

![7a7ef991f7ae487adc68a31ad2b38b9abc49c63d4be4a7914097947d6bd70585.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/7a7ef991f7ae487adc68a31ad2b38b9abc49c63d4be4a7914097947d6bd70585.jpg)

![7eb8352b86df525a93a372f780a703ca38609846519676f55b1d89fee4810c5b.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/7eb8352b86df525a93a372f780a703ca38609846519676f55b1d89fee4810c5b.jpg)

![7fb785eef3318621bcdbba67583b8406ef8813c4fef2200710768b6fe8c8d2ec.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/7fb785eef3318621bcdbba67583b8406ef8813c4fef2200710768b6fe8c8d2ec.jpg)

![802cb45e0ab235c9f1fa408123f1c9c640ae7b0268d4515e229419b6246abe0f.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/802cb45e0ab235c9f1fa408123f1c9c640ae7b0268d4515e229419b6246abe0f.jpg)

![81d7c80ebb2e0984537b3a1210907cd8ad5a54568a0cade59e949692c84b062e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/81d7c80ebb2e0984537b3a1210907cd8ad5a54568a0cade59e949692c84b062e.jpg)

![875f4e3e199e2bb0007196c9340cad6bb6df91e6bbbded75500f931e752932e9.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/875f4e3e199e2bb0007196c9340cad6bb6df91e6bbbded75500f931e752932e9.jpg)

![890327c1d07ecd834a6cc1d5cbfa749c0ba6267b5e20a42e2c211aedc5e002b7.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/890327c1d07ecd834a6cc1d5cbfa749c0ba6267b5e20a42e2c211aedc5e002b7.jpg)

![915bab95f8fe31453b3d3391f81a234ea8e22d73d5513873b9cad391eeb03ddf.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/915bab95f8fe31453b3d3391f81a234ea8e22d73d5513873b9cad391eeb03ddf.jpg)

![97ef7973a397908ba67aca068f51a04289aac2e0905a213498154baa7f3a1473.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/97ef7973a397908ba67aca068f51a04289aac2e0905a213498154baa7f3a1473.jpg)

![989d5c5312354cae50d50a99b362599486230e81122cdcd67d58153e051ed33c.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/989d5c5312354cae50d50a99b362599486230e81122cdcd67d58153e051ed33c.jpg)

![99ae645da33a7606490444e687f587cfd9ba72dda35e1e82303dee3322817eb1.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/99ae645da33a7606490444e687f587cfd9ba72dda35e1e82303dee3322817eb1.jpg)

![a060b77f4f0422b1a8ace1897147cfc0b270f10ae8e5dc01893025b230ff16da.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/a060b77f4f0422b1a8ace1897147cfc0b270f10ae8e5dc01893025b230ff16da.jpg)

![a24b81890b37a1f13d74ae47a472253a28c770be319fe2eb017d129e1386ebb1.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/a24b81890b37a1f13d74ae47a472253a28c770be319fe2eb017d129e1386ebb1.jpg)

![a41823ae9543ccc1c7f74050773dafb1d6962a7ed192c18b228b7fa5549153cd.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/a41823ae9543ccc1c7f74050773dafb1d6962a7ed192c18b228b7fa5549153cd.jpg)

![a6ee5f34038591ff6c89ffea99716d6cd9fe973c8bd8d809c9289d1c6043620e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/a6ee5f34038591ff6c89ffea99716d6cd9fe973c8bd8d809c9289d1c6043620e.jpg)

![aa753bd97e0c0759efcdc7b748abf006ee754bd9802d92575694aa8bc7ba47ca.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/aa753bd97e0c0759efcdc7b748abf006ee754bd9802d92575694aa8bc7ba47ca.jpg)

![adada4e75f4f2c4112f97bd9c6f30dc471b0f405c12d743449b2ec3993644e59.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/adada4e75f4f2c4112f97bd9c6f30dc471b0f405c12d743449b2ec3993644e59.jpg)

![b043d74c60661832d1fffb9c221d1523b4172f2864100c092e3e1b079fd07a80.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b043d74c60661832d1fffb9c221d1523b4172f2864100c092e3e1b079fd07a80.jpg)

![b18e48ba8faadbad2b0d9f4876b8f8b2c835fdce74a5c128665de3ed29f6b925.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b18e48ba8faadbad2b0d9f4876b8f8b2c835fdce74a5c128665de3ed29f6b925.jpg)

![b22a079fcd0535d9a452d48082bfe0b24700f9c6b61acb1eb9f3b2b5d9ede919.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b22a079fcd0535d9a452d48082bfe0b24700f9c6b61acb1eb9f3b2b5d9ede919.jpg)

![b3239830983259bc31e367deb359a79d27deacbd2cb6d1241fab95c8a3cdd833.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b3239830983259bc31e367deb359a79d27deacbd2cb6d1241fab95c8a3cdd833.jpg)

![b390c7e6230fe212dc08528cc81e52941b518adcd86a4ef2a5f55dd89c756912.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b390c7e6230fe212dc08528cc81e52941b518adcd86a4ef2a5f55dd89c756912.jpg)

![b3b89b8960179388d69e84985bbd61c157b947cd13024a95e0915f14ecf2c4fe.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/b3b89b8960179388d69e84985bbd61c157b947cd13024a95e0915f14ecf2c4fe.jpg)

![be4eb1239fd8c6617ed5ce57a8e5efeddccd36506e676d62b654fd349866096d.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/be4eb1239fd8c6617ed5ce57a8e5efeddccd36506e676d62b654fd349866096d.jpg)

![bf06b04f3088853122e7a46c284cd455ed592e121f89e7140bbabcd0bcdd110d.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/bf06b04f3088853122e7a46c284cd455ed592e121f89e7140bbabcd0bcdd110d.jpg)

![d85f8ab5c5804c50427371b48d9667c29e765cd2a68d1dc3bcb246fcd80f5f74.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/d85f8ab5c5804c50427371b48d9667c29e765cd2a68d1dc3bcb246fcd80f5f74.jpg)

![e1acd03bcdfb4337969d7a7b53089222d49a9d336a1785a4e9cdd9cf7baa2cb4.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e1acd03bcdfb4337969d7a7b53089222d49a9d336a1785a4e9cdd9cf7baa2cb4.jpg)

![e20d961e336deb4c330f0cd313bd939e4d002c66e0ba9e69c9c2070e6786e938.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e20d961e336deb4c330f0cd313bd939e4d002c66e0ba9e69c9c2070e6786e938.jpg)

![e30b938f719cea737e2c9855e807ae2f535c7b1f33b5e06d32442adbce137b23.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e30b938f719cea737e2c9855e807ae2f535c7b1f33b5e06d32442adbce137b23.jpg)

![e59c1f1c32b3e20348a850b1333d2c1321dde956f2fba79fd317bada56de6f0f.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e59c1f1c32b3e20348a850b1333d2c1321dde956f2fba79fd317bada56de6f0f.jpg)

![e5fde1665d2091ef53f51cd9c91377453f449b49831696689611eef1aff3ac08.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e5fde1665d2091ef53f51cd9c91377453f449b49831696689611eef1aff3ac08.jpg)

![e6a1cae2faaa762306efd75533851b693289b2ea11ba42c8993042726fae4ab6.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/e6a1cae2faaa762306efd75533851b693289b2ea11ba42c8993042726fae4ab6.jpg)

![eab0bd1c8ee7c267a141b4941ca42ded8312f3516abbdfe8ddb32964708fbb83.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/eab0bd1c8ee7c267a141b4941ca42ded8312f3516abbdfe8ddb32964708fbb83.jpg)

![ef4b62b6e9d463b24374f4601915578cf4464c4e5acc674fa48984473a78e7f5.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/ef4b62b6e9d463b24374f4601915578cf4464c4e5acc674fa48984473a78e7f5.jpg)

![f5892ca0cf7857ad0d950b64f6ce290c4ea41b4ef00b493fce9a6e89c7f3bca8.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/f5892ca0cf7857ad0d950b64f6ce290c4ea41b4ef00b493fce9a6e89c7f3bca8.jpg)

![f5d5b62c38388abcb23448caeacfb91473f2989a1a73aa47e44cdf6394106690.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/f5d5b62c38388abcb23448caeacfb91473f2989a1a73aa47e44cdf6394106690.jpg)

![f8ec5c426e015254cc2b586478bc89b57bbd8d9d45bd58f812454b3ccd3b3c8b.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/f8ec5c426e015254cc2b586478bc89b57bbd8d9d45bd58f812454b3ccd3b3c8b.jpg)

![fe25ca56105a767ff69201ff566d0045c44f296033679780d4772d790d1eab5f.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/images/fe25ca56105a767ff69201ff566d0045c44f296033679780d4772d790d1eab5f.jpg)

### Tables

![21731d2b1e5462c25eaa6cb76b1b2e8867610baf02b73250b085fe75d6ae4a95.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/21731d2b1e5462c25eaa6cb76b1b2e8867610baf02b73250b085fe75d6ae4a95.jpg)

![2703f58ebedb24a844d9e8d00d9b4b59e790d8cd0bb27bd978b6b66df5b9e966.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/2703f58ebedb24a844d9e8d00d9b4b59e790d8cd0bb27bd978b6b66df5b9e966.jpg)

![6afefddbe9715013ad1e77d262e43e5c1a14e07d54fa6e6277a040ce955e422e.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/6afefddbe9715013ad1e77d262e43e5c1a14e07d54fa6e6277a040ce955e422e.jpg)

![8d542c2faa1e328e461383b6b185abd2a68562fa7b2b3ce1671e85ac0aed31fb.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/8d542c2faa1e328e461383b6b185abd2a68562fa7b2b3ce1671e85ac0aed31fb.jpg)

![913d5bd0344912f52b930594536f916905758f106b31a52fce74d3e115cdee93.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/913d5bd0344912f52b930594536f916905758f106b31a52fce74d3e115cdee93.jpg)

![ab61e0a84d3b78c24100622ff747ad10bf0a24cca162de037f37cdcb468e1ae1.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/ab61e0a84d3b78c24100622ff747ad10bf0a24cca162de037f37cdcb468e1ae1.jpg)

![d9ad73719e83103656ef0b6567fb51e3e43936b6f15269880c26277944b4c921.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/d9ad73719e83103656ef0b6567fb51e3e43936b6f15269880c26277944b4c921.jpg)

![db88209728c766c7acf315f851a4fad89beee7ba5810cbb61c55a0d3cdeea2c2.jpg](../icml_results/2220_When%20Do%20Diffusion%20Models%20Learn%20to%20Generate%20Multiple%20Objects_/tables/db88209728c766c7acf315f851a4fad89beee7ba5810cbb61c55a0d3cdeea2c2.jpg)

<a id="paper-2192"></a>

## Context-free Recognition with Transformers


### Images

![79b0c7a721ecac7b1862128e6e02e5497467fa94563e90cf28b3319f05892216.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/images/79b0c7a721ecac7b1862128e6e02e5497467fa94563e90cf28b3319f05892216.jpg)

![bb0a27c58932d2cd29810edf91e20f0803285ba362f6691e2ad06c074dcf8ba6.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/images/bb0a27c58932d2cd29810edf91e20f0803285ba362f6691e2ad06c074dcf8ba6.jpg)

### Tables

![0c5d2020ea6463ad98248bd9f6edb2ee1ec8229c983170fa3dfd4ad39cdaa54e.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/0c5d2020ea6463ad98248bd9f6edb2ee1ec8229c983170fa3dfd4ad39cdaa54e.jpg)

![5b8b67c5de8cb95245fbdeacda4f12b37386eb85686a004c61e1f55a46ea80b7.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/5b8b67c5de8cb95245fbdeacda4f12b37386eb85686a004c61e1f55a46ea80b7.jpg)

![8eba546d6349eb89398a579126970e0a86b86160cd5f991fcb3e63a68a107868.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/8eba546d6349eb89398a579126970e0a86b86160cd5f991fcb3e63a68a107868.jpg)

![92e24d1bd8e966ba3861e3771b89fb39cf1e04f172164f1bbc067f306d8ec05e.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/92e24d1bd8e966ba3861e3771b89fb39cf1e04f172164f1bbc067f306d8ec05e.jpg)

![f6372f7ecdfb958e09d2ed8bcfe07239e092cab59c1a5f35cc9dab7925277ed3.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/f6372f7ecdfb958e09d2ed8bcfe07239e092cab59c1a5f35cc9dab7925277ed3.jpg)

![f909ac1ece7713532764e27854f3577a189f9350dc6b38b957701c594249087e.jpg](../icml_results/2221_Context-free%20Recognition%20with%20Transformers/tables/f909ac1ece7713532764e27854f3577a189f9350dc6b38b957701c594249087e.jpg)

<a id="paper-2193"></a>

## MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification


### Images

![1d69ba30e601c96d40a042af70a0153f41f93d706fc4692acf9f6e624519d010.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/1d69ba30e601c96d40a042af70a0153f41f93d706fc4692acf9f6e624519d010.jpg)

![1f9ac46a03e0c9798a3bfcd380b13115d904758b54ee50aa1a6c20b6bbfe9ee1.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/1f9ac46a03e0c9798a3bfcd380b13115d904758b54ee50aa1a6c20b6bbfe9ee1.jpg)

![22d20590c992d6ddfce2269b1a948868b8be9e64ff3209be0f8fd970acb5f2d3.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/22d20590c992d6ddfce2269b1a948868b8be9e64ff3209be0f8fd970acb5f2d3.jpg)

![293d5d97169580543dff36f11e732e8fe18ae785bb584efa4a7fbab6cbf917c7.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/293d5d97169580543dff36f11e732e8fe18ae785bb584efa4a7fbab6cbf917c7.jpg)

![2ac9f69e4289739fc62c5db0c38a35ccd20a1ae90be80719e5ad2c11011169a3.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/2ac9f69e4289739fc62c5db0c38a35ccd20a1ae90be80719e5ad2c11011169a3.jpg)

![3deba5ec015140061b4e8acf9025a2aa876b496ab79f9dc80e9a2778e9e0e7e2.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/3deba5ec015140061b4e8acf9025a2aa876b496ab79f9dc80e9a2778e9e0e7e2.jpg)

![476987191b843d0cc8fe9221f0b832b436c0945b0a13eef7701a7bdeb8cb2793.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/476987191b843d0cc8fe9221f0b832b436c0945b0a13eef7701a7bdeb8cb2793.jpg)

![47c04bb787a7785c0b94847475ee84e540e0ebf36c15af9f15704f4218b81ef6.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/47c04bb787a7785c0b94847475ee84e540e0ebf36c15af9f15704f4218b81ef6.jpg)

![7c12a840d70efc35853c25c8d22b9fe3d155532bb477a60f91299f02c8c08020.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/7c12a840d70efc35853c25c8d22b9fe3d155532bb477a60f91299f02c8c08020.jpg)

![7cb39024df8433b9bf0e4ec9765b028f71b07e54bf17ac346f8d4a1435953906.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/7cb39024df8433b9bf0e4ec9765b028f71b07e54bf17ac346f8d4a1435953906.jpg)

![7f0b17e794377d453d0d0a3dbe65c343921b50a79ffb23bccefdb4fe32455492.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/7f0b17e794377d453d0d0a3dbe65c343921b50a79ffb23bccefdb4fe32455492.jpg)

![aae278d73d02bdb07d1c739ff01576bc67c5c667bd51bf092d8d0c90a2ed9fe1.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/aae278d73d02bdb07d1c739ff01576bc67c5c667bd51bf092d8d0c90a2ed9fe1.jpg)

![ba021924a3f450cc7b77fd091912bbef609ea743f7ea5a6dab83d4c0c98cfab2.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/ba021924a3f450cc7b77fd091912bbef609ea743f7ea5a6dab83d4c0c98cfab2.jpg)

![c4eca9a912b170e70f193d416859b4dc22dcddef63775ea1b28ff9de4bf616d1.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/c4eca9a912b170e70f193d416859b4dc22dcddef63775ea1b28ff9de4bf616d1.jpg)

![ce1880ca5042068780c1ae60cafa6175a13cd8711aba4ae2e577fb3c157cd8cc.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/ce1880ca5042068780c1ae60cafa6175a13cd8711aba4ae2e577fb3c157cd8cc.jpg)

![dceec2e72fe9a0e04acef31d97feb2bfeca09b77bc52bf9e1f76c77255aa3d1b.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/dceec2e72fe9a0e04acef31d97feb2bfeca09b77bc52bf9e1f76c77255aa3d1b.jpg)

![f16af06c2196cf434437783d491b59ad15b33e831c3427e6a2ca4f8b85d7aec3.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/f16af06c2196cf434437783d491b59ad15b33e831c3427e6a2ca4f8b85d7aec3.jpg)

![f467964ff796b1b74f23229b8b9c2da673125db6a9b043ef315b1fada9c31048.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/images/f467964ff796b1b74f23229b8b9c2da673125db6a9b043ef315b1fada9c31048.jpg)

### Tables

![0314e862ab33e52128ab89a1dcfe14e4b2470ad202ef0fa7c491fdbe8ee5b7ae.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/tables/0314e862ab33e52128ab89a1dcfe14e4b2470ad202ef0fa7c491fdbe8ee5b7ae.jpg)

![9c24fd8cbf5dbd11f0e98506937d42860047df5f64bc6e7270ddc330130cc4ae.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/tables/9c24fd8cbf5dbd11f0e98506937d42860047df5f64bc6e7270ddc330130cc4ae.jpg)

![bb0355520495666416236824e34c0712f16da60660b04a2f5b3f1b657f1e0429.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/tables/bb0355520495666416236824e34c0712f16da60660b04a2f5b3f1b657f1e0429.jpg)

![e91b4629615d18ce6ecb3e3dc4638bd1c99be106496ad6be9ddf802f80d34f49.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/tables/e91b4629615d18ce6ecb3e3dc4638bd1c99be106496ad6be9ddf802f80d34f49.jpg)

![fce40a3a58496b18dfd2719eac6951e36da796b99614576bd770ad1b69390955.jpg](../icml_results/2222_MiniMax%20Learning%20of%20Interpretable%20Factored%20Stochastic%20Policies%20from%20Conjoint%20Data%2C%20with%20Uncertainty%20/tables/fce40a3a58496b18dfd2719eac6951e36da796b99614576bd770ad1b69390955.jpg)

<a id="paper-2194"></a>

## Tiny Brains, Giant Impact: Uncovering the Keystone Neurons of LLM with Just a Few Prompts


### Images

![0a442a3a0042b56667e94c768e5ee4be6d6711363a0a49935b11c11cddf2b4ec.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/0a442a3a0042b56667e94c768e5ee4be6d6711363a0a49935b11c11cddf2b4ec.jpg)

![0fd7a4e7d222b55c6cba0a6139403e396f84894124651b4c29c70445656efc18.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/0fd7a4e7d222b55c6cba0a6139403e396f84894124651b4c29c70445656efc18.jpg)

![10cfd7f4ab570380977bc545fdfe11d53e4991f30b60a3e20b90fb871f4dd1f6.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/10cfd7f4ab570380977bc545fdfe11d53e4991f30b60a3e20b90fb871f4dd1f6.jpg)

![5771835e3b94629084efc00a4d2f8acd1689eb1e18fa0c86c661b74000fb3ad2.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/5771835e3b94629084efc00a4d2f8acd1689eb1e18fa0c86c661b74000fb3ad2.jpg)

![8b45c646be066072da2ed6090e7238630fef770f87cb4dce0ad58b58eb98f08d.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/8b45c646be066072da2ed6090e7238630fef770f87cb4dce0ad58b58eb98f08d.jpg)

![90ccacf656d5def9da6b2db15407c9d98da2d18fe89cb08ff7f0389c5db666e4.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/images/90ccacf656d5def9da6b2db15407c9d98da2d18fe89cb08ff7f0389c5db666e4.jpg)

### Tables

![05f260c7efc33b5fc31f05dd34f039f65719a0866f07256b213da36b5fe1338e.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/05f260c7efc33b5fc31f05dd34f039f65719a0866f07256b213da36b5fe1338e.jpg)

![1b8fbff9f68602dc15663b1354dd1e021e2e807dc2aafb7f1dbdb34b5ed2b035.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/1b8fbff9f68602dc15663b1354dd1e021e2e807dc2aafb7f1dbdb34b5ed2b035.jpg)

![4b0ae84b3d1076d346b245a195198098accf85bcdf12e79f82a00e873ae04f27.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/4b0ae84b3d1076d346b245a195198098accf85bcdf12e79f82a00e873ae04f27.jpg)

![4b96372a290fd0e42ea10cefcc6991f1fc8e8d4442f2259e4c06716561fa7c5c.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/4b96372a290fd0e42ea10cefcc6991f1fc8e8d4442f2259e4c06716561fa7c5c.jpg)

![4c8d91a103a63aad604984e7bd29bdb3824edbb470b4b270c9e39212938c86f3.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/4c8d91a103a63aad604984e7bd29bdb3824edbb470b4b270c9e39212938c86f3.jpg)

![51343ab9578ecf57f283dbe53abf83d27bd0036ad43403d798acb0ef134c789f.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/51343ab9578ecf57f283dbe53abf83d27bd0036ad43403d798acb0ef134c789f.jpg)

![598a4283b77e9a64cf2b4290dd4c7d5d534f656a6c36b36bb15d2c3c76f3c158.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/598a4283b77e9a64cf2b4290dd4c7d5d534f656a6c36b36bb15d2c3c76f3c158.jpg)

![ac9a2ce59854b5a050a8ee37b5344ebc63212cad798c16c7607ba367698eece4.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/ac9a2ce59854b5a050a8ee37b5344ebc63212cad798c16c7607ba367698eece4.jpg)

![ad5aaa0c8b319b3bbd297a1e8170662d9fec8f1e5632140e1e14f57ce4a598da.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/ad5aaa0c8b319b3bbd297a1e8170662d9fec8f1e5632140e1e14f57ce4a598da.jpg)

![affdce0fbf682abd174c556277dbe573210f4118bfe25c0943dfb362319886db.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/affdce0fbf682abd174c556277dbe573210f4118bfe25c0943dfb362319886db.jpg)

![d2e05304c1d3f7117f9a59c1317139bec65c7fe7d0250bb18a19f2ed3cec4062.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/d2e05304c1d3f7117f9a59c1317139bec65c7fe7d0250bb18a19f2ed3cec4062.jpg)

![f83963c2a316d638ea102dce3a9b197c2799cb183c75df70036197197a27826d.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/f83963c2a316d638ea102dce3a9b197c2799cb183c75df70036197197a27826d.jpg)

![fe88dda290a66f59c019851b34b1e75d11322c6cd49955bab23f8d46ee59c9ca.jpg](../icml_results/2223_Tiny%20Brains%2C%20Giant%20Impact_%20Uncovering%20the%20Keystone%20Neurons%20of%20LLM%20with%20Just%20a%20Few%20Prompts/tables/fe88dda290a66f59c019851b34b1e75d11322c6cd49955bab23f8d46ee59c9ca.jpg)

<a id="paper-2195"></a>

## ShapCCS: Shapley-Driven Client Coreset Selection in Federated Learning


### Images

![2c22f9317c55dabc4dc686c55b79f8910d0d5dad2f77eb26973adbae0e89bedc.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/images/2c22f9317c55dabc4dc686c55b79f8910d0d5dad2f77eb26973adbae0e89bedc.jpg)

![8a5763b8ef39f8be1e78a572317064d22b091cc48af73d89fbc3cc9f736f9425.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/images/8a5763b8ef39f8be1e78a572317064d22b091cc48af73d89fbc3cc9f736f9425.jpg)

![9a2f4795a4a6a95170ea7b65e7f6daf41506b9e4a1954604a36f8ef21f4b1ea1.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/images/9a2f4795a4a6a95170ea7b65e7f6daf41506b9e4a1954604a36f8ef21f4b1ea1.jpg)

![e26ed4405ac78406d7ce5f0cda38bb2684c357f52daf892bd443ca3e19802fc5.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/images/e26ed4405ac78406d7ce5f0cda38bb2684c357f52daf892bd443ca3e19802fc5.jpg)

![ee84fe55847ccd4c9e80e8bacbd75777179fb71e8bac8628938fbb15dc07ef57.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/images/ee84fe55847ccd4c9e80e8bacbd75777179fb71e8bac8628938fbb15dc07ef57.jpg)

### Tables

![0f97a74a4c83c21968861f9b28864bee080ebc945729b3d4dcfdb0c8303206a5.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/0f97a74a4c83c21968861f9b28864bee080ebc945729b3d4dcfdb0c8303206a5.jpg)

![44bd3c335a799306539e5ac22b5048e789c6455f24aec2040612d63989a9cf64.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/44bd3c335a799306539e5ac22b5048e789c6455f24aec2040612d63989a9cf64.jpg)

![a6d843207caf2e13b7c4b07ab3280b09ff7ed454a4c43da67bd2afc3e89d38ed.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/a6d843207caf2e13b7c4b07ab3280b09ff7ed454a4c43da67bd2afc3e89d38ed.jpg)

![ad1e9da7f9a88002b8961be764a634e6f6c9284a144742bc2afdc8e8aa249daf.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/ad1e9da7f9a88002b8961be764a634e6f6c9284a144742bc2afdc8e8aa249daf.jpg)

![d78e8e9eea742803d3415f2396808facf07b9703abcf6964de17dcaa8def0bc3.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/d78e8e9eea742803d3415f2396808facf07b9703abcf6964de17dcaa8def0bc3.jpg)

![f3774ecd2c8d3c46f5706e7d65caf46c862dc1ece07b8bd86b1a2eb7bbffa563.jpg](../icml_results/2224_ShapCCS_%20Shapley-Driven%20Client%20Coreset%20Selection%20in%20Federated%20Learning/tables/f3774ecd2c8d3c46f5706e7d65caf46c862dc1ece07b8bd86b1a2eb7bbffa563.jpg)

<a id="paper-2196"></a>

## TopBench: A Benchmark for Implicit Predictive Reasoning in Tabular Question Answering


### Images

![0092cc132b4be5a0d3d9f0c9156358ffffc9fde2585fe15c2ab86a98bc629cbe.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/0092cc132b4be5a0d3d9f0c9156358ffffc9fde2585fe15c2ab86a98bc629cbe.jpg)

![01896a61be2cda7b167903406ebcb2b578c91d805e09f063e3042eef89f6dcb1.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/01896a61be2cda7b167903406ebcb2b578c91d805e09f063e3042eef89f6dcb1.jpg)

![0336c21991d7607fa5e174b684b01d5a1fc1bef70e9508f75a8c62133700d806.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/0336c21991d7607fa5e174b684b01d5a1fc1bef70e9508f75a8c62133700d806.jpg)

![0f35c1f090ac18fcfb078aeb3a373736fcdc51f5f015d2a4b2784eb3f9a282b5.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/0f35c1f090ac18fcfb078aeb3a373736fcdc51f5f015d2a4b2784eb3f9a282b5.jpg)

![31d6e4973af2a418ffecbd3ba3f4771cce9a896327861b6c497de027416a0a1b.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/31d6e4973af2a418ffecbd3ba3f4771cce9a896327861b6c497de027416a0a1b.jpg)

![385fd2a33909a7f53ec3fecb88c58602c74f77411cfce8cb8e303ea5d6a1259e.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/385fd2a33909a7f53ec3fecb88c58602c74f77411cfce8cb8e303ea5d6a1259e.jpg)

![4549f3e452f4cd8f3ed4d749f65b3dcd36608534d81b2be65670ed6fbdf7762d.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/4549f3e452f4cd8f3ed4d749f65b3dcd36608534d81b2be65670ed6fbdf7762d.jpg)

![49239e994e2026019f35c45efd57f9394242951c124be83774134c3de1fc779d.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/49239e994e2026019f35c45efd57f9394242951c124be83774134c3de1fc779d.jpg)

![5b602a079500150be1d517e2929a1a34006779b703d37bcc0cb835e6a3bfb67c.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/5b602a079500150be1d517e2929a1a34006779b703d37bcc0cb835e6a3bfb67c.jpg)

![63b45f6dab4e982bd273917a0e247f5fffebba5880bdc1430464f58db5ddb6de.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/63b45f6dab4e982bd273917a0e247f5fffebba5880bdc1430464f58db5ddb6de.jpg)

![738bbccde794e35716ca1554bf6f8a0f318c66e0e2e07b0c9e0d67169867590e.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/738bbccde794e35716ca1554bf6f8a0f318c66e0e2e07b0c9e0d67169867590e.jpg)

![764a96616b0b7af01abe3d2986099f92922f4e26f1ec9ae486134906b9a99d18.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/764a96616b0b7af01abe3d2986099f92922f4e26f1ec9ae486134906b9a99d18.jpg)

![841fb54b460beeb70c8c8615fee642ed5614e5a42a35f77702fa1fd5a5d40a5b.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/841fb54b460beeb70c8c8615fee642ed5614e5a42a35f77702fa1fd5a5d40a5b.jpg)

![9231d61aa23dfe675041a7859a01502907a9d5c0ae7216afba8f34c96b90201e.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/9231d61aa23dfe675041a7859a01502907a9d5c0ae7216afba8f34c96b90201e.jpg)

![9f1de3d3d90178941c070c6932eb365473968334ee5347341474d3453ed3699b.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/9f1de3d3d90178941c070c6932eb365473968334ee5347341474d3453ed3699b.jpg)

![ae4acdca4b574d6e43f8cb610d861e6ea797fa37884164a88a40ded89735a8f2.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/ae4acdca4b574d6e43f8cb610d861e6ea797fa37884164a88a40ded89735a8f2.jpg)

![ae70ab8e0002637807374911f32e02d81154cf8a29551ce263f8d3192ac92e48.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/ae70ab8e0002637807374911f32e02d81154cf8a29551ce263f8d3192ac92e48.jpg)

![cae9b6c9994d6fd29a9f5b161c9ba2b7741500860fe69987d06480053694eed6.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/cae9b6c9994d6fd29a9f5b161c9ba2b7741500860fe69987d06480053694eed6.jpg)

![e36bd7161d2c31afcebaf2f6d7994e058743f0e37bbf27ef6dd75fd40ae69e3c.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/e36bd7161d2c31afcebaf2f6d7994e058743f0e37bbf27ef6dd75fd40ae69e3c.jpg)

![ed5f2b360a9bf2ee0de1acc3474558fad9f501758449d73f548b9a22894ff80f.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/ed5f2b360a9bf2ee0de1acc3474558fad9f501758449d73f548b9a22894ff80f.jpg)

![f57db8ca748cec7cd6d5f9f0d952c310f2642fa3e488060d35725002f7cc8f63.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/images/f57db8ca748cec7cd6d5f9f0d952c310f2642fa3e488060d35725002f7cc8f63.jpg)

### Tables

![02ed1151c0742052ecd0ff7fcd113bcd2d878b2832e102f0df89caf59b47f2f0.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/02ed1151c0742052ecd0ff7fcd113bcd2d878b2832e102f0df89caf59b47f2f0.jpg)

![2ca33124fc1024b830919a7294ce2f3b78ae12fdeb33cc0c8131a067965a0629.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/2ca33124fc1024b830919a7294ce2f3b78ae12fdeb33cc0c8131a067965a0629.jpg)

![36fd1e1cc6e4fcc35129729419e4110b78f0ee76c970260273932af3f62d1153.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/36fd1e1cc6e4fcc35129729419e4110b78f0ee76c970260273932af3f62d1153.jpg)

![37f5474e5381ef92427a6422ebea853c48b90357f02424f4afeb893e50ef4742.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/37f5474e5381ef92427a6422ebea853c48b90357f02424f4afeb893e50ef4742.jpg)

![3b17242922432bba0d8cdee9f280b7129898351c8d9303ffc9d02f118f19d1d9.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/3b17242922432bba0d8cdee9f280b7129898351c8d9303ffc9d02f118f19d1d9.jpg)

![44ff4fb040639247e0e7d989ba6e964b3146a200345da9108d2cc6218f844a50.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/44ff4fb040639247e0e7d989ba6e964b3146a200345da9108d2cc6218f844a50.jpg)

![46f9d60bb2519a479a20380540a48e20a2ac4f61b516817838b8146be179c401.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/46f9d60bb2519a479a20380540a48e20a2ac4f61b516817838b8146be179c401.jpg)

![57a89448af17e98a4ff655397db2f040613adaa2a9a5530da8f7fbc2d91bbb9b.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/57a89448af17e98a4ff655397db2f040613adaa2a9a5530da8f7fbc2d91bbb9b.jpg)

![58ca1a77c0d5601d634b82fcf0bbd591cfca1595c09c7f4a5f37ae337961f06d.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/58ca1a77c0d5601d634b82fcf0bbd591cfca1595c09c7f4a5f37ae337961f06d.jpg)

![5ff67b59f4137baebb40b5a11b63ada144eb2b5e9cd802b2d047c668ab7e2f94.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/5ff67b59f4137baebb40b5a11b63ada144eb2b5e9cd802b2d047c668ab7e2f94.jpg)

![61f84904275bb45e21554a428e98613e93037f7b1f269770a80be9a4da51dad6.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/61f84904275bb45e21554a428e98613e93037f7b1f269770a80be9a4da51dad6.jpg)

![65a07a5128a3d7aa210b8ea539e292fdf415afcbf5405a48fc0760b31fd8aae0.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/65a07a5128a3d7aa210b8ea539e292fdf415afcbf5405a48fc0760b31fd8aae0.jpg)

![73d87640aa167af02a6f6fcfe0133bbf4966edc7995ecc7a331f646972a71d4d.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/73d87640aa167af02a6f6fcfe0133bbf4966edc7995ecc7a331f646972a71d4d.jpg)

![82ae4297b282600ffbf36902593a6e9fccb3b6af51e655ee3e5211f17a17bd94.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/82ae4297b282600ffbf36902593a6e9fccb3b6af51e655ee3e5211f17a17bd94.jpg)

![94fa10afd9e4addf15853b21b5e5f1f84585ef049b48bc42bccd59bd30dd91ac.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/94fa10afd9e4addf15853b21b5e5f1f84585ef049b48bc42bccd59bd30dd91ac.jpg)

![c82fe4093d789f57c8860cc381623bb61f24dbd133c13da50396551806c9480b.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/c82fe4093d789f57c8860cc381623bb61f24dbd133c13da50396551806c9480b.jpg)

![ce9b6e80ff81adbbc2fc587cc06a84c6389088c45d79f3ef7f06620c3e3fff3a.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/ce9b6e80ff81adbbc2fc587cc06a84c6389088c45d79f3ef7f06620c3e3fff3a.jpg)

![fcf2d32a835b1d06b059031e81ed406596a61fa75f1d059e771c609746bc30bf.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/fcf2d32a835b1d06b059031e81ed406596a61fa75f1d059e771c609746bc30bf.jpg)

![ff780e40c856b735744ada0274cfe9085a1232e75e3cdb355db90d86579daf6d.jpg](../icml_results/2225_TopBench_%20A%20Benchmark%20for%20Implicit%20Predictive%20Reasoning%20in%20Tabular%20Question%20Answering/tables/ff780e40c856b735744ada0274cfe9085a1232e75e3cdb355db90d86579daf6d.jpg)

<a id="paper-2197"></a>

## Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents


### Images

![18f700dffc9cb909a307443127a7cf554165669e57537f42121c3a396a8a807d.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/18f700dffc9cb909a307443127a7cf554165669e57537f42121c3a396a8a807d.jpg)

![2d90af278f5c3b382a0d7ea00c3df858bff6c78220fa2c6583288931ac5d6548.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/2d90af278f5c3b382a0d7ea00c3df858bff6c78220fa2c6583288931ac5d6548.jpg)

![36b2185713c35ce00156c4a6f4c7b90c0c6be501afc4ecbe2417ab7b14e7b1e6.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/36b2185713c35ce00156c4a6f4c7b90c0c6be501afc4ecbe2417ab7b14e7b1e6.jpg)

![5676b63d58f316a4b5fd69eabef60150e7d346590450d56976ec4edb850e2bd7.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/5676b63d58f316a4b5fd69eabef60150e7d346590450d56976ec4edb850e2bd7.jpg)

![c34960ed542cf37fe81ef170f2c195d6bdc52e0b320340547a04231774a14619.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/c34960ed542cf37fe81ef170f2c195d6bdc52e0b320340547a04231774a14619.jpg)

![c456b1898fca4d2f3b2a08d36f722f8a01dacd8fa45d1cd8264a0be1dc3235c0.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/c456b1898fca4d2f3b2a08d36f722f8a01dacd8fa45d1cd8264a0be1dc3235c0.jpg)

![ce6280dc850cddc88ffb3966da35feea97097aac5ea8976feb1647f4e6ebd353.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/ce6280dc850cddc88ffb3966da35feea97097aac5ea8976feb1647f4e6ebd353.jpg)

![f2ba6e6fffb733e954b892295a2f5e55494d24a2fd36d1e98c8f9cd50d5284eb.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/f2ba6e6fffb733e954b892295a2f5e55494d24a2fd36d1e98c8f9cd50d5284eb.jpg)

![f774211a729af380bbd364f4d824ac0ae2ba67c5db2528e2534e72adc15e6047.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/images/f774211a729af380bbd364f4d824ac0ae2ba67c5db2528e2534e72adc15e6047.jpg)

### Tables

![3028c81d475700c525ba7d856eacd66dc0e07378e3cbcda82af6124afbc02633.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/3028c81d475700c525ba7d856eacd66dc0e07378e3cbcda82af6124afbc02633.jpg)

![32ea42e40f38b318997ac75c220bb2d92e50e2c72db4721fa40d2ad81e90606b.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/32ea42e40f38b318997ac75c220bb2d92e50e2c72db4721fa40d2ad81e90606b.jpg)

![494c800001a44e03185de4ca6604597d7264ecab724085b33ff297625466af24.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/494c800001a44e03185de4ca6604597d7264ecab724085b33ff297625466af24.jpg)

![5af7eecc9515c2de3e3f8eba8f2f2afd3b17423050b42d0fc6cec1bc83de1e4f.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/5af7eecc9515c2de3e3f8eba8f2f2afd3b17423050b42d0fc6cec1bc83de1e4f.jpg)

![aaf0895f490bf3a11b474582c5967f79787b0207f0faa909f99faeec0765a235.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/aaf0895f490bf3a11b474582c5967f79787b0207f0faa909f99faeec0765a235.jpg)

![aeecae00f2bfa7e317bd87b53e710d8e0e0d7e2fb12f2bc14608eb7efb66102e.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/aeecae00f2bfa7e317bd87b53e710d8e0e0d7e2fb12f2bc14608eb7efb66102e.jpg)

![e188c8611a185834c6cad696cb467530278df26b79ae5a3fa0abdea2fcc46984.jpg](../icml_results/2226_Memory%20is%20Reconstructed%2C%20Not%20Retrieved_%20Graph%20Memory%20for%20LLM%20Agents/tables/e188c8611a185834c6cad696cb467530278df26b79ae5a3fa0abdea2fcc46984.jpg)

<a id="paper-2198"></a>

## Recovering Hidden Reward in Diffusion-Based Policies


### Images

![4295bc1152d6b6531b22fe75aae1910feadf177cea921588d10b8d774d96f866.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/4295bc1152d6b6531b22fe75aae1910feadf177cea921588d10b8d774d96f866.jpg)

![53a383e955af8189cd51c3be27d45810eca2c032adccf2809ea6eb81b84992ba.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/53a383e955af8189cd51c3be27d45810eca2c032adccf2809ea6eb81b84992ba.jpg)

![7b6a9682dee39cb38ec964f5e4830c7980097038f34241fff1c4f904084cfc9a.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/7b6a9682dee39cb38ec964f5e4830c7980097038f34241fff1c4f904084cfc9a.jpg)

![7bc8dfc1b1963efa156d923eee6360164ffdaae6bee18d217bda221925ca23c1.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/7bc8dfc1b1963efa156d923eee6360164ffdaae6bee18d217bda221925ca23c1.jpg)

![8d139fe51ed4029c54853e228f8843736a43e4dd2e90ac0cef3dd1d74c8c80fd.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/8d139fe51ed4029c54853e228f8843736a43e4dd2e90ac0cef3dd1d74c8c80fd.jpg)

![902abc87bf836ac3111cec2ffc33154100d9af3144cf2d94cad79741e9229aa7.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/902abc87bf836ac3111cec2ffc33154100d9af3144cf2d94cad79741e9229aa7.jpg)

![a9990f87adcf37d48fb9d45cb806c16b6ed888fc49ebf75d88eadb24a5a1b9b6.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/images/a9990f87adcf37d48fb9d45cb806c16b6ed888fc49ebf75d88eadb24a5a1b9b6.jpg)

### Tables

![17f53b1033e8598079e2c73cb5fbfdc5512a569d60b1e0266875df88aee2f59d.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/17f53b1033e8598079e2c73cb5fbfdc5512a569d60b1e0266875df88aee2f59d.jpg)

![186c92198e2516c7ee8e4d0fd86ad11bf001194e975424076ca2cbe5607cf805.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/186c92198e2516c7ee8e4d0fd86ad11bf001194e975424076ca2cbe5607cf805.jpg)

![19f1ec7bcb2e29c58c6c2f3a01a5d2fadc930c9465764e32e6d739e8b03af1fd.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/19f1ec7bcb2e29c58c6c2f3a01a5d2fadc930c9465764e32e6d739e8b03af1fd.jpg)

![51e2afd5657d307ab935b44531e18dd71a2dba34e1f18c7681df8998976c7b6a.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/51e2afd5657d307ab935b44531e18dd71a2dba34e1f18c7681df8998976c7b6a.jpg)

![7e9b6da63c602c6d526310505c07076bf59a3c46c8c9f1f18389492ef235bb6c.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/7e9b6da63c602c6d526310505c07076bf59a3c46c8c9f1f18389492ef235bb6c.jpg)

![d7d8147d5955f7f0c70667fb629f2018db1d57c28c1ebd92b092354749b7cfc6.jpg](../icml_results/2227_Recovering%20Hidden%20Reward%20in%20Diffusion-Based%20Policies/tables/d7d8147d5955f7f0c70667fb629f2018db1d57c28c1ebd92b092354749b7cfc6.jpg)

<a id="paper-2199"></a>

## Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles


### Images

![e9a47df2256049571d6ed54a318255a1bc9d3ea30b8fedfc1f7f631de5c74ef5.jpg](../icml_results/2228_Lower%20Complexity%20Bounds%20for%20Nonconvex-Strongly-Convex%20Bilevel%20Optimization%20with%20First-Order%20Oracles/images/e9a47df2256049571d6ed54a318255a1bc9d3ea30b8fedfc1f7f631de5c74ef5.jpg)

<a id="paper-2200"></a>

## DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing


### Images

![058be634f70474aefeb3a68190fc7faf0fb8c9be954107160555b246d2441979.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/058be634f70474aefeb3a68190fc7faf0fb8c9be954107160555b246d2441979.jpg)

![0f45249c6ea32046bb5d472ca46cb62597c596df9dddad65056548379dc65f06.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/0f45249c6ea32046bb5d472ca46cb62597c596df9dddad65056548379dc65f06.jpg)

![10817c62e45c3926c0dd820227a89065cfe0dd7e9dfb53bb90d2c6a788944a51.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/10817c62e45c3926c0dd820227a89065cfe0dd7e9dfb53bb90d2c6a788944a51.jpg)

![22a1390df0a582c62e9345dfc038795e29332d2a28ace8c4bddb04433f7d77c2.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/22a1390df0a582c62e9345dfc038795e29332d2a28ace8c4bddb04433f7d77c2.jpg)

![53a05ad4876e1100f177d4c19fe73514e657a1e215b1050c3e4f22cc5b2840cb.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/53a05ad4876e1100f177d4c19fe73514e657a1e215b1050c3e4f22cc5b2840cb.jpg)

![696a2c174900758653187f89a66d2fe0e2f98cb676712fab823c7740a3b11f45.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/696a2c174900758653187f89a66d2fe0e2f98cb676712fab823c7740a3b11f45.jpg)

![b2c4a22b8d9a567b2f9c3772266decd75c90a8cc39f96eb570c2a62718255b1f.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/b2c4a22b8d9a567b2f9c3772266decd75c90a8cc39f96eb570c2a62718255b1f.jpg)

![fb96674cd53720585d79ca8fd776c1af77db3c1da3bc192d1469aa5859b60ff1.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/images/fb96674cd53720585d79ca8fd776c1af77db3c1da3bc192d1469aa5859b60ff1.jpg)

### Tables

![091e23975a4f55e0b22a56b6d8db9e78c72dcb092fc3bbcff35b97a364f42645.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/091e23975a4f55e0b22a56b6d8db9e78c72dcb092fc3bbcff35b97a364f42645.jpg)

![0efc2dc4496d5d804fbeb2d2b5d248ca23926b7f196758c97322280e3628d053.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/0efc2dc4496d5d804fbeb2d2b5d248ca23926b7f196758c97322280e3628d053.jpg)

![0f95dd28a875425711ebba86399d725a87f084f3e6ffe53e82e2413a737e0ad8.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/0f95dd28a875425711ebba86399d725a87f084f3e6ffe53e82e2413a737e0ad8.jpg)

![114993f135ab18f174ea7dde83fe5192ea8a9f83fa2e0e2fdee8bac7acdbaba7.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/114993f135ab18f174ea7dde83fe5192ea8a9f83fa2e0e2fdee8bac7acdbaba7.jpg)

![133138a3c515848c7f1da58d3dda90de039fc9b43853b54373d73a38cc5d25e7.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/133138a3c515848c7f1da58d3dda90de039fc9b43853b54373d73a38cc5d25e7.jpg)

![2dfb674ba93778985e98e3592104a64466e33a5d1ce1e90ee61dfe17b3da9582.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/2dfb674ba93778985e98e3592104a64466e33a5d1ce1e90ee61dfe17b3da9582.jpg)

![362d4da148fe63120de6aea18ef84ca4e84fdb9ec30b1514100c287e0f0ee6c4.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/362d4da148fe63120de6aea18ef84ca4e84fdb9ec30b1514100c287e0f0ee6c4.jpg)

![45b6d0f5faf54cdb098969c96c2afa08706e212e2ad598e765fb7940906b58e4.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/45b6d0f5faf54cdb098969c96c2afa08706e212e2ad598e765fb7940906b58e4.jpg)

![52f9fec43d81f74a78d83833c8c8154a31474bc89f95da14d3ae0893ba3ded26.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/52f9fec43d81f74a78d83833c8c8154a31474bc89f95da14d3ae0893ba3ded26.jpg)

![970135b3f1993f0bb3fb9ddf505963826fb8156545ad02c933f5fd454bdca31b.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/970135b3f1993f0bb3fb9ddf505963826fb8156545ad02c933f5fd454bdca31b.jpg)

![cc415dea3ee8a17b485e52eb8000f31c18ad6e7843e47fe2af1b3a95de871a4f.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/cc415dea3ee8a17b485e52eb8000f31c18ad6e7843e47fe2af1b3a95de871a4f.jpg)

![fd9f6569c267a9e8fdbe5ae29495e71be03c907da3d7264499d6f9d0b773561b.jpg](../icml_results/2229_DiscoForcing_%20A%20Unified%20Framework%20for%20Real-Time%20Audio-Driven%20Character%20Control%20with%20Diffusion%20Forcin/tables/fd9f6569c267a9e8fdbe5ae29495e71be03c907da3d7264499d6f9d0b773561b.jpg)

<a id="paper-2201"></a>

## Learn-to-learn on Arbitrary Textual Conditioning: A Hypernetwork-Driven Meta-gated LLM


### Images

![133e9b5acbd6750d132c2f9dfe8ff2cd8f3a98bf9deaece0ae84f462d5db7933.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/133e9b5acbd6750d132c2f9dfe8ff2cd8f3a98bf9deaece0ae84f462d5db7933.jpg)

![1bed7ccb324cda348a053bfee6c3a7d19f8daa2ee79b3ee098c00fe987ebeec5.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/1bed7ccb324cda348a053bfee6c3a7d19f8daa2ee79b3ee098c00fe987ebeec5.jpg)

![32498dbc851b92c5ecd6d1368b26f6a2813cb761e1adb57387037f4dc316ebaa.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/32498dbc851b92c5ecd6d1368b26f6a2813cb761e1adb57387037f4dc316ebaa.jpg)

![3e7273446f27808488f88f91fd366cdb663e0b17800c0bd87308a820de73ac95.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/3e7273446f27808488f88f91fd366cdb663e0b17800c0bd87308a820de73ac95.jpg)

![3fb921fb7cf8e7dfd767e7eb848f22e65bcb327ebd80b929e8c500ea26738c85.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/3fb921fb7cf8e7dfd767e7eb848f22e65bcb327ebd80b929e8c500ea26738c85.jpg)

![5a4f2e9758d5ee8009cd95c975f49220ad9df05bafa44f8cc71fab80703d2d6e.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/5a4f2e9758d5ee8009cd95c975f49220ad9df05bafa44f8cc71fab80703d2d6e.jpg)

![68f898c245d063693b8f23cda065232deb994e03e33569725fa950841421d05e.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/68f898c245d063693b8f23cda065232deb994e03e33569725fa950841421d05e.jpg)

![72dc1baf06c0a6a1b6103b9c45ff0ca59f3fac775b5427173f85589712983dc2.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/72dc1baf06c0a6a1b6103b9c45ff0ca59f3fac775b5427173f85589712983dc2.jpg)

![930c76151c90eb202b6bd1926a4fa93f0a1fb6375210d7953620505ec5d535d2.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/930c76151c90eb202b6bd1926a4fa93f0a1fb6375210d7953620505ec5d535d2.jpg)

![a4de970aefd2ccaa2bf25cb685235d6ee322e92faed2a9bdbbb9c94aa39cb13e.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/a4de970aefd2ccaa2bf25cb685235d6ee322e92faed2a9bdbbb9c94aa39cb13e.jpg)

![b549b84ca9789de8034417661e3ce10feab1f70d9513649d814a0c5c6650e923.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/b549b84ca9789de8034417661e3ce10feab1f70d9513649d814a0c5c6650e923.jpg)

![bc2011185353d968555a618d4837c802d94eae223867d977bb8e55cdfbba4f9d.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/bc2011185353d968555a618d4837c802d94eae223867d977bb8e55cdfbba4f9d.jpg)

![ddb0acdb1d41f147d6f3d5dc8d64f765ee22b77f3bfb2f0249539853fd9089cc.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/ddb0acdb1d41f147d6f3d5dc8d64f765ee22b77f3bfb2f0249539853fd9089cc.jpg)

![e96be726b4722c7053eca7c790273b8105a029d33bf9745aad29def6d70e1770.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/e96be726b4722c7053eca7c790273b8105a029d33bf9745aad29def6d70e1770.jpg)

![f6718f3b0b6f6eda52ec5ee89a5caa34b801acd39a4f1965bee87bbc5f8b5bc6.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/f6718f3b0b6f6eda52ec5ee89a5caa34b801acd39a4f1965bee87bbc5f8b5bc6.jpg)

![f69425e44d63add55173a40a73459a0c0de9f13a9e249722984715fa1356b422.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/images/f69425e44d63add55173a40a73459a0c0de9f13a9e249722984715fa1356b422.jpg)

### Tables

![020bbb76c1f9dd839db39cc7ee50dd2fb9335ff1c38c5e595a72d50eea82c910.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/020bbb76c1f9dd839db39cc7ee50dd2fb9335ff1c38c5e595a72d50eea82c910.jpg)

![096b1f7a633e346658154bc2048395992d8735545dd8c46eb3e8a954d9ff78b6.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/096b1f7a633e346658154bc2048395992d8735545dd8c46eb3e8a954d9ff78b6.jpg)

![0be1b468da3072465dd9714cc700a3db3c5062df754670b2bc327776f25a6f80.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/0be1b468da3072465dd9714cc700a3db3c5062df754670b2bc327776f25a6f80.jpg)

![14ec52f1dce2a905e47c5bf0e23978363f1e1a5c96b235d33e419e2b026fd9bf.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/14ec52f1dce2a905e47c5bf0e23978363f1e1a5c96b235d33e419e2b026fd9bf.jpg)

![2a6f7f79a012b57845f5799f2982399c2442fa2124824820551afd0fd44fc751.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/2a6f7f79a012b57845f5799f2982399c2442fa2124824820551afd0fd44fc751.jpg)

![3ac77f125a28d7af633cf2506ca5410813d1aef163285d9061454c000451729e.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/3ac77f125a28d7af633cf2506ca5410813d1aef163285d9061454c000451729e.jpg)

![4db38d6f160e841fb27a5e984da2dd7e2df86f46f3ae6d3d31a4192053dc90bb.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/4db38d6f160e841fb27a5e984da2dd7e2df86f46f3ae6d3d31a4192053dc90bb.jpg)

![561a302299d7b848551662af9d0811d1fa5643d8762394b1c0b99d6bf6fb46f1.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/561a302299d7b848551662af9d0811d1fa5643d8762394b1c0b99d6bf6fb46f1.jpg)

![5c614d3473421ddef1857d52a9f7fca32ee05428a456fd1b7a818c606c0c834d.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/5c614d3473421ddef1857d52a9f7fca32ee05428a456fd1b7a818c606c0c834d.jpg)

![88814d18890c98335ee86c5c7b5e5bc49246ff65924a42b16e88e3e04c6eb772.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/88814d18890c98335ee86c5c7b5e5bc49246ff65924a42b16e88e3e04c6eb772.jpg)

![88ac51f7cb911ebba12a9a295189938e9726fddff9e78ac8b4f58bea0b8aaebd.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/88ac51f7cb911ebba12a9a295189938e9726fddff9e78ac8b4f58bea0b8aaebd.jpg)

![96398d174f23afc093b7ec21dca70c9446907e03e0ba46e6070c77fb7845e455.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/96398d174f23afc093b7ec21dca70c9446907e03e0ba46e6070c77fb7845e455.jpg)

![a1e3ff55d2fb12faebf52027a8f7d6cb06191bda14ad17e2412423380f70e87d.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/a1e3ff55d2fb12faebf52027a8f7d6cb06191bda14ad17e2412423380f70e87d.jpg)

![a2f712fd26a82b19faed0caef1f5d7e198c7c8620ba62972b0e5a8fe61c79ba6.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/a2f712fd26a82b19faed0caef1f5d7e198c7c8620ba62972b0e5a8fe61c79ba6.jpg)

![b507160b89d15513705359e9b1ba6e0daa7376bd0c69379de6d5155ca066699c.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/b507160b89d15513705359e9b1ba6e0daa7376bd0c69379de6d5155ca066699c.jpg)

![bc5c73023b37e6baceb9905e58a63e71e0913be4a6fac3bc497ce66405460193.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/bc5c73023b37e6baceb9905e58a63e71e0913be4a6fac3bc497ce66405460193.jpg)

![db90d8067f0fc756d47d1f1ee7ff290a23aea05708d3dc1a73c62ebc0535f911.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/db90d8067f0fc756d47d1f1ee7ff290a23aea05708d3dc1a73c62ebc0535f911.jpg)

![e15e4c45f9f96285f470cbdef7965552e0e72ff14a56ea12c2f937caceb384da.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/e15e4c45f9f96285f470cbdef7965552e0e72ff14a56ea12c2f937caceb384da.jpg)

![faecaf72e168e7e40ed2d29bcd9cbc4a87a6338d27586da33b3ee2e58c0f054e.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/faecaf72e168e7e40ed2d29bcd9cbc4a87a6338d27586da33b3ee2e58c0f054e.jpg)

![ffba5d2001c54fc0fe07a361e72e4c977bbe4dd8454df28e5100700dc16b79e3.jpg](../icml_results/2230_Learn-to-learn%20on%20Arbitrary%20Textual%20Conditioning_%20A%20Hypernetwork-Driven%20Meta-gated%20LLM/tables/ffba5d2001c54fc0fe07a361e72e4c977bbe4dd8454df28e5100700dc16b79e3.jpg)

<a id="paper-2202"></a>

## Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening


### Images

![1800c005999931bd139352ff0d2ca119d987bb26e35efd2a6eb687953b21a383.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/1800c005999931bd139352ff0d2ca119d987bb26e35efd2a6eb687953b21a383.jpg)

![45715f202b375f6e700749655adc38d9560db9b26cca923ad2abc8e28248a490.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/45715f202b375f6e700749655adc38d9560db9b26cca923ad2abc8e28248a490.jpg)

![8716e60ae282709e47e8c2743d67f25b00729b660bbd4f413825f1fbd6c45f37.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/8716e60ae282709e47e8c2743d67f25b00729b660bbd4f413825f1fbd6c45f37.jpg)

![bc7732faad8317212fe8bde6e977ae04faa51f8a61c41e51fbd18d0121e99619.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/bc7732faad8317212fe8bde6e977ae04faa51f8a61c41e51fbd18d0121e99619.jpg)

![c3fa8d88787556cf6d4d50dfd06647bdaaedc496c9e0596aaa8df690110d64bf.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/c3fa8d88787556cf6d4d50dfd06647bdaaedc496c9e0596aaa8df690110d64bf.jpg)

![c724bb590b08a4bae2f760020eafb8cdf294c22f85539e11448adbfb294bb845.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/images/c724bb590b08a4bae2f760020eafb8cdf294c22f85539e11448adbfb294bb845.jpg)

### Tables

![0b5c4ae9ab72122b06ed20fa2fc5ed580b2c58f6c0d05cfdcf63f9d24f9376db.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/0b5c4ae9ab72122b06ed20fa2fc5ed580b2c58f6c0d05cfdcf63f9d24f9376db.jpg)

![100242dbe3ceee4682a7c682fda848d31fcb216f748bce47e179b50ce17bcdb9.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/100242dbe3ceee4682a7c682fda848d31fcb216f748bce47e179b50ce17bcdb9.jpg)

![1f0f34a1c3718a1099cef14e2f1826bb676f2dd7648fa8ef81aa7b5f7c99276d.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/1f0f34a1c3718a1099cef14e2f1826bb676f2dd7648fa8ef81aa7b5f7c99276d.jpg)

![70add58f89f98aef889bc464e77acd0aa712be7672bbc35655875cd98e82f601.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/70add58f89f98aef889bc464e77acd0aa712be7672bbc35655875cd98e82f601.jpg)

![72ceb23668fe241a177e51299658df3424d7c5555f2ab47867009d2b46015fdf.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/72ceb23668fe241a177e51299658df3424d7c5555f2ab47867009d2b46015fdf.jpg)

![95743c8426386be5957a2e6aed3e3caaf5503f9e010af08ea96d833cde5d7c6b.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/95743c8426386be5957a2e6aed3e3caaf5503f9e010af08ea96d833cde5d7c6b.jpg)

![9db293aec898c456cd5d101f3982f920537f454e6ce622296f3f5f4a63b5f2b7.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/9db293aec898c456cd5d101f3982f920537f454e6ce622296f3f5f4a63b5f2b7.jpg)

![d3ffd7d5602533d3bb342ee5351ff3255e9d229ef9ffe558ab375f74ddfd37fe.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/d3ffd7d5602533d3bb342ee5351ff3255e9d229ef9ffe558ab375f74ddfd37fe.jpg)

![d843ad15fd7cf16327920b654ed8fb3c277ba52ad42da781bd6b11a3f687cf7f.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/d843ad15fd7cf16327920b654ed8fb3c277ba52ad42da781bd6b11a3f687cf7f.jpg)

![ed1f89d4e0d9b18b1f27db8213d09962dafb20849b329343781d1fa78293af4a.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/ed1f89d4e0d9b18b1f27db8213d09962dafb20849b329343781d1fa78293af4a.jpg)

![ed2b19bdfc0be60aee81265d47f7b48bf4d8084a2cd9c5dd14403b6c89bd1620.jpg](../icml_results/2231_Scalable%20Power%20Sampling_%20Unlocking%20Efficient%2C%20Training-Free%20Reasoning%20for%20LLMs%20via%20Distribution%20Shar/tables/ed2b19bdfc0be60aee81265d47f7b48bf4d8084a2cd9c5dd14403b6c89bd1620.jpg)

<a id="paper-2203"></a>

## HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control


### Images

![0854e8b743644d2d0da323845e225b032b3f9f4130c1454cb029b21b48620441.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/0854e8b743644d2d0da323845e225b032b3f9f4130c1454cb029b21b48620441.jpg)

![207703bbc02e9ac03a3473ddc1150ca8f4d03648267ff12c1ba390c87de381e1.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/207703bbc02e9ac03a3473ddc1150ca8f4d03648267ff12c1ba390c87de381e1.jpg)

![3a054f45c2e56a5bee7e275665b48400f26421aa132134c08dedd6df17b56244.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/3a054f45c2e56a5bee7e275665b48400f26421aa132134c08dedd6df17b56244.jpg)

![6037b6377172f0e9903be81afff8e2882f656995eb6bd6ce0c6c36b7389ac05a.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/6037b6377172f0e9903be81afff8e2882f656995eb6bd6ce0c6c36b7389ac05a.jpg)

![6c4f7fe10c79590595bd9476cecdf59bfaa4ae3377dd99bc200728708158cafe.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/6c4f7fe10c79590595bd9476cecdf59bfaa4ae3377dd99bc200728708158cafe.jpg)

![8180e34ab66bb39e6227d6dd748a3d7124b1c16e5407f247ffc7b45a04b3d4f8.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/8180e34ab66bb39e6227d6dd748a3d7124b1c16e5407f247ffc7b45a04b3d4f8.jpg)

![bca16204df40ba667991ac6a654520ebb02c1fa3a1be99ad30cb87527a313b88.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/bca16204df40ba667991ac6a654520ebb02c1fa3a1be99ad30cb87527a313b88.jpg)

![c22d9de4fbf669945966494f0630c0dfe7cad159802aa42adb2d8b6fa7baaaa9.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/c22d9de4fbf669945966494f0630c0dfe7cad159802aa42adb2d8b6fa7baaaa9.jpg)

![e3a4ca74582ae376d445871ebe4a92e7db9a60f13a035eea0a77a7dcda52dcdf.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/e3a4ca74582ae376d445871ebe4a92e7db9a60f13a035eea0a77a7dcda52dcdf.jpg)

![f34630fccf3fe5cdc622b1ccd60259f348bac8d60ac83faa37617e5ac3b163ee.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/f34630fccf3fe5cdc622b1ccd60259f348bac8d60ac83faa37617e5ac3b163ee.jpg)

![f958dad8d5c71ae7eef4e1e71ed3434c83eefb06b11d3211ea5a170aa27502a7.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/images/f958dad8d5c71ae7eef4e1e71ed3434c83eefb06b11d3211ea5a170aa27502a7.jpg)

### Tables

![1bce7ace870e2ef25f68d76085efef0dee994f5dba1ee86932d1d4d33bd12686.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/1bce7ace870e2ef25f68d76085efef0dee994f5dba1ee86932d1d4d33bd12686.jpg)

![356583a31d7a8db1b0bc9cccd6d0c0278291544a4dd3198da396f742b5e4afe4.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/356583a31d7a8db1b0bc9cccd6d0c0278291544a4dd3198da396f742b5e4afe4.jpg)

![4781c1bbd120cefd6428533f65b4ac92ddcafc125ba18212649684c969d876e8.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/4781c1bbd120cefd6428533f65b4ac92ddcafc125ba18212649684c969d876e8.jpg)

![5980d545a9ab03d7cc5f42dc62815f84ed9a56d443bf0c0d17dea7220b879531.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/5980d545a9ab03d7cc5f42dc62815f84ed9a56d443bf0c0d17dea7220b879531.jpg)

![8e0cba2c61b4a8a0dda470a387eba747b34ba35dd30d86ba3ea41395f5407c48.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/8e0cba2c61b4a8a0dda470a387eba747b34ba35dd30d86ba3ea41395f5407c48.jpg)

![8fbc1c085dc019e93dcb28d2550431b8d2c65d1e36347baaee6cba1d6ada8cdd.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/8fbc1c085dc019e93dcb28d2550431b8d2c65d1e36347baaee6cba1d6ada8cdd.jpg)

![9fb0bc310f63f9482efbefb187fdd59cd6a3289ad7e4593b7fccc6ad987397d9.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/9fb0bc310f63f9482efbefb187fdd59cd6a3289ad7e4593b7fccc6ad987397d9.jpg)

![a4d79d1e7701616f8966ad12abc43965ecf53c6f08322906cbef71aa7ae1c8bc.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/a4d79d1e7701616f8966ad12abc43965ecf53c6f08322906cbef71aa7ae1c8bc.jpg)

![a901a32dd3b62499b5cabe3a676c2c604a60ecd18268a9764b2762bb4b7247d1.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/a901a32dd3b62499b5cabe3a676c2c604a60ecd18268a9764b2762bb4b7247d1.jpg)

![b947796c2061db76b81c33721cda0cbd20ef381d0df36af1bacaa0fa5dfeedc6.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/b947796c2061db76b81c33721cda0cbd20ef381d0df36af1bacaa0fa5dfeedc6.jpg)

![f566043ea1c7d8b433956e5692f2a750cfecd7a37d00d7c83b171e4e6efe9ae6.jpg](../icml_results/2232_HiMe_%20Hierarchical%20Embodied%20Memory%20for%20Long-Horizon%20Vision-Language-Action%20Control/tables/f566043ea1c7d8b433956e5692f2a750cfecd7a37d00d7c83b171e4e6efe9ae6.jpg)

<a id="paper-2204"></a>

## Zero-Shot Text-to-Motion Evaluation using Video Language Models


### Images

![0e2e70608cddd3bc3ef4fba37d78a35c0c53e82c71a6996954e98a1a337e3f13.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/images/0e2e70608cddd3bc3ef4fba37d78a35c0c53e82c71a6996954e98a1a337e3f13.jpg)

![27728ac7cff409be8b40978573a0ecb9b8ed41048f051da35553b13ea0b161fd.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/images/27728ac7cff409be8b40978573a0ecb9b8ed41048f051da35553b13ea0b161fd.jpg)

![37e97dc2de98aa2b9f8e4fcc81672a3dc068b718155caf013c29fcfdc01d11c2.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/images/37e97dc2de98aa2b9f8e4fcc81672a3dc068b718155caf013c29fcfdc01d11c2.jpg)

![54b07b538f4c4b06482c232b862b861c9708177cef93a9b346132669ff15976f.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/images/54b07b538f4c4b06482c232b862b861c9708177cef93a9b346132669ff15976f.jpg)

![9c1f09165d896792b0aeac155b14aa14940408722861f59f0c986e7fea82fff6.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/images/9c1f09165d896792b0aeac155b14aa14940408722861f59f0c986e7fea82fff6.jpg)

### Tables

![1301290360697f8f7217ec61c6a544bb1f8bb4babaffb4a51dcf607c9929a20c.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/1301290360697f8f7217ec61c6a544bb1f8bb4babaffb4a51dcf607c9929a20c.jpg)

![2d006aa297d800431d5287a261d79bf3b9b102d2a8774c851715b80ecfda7192.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/2d006aa297d800431d5287a261d79bf3b9b102d2a8774c851715b80ecfda7192.jpg)

![300ec6af57bc2fbc91fd9e003094bfb9d10f63edc97bde00555d1e2592a685af.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/300ec6af57bc2fbc91fd9e003094bfb9d10f63edc97bde00555d1e2592a685af.jpg)

![52fd574449fcf67f7f025a8d25d5bb61fff744e1db20c2dfc6091b7ab713f41e.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/52fd574449fcf67f7f025a8d25d5bb61fff744e1db20c2dfc6091b7ab713f41e.jpg)

![5bc87b0f4ed80e818655798f07bca1ca623dd1ef7f8bf9a6b89214533250c02d.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/5bc87b0f4ed80e818655798f07bca1ca623dd1ef7f8bf9a6b89214533250c02d.jpg)

![966b07b6810326a1f857e182943c7c6dad0114627ec96272cffb611a033d6c52.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/966b07b6810326a1f857e182943c7c6dad0114627ec96272cffb611a033d6c52.jpg)

![b32a9d4ad8390372359060a8d41c9096a1365a43396a0aa194be407eb1bdd9a9.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/b32a9d4ad8390372359060a8d41c9096a1365a43396a0aa194be407eb1bdd9a9.jpg)

![babde92c529761961b0448a506ba7a9caf644b72785cf81831d588a61eeffe5d.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/babde92c529761961b0448a506ba7a9caf644b72785cf81831d588a61eeffe5d.jpg)

![bd4a287f4ab130de4223d8defd60910035d2c1f7702f2a014e8dc4b33504baf8.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/bd4a287f4ab130de4223d8defd60910035d2c1f7702f2a014e8dc4b33504baf8.jpg)

![c6b26c15a7aa061f65cf7b6bd92b75903d6fe181cb0ed4f72cffed77b53b94e3.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/c6b26c15a7aa061f65cf7b6bd92b75903d6fe181cb0ed4f72cffed77b53b94e3.jpg)

![d96016df7d42ee554d1757582706046a1744fe8ecd5680d4a1b0d113ecda405f.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/d96016df7d42ee554d1757582706046a1744fe8ecd5680d4a1b0d113ecda405f.jpg)

![ec09a37bead11c035b60a8c2f7636dc3bfdb1b34f5ec68572258907be44bfc01.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/ec09a37bead11c035b60a8c2f7636dc3bfdb1b34f5ec68572258907be44bfc01.jpg)

![f4efe9b5570ba3e049794ebc8496f12709d4108446e1d701aae5cc718c389a86.jpg](../icml_results/2233_Zero-Shot%20Text-to-Motion%20Evaluation%20using%20Video%20Language%20Models/tables/f4efe9b5570ba3e049794ebc8496f12709d4108446e1d701aae5cc718c389a86.jpg)

<a id="paper-2205"></a>

## SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling


### Images

![29957c9116001d5b68c5841dea87529e368d08276ef9798293b42eae23b2ac05.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/images/29957c9116001d5b68c5841dea87529e368d08276ef9798293b42eae23b2ac05.jpg)

![4e970fd70fa291913f7571631266cbfce7e074b8796d1c358644185e53b60be7.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/images/4e970fd70fa291913f7571631266cbfce7e074b8796d1c358644185e53b60be7.jpg)

![5361326edbcf734a5636f5ea9b7e66a0fdbdac548e99957bb6b2529137c6b80d.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/images/5361326edbcf734a5636f5ea9b7e66a0fdbdac548e99957bb6b2529137c6b80d.jpg)

![7897701dc2ff24ec1a6315e20bc384b89e5cd0f88226fa58e79ae767758bb8d2.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/images/7897701dc2ff24ec1a6315e20bc384b89e5cd0f88226fa58e79ae767758bb8d2.jpg)

![a78a996818e843d1c3a7dd5a98efeedc18010325b9ca5a0ec23f6b2050d025e9.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/images/a78a996818e843d1c3a7dd5a98efeedc18010325b9ca5a0ec23f6b2050d025e9.jpg)

### Tables

![19832611f36ea4eccd447a135f871bebae41c9604be4c97cff36378adc7b959e.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/19832611f36ea4eccd447a135f871bebae41c9604be4c97cff36378adc7b959e.jpg)

![52a08f10ed837f02467e615c9e67289105565ddfdf201b69db7d787a8b04ec52.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/52a08f10ed837f02467e615c9e67289105565ddfdf201b69db7d787a8b04ec52.jpg)

![637a1338388fd81aa4bea0840d0efb013d1dfbb484d2f00389b13cabf684143b.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/637a1338388fd81aa4bea0840d0efb013d1dfbb484d2f00389b13cabf684143b.jpg)

![69592ee8fd3f24dd114ff8b21c6cda5bdbb05d6621f3016ca812a43e0ca8d9fe.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/69592ee8fd3f24dd114ff8b21c6cda5bdbb05d6621f3016ca812a43e0ca8d9fe.jpg)

![71cb711e406729ef3008d3ee62ecdae2a7324e212562ace6a01e448076d746a4.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/71cb711e406729ef3008d3ee62ecdae2a7324e212562ace6a01e448076d746a4.jpg)

![77699eca9fbdcffc90b5b59bd194ed8d4f32061b7e51ae89728919fdb98ba1aa.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/77699eca9fbdcffc90b5b59bd194ed8d4f32061b7e51ae89728919fdb98ba1aa.jpg)

![8b3b786bad3719523aed507502e07dba64cc091cd9b45769d1b2e3ffe269668c.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/8b3b786bad3719523aed507502e07dba64cc091cd9b45769d1b2e3ffe269668c.jpg)

![91457a220f1d1aabbe875fcf87bf3b58d83c473e644be21be1707cc9df58a77f.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/91457a220f1d1aabbe875fcf87bf3b58d83c473e644be21be1707cc9df58a77f.jpg)

![d2fe7d693d075e555747ec1d94a283cfdfd390216a52b5697e4e8854f1a58be0.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/d2fe7d693d075e555747ec1d94a283cfdfd390216a52b5697e4e8854f1a58be0.jpg)

![e8ca83e30ba80a19c29b45fbf780f5614e38c2f4402ac3126b4b4114dbacd7a5.jpg](../icml_results/2234_SALE%20_%20Low-bit%20Estimation%20for%20Efficient%20Sparse%20Attention%20in%20Long-context%20LLM%20Prefilling/tables/e8ca83e30ba80a19c29b45fbf780f5614e38c2f4402ac3126b4b4114dbacd7a5.jpg)

<a id="paper-2206"></a>

## Near-Optimal Regret for KL-Regularized Multi-Armed Bandits


### Images

![12db9eb82c91d0f008eb4e577e6ef39ab2187871b52fb0ad250520098f1e101e.jpg](../icml_results/2235_Near-Optimal%20Regret%20for%20KL-Regularized%20Multi-Armed%20Bandits/images/12db9eb82c91d0f008eb4e577e6ef39ab2187871b52fb0ad250520098f1e101e.jpg)

![4cc7120590d6514998d58f9b3d2e8a3127b8d3183576fd5891b786488f6e44a8.jpg](../icml_results/2235_Near-Optimal%20Regret%20for%20KL-Regularized%20Multi-Armed%20Bandits/images/4cc7120590d6514998d58f9b3d2e8a3127b8d3183576fd5891b786488f6e44a8.jpg)

![4de908f0711dfc32681ab9b16ef79c5668b9ed12591bd0e92e6b9293b493e571.jpg](../icml_results/2235_Near-Optimal%20Regret%20for%20KL-Regularized%20Multi-Armed%20Bandits/images/4de908f0711dfc32681ab9b16ef79c5668b9ed12591bd0e92e6b9293b493e571.jpg)

![56a08bc2548ed80ece84131e5fab6acaf185d7f2d5e935903cfe2b20d5f1b2c1.jpg](../icml_results/2235_Near-Optimal%20Regret%20for%20KL-Regularized%20Multi-Armed%20Bandits/images/56a08bc2548ed80ece84131e5fab6acaf185d7f2d5e935903cfe2b20d5f1b2c1.jpg)

### Tables

![65e084cb87fbd7c8fe976105d471f3fe3e17915a65d5fbd6c99f1671c9485bfd.jpg](../icml_results/2235_Near-Optimal%20Regret%20for%20KL-Regularized%20Multi-Armed%20Bandits/tables/65e084cb87fbd7c8fe976105d471f3fe3e17915a65d5fbd6c99f1671c9485bfd.jpg)

<a id="paper-2207"></a>

## Sparse ActionGen: Accelerating Diffusion Policy with Real-time Pruning


### Images

![07b252487a77d921e197dbd5e06c91d15e3ced8a906ea3d68232c36618ab9d43.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/07b252487a77d921e197dbd5e06c91d15e3ced8a906ea3d68232c36618ab9d43.jpg)

![2bf04825bd2ff30e9dd868c241a315f2c4b9498478bc5b88e00cc7e6e83b540d.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/2bf04825bd2ff30e9dd868c241a315f2c4b9498478bc5b88e00cc7e6e83b540d.jpg)

![2e58c26affff47099a66a31142201a51c7baa08767513e68406b37be63b5d4d9.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/2e58c26affff47099a66a31142201a51c7baa08767513e68406b37be63b5d4d9.jpg)

![35c3302c5965cf046ed9006c5677f873fe82fd526706464022b0618ce00fd7fe.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/35c3302c5965cf046ed9006c5677f873fe82fd526706464022b0618ce00fd7fe.jpg)

![366e6f4df97ae13989f779a3bf0463e068ef7ef4b41f1cdd3737504946b476be.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/366e6f4df97ae13989f779a3bf0463e068ef7ef4b41f1cdd3737504946b476be.jpg)

![3735b511184203771c47499da6d8086b09b3429b68f7d85acdadcb1f7d247fe9.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/3735b511184203771c47499da6d8086b09b3429b68f7d85acdadcb1f7d247fe9.jpg)

![484c929ca7b1eef80d30096d1835a51d81a9cfafb1de1cd253244452d1a31ebe.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/484c929ca7b1eef80d30096d1835a51d81a9cfafb1de1cd253244452d1a31ebe.jpg)

![56b47995ddf559092d2ab06f3cccc4af78cc218d1c4a41837afd7a9be7500ea6.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/56b47995ddf559092d2ab06f3cccc4af78cc218d1c4a41837afd7a9be7500ea6.jpg)

![604b03846463687561e204e47b8900dc549cc3046db6a9bb0143cb10cff9df6c.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/604b03846463687561e204e47b8900dc549cc3046db6a9bb0143cb10cff9df6c.jpg)

![63d52dd943b19cef08bfb19d93cb33418aeed51cba4c35b79d718817e206f07c.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/63d52dd943b19cef08bfb19d93cb33418aeed51cba4c35b79d718817e206f07c.jpg)

![9f53099e31f03482e5c48bde08868dbf4f7230638c5038072d0d343b4db9c645.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/9f53099e31f03482e5c48bde08868dbf4f7230638c5038072d0d343b4db9c645.jpg)

![aa02c755bbd110d95f4f4c6f0fb11966246103c707f17e1604c4fc4dedfaa017.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/aa02c755bbd110d95f4f4c6f0fb11966246103c707f17e1604c4fc4dedfaa017.jpg)

![b0f49546ea33c6c78db079bbd72e2b302f7d13be4cfab126a870970fc955bb7e.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/b0f49546ea33c6c78db079bbd72e2b302f7d13be4cfab126a870970fc955bb7e.jpg)

![bccbae2e20caf68fd849fd382f2b5c5dd7e2bbabd5bd69c2141b48ec6a6769b4.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/bccbae2e20caf68fd849fd382f2b5c5dd7e2bbabd5bd69c2141b48ec6a6769b4.jpg)

![f51d2e576347eea673b48b3cefaec1f7a1933913def89cf22d7c5720fdbd0e04.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/images/f51d2e576347eea673b48b3cefaec1f7a1933913def89cf22d7c5720fdbd0e04.jpg)

### Tables

![0742b6b5d553c3f4c323362c1ffc48c1fba4baa6b3febe5d92172b75e5de3d3f.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/0742b6b5d553c3f4c323362c1ffc48c1fba4baa6b3febe5d92172b75e5de3d3f.jpg)

![0d3214c37ffaf423e962f618426eb0e70ca95db345e4d4b49fa73f6ce2bcd7bb.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/0d3214c37ffaf423e962f618426eb0e70ca95db345e4d4b49fa73f6ce2bcd7bb.jpg)

![22d463078fb800742eb9019705cc6e8df345c415fa938cfd8cd143fa3890a731.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/22d463078fb800742eb9019705cc6e8df345c415fa938cfd8cd143fa3890a731.jpg)

![24774b1a4891f6616f7aeb7d5c8a9dc8200fed11a52dc0440b5d7f47cf1a14cc.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/24774b1a4891f6616f7aeb7d5c8a9dc8200fed11a52dc0440b5d7f47cf1a14cc.jpg)

![33234d6054d3c03de0488e4fa2aae1264dc75a381fa81851d414b868a6100329.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/33234d6054d3c03de0488e4fa2aae1264dc75a381fa81851d414b868a6100329.jpg)

![37a1e324f1c742167ef0273e74f55641356ee44a0b694748ec1e3149a16ef82a.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/37a1e324f1c742167ef0273e74f55641356ee44a0b694748ec1e3149a16ef82a.jpg)

![3f1b4f928689f63fdd8bca0e0919ae4e35a2a482bed41a6f42bf206fae258f73.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/3f1b4f928689f63fdd8bca0e0919ae4e35a2a482bed41a6f42bf206fae258f73.jpg)

![733693daf99b34861eccacf442c6e2bd3ab4e2ed86952cd7b35f9a1fb7b1bc6c.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/733693daf99b34861eccacf442c6e2bd3ab4e2ed86952cd7b35f9a1fb7b1bc6c.jpg)

![8097e10b5353e81dd288bc1edfba3d999118d303ae9cb125e69b6ae4b4657b42.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/8097e10b5353e81dd288bc1edfba3d999118d303ae9cb125e69b6ae4b4657b42.jpg)

![96b7dd771e66ba526bcebf44dd8d00ea05f8ece969eb7175929a2e3d714c977c.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/96b7dd771e66ba526bcebf44dd8d00ea05f8ece969eb7175929a2e3d714c977c.jpg)

![c192465aedef3291c6667186af89fb449935fb819368b8950a2b23ae1e29092d.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/c192465aedef3291c6667186af89fb449935fb819368b8950a2b23ae1e29092d.jpg)

![d625f37b643403219e9a11e30fbddf4b03a70307144d3267115b0abd8564ac3f.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/d625f37b643403219e9a11e30fbddf4b03a70307144d3267115b0abd8564ac3f.jpg)

![e52a53cc843eb48d66f01afea6a2481d86a6ad150b68afa1b0e898762311a1b6.jpg](../icml_results/2236_Sparse%20ActionGen_%20Accelerating%20Diffusion%20Policy%20with%20Real-time%20Pruning/tables/e52a53cc843eb48d66f01afea6a2481d86a6ad150b68afa1b0e898762311a1b6.jpg)

<a id="paper-2208"></a>

## VT-Bench: A Unified Benchmark for Visual-Tabular Multi-Modal Learning


### Images

![375654e4a041476483fd1c75aad0ef7f53d55a21dabc3f9469675147ab6e9430.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/375654e4a041476483fd1c75aad0ef7f53d55a21dabc3f9469675147ab6e9430.jpg)

![4be05354bd12769b883d33013c19d324f1fa677d8f77b978bbda19394679aeb8.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/4be05354bd12769b883d33013c19d324f1fa677d8f77b978bbda19394679aeb8.jpg)

![6cf3a03b0bc2f3d28c93b63c82a240cb0a12976026e139b4163e69c1b05c6a84.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/6cf3a03b0bc2f3d28c93b63c82a240cb0a12976026e139b4163e69c1b05c6a84.jpg)

![a42e2cb77fea2abc30d296753ee53b156604135df348c8e9ff07c5c5ab7472e1.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/a42e2cb77fea2abc30d296753ee53b156604135df348c8e9ff07c5c5ab7472e1.jpg)

![a4ef950691d9c145ebb9f3f36b6fbb8363eb30c06006dc6a324822c3c3601d20.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/a4ef950691d9c145ebb9f3f36b6fbb8363eb30c06006dc6a324822c3c3601d20.jpg)

![c1b06ddcca4d1b37d8b815bbdbe0ca893568b741003f61f2d14121c64f7fbd98.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/c1b06ddcca4d1b37d8b815bbdbe0ca893568b741003f61f2d14121c64f7fbd98.jpg)

![d4d6e7776a589c9ea5acddbfdebb333f4346f4228e1cb993cc5ee03b4d3866f0.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/d4d6e7776a589c9ea5acddbfdebb333f4346f4228e1cb993cc5ee03b4d3866f0.jpg)

![fd002bc21b6842f99bf1c46e1e3cfa4573baeb0acb087a91f402a12b7af8a067.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/images/fd002bc21b6842f99bf1c46e1e3cfa4573baeb0acb087a91f402a12b7af8a067.jpg)

### Tables

![00b30a8c6af446e8372ea5b64a8e693946f081c3e37186fa7bbac4b6eb55add3.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/00b30a8c6af446e8372ea5b64a8e693946f081c3e37186fa7bbac4b6eb55add3.jpg)

![0ced27c211361f70019f7094e71789923b2627e6be4f13234101685f7b011914.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/0ced27c211361f70019f7094e71789923b2627e6be4f13234101685f7b011914.jpg)

![0d2f948d554e620f5d4fae76d32185e1531d97e9667d3660e5b8a1e566f45e25.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/0d2f948d554e620f5d4fae76d32185e1531d97e9667d3660e5b8a1e566f45e25.jpg)

![1657404aaf97e51a9c18ab942f87b58600faec7fb881108fb9a0620f90d0781f.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/1657404aaf97e51a9c18ab942f87b58600faec7fb881108fb9a0620f90d0781f.jpg)

![3765ec37044b3bba0043e90f4031cbf230c5af41e915c9aa3d10972d735f192c.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/3765ec37044b3bba0043e90f4031cbf230c5af41e915c9aa3d10972d735f192c.jpg)

![874f0a81a46f257a36c892fae1a1facb6289308378918c0a3b406fc6392e94be.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/874f0a81a46f257a36c892fae1a1facb6289308378918c0a3b406fc6392e94be.jpg)

![9fd24fdce5ab653b41ad60daace33d77d3317d64b4c4d005e73f284bee4377c1.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/9fd24fdce5ab653b41ad60daace33d77d3317d64b4c4d005e73f284bee4377c1.jpg)

![ae840fdbc444ef4bb05f3ddc96c318ccbe503b366d9b3f407355634460959cc1.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/ae840fdbc444ef4bb05f3ddc96c318ccbe503b366d9b3f407355634460959cc1.jpg)

![b48af92566f3458c9ae81b9f2be66791472675f79d4781bd60dc88897aeeb940.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/b48af92566f3458c9ae81b9f2be66791472675f79d4781bd60dc88897aeeb940.jpg)

![bd64ae9c135bd4aae44c7106e46110b0d6e987c40942c72a2bc40d86b05c28fb.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/bd64ae9c135bd4aae44c7106e46110b0d6e987c40942c72a2bc40d86b05c28fb.jpg)

![c992cabeba9502a294f6b71dc7f089fdd64f79afab50ecbe5c9023e55d659d18.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/c992cabeba9502a294f6b71dc7f089fdd64f79afab50ecbe5c9023e55d659d18.jpg)

![d6b357d036a92223395f11b97cc3a14431a04d48fe6a84ec9103a439ab98bcd2.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/d6b357d036a92223395f11b97cc3a14431a04d48fe6a84ec9103a439ab98bcd2.jpg)

![e46bd907202ac72b0f64e311b53f4e3d36a80f136964c4354c8558fc22ab60f7.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/e46bd907202ac72b0f64e311b53f4e3d36a80f136964c4354c8558fc22ab60f7.jpg)

![e99db2c372017c33124c5b4dc4646c0c151dbea5e011769982607d93b8dacc31.jpg](../icml_results/2237_VT-Bench_%20A%20Unified%20Benchmark%20for%20Visual-Tabular%20Multi-Modal%20Learning/tables/e99db2c372017c33124c5b4dc4646c0c151dbea5e011769982607d93b8dacc31.jpg)

<a id="paper-2209"></a>

## Quantitative Estimation of Target Task Performance from Unsupervised Pretext Task in Semi/Self-Supervised Learning


### Images

![080055b562dfa9d6157af24897902e122afba12d018abffdb46d29cf06b6d010.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/080055b562dfa9d6157af24897902e122afba12d018abffdb46d29cf06b6d010.jpg)

![0a4ceba4e79ea242088a9c798ddbaefa020ed1d4ca3c45eb6265164b22e9f36a.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/0a4ceba4e79ea242088a9c798ddbaefa020ed1d4ca3c45eb6265164b22e9f36a.jpg)

![1d38e7025c62e6abe6ea837dfea99a208fa124738a402eb12405b11a55a854b8.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/1d38e7025c62e6abe6ea837dfea99a208fa124738a402eb12405b11a55a854b8.jpg)

![47dc8918eb62264a2f05f49fcee2d38e559008c3e1d911120000017429c41142.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/47dc8918eb62264a2f05f49fcee2d38e559008c3e1d911120000017429c41142.jpg)

![4f4124a0c0af4d20339d2e249cc752c261b68f55b9b88154482f77069f895bb4.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/4f4124a0c0af4d20339d2e249cc752c261b68f55b9b88154482f77069f895bb4.jpg)

![57d1296d94ef59da7b610a6fa356db1a7fddd9c2e79dbde73045816e07bd6902.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/57d1296d94ef59da7b610a6fa356db1a7fddd9c2e79dbde73045816e07bd6902.jpg)

![69e0f6cc6641e064ea54fd6c2895ea9e96edb8c62bde3d8f04eb5d85a81ec1ea.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/69e0f6cc6641e064ea54fd6c2895ea9e96edb8c62bde3d8f04eb5d85a81ec1ea.jpg)

![6f80acfe95a26973652c412daae3efb3c632c8f1b089bcc4f566d9601c70595b.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/6f80acfe95a26973652c412daae3efb3c632c8f1b089bcc4f566d9601c70595b.jpg)

![72fdf9d9a38c0e934569ec1adf9b5b9d5ffc7efcf98695bf58bfe44b40b468c0.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/72fdf9d9a38c0e934569ec1adf9b5b9d5ffc7efcf98695bf58bfe44b40b468c0.jpg)

![844e5c29cb15690a1d2f9a9c0ae26d96e7d7f54499bb77a421cf34e7543853f2.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/844e5c29cb15690a1d2f9a9c0ae26d96e7d7f54499bb77a421cf34e7543853f2.jpg)

![9ea9b3fb85522a2405918ac882590e9012c64bf227144e509630a83f4647f560.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/9ea9b3fb85522a2405918ac882590e9012c64bf227144e509630a83f4647f560.jpg)

![9ff26f06c688dd60bf73b63f2c12616d1a765bc20f45834e94534faced68451d.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/9ff26f06c688dd60bf73b63f2c12616d1a765bc20f45834e94534faced68451d.jpg)

![afd5e8d65606210b0c557d7aa4f91800fadf54f5c4ed8c86542a130800940522.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/afd5e8d65606210b0c557d7aa4f91800fadf54f5c4ed8c86542a130800940522.jpg)

![b41391e1b6f482da42702cfa3d4b3cf39dcb16f2b9cfea66964151379742e31f.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/b41391e1b6f482da42702cfa3d4b3cf39dcb16f2b9cfea66964151379742e31f.jpg)

![ceb61d74808e66780f431a3b0f4272fdb7ab6d0ca038b12bc6c10eeeb89de26a.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/ceb61d74808e66780f431a3b0f4272fdb7ab6d0ca038b12bc6c10eeeb89de26a.jpg)

![d6011823e0f398284829ba81a4f0a684d51feef0ba449aefa907021d51f79aff.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/d6011823e0f398284829ba81a4f0a684d51feef0ba449aefa907021d51f79aff.jpg)

![dcc7a9198ec9765c5fe6f0bfe9ad0f6d11feb7a06013ddfe80e72e4b06c9e297.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/dcc7a9198ec9765c5fe6f0bfe9ad0f6d11feb7a06013ddfe80e72e4b06c9e297.jpg)

![df346143bf85853fc780c27ae07b969e1c96f9fc4f3fd5d3f238f696d611a930.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/df346143bf85853fc780c27ae07b969e1c96f9fc4f3fd5d3f238f696d611a930.jpg)

![ebe57bc2117ac5589b987b70dfa58c9aede0cc56194a5fa833610f7453185db4.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/ebe57bc2117ac5589b987b70dfa58c9aede0cc56194a5fa833610f7453185db4.jpg)

![faf0925c3df9da755cc46cab971acd87fee562cf117944b14aa9c189bb2c6d03.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/faf0925c3df9da755cc46cab971acd87fee562cf117944b14aa9c189bb2c6d03.jpg)

![fb3cbff51dcc429069b2a52384d7d93137e5a9fab1ce81b3228b9e400903a7e8.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/fb3cbff51dcc429069b2a52384d7d93137e5a9fab1ce81b3228b9e400903a7e8.jpg)

![ff3b734cdd7d5ee108fea033d548bc83db1d904338a191b9a268e45a05d8c361.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/images/ff3b734cdd7d5ee108fea033d548bc83db1d904338a191b9a268e45a05d8c361.jpg)

### Tables

![238cf123c3a89b3686ee29d4cc9afb1f3560c3a99ae71c5907f29e5cbcfc857a.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/tables/238cf123c3a89b3686ee29d4cc9afb1f3560c3a99ae71c5907f29e5cbcfc857a.jpg)

![d5f9d16b688e54640f1e1163e04599a87d7323b8d9062956d95d319602592d77.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/tables/d5f9d16b688e54640f1e1163e04599a87d7323b8d9062956d95d319602592d77.jpg)

![e329dd31fb2e92650090b55bdd2193b064a34f98ec8ad84f8a3121b3b28c0241.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/tables/e329dd31fb2e92650090b55bdd2193b064a34f98ec8ad84f8a3121b3b28c0241.jpg)

![ebe383806445ecc957577fd9e59a2fbc1ed9238e3515ccf814715edf8b3f6543.jpg](../icml_results/2238_Quantitative%20Estimation%20of%20Target%20Task%20Performance%20from%20Unsupervised%20Pretext%20Task%20in%20Semi_Self-Super/tables/ebe383806445ecc957577fd9e59a2fbc1ed9238e3515ccf814715edf8b3f6543.jpg)

<a id="paper-2210"></a>

## From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models


### Images

![15c7b7d211cce6478ef78d1148b5cf9954b9810233832b4efe6d70c092d9c68c.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/15c7b7d211cce6478ef78d1148b5cf9954b9810233832b4efe6d70c092d9c68c.jpg)

![3348cde7fc1d178992b6e509fb701fd2a5db2c8041fed262901c129545dd1184.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/3348cde7fc1d178992b6e509fb701fd2a5db2c8041fed262901c129545dd1184.jpg)

![4466707837a3fd51fb8b01e9833d8d64d25844b49938e81855e827985407a201.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/4466707837a3fd51fb8b01e9833d8d64d25844b49938e81855e827985407a201.jpg)

![47e8974d932943ed113b893376336f09e2012be95eac8fb5192e545ce99d9356.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/47e8974d932943ed113b893376336f09e2012be95eac8fb5192e545ce99d9356.jpg)

![51ff37c266682b274b76203a76956d3ce5899faa91e4d58d8b76dea9a1b43d3f.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/51ff37c266682b274b76203a76956d3ce5899faa91e4d58d8b76dea9a1b43d3f.jpg)

![8271b1b0da15a5bf505ca24fb150d8f774985f1787d0e11d2c72f9b58cd6300f.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/images/8271b1b0da15a5bf505ca24fb150d8f774985f1787d0e11d2c72f9b58cd6300f.jpg)

### Tables

![3aa1b560e397c134eafcd2b5e5ab6fd5c790339b75d232be828aa0b69a70e013.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/tables/3aa1b560e397c134eafcd2b5e5ab6fd5c790339b75d232be828aa0b69a70e013.jpg)

![86b7aa9d4fb14f8749bb92ddf7bb64c3a3c755fb054c30f863dab3af2acfa1ce.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/tables/86b7aa9d4fb14f8749bb92ddf7bb64c3a3c755fb054c30f863dab3af2acfa1ce.jpg)

![a5a08b1783b2d4a1fe92b3386824d610df66dff696c43def05ff4da5a8d701b4.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/tables/a5a08b1783b2d4a1fe92b3386824d610df66dff696c43def05ff4da5a8d701b4.jpg)

![da7ba857559ee4ab8b65c70ce3df80839c4672ff538a6054106cc30688033d67.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/tables/da7ba857559ee4ab8b65c70ce3df80839c4672ff538a6054106cc30688033d67.jpg)

![f5890064d8aa123aba3f4cdb55049a5cbf2fee9ca42662f218bd919d465cf362.jpg](../icml_results/2239_From%20Blind%20Spots%20to%20Gains_%20Diagnostic-Driven%20Iterative%20Training%20for%20Large%20Multimodal%20Models/tables/f5890064d8aa123aba3f4cdb55049a5cbf2fee9ca42662f218bd919d465cf362.jpg)

<a id="paper-2211"></a>

## EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models

### Images

![0a156aba80fe4e2ef2bea3a55d714feccf9c4fe00b7eedae1c48a95a276cfb7b.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/0a156aba80fe4e2ef2bea3a55d714feccf9c4fe00b7eedae1c48a95a276cfb7b.jpg)

![1e4cd4e59f7a41fec9fe993de5403a22a62a006d87054696aeaaca24621074b1.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/1e4cd4e59f7a41fec9fe993de5403a22a62a006d87054696aeaaca24621074b1.jpg)

![380bf9301ae06e20d5f4acc4352ea3e291b30d905ee8785b4760b3f9fad6cf8a.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/380bf9301ae06e20d5f4acc4352ea3e291b30d905ee8785b4760b3f9fad6cf8a.jpg)

![777abd7bb66316c075e5f03a59f32f45cefd404a3ba998998595678460279689.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/777abd7bb66316c075e5f03a59f32f45cefd404a3ba998998595678460279689.jpg)

![a61499b30a72c0ed308d1e253025b1a3c82da9236b0a7f299d7a507c4148dad4.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/a61499b30a72c0ed308d1e253025b1a3c82da9236b0a7f299d7a507c4148dad4.jpg)

![b2484667abb3f92e2dbd05f38d98d10c49a27a200933225f92ec328695002f06.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/b2484667abb3f92e2dbd05f38d98d10c49a27a200933225f92ec328695002f06.jpg)

![f47572926d548f24c1d7ae89d0ce88a7f2b52547252a1760c8205dc3bcb50df3.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/f47572926d548f24c1d7ae89d0ce88a7f2b52547252a1760c8205dc3bcb50df3.jpg)

![f4ab8bb5bca86bda498b38ef274f8fd985f19970f6d409cbd8386f8d9528b305.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/images/f4ab8bb5bca86bda498b38ef274f8fd985f19970f6d409cbd8386f8d9528b305.jpg)

### Tables

![00b729876877dad574f3098077e59b780a39a6a3f5cf529c69f99c0554fba921.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/00b729876877dad574f3098077e59b780a39a6a3f5cf529c69f99c0554fba921.jpg)

![0263ba10c5e3cce57728d4aebe6e51789f3c985a1e70140592087863891d6d70.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/0263ba10c5e3cce57728d4aebe6e51789f3c985a1e70140592087863891d6d70.jpg)

![10169c955325fd698087c2fc3632ffbf43882fe4fb94b80efffc53a02b96d51e.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/10169c955325fd698087c2fc3632ffbf43882fe4fb94b80efffc53a02b96d51e.jpg)

![14cce3938be81f7c1448988e6b85c913b0807eb214927c81236bb69cc69ab84d.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/14cce3938be81f7c1448988e6b85c913b0807eb214927c81236bb69cc69ab84d.jpg)

![1a2d8d9bc7eac19e95f42a0110d6f1bef575121031a9d89b29581ea02fe63a2d.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/1a2d8d9bc7eac19e95f42a0110d6f1bef575121031a9d89b29581ea02fe63a2d.jpg)

![32d8700326a6fd323cc0e72c3dff03ded97e30a50550439fde537dbd991d1615.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/32d8700326a6fd323cc0e72c3dff03ded97e30a50550439fde537dbd991d1615.jpg)

![39980fad915ca3fd603e2e63b4aa18a1d218e5077c2bbd426f89875312b1665a.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/39980fad915ca3fd603e2e63b4aa18a1d218e5077c2bbd426f89875312b1665a.jpg)

![491954093f1dcb7c3a36278f30cd4a1c3a13ce9e9d46573f0da1bc6161086333.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/491954093f1dcb7c3a36278f30cd4a1c3a13ce9e9d46573f0da1bc6161086333.jpg)

![4945608f61e09cfd1c29825a5dc17ea5a9ac0f8e96b0002c6ae47c7b7ebbba29.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/4945608f61e09cfd1c29825a5dc17ea5a9ac0f8e96b0002c6ae47c7b7ebbba29.jpg)

![512d8bd0846d2e77dbcbd54396d50697a393a288f553c99f8683c6ab6dd96072.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/512d8bd0846d2e77dbcbd54396d50697a393a288f553c99f8683c6ab6dd96072.jpg)

![5c86e3e8de2862455786de61b22c287153af9d91d951fb7acfe91e6cd4b7ecc8.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/5c86e3e8de2862455786de61b22c287153af9d91d951fb7acfe91e6cd4b7ecc8.jpg)

![7256c5b04efcc741ec18730ab55c0505ba493f1eeae08069e6cfe090f6cf2b5c.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/7256c5b04efcc741ec18730ab55c0505ba493f1eeae08069e6cfe090f6cf2b5c.jpg)

![7afd42a1436c51a8d3a717928bf0bc696ca6063ab489974b44c8c683b8434f67.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/7afd42a1436c51a8d3a717928bf0bc696ca6063ab489974b44c8c683b8434f67.jpg)

![8392028b365438b23236eab87b3a28d712a726e4bb4aed8ac8e033f103e9f4c8.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/8392028b365438b23236eab87b3a28d712a726e4bb4aed8ac8e033f103e9f4c8.jpg)

![8a895461c0416ef26e1b736119b7ae79c628dbd31494cb88cd9e29e2c6519388.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/8a895461c0416ef26e1b736119b7ae79c628dbd31494cb88cd9e29e2c6519388.jpg)

![9a3e086bfe7719f2623ad168988d3a07a1f484a7932a86d109bf9f50c4fe2a4c.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/9a3e086bfe7719f2623ad168988d3a07a1f484a7932a86d109bf9f50c4fe2a4c.jpg)

![ac924760b4af4313982663c0b416b970365779f15616f18bef2b3234cad4e3b3.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/ac924760b4af4313982663c0b416b970365779f15616f18bef2b3234cad4e3b3.jpg)

![c0f87a44239b1f5f0ccf85f189aba7e4bc240ec4a69085baf15b8dc2af071d06.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c0f87a44239b1f5f0ccf85f189aba7e4bc240ec4a69085baf15b8dc2af071d06.jpg)

![c217160f53f9ac6c491563588ea923250afa956d77aeed271ae33afcdc7a914b.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c217160f53f9ac6c491563588ea923250afa956d77aeed271ae33afcdc7a914b.jpg)

![c2b5baf107b2abe3536829f54d5d6cd3e336f0c9df2709aa35c78415f502059d.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c2b5baf107b2abe3536829f54d5d6cd3e336f0c9df2709aa35c78415f502059d.jpg)

![c40793aacf8d17312ba8641cf7dce53668d250a96a316d491811e371d042d8fd.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c40793aacf8d17312ba8641cf7dce53668d250a96a316d491811e371d042d8fd.jpg)

![c45bd7cf2123d1ab703f92c8aab5e944c74ffa9bcb5f6bd64888dc0ef6181c8b.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c45bd7cf2123d1ab703f92c8aab5e944c74ffa9bcb5f6bd64888dc0ef6181c8b.jpg)

![c89cc4a524b75572bf216814e26f81be7af35211578b6c18eb36db251b2e62c1.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/c89cc4a524b75572bf216814e26f81be7af35211578b6c18eb36db251b2e62c1.jpg)

![d75fe7babf5d0cbbf3f5ed6e0ef5519bde8b3f8b43cd9b51ed421ce400103276.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/d75fe7babf5d0cbbf3f5ed6e0ef5519bde8b3f8b43cd9b51ed421ce400103276.jpg)

![dd56575b3e8ed8dba2897ac0a630d4a087699b2ff5f2d2ec531d4248b28254ad.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/dd56575b3e8ed8dba2897ac0a630d4a087699b2ff5f2d2ec531d4248b28254ad.jpg)

![de978dd990cb35660183b4e98606b28b694d74943eb42f8c780049c9f0df3c64.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/de978dd990cb35660183b4e98606b28b694d74943eb42f8c780049c9f0df3c64.jpg)

![e17685c3a4c682f1dfaccbfacf6c04472f1be8ea0df561955c81ce8f66881ef0.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/e17685c3a4c682f1dfaccbfacf6c04472f1be8ea0df561955c81ce8f66881ef0.jpg)

![f0884203f656e97112c281c15c7617d2e98b533691671dfb8394e8813fb41dd9.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/f0884203f656e97112c281c15c7617d2e98b533691671dfb8394e8813fb41dd9.jpg)

![f9ad395a26ddade10d0784c51d2f568d1164b18c9e458b8a08e5cdb40f51d8ba.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/f9ad395a26ddade10d0784c51d2f568d1164b18c9e458b8a08e5cdb40f51d8ba.jpg)

![fa78aa0cb7352c76766127a78e98271b6f34f08c8cd05e8d546916a44fd72e53.jpg](../icml_results/2240_EmBrace_%20A%20Collective%20Knowledge%20Fusion%20Framework%20Toward%20Unified%20EEG%20Foundation%20Models/tables/fa78aa0cb7352c76766127a78e98271b6f34f08c8cd05e8d546916a44fd72e53.jpg)
