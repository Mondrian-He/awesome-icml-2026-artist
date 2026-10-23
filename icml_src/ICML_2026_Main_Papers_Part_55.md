# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 55 of 197

## 目录 (Table of Contents)

1. [BiRQA: Bidirectional Robust Quality Assessment for Images](#paper-1783)
2. [Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](#paper-1784)
3. [Fourier Features Let Agents Learn High Precision Policies with Imitation Learning](#paper-1785)
4. [Front-Loaded Robust Conformal Prediction: Heavy Calibration, Minimal Test-Time Cost](#paper-1786)
5. [Beyond Correctness: Distance-Based Social Dynamics of Multi-Agent Debate](#paper-1787)
6. [Unlearning’s Blind Spots: Over-Unlearning and Prototypical Relearning Attack](#paper-1788)
7. [Geodesic Flow Matching for Denoising High-Dimensional Structured Representations](#paper-1789)
8. [GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data](#paper-1790)
9. [Forward-KL Convergence of Time-Inhomogeneous Langevin Diffusions](#paper-1791)
10. [Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space](#paper-1792)
11. [Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](#paper-1793)
12. [Architecture Matters for Multi-Agent Security](#paper-1794)
13. [Formalizing and Falsifying Causal Pathways of Rare Events](#paper-1795)
14. [Pluralistic Leaderboards](#paper-1796)
15. [SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model](#paper-1797)
16. [Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases](#paper-1798)
17. [Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](#paper-1799)
18. [Lower Bounds for Frank-Wolfe on Strongly Convex Sets](#paper-1800)
19. [Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling](#paper-1801)
20. [Maximin Relative Improvement: Fair Learning as a Bargaining Problem](#paper-1802)
21. [Weight Decay Improves Language Model Plasticity](#paper-1803)
22. [Bridging Local–Global Dissonance: Learning from Compressive Measurements for Hyperspectral Reconstruction](#paper-1804)
23. [TSFAdv: Frequency-Guided Black-Box Adversarial Attacks on Time Series Forecasting](#paper-1805)
24. [Revealing Long-context Potential of Attention Heads via Frequency Kernels](#paper-1806)
25. [Learning Permutation-invariant Macroscopic Dynamics](#paper-1807)
26. [Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials](#paper-1808)
27. [EGG: An Expert-Guided Agent Framework for Kernel Generation](#paper-1809)
28. [Transferable Reinforcement Learning via Probabilistic Latent Embeddings and Dynamic Policy Adaptation for Sim-to-Real Deployment](#paper-1810)
29. [Expert-level Leaf Cell Layout Generation via Preference-Optimized LLM](#paper-1811)
30. [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](#paper-1812)
31. [Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them](#paper-1813)
32. [DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation](#paper-1814)
33. [The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs](#paper-1815)

---


<a id="paper-1783"></a>

## BiRQA: Bidirectional Robust Quality Assessment for Images

### Images

![0f645fa1cc7111cd7b508fbe167f36f6dd3a35c95fc7d46e73f37a1cc63f3797.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/0f645fa1cc7111cd7b508fbe167f36f6dd3a35c95fc7d46e73f37a1cc63f3797.jpg)

![19f923a31972624f0d61bed5abed6629f0a1452d69b9171edbc901c3dc5a8bee.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/19f923a31972624f0d61bed5abed6629f0a1452d69b9171edbc901c3dc5a8bee.jpg)

![219f4f84e3819209f13195e69e8fc9610ce0b3b5d09d4f4822f09c82b1137dc5.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/219f4f84e3819209f13195e69e8fc9610ce0b3b5d09d4f4822f09c82b1137dc5.jpg)

![2fbd0c8389eb8efe62a5b501773c2a1e0f0ba9a8e6adb46b275f15cb4b50e42d.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/2fbd0c8389eb8efe62a5b501773c2a1e0f0ba9a8e6adb46b275f15cb4b50e42d.jpg)

![3589f69af8844aaf1fb6372c899bfe2587c4996da241fcb18ed8a719e6775f90.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/3589f69af8844aaf1fb6372c899bfe2587c4996da241fcb18ed8a719e6775f90.jpg)

![66280a684f3f972ddaab7e5c8323d8d2b098a7206b90326e597539624ff9f91f.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/66280a684f3f972ddaab7e5c8323d8d2b098a7206b90326e597539624ff9f91f.jpg)

![90f4c853dceafbb13bd9f2d613ceab4996a10323b2fd810017497622763f6b6b.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/90f4c853dceafbb13bd9f2d613ceab4996a10323b2fd810017497622763f6b6b.jpg)

![99576da1e8f6a9b69d4e5443841495754d00cd21ad74f4461039e529cc87fd87.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/99576da1e8f6a9b69d4e5443841495754d00cd21ad74f4461039e529cc87fd87.jpg)

![99f7926915291309c7ee4133b47f5d8e12e4774d1e982d8ecd039b0673dd171b.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/99f7926915291309c7ee4133b47f5d8e12e4774d1e982d8ecd039b0673dd171b.jpg)

![a5ec07bed0d3ff9711122671f17a481125869f7614a1d0113b6a22303b7a0024.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/a5ec07bed0d3ff9711122671f17a481125869f7614a1d0113b6a22303b7a0024.jpg)

![a9da1c41c930d21f673930a5cbfc4f1b938aaaee6fb2dc015d9581dcdfccd536.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/a9da1c41c930d21f673930a5cbfc4f1b938aaaee6fb2dc015d9581dcdfccd536.jpg)

![b22ad2cddfd33d715acaa421d0ab1d02595b0b1299bc86e056046912ce37c233.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/b22ad2cddfd33d715acaa421d0ab1d02595b0b1299bc86e056046912ce37c233.jpg)

![b471bb5c09256bcc895a1088770134486af591430b0894e258301e5965bd8b82.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/b471bb5c09256bcc895a1088770134486af591430b0894e258301e5965bd8b82.jpg)

![c16b75d42cb4b0f4cacfed29da0b70f8c1607840c737b950fa86278853907f10.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/c16b75d42cb4b0f4cacfed29da0b70f8c1607840c737b950fa86278853907f10.jpg)

![c453aca134495d09e4b176d3defd1554b4a17242f4556e708c1ccd18f1e15155.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/c453aca134495d09e4b176d3defd1554b4a17242f4556e708c1ccd18f1e15155.jpg)

![ca59eeae1c4161b474fa46374306a22d69f4eaf9129ad81967b96f8c40ab4e6d.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/ca59eeae1c4161b474fa46374306a22d69f4eaf9129ad81967b96f8c40ab4e6d.jpg)

![dacf0c490a72ee3f753ce1bb27f7198b5ae292715c7356bb889b34f362f3661a.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/dacf0c490a72ee3f753ce1bb27f7198b5ae292715c7356bb889b34f362f3661a.jpg)

![dfe83935ec83804e429540c90f77955c492cf35b96142ed9109c4427fd95d71f.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/dfe83935ec83804e429540c90f77955c492cf35b96142ed9109c4427fd95d71f.jpg)

![e72736eaeda0eb40aa903ef20cc796ce97bb23e1f3eafbec31be6ed0ae612ef9.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/e72736eaeda0eb40aa903ef20cc796ce97bb23e1f3eafbec31be6ed0ae612ef9.jpg)

![f0bc6fa228c631511d4c94e472fbe1a0ff0cecfa520f1db1ac37965cb1a08c3d.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/f0bc6fa228c631511d4c94e472fbe1a0ff0cecfa520f1db1ac37965cb1a08c3d.jpg)

![f8e86926e5a1612cf9ec77ec7762e11fabf5f03554a9ae44c5158af4dd14873d.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/images/f8e86926e5a1612cf9ec77ec7762e11fabf5f03554a9ae44c5158af4dd14873d.jpg)

### Tables

![2fad81ae88b1ba9a097b15eacd444cdf83af6c7cdefb966e5864ffb7de059037.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/tables/2fad81ae88b1ba9a097b15eacd444cdf83af6c7cdefb966e5864ffb7de059037.jpg)

![4c3deea87f2098e05aa031499d08c2f1d2ee537140dc3dec639c3037843612ab.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/tables/4c3deea87f2098e05aa031499d08c2f1d2ee537140dc3dec639c3037843612ab.jpg)

![7e1072040d6db0b89e53e64cd609ecd6af614548a3ddae176cbf5739766649e4.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/tables/7e1072040d6db0b89e53e64cd609ecd6af614548a3ddae176cbf5739766649e4.jpg)

![e0f9b97d1dd14cc8731039bd10b0333ff1d7d67d000314be1f8ff578fc66a9e9.jpg](../icml_results/1805_FLIPS_%20Instance-Fingerprinting%20for%20LLMs%20via%20Pseudo-random%20Sequences/tables/e0f9b97d1dd14cc8731039bd10b0333ff1d7d67d000314be1f8ff578fc66a9e9.jpg)

## BiRQA: Bidirectional Robust Quality Assessment for Images


### Images

![00c10ecfb25e4d8a031ece68824b8ecbe4e5d78acc1d0861cbc00211a0308dee.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/00c10ecfb25e4d8a031ece68824b8ecbe4e5d78acc1d0861cbc00211a0308dee.jpg)

![6c85149f513eeb570df955adb2500813e7e6525842c432234d82637d55e9d48a.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/6c85149f513eeb570df955adb2500813e7e6525842c432234d82637d55e9d48a.jpg)

![9a3e34e7ce2558d4c07e1354b0599319bed0aee815add05fc152f4ae2e792932.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/9a3e34e7ce2558d4c07e1354b0599319bed0aee815add05fc152f4ae2e792932.jpg)

![be00a15795c4e086962481cf5350111de3a29587a815c742cc6efbe85a92844f.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/be00a15795c4e086962481cf5350111de3a29587a815c742cc6efbe85a92844f.jpg)

![c5297471a114fd58dc1fcdb566813758bec5473d498efa4fc0259f85e550b32d.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/c5297471a114fd58dc1fcdb566813758bec5473d498efa4fc0259f85e550b32d.jpg)

![dac804645a6b423a1c5b403ea02193b861c8d3c60f4fa4f64ecc56a5a8331e02.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/images/dac804645a6b423a1c5b403ea02193b861c8d3c60f4fa4f64ecc56a5a8331e02.jpg)

### Tables

![0696881e729071c945814bdde4ef70a85a216f2251fb78d89264f1db2d6994f9.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/0696881e729071c945814bdde4ef70a85a216f2251fb78d89264f1db2d6994f9.jpg)

![189cc368e1c5c057b8744b6195b138aad6e83bc94e83e381ae2688048b35ff83.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/189cc368e1c5c057b8744b6195b138aad6e83bc94e83e381ae2688048b35ff83.jpg)

![2f129812e1e6fa6eb6d8345ce25660dfff1d869cc490d76d0f04c0518bdd8e08.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/2f129812e1e6fa6eb6d8345ce25660dfff1d869cc490d76d0f04c0518bdd8e08.jpg)

![718ee6a867bb45b067f10d466d2acb4ee301bcba0addbaffb6cbb2e8b98e72a5.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/718ee6a867bb45b067f10d466d2acb4ee301bcba0addbaffb6cbb2e8b98e72a5.jpg)

![7ac8cb49cbdbd28b61a72694a034b6f7755a955604feaf5b69987fa0fac41fd6.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/7ac8cb49cbdbd28b61a72694a034b6f7755a955604feaf5b69987fa0fac41fd6.jpg)

![7cec3cdfe7e26f433ebc575a887a202d61553dba9d4cb69db1869b3f0d15c411.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/7cec3cdfe7e26f433ebc575a887a202d61553dba9d4cb69db1869b3f0d15c411.jpg)

![81bcd80d7e6d6db7319ec7bf7748f2ef1f6670c556eb126f70c4f839983acfef.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/81bcd80d7e6d6db7319ec7bf7748f2ef1f6670c556eb126f70c4f839983acfef.jpg)

![8e428790a0343bd5658a24748265f72eac5c4e7ca03b38816137b6469eae6da9.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/8e428790a0343bd5658a24748265f72eac5c4e7ca03b38816137b6469eae6da9.jpg)

![ae67a2acc6efcdc452bfc58565837f87f5d87fc7a44dc78666bd23be55ba20b4.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/ae67a2acc6efcdc452bfc58565837f87f5d87fc7a44dc78666bd23be55ba20b4.jpg)

![b941e7be6d7a1d0aa87496e1a7fcbc3ffae3dd1e04c4b03c83e801fe0a935619.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/b941e7be6d7a1d0aa87496e1a7fcbc3ffae3dd1e04c4b03c83e801fe0a935619.jpg)

![dafdd691efe05a9b08745e60e897fcde640797768c661b68fb4271902e51565d.jpg](../icml_results/1806_BiRQA_%20Bidirectional%20Robust%20Quality%20Assessment%20for%20Images/tables/dafdd691efe05a9b08745e60e897fcde640797768c661b68fb4271902e51565d.jpg)

<a id="paper-1784"></a>

## Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design


### Images

![16ec35521d7e671143802a40ce911d6cf1f2cd5cfb1d3786f053e67cc13b8672.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/16ec35521d7e671143802a40ce911d6cf1f2cd5cfb1d3786f053e67cc13b8672.jpg)

![1e91b99e04289c65a107606175201f9f2e2fe825974918ded881d80c5b008d6e.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/1e91b99e04289c65a107606175201f9f2e2fe825974918ded881d80c5b008d6e.jpg)

![291fad030ac48a84b281758d7dcebf5fc257b8f6ccebcc9f3b643dcd5c1d78cd.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/291fad030ac48a84b281758d7dcebf5fc257b8f6ccebcc9f3b643dcd5c1d78cd.jpg)

![37313a1ffd5eeb97a4af3bcda49f846165f3df8b38b78dc5058a70e89ddc4d90.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/37313a1ffd5eeb97a4af3bcda49f846165f3df8b38b78dc5058a70e89ddc4d90.jpg)

![4dfca12031bbe2b8447adb9a55346af56a6f12d80a510f20e52c588e7fdf430a.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/4dfca12031bbe2b8447adb9a55346af56a6f12d80a510f20e52c588e7fdf430a.jpg)

![6f815db6179ce60926e0863b6f97177f278c31aba12c60a407e59a041dc4a2f9.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/6f815db6179ce60926e0863b6f97177f278c31aba12c60a407e59a041dc4a2f9.jpg)

![7c53eba44e6dc2d996bd99645a76f1f8db63bccad5f3453713836a6e15ab738a.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/7c53eba44e6dc2d996bd99645a76f1f8db63bccad5f3453713836a6e15ab738a.jpg)

![7e54d0944fdc34cb5c7350f6238692a11279c9254b6719a654e070bf5e397adf.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/7e54d0944fdc34cb5c7350f6238692a11279c9254b6719a654e070bf5e397adf.jpg)

![88b18f505e905a31be60a6e325150858d9ef227b8dd4f87df9c5708e9967d771.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/88b18f505e905a31be60a6e325150858d9ef227b8dd4f87df9c5708e9967d771.jpg)

![af8315c97595b5bea04320a35319f15b3846238ab3b32111bbec5d48eeb96555.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/af8315c97595b5bea04320a35319f15b3846238ab3b32111bbec5d48eeb96555.jpg)

![b254e5e47d7d31ff7966a1620b54df277e288c4c5494ecf12245dc1e12e3b1ea.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/b254e5e47d7d31ff7966a1620b54df277e288c4c5494ecf12245dc1e12e3b1ea.jpg)

![e0beca643d1b2f9c122247ad81937751379457cbc0675150dcba9a8a0f7ffb18.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/e0beca643d1b2f9c122247ad81937751379457cbc0675150dcba9a8a0f7ffb18.jpg)

![e15e30315d80efc0ad666dbb8e72ee9b52ac2b4376bbf898a9b5d4cbb861181c.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/images/e15e30315d80efc0ad666dbb8e72ee9b52ac2b4376bbf898a9b5d4cbb861181c.jpg)

### Tables

![0f79f30b3a280e02d68c7e7fcd4ebc74b2f89d4984c67846617e5bba08b7fdbf.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/0f79f30b3a280e02d68c7e7fcd4ebc74b2f89d4984c67846617e5bba08b7fdbf.jpg)

![16fbf95ac4c6dd642be1453a4f8d46cbafecdee56077ef157f176908e91e5450.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/16fbf95ac4c6dd642be1453a4f8d46cbafecdee56077ef157f176908e91e5450.jpg)

![1badcfa8a6dfb4b0fa5762020ec504b7845bb402a19146a80c78cab7e15fd1a0.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/1badcfa8a6dfb4b0fa5762020ec504b7845bb402a19146a80c78cab7e15fd1a0.jpg)

![358645b4d4ffe5898369082ce4b5b2517ef11c3ead9b9bc4c0c4e0d8794ec8d2.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/358645b4d4ffe5898369082ce4b5b2517ef11c3ead9b9bc4c0c4e0d8794ec8d2.jpg)

![5b79d0d36e23a5f6fb0ed5ae33d7b5281bf5dd09718f3c8c94495fd0661d2b8f.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/5b79d0d36e23a5f6fb0ed5ae33d7b5281bf5dd09718f3c8c94495fd0661d2b8f.jpg)

![649abc55e993857432bf479246a51c3e3b7d1d57bbdb587266942a1b5a237b5e.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/649abc55e993857432bf479246a51c3e3b7d1d57bbdb587266942a1b5a237b5e.jpg)

![9df892f1af886a7488b90a448f6faa7a296e12de202b17c030a7ad39382ee2b0.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/9df892f1af886a7488b90a448f6faa7a296e12de202b17c030a7ad39382ee2b0.jpg)

![d7517a4bd7f2c9c846cc6867f93d2c5f16d21fb64803e50c50dad870c92da15e.jpg](../icml_results/1807_Constrained%20Flow%20Optimization%20via%20Sequential%20Fine-Tuning%20for%20Molecular%20Design/tables/d7517a4bd7f2c9c846cc6867f93d2c5f16d21fb64803e50c50dad870c92da15e.jpg)

<a id="paper-1785"></a>

## Fourier Features Let Agents Learn High Precision Policies with Imitation Learning


### Images

![0a36f31c9a9105829ca52b66cacd9c7a5a90a59c16d0dc206387cf85dcdb2ac5.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/0a36f31c9a9105829ca52b66cacd9c7a5a90a59c16d0dc206387cf85dcdb2ac5.jpg)

![135ad3032b6bd94b157e66cd5f2ac33466163438516aaec4adcd35e838c38726.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/135ad3032b6bd94b157e66cd5f2ac33466163438516aaec4adcd35e838c38726.jpg)

![271be801f83d855e74a44ee55fb0aa81b6e2cf3d45902056cf008aa57fb3e630.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/271be801f83d855e74a44ee55fb0aa81b6e2cf3d45902056cf008aa57fb3e630.jpg)

![4d4d3930ac020ca1cd791229c1101364f29d1c1d825feac2be940c6d6bb5620c.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/4d4d3930ac020ca1cd791229c1101364f29d1c1d825feac2be940c6d6bb5620c.jpg)

![87985aa1b50120591fa93d0f1c4966992b7d2f6850b319a36a866762eaf58617.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/87985aa1b50120591fa93d0f1c4966992b7d2f6850b319a36a866762eaf58617.jpg)

![989c3d970c17c53cf9be1fd69a63f785849350cb46477383e364f05f19799dd7.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/989c3d970c17c53cf9be1fd69a63f785849350cb46477383e364f05f19799dd7.jpg)

![a750d712e9330462bc079f64a18fbe389d87a7a9ea338a0e25cd5878ad246b0e.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/a750d712e9330462bc079f64a18fbe389d87a7a9ea338a0e25cd5878ad246b0e.jpg)

![b21270a1ef8f20b3f3fcfb9730858e4aa8592ee5432a96c914760f5b03384774.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/b21270a1ef8f20b3f3fcfb9730858e4aa8592ee5432a96c914760f5b03384774.jpg)

![d1fe3aece6b9e93b20cbba17a88d2a2660d0e1b488906fb50156267c4c4cfac5.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/d1fe3aece6b9e93b20cbba17a88d2a2660d0e1b488906fb50156267c4c4cfac5.jpg)

![d7f67a2eea180b380505fc07472db112a53538b5b01308576fcdde29863b29c0.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/d7f67a2eea180b380505fc07472db112a53538b5b01308576fcdde29863b29c0.jpg)

![e4a7e2386af342b74d350a35fb96c9d312aab9e8fee26f8c68bd3e3e9f4c3be4.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/e4a7e2386af342b74d350a35fb96c9d312aab9e8fee26f8c68bd3e3e9f4c3be4.jpg)

![ed7e25132d5263716c79821b94543448f23d2ca9e513243d02e3f2818ac31aad.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/ed7e25132d5263716c79821b94543448f23d2ca9e513243d02e3f2818ac31aad.jpg)

![f31da120f52d4cb0ddb4af58f0dd61931aa2d20fdb82013f4a6ca6c5f676315c.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/images/f31da120f52d4cb0ddb4af58f0dd61931aa2d20fdb82013f4a6ca6c5f676315c.jpg)

### Tables

![266111e5499089b527ab0b82de844e28c41829876c09545f4ad64f0e36fcb95a.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/266111e5499089b527ab0b82de844e28c41829876c09545f4ad64f0e36fcb95a.jpg)

![2b2c60d2231a4239ea4a9bf309d945adceb8a98b5193a755ab5269cc0964edd3.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/2b2c60d2231a4239ea4a9bf309d945adceb8a98b5193a755ab5269cc0964edd3.jpg)

![36527a93a05be9b8900089dccc030b31e1dd18f209327109ce209d2854cd17a5.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/36527a93a05be9b8900089dccc030b31e1dd18f209327109ce209d2854cd17a5.jpg)

![3a4f0c56fef9fc144031734bec528ac4e5fa10725e3739a91ac994f01c271ca0.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/3a4f0c56fef9fc144031734bec528ac4e5fa10725e3739a91ac994f01c271ca0.jpg)

![49886076991d6300e905090129df5d5ba6fda4aba8ade5a38287cec4d6fb5aa2.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/49886076991d6300e905090129df5d5ba6fda4aba8ade5a38287cec4d6fb5aa2.jpg)

![5413542fd8ea13ce24b31f848d84b5a915613cebd76ff330cfb263a06a886536.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/5413542fd8ea13ce24b31f848d84b5a915613cebd76ff330cfb263a06a886536.jpg)

![5dc0c3e1e6b5740d09722b6c8271be0a7907253b1fa4a61b586fe6ca1cae2d58.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/5dc0c3e1e6b5740d09722b6c8271be0a7907253b1fa4a61b586fe6ca1cae2d58.jpg)

![5f65872172d475b68f510d8ccb129264621a4e566fad64204be6fd3fb69e79b2.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/5f65872172d475b68f510d8ccb129264621a4e566fad64204be6fd3fb69e79b2.jpg)

![bba5b58a345c921842d85ec6112d2e22ceb2cb6ccec782aeb6966b057cdd239c.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/bba5b58a345c921842d85ec6112d2e22ceb2cb6ccec782aeb6966b057cdd239c.jpg)

![c912e2b604abf077a25e3d17cd8f1873063da3a3e3120313374875aa4731885b.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/c912e2b604abf077a25e3d17cd8f1873063da3a3e3120313374875aa4731885b.jpg)

![e59fe50363e56c4dd71ea42896bdb8572640ebc796453fd412b78aac980180d9.jpg](../icml_results/1808_Fourier%20Features%20Let%20Agents%20Learn%20High%20Precision%20Policies%20with%20Imitation%20Learning/tables/e59fe50363e56c4dd71ea42896bdb8572640ebc796453fd412b78aac980180d9.jpg)

<a id="paper-1786"></a>

## Front-Loaded Robust Conformal Prediction: Heavy Calibration, Minimal Test-Time Cost


### Images

![1e41c2b4950d7bd20ae6473333d11a08a11356853b5eb21da07c743baaeedfd7.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/1e41c2b4950d7bd20ae6473333d11a08a11356853b5eb21da07c743baaeedfd7.jpg)

![43d70df897f701473f842aeae088aa52026c41d2f6e9fe0c25c41f55ebf79233.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/43d70df897f701473f842aeae088aa52026c41d2f6e9fe0c25c41f55ebf79233.jpg)

![4f5a52f8ac4739fae52c44ad457fcd868745127a4c7a61e6b26ae10d75a9d116.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/4f5a52f8ac4739fae52c44ad457fcd868745127a4c7a61e6b26ae10d75a9d116.jpg)

![649308c4c8539a8a035b4fb4a13cd47cceb132732086ea343c53e112ea6c5bfd.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/649308c4c8539a8a035b4fb4a13cd47cceb132732086ea343c53e112ea6c5bfd.jpg)

![6de712682da473e4aec345b9177834906022bbec6c46c18bc105a808e33890b8.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/6de712682da473e4aec345b9177834906022bbec6c46c18bc105a808e33890b8.jpg)

![770100e09f03df62a46106a304b99bebd51a084567b7e658d073c9b42a9f6d09.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/770100e09f03df62a46106a304b99bebd51a084567b7e658d073c9b42a9f6d09.jpg)

![ae92b873760e5c2d52d5e56aaaa583fab64fc8cfc467fdcc039e12f18104af27.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/ae92b873760e5c2d52d5e56aaaa583fab64fc8cfc467fdcc039e12f18104af27.jpg)

![eb69352d05979f27f1865bc125c7235c047b3844becacd2f146322b4e9efabec.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/images/eb69352d05979f27f1865bc125c7235c047b3844becacd2f146322b4e9efabec.jpg)

### Tables

![1ca369c31947462131c6aa4de1d10fff50d3f7105f07346daac3d5e546f517c0.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/1ca369c31947462131c6aa4de1d10fff50d3f7105f07346daac3d5e546f517c0.jpg)

![3424d0b38f50c93b35e0b26b7c6b4e25f043523defdac3fc2ba913107876567c.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/3424d0b38f50c93b35e0b26b7c6b4e25f043523defdac3fc2ba913107876567c.jpg)

![382feff75b333703bb287a2d92c80d158b97c6eca5dbe93f7552743dfbf6407c.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/382feff75b333703bb287a2d92c80d158b97c6eca5dbe93f7552743dfbf6407c.jpg)

![49f2ee326cf24257eba9c65178e3e3b8c016544515664575fb415abd5f2a88ea.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/49f2ee326cf24257eba9c65178e3e3b8c016544515664575fb415abd5f2a88ea.jpg)

![68e461c8e48b14160a04feefde20ef5d8bf0f87c7db9926891d4b708bdbf5015.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/68e461c8e48b14160a04feefde20ef5d8bf0f87c7db9926891d4b708bdbf5015.jpg)

![7b7d69f979670ce34214fc0bd3a79d63fdf4c4bd5600e2d210de4eef68b37f39.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/7b7d69f979670ce34214fc0bd3a79d63fdf4c4bd5600e2d210de4eef68b37f39.jpg)

![ec47db67ecc40610a832078533ab8b5789d22a88a45a115254cb4276cfcb293b.jpg](../icml_results/1809_Front-Loaded%20Robust%20Conformal%20Prediction_%20Heavy%20Calibration%2C%20Minimal%20Test-Time%20Cost/tables/ec47db67ecc40610a832078533ab8b5789d22a88a45a115254cb4276cfcb293b.jpg)

<a id="paper-1787"></a>

## Beyond Correctness: Distance-Based Social Dynamics of Multi-Agent Debate


### Images

![070f000652f657751e59494c57da5f2245e05b1709c9a77e79b4a6c519631bc9.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/070f000652f657751e59494c57da5f2245e05b1709c9a77e79b4a6c519631bc9.jpg)

![1ad0995c0d19364806fa80516c52c4d67128398668abf81fa96abaa2498d2bbb.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/1ad0995c0d19364806fa80516c52c4d67128398668abf81fa96abaa2498d2bbb.jpg)

![1f00b634d59e3c11e03e07fdcbc440e2d461dcdb1e385ddea25cd1f2a6ae11f5.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/1f00b634d59e3c11e03e07fdcbc440e2d461dcdb1e385ddea25cd1f2a6ae11f5.jpg)

![27d8eb2a1609e14078f4bded899cc9b373787436b313b1db7561f243520cdf9f.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/27d8eb2a1609e14078f4bded899cc9b373787436b313b1db7561f243520cdf9f.jpg)

![325ae51c98043002d1c3b7255ea99032acd63ca94af6b92e0f5f8dcff315636b.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/325ae51c98043002d1c3b7255ea99032acd63ca94af6b92e0f5f8dcff315636b.jpg)

![44ccc92fc6695d97ed07314b1a36eb68eda5f95dced606e82d20c8ee81ba8f71.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/44ccc92fc6695d97ed07314b1a36eb68eda5f95dced606e82d20c8ee81ba8f71.jpg)

![46431e7f1f439e53b7c7f3ff6e31b7cdf6ecfec9b57b82ae37978e198b677f81.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/46431e7f1f439e53b7c7f3ff6e31b7cdf6ecfec9b57b82ae37978e198b677f81.jpg)

![5601d1a754c6b4f583eb70b56d6018ae83df759748165cc96218fa58d4b50626.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/5601d1a754c6b4f583eb70b56d6018ae83df759748165cc96218fa58d4b50626.jpg)

![62c2c169bcdb7399c81528c4c0ba969f1e9f0ca9f2d1f5e8154e6c40da864b5e.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/62c2c169bcdb7399c81528c4c0ba969f1e9f0ca9f2d1f5e8154e6c40da864b5e.jpg)

![83ebeb8a2fde1ee8512ca06efe84569a32d4d5558a1eed06f3b5360c57c1323c.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/83ebeb8a2fde1ee8512ca06efe84569a32d4d5558a1eed06f3b5360c57c1323c.jpg)

![85585af22768b3c983e18ac31b0fe7281b8ee1e88784855bf0d89098b831b26d.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/85585af22768b3c983e18ac31b0fe7281b8ee1e88784855bf0d89098b831b26d.jpg)

![9b7201e1d1bf121ace53f01d73f8d7cb38eefe091184da6a1a105b38f6e62a6a.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/9b7201e1d1bf121ace53f01d73f8d7cb38eefe091184da6a1a105b38f6e62a6a.jpg)

![c36911eb603f38de445f2d12c6aada619fd8947a193639b2f685aa2ec09c0b95.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/c36911eb603f38de445f2d12c6aada619fd8947a193639b2f685aa2ec09c0b95.jpg)

![c4963b20107119bcbb052c8e64ee4bf07869d360665e510c25f780957aaf864c.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/c4963b20107119bcbb052c8e64ee4bf07869d360665e510c25f780957aaf864c.jpg)

![d5d73b7220a9b5c5214de670c44eabbd504cf2b11720a7bbe935ff48709b2fae.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/d5d73b7220a9b5c5214de670c44eabbd504cf2b11720a7bbe935ff48709b2fae.jpg)

![d8b0a5477293d65f1489230f1b46b27a682f1e9006287014d34083e88913a741.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/d8b0a5477293d65f1489230f1b46b27a682f1e9006287014d34083e88913a741.jpg)

![e1acdfe0bf64c0095c1954aa9f44344bd65a9da4435c5e7fd7daef12f027e402.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/e1acdfe0bf64c0095c1954aa9f44344bd65a9da4435c5e7fd7daef12f027e402.jpg)

![fe22131c7e0c8585c2b414cec3eed78f9731e1a20f85de6dee763b80d9f474b9.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/images/fe22131c7e0c8585c2b414cec3eed78f9731e1a20f85de6dee763b80d9f474b9.jpg)

### Tables

![0bbee95bd82a6e7ee2520b6dc59bf42a8a9e2ee553dd3ff2d2d73afe84286a11.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/tables/0bbee95bd82a6e7ee2520b6dc59bf42a8a9e2ee553dd3ff2d2d73afe84286a11.jpg)

![aec7f1c9d7e21d2914d3641c530be8e527c5133b6a174108416c3ab1a36e8f18.jpg](../icml_results/1810_Beyond%20Correctness_%20Distance-Based%20Social%20Dynamics%20of%20Multi-Agent%20Debate/tables/aec7f1c9d7e21d2914d3641c530be8e527c5133b6a174108416c3ab1a36e8f18.jpg)

<a id="paper-1788"></a>

## Unlearning’s Blind Spots: Over-Unlearning and Prototypical Relearning Attack


### Images

![073d7e0d4eab36acae73c0ad97e26bcf96495cf99b4d5bf6380182b89cf34aa9.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/images/073d7e0d4eab36acae73c0ad97e26bcf96495cf99b4d5bf6380182b89cf34aa9.jpg)

![0c7a285c611e404c150ad1ceb919d36bff0643260b4b94043a2bd89bf6ebbf84.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/images/0c7a285c611e404c150ad1ceb919d36bff0643260b4b94043a2bd89bf6ebbf84.jpg)

![2536e99b41583a83b102a36e89b750642d9e81cfe8a59b91b79cde97bd395565.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/images/2536e99b41583a83b102a36e89b750642d9e81cfe8a59b91b79cde97bd395565.jpg)

![30b10460835efb1c0a0c6b2ba8c3888b7a98140b32fc1bd39330be8393bec493.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/images/30b10460835efb1c0a0c6b2ba8c3888b7a98140b32fc1bd39330be8393bec493.jpg)

![55b0518792b9bf8603da925d2c8aa65ee7e5f15f86c6f94d1ff1596a39b35e59.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/images/55b0518792b9bf8603da925d2c8aa65ee7e5f15f86c6f94d1ff1596a39b35e59.jpg)

### Tables

![1e868a8ab0ed2b7e45b73c9ff25dce29afe43b678cc1d32359de151554374311.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/1e868a8ab0ed2b7e45b73c9ff25dce29afe43b678cc1d32359de151554374311.jpg)

![23d48f138b3454d9dd8eed37978ee2863ad5aae70f1c4f0d3fa4cc488aeb5a13.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/23d48f138b3454d9dd8eed37978ee2863ad5aae70f1c4f0d3fa4cc488aeb5a13.jpg)

![4214b1829f6b8452ef92f703f229e0ef3ff1a4eba8696d320102016bd01278b5.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/4214b1829f6b8452ef92f703f229e0ef3ff1a4eba8696d320102016bd01278b5.jpg)

![6493385f1028eb569fbe3bd393f105a0b47a6f5f1894806418270895d1361388.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/6493385f1028eb569fbe3bd393f105a0b47a6f5f1894806418270895d1361388.jpg)

![6654416f1518a360338c2969695f57b78dfc0b7c9d1599ce90e083c9bd3445c9.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/6654416f1518a360338c2969695f57b78dfc0b7c9d1599ce90e083c9bd3445c9.jpg)

![6f8eaa8d1de6bcf37e777fe12a810e4bee4c4b2a59ea1dbdb388e83cb9d47d6d.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/6f8eaa8d1de6bcf37e777fe12a810e4bee4c4b2a59ea1dbdb388e83cb9d47d6d.jpg)

![727c81c56150badc85db9d9859be8aa87279a0c781b818efa60ab03474ba4f6e.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/727c81c56150badc85db9d9859be8aa87279a0c781b818efa60ab03474ba4f6e.jpg)

![7420e7261c2fd0dd5cefc7dce981814ac5dc7b6e7255352237cbd3645c8c323a.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/7420e7261c2fd0dd5cefc7dce981814ac5dc7b6e7255352237cbd3645c8c323a.jpg)

![a49678b1d6bb2363fb88bad684bf09f1814446952261ceb49980b7d162a32644.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/a49678b1d6bb2363fb88bad684bf09f1814446952261ceb49980b7d162a32644.jpg)

![aee3a23d5b765c453ad5944faf3ea08650605615558bae0b7acd6f7acba5e308.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/aee3a23d5b765c453ad5944faf3ea08650605615558bae0b7acd6f7acba5e308.jpg)

![c2076299915255cf1864dbf41c6c97127bd5366867e2fca3031dfbc69dcbeeb6.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/c2076299915255cf1864dbf41c6c97127bd5366867e2fca3031dfbc69dcbeeb6.jpg)

![cc76a5ce6594170bd22fffd7edb179714da874a1593a6e28162ad5450fd8876a.jpg](../icml_results/1811_Unlearning%E2%80%99s%20Blind%20Spots_%20Over-Unlearning%20and%20Prototypical%20Relearning%20Attack/tables/cc76a5ce6594170bd22fffd7edb179714da874a1593a6e28162ad5450fd8876a.jpg)

<a id="paper-1789"></a>

## Geodesic Flow Matching for Denoising High-Dimensional Structured Representations


### Images

![07878f27f14b0ce31d100312f557192f68467ca0d5f9dce83bf1a4a2eb897df2.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/images/07878f27f14b0ce31d100312f557192f68467ca0d5f9dce83bf1a4a2eb897df2.jpg)

![21f66c35db14e357d2238b64d03033ce80e5f81500b455c1dbc5fcf65e5e0771.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/images/21f66c35db14e357d2238b64d03033ce80e5f81500b455c1dbc5fcf65e5e0771.jpg)

![498a3da5385e000a0bf2f1c5d8c2b84ff32b9ef3d7cfeb7cabf6937b866f05b1.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/images/498a3da5385e000a0bf2f1c5d8c2b84ff32b9ef3d7cfeb7cabf6937b866f05b1.jpg)

![beb0a6604a500bced55dda96f91e2b5007ef6ff36ebad75383d456423a527005.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/images/beb0a6604a500bced55dda96f91e2b5007ef6ff36ebad75383d456423a527005.jpg)

![c244159a680d6d6d943c5a5e49741944b872790f9b5e4886293ce8902e498a7d.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/images/c244159a680d6d6d943c5a5e49741944b872790f9b5e4886293ce8902e498a7d.jpg)

### Tables

![8adcdc262a3ebd3f680dbe3a9a9eb7d381d4597d3e8bf9ee66c3d491e125f211.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/tables/8adcdc262a3ebd3f680dbe3a9a9eb7d381d4597d3e8bf9ee66c3d491e125f211.jpg)

![aa48e9dc224deebe82df1d7d49a9440fe8673979afeda7243dea07699d632408.jpg](../icml_results/1812_Geodesic%20Flow%20Matching%20for%20Denoising%20High-Dimensional%20Structured%20Representations/tables/aa48e9dc224deebe82df1d7d49a9440fe8673979afeda7243dea07699d632408.jpg)

<a id="paper-1790"></a>

## GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data


### Images

![08a8698b6ca1b3f20d9c0c8a6f4b27ab6e49620cb72a9f674e0d07ce43361439.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/08a8698b6ca1b3f20d9c0c8a6f4b27ab6e49620cb72a9f674e0d07ce43361439.jpg)

![08e4904af6dab2e76aa8e3a816ccdc53f573154ab298cbe3fe3a5ee9fc34ac4f.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/08e4904af6dab2e76aa8e3a816ccdc53f573154ab298cbe3fe3a5ee9fc34ac4f.jpg)

![091dd13c14f9e8ae315c04fee12652d0fe9b7985ba3d7317a59436a89c93dbb3.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/091dd13c14f9e8ae315c04fee12652d0fe9b7985ba3d7317a59436a89c93dbb3.jpg)

![09e975bf6f1cb8c1ad6fddb34eec54d63331e74dbd119539260470cc615e02cf.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/09e975bf6f1cb8c1ad6fddb34eec54d63331e74dbd119539260470cc615e02cf.jpg)

![18754cb2f1d1e05d339c61eba84cb018cafe68fef0db177929b37c5c27568a19.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/18754cb2f1d1e05d339c61eba84cb018cafe68fef0db177929b37c5c27568a19.jpg)

![1f6610191c6c8e29f63d6740d8e1bfd01c673c36d284235717986f456673b545.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/1f6610191c6c8e29f63d6740d8e1bfd01c673c36d284235717986f456673b545.jpg)

![2554c4a8d534ea9208fce065bb48fa42f762e964eb2adb1124260182ea044545.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/2554c4a8d534ea9208fce065bb48fa42f762e964eb2adb1124260182ea044545.jpg)

![30cbe07b05b50fccf9cc2df82814950a13c94d2434c0a73d15120be6174ebf86.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/30cbe07b05b50fccf9cc2df82814950a13c94d2434c0a73d15120be6174ebf86.jpg)

![327456dc36bb38c4f9ba78b680d9ce99fa523ce984dcef08a51de71567e827a3.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/327456dc36bb38c4f9ba78b680d9ce99fa523ce984dcef08a51de71567e827a3.jpg)

![35745d3056094d360164764849e33ce7385e00a0eb3f246e2db6098f17f05ccd.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/35745d3056094d360164764849e33ce7385e00a0eb3f246e2db6098f17f05ccd.jpg)

![64ae290d1b67895cfc806eba42e963612a59e918f1f22e2ba6f9eed6c47339c3.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/64ae290d1b67895cfc806eba42e963612a59e918f1f22e2ba6f9eed6c47339c3.jpg)

![64b08d6d7da6c837444fd7b9aeb26b656bf36d03fd3ce1ec29719627729ce628.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/64b08d6d7da6c837444fd7b9aeb26b656bf36d03fd3ce1ec29719627729ce628.jpg)

![7121f38df2b44a843cfbed2ea9563d362c9a04e1480fa468cd476f2d845d2fcd.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/7121f38df2b44a843cfbed2ea9563d362c9a04e1480fa468cd476f2d845d2fcd.jpg)

![756604f0b1ee1b413d7e0758687ec0a7fd279284396d93ad5d6b3d12d4623ad9.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/756604f0b1ee1b413d7e0758687ec0a7fd279284396d93ad5d6b3d12d4623ad9.jpg)

![7f135dc027e9eb119879a77058b6c9733576c2cca11d6b445516d16d2c22ec0b.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/7f135dc027e9eb119879a77058b6c9733576c2cca11d6b445516d16d2c22ec0b.jpg)

![7f93c3feac0e1b6aa3dc9491eaf56884b47fb2472aad142b300ee7ed746f69c3.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/7f93c3feac0e1b6aa3dc9491eaf56884b47fb2472aad142b300ee7ed746f69c3.jpg)

![86b58d304fde3e6388512a3d634314cf3d29d50dafb3c7093820bef0cb2c2e1f.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/86b58d304fde3e6388512a3d634314cf3d29d50dafb3c7093820bef0cb2c2e1f.jpg)

![86d94a6c2e21ae1a6f46f83632ba6915db9773050d065954855554bc277e34af.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/86d94a6c2e21ae1a6f46f83632ba6915db9773050d065954855554bc277e34af.jpg)

![8ae2b2e2d608aeb3a5a638e43b491288b22a1c001429cbfc2c4e990e3af37ce9.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/8ae2b2e2d608aeb3a5a638e43b491288b22a1c001429cbfc2c4e990e3af37ce9.jpg)

![98a34ed9ae1c3b2410ca3921501f2af1392e4f386371e07e93d5f4c504e074b8.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/98a34ed9ae1c3b2410ca3921501f2af1392e4f386371e07e93d5f4c504e074b8.jpg)

![98a4a2290e733857782357250f41e4e8c6801d8d82e4c795356f05f4a1a9b705.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/98a4a2290e733857782357250f41e4e8c6801d8d82e4c795356f05f4a1a9b705.jpg)

![9b32ae5f108749e00f3e95d8ed56a9263849f126b9dc7936848cc17ac4ca0b6a.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/9b32ae5f108749e00f3e95d8ed56a9263849f126b9dc7936848cc17ac4ca0b6a.jpg)

![9d839967ee09f6f9c353d210a3ebe9e1b879746d5bebd3e5fe7708578cd1d483.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/9d839967ee09f6f9c353d210a3ebe9e1b879746d5bebd3e5fe7708578cd1d483.jpg)

![aad8ec0535f23ab78464cccc820775e135215fc3116af9093817981df569f98e.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/aad8ec0535f23ab78464cccc820775e135215fc3116af9093817981df569f98e.jpg)

![ab5e46b6b93bfc0ef4bdf393746adffcd2a866a064eaf3da1b6fd72626236fba.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/ab5e46b6b93bfc0ef4bdf393746adffcd2a866a064eaf3da1b6fd72626236fba.jpg)

![ae8950466d4eb8497d808247b561b7a503c793400d8cbd570ccc0bb67cea2d1e.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/ae8950466d4eb8497d808247b561b7a503c793400d8cbd570ccc0bb67cea2d1e.jpg)

![b7b793f8c80c295736c55fc80addb1bbff884931cffac7013d44af180e907a72.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/b7b793f8c80c295736c55fc80addb1bbff884931cffac7013d44af180e907a72.jpg)

![d0ce3e03902bb742d99217043ee29ef43d203541d9b86ee051ccf64d9d626209.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/d0ce3e03902bb742d99217043ee29ef43d203541d9b86ee051ccf64d9d626209.jpg)

![d64ccd1ccf354ca67fbffa41787e949e8bd29f97688b910ac6c6eda3a18f61d9.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/d64ccd1ccf354ca67fbffa41787e949e8bd29f97688b910ac6c6eda3a18f61d9.jpg)

![d7c79f702523351cd83cd9078bb6ec2160c311060c68d4e8c3594ac1c6fa019a.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/d7c79f702523351cd83cd9078bb6ec2160c311060c68d4e8c3594ac1c6fa019a.jpg)

![df0bdf50d5227914820ba179e422d11e18088da8c84a0273295878ec3da28556.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/df0bdf50d5227914820ba179e422d11e18088da8c84a0273295878ec3da28556.jpg)

![e445e579d28e2e33b6eaf035ef781d2186efb697ad303b1919aef68eca43c441.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/e445e579d28e2e33b6eaf035ef781d2186efb697ad303b1919aef68eca43c441.jpg)

![e60cd7ec85c4592a865be0bca7d526219a87c532540f0482b4878162459dac8a.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/e60cd7ec85c4592a865be0bca7d526219a87c532540f0482b4878162459dac8a.jpg)

![f768d56ea47a891a3e56115c9bd40c1815df0dac741f7c8aa9b687100fb7e7a3.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/f768d56ea47a891a3e56115c9bd40c1815df0dac741f7c8aa9b687100fb7e7a3.jpg)

![f88f6d05518e84fbb06fcd67a88fc5d6c0af01f7cfdb74078d73b6a7926b894d.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/f88f6d05518e84fbb06fcd67a88fc5d6c0af01f7cfdb74078d73b6a7926b894d.jpg)

![fd08df89efe7883b4d8a4b18971e8cb5e652037831f824d2b20ddd9d79415c1c.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/images/fd08df89efe7883b4d8a4b18971e8cb5e652037831f824d2b20ddd9d79415c1c.jpg)

### Tables

![0219ab000d64176bb0e04dd980cf1ee9146c5a962462a9afeea25308ca6bc9d1.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/0219ab000d64176bb0e04dd980cf1ee9146c5a962462a9afeea25308ca6bc9d1.jpg)

![04734592a1ae86a5bb4ddb61b1d062b4936e979e5f18af5086af27bff4c43518.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/04734592a1ae86a5bb4ddb61b1d062b4936e979e5f18af5086af27bff4c43518.jpg)

![0d13d571f8020cd1aa491ff6eb69232f312d846b210958b96cc6188e8c3a4fa1.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/0d13d571f8020cd1aa491ff6eb69232f312d846b210958b96cc6188e8c3a4fa1.jpg)

![11c8e9fa22f231ac09bc144c5bce4c050a3182069e6c1deb622335be2139118c.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/11c8e9fa22f231ac09bc144c5bce4c050a3182069e6c1deb622335be2139118c.jpg)

![176d2193bc14ceeee44aae0bf773acfbb8002db1082a1e358c5e45f329be7b03.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/176d2193bc14ceeee44aae0bf773acfbb8002db1082a1e358c5e45f329be7b03.jpg)

![18ebbfc193b44aabaf84786e3383c9addf4e37e2ece517ba797fd52628000edf.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/18ebbfc193b44aabaf84786e3383c9addf4e37e2ece517ba797fd52628000edf.jpg)

![21796c50f954b2b8e1b5c9ed6fcf84992bc62b95b69c631b7c703eae10bc3ae0.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/21796c50f954b2b8e1b5c9ed6fcf84992bc62b95b69c631b7c703eae10bc3ae0.jpg)

![275c6cccd3b2d5b2fb64ce9a86cbcac222c332e260b6004d583e498ac6ea1269.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/275c6cccd3b2d5b2fb64ce9a86cbcac222c332e260b6004d583e498ac6ea1269.jpg)

![28847d731c21c1b015f4de5233e601312115e2cef07db069748a85a38c0490fc.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/28847d731c21c1b015f4de5233e601312115e2cef07db069748a85a38c0490fc.jpg)

![2f1a79aff1dfcbac6a3a77e7a0fd7410aaadaa46971f60a3f9bd35b0f59cb745.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/2f1a79aff1dfcbac6a3a77e7a0fd7410aaadaa46971f60a3f9bd35b0f59cb745.jpg)

![34061c23d07d735e453d367c9fb6eb2b9f371de7242547fd25ed8cbe6ac31a84.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/34061c23d07d735e453d367c9fb6eb2b9f371de7242547fd25ed8cbe6ac31a84.jpg)

![39b075b93d734d87e258cdaec19d3e84602b4e82791fcf1db7c6c98fb8ceafc4.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/39b075b93d734d87e258cdaec19d3e84602b4e82791fcf1db7c6c98fb8ceafc4.jpg)

![3def3b48e4879c4b699513a4507dd56a4a534ce2f0ada584a2002e37b307492b.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/3def3b48e4879c4b699513a4507dd56a4a534ce2f0ada584a2002e37b307492b.jpg)

![466eb1e4706caba59be00684d41bdbe3eb4a488bb1142350ff32e2b0524d5b8c.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/466eb1e4706caba59be00684d41bdbe3eb4a488bb1142350ff32e2b0524d5b8c.jpg)

![4724a2de03240bcc698b2fb25ea011ef4c7bb0081f2ccfef7ced359581e383f9.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/4724a2de03240bcc698b2fb25ea011ef4c7bb0081f2ccfef7ced359581e383f9.jpg)

![49dd8a4b260cca8fede2b1be73036459ce337c267360a6217e6e188e9da44e60.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/49dd8a4b260cca8fede2b1be73036459ce337c267360a6217e6e188e9da44e60.jpg)

![4ecbfd58cc918f5ce5e5a80e36e675e54987c68e91a6b2c7e9702abbc72dee0a.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/4ecbfd58cc918f5ce5e5a80e36e675e54987c68e91a6b2c7e9702abbc72dee0a.jpg)

![62c60f9094b727e204b9392f7622ea8d9055d1f75ae557c206319d04c92edf73.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/62c60f9094b727e204b9392f7622ea8d9055d1f75ae557c206319d04c92edf73.jpg)

![70963e733bb7958a1749c35803b6cf327506e96800cd8c8d96af56bc6bad7d1e.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/70963e733bb7958a1749c35803b6cf327506e96800cd8c8d96af56bc6bad7d1e.jpg)

![8384765b088d1d859b6c1100dfda24c975dc51f6a011f53aab5537e39cc04225.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/8384765b088d1d859b6c1100dfda24c975dc51f6a011f53aab5537e39cc04225.jpg)

![8c8b08fa71801a6591a287c77376bf0f2c3d8b932bc46a52629b98b873506eeb.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/8c8b08fa71801a6591a287c77376bf0f2c3d8b932bc46a52629b98b873506eeb.jpg)

![8e475cf5d69b05656a020d2c157a7ef76eb3e174ef1c8c242a695996488ef686.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/8e475cf5d69b05656a020d2c157a7ef76eb3e174ef1c8c242a695996488ef686.jpg)

![915f69dd966c4dc60023be45ca5aaa5bce97268da2b8ab23b639f0e5909f8fb2.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/915f69dd966c4dc60023be45ca5aaa5bce97268da2b8ab23b639f0e5909f8fb2.jpg)

![afcaefa9d663dd933ffb800dd5961bca7bff1b411706f182976a56a6788eb38e.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/afcaefa9d663dd933ffb800dd5961bca7bff1b411706f182976a56a6788eb38e.jpg)

![c7aba75e7303c5f3fff73517b9c1ef83ac7590ff3ec6cf4613d1963cfdc49a4b.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/c7aba75e7303c5f3fff73517b9c1ef83ac7590ff3ec6cf4613d1963cfdc49a4b.jpg)

![c8a7b2423943874b2ab7ed244f7ef321b489343f5722aa8a1b3bca7ed6e2d70d.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/c8a7b2423943874b2ab7ed244f7ef321b489343f5722aa8a1b3bca7ed6e2d70d.jpg)

![d1ea21264aa02698ddd2cda6ebbfaa2302af5559eb5667595ef9bdfbd9036e0a.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/d1ea21264aa02698ddd2cda6ebbfaa2302af5559eb5667595ef9bdfbd9036e0a.jpg)

![e06d11647c3f657901cebce22f6ca03414873be93f1843d3916685ce910539ba.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/e06d11647c3f657901cebce22f6ca03414873be93f1843d3916685ce910539ba.jpg)

![e1f8b1b567ee2447dbaf52dc90b9b506116f06385211d516c3299fed705bf2fe.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/e1f8b1b567ee2447dbaf52dc90b9b506116f06385211d516c3299fed705bf2fe.jpg)

![e1ff703d8ed61f284925f50d7c900a76277f48ec12fcc3acf68ed0fb26f94d97.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/e1ff703d8ed61f284925f50d7c900a76277f48ec12fcc3acf68ed0fb26f94d97.jpg)

![eeff28a6b6809e00a20b24b92b0a747b1008679cf4c859daaffe33cb8ad0e6cf.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/eeff28a6b6809e00a20b24b92b0a747b1008679cf4c859daaffe33cb8ad0e6cf.jpg)

![f03e9582ed92ad8c06cc5cca35954adcde50abe40b74f9dbc14fe500c5c5a298.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/f03e9582ed92ad8c06cc5cca35954adcde50abe40b74f9dbc14fe500c5c5a298.jpg)

![f5f4234466c9de30bb31ecfd69ee94928f56258860a4a0a7ad42d5f320adb168.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/f5f4234466c9de30bb31ecfd69ee94928f56258860a4a0a7ad42d5f320adb168.jpg)

![fa05963b5d4cc725d7f0f8a21ae62338f73f0e537f3b25d8f18b2c157dd05d89.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/fa05963b5d4cc725d7f0f8a21ae62338f73f0e537f3b25d8f18b2c157dd05d89.jpg)

![fecfa13bf3671397cee971a749d0d28935a1d93de41eb90168aea8b1cf847e14.jpg](../icml_results/1813_GOTabPFN_%20From%20Feature%20Ordering%20to%20Compact%20Tokenization%20for%20Tabular%20Foundation%20Models%20on%20High-Dimens/tables/fecfa13bf3671397cee971a749d0d28935a1d93de41eb90168aea8b1cf847e14.jpg)

<a id="paper-1791"></a>

## Forward-KL Convergence of Time-Inhomogeneous Langevin Diffusions


### Images

![1e3f357a98c5f6fd3bed4e1705ca9aa38444b41c8e807e44fe7862599ab4c795.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/1e3f357a98c5f6fd3bed4e1705ca9aa38444b41c8e807e44fe7862599ab4c795.jpg)

![34cd3305cc361fef49a44b9d2b5d0dc0118c17da27705e9e606c3c544b5630a1.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/34cd3305cc361fef49a44b9d2b5d0dc0118c17da27705e9e606c3c544b5630a1.jpg)

![4381927c1861482d454d51f591b43b3b85287774881d350573c8c0f9a33a4f37.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/4381927c1861482d454d51f591b43b3b85287774881d350573c8c0f9a33a4f37.jpg)

![6947292510f03242ef575cce15009441a2f17911ef4c24cb0e2746a006f24592.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/6947292510f03242ef575cce15009441a2f17911ef4c24cb0e2746a006f24592.jpg)

![701b8222fc87b78fc20875aeb0968599789716080979fc8f5a72c38d64e441a9.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/701b8222fc87b78fc20875aeb0968599789716080979fc8f5a72c38d64e441a9.jpg)

![84f33ef10ff4496e7131a2ebb4bf533504a01f562279e4ea5e3a6aedcd667048.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/84f33ef10ff4496e7131a2ebb4bf533504a01f562279e4ea5e3a6aedcd667048.jpg)

![c54f8f9100f73237da3c40e4b2280defbca1ae7fef5db987d938927abae046f5.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/c54f8f9100f73237da3c40e4b2280defbca1ae7fef5db987d938927abae046f5.jpg)

![e5cf4e045ae033bb47a1d6cd03a90bf711f08fdf91009b911dbb15758d7538f4.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/e5cf4e045ae033bb47a1d6cd03a90bf711f08fdf91009b911dbb15758d7538f4.jpg)

![ea5d8dc88c65a50352ff9f978747318df4a475933f397fb05cef4e56478e4f96.jpg](../icml_results/1814_Forward-KL%20Convergence%20of%20Time-Inhomogeneous%20Langevin%20Diffusions/images/ea5d8dc88c65a50352ff9f978747318df4a475933f397fb05cef4e56478e4f96.jpg)

<a id="paper-1792"></a>

## Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space


### Images

![047fb7096ea28111548777bde20495ba98a79828c8297ba4d6a48f8a6a1d84aa.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/047fb7096ea28111548777bde20495ba98a79828c8297ba4d6a48f8a6a1d84aa.jpg)

![267be1a8f437cb0f2d17f4bb4fc8511d64e7d8546733f86250c7c2388770f9d5.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/267be1a8f437cb0f2d17f4bb4fc8511d64e7d8546733f86250c7c2388770f9d5.jpg)

![5fd3501a086cafef808abf01daf4521e8afb0afe36335bcccddb4359bc8cccef.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/5fd3501a086cafef808abf01daf4521e8afb0afe36335bcccddb4359bc8cccef.jpg)

![6a81066875294a001153db8cbfe9546d65ffaae5cff623ce71b98eb3ac0b6d72.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/6a81066875294a001153db8cbfe9546d65ffaae5cff623ce71b98eb3ac0b6d72.jpg)

![718d6fb74d0cad0eb7e8353fd398e2ed021eb13cbb3838c46cb8c57827ca3697.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/718d6fb74d0cad0eb7e8353fd398e2ed021eb13cbb3838c46cb8c57827ca3697.jpg)

![78e57b92031913407257d2e8615fec3d947bc0ad22b9f63755a7987b4e98859e.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/78e57b92031913407257d2e8615fec3d947bc0ad22b9f63755a7987b4e98859e.jpg)

![9329b10e908aab0cd055abebd3eb3e3ef2d677fc26fd3955a4c35aaae4d6f804.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/9329b10e908aab0cd055abebd3eb3e3ef2d677fc26fd3955a4c35aaae4d6f804.jpg)

![9754c897936c0fd61f52ebc8b04516ee446f5ce3cbb1502de5489488d66383a7.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/9754c897936c0fd61f52ebc8b04516ee446f5ce3cbb1502de5489488d66383a7.jpg)

![9d303c8c66d7bd85ee9b8653ec091b0c083bd9ba5877c4d6a286682771f6fd29.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/9d303c8c66d7bd85ee9b8653ec091b0c083bd9ba5877c4d6a286682771f6fd29.jpg)

![9f7757e411d2d9fa5d13aff9f15372c60388005e464a49020503f17db5ede4a7.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/9f7757e411d2d9fa5d13aff9f15372c60388005e464a49020503f17db5ede4a7.jpg)

![a28282682b2b7f6fe3a67aea00de528d24f40821829a817af121e413f745710e.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/a28282682b2b7f6fe3a67aea00de528d24f40821829a817af121e413f745710e.jpg)

![a358a8247bdb1e724bcabb1b827b57c59d43d54dfd124f64e951ea4b636c2ce5.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/a358a8247bdb1e724bcabb1b827b57c59d43d54dfd124f64e951ea4b636c2ce5.jpg)

![bafd872f2ca5b1b8b87d1772e8bd27817f9007febc94787506c1df8f9b5c1106.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/bafd872f2ca5b1b8b87d1772e8bd27817f9007febc94787506c1df8f9b5c1106.jpg)

![ca7e73cc2a4422bdf0cac51d92e67d5c9d1fbad11f465d07dfa9313ad26655dd.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/ca7e73cc2a4422bdf0cac51d92e67d5c9d1fbad11f465d07dfa9313ad26655dd.jpg)

![e51b9e7419d47b7bf652a0365e27a7ef2ff2c79efe9d4bd1b93a2112d16e2447.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/e51b9e7419d47b7bf652a0365e27a7ef2ff2c79efe9d4bd1b93a2112d16e2447.jpg)

![f78179b728ecf17253cb523a6d55528721b89f2a0a485fcb417e5e06156ad548.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/f78179b728ecf17253cb523a6d55528721b89f2a0a485fcb417e5e06156ad548.jpg)

![fdb89d4c087ddebdefdfb725a326297f1076fb15d579fb63b3c98e70bb5ed319.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/images/fdb89d4c087ddebdefdfb725a326297f1076fb15d579fb63b3c98e70bb5ed319.jpg)

### Tables

![e66d197f8241f62402457a86fdd331edbf3c9fc14c2a2544cdb8e4c40524d003.jpg](../icml_results/1815_Target-Aware%20Bandit%20Allocation%20for%20Scalable%20Surrogate%20Optimization%20in%20Chemical%20Space/tables/e66d197f8241f62402457a86fdd331edbf3c9fc14c2a2544cdb8e4c40524d003.jpg)

<a id="paper-1793"></a>

## Hierarchical Image Tokenization for Multi-Scale Image Super Resolution


### Images

![27ec4eddce4ced56150fa53d4ac75a3248ef80fa74f6ad267d1440fc0317dbfa.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/27ec4eddce4ced56150fa53d4ac75a3248ef80fa74f6ad267d1440fc0317dbfa.jpg)

![4d4e960a03daa5d63d0267e93b7ea2a684da857caebfacf9e8d8b497a9b194bc.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/4d4e960a03daa5d63d0267e93b7ea2a684da857caebfacf9e8d8b497a9b194bc.jpg)

![59cc8a470f622dfcea2b34ea74a6d62c7224d6a13d7215d956431b97db587085.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/59cc8a470f622dfcea2b34ea74a6d62c7224d6a13d7215d956431b97db587085.jpg)

![af5eda8a8bbb7ba7551e0b4c28355140ffe67ba147dd4d334c5db21d1dfe54e9.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/af5eda8a8bbb7ba7551e0b4c28355140ffe67ba147dd4d334c5db21d1dfe54e9.jpg)

![b3666bd49a0f6c451ff723e33bcac0f7455b880be68dd30a4daf805eefccbf71.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/b3666bd49a0f6c451ff723e33bcac0f7455b880be68dd30a4daf805eefccbf71.jpg)

![bc5e78afb08a9271694ea77d0147696239ee9bde8fee4e2e3ff2d5b9158e0436.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/images/bc5e78afb08a9271694ea77d0147696239ee9bde8fee4e2e3ff2d5b9158e0436.jpg)

### Tables

![0041017f19247ed6f04f098df529ef22377d40e3fa0fc1aedba59762cdd1738d.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/0041017f19247ed6f04f098df529ef22377d40e3fa0fc1aedba59762cdd1738d.jpg)

![018fc830d43a497d285c6ece73d594fb7d26b1df1f1bae1b87adea18e27b4557.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/018fc830d43a497d285c6ece73d594fb7d26b1df1f1bae1b87adea18e27b4557.jpg)

![8c4c14f83f356c8a033712ff38a3fc651d233a6ed34d86c4f0f08262347b53d0.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/8c4c14f83f356c8a033712ff38a3fc651d233a6ed34d86c4f0f08262347b53d0.jpg)

![901857a10f30ee6a1ef2dfcbd212fba98f123104998b645f63932da74284e8ed.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/901857a10f30ee6a1ef2dfcbd212fba98f123104998b645f63932da74284e8ed.jpg)

![94f06fa080e52e1c1b511fee2a70e3bfcf1c6165f417bbe29d6df62d27c87567.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/94f06fa080e52e1c1b511fee2a70e3bfcf1c6165f417bbe29d6df62d27c87567.jpg)

![cd1a8e900135c9973faf4e0c855b48a56bd2ad495d60a72f037b870fe684c97f.jpg](../icml_results/1816_Hierarchical%20Image%20Tokenization%20for%20Multi-Scale%20Image%20Super%20Resolution/tables/cd1a8e900135c9973faf4e0c855b48a56bd2ad495d60a72f037b870fe684c97f.jpg)

<a id="paper-1794"></a>

## Architecture Matters for Multi-Agent Security


### Images

![d5abba9778c9058224131ece8625ca27cf75ee4861e2671ab12aa11b4652d7dc.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/images/d5abba9778c9058224131ece8625ca27cf75ee4861e2671ab12aa11b4652d7dc.jpg)

### Tables

![0b4e0a8b7dabc8d28ef4ce4436321b9d61088b8ec66a6ac4e355b2797e108007.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/0b4e0a8b7dabc8d28ef4ce4436321b9d61088b8ec66a6ac4e355b2797e108007.jpg)

![1e84c8f737f39eb888df54986182649fbd3467105da8ba7f23e4b6c3d1366fa8.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/1e84c8f737f39eb888df54986182649fbd3467105da8ba7f23e4b6c3d1366fa8.jpg)

![27a1476424074dfff01f0bda122f10cfd9244f85dbfc714402e56adb923d512d.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/27a1476424074dfff01f0bda122f10cfd9244f85dbfc714402e56adb923d512d.jpg)

![38c25c4bbf9167abe4e3057117bd498dc90728d3c3b2d047e706e4a366b25d27.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/38c25c4bbf9167abe4e3057117bd498dc90728d3c3b2d047e706e4a366b25d27.jpg)

![3ce87abf2981d856f818ca2bb129825aa4f818bb514734fe738902bf2217c35b.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/3ce87abf2981d856f818ca2bb129825aa4f818bb514734fe738902bf2217c35b.jpg)

![465d9b0a33463443e912cd5f2d27ac2891a4355efa274f899e81ad3178d358ab.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/465d9b0a33463443e912cd5f2d27ac2891a4355efa274f899e81ad3178d358ab.jpg)

![49b4348af3af297609e8a8db4d298f92bc05ac176b9da86c28c2377e788d14dd.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/49b4348af3af297609e8a8db4d298f92bc05ac176b9da86c28c2377e788d14dd.jpg)

![55bcfb9ba876335f81ba33bf2b61e53e10755962c230cc13c71aa25ceb47afd8.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/55bcfb9ba876335f81ba33bf2b61e53e10755962c230cc13c71aa25ceb47afd8.jpg)

![a55f1724af1085264afae9c6aab7a0a49aeebd70214333eb74479412d90c5c13.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/a55f1724af1085264afae9c6aab7a0a49aeebd70214333eb74479412d90c5c13.jpg)

![b3232da87ab056c23c9499cb996443ae75530c040025db37b759a99ff35057da.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/b3232da87ab056c23c9499cb996443ae75530c040025db37b759a99ff35057da.jpg)

![c2ba04d7040607295439e5c6c35a4ba1b58df26d635ca237f1e58b78e16e4011.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/c2ba04d7040607295439e5c6c35a4ba1b58df26d635ca237f1e58b78e16e4011.jpg)

![cffe736b89a7c09cd5eccb3376283fc58fa345f4e3915b97131996d48cb111fe.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/cffe736b89a7c09cd5eccb3376283fc58fa345f4e3915b97131996d48cb111fe.jpg)

![dcb48affc68e48efc9479d8c72abdea86ccc7019a684db939fe454b75fa9266a.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/dcb48affc68e48efc9479d8c72abdea86ccc7019a684db939fe454b75fa9266a.jpg)

![dec9b86ec2525c09d3fca3a99471c119379b0a954e2d4bbb9a48ff341f4d3b54.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/dec9b86ec2525c09d3fca3a99471c119379b0a954e2d4bbb9a48ff341f4d3b54.jpg)

![eb4464e0bde05cf25fd620feb0b2723fafa31b2f681e4b3956d8f69b5b65ea06.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/eb4464e0bde05cf25fd620feb0b2723fafa31b2f681e4b3956d8f69b5b65ea06.jpg)

![f3b911cf591e3848d8d3450313bdf7461019cd4cd8e9350063b79182341f6384.jpg](../icml_results/1817_Architecture%20Matters%20for%20Multi-Agent%20Security/tables/f3b911cf591e3848d8d3450313bdf7461019cd4cd8e9350063b79182341f6384.jpg)

<a id="paper-1795"></a>

## Formalizing and Falsifying Causal Pathways of Rare Events


### Images

![3cce0155898dfb9c0649c2bd034c10f4f6c765319cb03972ce1c22650f9365e7.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/images/3cce0155898dfb9c0649c2bd034c10f4f6c765319cb03972ce1c22650f9365e7.jpg)

![6e269238e1654ff750b6f00d55dd4edf6395a6588d78a192c00f1a642ef1a00f.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/images/6e269238e1654ff750b6f00d55dd4edf6395a6588d78a192c00f1a642ef1a00f.jpg)

![71491a0aee2dcb51c8e80eae25e016ac9ae6dc86e8c5b69faa4dcd49547a9215.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/images/71491a0aee2dcb51c8e80eae25e016ac9ae6dc86e8c5b69faa4dcd49547a9215.jpg)

![75c6ff74bd4d47c904c6b16b6d6fdd97ff08b721731b13ed13b445a25a6cfc79.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/images/75c6ff74bd4d47c904c6b16b6d6fdd97ff08b721731b13ed13b445a25a6cfc79.jpg)

![a0f3408664b7dc3bc2aafe48387000546a7ded467c6af8e7aa5184dc928fe3e9.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/images/a0f3408664b7dc3bc2aafe48387000546a7ded467c6af8e7aa5184dc928fe3e9.jpg)

### Tables

![2cf8338db6a07af29bacd21d725c9cafd1d7634bc9fea0c6864b1793988ee4d9.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/tables/2cf8338db6a07af29bacd21d725c9cafd1d7634bc9fea0c6864b1793988ee4d9.jpg)

![6dea266346c256f836bbd0dac9c48b95edd5f7eb55ffea31dd284d910330d51f.jpg](../icml_results/1818_Formalizing%20and%20Falsifying%20Causal%20Pathways%20of%20Rare%20Events/tables/6dea266346c256f836bbd0dac9c48b95edd5f7eb55ffea31dd284d910330d51f.jpg)

<a id="paper-1796"></a>

## Pluralistic Leaderboards


### Images

![13e1cda6be16564f6d29e9e84a20e80800abbb2f97010ee9c3d12c7889aa4b3d.jpg](../icml_results/1819_Pluralistic%20Leaderboards/images/13e1cda6be16564f6d29e9e84a20e80800abbb2f97010ee9c3d12c7889aa4b3d.jpg)

![2a459564ad558d3cb22036cc6df6605a86f16329a1f60cf1b54db7281a404d4c.jpg](../icml_results/1819_Pluralistic%20Leaderboards/images/2a459564ad558d3cb22036cc6df6605a86f16329a1f60cf1b54db7281a404d4c.jpg)

![2d057a319e9f32ebaaff786df907421caec11113dbecf10a95e5ec1e0c1ecdd2.jpg](../icml_results/1819_Pluralistic%20Leaderboards/images/2d057a319e9f32ebaaff786df907421caec11113dbecf10a95e5ec1e0c1ecdd2.jpg)

![96ca50235ba6dea0bc9cb1e1ae6a56d496f71ab5cf3dbf7d9d16c600e3c1f109.jpg](../icml_results/1819_Pluralistic%20Leaderboards/images/96ca50235ba6dea0bc9cb1e1ae6a56d496f71ab5cf3dbf7d9d16c600e3c1f109.jpg)

![e31a97e56a12c6da1f7ca3b0181223513fa7e54ef07bbfd916a679ee51c7c5f1.jpg](../icml_results/1819_Pluralistic%20Leaderboards/images/e31a97e56a12c6da1f7ca3b0181223513fa7e54ef07bbfd916a679ee51c7c5f1.jpg)

### Tables

![2c05499ee86c7b5d05b54f5881c8d36a94bc9e604896744e76b34324e1975fc0.jpg](../icml_results/1819_Pluralistic%20Leaderboards/tables/2c05499ee86c7b5d05b54f5881c8d36a94bc9e604896744e76b34324e1975fc0.jpg)

![55694ae8e07c0862399275c674b80062e3600fc7b1b48cd21f45159e66a89224.jpg](../icml_results/1819_Pluralistic%20Leaderboards/tables/55694ae8e07c0862399275c674b80062e3600fc7b1b48cd21f45159e66a89224.jpg)

![cf5e7e99f6f8f432f0d698340b0f1b0648de82b2bb2465c88fc61387fce66678.jpg](../icml_results/1819_Pluralistic%20Leaderboards/tables/cf5e7e99f6f8f432f0d698340b0f1b0648de82b2bb2465c88fc61387fce66678.jpg)

![fb5e9b4cd5ca602813e9bbe9e549e3d37c2010b05ff6d91ba4b100508568162f.jpg](../icml_results/1819_Pluralistic%20Leaderboards/tables/fb5e9b4cd5ca602813e9bbe9e549e3d37c2010b05ff6d91ba4b100508568162f.jpg)

<a id="paper-1797"></a>

## SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model


### Images

![02b0f674b52762fc01fa4143914967ff96935bc0d9a5353ea85388df20d26ca1.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/02b0f674b52762fc01fa4143914967ff96935bc0d9a5353ea85388df20d26ca1.jpg)

![082448af918e7585f850f2251c6edf47d6eecc015a84aa2d5fb424c9af097d4f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/082448af918e7585f850f2251c6edf47d6eecc015a84aa2d5fb424c9af097d4f.jpg)

![0c78ea369c0e695beb8f21d6bac71afaa97c78110eb185defebb36a83899e6c8.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/0c78ea369c0e695beb8f21d6bac71afaa97c78110eb185defebb36a83899e6c8.jpg)

![1914b4a58db999574eed92cfcb7a74d2e062e6f6be561bc24464ffc352faedab.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/1914b4a58db999574eed92cfcb7a74d2e062e6f6be561bc24464ffc352faedab.jpg)

![1c270a412a1f4ca4d092a8559b349fbbae633bdb574e696100338f045ab93914.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/1c270a412a1f4ca4d092a8559b349fbbae633bdb574e696100338f045ab93914.jpg)

![21bbce085238192370da4391505c163dca0f822d8ad2b2e934576b2b18ff584e.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/21bbce085238192370da4391505c163dca0f822d8ad2b2e934576b2b18ff584e.jpg)

![265df0762558d633a7e5e584e8f04ba8bcb1987eb1a55932c4f5323567b7fab0.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/265df0762558d633a7e5e584e8f04ba8bcb1987eb1a55932c4f5323567b7fab0.jpg)

![34a4ee8d49b34d61dfb23393848f3398fad34e5d589734759daf11bca3cae93b.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/34a4ee8d49b34d61dfb23393848f3398fad34e5d589734759daf11bca3cae93b.jpg)

![3783fe49901eed513a5f1f47174bb872457767f05dcbeab26e23797488616199.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/3783fe49901eed513a5f1f47174bb872457767f05dcbeab26e23797488616199.jpg)

![4046603b3c3c5515f745a1fd4dbc264e6a0a0a2a2d7d06f4ad575af39fa8b40f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/4046603b3c3c5515f745a1fd4dbc264e6a0a0a2a2d7d06f4ad575af39fa8b40f.jpg)

![6509f2428393419c52b28b54d69f0fddacaf097fa24e863c6ef4105cfd9e6c3b.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/6509f2428393419c52b28b54d69f0fddacaf097fa24e863c6ef4105cfd9e6c3b.jpg)

![6a0c662e87653f533dcbfa45a5ca175ed6a465239235afff25d0b6e2cbd960f2.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/6a0c662e87653f533dcbfa45a5ca175ed6a465239235afff25d0b6e2cbd960f2.jpg)

![6fdfe6d30a4bbf7b6a1b905a78929068692a78d1ba102bf6e4eb6eb06e2c96ae.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/6fdfe6d30a4bbf7b6a1b905a78929068692a78d1ba102bf6e4eb6eb06e2c96ae.jpg)

![74c824709b00c286b3c65282a8e87c7cfa25c5123073de68365d1abd37a0333d.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/74c824709b00c286b3c65282a8e87c7cfa25c5123073de68365d1abd37a0333d.jpg)

![7b19faf20157094358d8abbcf97c614de42d6af6a2dcf87674938bc85a3253b9.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/7b19faf20157094358d8abbcf97c614de42d6af6a2dcf87674938bc85a3253b9.jpg)

![7fe561f633db2b430b00818b6699a7b5ec3f4f524ad544a2acba1ec683503126.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/7fe561f633db2b430b00818b6699a7b5ec3f4f524ad544a2acba1ec683503126.jpg)

![8320dbd0aeb27ddcee3a8aeddcd5eddf150a526474bcee6af3a8c6766c6bfb16.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/8320dbd0aeb27ddcee3a8aeddcd5eddf150a526474bcee6af3a8c6766c6bfb16.jpg)

![85c7f39d2f83fc47c1dd6f906a2aabd0f0820e2d45f86edb6359247a2af053c1.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/85c7f39d2f83fc47c1dd6f906a2aabd0f0820e2d45f86edb6359247a2af053c1.jpg)

![8db6dab729160c9e045c79bc2fae6f2ec0c7fd4b389a70f92cb4099deda4a979.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/8db6dab729160c9e045c79bc2fae6f2ec0c7fd4b389a70f92cb4099deda4a979.jpg)

![9995ca39900a6bdf32c34be353681ade3340f4c24bedca6654c402ef7839bc53.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/9995ca39900a6bdf32c34be353681ade3340f4c24bedca6654c402ef7839bc53.jpg)

![aca58b939557008ae82f8c72636c2ec5a00fff4ae707f602de31b739cf8474ad.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/aca58b939557008ae82f8c72636c2ec5a00fff4ae707f602de31b739cf8474ad.jpg)

![ae7676bb17809c9d1467c5ea6c0fb4d39ec87006ca7df98238e8389a360a936e.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/ae7676bb17809c9d1467c5ea6c0fb4d39ec87006ca7df98238e8389a360a936e.jpg)

![b5e1a79a7da2236eeaf1dd39b84f0a7b01162ba58f784103b3a5c185cc88b8e1.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/b5e1a79a7da2236eeaf1dd39b84f0a7b01162ba58f784103b3a5c185cc88b8e1.jpg)

![ca4422f3d02f202493534e7532b4db1af7b5386cc7e07a9cc6c8dc5aa3ae3a46.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/ca4422f3d02f202493534e7532b4db1af7b5386cc7e07a9cc6c8dc5aa3ae3a46.jpg)

![cf3da8950b3fa74d7b47fb418bb25002ea0cbcf470fd679776cef1dc9f8359e3.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/cf3da8950b3fa74d7b47fb418bb25002ea0cbcf470fd679776cef1dc9f8359e3.jpg)

![e2273ad088a6665392e03c10ed09684aa4915d462a5f4bd278df67372b004522.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/e2273ad088a6665392e03c10ed09684aa4915d462a5f4bd278df67372b004522.jpg)

![ea056195e9408cccfe756f7b83e4c385d78797aafd8014a27247d6c53df517a3.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/ea056195e9408cccfe756f7b83e4c385d78797aafd8014a27247d6c53df517a3.jpg)

![ef413e87ca58b09f61c9b5b986ab8f3e5934d2dbc0a3386dc3a41e4e2273db89.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/ef413e87ca58b09f61c9b5b986ab8f3e5934d2dbc0a3386dc3a41e4e2273db89.jpg)

![f1da0079c078d47727077c8a47939b626ff8332de0fb3f9f4ae916ced9d20506.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/f1da0079c078d47727077c8a47939b626ff8332de0fb3f9f4ae916ced9d20506.jpg)

![f8a02559e2574228515e56b9809456f56767d018ade93f92311feb1399cfef3f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/f8a02559e2574228515e56b9809456f56767d018ade93f92311feb1399cfef3f.jpg)

![ff680db00b4d65d42a356d263313dc693cb36fbdbde73213047d6ea723c5eded.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/images/ff680db00b4d65d42a356d263313dc693cb36fbdbde73213047d6ea723c5eded.jpg)

### Tables

![0359ace10e288edd7f16484d7b706b653951640655920b97576bbfcad30987c4.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/0359ace10e288edd7f16484d7b706b653951640655920b97576bbfcad30987c4.jpg)

![1a37c62f6bacd7702a413b785f7bfb4384160adce5279fc791f81d76a0196b8f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/1a37c62f6bacd7702a413b785f7bfb4384160adce5279fc791f81d76a0196b8f.jpg)

![3712d3a6ec1149327c8d4259100ce46c5b7aa676d87d6c2b6e38512e71585da0.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/3712d3a6ec1149327c8d4259100ce46c5b7aa676d87d6c2b6e38512e71585da0.jpg)

![398318fd7625579a9ce6d62bce0a896dfad52f303fd0a4a245042bab28552d1d.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/398318fd7625579a9ce6d62bce0a896dfad52f303fd0a4a245042bab28552d1d.jpg)

![3e09d431e76d36872f1a5904850065bf898bb5b8432f34c9e35c261ab649a27b.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/3e09d431e76d36872f1a5904850065bf898bb5b8432f34c9e35c261ab649a27b.jpg)

![4e44d006cd934bffe171bc5d459bcb5ca657ed8c7b9f412f9a12e889acdda4d8.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/4e44d006cd934bffe171bc5d459bcb5ca657ed8c7b9f412f9a12e889acdda4d8.jpg)

![56cc95351847227b2b86bb22a1b7fb7c876e97ca440ecc19d651eede3bcba83a.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/56cc95351847227b2b86bb22a1b7fb7c876e97ca440ecc19d651eede3bcba83a.jpg)

![5afe449b8ead2b3187174b8916ab304435ba603b437a694e6dffc8fba9d3b3b7.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/5afe449b8ead2b3187174b8916ab304435ba603b437a694e6dffc8fba9d3b3b7.jpg)

![5f8020ac12a09b3c368177730d4e0079b6197ea35487e21a1abdd220cec4fab7.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/5f8020ac12a09b3c368177730d4e0079b6197ea35487e21a1abdd220cec4fab7.jpg)

![822638239e1749c161e78654b825c490e9d5a94cc12a8b0b217adf11bd6a8554.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/822638239e1749c161e78654b825c490e9d5a94cc12a8b0b217adf11bd6a8554.jpg)

![8498dfb02b534db380b0c0d4a7187286f9619a367cbffc8f672cc5e19cd85066.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/8498dfb02b534db380b0c0d4a7187286f9619a367cbffc8f672cc5e19cd85066.jpg)

![8e39df8c693e2338351b36f226f048af7e987e6ce7abb4821623f5cf474b73a4.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/8e39df8c693e2338351b36f226f048af7e987e6ce7abb4821623f5cf474b73a4.jpg)

![8edc18303a224adabcae48d39c7eb1bd2f7732ee08eb5ad89fbaaa1dfedaa393.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/8edc18303a224adabcae48d39c7eb1bd2f7732ee08eb5ad89fbaaa1dfedaa393.jpg)

![9f38a1e6e545ee48e0d10a2c2daf4eb657911db91bec0d48d44f0090500a7edb.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/9f38a1e6e545ee48e0d10a2c2daf4eb657911db91bec0d48d44f0090500a7edb.jpg)

![a3f8f3731f1cb2a750857b2fcaa2270dbe6e3b06a0710f5060fc6168ed412491.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/a3f8f3731f1cb2a750857b2fcaa2270dbe6e3b06a0710f5060fc6168ed412491.jpg)

![a7bd03a3955411eb3b5de562041e1ab225a191262a65045c49223b44460da227.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/a7bd03a3955411eb3b5de562041e1ab225a191262a65045c49223b44460da227.jpg)

![ac4bb58d74d80739677b4630fe6c01219449af366b964f223003099b61270e03.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/ac4bb58d74d80739677b4630fe6c01219449af366b964f223003099b61270e03.jpg)

![aeb9985e3859348a26d464904b4012f7764cbaf3fd530e1e034c7f7f3d83e9c5.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/aeb9985e3859348a26d464904b4012f7764cbaf3fd530e1e034c7f7f3d83e9c5.jpg)

![beaa0292ac6e04a0248643714c7bcc5a5bf34bfe1017d461165407e257f3f9c5.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/beaa0292ac6e04a0248643714c7bcc5a5bf34bfe1017d461165407e257f3f9c5.jpg)

![c0fc037ccc09c4f0dacb99fc3aae48a39068cd74445b5502ae46bbe94a83edb4.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/c0fc037ccc09c4f0dacb99fc3aae48a39068cd74445b5502ae46bbe94a83edb4.jpg)

![cb7013452fb3e90b57df175fbcac564ef3633c0faff5250c27887c439081e857.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/cb7013452fb3e90b57df175fbcac564ef3633c0faff5250c27887c439081e857.jpg)

![d4cec8dd79dc7318088d1918fa87edb78e0b62f11f2b78f538e8b20b3ad8ade6.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/d4cec8dd79dc7318088d1918fa87edb78e0b62f11f2b78f538e8b20b3ad8ade6.jpg)

![d68b081e09bbb0aa9ec44f857a74b388669e9767eb7b3587704f67da0e7c4d97.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/d68b081e09bbb0aa9ec44f857a74b388669e9767eb7b3587704f67da0e7c4d97.jpg)

![e217bbc082697f55f75b8e2bffda13a539d6f8d736b792883b36d610dffa405f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/e217bbc082697f55f75b8e2bffda13a539d6f8d736b792883b36d610dffa405f.jpg)

![e7066ebefcaa17c7510cf0767230afad61b93ca81686b3fae3ae37145dbef22f.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/e7066ebefcaa17c7510cf0767230afad61b93ca81686b3fae3ae37145dbef22f.jpg)

![efc852297db58d4484e44e26cdb3eb69288e02b776b10fa5aa0f2360369d01be.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/efc852297db58d4484e44e26cdb3eb69288e02b776b10fa5aa0f2360369d01be.jpg)

![f4415c9ece7c40e6ba45df9983d24ae8e13970b3ff542dd5a257a85abb57fbac.jpg](../icml_results/1820_SMART_%20Scalable%20Mesh-free%20Aerodynamic%20Simulations%20from%20Raw%20Geometries%20using%20a%20Transformer-based%20Surr/tables/f4415c9ece7c40e6ba45df9983d24ae8e13970b3ff542dd5a257a85abb57fbac.jpg)

<a id="paper-1798"></a>

## Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases


### Images

![0c95190c2191b07ad146270e5c902b3f5f51c78de1161ad85d0f7f157796c0ef.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/0c95190c2191b07ad146270e5c902b3f5f51c78de1161ad85d0f7f157796c0ef.jpg)

![0e879be6eff32dc6ff77ff4cd47a78f563c4c9ff3e0cc456bfba9660abc810b5.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/0e879be6eff32dc6ff77ff4cd47a78f563c4c9ff3e0cc456bfba9660abc810b5.jpg)

![18cbf2885c0409dd94e5a2b83e2321e4aaf5112b87b12a5b15ba7f7eb83d93de.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/18cbf2885c0409dd94e5a2b83e2321e4aaf5112b87b12a5b15ba7f7eb83d93de.jpg)

![1950263fc89f6c2ead4eee5d55e858c8bbfe77fe1eb82e06ea33d7884a4f7ea9.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/1950263fc89f6c2ead4eee5d55e858c8bbfe77fe1eb82e06ea33d7884a4f7ea9.jpg)

![28e8a195222b64164bf5700a40c8728f1d8b49f99dedacd002c6f6291910321d.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/28e8a195222b64164bf5700a40c8728f1d8b49f99dedacd002c6f6291910321d.jpg)

![40093ed46c6274bbe6be5ebe229abca7f80ecc236adc22f18d2b9bb9dc2bb673.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/40093ed46c6274bbe6be5ebe229abca7f80ecc236adc22f18d2b9bb9dc2bb673.jpg)

![5f3e5f2fcd923ebe473038e2b9e62ea02880dc1fde1b90d330aba3c2df08420c.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/5f3e5f2fcd923ebe473038e2b9e62ea02880dc1fde1b90d330aba3c2df08420c.jpg)

![8091e7f75e82462f37aa2afc123293ed370d41e14423e87ffede3a7e42aee7ff.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/8091e7f75e82462f37aa2afc123293ed370d41e14423e87ffede3a7e42aee7ff.jpg)

![bd987c3d18e238fb58f4473f7d5ef98ac9a73d994b820b16d51db5932fd0b8b0.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/bd987c3d18e238fb58f4473f7d5ef98ac9a73d994b820b16d51db5932fd0b8b0.jpg)

![d3552cac535ddd54f9800d85787ca4d60d56d3199c0b4440643193ae5808dccf.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/d3552cac535ddd54f9800d85787ca4d60d56d3199c0b4440643193ae5808dccf.jpg)

![dc5e66bb4a81692d01965142278a6c76e241d40f3a5148518159c68e0fdd51da.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/dc5e66bb4a81692d01965142278a6c76e241d40f3a5148518159c68e0fdd51da.jpg)

![df40e53343a5f58bd3b979477ac1d496f1ed29f08175a9650311f7c5699a4403.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/df40e53343a5f58bd3b979477ac1d496f1ed29f08175a9650311f7c5699a4403.jpg)

![e57993227d944e0e90e2d93b3360f6f79e8b5d06aea77441101ddfaa93c9119c.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/e57993227d944e0e90e2d93b3360f6f79e8b5d06aea77441101ddfaa93c9119c.jpg)

![ebf376bccb0b1b57168dcaeae2e182f4d3fa4ab49feece696189f04c44275630.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/ebf376bccb0b1b57168dcaeae2e182f4d3fa4ab49feece696189f04c44275630.jpg)

![f72260033058b18d336db5321594a671c3f73e101c536058d490ab21af7c79a7.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/images/f72260033058b18d336db5321594a671c3f73e101c536058d490ab21af7c79a7.jpg)

### Tables

![04b01ca4525717a228f7324bd84517336f438b6561a5f1f1f6670757581abd73.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/04b01ca4525717a228f7324bd84517336f438b6561a5f1f1f6670757581abd73.jpg)

![10a6059fa3837d38a3e01693dad6ed263aa93191211f930712fd83cf9de7d7d4.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/10a6059fa3837d38a3e01693dad6ed263aa93191211f930712fd83cf9de7d7d4.jpg)

![113d76126d48b5bfe6aca206fb31509842d48889335f1c708f05b0c4e0ec4497.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/113d76126d48b5bfe6aca206fb31509842d48889335f1c708f05b0c4e0ec4497.jpg)

![55050981ecc3e958c18a6d1bd7ffda61aefcc586e722281046e1e9d97810c693.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/55050981ecc3e958c18a6d1bd7ffda61aefcc586e722281046e1e9d97810c693.jpg)

![60b0bf844617ec25ab2ae81227e0d6be6a2c3b81004a41db858306bfca34664b.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/60b0bf844617ec25ab2ae81227e0d6be6a2c3b81004a41db858306bfca34664b.jpg)

![629c6dcfa486f12ca9274a02ea8919d1e8ad1bcac6acba6c058580987916d787.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/629c6dcfa486f12ca9274a02ea8919d1e8ad1bcac6acba6c058580987916d787.jpg)

![7eb0366dc3a4a5bea28dfc8e9d3e78962e64dc43bbecf1e138d46e632bbb77b1.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/7eb0366dc3a4a5bea28dfc8e9d3e78962e64dc43bbecf1e138d46e632bbb77b1.jpg)

![86ef4169265011a68f31873e13f9c052f174f62c4ba841f85226621da5d40d7f.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/86ef4169265011a68f31873e13f9c052f174f62c4ba841f85226621da5d40d7f.jpg)

![92ba0d9f31f50cb331d274df132ca190bf8a84fbfcc9ceb7efa5a0d67d5f1fce.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/92ba0d9f31f50cb331d274df132ca190bf8a84fbfcc9ceb7efa5a0d67d5f1fce.jpg)

![95def7a34d5467260c91232c547e1e89e441cc2bd2358bbd2e30b21163bc0f27.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/95def7a34d5467260c91232c547e1e89e441cc2bd2358bbd2e30b21163bc0f27.jpg)

![9b677d0fdb2f05576a1c278698c6f26a63570e2d1c95f558f8edf193af58dfc0.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/9b677d0fdb2f05576a1c278698c6f26a63570e2d1c95f558f8edf193af58dfc0.jpg)

![a2749a2fc9ec6e07b8b0c02284de11fd66d338c375d3e6db7bf7c21acb141ea5.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/a2749a2fc9ec6e07b8b0c02284de11fd66d338c375d3e6db7bf7c21acb141ea5.jpg)

![bb413fb889636bdb5f69b76c3b6954a21d6a9aa7371641f5e64a5c48be0c73c5.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/bb413fb889636bdb5f69b76c3b6954a21d6a9aa7371641f5e64a5c48be0c73c5.jpg)

![cc02df97b48e55114d64f2d8929d0d6533bc5d3390deb21bf2102b6df6a48eba.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/cc02df97b48e55114d64f2d8929d0d6533bc5d3390deb21bf2102b6df6a48eba.jpg)

![cf785200cb5f6a1efed86b8daa3afd1ef6490b70015b9d3c265f077cb5f2db84.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/cf785200cb5f6a1efed86b8daa3afd1ef6490b70015b9d3c265f077cb5f2db84.jpg)

![cfcb33f98cef6bcb3060a7968b49ca7688e448a9e7f1af8c51e49e8ef87a040e.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/cfcb33f98cef6bcb3060a7968b49ca7688e448a9e7f1af8c51e49e8ef87a040e.jpg)

![e7d9f2f01b7f5e023cebf900dd5725f7c7c7f0cfeda456e55cbe4fec02cb889e.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/e7d9f2f01b7f5e023cebf900dd5725f7c7c7f0cfeda456e55cbe4fec02cb889e.jpg)

![ece3690fa49043b4b694053842652f7fc57a11766cde07179997ae77270c32d8.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/ece3690fa49043b4b694053842652f7fc57a11766cde07179997ae77270c32d8.jpg)

![fe927bc1fbf9da12bf7e95687803cdbd7103cb488fc65adef47da7280c7dde25.jpg](../icml_results/1821_Alignment%20Tampering_%20How%20Reinforcement%20Learning%20from%20Human%20Feedback%20Is%20Exploited%20to%20Optimize%20Misalig/tables/fe927bc1fbf9da12bf7e95687803cdbd7103cb488fc65adef47da7280c7dde25.jpg)

<a id="paper-1799"></a>

## Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models


### Images

![11acf198ab53e48fa4a893af6f37072378628b2688ab09f5ab0971afdfa70204.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/images/11acf198ab53e48fa4a893af6f37072378628b2688ab09f5ab0971afdfa70204.jpg)

![bb183575cdbafc416e0ab50ec43301172913d54197f3b92c24671734912503ac.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/images/bb183575cdbafc416e0ab50ec43301172913d54197f3b92c24671734912503ac.jpg)

![c497b8cf4cc90f5b22dc6e8db82ff626a9cd6d5c85a22a17baead295fdcf632d.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/images/c497b8cf4cc90f5b22dc6e8db82ff626a9cd6d5c85a22a17baead295fdcf632d.jpg)

![d0fc1ee3e41fc2cf5a07a6cc03b5d571909da71937b1d38be68da84bd050d6ae.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/images/d0fc1ee3e41fc2cf5a07a6cc03b5d571909da71937b1d38be68da84bd050d6ae.jpg)

![f129c14a556c1ca5584fe0b8bc3fd6e9f406dcd76e3b0d02c40f894fd70472d2.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/images/f129c14a556c1ca5584fe0b8bc3fd6e9f406dcd76e3b0d02c40f894fd70472d2.jpg)

### Tables

![19d0b31fe343c3efa6303233bd3a3c20be8f34f71c534f6b20528113852176f5.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/19d0b31fe343c3efa6303233bd3a3c20be8f34f71c534f6b20528113852176f5.jpg)

![33f82402f539e1ce7875068ed0ccd65be49e6e186c83a805de331f94df3df151.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/33f82402f539e1ce7875068ed0ccd65be49e6e186c83a805de331f94df3df151.jpg)

![3b3ad61d73d4ac36fd7d6bf77c2a8b129e2069668baeb0fbb4546315736d8955.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/3b3ad61d73d4ac36fd7d6bf77c2a8b129e2069668baeb0fbb4546315736d8955.jpg)

![58e0cc90367d44927b23270140ab5650c6aaa6e2eba1422dfb233e93f3cb906c.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/58e0cc90367d44927b23270140ab5650c6aaa6e2eba1422dfb233e93f3cb906c.jpg)

![837a0d12d2478d40eebe8ced8a85aef274303ebbc1c69fbe57e330fd46f3280e.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/837a0d12d2478d40eebe8ced8a85aef274303ebbc1c69fbe57e330fd46f3280e.jpg)

![9095a75bd50ab5e76b04cfa57a816b0363cafb7648662d79561137c0c42d73d6.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/9095a75bd50ab5e76b04cfa57a816b0363cafb7648662d79561137c0c42d73d6.jpg)

![91b9ac1a9e81c9e42d8baa495b04142a7aa4ac0a0b8dcd58ec96ac54e6911d48.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/91b9ac1a9e81c9e42d8baa495b04142a7aa4ac0a0b8dcd58ec96ac54e6911d48.jpg)

![a19e9e2868975f6c471ecf3d16716e346ca7cce67967c39fc162fab030b69489.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/a19e9e2868975f6c471ecf3d16716e346ca7cce67967c39fc162fab030b69489.jpg)

![ae6e44190dd9b6965a8273e2c16b2fdf29a34a650e166f7873df63209cf4e81f.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/ae6e44190dd9b6965a8273e2c16b2fdf29a34a650e166f7873df63209cf4e81f.jpg)

![d5dc7785df30933abd069226990bafdd4c524d855803def7f70ff9bcbe15d1a6.jpg](../icml_results/1822_Towards%20Fine-Grained%20Robustness_%20Attention-Guided%20Test-Time%20Prompt%20Tuning%20for%20Vision-Language%20Models/tables/d5dc7785df30933abd069226990bafdd4c524d855803def7f70ff9bcbe15d1a6.jpg)

<a id="paper-1800"></a>

## Lower Bounds for Frank-Wolfe on Strongly Convex Sets


### Images

![3e098b2694538e3da6052b27658adeb024e63a210565dcbe4de6bc8e73af66f6.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/3e098b2694538e3da6052b27658adeb024e63a210565dcbe4de6bc8e73af66f6.jpg)

![47b19e78d7d0c37b22308d6ff01c802b2fae7ccb77da1e88e73cf7b8fd290062.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/47b19e78d7d0c37b22308d6ff01c802b2fae7ccb77da1e88e73cf7b8fd290062.jpg)

![9740fbddcfc493bd8ca2890c477e4ebd916fd8f9ee9d4e3dfaf7ac97b3464284.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/9740fbddcfc493bd8ca2890c477e4ebd916fd8f9ee9d4e3dfaf7ac97b3464284.jpg)

![a6c2ba65863903f91a6490aa894b0b78a670d3afea525a4f391535d84385428b.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/a6c2ba65863903f91a6490aa894b0b78a670d3afea525a4f391535d84385428b.jpg)

![a851b3018dbf7c5e9f2a27adc8391154612a711d2f701cd3fb1fa41d065a59d2.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/a851b3018dbf7c5e9f2a27adc8391154612a711d2f701cd3fb1fa41d065a59d2.jpg)

![acb1dcf045cc192d465d9713256db018659339aedaa53835e39fafe9305f939d.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/acb1dcf045cc192d465d9713256db018659339aedaa53835e39fafe9305f939d.jpg)

![bfc5aa9d249ade16b8e69214fe395e8f52f2ebf89660d139b9184614e9207c36.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/bfc5aa9d249ade16b8e69214fe395e8f52f2ebf89660d139b9184614e9207c36.jpg)

![c80197461eab351bb7875b3d4cc73544d69a87b884468f42a7f8f8b0aee226e9.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/c80197461eab351bb7875b3d4cc73544d69a87b884468f42a7f8f8b0aee226e9.jpg)

![dbff71724c1522876096b09233643dbde8e2e169255722d49e56cf9e1c2a1182.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/images/dbff71724c1522876096b09233643dbde8e2e169255722d49e56cf9e1c2a1182.jpg)

### Tables

![1c3fc050ee04ab747586d143b03a5b1d78c77f9fb4c65f3599d12088f5cd8141.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/tables/1c3fc050ee04ab747586d143b03a5b1d78c77f9fb4c65f3599d12088f5cd8141.jpg)

![b652352f1b4c367758b05d768c2b5de62c161f043663912bc4169cc5627d5b88.jpg](../icml_results/1823_Lower%20Bounds%20for%20Frank-Wolfe%20on%20Strongly%20Convex%20Sets/tables/b652352f1b4c367758b05d768c2b5de62c161f043663912bc4169cc5627d5b88.jpg)

<a id="paper-1801"></a>

## Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling


### Images

![1327d68f36ddba1e46289e450aa9ccf4c69b2dd7bc1a5adcb8d4c37a3c8cccb3.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/1327d68f36ddba1e46289e450aa9ccf4c69b2dd7bc1a5adcb8d4c37a3c8cccb3.jpg)

![18b066e8c6229887c262b5fdaf079bfae1db7c29f32ad21da1c0a36b133c407b.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/18b066e8c6229887c262b5fdaf079bfae1db7c29f32ad21da1c0a36b133c407b.jpg)

![31d4828a3130e2fb0506572ae0d018e389188e74ccb1f3588945eafa97d5935a.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/31d4828a3130e2fb0506572ae0d018e389188e74ccb1f3588945eafa97d5935a.jpg)

![53acba906aca7dd57fe67b90104e82aee60aa25098d1d2bee31ff330b37dd6c4.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/53acba906aca7dd57fe67b90104e82aee60aa25098d1d2bee31ff330b37dd6c4.jpg)

![546b6ef12fd945ddd9578cc3884cd9550fd22f08940caebfefb0e10d85208d47.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/546b6ef12fd945ddd9578cc3884cd9550fd22f08940caebfefb0e10d85208d47.jpg)

![619d646b5e816ad8e72e1180c5c08cc79aaa74db310b783e82c5991698926841.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/619d646b5e816ad8e72e1180c5c08cc79aaa74db310b783e82c5991698926841.jpg)

![66eb9933145c893107e81d0aed84cfc174f677a42ca554d9448b1f1103bbb64f.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/66eb9933145c893107e81d0aed84cfc174f677a42ca554d9448b1f1103bbb64f.jpg)

![67b9abedca1a8b8e00163564254ab6cc6d85a54e1a6fbc08e4da5a5bb2175cc6.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/67b9abedca1a8b8e00163564254ab6cc6d85a54e1a6fbc08e4da5a5bb2175cc6.jpg)

![69e8f4fc481ea77f9ddda885b4bad3e3f6f1fc1659e7055ae17a3218eb51bed7.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/69e8f4fc481ea77f9ddda885b4bad3e3f6f1fc1659e7055ae17a3218eb51bed7.jpg)

![7a4f63ef9b19c48e561f328e20ad2a63a3462d1a5c36d51cfab429df45978fc8.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/7a4f63ef9b19c48e561f328e20ad2a63a3462d1a5c36d51cfab429df45978fc8.jpg)

![7f734f7ac8fb1455949ad09f83ab13d7e8aea9e7537b774dec652f5b77cd4b33.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/7f734f7ac8fb1455949ad09f83ab13d7e8aea9e7537b774dec652f5b77cd4b33.jpg)

![9b3e487a454aa0786d5617448d03ed7eb364dbd050275973e65dd5d9e5b2a1d9.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/9b3e487a454aa0786d5617448d03ed7eb364dbd050275973e65dd5d9e5b2a1d9.jpg)

![a09c0388e18a8845f025daaa9f6a0fdc7dab196faad9485fa3d18e1222eb5fa6.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/a09c0388e18a8845f025daaa9f6a0fdc7dab196faad9485fa3d18e1222eb5fa6.jpg)

![a65dc1d902bfc2eb9368999c8ddd2af707089432b9555ab3b6b7c2ad56f4ad15.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/a65dc1d902bfc2eb9368999c8ddd2af707089432b9555ab3b6b7c2ad56f4ad15.jpg)

![ab968f65be7f13f39cb73436e875f2c066c63f3923d04ad7c0458aa05efba7f6.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/ab968f65be7f13f39cb73436e875f2c066c63f3923d04ad7c0458aa05efba7f6.jpg)

![ad0b07786fefe98f0b37e816648bbb03f922f5e54f571040b99e9af61f1117ce.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/ad0b07786fefe98f0b37e816648bbb03f922f5e54f571040b99e9af61f1117ce.jpg)

![be42698111042c2ca885fbcd5df1876b6e1d505794ab281d6bfb613e573559da.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/be42698111042c2ca885fbcd5df1876b6e1d505794ab281d6bfb613e573559da.jpg)

![c09a202242941c196dee88fc1f7eeaf6845bf60c4c610fac6038ab14e326d5d8.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/c09a202242941c196dee88fc1f7eeaf6845bf60c4c610fac6038ab14e326d5d8.jpg)

![c9f64fdbf93177d63454179bc0d54b13373f1ef85599338a08c23ca8a7d83cdf.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/c9f64fdbf93177d63454179bc0d54b13373f1ef85599338a08c23ca8a7d83cdf.jpg)

![d0f33398b25d667243acc530918a7138fa6ecd090514778129c4becd90f1302b.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/d0f33398b25d667243acc530918a7138fa6ecd090514778129c4becd90f1302b.jpg)

![d95053e9caad5ccfbd8ed0be990e2e00607356302b84ed05d02d09454bd7e7f5.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/images/d95053e9caad5ccfbd8ed0be990e2e00607356302b84ed05d02d09454bd7e7f5.jpg)

### Tables

![ae986688f212d8316d62b14412acbd5fc463de35b718eb3e54ea1ffcf5e3c4a8.jpg](../icml_results/1824_Demystifying%20LLM-as-a-Judge_%20Analytically%20Tractable%20Model%20for%20Inference-Time%20Scaling/tables/ae986688f212d8316d62b14412acbd5fc463de35b718eb3e54ea1ffcf5e3c4a8.jpg)

<a id="paper-1802"></a>

## Maximin Relative Improvement: Fair Learning as a Bargaining Problem


### Images

![13e1f3227c023ebc61292b74d9ac08cbb1407d837b6d5af056bd578a099fde76.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/13e1f3227c023ebc61292b74d9ac08cbb1407d837b6d5af056bd578a099fde76.jpg)

![152c3e9764c42022c81803dbe495e1ffc50c7e480db090b86d3fb1a81470ec8c.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/152c3e9764c42022c81803dbe495e1ffc50c7e480db090b86d3fb1a81470ec8c.jpg)

![1c0192f8ae5ab5f7477344d46d8f3ab23723aebcc004f887c0727dc892a26c72.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/1c0192f8ae5ab5f7477344d46d8f3ab23723aebcc004f887c0727dc892a26c72.jpg)

![1f202ab3e6ae03a8eb92248371bd7738f56560b35c10a6c74b9d382c4525e3fe.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/1f202ab3e6ae03a8eb92248371bd7738f56560b35c10a6c74b9d382c4525e3fe.jpg)

![374faf0f1d4af684a9e39bb1974014cf1b4587536d52fc193c6bd0d47202468d.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/374faf0f1d4af684a9e39bb1974014cf1b4587536d52fc193c6bd0d47202468d.jpg)

![4bcacddb72e8a833349a08a1687596c6369429b91ce56c2c7681e4133f5f2f98.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/4bcacddb72e8a833349a08a1687596c6369429b91ce56c2c7681e4133f5f2f98.jpg)

![5b02ae948f7c2aa5664a225930b4bc906b84684c14b0ac333b7eff97e1eca194.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/5b02ae948f7c2aa5664a225930b4bc906b84684c14b0ac333b7eff97e1eca194.jpg)

![725cca2b2876bbacc3b55716aa223489f1365a1ab3db081f344f469c9da8df42.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/725cca2b2876bbacc3b55716aa223489f1365a1ab3db081f344f469c9da8df42.jpg)

![7ae2f708869cff5fe5a86aa92605e0e96aedbe52d813377eed64d67476f04cc5.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/7ae2f708869cff5fe5a86aa92605e0e96aedbe52d813377eed64d67476f04cc5.jpg)

![819f6947ede26dd6fbe18a08f1e883cbc1177ad8c040d95fe1ceb59f5075b5ef.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/819f6947ede26dd6fbe18a08f1e883cbc1177ad8c040d95fe1ceb59f5075b5ef.jpg)

![b9431c0c3e62f6053caf468eb7a9c5b512a576e27910f69094fdff3f0af4e328.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/images/b9431c0c3e62f6053caf468eb7a9c5b512a576e27910f69094fdff3f0af4e328.jpg)

### Tables

![3e5d5aacba6b16183c0764f8efe0e45dd11607055d22a267a94db21849f9cbf7.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/tables/3e5d5aacba6b16183c0764f8efe0e45dd11607055d22a267a94db21849f9cbf7.jpg)

![c64560ed1ec5483594ac447be1e794dfe16ca20789a72635fb2e1768914cd6ea.jpg](../icml_results/1825_Maximin%20Relative%20Improvement_%20Fair%20Learning%20as%20a%20Bargaining%20Problem/tables/c64560ed1ec5483594ac447be1e794dfe16ca20789a72635fb2e1768914cd6ea.jpg)

<a id="paper-1803"></a>

## Weight Decay Improves Language Model Plasticity


### Images

![006e6c341434f82fcd1f10bece63a3763ecb83f0b37df930cb0cadf90fb09d6f.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/006e6c341434f82fcd1f10bece63a3763ecb83f0b37df930cb0cadf90fb09d6f.jpg)

![018af3a91465336a10575fe5ff633ed0aa70b4b70208112cfa462b8779229652.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/018af3a91465336a10575fe5ff633ed0aa70b4b70208112cfa462b8779229652.jpg)

![09df3462befe8916c9ef635cd182b6a37237752f0b786bc89f12ee5e1599be5f.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/09df3462befe8916c9ef635cd182b6a37237752f0b786bc89f12ee5e1599be5f.jpg)

![200730c8ace115cff22edbee885e94256387205bc39f148913b048e044d094bc.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/200730c8ace115cff22edbee885e94256387205bc39f148913b048e044d094bc.jpg)

![29ccbdfe48b29ccb3037442a85fd04861ef6ebc62665212b20897d2d8c8e21c5.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/29ccbdfe48b29ccb3037442a85fd04861ef6ebc62665212b20897d2d8c8e21c5.jpg)

![2ad60f7a500d919209173be1b53e28df280b843409b08f9d74eb9fe8ffc9dea6.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/2ad60f7a500d919209173be1b53e28df280b843409b08f9d74eb9fe8ffc9dea6.jpg)

![2bc2da82233fbb3bf52b0ab72aacbde5249c5b5b8fc0ba5723b8e0dc815fc4e6.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/2bc2da82233fbb3bf52b0ab72aacbde5249c5b5b8fc0ba5723b8e0dc815fc4e6.jpg)

![46ce9386fa1b52f19a153269d9c06101ab2aab0cba97e0675f971fee434d2d72.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/46ce9386fa1b52f19a153269d9c06101ab2aab0cba97e0675f971fee434d2d72.jpg)

![544bfab541b51ceb448649b60b9587bbbc8061985332d5e22c6c003040635e20.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/544bfab541b51ceb448649b60b9587bbbc8061985332d5e22c6c003040635e20.jpg)

![5d1c434280b51ed7746515778c3a473c5522e74fc768cccf24d9ccbae6f4e820.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/5d1c434280b51ed7746515778c3a473c5522e74fc768cccf24d9ccbae6f4e820.jpg)

![5eac704ef09511592c7c79605231abc317292a2f4c7c757467a9fc36538a75d4.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/5eac704ef09511592c7c79605231abc317292a2f4c7c757467a9fc36538a75d4.jpg)

![68a389dfeb0a7991618550b9cc59629af84f6af4d9c0d77606512bd84d5ed02e.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/68a389dfeb0a7991618550b9cc59629af84f6af4d9c0d77606512bd84d5ed02e.jpg)

![696ec608f7f04dc377ede8edb5937278f30e60add480b3bfbbb6d2a842b391d5.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/696ec608f7f04dc377ede8edb5937278f30e60add480b3bfbbb6d2a842b391d5.jpg)

![6b06d9c706502752363e206e32886a2328a626212b0b4b7441ebfb1580b90535.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/6b06d9c706502752363e206e32886a2328a626212b0b4b7441ebfb1580b90535.jpg)

![7717d29d08379d249147dbe848be147dfa4a791f27b3bbdce875d7ce0c8fb48c.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/7717d29d08379d249147dbe848be147dfa4a791f27b3bbdce875d7ce0c8fb48c.jpg)

![7bc4c6b4c875ddc0bd78c8ade5e2851fa7042b71add6fd70b228c1f5551c0d96.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/7bc4c6b4c875ddc0bd78c8ade5e2851fa7042b71add6fd70b228c1f5551c0d96.jpg)

![8fd082216bfb462934bd1d9dc677052494b22b426d59b2bde8db5dacdfdc7fbf.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/8fd082216bfb462934bd1d9dc677052494b22b426d59b2bde8db5dacdfdc7fbf.jpg)

![93c3f711959ffb037e7d140961cb75459efbb478b304f3699dc706dcf157a101.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/93c3f711959ffb037e7d140961cb75459efbb478b304f3699dc706dcf157a101.jpg)

![9786658fc0e6fe9efbdebf5a49ec0b30c6612502d644f9d4273a18002461424b.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/9786658fc0e6fe9efbdebf5a49ec0b30c6612502d644f9d4273a18002461424b.jpg)

![97faea064e3c38fc9264f1cc21e9cbe91bc06dba6a5ad820d4ee445f1e6d8d82.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/97faea064e3c38fc9264f1cc21e9cbe91bc06dba6a5ad820d4ee445f1e6d8d82.jpg)

![a1d119ce514b7549b0e68604852112da2f826caa37006bb3ae5de435a630e6af.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/a1d119ce514b7549b0e68604852112da2f826caa37006bb3ae5de435a630e6af.jpg)

![a9836398e434aaac2694d88b667ad375e1ee086e68be49919ca63c0f3ba8ae59.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/a9836398e434aaac2694d88b667ad375e1ee086e68be49919ca63c0f3ba8ae59.jpg)

![b580741d3ae7715e53362fa4eaedc4061b3db08d21dfa81c015e1bd90cf0f033.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/b580741d3ae7715e53362fa4eaedc4061b3db08d21dfa81c015e1bd90cf0f033.jpg)

![b6bbf212005706e6b6592ded14c143ed7a5b8e17877b015d22a7062c568c79d1.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/b6bbf212005706e6b6592ded14c143ed7a5b8e17877b015d22a7062c568c79d1.jpg)

![ba1af6fe47fb204ad46ea21f02660f1038be9f8cf575d88d9f2c34ec82c2b07b.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/ba1af6fe47fb204ad46ea21f02660f1038be9f8cf575d88d9f2c34ec82c2b07b.jpg)

![dbd0ebdad6cc4c7272596d827f42dc3ce3de8a14a05151bd6fd4c7ff4dbd352e.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/dbd0ebdad6cc4c7272596d827f42dc3ce3de8a14a05151bd6fd4c7ff4dbd352e.jpg)

![dd98d11e05b6b5492ce2182c162d6d352191f531b90b64f65f09d45513c8e68c.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/dd98d11e05b6b5492ce2182c162d6d352191f531b90b64f65f09d45513c8e68c.jpg)

![dfe31b904fd0904abc3a3d5d0a2756283e77f4b0f45da29a23e3b3b2f1241e63.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/dfe31b904fd0904abc3a3d5d0a2756283e77f4b0f45da29a23e3b3b2f1241e63.jpg)

![ed4f19d54aa20e1ba3dedf6d7c51208787e427b0eecf663ce1d38a3abdd451df.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/ed4f19d54aa20e1ba3dedf6d7c51208787e427b0eecf663ce1d38a3abdd451df.jpg)

![fec41d90fe0423e769e0bd96ae363225eee12fb2285594c7368737dbbc3858f1.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/images/fec41d90fe0423e769e0bd96ae363225eee12fb2285594c7368737dbbc3858f1.jpg)

### Tables

![4f2a98bc6e12fb928b5c1e33356e8081d392a80b15959931ebb90af461c18cc1.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/4f2a98bc6e12fb928b5c1e33356e8081d392a80b15959931ebb90af461c18cc1.jpg)

![52b6b02cd55847f36ee6c3ca2f70aa598afa5d36e485a099938ed4f616b5ecba.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/52b6b02cd55847f36ee6c3ca2f70aa598afa5d36e485a099938ed4f616b5ecba.jpg)

![6e06a3f10205a0c29ba627ebaa3c8640d607cda95e443d78feb2e4b3b6c6137a.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/6e06a3f10205a0c29ba627ebaa3c8640d607cda95e443d78feb2e4b3b6c6137a.jpg)

![c115af0c11ce565f51db803b60e8407189bb3a0da704b8e37d2343d52c58791b.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/c115af0c11ce565f51db803b60e8407189bb3a0da704b8e37d2343d52c58791b.jpg)

![dd7ae5dfcbbb4d6a7d68d1a5e22dbbc485502493139c6121c085802c6e733597.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/dd7ae5dfcbbb4d6a7d68d1a5e22dbbc485502493139c6121c085802c6e733597.jpg)

![f343126f4708a7567b20eca9231afb7491c7f8daa1434a03efc76f9af591e6d1.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/f343126f4708a7567b20eca9231afb7491c7f8daa1434a03efc76f9af591e6d1.jpg)

![f6b0aad69f2e525f45d60008e2603bddd855a9b44e020f8788193fe1c5fd9e89.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/f6b0aad69f2e525f45d60008e2603bddd855a9b44e020f8788193fe1c5fd9e89.jpg)

![f7c8f719f51489b8f80d9e3fb68d8127b9a14dcfa480cc6994d8db3d365058dd.jpg](../icml_results/1826_Weight%20Decay%20Improves%20Language%20Model%20Plasticity/tables/f7c8f719f51489b8f80d9e3fb68d8127b9a14dcfa480cc6994d8db3d365058dd.jpg)

<a id="paper-1804"></a>

## Bridging Local–Global Dissonance: Learning from Compressive Measurements for Hyperspectral Reconstruction


### Images

![3caa3469e27adc0d2d5de1e3405aef6b43fb3ab765de8626e628d67c88fefb7f.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/3caa3469e27adc0d2d5de1e3405aef6b43fb3ab765de8626e628d67c88fefb7f.jpg)

![72b758a82eb80195bda95d6b8d84371798c4123045825789d36802a974ab52de.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/72b758a82eb80195bda95d6b8d84371798c4123045825789d36802a974ab52de.jpg)

![8d24a4c7c2a13180e49d79c30ecda4911fb71714215670db86ea393b14967c0d.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/8d24a4c7c2a13180e49d79c30ecda4911fb71714215670db86ea393b14967c0d.jpg)

![c13d1fd42dbdbef42aa9bf4198bbd41a62c823e2237ffe3de5a6e7c3953e2cac.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/c13d1fd42dbdbef42aa9bf4198bbd41a62c823e2237ffe3de5a6e7c3953e2cac.jpg)

![c627aeaf16714f17cc924efb5e8d02f8b4572ab991f6d7519c13d0d15554a287.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/c627aeaf16714f17cc924efb5e8d02f8b4572ab991f6d7519c13d0d15554a287.jpg)

![deaf0f2ae77024e9140d16ea37f7ef3043e88361c2bd02efd8b49c3bae9a5e7e.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/deaf0f2ae77024e9140d16ea37f7ef3043e88361c2bd02efd8b49c3bae9a5e7e.jpg)

![e8910628d65d0be6f6cfeb30654e891965560fb0bf100454326959de00a56a7b.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/images/e8910628d65d0be6f6cfeb30654e891965560fb0bf100454326959de00a56a7b.jpg)

### Tables

![02c856d1b6f28316da2c488f7603cadeb6b060d906ee0be39e2f18d7e2070c0d.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/tables/02c856d1b6f28316da2c488f7603cadeb6b060d906ee0be39e2f18d7e2070c0d.jpg)

![9b27504cd6e7639181420011d84038caf0b457c2715a43c4a2fbf581e330c094.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/tables/9b27504cd6e7639181420011d84038caf0b457c2715a43c4a2fbf581e330c094.jpg)

![b01713543cf63f16d791116057e6df64e4aec6be0603ed2d49b28ce8fd28a9c7.jpg](../icml_results/1827_Bridging%20Local%E2%80%93Global%20Dissonance_%20Learning%20from%20Compressive%20Measurements%20for%20Hyperspectral%20Reconstru/tables/b01713543cf63f16d791116057e6df64e4aec6be0603ed2d49b28ce8fd28a9c7.jpg)

<a id="paper-1805"></a>

## TSFAdv: Frequency-Guided Black-Box Adversarial Attacks on Time Series Forecasting


### Images

![0383ee145f0be9f983c44c51fba093eeaf834552be8b69e0965733a45c0d9117.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/0383ee145f0be9f983c44c51fba093eeaf834552be8b69e0965733a45c0d9117.jpg)

![15cd10244e53ae356a4816b7a5fd748811b067708d6963aa10c319e88850fe48.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/15cd10244e53ae356a4816b7a5fd748811b067708d6963aa10c319e88850fe48.jpg)

![28cbf6576777ef23ac6567878335b7c6345e9fb1adeb64dc07e685d1f04dada5.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/28cbf6576777ef23ac6567878335b7c6345e9fb1adeb64dc07e685d1f04dada5.jpg)

![2986be991a6477ab62d791807a0cf6a581839e0c91ed6bed892e1b3d59dc43eb.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/2986be991a6477ab62d791807a0cf6a581839e0c91ed6bed892e1b3d59dc43eb.jpg)

![298b79f2d5b65e39c71d7dd5f977c2ed7cf414bafb17a461d4d0ff109c97a4b5.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/298b79f2d5b65e39c71d7dd5f977c2ed7cf414bafb17a461d4d0ff109c97a4b5.jpg)

![437c98c060edadc015c07d654debf24a3ed5ff698400658c9a41436a1c6ff4b9.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/437c98c060edadc015c07d654debf24a3ed5ff698400658c9a41436a1c6ff4b9.jpg)

![5adf0cc009ce149d1f4f67b0d3fc727ff7443f0e50506ee64dacc0e153790b5c.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/5adf0cc009ce149d1f4f67b0d3fc727ff7443f0e50506ee64dacc0e153790b5c.jpg)

![b21242225518afea697327604748435c2f322534b4a37432ec85942ec7a0d1d5.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/b21242225518afea697327604748435c2f322534b4a37432ec85942ec7a0d1d5.jpg)

![bb927aafcc41f38598f7770a005b1fc92150c82b5ad4a4b4c0087fff18323001.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/bb927aafcc41f38598f7770a005b1fc92150c82b5ad4a4b4c0087fff18323001.jpg)

![bcfcf59120e68e31463a7351bc4706d4f9831fdd69fbf6b96eab91e919e10607.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/bcfcf59120e68e31463a7351bc4706d4f9831fdd69fbf6b96eab91e919e10607.jpg)

![e778277703da9a30ac347757693b31e49ab427e18ff511beec140cc60815b164.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/images/e778277703da9a30ac347757693b31e49ab427e18ff511beec140cc60815b164.jpg)

### Tables

![1ee03eb6942e9508e60a680705d6bf1943862a8cdd9e71e7ca0cee9a5357f0af.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/1ee03eb6942e9508e60a680705d6bf1943862a8cdd9e71e7ca0cee9a5357f0af.jpg)

![22b90832820006799822d9e41812118d9ba34e11454a7588ac04a842831f63ba.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/22b90832820006799822d9e41812118d9ba34e11454a7588ac04a842831f63ba.jpg)

![2625cbbfac9340d3b478273d5e2745cfe403a0c6e087fc84b1ab656d22f42c2c.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/2625cbbfac9340d3b478273d5e2745cfe403a0c6e087fc84b1ab656d22f42c2c.jpg)

![6149f0aca87b31beff2cbd6f6e37f9abeaba78d77d16e1e89c992142bd1c8ebc.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/6149f0aca87b31beff2cbd6f6e37f9abeaba78d77d16e1e89c992142bd1c8ebc.jpg)

![b4c0ed0b3a7033d731c18fc84a8a0be1b4cc986091c660897d00896795edb622.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/b4c0ed0b3a7033d731c18fc84a8a0be1b4cc986091c660897d00896795edb622.jpg)

![e48a0b9a401e61c704864089705fe9dd2f4030923af574490a04c5d80b0e2827.jpg](../icml_results/1828_TSFAdv_%20Frequency-Guided%20Black-Box%20Adversarial%20Attacks%20on%20Time%20Series%20Forecasting/tables/e48a0b9a401e61c704864089705fe9dd2f4030923af574490a04c5d80b0e2827.jpg)

<a id="paper-1806"></a>

## Revealing Long-context Potential of Attention Heads via Frequency Kernels


### Images

![10292574383854027d84abc1fe9bd4a4242a4a5699a52fc6935be458462ad74e.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/10292574383854027d84abc1fe9bd4a4242a4a5699a52fc6935be458462ad74e.jpg)

![1311053ac954e9eaefaede9e9b54f75398970d4a6e265ffdeb1d4e210f692d9e.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/1311053ac954e9eaefaede9e9b54f75398970d4a6e265ffdeb1d4e210f692d9e.jpg)

![46f4f96b7057cad168ab565b3af9cce3849ffe5449e541d2874a1fa66263fa42.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/46f4f96b7057cad168ab565b3af9cce3849ffe5449e541d2874a1fa66263fa42.jpg)

![485495b135da00055b5fc0851066ae91f67bf936d85bf73bb4105fe2d9c9760c.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/485495b135da00055b5fc0851066ae91f67bf936d85bf73bb4105fe2d9c9760c.jpg)

![69bcdf37f0e68ead9fee036d0633bc57c6a0487084b585fd8a34d49111b3a78d.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/69bcdf37f0e68ead9fee036d0633bc57c6a0487084b585fd8a34d49111b3a78d.jpg)

![71f52bb4914f7900b9ddb66e9096eff5c38e268c8a2daa6bb84d8c2b65564c24.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/71f52bb4914f7900b9ddb66e9096eff5c38e268c8a2daa6bb84d8c2b65564c24.jpg)

![add3655a405d6f2344d04c0c56dd48b4f17130bdd0a9d65ad138090e3ef6e10e.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/add3655a405d6f2344d04c0c56dd48b4f17130bdd0a9d65ad138090e3ef6e10e.jpg)

![b13953f981d7c4f2637a0ec60fa72a3d7bc764545ab46b444b692472e3322372.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/b13953f981d7c4f2637a0ec60fa72a3d7bc764545ab46b444b692472e3322372.jpg)

![f4030bb6714710ad6e010497511941eeef21dd113e1d0c5643831d33531daa50.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/images/f4030bb6714710ad6e010497511941eeef21dd113e1d0c5643831d33531daa50.jpg)

### Tables

![94a5c4e6ab37e70a25b3b9ccdd853f2342614dd700b088cde9dfdec7cfd6285b.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/tables/94a5c4e6ab37e70a25b3b9ccdd853f2342614dd700b088cde9dfdec7cfd6285b.jpg)

![9fb72adbe356fab5d638ffe27bb0b54a939a8798c3a83d689f98c32c253c1e5f.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/tables/9fb72adbe356fab5d638ffe27bb0b54a939a8798c3a83d689f98c32c253c1e5f.jpg)

![c1252aa92a2c956897b178cd38e33070209305e9f74506eab74039ce7fef96cc.jpg](../icml_results/1829_Revealing%20Long-context%20Potential%20of%20Attention%20Heads%20via%20Frequency%20Kernels/tables/c1252aa92a2c956897b178cd38e33070209305e9f74506eab74039ce7fef96cc.jpg)

<a id="paper-1807"></a>

## Learning Permutation-invariant Macroscopic Dynamics


### Images

![09212ba3250ab8585b9493cd2439ff4e0aba4a1276fff090b91375076f83b73b.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/09212ba3250ab8585b9493cd2439ff4e0aba4a1276fff090b91375076f83b73b.jpg)

![48c419a893be033273d4d996adebe33a5549d987c51f128432b5e4a2d1a31e30.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/48c419a893be033273d4d996adebe33a5549d987c51f128432b5e4a2d1a31e30.jpg)

![4a0bbe1fa0ab605eca2ea22d6c90d2dd16dfb08e5cd1bc8328abc0d29233e3a7.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/4a0bbe1fa0ab605eca2ea22d6c90d2dd16dfb08e5cd1bc8328abc0d29233e3a7.jpg)

![5e719ba70c411999a8f195ea1ec9d14362d01bbfbaa68c0d0f9dbc9e20a9a9f7.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/5e719ba70c411999a8f195ea1ec9d14362d01bbfbaa68c0d0f9dbc9e20a9a9f7.jpg)

![666e51ac5fcdd1168533cc5858a29d66c31bcf66831b5d7f0dfbe3609de69e12.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/666e51ac5fcdd1168533cc5858a29d66c31bcf66831b5d7f0dfbe3609de69e12.jpg)

![8b52d0d138a3dd0eec716ee5695e1eb9d012a3dec638acc75e0ccac7f15163ab.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/8b52d0d138a3dd0eec716ee5695e1eb9d012a3dec638acc75e0ccac7f15163ab.jpg)

![96cd8737972360f20f48babe5f2bc93596d7ce69837c7b33a319f5acc77e439e.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/96cd8737972360f20f48babe5f2bc93596d7ce69837c7b33a319f5acc77e439e.jpg)

![c2ae34dfaca850b952f0d68f8e27f29988f24f236cae01321dabecad1d98c1e3.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/c2ae34dfaca850b952f0d68f8e27f29988f24f236cae01321dabecad1d98c1e3.jpg)

![d6d8bb568ca550462e9d0a7cd233a7955d7a9cfc0ac619626ee6b7b42fd63080.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/d6d8bb568ca550462e9d0a7cd233a7955d7a9cfc0ac619626ee6b7b42fd63080.jpg)

![efe773336762b047653c8c192b0853397814a47774d39081384c107bf51433b4.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/images/efe773336762b047653c8c192b0853397814a47774d39081384c107bf51433b4.jpg)

### Tables

![09192393a5ce75dd9b0f60d132f26d19af03c2808c60c1d1a9d5ad4515d7cb3a.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/tables/09192393a5ce75dd9b0f60d132f26d19af03c2808c60c1d1a9d5ad4515d7cb3a.jpg)

![1707fac36ebf577c9e919ae74431fb5b2288b7704586cfa7235b703f24152723.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/tables/1707fac36ebf577c9e919ae74431fb5b2288b7704586cfa7235b703f24152723.jpg)

![b39c984b5d9d1af77b10766cb13dd4f96e2fa51f25f886c533ba581cd17d41c8.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/tables/b39c984b5d9d1af77b10766cb13dd4f96e2fa51f25f886c533ba581cd17d41c8.jpg)

![ea81f71b1dc9daa8bcc6608e56c562477c24f202c75f2108dd8c9d70883d9869.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/tables/ea81f71b1dc9daa8bcc6608e56c562477c24f202c75f2108dd8c9d70883d9869.jpg)

![fde2890f90c9e69a68ed5550c4b332c3ee5bb24cddfb05eaa9bd12f20a99027e.jpg](../icml_results/1830_Learning%20Permutation-invariant%20Macroscopic%20Dynamics/tables/fde2890f90c9e69a68ed5550c4b332c3ee5bb24cddfb05eaa9bd12f20a99027e.jpg)

<a id="paper-1808"></a>

## Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials


### Images

![5334762931437afa7fed260894f840d5a5fd95ed4f0dbcb2cafed8df2c142ebe.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/5334762931437afa7fed260894f840d5a5fd95ed4f0dbcb2cafed8df2c142ebe.jpg)

![60ee516ca9ecc2cdbb1d0f325e43d0b9ccb8683a0de37b1089100248a5e4f4c1.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/60ee516ca9ecc2cdbb1d0f325e43d0b9ccb8683a0de37b1089100248a5e4f4c1.jpg)

![7dcb6154f3a4636d8ee16865563aa54fdadbc3252c394f0a588ffef153d92c9b.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/7dcb6154f3a4636d8ee16865563aa54fdadbc3252c394f0a588ffef153d92c9b.jpg)

![8480489efe6de3c43acc74f90b9adfb56feeca692adbc474f12a251c5ef681ca.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/8480489efe6de3c43acc74f90b9adfb56feeca692adbc474f12a251c5ef681ca.jpg)

![a54f72ccf05b887408535efcda8c1c2f327a2f038d66ff9d06f5517d70529482.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/a54f72ccf05b887408535efcda8c1c2f327a2f038d66ff9d06f5517d70529482.jpg)

![af4f7993be70505b14dd5649d85f44a1bd58ab396a19c8485e3da13d6b4c60b7.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/af4f7993be70505b14dd5649d85f44a1bd58ab396a19c8485e3da13d6b4c60b7.jpg)

![b4d54c23a9d8a62241305e73ac34b0a00cfaacb8dc27db26105bd2143ed5b7c4.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/b4d54c23a9d8a62241305e73ac34b0a00cfaacb8dc27db26105bd2143ed5b7c4.jpg)

![c2846dcff649322625460603688b2217760b04278cd3c0793009a18f03f995c7.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/c2846dcff649322625460603688b2217760b04278cd3c0793009a18f03f995c7.jpg)

![c47b72c443e8f84a67ab592041911c6ed0d6be9f6fd358e36715b2b3cff8a8b0.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/c47b72c443e8f84a67ab592041911c6ed0d6be9f6fd358e36715b2b3cff8a8b0.jpg)

![d343e4a412fd62e0ea00024f7e1448b88c21031237a17f414abf61ba7a9f9c40.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/d343e4a412fd62e0ea00024f7e1448b88c21031237a17f414abf61ba7a9f9c40.jpg)

![fe2cb1dadf395cf53cde8e808b35998f138dbd5d22d904ebcca64e6c4cf776d6.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/images/fe2cb1dadf395cf53cde8e808b35998f138dbd5d22d904ebcca64e6c4cf776d6.jpg)

### Tables

![4ee8561f56a31d83e41bc06b838d026174bceeb3a0c391575e46b16252be1e34.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/4ee8561f56a31d83e41bc06b838d026174bceeb3a0c391575e46b16252be1e34.jpg)

![6ce8db8647a1dd12d1f63800a51312e1968de453354dca41ab891e3dde535604.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/6ce8db8647a1dd12d1f63800a51312e1968de453354dca41ab891e3dde535604.jpg)

![80ab0093fc9a530f6f24e46d5b6af32fa50b28c3e4bb3288dcefb13ebf939418.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/80ab0093fc9a530f6f24e46d5b6af32fa50b28c3e4bb3288dcefb13ebf939418.jpg)

![a65bf6ea58fb542ae178987b5363a095edd66af2837904788869f2e227e6383d.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/a65bf6ea58fb542ae178987b5363a095edd66af2837904788869f2e227e6383d.jpg)

![ae309c7110705ef66617c2f52f2f76ebae3dfde7d9a561b9ba193b504031f141.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/ae309c7110705ef66617c2f52f2f76ebae3dfde7d9a561b9ba193b504031f141.jpg)

![b954d8cd31a1391d3f34a46d2bce102e61ca551259c4a4a5d98ecf8674d84a13.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/b954d8cd31a1391d3f34a46d2bce102e61ca551259c4a4a5d98ecf8674d84a13.jpg)

![ee52f1680a20866a91e27c75919d205c68c3c67c2e5c017723a187324111e1da.jpg](../icml_results/1831_Smooth%20Dynamic%20Cutoffs%20for%20Machine%20Learning%20Interatomic%20Potentials/tables/ee52f1680a20866a91e27c75919d205c68c3c67c2e5c017723a187324111e1da.jpg)

<a id="paper-1809"></a>

## EGG: An Expert-Guided Agent Framework for Kernel Generation


### Images

![2fc75a4f66bd54f39ecec62c8a90350987f6241bfcebfb1de2c3fcdefa48aade.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/2fc75a4f66bd54f39ecec62c8a90350987f6241bfcebfb1de2c3fcdefa48aade.jpg)

![31aa01aba9aaa8e464691933a0f1189ffd459b91dc952c32c3cb98e29dd736c0.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/31aa01aba9aaa8e464691933a0f1189ffd459b91dc952c32c3cb98e29dd736c0.jpg)

![4fdd4c779c3d701e341c0c96783f3fdc595f98035e3bd43b03182de0f44f7f6f.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/4fdd4c779c3d701e341c0c96783f3fdc595f98035e3bd43b03182de0f44f7f6f.jpg)

![52aa2806ba62cc3eac406a2c33d169e5312013b5793ca697b83b993eaa162ccb.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/52aa2806ba62cc3eac406a2c33d169e5312013b5793ca697b83b993eaa162ccb.jpg)

![5a753a9f4485ecc45fa3029c990e8d6980d319d50f1a54307139333e2d1294b9.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/5a753a9f4485ecc45fa3029c990e8d6980d319d50f1a54307139333e2d1294b9.jpg)

![67121e0513688df0a5c2db210dcd29cbbbafc495fbcae6461370b9ab10811b33.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/67121e0513688df0a5c2db210dcd29cbbbafc495fbcae6461370b9ab10811b33.jpg)

![6ac3629dac215543c6a2c2d52a5676b35f3671902bf71c2f8dec61c8a5d0d5b5.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/6ac3629dac215543c6a2c2d52a5676b35f3671902bf71c2f8dec61c8a5d0d5b5.jpg)

![d471a0384204059c9eb6e55d40f73ed093c881e533ad76653482e0d1021e7fca.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/images/d471a0384204059c9eb6e55d40f73ed093c881e533ad76653482e0d1021e7fca.jpg)

### Tables

![2254ef54487da197c0c43a88524b3d134d01c91fc259d52292d3ce8468efcc91.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/2254ef54487da197c0c43a88524b3d134d01c91fc259d52292d3ce8468efcc91.jpg)

![27475321faa162880799c554e78bb96a034d92f6d4d16c57b1c4c91c30f433cd.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/27475321faa162880799c554e78bb96a034d92f6d4d16c57b1c4c91c30f433cd.jpg)

![3fdba0617398009ab8df15e73fd2d0a020b9f6082125d07ae1ad42443dfedc74.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/3fdba0617398009ab8df15e73fd2d0a020b9f6082125d07ae1ad42443dfedc74.jpg)

![424c73ef426015784f8f68b3fd3e6a00618963aa65ebaae89da55a12f078804e.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/424c73ef426015784f8f68b3fd3e6a00618963aa65ebaae89da55a12f078804e.jpg)

![430a02a30cabd83dc2c6cfa9a031ef2f722834581dcb49c3a3530459fddf2d0c.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/430a02a30cabd83dc2c6cfa9a031ef2f722834581dcb49c3a3530459fddf2d0c.jpg)

![4cc8fe7f502f74277ac919ab38b13ec981cba8c059c42ceeb6b96251010bc919.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/4cc8fe7f502f74277ac919ab38b13ec981cba8c059c42ceeb6b96251010bc919.jpg)

![6591436d72e7c3afa0b8ebd3455e1949e4fca316b5972aa8ce6dfd02abb73a8d.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/6591436d72e7c3afa0b8ebd3455e1949e4fca316b5972aa8ce6dfd02abb73a8d.jpg)

![7566f999757c96b889767e0ddb1f0747af3fb2002490d4a606b19c34d279ba65.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/7566f999757c96b889767e0ddb1f0747af3fb2002490d4a606b19c34d279ba65.jpg)

![8f7a9d9f6caf92571158772aeb6cd39b0e282f8dcae2be8f0aa0f5967397f557.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/8f7a9d9f6caf92571158772aeb6cd39b0e282f8dcae2be8f0aa0f5967397f557.jpg)

![9169478228548bbf6e09b329956e8908aac9ec8994b7595caa9a7fb73fc14251.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/9169478228548bbf6e09b329956e8908aac9ec8994b7595caa9a7fb73fc14251.jpg)

![ba70bfea9de5dd57dcc8faf7ada74f9ec76b641f893e6e594a445eb4a09fac4c.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/ba70bfea9de5dd57dcc8faf7ada74f9ec76b641f893e6e594a445eb4a09fac4c.jpg)

![c92f5dca49f57c6fefb3728ad0650ab895b8d484c9879d14b07be1f8badad05e.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/c92f5dca49f57c6fefb3728ad0650ab895b8d484c9879d14b07be1f8badad05e.jpg)

![cd088afd6acb9c4e56791d6f5906fb21cd5ccccbb75c8fb9ddba860dcef7c15d.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/cd088afd6acb9c4e56791d6f5906fb21cd5ccccbb75c8fb9ddba860dcef7c15d.jpg)

![d864d1d944f30371e9e3a12b0977f9f51f0391011a5125deb3719ba19ce41b37.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/d864d1d944f30371e9e3a12b0977f9f51f0391011a5125deb3719ba19ce41b37.jpg)

![e406fda1bcc60344411bf1e13a3a82d98ed3f1ac2e57ebf7fa12fbcb9d3f2d7d.jpg](../icml_results/1832_EGG_%20An%20Expert-Guided%20Agent%20Framework%20for%20Kernel%20Generation/tables/e406fda1bcc60344411bf1e13a3a82d98ed3f1ac2e57ebf7fa12fbcb9d3f2d7d.jpg)

<a id="paper-1810"></a>

## Transferable Reinforcement Learning via Probabilistic Latent Embeddings and Dynamic Policy Adaptation for Sim-to-Real Deployment


### Images

![1576fc9fa675d1f7e5bfd7b733cd69a08ea42b22a4b10bedab046b5c6fe9dfbc.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/1576fc9fa675d1f7e5bfd7b733cd69a08ea42b22a4b10bedab046b5c6fe9dfbc.jpg)

![5145fae93b6e105df4b9389365927e077718e841d5011c3867ae81aee0912f8a.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/5145fae93b6e105df4b9389365927e077718e841d5011c3867ae81aee0912f8a.jpg)

![804861767293c26542f5dac72886b2e8ef780d5ef0df8b588d84cb246737a17c.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/804861767293c26542f5dac72886b2e8ef780d5ef0df8b588d84cb246737a17c.jpg)

![8636b304ef4e0026d8e7d6514b201009c6717b5525225e82e7ce77915859e3a7.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/8636b304ef4e0026d8e7d6514b201009c6717b5525225e82e7ce77915859e3a7.jpg)

![b5d1a832fe4f4cc39880619efd4893f1fb10739d0aac68d754baf33d3182294c.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/b5d1a832fe4f4cc39880619efd4893f1fb10739d0aac68d754baf33d3182294c.jpg)

![c8fe16d3341bf567bc8adde93ce1b410baa538db14235715d5596e8f27c77865.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/c8fe16d3341bf567bc8adde93ce1b410baa538db14235715d5596e8f27c77865.jpg)

![f3b4fae39317fded3bdaae92ccc8c6970921436765705403dd84ad45f8fe5434.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/images/f3b4fae39317fded3bdaae92ccc8c6970921436765705403dd84ad45f8fe5434.jpg)

### Tables

![2291fc2029042b8362c54621456276a06181c6fddee5c6d6008ba0a58478dd14.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/2291fc2029042b8362c54621456276a06181c6fddee5c6d6008ba0a58478dd14.jpg)

![3d444e0d3ee28e8c335f6b85199131b9ad0bf79c8aa43d0594b5bf61ab106e19.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/3d444e0d3ee28e8c335f6b85199131b9ad0bf79c8aa43d0594b5bf61ab106e19.jpg)

![4a030dbbd9ce84c40e1de99b05461367c4081fe4e6361bf526d2404764f2566e.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/4a030dbbd9ce84c40e1de99b05461367c4081fe4e6361bf526d2404764f2566e.jpg)

![87133119a0bc1452041ba7a1d7655c4253e8b76255aa351f7a4a1e1a17c0718c.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/87133119a0bc1452041ba7a1d7655c4253e8b76255aa351f7a4a1e1a17c0718c.jpg)

![a0969febb38407b5080d4e25de8df281e5c04f2bd6a791c0bd682bc208188c3c.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/a0969febb38407b5080d4e25de8df281e5c04f2bd6a791c0bd682bc208188c3c.jpg)

![cd0c23bdd61ef58b4646e99fde2c53e1d8b2d27cfe2be63202bcf737403a5439.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/cd0c23bdd61ef58b4646e99fde2c53e1d8b2d27cfe2be63202bcf737403a5439.jpg)

![d4153ed86a2bbd14088707dbbd3ca41d2fdeee1955d1f0e6f0a935085b311aad.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/d4153ed86a2bbd14088707dbbd3ca41d2fdeee1955d1f0e6f0a935085b311aad.jpg)

![e55abcc17fb8a5d8cd21faf814fdc69f2f02e376a0d6d61e56f828ec1ff3b0af.jpg](../icml_results/1833_Transferable%20Reinforcement%20Learning%20via%20Probabilistic%20Latent%20Embeddings%20and%20Dynamic%20Policy%20Adaptatio/tables/e55abcc17fb8a5d8cd21faf814fdc69f2f02e376a0d6d61e56f828ec1ff3b0af.jpg)

<a id="paper-1811"></a>

## Expert-level Leaf Cell Layout Generation via Preference-Optimized LLM


### Images

![14b82bb1b0d14391bf0200e7d66d9b394fb4d40c1f175441d9fce93bb7bd7570.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/14b82bb1b0d14391bf0200e7d66d9b394fb4d40c1f175441d9fce93bb7bd7570.jpg)

![1ec8b4ec9863ba966e0b7aebbe33edd8bbb2f93e33bce398055fcc41678ee1f7.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/1ec8b4ec9863ba966e0b7aebbe33edd8bbb2f93e33bce398055fcc41678ee1f7.jpg)

![23ec4604e9896472b46504a52d1b0bdbd3e81221ca1c3c9428eecbdb1ad0b447.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/23ec4604e9896472b46504a52d1b0bdbd3e81221ca1c3c9428eecbdb1ad0b447.jpg)

![64b80b4d0aa3817f578f44a9584798e4211cadd7ddd0d3fa1807a17c2158d86e.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/64b80b4d0aa3817f578f44a9584798e4211cadd7ddd0d3fa1807a17c2158d86e.jpg)

![8f150057294bfbdd4da8acd84f18bcce71536e6201df7b4e756ab0718c7ef8a4.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/8f150057294bfbdd4da8acd84f18bcce71536e6201df7b4e756ab0718c7ef8a4.jpg)

![aca7f0e12173f3ed05fd2f008958f8a65914eb8712b340fd9ee20d8d9ff7852e.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/aca7f0e12173f3ed05fd2f008958f8a65914eb8712b340fd9ee20d8d9ff7852e.jpg)

![b7dac457414a42fe9167106509f37f2af4400484e0aaa30c1e1d082a25255016.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/images/b7dac457414a42fe9167106509f37f2af4400484e0aaa30c1e1d082a25255016.jpg)

### Tables

![9bae99b174239bb70a46329670f6fb759b49d2a9a82c5502eeb3a2275c22434a.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/9bae99b174239bb70a46329670f6fb759b49d2a9a82c5502eeb3a2275c22434a.jpg)

![aeb5a76c452daaef693af04f623f173b951068d9a708e1cc1506c58b4212dc38.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/aeb5a76c452daaef693af04f623f173b951068d9a708e1cc1506c58b4212dc38.jpg)

![b66ce188036ebe102aea20728375eefc358d1fa15ec0af9b0febbe60fa9fbd9f.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/b66ce188036ebe102aea20728375eefc358d1fa15ec0af9b0febbe60fa9fbd9f.jpg)

![bf001d2b7a343cb857759d015fde2e7cb17aec673d95c519bd43e5e3f5cb5285.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/bf001d2b7a343cb857759d015fde2e7cb17aec673d95c519bd43e5e3f5cb5285.jpg)

![de72c17c19b068ce919fc998705a40a6df14c58a142a73eb8cbd964df6faf49c.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/de72c17c19b068ce919fc998705a40a6df14c58a142a73eb8cbd964df6faf49c.jpg)

![e6267cd39954b0224bb85afec54393ff171b323fa8a88753136390cc29f95e55.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/e6267cd39954b0224bb85afec54393ff171b323fa8a88753136390cc29f95e55.jpg)

![f910889176399816bd4457b76ef7bf73030c2f15a1cecfaaa08fa3f75e0bf28a.jpg](../icml_results/1834_Expert-level%20Leaf%20Cell%20Layout%20Generation%20via%20Preference-Optimized%20LLM/tables/f910889176399816bd4457b76ef7bf73030c2f15a1cecfaaa08fa3f75e0bf28a.jpg)

<a id="paper-1812"></a>

## PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding


### Images

![20e5f61c82d1196a54737c837370f69b5695eeec09a7668a33d31927f0b5af2b.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/20e5f61c82d1196a54737c837370f69b5695eeec09a7668a33d31927f0b5af2b.jpg)

![751d572d5b24747d6e15c11fad8d254e6d623da47b4c8e6edb70f1a70d125e39.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/751d572d5b24747d6e15c11fad8d254e6d623da47b4c8e6edb70f1a70d125e39.jpg)

![95116d3c83560cd7bd41ddfb6108a388a4e504c3058a933770b5031373b9ea57.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/95116d3c83560cd7bd41ddfb6108a388a4e504c3058a933770b5031373b9ea57.jpg)

![a30df1683b229afb727772fa693a5af02827d9ae5f81b190a228941e68014ba2.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/a30df1683b229afb727772fa693a5af02827d9ae5f81b190a228941e68014ba2.jpg)

![c4cf220b695653b2fbf271012e388aa17a5a1f2741023d88cfe465817e3fffad.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/c4cf220b695653b2fbf271012e388aa17a5a1f2741023d88cfe465817e3fffad.jpg)

![c98fa7d8dde21a39a1cc3b435a72c8cfc568aca3627636bcb552dc0df807593c.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/images/c98fa7d8dde21a39a1cc3b435a72c8cfc568aca3627636bcb552dc0df807593c.jpg)

### Tables

![18a923ed87c375a2f66532ae1ff32e12f5c7af33add4e82d5468e9c7337e4e26.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/18a923ed87c375a2f66532ae1ff32e12f5c7af33add4e82d5468e9c7337e4e26.jpg)

![6cc0ad6749cae7945ff270cd9cb01940b33ea8749ad1408afe768ae27cd888b0.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/6cc0ad6749cae7945ff270cd9cb01940b33ea8749ad1408afe768ae27cd888b0.jpg)

![7b8e56cc47492af3cf46ada241384ceadf483bc0c4dbdd8d72dd792f476a4c5b.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/7b8e56cc47492af3cf46ada241384ceadf483bc0c4dbdd8d72dd792f476a4c5b.jpg)

![a139337b8bcbecd5adeae0fcbb99aefd5e28410c542c063ace9270ee8c0ead8c.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/a139337b8bcbecd5adeae0fcbb99aefd5e28410c542c063ace9270ee8c0ead8c.jpg)

![bd648c34f49b00120a149ff344c3195e79e062d5499bb0049fd68b20941f942b.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/bd648c34f49b00120a149ff344c3195e79e062d5499bb0049fd68b20941f942b.jpg)

![bf3b7172790188098747734069816af585af2e62d68158de3f7b5113a7859970.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/bf3b7172790188098747734069816af585af2e62d68158de3f7b5113a7859970.jpg)

![dd6b76cf316622fc4bc9b6f30dc51ecc6cf35c8c90764b54d8ebca8534685222.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/dd6b76cf316622fc4bc9b6f30dc51ecc6cf35c8c90764b54d8ebca8534685222.jpg)

![e5e891a0b9116fad52e2b751d80b7bd94f7e18653bcb461533adb4234e99f6f3.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/e5e891a0b9116fad52e2b751d80b7bd94f7e18653bcb461533adb4234e99f6f3.jpg)

![f3e8a5162bdd21c27c4b692f239ab554104f98b7208b0faf536cf6f53a8b7bdc.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/f3e8a5162bdd21c27c4b692f239ab554104f98b7208b0faf536cf6f53a8b7bdc.jpg)

![f6d9a4dbcccbdb4b034658386e065f5f9b0b841c5941c4eefcdc87c7f5afcc1c.jpg](../icml_results/1835_PipeSD_%20An%20Efficient%20Cloud-Edge%20Collaborative%20Pipeline%20Inference%20Framework%20with%20Speculative%20Decoding/tables/f6d9a4dbcccbdb4b034658386e065f5f9b0b841c5941c4eefcdc87c7f5afcc1c.jpg)

<a id="paper-1813"></a>

## Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them


### Images

![06fb63c692a3fad7295713aa5d3d17af387a898b9eb709ce939f12711d09ed95.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/06fb63c692a3fad7295713aa5d3d17af387a898b9eb709ce939f12711d09ed95.jpg)

![1b349e44feafeb12cbdf5ecb118a7541f0494203ef807930226f49d9618c8d25.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/1b349e44feafeb12cbdf5ecb118a7541f0494203ef807930226f49d9618c8d25.jpg)

![1d71b5a99a5d21ea0524f0b0583d19a6f0d8c8c7976b5279666805337dddb443.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/1d71b5a99a5d21ea0524f0b0583d19a6f0d8c8c7976b5279666805337dddb443.jpg)

![1ecfd94e964e230490f31f0ac9b663c2bfd4b50578e137bfc23e9f27a42af2e0.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/1ecfd94e964e230490f31f0ac9b663c2bfd4b50578e137bfc23e9f27a42af2e0.jpg)

![3135de2a8cddbbe3f78b4b776de4dca080064e9372773c0c6d6afe953037ba0e.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/3135de2a8cddbbe3f78b4b776de4dca080064e9372773c0c6d6afe953037ba0e.jpg)

![36cc98519eae0f470e2c8e0c74c0bec7c4636bdbe7c3fabe4f9a022878acee60.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/36cc98519eae0f470e2c8e0c74c0bec7c4636bdbe7c3fabe4f9a022878acee60.jpg)

![497f372b006c4cf88b673eb39d8728fa4d34147836bec33a68d7392955e3ee84.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/497f372b006c4cf88b673eb39d8728fa4d34147836bec33a68d7392955e3ee84.jpg)

![4a6a48c97e5eaf7b7a62eb1a049a31f82fc5038f37f8b03a82ffdccbefae7c2d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/4a6a48c97e5eaf7b7a62eb1a049a31f82fc5038f37f8b03a82ffdccbefae7c2d.jpg)

![4cf3ac204e58ef377aff4a1f1826b1960ffe34873af91bd7f8b3f3976ff75a2d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/4cf3ac204e58ef377aff4a1f1826b1960ffe34873af91bd7f8b3f3976ff75a2d.jpg)

![50f3a7e45b6508647429ecfecbc748f7fa315701af17485787a121b0d79fc5f5.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/50f3a7e45b6508647429ecfecbc748f7fa315701af17485787a121b0d79fc5f5.jpg)

![7ddcafa7c2006798683ab58036c58a84ebb4ab7e26d5de633f5e36c64f6c70ef.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/7ddcafa7c2006798683ab58036c58a84ebb4ab7e26d5de633f5e36c64f6c70ef.jpg)

![872755faa83e06afd4cd76aa03fe7ad6517965416c3aac01544c543eb5cc814d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/872755faa83e06afd4cd76aa03fe7ad6517965416c3aac01544c543eb5cc814d.jpg)

![873497f1591f22f816a8d271ed1e1c14f34696f9d78242dda176181b567e3dc9.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/873497f1591f22f816a8d271ed1e1c14f34696f9d78242dda176181b567e3dc9.jpg)

![8a378d86d6a927a74a572d6dff67f1fd398cf3372f348fa87f899985b343a967.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/8a378d86d6a927a74a572d6dff67f1fd398cf3372f348fa87f899985b343a967.jpg)

![8c911cc15ee1f723efa05ca2594363eda6e2d53312cfee22ab2e909e66755fa3.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/8c911cc15ee1f723efa05ca2594363eda6e2d53312cfee22ab2e909e66755fa3.jpg)

![9428902327bae97446ec9afcdd812eccaa8b2b38641bbe62593f855c413edf2e.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/9428902327bae97446ec9afcdd812eccaa8b2b38641bbe62593f855c413edf2e.jpg)

![99f149eccaa603ca5c6197294e5bf4ee1ea8dfc81bd4900f2466b5e87c285c6a.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/99f149eccaa603ca5c6197294e5bf4ee1ea8dfc81bd4900f2466b5e87c285c6a.jpg)

![9a764e581522440812cebdf2de5d8138cdc3d1fba7609984323ba3fb19724bfc.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/9a764e581522440812cebdf2de5d8138cdc3d1fba7609984323ba3fb19724bfc.jpg)

![9e322b57eb6b7eeceb8cbdb69cd5ac4320476c6e6a26e95540029750a17524c6.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/9e322b57eb6b7eeceb8cbdb69cd5ac4320476c6e6a26e95540029750a17524c6.jpg)

![a0c9f6843dfaf16fb1537c5cb509a46d086eefdd476a46ce491302fdba9ca478.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/a0c9f6843dfaf16fb1537c5cb509a46d086eefdd476a46ce491302fdba9ca478.jpg)

![a1d9b9090c9fa80f7ae72177aa5e58b46d60cdab880128360ef91ae68b93887d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/a1d9b9090c9fa80f7ae72177aa5e58b46d60cdab880128360ef91ae68b93887d.jpg)

![adc6d106d889c51e311ec2f08ed298412fe117606167b5bbfbf200b0a644e9fb.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/adc6d106d889c51e311ec2f08ed298412fe117606167b5bbfbf200b0a644e9fb.jpg)

![add05d8a01396085a8852db4706197e7a3a717ae829004fb6ed2741b2fc02126.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/add05d8a01396085a8852db4706197e7a3a717ae829004fb6ed2741b2fc02126.jpg)

![b30a1c17440bbaf1ae66e9e578cb67e9f56f6b7583618688745c42924c0a6100.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/b30a1c17440bbaf1ae66e9e578cb67e9f56f6b7583618688745c42924c0a6100.jpg)

![b3dff6fc232215496959510de04dafcdb02e2e389147e0faf1f1c1d657f5b0d3.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/b3dff6fc232215496959510de04dafcdb02e2e389147e0faf1f1c1d657f5b0d3.jpg)

![c0973f2f83894f4f0076dd12417766184d8721584b188b030c4ecd5dc307f10d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/c0973f2f83894f4f0076dd12417766184d8721584b188b030c4ecd5dc307f10d.jpg)

![c759e7247afaf512325038f8c55055625ca0c6bf6be2862a9b70c132f3af5947.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/c759e7247afaf512325038f8c55055625ca0c6bf6be2862a9b70c132f3af5947.jpg)

![ccabe2209259c501a2a5d044f33e687398b119f14328dfbac8a38d5d19282286.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/ccabe2209259c501a2a5d044f33e687398b119f14328dfbac8a38d5d19282286.jpg)

![cd0fcf6dd8ef0a1f6168c167f14c9ebc77682b8f45fab9c876140cb2c4991e17.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/cd0fcf6dd8ef0a1f6168c167f14c9ebc77682b8f45fab9c876140cb2c4991e17.jpg)

![d0a92e686d5dd54d504202913f180cac0b14069a276799ee8f737683e1846107.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/d0a92e686d5dd54d504202913f180cac0b14069a276799ee8f737683e1846107.jpg)

![dac97ae2e331533f9fc202b81e7d7e5fe9ae303dd0cab782a04217d2fc07c79c.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/dac97ae2e331533f9fc202b81e7d7e5fe9ae303dd0cab782a04217d2fc07c79c.jpg)

![dae25f120ac6ae86682619ad43948c896ead366af4dff91b62fc9ceb36206de5.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/dae25f120ac6ae86682619ad43948c896ead366af4dff91b62fc9ceb36206de5.jpg)

![de506d9c1c124616c6f5eb829067dea978c04a4dd15ed8fd982211a35de9b836.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/de506d9c1c124616c6f5eb829067dea978c04a4dd15ed8fd982211a35de9b836.jpg)

![e265f757a238e709ceb767708ddb8a779ec5459ad4f86b4aca73acf93f5f063d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/e265f757a238e709ceb767708ddb8a779ec5459ad4f86b4aca73acf93f5f063d.jpg)

![f555200b227f12c2600e660d6fdb9d8b6fb1767a126195acf7681c85f1736b12.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/f555200b227f12c2600e660d6fdb9d8b6fb1767a126195acf7681c85f1736b12.jpg)

![f9a5fb6b168b384166a3ec3b0c36bd4b8a596e3efafbfb590bb0270b53f12942.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/f9a5fb6b168b384166a3ec3b0c36bd4b8a596e3efafbfb590bb0270b53f12942.jpg)

![fa22aee5f64db4da237d0c2ab931355a8920a63f9a40088295f523cec1e447cc.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/fa22aee5f64db4da237d0c2ab931355a8920a63f9a40088295f523cec1e447cc.jpg)

![fbf5e6cddad24d8c4114a905134921623fb59595dc9954b34f628978ddfb4a47.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/images/fbf5e6cddad24d8c4114a905134921623fb59595dc9954b34f628978ddfb4a47.jpg)

### Tables

![193bca142545e855530306f0ac0291ec23c5b8875152cde498b989fb7e7068a3.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/193bca142545e855530306f0ac0291ec23c5b8875152cde498b989fb7e7068a3.jpg)

![421a38df7077e0748584b014ffbd4f154986adfe195022ba13402577166a9dd3.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/421a38df7077e0748584b014ffbd4f154986adfe195022ba13402577166a9dd3.jpg)

![4889b11d111e832717ac0871529632420fa8ada613562c96b526cb4e8f8f7832.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/4889b11d111e832717ac0871529632420fa8ada613562c96b526cb4e8f8f7832.jpg)

![564ca0e15e083c5cb694ecec3982318e794b13d5ddb4fd72b38f93acda05be2b.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/564ca0e15e083c5cb694ecec3982318e794b13d5ddb4fd72b38f93acda05be2b.jpg)

![5fb874b9d11dd15390ed42c9da5a71cac707680c8aac7b68af617664619e5027.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/5fb874b9d11dd15390ed42c9da5a71cac707680c8aac7b68af617664619e5027.jpg)

![7b76f712762a8191f75e9c2ff3baa01ddf4bd70217b99cd8a4a824071faa53e4.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/7b76f712762a8191f75e9c2ff3baa01ddf4bd70217b99cd8a4a824071faa53e4.jpg)

![7db3438d6ec25b6b07606da34afb0d88dc4cb0939a5103cc7f6002d3e1ea88e6.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/7db3438d6ec25b6b07606da34afb0d88dc4cb0939a5103cc7f6002d3e1ea88e6.jpg)

![850a42440caf9f59460d2b665408da6c57dc12db4017e9f08f0fcc1c40bf9ec0.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/850a42440caf9f59460d2b665408da6c57dc12db4017e9f08f0fcc1c40bf9ec0.jpg)

![9807b1fc0b72e70c25b82b0afc136f3201ffb16d03169af15f57914f91c4461f.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/9807b1fc0b72e70c25b82b0afc136f3201ffb16d03169af15f57914f91c4461f.jpg)

![a6b2c122bc4d4bdfa7a15ce14731a9e86dcc6a78f290bca19e532120ae781c4d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/a6b2c122bc4d4bdfa7a15ce14731a9e86dcc6a78f290bca19e532120ae781c4d.jpg)

![ad2ce21ef218e54de4999571374fccacf657d3fe9c0e15d3383231592f048969.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/ad2ce21ef218e54de4999571374fccacf657d3fe9c0e15d3383231592f048969.jpg)

![ae9538a9f2361c045b456a598e5d959b3ed459153d87a3ed0eee5529b3a773e8.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/ae9538a9f2361c045b456a598e5d959b3ed459153d87a3ed0eee5529b3a773e8.jpg)

![b2c360e8973f92f109333416fb1abf1c2f05a3d2b6c1a0613a03f36d745eee67.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/b2c360e8973f92f109333416fb1abf1c2f05a3d2b6c1a0613a03f36d745eee67.jpg)

![bc58a6ecd11b92150c3a548b97166c403fef15bf4dbf1dfa4ecf0a0e110a33c6.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/bc58a6ecd11b92150c3a548b97166c403fef15bf4dbf1dfa4ecf0a0e110a33c6.jpg)

![ce786b54d9a95329bb15ffa600b543471c4d98ed57a4454821786370be1e9cf6.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/ce786b54d9a95329bb15ffa600b543471c4d98ed57a4454821786370be1e9cf6.jpg)

![d7002e9534eba2c33cb2f2fedf06e62cef56adac162ed3cc3a32c9d5e15c3727.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/d7002e9534eba2c33cb2f2fedf06e62cef56adac162ed3cc3a32c9d5e15c3727.jpg)

![ec50e8cffe6b8d227abea2b013e9c00ca3656242bfe4af7b7926e3b9fdd4bf8d.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/ec50e8cffe6b8d227abea2b013e9c00ca3656242bfe4af7b7926e3b9fdd4bf8d.jpg)

![fbd03a10d4e823610eb706beb5ad1057b30e7c87d496f946ed4996ede43adb20.jpg](../icml_results/1836_Physics%20in%202-Steps_%20Locking%20Motion%20Priors%20Before%20Visual%20Refinement%20Erases%20Them/tables/fbd03a10d4e823610eb706beb5ad1057b30e7c87d496f946ed4996ede43adb20.jpg)

<a id="paper-1814"></a>

## DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation


### Images

![13212c2a0628543515557d73fd0af50fa1a8cb8b98a599703feb75206dbf36b7.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/13212c2a0628543515557d73fd0af50fa1a8cb8b98a599703feb75206dbf36b7.jpg)

![3c94bad8f5cf6d918ac34da43414d1a67ee16c19b49c985b377ce35beba8c080.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/3c94bad8f5cf6d918ac34da43414d1a67ee16c19b49c985b377ce35beba8c080.jpg)

![50f96d1cb08b3b81ad227a627b0bceceec2fbdb5ec29d242418201dbf05b2a71.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/50f96d1cb08b3b81ad227a627b0bceceec2fbdb5ec29d242418201dbf05b2a71.jpg)

![8a49ea427fc3c2ca959896f258af5163bd263156a24438f685a2523c5f02bd26.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/8a49ea427fc3c2ca959896f258af5163bd263156a24438f685a2523c5f02bd26.jpg)

![9b12cfbfb14831ed3c79c197e0593c009ec01f25e90930e90540610893e32b79.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/9b12cfbfb14831ed3c79c197e0593c009ec01f25e90930e90540610893e32b79.jpg)

![c4115788c2b6fa39e2a607b8affef8c754a30cbfb14b4ff558d2bc15336b0fdd.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/images/c4115788c2b6fa39e2a607b8affef8c754a30cbfb14b4ff558d2bc15336b0fdd.jpg)

### Tables

![11369af5e855bee6bf98700515567c48fd9ab6ade0c22b0b55b88b2cb2d304d3.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/11369af5e855bee6bf98700515567c48fd9ab6ade0c22b0b55b88b2cb2d304d3.jpg)

![16029d6eda36162f1c227b5f773d85df121823f21c3d73e75c18e36ad9fd8bb2.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/16029d6eda36162f1c227b5f773d85df121823f21c3d73e75c18e36ad9fd8bb2.jpg)

![2eb4a656662913a8450fe5f8800cb7fa9adb56607467314b9f6db6103aa0554b.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/2eb4a656662913a8450fe5f8800cb7fa9adb56607467314b9f6db6103aa0554b.jpg)

![3573a30798179b961ee9f67f72e6706616199d51fa7b78b3b144fea57511d9de.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/3573a30798179b961ee9f67f72e6706616199d51fa7b78b3b144fea57511d9de.jpg)

![377a825c8a56ddc31af0c4b6494d02f7f4ab691959cb604ec5842d9f76eb5e5c.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/377a825c8a56ddc31af0c4b6494d02f7f4ab691959cb604ec5842d9f76eb5e5c.jpg)

![404c2e9893223aae31b08881070e2b23b92b7a9d098dd8fd39889595b0d6c08e.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/404c2e9893223aae31b08881070e2b23b92b7a9d098dd8fd39889595b0d6c08e.jpg)

![4334f8d4ece1fcfe9a3df3780055b81c22f6f0b435b59d9e08fd7b7766eb7dee.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/4334f8d4ece1fcfe9a3df3780055b81c22f6f0b435b59d9e08fd7b7766eb7dee.jpg)

![48f1da22b6101d739376bb6a94e07921f73175a26a654e6c447a211389064f94.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/48f1da22b6101d739376bb6a94e07921f73175a26a654e6c447a211389064f94.jpg)

![4c2bc3019ce6dde46f358ec27f05ad9e623d7da3fa3c73ea7e4571524d0ccaca.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/4c2bc3019ce6dde46f358ec27f05ad9e623d7da3fa3c73ea7e4571524d0ccaca.jpg)

![5526b7b0fa139718e5f06b5f3bcf4e3cb24c57613ec6c0b85e1780f2e0e68fce.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/5526b7b0fa139718e5f06b5f3bcf4e3cb24c57613ec6c0b85e1780f2e0e68fce.jpg)

![5a3bd8ba77f8192cde520e4d63dcf3830c1198b0cb621b9ff6dbee320f4e9cde.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/5a3bd8ba77f8192cde520e4d63dcf3830c1198b0cb621b9ff6dbee320f4e9cde.jpg)

![5b123dc4ff85f6a15b6d94d268502c9bc42f65e28990ae8f0e2a6225e6ad2568.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/5b123dc4ff85f6a15b6d94d268502c9bc42f65e28990ae8f0e2a6225e6ad2568.jpg)

![6aacdc8603101d3815c516806351a861b31aa8cd3e2a721926d258e9133da6fe.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/6aacdc8603101d3815c516806351a861b31aa8cd3e2a721926d258e9133da6fe.jpg)

![78150242d5fee33c7bc48f7ef39f0d9e5b16662977dc8456d084cd021f4c5975.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/78150242d5fee33c7bc48f7ef39f0d9e5b16662977dc8456d084cd021f4c5975.jpg)

![9d2b1dd2a5f9e628fd8ba374b8f724f3a1d26ecc2347482ebc8aea3218094744.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/9d2b1dd2a5f9e628fd8ba374b8f724f3a1d26ecc2347482ebc8aea3218094744.jpg)

![9ddf9e6ad3706874a55a2f4532ad1d7011ddc6955b8a0eec295c751a555380c9.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/9ddf9e6ad3706874a55a2f4532ad1d7011ddc6955b8a0eec295c751a555380c9.jpg)

![a6a40e2c9bfbc072857e9af821138f0b4ceca591b13a58117bd8228f8610cad2.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/a6a40e2c9bfbc072857e9af821138f0b4ceca591b13a58117bd8228f8610cad2.jpg)

![af3cff44a8e15bbf604584bc6fbb3a0a3975f4c92c71618d8708a1234b8fe21a.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/af3cff44a8e15bbf604584bc6fbb3a0a3975f4c92c71618d8708a1234b8fe21a.jpg)

![b084a8c496608cf4ceeb3ad4ec0a3699eaa687d7120e0a2f81080db95f5f86ad.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/b084a8c496608cf4ceeb3ad4ec0a3699eaa687d7120e0a2f81080db95f5f86ad.jpg)

![b855de15f1702d27097ce07189c3761a52249a293ce1f39bad2d4265a735e40d.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/b855de15f1702d27097ce07189c3761a52249a293ce1f39bad2d4265a735e40d.jpg)

![bbb35fedeaf381da36ecf4b2db623df1d1f869529797ab7152de7dc9a0ef62d5.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/bbb35fedeaf381da36ecf4b2db623df1d1f869529797ab7152de7dc9a0ef62d5.jpg)

![c2f3dc6a5c2f22e9a6595433c2bc30dc2a508148db80a130762fae4122c4907a.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/c2f3dc6a5c2f22e9a6595433c2bc30dc2a508148db80a130762fae4122c4907a.jpg)

![da775070d4445b549a791d4ffe44bcf72f48e0816cd0912919dc15e53b7b4696.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/da775070d4445b549a791d4ffe44bcf72f48e0816cd0912919dc15e53b7b4696.jpg)

![ecff9b51d595f153711089098ba0b03f47d670622f180b7c0b71743394abfdf3.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/ecff9b51d595f153711089098ba0b03f47d670622f180b7c0b71743394abfdf3.jpg)

![f7d7680d71de95ed341fac193ef121022948bca170b7978e0a3238a23575bae6.jpg](../icml_results/1837_DEER_%20A%20Benchmark%20for%20Evaluating%20Deep%20Research%20Agents%20on%20Expert%20Report%20Generation/tables/f7d7680d71de95ed341fac193ef121022948bca170b7978e0a3238a23575bae6.jpg)

<a id="paper-1815"></a>

## The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs

### Images

![0c51d8f9e04513736ce0aaa94f7c394fc478599f51dcd21fc5585d7996548a6a.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/0c51d8f9e04513736ce0aaa94f7c394fc478599f51dcd21fc5585d7996548a6a.jpg)

![23d9cf5eef00b2ba9f5f1c8e51b112e4cb45e876d77f04c303901c3f6151faa0.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/23d9cf5eef00b2ba9f5f1c8e51b112e4cb45e876d77f04c303901c3f6151faa0.jpg)

![2b039024375293a9f38f84362d18ec15befcc9c794b0e528b5aad881b8bd57af.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/2b039024375293a9f38f84362d18ec15befcc9c794b0e528b5aad881b8bd57af.jpg)

![5c084c0d4a3abfa56d20ff5e2ad18879c877e7174362ab662e517ebd21f4ebe7.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/5c084c0d4a3abfa56d20ff5e2ad18879c877e7174362ab662e517ebd21f4ebe7.jpg)

![6096889bd935827b4a9944eda328c96865d9517e33464d39a893bd72d9fd6042.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/6096889bd935827b4a9944eda328c96865d9517e33464d39a893bd72d9fd6042.jpg)

![8624f75651093f531eb61a6ad9a20098105bd46dbfac1135a4515ccd7147326d.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/8624f75651093f531eb61a6ad9a20098105bd46dbfac1135a4515ccd7147326d.jpg)

![abd3d55b32af66497c27d6a9e560a1daf1f1d2ac3057b59cf7f5ecf32d9b9a6c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/abd3d55b32af66497c27d6a9e560a1daf1f1d2ac3057b59cf7f5ecf32d9b9a6c.jpg)

![cb024d7065a032f7b903ff2fcb3d6269ec9599c985c99cdfb610436b5ee4fd64.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/cb024d7065a032f7b903ff2fcb3d6269ec9599c985c99cdfb610436b5ee4fd64.jpg)

![daa4aea6cfe427365159eae1e2b62821f0c0e9a284617a3f816fb91a4724ae3d.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/images/daa4aea6cfe427365159eae1e2b62821f0c0e9a284617a3f816fb91a4724ae3d.jpg)

### Tables

![0a1854bbb45d728959e80b30818572120c41a5eeee9b73b90f27f14358bfb5e7.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/0a1854bbb45d728959e80b30818572120c41a5eeee9b73b90f27f14358bfb5e7.jpg)

![0ba034818a8cc4ed27f729d0647f9b6c61decfc28045fcbb253712368eb7c174.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/0ba034818a8cc4ed27f729d0647f9b6c61decfc28045fcbb253712368eb7c174.jpg)

![2df5e8c8f0ab05f46a4e57face601e098bf0bca9dcfca10216c23f8dbc947db1.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/2df5e8c8f0ab05f46a4e57face601e098bf0bca9dcfca10216c23f8dbc947db1.jpg)

![5f35781f4f5bdb61ffa253487b8ffe98a65ad2e49be4098d9bf5ec6cb1fb9b99.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/5f35781f4f5bdb61ffa253487b8ffe98a65ad2e49be4098d9bf5ec6cb1fb9b99.jpg)

![7ed18a696d8063b642649af3f0deaa8175b24eab6b307ddf59e5af8b0a836f4c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/7ed18a696d8063b642649af3f0deaa8175b24eab6b307ddf59e5af8b0a836f4c.jpg)

![85e15b45145953dda77b84ed91ed5e7f6dd16dfac4a7e36404fae5bd9a306e3e.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/85e15b45145953dda77b84ed91ed5e7f6dd16dfac4a7e36404fae5bd9a306e3e.jpg)

![95f7f896dbeb10197be845aab0f16ae0972763f2a4bdf8ca030b2c66f95f5c0e.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/95f7f896dbeb10197be845aab0f16ae0972763f2a4bdf8ca030b2c66f95f5c0e.jpg)

![a0318afb732560ff134b3253a0c5ff73ad37592edaf14f05910929210ece83b2.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/a0318afb732560ff134b3253a0c5ff73ad37592edaf14f05910929210ece83b2.jpg)

![a3ca3e063af6ebeb8d0ee9a54c42dde8fa260b7b53b9c95e4c1b842f20699aee.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/a3ca3e063af6ebeb8d0ee9a54c42dde8fa260b7b53b9c95e4c1b842f20699aee.jpg)

![aac06f07abb1be5af3eff532ec525e4432033568cc8ba649942210cc83485769.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/aac06f07abb1be5af3eff532ec525e4432033568cc8ba649942210cc83485769.jpg)

![d287b1204f0fdb90f70974513c539020e019cdca96ba62d72bf707a675de1c0c.jpg](../icml_results/1838_The%20Personality%20Illusion_%20Revealing%20Dissociation%20Between%20Self-Reports%20%26%20Behavior%20in%20LLMs/tables/d287b1204f0fdb90f70974513c539020e019cdca96ba62d72bf707a675de1c0c.jpg)
