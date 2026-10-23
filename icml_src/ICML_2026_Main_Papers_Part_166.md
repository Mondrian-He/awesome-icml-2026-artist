# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 166 of 197

## 目录 (Table of Contents)

1. [PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating Neural Network Training](#paper-5446)
2. [Leveraging Lineage Barcodes as Natural Augmentations for Contrastive Learning of Cell Fate in scRNA-seq Data](#paper-5447)
3. [cMoLLM at Scale: Horizontal Scaling Laws for Convolutionally-Gated Mixture-of-LLMs](#paper-5448)
4. [DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization](#paper-5449)
5. [Logarithmic Switching Regret for Online Convex Optimization](#paper-5450)
6. [PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](#paper-5451)
7. [Fast Mixture of Curvature-Aware Experts for Diverse and Dynamic Graph Topologies](#paper-5452)
8. [Autoregressive, Yet Revisable: In Decoding Revision for Secure Code Generation](#paper-5453)
9. [Robust Cross-Modal Retrieval via Generative Semantic Refinement and Exclusion-Guided Adaptation](#paper-5454)
10. [Inner-layer Token Self-modulation as Another Scaling Axis for LLMs](#paper-5455)
11. [Scene Graph Thinking: Reinforcing Structured Visual Reasoning for Multimodal Large Language Models](#paper-5456)
12. [Simple yet Effective: Low-Rank Spatial Attention for Neural Operators](#paper-5457)
13. [RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models](#paper-5458)
14. [Efficient Preference Poisoning Attack on Offline RLHF](#paper-5459)
15. [A Hypertoroidal Covering for Perfect Color Equivariance](#paper-5460)
16. [Masked Multi-path Contrast with Confidence-Gated Semantic Imputation for Incomplete Multi-view Clustering](#paper-5461)
17. [Do Vision and Text Cues Exhibit Evidential Coupling? UFO: A Benchmark for Compositional Multimodal Reasoning in Unified Models](#paper-5462)
18. [Prompt Tuning for CLIP on the Pretrained Manifold](#paper-5463)
19. [Merge to Remember: Sharpness-Aware Isotropic Merging for Continual Learning](#paper-5464)
20. [Factored Causal Representation Learning for Robust Reward Modeling in RLHF](#paper-5465)
21. [Decentralized Online Convex Optimization with Efficient Communication: Improved Algorithm and Lower Bounds](#paper-5466)
22. [SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations](#paper-5467)
23. [SIPO: Stabilized and Improved Preference Optimization for Aligning Diffusion Models](#paper-5468)
24. [Criterion-Conditional In-Context Learning: Evaluating Criterion-Shift Adaptation in Vision-Language Models](#paper-5469)
25. [DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing](#paper-5470)
26. [Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents](#paper-5471)
27. [Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](#paper-5472)
28. [SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval](#paper-5473)
29. [Distinguishable Deletion: Unifying Knowledge Erasure and Refusal for Large Language Model Unlearning](#paper-5474)
30. [Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](#paper-5475)
31. [Accelerated Dual Method for Distributed Optimization: An Inexact-Gradient View of Local Updates](#paper-5476)
32. [Brain Networks Should Be Learned, Not Constructed](#paper-5477)
33. [Speech-Audio Compositional Attacks on Multimodal LLMs and Their Defense with SALMONN-Guard](#paper-5478)

---


<a id="paper-5446"></a>

## PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating Neural Network Training

### Images

![1c7fc7016ebe2abda7b328f92bfdb073af1ce4c0e3fe6da2edf9b3c603cfe932.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/1c7fc7016ebe2abda7b328f92bfdb073af1ce4c0e3fe6da2edf9b3c603cfe932.jpg)

![ae63d09afff8d70f54c585b92a577cf78942c7d5e06a9f0cd74932e27f04a7a0.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/ae63d09afff8d70f54c585b92a577cf78942c7d5e06a9f0cd74932e27f04a7a0.jpg)

![cd60fb6fe08b36925206668744bbab7a976a14c4776457e1bede47b5b8d6bcf8.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/cd60fb6fe08b36925206668744bbab7a976a14c4776457e1bede47b5b8d6bcf8.jpg)

![dd508242f4e1504ddf18e8d46e3dabc925da3a2cfe1bfe5bfbd717864c25ff34.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/dd508242f4e1504ddf18e8d46e3dabc925da3a2cfe1bfe5bfbd717864c25ff34.jpg)

![ed37c32254e2ae5700f881a0c6324214857d1c4cc0c008c2bf146ef4c93c691d.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/ed37c32254e2ae5700f881a0c6324214857d1c4cc0c008c2bf146ef4c93c691d.jpg)

![f687c5e7ad584cb2c9d30d93edb71b8b9c8f04f683fe078e3ea242e4a69292b6.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/images/f687c5e7ad584cb2c9d30d93edb71b8b9c8f04f683fe078e3ea242e4a69292b6.jpg)

### Tables

![78f561734c82fcc19f2bfde218c8cb33db19a53bdaf417bbc59369725e90407a.jpg](../icml_results/5493_Accelerating%20Langevin%20Monte%20Carlo%20via%20Efficient%20Stochastic%20Runge-Kutta%20Methods%20beyond%20Log-Concavity/tables/78f561734c82fcc19f2bfde218c8cb33db19a53bdaf417bbc59369725e90407a.jpg)

## PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating Neural Network Training


### Images

![3090edab6b79af542eec9cba3437ef504e9e2d5284dbec13a098aefdf626afcb.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/3090edab6b79af542eec9cba3437ef504e9e2d5284dbec13a098aefdf626afcb.jpg)

![5957a154ce58850707cdadb1fd604ab212e9a224605e23bc1bf750e31f5c41fb.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/5957a154ce58850707cdadb1fd604ab212e9a224605e23bc1bf750e31f5c41fb.jpg)

![86ceed004a39e2fa0592ba4a68fd5641ca3b949bafd8cfd106103c63fcd73f37.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/86ceed004a39e2fa0592ba4a68fd5641ca3b949bafd8cfd106103c63fcd73f37.jpg)

![a4fbc83d4626a2b738697280b0373b101cde3ae2363b27878ca11ba7fb81fb13.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/a4fbc83d4626a2b738697280b0373b101cde3ae2363b27878ca11ba7fb81fb13.jpg)

![bff614bb9121809a0b707196cba94239253b6de7edbe520e7c725c56d5df7630.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/bff614bb9121809a0b707196cba94239253b6de7edbe520e7c725c56d5df7630.jpg)

![d9a7ffc9ebfa50d4962a8ef5bb19cf035b1290b2cbaed23e29235f530c2f0911.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/d9a7ffc9ebfa50d4962a8ef5bb19cf035b1290b2cbaed23e29235f530c2f0911.jpg)

![dc14f9e794c1d92f9579510853ab9d39fbf854bcad9a79ed4ad7381ccdf843ef.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/dc14f9e794c1d92f9579510853ab9d39fbf854bcad9a79ed4ad7381ccdf843ef.jpg)

![e0c7ce67652346a3e6cf1c8e67808fb96209d6ff762e6d7aa86621c93f1abb7d.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/e0c7ce67652346a3e6cf1c8e67808fb96209d6ff762e6d7aa86621c93f1abb7d.jpg)

![e4eecdf8e9cb391f36e5f017cfba31360bcf2edaacd172eb98922cb344e7cc64.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/e4eecdf8e9cb391f36e5f017cfba31360bcf2edaacd172eb98922cb344e7cc64.jpg)

![ecda96e754dd1198d63ba959f7bc9a07dcebc6a5578a1b9f03ed9070cdacffa5.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/ecda96e754dd1198d63ba959f7bc9a07dcebc6a5578a1b9f03ed9070cdacffa5.jpg)

![ed3305c31a71c17dd6229de53b2bfbfa4fe28d805729016ee453a2a35dc0cac5.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/images/ed3305c31a71c17dd6229de53b2bfbfa4fe28d805729016ee453a2a35dc0cac5.jpg)

### Tables

![821fc82ca98de7329fa9e673a7ed6985e2359df19e83836491faed7580490fdd.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/tables/821fc82ca98de7329fa9e673a7ed6985e2359df19e83836491faed7580490fdd.jpg)

![b7bbd392a0e8367530f1cf427d349d824f0c47e7cc16f16a031dc15ca00df20b.jpg](../icml_results/5494_PRISM_%20Distribution-free%20Adaptive%20Computation%20of%20Matrix%20Functions%20for%20Accelerating%20Neural%20Network%20Tr/tables/b7bbd392a0e8367530f1cf427d349d824f0c47e7cc16f16a031dc15ca00df20b.jpg)

<a id="paper-5447"></a>

## Leveraging Lineage Barcodes as Natural Augmentations for Contrastive Learning of Cell Fate in scRNA-seq Data


### Images

![08f25939847923a56df5baa77dadd12698e68c3be521395431c20a3f4c0df3ca.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/08f25939847923a56df5baa77dadd12698e68c3be521395431c20a3f4c0df3ca.jpg)

![20d011f06d4541ae02327559ddb8eb981ecec5b753740d70bf099e73363466f5.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/20d011f06d4541ae02327559ddb8eb981ecec5b753740d70bf099e73363466f5.jpg)

![224f9c401cb9e72276e2e6ab31e9555e9c7066e677174484d8e5eb450591ba2a.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/224f9c401cb9e72276e2e6ab31e9555e9c7066e677174484d8e5eb450591ba2a.jpg)

![3a6c52602cc2ef963d022383cc7c08969b383ffa3161a74e6a0de70f6af68ed3.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/3a6c52602cc2ef963d022383cc7c08969b383ffa3161a74e6a0de70f6af68ed3.jpg)

![3ac18095122462e5354874aa0c6c45125b1001f8ec1b2cc20eb7012fb9414fea.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/3ac18095122462e5354874aa0c6c45125b1001f8ec1b2cc20eb7012fb9414fea.jpg)

![3e2324d6d3af6d3a0901e67c6bdf3081d494ea9a67751af052e4046829aea33f.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/3e2324d6d3af6d3a0901e67c6bdf3081d494ea9a67751af052e4046829aea33f.jpg)

![410fc6d1a4fbc0d6ec5c7a60b7f2ddd8be6991d0ea00fce0f27c7cc36d799101.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/410fc6d1a4fbc0d6ec5c7a60b7f2ddd8be6991d0ea00fce0f27c7cc36d799101.jpg)

![4b2ce0989c3e50375c5f6f0fa311c7ef52659560c3e2edb6918c783a0c10a005.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/4b2ce0989c3e50375c5f6f0fa311c7ef52659560c3e2edb6918c783a0c10a005.jpg)

![527945d8691f63e9836f52c5a79136ba5872093506cf78f701aca0069879ece8.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/527945d8691f63e9836f52c5a79136ba5872093506cf78f701aca0069879ece8.jpg)

![6053b29092aed815faf371a802e79364d99e2570e74cef8a6e9ebeadbb155850.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/6053b29092aed815faf371a802e79364d99e2570e74cef8a6e9ebeadbb155850.jpg)

![6827cea153efff61ca2d46bfa95caab7d67f79c9367303a5b07ce89d7509251c.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/6827cea153efff61ca2d46bfa95caab7d67f79c9367303a5b07ce89d7509251c.jpg)

![6dfff23de2d42f035132120b9a93eb13186c0bd3325c9fb4f206b2a8354ba460.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/6dfff23de2d42f035132120b9a93eb13186c0bd3325c9fb4f206b2a8354ba460.jpg)

![6f83150fe02fc777854a6f0cbe4a2ba5a1a4a87f2daa1931fb61dc1ad0fb65d8.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/6f83150fe02fc777854a6f0cbe4a2ba5a1a4a87f2daa1931fb61dc1ad0fb65d8.jpg)

![758553eafd4a1a395b854bf890e61443a1f6f460e770d9fc2241b7a434b38cbb.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/758553eafd4a1a395b854bf890e61443a1f6f460e770d9fc2241b7a434b38cbb.jpg)

![974d373824ae590d3f43954693f554684257210c83c7fd4e3da279dd49f73b36.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/974d373824ae590d3f43954693f554684257210c83c7fd4e3da279dd49f73b36.jpg)

![b1af6e2d7ec14294235da6c6f375b3bb22c02831e83986d9d6ad889480ebc18c.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/b1af6e2d7ec14294235da6c6f375b3bb22c02831e83986d9d6ad889480ebc18c.jpg)

![b294242172e830a86e7684dd86afed95b4ce832279dc59dfa11701b3b884f607.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/b294242172e830a86e7684dd86afed95b4ce832279dc59dfa11701b3b884f607.jpg)

![bcb96bd1790d5fda4f3fc4b57ef3d10b3b50b0afe2ba41ae97fb6e6d84f8f017.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/bcb96bd1790d5fda4f3fc4b57ef3d10b3b50b0afe2ba41ae97fb6e6d84f8f017.jpg)

![c020fc91318b773eb3c05baa855bd8ed8972bbb5ec25ce28c896e9b88d418bd3.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/c020fc91318b773eb3c05baa855bd8ed8972bbb5ec25ce28c896e9b88d418bd3.jpg)

![c72c0aad0d420cd3bfb63fbbbbc5a0e0edb78bbc10894885af75c7d0cda49775.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/c72c0aad0d420cd3bfb63fbbbbc5a0e0edb78bbc10894885af75c7d0cda49775.jpg)

![d435ea9c5bde6029c572a1c7850b39784ff6b2fc3eefb68db03bbc8ea8d7b9fe.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/d435ea9c5bde6029c572a1c7850b39784ff6b2fc3eefb68db03bbc8ea8d7b9fe.jpg)

![e0c56073563daec1afeb74879b4214f4ff18f8e339550f1f156763ab66f3011d.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/e0c56073563daec1afeb74879b4214f4ff18f8e339550f1f156763ab66f3011d.jpg)

![f17674b461dd39db64e693027e05b18748d1d78a880125d1dfe7cd84bc641d87.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/f17674b461dd39db64e693027e05b18748d1d78a880125d1dfe7cd84bc641d87.jpg)

![f7ea61a94b376a732e77fce33879c95a75813e21d5f4ad8a30161a60808d2c8b.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/images/f7ea61a94b376a732e77fce33879c95a75813e21d5f4ad8a30161a60808d2c8b.jpg)

### Tables

![03ab9101075cbb199f47f00cef115dc453c9aa5b08ff47fec66f660a467d9d99.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/03ab9101075cbb199f47f00cef115dc453c9aa5b08ff47fec66f660a467d9d99.jpg)

![089d04cb7920c4bea0a52fdfa5124d10c7b7e1bd29e4d42dc79727516d96066e.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/089d04cb7920c4bea0a52fdfa5124d10c7b7e1bd29e4d42dc79727516d96066e.jpg)

![0ed5b854ab987443b07237bd87a879df23619e52aff511acea4b26a76f484700.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/0ed5b854ab987443b07237bd87a879df23619e52aff511acea4b26a76f484700.jpg)

![19d973937fd74f6974bfde24c9ab9b4a630bc84e5744c0a870f771bd52ce38dd.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/19d973937fd74f6974bfde24c9ab9b4a630bc84e5744c0a870f771bd52ce38dd.jpg)

![46d422dfb05834f79488063b6514052908eca4a52c0dee35c0d96aead93c2c38.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/46d422dfb05834f79488063b6514052908eca4a52c0dee35c0d96aead93c2c38.jpg)

![4880ce879830a0bf96c9d9f303ac688744bcb84e8b8d5c459d54c4110f7f10a9.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/4880ce879830a0bf96c9d9f303ac688744bcb84e8b8d5c459d54c4110f7f10a9.jpg)

![4d22e6059d188acead4e1a33dd7ae9983d9776f1a87c7bd7508d74a0241615b6.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/4d22e6059d188acead4e1a33dd7ae9983d9776f1a87c7bd7508d74a0241615b6.jpg)

![58ab0c9694662a89bffd1d2b364dacc2e42e404368e7eb5f3525b2254d2d4305.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/58ab0c9694662a89bffd1d2b364dacc2e42e404368e7eb5f3525b2254d2d4305.jpg)

![6f37355977a00926d93062be731323bd7a64abd85180419f46d70ad78c8e2a77.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/6f37355977a00926d93062be731323bd7a64abd85180419f46d70ad78c8e2a77.jpg)

![7b381bbb7a0b59bf52881b0e4a871d94b3f8de23c34bd5f0a95bcd356d1402fd.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/7b381bbb7a0b59bf52881b0e4a871d94b3f8de23c34bd5f0a95bcd356d1402fd.jpg)

![8158ce4d4c673a9bb189689e4ad4a3b65e05f2d344e13215581faec68f731669.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/8158ce4d4c673a9bb189689e4ad4a3b65e05f2d344e13215581faec68f731669.jpg)

![8c38810bfb0f60e4fe8041409fae965e528043923cc414efaf881ce84a5b5524.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/8c38810bfb0f60e4fe8041409fae965e528043923cc414efaf881ce84a5b5524.jpg)

![949d1b2c12c0c1ce7ac9452bca18f08e970a7df781512936ccace23beba05f1f.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/949d1b2c12c0c1ce7ac9452bca18f08e970a7df781512936ccace23beba05f1f.jpg)

![b73f1756df0bddc38f9b7d0dc0ed22606c61e162dd829930666ac82dc8f8f620.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/b73f1756df0bddc38f9b7d0dc0ed22606c61e162dd829930666ac82dc8f8f620.jpg)

![dbf116ab16a631ee3e6ee42b4896aeb496dbda6663644d1f179036de53443012.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/dbf116ab16a631ee3e6ee42b4896aeb496dbda6663644d1f179036de53443012.jpg)

![ed29d68b50919ffbaddf0908fe4b11f0ca8f2f70c83c3a015ac5af9a1e88dab8.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/ed29d68b50919ffbaddf0908fe4b11f0ca8f2f70c83c3a015ac5af9a1e88dab8.jpg)

![fceda3737da0ee8c8b059d030503bb0f2123890d582e9dfe9945f0d58f3e95b8.jpg](../icml_results/5495_Leveraging%20Lineage%20Barcodes%20as%20Natural%20Augmentations%20for%20Contrastive%20Learning%20of%20Cell%20Fate%20in%20scRNA-/tables/fceda3737da0ee8c8b059d030503bb0f2123890d582e9dfe9945f0d58f3e95b8.jpg)

<a id="paper-5448"></a>

## cMoLLM at Scale: Horizontal Scaling Laws for Convolutionally-Gated Mixture-of-LLMs


### Images

![3dd0f34744d3e59dc9c5571ae16ba360a7e505c4efdbc1c4141725f416cd0c95.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/images/3dd0f34744d3e59dc9c5571ae16ba360a7e505c4efdbc1c4141725f416cd0c95.jpg)

![6234379ca28cceb4d36b1ad2993af895d86dbc94f5fc6a9a8a8e18ed48b3b015.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/images/6234379ca28cceb4d36b1ad2993af895d86dbc94f5fc6a9a8a8e18ed48b3b015.jpg)

![e673b6c3834b2dff01ae4f66459ade16fffd3126ad95568ec252c5e21bcbb2da.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/images/e673b6c3834b2dff01ae4f66459ade16fffd3126ad95568ec252c5e21bcbb2da.jpg)

### Tables

![2f7219f1070a7150957f4c4ddda9911ca13c5e200249a0995fc859fd4ee39987.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/2f7219f1070a7150957f4c4ddda9911ca13c5e200249a0995fc859fd4ee39987.jpg)

![3217e3a8991d4b65afbad417366cc3ea3305ca7c1efb4bb9f2a217f67894afdd.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/3217e3a8991d4b65afbad417366cc3ea3305ca7c1efb4bb9f2a217f67894afdd.jpg)

![642171cbefcd39521be40c1b7d12866acfeb0d077fc7b69d7d4dde63c6b2f794.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/642171cbefcd39521be40c1b7d12866acfeb0d077fc7b69d7d4dde63c6b2f794.jpg)

![962fbfc18bd0e31c15349def280c691880a96ee67d1e97d2b52b19b1e06b2344.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/962fbfc18bd0e31c15349def280c691880a96ee67d1e97d2b52b19b1e06b2344.jpg)

![c98d6c79a570e660612501b01d4498673db4bd86854f00523cdb75cfa43674dc.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/c98d6c79a570e660612501b01d4498673db4bd86854f00523cdb75cfa43674dc.jpg)

![d752a2d1c50ff863f9f2d3febd09a13e0df77bc531c8afbae388aff73738b56f.jpg](../icml_results/5496_cMoLLM%20at%20Scale_%20Horizontal%20Scaling%20Laws%20for%20Convolutionally-Gated%20Mixture-of-LLMs/tables/d752a2d1c50ff863f9f2d3febd09a13e0df77bc531c8afbae388aff73738b56f.jpg)

<a id="paper-5449"></a>

## DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization


### Images

![062cd737d2bd39fba3b237fe28bab6da7cb7378092a5f07bc0979df13811751b.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/062cd737d2bd39fba3b237fe28bab6da7cb7378092a5f07bc0979df13811751b.jpg)

![1071dc3c59a159a28864cacf92f97c6d49397fea57c5f5dd342406e373704b40.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/1071dc3c59a159a28864cacf92f97c6d49397fea57c5f5dd342406e373704b40.jpg)

![10750605227b9e824bab901ba8eda03ca9be8dfe222697e0b28f40b9e2072664.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/10750605227b9e824bab901ba8eda03ca9be8dfe222697e0b28f40b9e2072664.jpg)

![16859e994b201d8634260a88371a9b66efe0bd195ab25cc492888dbbc0004af0.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/16859e994b201d8634260a88371a9b66efe0bd195ab25cc492888dbbc0004af0.jpg)

![2c07251640c1d23094f26860af9bb2d2198959cc26636ff65a41a99347743750.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/2c07251640c1d23094f26860af9bb2d2198959cc26636ff65a41a99347743750.jpg)

![532e114414480409ab3f0cf64aeec3b841803c8cb34ee22a68d2ce9a50beadab.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/532e114414480409ab3f0cf64aeec3b841803c8cb34ee22a68d2ce9a50beadab.jpg)

![56ce396315a88944f3f422ab28cbaab7c0e4bbfa669a9ca2d1ff2b9e43c48e3b.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/56ce396315a88944f3f422ab28cbaab7c0e4bbfa669a9ca2d1ff2b9e43c48e3b.jpg)

![97d8d517504733c1f5ce6b00b82ccc0accd5574588e3e14aa2d64fd085f7abe7.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/97d8d517504733c1f5ce6b00b82ccc0accd5574588e3e14aa2d64fd085f7abe7.jpg)

![9d26706a1dfeb1d9aa28e3cbc697b4f5b2b10d96b979f9d706143c2730e0c4de.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/9d26706a1dfeb1d9aa28e3cbc697b4f5b2b10d96b979f9d706143c2730e0c4de.jpg)

![bf53ff910fa72f6b49f9ff92c45ef92fc2cadd138af5672362158c737ffe9280.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/bf53ff910fa72f6b49f9ff92c45ef92fc2cadd138af5672362158c737ffe9280.jpg)

![c5debb0eded37015b50fe7dbe123c3fd440aff3f0bb998ffad1e28c9a99f58c6.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/c5debb0eded37015b50fe7dbe123c3fd440aff3f0bb998ffad1e28c9a99f58c6.jpg)

![d224f771bada9568b06b9804ef0438c12856c3790c8e092120dd53091370436c.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/images/d224f771bada9568b06b9804ef0438c12856c3790c8e092120dd53091370436c.jpg)

### Tables

![5228ccb80c0d30567d9006e6da1ea07286c07fe4647033fd031b2a48596ec7a5.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/tables/5228ccb80c0d30567d9006e6da1ea07286c07fe4647033fd031b2a48596ec7a5.jpg)

![7d1665971b8b07f72c83b63583573bdcfc710a1f18de5943b9223c8803840f81.jpg](../icml_results/5497_DiffStyle3D_%20Consistent%203D%20Gaussian%20Stylization%20via%20Attention%20Optimization/tables/7d1665971b8b07f72c83b63583573bdcfc710a1f18de5943b9223c8803840f81.jpg)

<a id="paper-5450"></a>

## Logarithmic Switching Regret for Online Convex Optimization


### Images

![1efd88cbd8d3e33df840bfab5fd57aa1289bf9c145f092d294c34acb7bce9273.jpg](../icml_results/5498_Logarithmic%20Switching%20Regret%20for%20Online%20Convex%20Optimization/images/1efd88cbd8d3e33df840bfab5fd57aa1289bf9c145f092d294c34acb7bce9273.jpg)

![4355b1b57ab4b93f89cf6c78d97d2b337fa4d8ca960192d23597759c8eda8e25.jpg](../icml_results/5498_Logarithmic%20Switching%20Regret%20for%20Online%20Convex%20Optimization/images/4355b1b57ab4b93f89cf6c78d97d2b337fa4d8ca960192d23597759c8eda8e25.jpg)

![5a88606d84b9583b3ab6ddd3055ba57727553ded78493d28d990b54a490ab8e6.jpg](../icml_results/5498_Logarithmic%20Switching%20Regret%20for%20Online%20Convex%20Optimization/images/5a88606d84b9583b3ab6ddd3055ba57727553ded78493d28d990b54a490ab8e6.jpg)

![e22fb047917b863ec3debd5dae5cc3018907542eb53e528bac06fc2709356318.jpg](../icml_results/5498_Logarithmic%20Switching%20Regret%20for%20Online%20Convex%20Optimization/images/e22fb047917b863ec3debd5dae5cc3018907542eb53e528bac06fc2709356318.jpg)

<a id="paper-5451"></a>

## PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World


### Images

![1e3fa6be1059ecfed26c6c795d11599a4e121b89c3ea2507256855f60578526c.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/1e3fa6be1059ecfed26c6c795d11599a4e121b89c3ea2507256855f60578526c.jpg)

![46395ac6c18bed5ea2979eff145c862f10a36868f3f9e3fa7adfc4120b1585c0.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/46395ac6c18bed5ea2979eff145c862f10a36868f3f9e3fa7adfc4120b1585c0.jpg)

![ad9d31982aea73f61d0b1a1dec36dbcfcffe7ca92f6c9e594b7cb13b089f13c8.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/ad9d31982aea73f61d0b1a1dec36dbcfcffe7ca92f6c9e594b7cb13b089f13c8.jpg)

![d28a6e73eb634ca289790f883a11226a61944108b53c2954709c67d2aae9c3c1.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/d28a6e73eb634ca289790f883a11226a61944108b53c2954709c67d2aae9c3c1.jpg)

![eb111a48ffb8b5f2b48db88e6872d19a59da4c69b31f8c0a11e39bbc8068fb9b.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/eb111a48ffb8b5f2b48db88e6872d19a59da4c69b31f8c0a11e39bbc8068fb9b.jpg)

![ed141c39379010fd37656ca518818af90d5b4fbb082888474eead24bb24bfdaa.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/images/ed141c39379010fd37656ca518818af90d5b4fbb082888474eead24bb24bfdaa.jpg)

### Tables

![765763bdaf4a9c21951cc20725ecc619a58938be6ae00f4d84de23c88f41c3a7.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/tables/765763bdaf4a9c21951cc20725ecc619a58938be6ae00f4d84de23c88f41c3a7.jpg)

![8672cc2059c66da82c40a711c274832c38f3bf9a1fa6480d1490e2b7bbb04799.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/tables/8672cc2059c66da82c40a711c274832c38f3bf9a1fa6480d1490e2b7bbb04799.jpg)

![a307eb851a89917dbfe613976781259dc23044ef5c3d12f2e475a9e889d26fe5.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/tables/a307eb851a89917dbfe613976781259dc23044ef5c3d12f2e475a9e889d26fe5.jpg)

![a81263b236b268154d8d13e785142d610ab07e0ecea5ef8afe58d556ea68f242.jpg](../icml_results/5499_PhysForge_%20Generating%20Physics-Grounded%203D%20Assets%20for%20Interactive%20Virtual%20World/tables/a81263b236b268154d8d13e785142d610ab07e0ecea5ef8afe58d556ea68f242.jpg)

<a id="paper-5452"></a>

## Fast Mixture of Curvature-Aware Experts for Diverse and Dynamic Graph Topologies


### Images

![151771c89917757a87aa5c3bccfb61e2ab760d4b3404c6026b848526d237e43b.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/151771c89917757a87aa5c3bccfb61e2ab760d4b3404c6026b848526d237e43b.jpg)

![3538a46f12355075172b150971239f87df2c5a71182776c9e99300e0438cc587.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/3538a46f12355075172b150971239f87df2c5a71182776c9e99300e0438cc587.jpg)

![5e0b88b2e707f032077f3acfbadd5efa1fd96999ec6d14a2c9ec5bcaa4439cab.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/5e0b88b2e707f032077f3acfbadd5efa1fd96999ec6d14a2c9ec5bcaa4439cab.jpg)

![6579d7d65d388b31879755dfaa2e4eb491e4cc9ceb592e884700e525b4171e0a.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/6579d7d65d388b31879755dfaa2e4eb491e4cc9ceb592e884700e525b4171e0a.jpg)

![96b12db634177b9c59ed0252dc625661c58cac6bed8cfeb8eed675e7a508d3f9.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/96b12db634177b9c59ed0252dc625661c58cac6bed8cfeb8eed675e7a508d3f9.jpg)

![a2aa43eb2ffc8a56159912ba3077e4ad9445ff45511f78aaf7d0f15a984ce52d.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/a2aa43eb2ffc8a56159912ba3077e4ad9445ff45511f78aaf7d0f15a984ce52d.jpg)

![a4cfd7981ed0c411bc06fd136474c1ef2972eb8b810fdcc9af09d7f5fec09e25.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/a4cfd7981ed0c411bc06fd136474c1ef2972eb8b810fdcc9af09d7f5fec09e25.jpg)

![f67e862d98a531d1d1d3440aadf02c4c81eede1a7a90a43f5b9a98f89ed55d85.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/f67e862d98a531d1d1d3440aadf02c4c81eede1a7a90a43f5b9a98f89ed55d85.jpg)

![f99287f920b020f2db0911f90e7ceaf47a670a6c862938b157195c5487d1897f.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/images/f99287f920b020f2db0911f90e7ceaf47a670a6c862938b157195c5487d1897f.jpg)

### Tables

![0e1244ac8e3e7f911f8ff4a945e94cb25b678b593c6c2ee54dd3ec3276fd4efa.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/0e1244ac8e3e7f911f8ff4a945e94cb25b678b593c6c2ee54dd3ec3276fd4efa.jpg)

![0f9d885890aa938bfcff906970f2575b735d69b59676f2bc59689b958050bde8.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/0f9d885890aa938bfcff906970f2575b735d69b59676f2bc59689b958050bde8.jpg)

![69299a4ada34cd99ed2d7fd9a585b3f902484f67fc25cab8de13c89871ce86c0.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/69299a4ada34cd99ed2d7fd9a585b3f902484f67fc25cab8de13c89871ce86c0.jpg)

![85f0557e2b8446b1df43261c209ea045ac32e0a3cbc8b010a776c0b4fcd9c2f7.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/85f0557e2b8446b1df43261c209ea045ac32e0a3cbc8b010a776c0b4fcd9c2f7.jpg)

![96fdfb82c26bb90724df6c8afbba346d9f8d9b5f87f071074ff0da4a5baf1a52.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/96fdfb82c26bb90724df6c8afbba346d9f8d9b5f87f071074ff0da4a5baf1a52.jpg)

![976648b9949484038399fc43d922aaecbc58835c87a180c2009a4f1fae4bbbad.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/976648b9949484038399fc43d922aaecbc58835c87a180c2009a4f1fae4bbbad.jpg)

![a554e033702ba0e915014b24e9686d4211fd46e331d9826c2beb395b6dcd433b.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/a554e033702ba0e915014b24e9686d4211fd46e331d9826c2beb395b6dcd433b.jpg)

![c891cfc2ec8e4aaedcd271cef6ad44e24451c7d363138e794c6090fbb99fd0e5.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/c891cfc2ec8e4aaedcd271cef6ad44e24451c7d363138e794c6090fbb99fd0e5.jpg)

![d53154f3786de2b81d2d4910c323dc6f6db4c067b10cd9e95cac8d3202a524d9.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/d53154f3786de2b81d2d4910c323dc6f6db4c067b10cd9e95cac8d3202a524d9.jpg)

![eef375e20a2e22fe27214a06dd620448559871688f24b6edbb22b2d8170e890b.jpg](../icml_results/5500_Fast%20Mixture%20of%20Curvature-Aware%20Experts%20for%20Diverse%20and%20Dynamic%20Graph%20Topologies/tables/eef375e20a2e22fe27214a06dd620448559871688f24b6edbb22b2d8170e890b.jpg)

<a id="paper-5453"></a>

## Autoregressive, Yet Revisable: In Decoding Revision for Secure Code Generation


### Images

![1f319ef5e77df3240d931c77d088c71d86a44b70db6d09c5ebba8ffd560a2c90.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/images/1f319ef5e77df3240d931c77d088c71d86a44b70db6d09c5ebba8ffd560a2c90.jpg)

![3e7b30dd4534ba333b5a5bab1fa7e633f87f6d92784f575149a6ed15e2c39b57.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/images/3e7b30dd4534ba333b5a5bab1fa7e633f87f6d92784f575149a6ed15e2c39b57.jpg)

![9f95d754af168e226112333d35ef53f629403eb2e07c008004fe4b47aa5f51a3.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/images/9f95d754af168e226112333d35ef53f629403eb2e07c008004fe4b47aa5f51a3.jpg)

![b9c839ba1d52c5c36e408eb8e657dbdbfebe2d56b22d6877994226641883d86c.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/images/b9c839ba1d52c5c36e408eb8e657dbdbfebe2d56b22d6877994226641883d86c.jpg)

![dcd66211ccf934ecefbe30f08fe8b75d7a6574e6a738d5c1cd7f3aab6f32696a.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/images/dcd66211ccf934ecefbe30f08fe8b75d7a6574e6a738d5c1cd7f3aab6f32696a.jpg)

### Tables

![5713f06f8e14c974abee4a09f010dc78630d3f78f3d7376e58baa6dbdc09c7b6.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/tables/5713f06f8e14c974abee4a09f010dc78630d3f78f3d7376e58baa6dbdc09c7b6.jpg)

![72da8b5f2af10377109e80165f311421310120c3f1032e6d2cfb209df8255dfb.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/tables/72da8b5f2af10377109e80165f311421310120c3f1032e6d2cfb209df8255dfb.jpg)

![8554ef408a933f4052423b64cbc2fbda54cb917fcbf792ef113ddd2f90ed30aa.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/tables/8554ef408a933f4052423b64cbc2fbda54cb917fcbf792ef113ddd2f90ed30aa.jpg)

![c7f051441b987afa7b1934a6ed45d5268827284ec0f33e3a7d6ff56688d37c88.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/tables/c7f051441b987afa7b1934a6ed45d5268827284ec0f33e3a7d6ff56688d37c88.jpg)

![ce059e7ae684023094846e8ee45c2c23b74f045142784e08858dcfbcbba2fbaf.jpg](../icml_results/5501_Autoregressive%2C%20Yet%20Revisable_%20In%20Decoding%20Revision%20for%20Secure%20Code%20Generation/tables/ce059e7ae684023094846e8ee45c2c23b74f045142784e08858dcfbcbba2fbaf.jpg)

<a id="paper-5454"></a>

## Robust Cross-Modal Retrieval via Generative Semantic Refinement and Exclusion-Guided Adaptation


### Images

![108d1c2044968229c789ade31d18f82b10ba831431c05521e72d9bf538ecd9d8.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/images/108d1c2044968229c789ade31d18f82b10ba831431c05521e72d9bf538ecd9d8.jpg)

![34d2223f35dc776cd01e6666b0592288a9f557c2d02521577f2ad5c7e4397adc.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/images/34d2223f35dc776cd01e6666b0592288a9f557c2d02521577f2ad5c7e4397adc.jpg)

![5e17acef39fc4725df08ca17c624fa9e8ec0fc90643e5a59177e7cf1b22928d6.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/images/5e17acef39fc4725df08ca17c624fa9e8ec0fc90643e5a59177e7cf1b22928d6.jpg)

![dc3a4976eeaa7acf3e72e92bf16a03e2890321343e57ec03b76768496b3ef712.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/images/dc3a4976eeaa7acf3e72e92bf16a03e2890321343e57ec03b76768496b3ef712.jpg)

![dd3ce179ae9444f437bd8685f4439322faef19c407b230789a8ba0f6ef2539df.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/images/dd3ce179ae9444f437bd8685f4439322faef19c407b230789a8ba0f6ef2539df.jpg)

### Tables

![2944f41d05bf65f47f3659b118bee56d46884ec98aa776520070dbddcc8b5e69.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/2944f41d05bf65f47f3659b118bee56d46884ec98aa776520070dbddcc8b5e69.jpg)

![3fe2167479f821a16c7f8c99ea2abd7febeb505a5e878611cce86136af34f49d.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/3fe2167479f821a16c7f8c99ea2abd7febeb505a5e878611cce86136af34f49d.jpg)

![4a47facf2b3335eda3cb0b5dcccd513d218f728aa53e7424dcd5d14ebd946edc.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/4a47facf2b3335eda3cb0b5dcccd513d218f728aa53e7424dcd5d14ebd946edc.jpg)

![5e07740d3d1fe098cc0856ae5c500ea2610cf331b529283802e1b29e7fbfdf35.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/5e07740d3d1fe098cc0856ae5c500ea2610cf331b529283802e1b29e7fbfdf35.jpg)

![61e5e3c39f379cac4e923ed1197a10a7bc053c2eff9b17e4d7bf62ba0cbfe09d.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/61e5e3c39f379cac4e923ed1197a10a7bc053c2eff9b17e4d7bf62ba0cbfe09d.jpg)

![96623b0b0146fe3fbbd4da4120e2365d214941989afa2df127dda394592ffacf.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/96623b0b0146fe3fbbd4da4120e2365d214941989afa2df127dda394592ffacf.jpg)

![9fe5ab8837f3c078f359a0ae15b23a9b7c97b2ad2193aa09d20d0777a880db44.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/9fe5ab8837f3c078f359a0ae15b23a9b7c97b2ad2193aa09d20d0777a880db44.jpg)

![a3cb444d88e41d59bb4d633210a0e2dcfb2aedccdd8b14fb0f649a7c4e3ce5fb.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/a3cb444d88e41d59bb4d633210a0e2dcfb2aedccdd8b14fb0f649a7c4e3ce5fb.jpg)

![c68fe3b2ecab2b32370464cc72903151399421bead3979f41eb0eb979af5427f.jpg](../icml_results/5502_Robust%20Cross-Modal%20Retrieval%20via%20Generative%20Semantic%20Refinement%20and%20Exclusion-Guided%20Adaptation/tables/c68fe3b2ecab2b32370464cc72903151399421bead3979f41eb0eb979af5427f.jpg)

<a id="paper-5455"></a>

## Inner-layer Token Self-modulation as Another Scaling Axis for LLMs


### Images

![0a323569f024fec31743b0ae1bbf1d0831339c7e13ee89ea9b4e1a73028bbca2.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/0a323569f024fec31743b0ae1bbf1d0831339c7e13ee89ea9b4e1a73028bbca2.jpg)

![0a4886b2d20903be147c0c8855710cf2e51e9d2271164500892868fefab00bf3.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/0a4886b2d20903be147c0c8855710cf2e51e9d2271164500892868fefab00bf3.jpg)

![20d4b934f9c88121afd18a0bcbc893ca6a5693314201f951c786a1065f92c3f4.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/20d4b934f9c88121afd18a0bcbc893ca6a5693314201f951c786a1065f92c3f4.jpg)

![71f2fa1f1766117672df0ed3bf5063bba0398d09f1b720101e8cbf3a18259ace.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/71f2fa1f1766117672df0ed3bf5063bba0398d09f1b720101e8cbf3a18259ace.jpg)

![acb8cc30e2ad5bf7ddf7259c674baf61172ef9197718bbab10b7f3fee38cab0c.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/acb8cc30e2ad5bf7ddf7259c674baf61172ef9197718bbab10b7f3fee38cab0c.jpg)

![d725cf925e7c098aac91d25d1370028a83c930fa2c665c421b4a8c6646f4e452.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/d725cf925e7c098aac91d25d1370028a83c930fa2c665c421b4a8c6646f4e452.jpg)

![dcd50dbd1fc5bd47ddf6eb71f331da3fe09eec44077ba03e3793b752d3ebbd85.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/dcd50dbd1fc5bd47ddf6eb71f331da3fe09eec44077ba03e3793b752d3ebbd85.jpg)

![fb53d82016522d8f2313e7505908070e9bfc2e7de15f7c11c7c17e514941c4f8.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/images/fb53d82016522d8f2313e7505908070e9bfc2e7de15f7c11c7c17e514941c4f8.jpg)

### Tables

![007b6382f3976fc3001d0a8498cdfb88a172edf4547d8bc2eb6fe2199e421ff7.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/007b6382f3976fc3001d0a8498cdfb88a172edf4547d8bc2eb6fe2199e421ff7.jpg)

![13db79efbad6a660881f3c13e73a646674df192fd27f4950d6357a52d99f0577.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/13db79efbad6a660881f3c13e73a646674df192fd27f4950d6357a52d99f0577.jpg)

![27fd46ba9194af6f6dad5c4f7e930bc837e2ad0c0e11d4e751951fdfeae42ced.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/27fd46ba9194af6f6dad5c4f7e930bc837e2ad0c0e11d4e751951fdfeae42ced.jpg)

![54295bd626928b2f6fc1a999e342714f7bc43cf4075cc2069b4eb91c6e63598d.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/54295bd626928b2f6fc1a999e342714f7bc43cf4075cc2069b4eb91c6e63598d.jpg)

![7243159bac732dc02d0e52a687635f6745f617d69da4ee0f9851c4960854ba1f.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/7243159bac732dc02d0e52a687635f6745f617d69da4ee0f9851c4960854ba1f.jpg)

![822561497a3fe17f2cc7903030e5f3e4cd71677b2c85370dead052f0353c7a12.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/822561497a3fe17f2cc7903030e5f3e4cd71677b2c85370dead052f0353c7a12.jpg)

![e5238ba683385cc20e00860c030038adb4887ce0fbbbb621abe9c69f4e9f9dc0.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/e5238ba683385cc20e00860c030038adb4887ce0fbbbb621abe9c69f4e9f9dc0.jpg)

![ebda0d9251cc3e41b65381f18aa9181d59b01e9d478b80554bc85fce1ccc96c7.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/ebda0d9251cc3e41b65381f18aa9181d59b01e9d478b80554bc85fce1ccc96c7.jpg)

![fe399276d0d09d45524b9145aed788158be4568bf79091b1ef559c84e87af5c7.jpg](../icml_results/5503_Inner-layer%20Token%20Self-modulation%20as%20Another%20Scaling%20Axis%20for%20LLMs/tables/fe399276d0d09d45524b9145aed788158be4568bf79091b1ef559c84e87af5c7.jpg)

<a id="paper-5456"></a>

## Scene Graph Thinking: Reinforcing Structured Visual Reasoning for Multimodal Large Language Models


### Images

![06e355b0209b59b91c3155aa1491552d298366bb3c9ea0a6fd212b84f4a096b3.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/06e355b0209b59b91c3155aa1491552d298366bb3c9ea0a6fd212b84f4a096b3.jpg)

![12e40f326c32ef3e2ef9b036d1d129fe2c0db95c0d6cd32337bb9c1c01a29a4d.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/12e40f326c32ef3e2ef9b036d1d129fe2c0db95c0d6cd32337bb9c1c01a29a4d.jpg)

![16748a649e3622ef42a69b51812c61ae99e6b7aa476cf48270f8798679f3bba1.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/16748a649e3622ef42a69b51812c61ae99e6b7aa476cf48270f8798679f3bba1.jpg)

![1ea4416fd901e4980561d53d4dfc87d03f05e78ce2eaeeea63e7750d3ec8949c.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/1ea4416fd901e4980561d53d4dfc87d03f05e78ce2eaeeea63e7750d3ec8949c.jpg)

![20cc22dcebcaf3356ce21aa31006f69a35165a758321609975ece46eafb66945.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/20cc22dcebcaf3356ce21aa31006f69a35165a758321609975ece46eafb66945.jpg)

![2866d3a5798d4fd2f2476146ce4f057cd81cb8b4b78c452e8b42dd4c290feb9c.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/2866d3a5798d4fd2f2476146ce4f057cd81cb8b4b78c452e8b42dd4c290feb9c.jpg)

![294da2a3a92971ea60e3f2551f08ebb03ac04b8ee74f6a6df6c51d196e9076b6.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/294da2a3a92971ea60e3f2551f08ebb03ac04b8ee74f6a6df6c51d196e9076b6.jpg)

![299c408f5a44ae3766896f6b56a7358e5db28986cc7d07b95c278d34bd490cc9.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/299c408f5a44ae3766896f6b56a7358e5db28986cc7d07b95c278d34bd490cc9.jpg)

![2ff451da5dd1ae18d21cbcf8339381f1267173f49fb592864bee5630d69ccac3.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/2ff451da5dd1ae18d21cbcf8339381f1267173f49fb592864bee5630d69ccac3.jpg)

![4f202d6a1536b51c463a538e8827b83cf42c038223fe79313119883836a63e7a.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/4f202d6a1536b51c463a538e8827b83cf42c038223fe79313119883836a63e7a.jpg)

![5c4c6d9f777d58a8e821b5670c1e2cbaf46179709655c217ec883857b581d2ec.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/5c4c6d9f777d58a8e821b5670c1e2cbaf46179709655c217ec883857b581d2ec.jpg)

![5f1d17cadd873ff56e48a8c908b014c84066d744c41bcfc4805e48d8feaf1553.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/5f1d17cadd873ff56e48a8c908b014c84066d744c41bcfc4805e48d8feaf1553.jpg)

![61b44b1fcae20b627a1548f26d2a836c7d37fa632466b3cd7ddd45ba1592e630.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/61b44b1fcae20b627a1548f26d2a836c7d37fa632466b3cd7ddd45ba1592e630.jpg)

![62763e47b3c8a9059eb6ddf2c48d34a210f9554b6a08fb30d086b38deecc1376.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/62763e47b3c8a9059eb6ddf2c48d34a210f9554b6a08fb30d086b38deecc1376.jpg)

![65cbc3bec3f1ed2332efb4d342a2acdcc7742b39a2a83ca00d9b06a5337315f1.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/65cbc3bec3f1ed2332efb4d342a2acdcc7742b39a2a83ca00d9b06a5337315f1.jpg)

![6da2dd0969df5a510e94b1a5ddc236192aa0a0c70245314272efbd25c271798f.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/6da2dd0969df5a510e94b1a5ddc236192aa0a0c70245314272efbd25c271798f.jpg)

![6db7617d65e2ba7dcdafe1d4fe15d8af6222058a8e1a6e43c9a77bf31fdeb7cb.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/6db7617d65e2ba7dcdafe1d4fe15d8af6222058a8e1a6e43c9a77bf31fdeb7cb.jpg)

![6ddea57eeac04b5f18944544f679339c5a037a1339825516956ac322b9295fc2.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/6ddea57eeac04b5f18944544f679339c5a037a1339825516956ac322b9295fc2.jpg)

![6f92f49784c19fe447e6d47a83a5ce7d904c71c5d2906934c36f7eca1243defc.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/6f92f49784c19fe447e6d47a83a5ce7d904c71c5d2906934c36f7eca1243defc.jpg)

![704824e382472cbdeb9b7d6039a3ec3f8ae8165c2950f5ed5ad48003ff522685.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/704824e382472cbdeb9b7d6039a3ec3f8ae8165c2950f5ed5ad48003ff522685.jpg)

![70f29276b32fdd6f1788f6b54170a8e038162f68dc90b39bad4e3f493eb251af.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/70f29276b32fdd6f1788f6b54170a8e038162f68dc90b39bad4e3f493eb251af.jpg)

![74763dfa8830c35c645a7b07ca583bb0236066f06ce083c2274012fd76839712.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/74763dfa8830c35c645a7b07ca583bb0236066f06ce083c2274012fd76839712.jpg)

![7f802f51de05b4010c0310bc4809f9a421781926e05ff531d5ac2197407f57d4.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/7f802f51de05b4010c0310bc4809f9a421781926e05ff531d5ac2197407f57d4.jpg)

![85ec0573aae98d7da1a478e1c9be95db97b224b5296ebfa7dacefa64fdc4ca13.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/85ec0573aae98d7da1a478e1c9be95db97b224b5296ebfa7dacefa64fdc4ca13.jpg)

![92056a10137cf781012cf1ffc69f1ca163e5b886f03fe710265ca07389865ca2.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/92056a10137cf781012cf1ffc69f1ca163e5b886f03fe710265ca07389865ca2.jpg)

![996a7c1e8f55d27a1da6d388785eaa45d906e8f15bae5ec5a38ab3be3cb8e825.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/996a7c1e8f55d27a1da6d388785eaa45d906e8f15bae5ec5a38ab3be3cb8e825.jpg)

![9efba69e2a5880059d6dd1282879d50b71c47d8abb7767606cd19cfb8c67d594.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/9efba69e2a5880059d6dd1282879d50b71c47d8abb7767606cd19cfb8c67d594.jpg)

![9fc8c357968da04f60d56bf6c0a61e0ea8e9267e868ffe8ba983ba3babe81b06.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/9fc8c357968da04f60d56bf6c0a61e0ea8e9267e868ffe8ba983ba3babe81b06.jpg)

![bab78f3a159466752eadb12a4cd2004bd4fdaf2d14edb9dac17ae073b7d00991.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/bab78f3a159466752eadb12a4cd2004bd4fdaf2d14edb9dac17ae073b7d00991.jpg)

![bb932cddcb32dd6a46a453fd2d2702001b43b04c8858de2a024dfa0c94d5fa35.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/bb932cddcb32dd6a46a453fd2d2702001b43b04c8858de2a024dfa0c94d5fa35.jpg)

![bfa61a8e27417b42c15001a8beb08b5d82c3e3100e58027bdded10147854e111.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/bfa61a8e27417b42c15001a8beb08b5d82c3e3100e58027bdded10147854e111.jpg)

![c7a43d2cf2c6e1ec66a6ccbdd59051864244e695eca910a2e3a254e980c0997f.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/c7a43d2cf2c6e1ec66a6ccbdd59051864244e695eca910a2e3a254e980c0997f.jpg)

![c895fa163a72b79445384474a58a961d3ffe010ffb6c206f8cf679d8ee614685.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/c895fa163a72b79445384474a58a961d3ffe010ffb6c206f8cf679d8ee614685.jpg)

![d6bf09743214f52701b67315b3525c4b2338675a71ffb23d18c8e5d2d7ae0424.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/d6bf09743214f52701b67315b3525c4b2338675a71ffb23d18c8e5d2d7ae0424.jpg)

![f78ec03552fbc3f2bd1adb66e0f71ff9e26a4cb643b21b0b3f103393946cf419.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/images/f78ec03552fbc3f2bd1adb66e0f71ff9e26a4cb643b21b0b3f103393946cf419.jpg)

### Tables

![3a3a38f8ba15f5664856d509d12efd26a4944aa16bae44c228af33ab7e2cfc0c.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/3a3a38f8ba15f5664856d509d12efd26a4944aa16bae44c228af33ab7e2cfc0c.jpg)

![9fdd0965405d62ab21504db186d89b60676a4b4cf7d6f607c1b057e504b9cb28.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/9fdd0965405d62ab21504db186d89b60676a4b4cf7d6f607c1b057e504b9cb28.jpg)

![b684046b4338deaf9323945a2651269160fdb880798bb187dce22147f6e144d7.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/b684046b4338deaf9323945a2651269160fdb880798bb187dce22147f6e144d7.jpg)

![c9e1ee4c64027c122e688417e84a4a4aa9f5dedf60939b4d8a4470d12dd17ab7.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/c9e1ee4c64027c122e688417e84a4a4aa9f5dedf60939b4d8a4470d12dd17ab7.jpg)

![ca285c8eb4ac9fc002fc9949160f3f31938391b6456817abea5b59e68857838f.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/ca285c8eb4ac9fc002fc9949160f3f31938391b6456817abea5b59e68857838f.jpg)

![d27f70d2b922dedf15f8cb8f264720dc18eebe8a744487e92f423bb6f09429d9.jpg](../icml_results/5504_Scene%20Graph%20Thinking_%20Reinforcing%20Structured%20Visual%20Reasoning%20for%20Multimodal%20Large%20Language%20Models/tables/d27f70d2b922dedf15f8cb8f264720dc18eebe8a744487e92f423bb6f09429d9.jpg)

<a id="paper-5457"></a>

## Simple yet Effective: Low-Rank Spatial Attention for Neural Operators


### Images

![3bfe945e26e8cb7f3ff37bc78e5325e22107688877327feb8c7f5bf6682f6433.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/images/3bfe945e26e8cb7f3ff37bc78e5325e22107688877327feb8c7f5bf6682f6433.jpg)

![4ad4642e044460b25542bb45a2d18b1981c154eef4ccf78a778f6820a4368ecc.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/images/4ad4642e044460b25542bb45a2d18b1981c154eef4ccf78a778f6820a4368ecc.jpg)

![68fab5df53484a3a8e6a8142199bac3de694f72ae8b9dc8751cd07febc1e453a.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/images/68fab5df53484a3a8e6a8142199bac3de694f72ae8b9dc8751cd07febc1e453a.jpg)

![d6e6ab936f2d2284697db84c531dd45137b3dce7321cf09119797c349c4cfce4.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/images/d6e6ab936f2d2284697db84c531dd45137b3dce7321cf09119797c349c4cfce4.jpg)

![f4100539b9625d18a1de9b4554ff284b895c771f636a75cee81b060516d5e298.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/images/f4100539b9625d18a1de9b4554ff284b895c771f636a75cee81b060516d5e298.jpg)

### Tables

![01917b73cb3c74eb62231afff07e76a6e34535d0bd64d171ede87f406ca93f52.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/01917b73cb3c74eb62231afff07e76a6e34535d0bd64d171ede87f406ca93f52.jpg)

![0b853b6f65dcf4cc81d15bfcbb8f54955dfaaf2d61f3c24546b4af1b569968eb.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/0b853b6f65dcf4cc81d15bfcbb8f54955dfaaf2d61f3c24546b4af1b569968eb.jpg)

![13cd28f9dd89095a671c08535db309cde71cc82f0272de0820e43d1ec506cd21.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/13cd28f9dd89095a671c08535db309cde71cc82f0272de0820e43d1ec506cd21.jpg)

![3f9079c5683768f1d4cd072e43f7159dced30a9d5722f5ec3820befebaaff9ad.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/3f9079c5683768f1d4cd072e43f7159dced30a9d5722f5ec3820befebaaff9ad.jpg)

![41960fa5e29ab8bb94de34ec09ac709df77c1ae4c9645f85524e616882ef7309.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/41960fa5e29ab8bb94de34ec09ac709df77c1ae4c9645f85524e616882ef7309.jpg)

![56b2acacb6763fbfb410bf7cec232d26b5357ef3fcf45ba2213bae10777f24b3.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/56b2acacb6763fbfb410bf7cec232d26b5357ef3fcf45ba2213bae10777f24b3.jpg)

![590527cca2e4fc885dbdaf96c4292ff6fc75aad6f508e325eb0c87c2036a8e8f.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/590527cca2e4fc885dbdaf96c4292ff6fc75aad6f508e325eb0c87c2036a8e8f.jpg)

![9d98e51fc0a928c62e0b8b8a3649cc23db95c2949562e848ac3b8248ed9569af.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/9d98e51fc0a928c62e0b8b8a3649cc23db95c2949562e848ac3b8248ed9569af.jpg)

![bb8a5041d4a16c5d5e9484f52e6337ffb00b9a43371f452da082041803a31e11.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/bb8a5041d4a16c5d5e9484f52e6337ffb00b9a43371f452da082041803a31e11.jpg)

![bc770fe091f16a62a5d6790bdd3089ee725f98eb413ab2a4714cfc335915738c.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/bc770fe091f16a62a5d6790bdd3089ee725f98eb413ab2a4714cfc335915738c.jpg)

![c6c6a2cb878bb7006cbade3c404e32a5d6c2c53a6f81425d1411a7e9b04d5e38.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/c6c6a2cb878bb7006cbade3c404e32a5d6c2c53a6f81425d1411a7e9b04d5e38.jpg)

![d2fa8d7ce48ded08fd6ab4ae96290903fafa7126402b845924af20cf2a109dc9.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/d2fa8d7ce48ded08fd6ab4ae96290903fafa7126402b845924af20cf2a109dc9.jpg)

![e872f0869b3636131df661ec13d5137d530c6dfe8959ae799fe6846d47698182.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/e872f0869b3636131df661ec13d5137d530c6dfe8959ae799fe6846d47698182.jpg)

![f7840f684e140804ad60d13464814295e85bee851b3db6acc8024a4edf7174c5.jpg](../icml_results/5505_Simple%20yet%20Effective_%20Low-Rank%20Spatial%20Attention%20for%20Neural%20Operators/tables/f7840f684e140804ad60d13464814295e85bee851b3db6acc8024a4edf7174c5.jpg)

<a id="paper-5458"></a>

## RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models


### Images

![0512dcc82c31040a80df60823b8fb1039ed4af86faabdc0a9eec837c4810cf9c.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/0512dcc82c31040a80df60823b8fb1039ed4af86faabdc0a9eec837c4810cf9c.jpg)

![4f3b99fed1696e5308acb226c45560b29d71232cf8540c97d9838528dd9bc046.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/4f3b99fed1696e5308acb226c45560b29d71232cf8540c97d9838528dd9bc046.jpg)

![b709c7f3554ff0a39128c2babebf603b1ddeaae280b50f291965770115faae7c.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/b709c7f3554ff0a39128c2babebf603b1ddeaae280b50f291965770115faae7c.jpg)

![d0855ba271563393f1be6142297cbc1818ffa2688ed1b61b648fce25ee44cdb4.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/d0855ba271563393f1be6142297cbc1818ffa2688ed1b61b648fce25ee44cdb4.jpg)

![d7e7fa6f5694a80232503a4eddf06bbfe64454e212184c0e314df066148187bf.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/d7e7fa6f5694a80232503a4eddf06bbfe64454e212184c0e314df066148187bf.jpg)

![ddbed45a6893fe05aa03be3175bf9df2fbb1556a422c4e9e43dc8b504408c4e8.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/ddbed45a6893fe05aa03be3175bf9df2fbb1556a422c4e9e43dc8b504408c4e8.jpg)

![f1e64dca6d75e0f9bd02694a857d302ab911f481173875004f11481829fc1a0c.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/f1e64dca6d75e0f9bd02694a857d302ab911f481173875004f11481829fc1a0c.jpg)

![f2d13482f58df4032717deb4470b47bdd2efd6eeee0af8f287e92374926265f8.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/f2d13482f58df4032717deb4470b47bdd2efd6eeee0af8f287e92374926265f8.jpg)

![f827bb632d38ecce12a89b563cda9e9dc8f4424b76a5becec739e85b04186c15.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/f827bb632d38ecce12a89b563cda9e9dc8f4424b76a5becec739e85b04186c15.jpg)

![ffc02706214ecf91e4d4bd687cdf74e37b876ed96a2d04d933210ed0dfff7f25.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/images/ffc02706214ecf91e4d4bd687cdf74e37b876ed96a2d04d933210ed0dfff7f25.jpg)

### Tables

![2eff96846c7e8e0d5f1fd954a9006bdcc150b3c1e636734a74d5248a3a980c72.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/2eff96846c7e8e0d5f1fd954a9006bdcc150b3c1e636734a74d5248a3a980c72.jpg)

![3c6fef019d99e632258bf87d34f1750e2083b2bdd6ee91e17e87886e830d3dc3.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/3c6fef019d99e632258bf87d34f1750e2083b2bdd6ee91e17e87886e830d3dc3.jpg)

![69732eb79e3343b8572198225bcc9e199e9d5b1ed5b99003ccd03de7e0db6b11.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/69732eb79e3343b8572198225bcc9e199e9d5b1ed5b99003ccd03de7e0db6b11.jpg)

![6f2f719fe45a34c6873acc7867e9c9d4c82a87dfc5820a811955b899611a79bf.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/6f2f719fe45a34c6873acc7867e9c9d4c82a87dfc5820a811955b899611a79bf.jpg)

![7014692e87f2994fd32be8b3e42d078fdbc532775e9ad03b61c9b002267ecc0e.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/7014692e87f2994fd32be8b3e42d078fdbc532775e9ad03b61c9b002267ecc0e.jpg)

![7040c7a66b0d95594a3d1992d05e80f95e308a935731bf037d554f5b3055d54d.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/7040c7a66b0d95594a3d1992d05e80f95e308a935731bf037d554f5b3055d54d.jpg)

![cffb27790f0461aa7965d4968803d11c450be64137a941cade110189c3ccdce8.jpg](../icml_results/5506_RLIE_%20Rule%20Generation%20with%20Logistic%20Regression%2C%20Iterative%20Refinement%2C%20and%20Evaluation%20for%20Large%20Langu/tables/cffb27790f0461aa7965d4968803d11c450be64137a941cade110189c3ccdce8.jpg)

<a id="paper-5459"></a>

## Efficient Preference Poisoning Attack on Offline RLHF


### Images

![035b6131291c25fc267ef63e7d03737dcd79d13c54d6296255fbd51175b4864b.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/035b6131291c25fc267ef63e7d03737dcd79d13c54d6296255fbd51175b4864b.jpg)

![09b2d9f0165c3915801fa4762575a79977739063bff62cb19db53c9bc4698e43.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/09b2d9f0165c3915801fa4762575a79977739063bff62cb19db53c9bc4698e43.jpg)

![173a26849e4d9e7eaf18b5ceb5706fcc1a3f23f8ebb8a28759fc2859922af6d0.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/173a26849e4d9e7eaf18b5ceb5706fcc1a3f23f8ebb8a28759fc2859922af6d0.jpg)

![37a03087d49f0edef031fef01a467c131c4b3ec07f47ec9f47fd83a8ca0e6749.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/37a03087d49f0edef031fef01a467c131c4b3ec07f47ec9f47fd83a8ca0e6749.jpg)

![3cac4c845bf4b9489a05d25ac223bfd5480a4434ff5cbbf8b1d1a1575e9d62fb.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/3cac4c845bf4b9489a05d25ac223bfd5480a4434ff5cbbf8b1d1a1575e9d62fb.jpg)

![7317c92a39014dccf992506053cfc0e72c469c6a413b9665d63fe9baee5e3a28.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/7317c92a39014dccf992506053cfc0e72c469c6a413b9665d63fe9baee5e3a28.jpg)

![88a46a335adf503dd28236143f463aa0a74dbb7bf7d9aa127224d53eab24bf79.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/88a46a335adf503dd28236143f463aa0a74dbb7bf7d9aa127224d53eab24bf79.jpg)

![a8dde6227b363dfbbc606053b5abfc1dc4c2ba631605ddc24b9e2443d8731d86.jpg](../icml_results/5507_Efficient%20Preference%20Poisoning%20Attack%20on%20Offline%20RLHF/images/a8dde6227b363dfbbc606053b5abfc1dc4c2ba631605ddc24b9e2443d8731d86.jpg)

<a id="paper-5460"></a>

## A Hypertoroidal Covering for Perfect Color Equivariance


### Images

![04222b1f412683b70d9b4a076cf56341f8bde8999c578f9023d172361bc245bd.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/04222b1f412683b70d9b4a076cf56341f8bde8999c578f9023d172361bc245bd.jpg)

![09c0527031249c094ddae2439c130a0016f73aa69d3232876a85d8b242b98d02.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/09c0527031249c094ddae2439c130a0016f73aa69d3232876a85d8b242b98d02.jpg)

![216189bb62f3542a9d2ccce85662821a2f0df9d3f17087513e16fa3ea4fc2cad.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/216189bb62f3542a9d2ccce85662821a2f0df9d3f17087513e16fa3ea4fc2cad.jpg)

![3003c052b4132a24ab7d6d9cb718444c15b95b933f850712afca23a5bbaaeaa5.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/3003c052b4132a24ab7d6d9cb718444c15b95b933f850712afca23a5bbaaeaa5.jpg)

![676b72698c52baa67a016698f170c3e8b4384ac9694f04da6734337ff91b6bef.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/676b72698c52baa67a016698f170c3e8b4384ac9694f04da6734337ff91b6bef.jpg)

![706f4907e742faf8d5a03618fcc8757a89cf79ba8f2f0462c1c1d805d08538ff.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/706f4907e742faf8d5a03618fcc8757a89cf79ba8f2f0462c1c1d805d08538ff.jpg)

![818d57e29565075ef2446d9a5755782432fdbe15b1e94b1505d6fe60c5675942.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/818d57e29565075ef2446d9a5755782432fdbe15b1e94b1505d6fe60c5675942.jpg)

![85315521817efcc39ac268077781c9b404ba71ce8b2717431b007390b48258d9.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/85315521817efcc39ac268077781c9b404ba71ce8b2717431b007390b48258d9.jpg)

![a50e673e4ab4172debf6bc897aaa5f32d25ce1390bc52d5217c64c54bb4601bf.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/a50e673e4ab4172debf6bc897aaa5f32d25ce1390bc52d5217c64c54bb4601bf.jpg)

![bfc91dfbd04e20340f1e7dc24c68fa78c881ae655f6f811c60dabbce51876ab6.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/bfc91dfbd04e20340f1e7dc24c68fa78c881ae655f6f811c60dabbce51876ab6.jpg)

![c8dae6b5452e5ca01a8557a0cdd77400484c62a12336b74b52a59d7b35ef1ba4.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/c8dae6b5452e5ca01a8557a0cdd77400484c62a12336b74b52a59d7b35ef1ba4.jpg)

![ccde6edf4ea06af7b3a9c34414b41d831eaf5a1a18851119a79357c154edd5bc.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/ccde6edf4ea06af7b3a9c34414b41d831eaf5a1a18851119a79357c154edd5bc.jpg)

![d79b415232657452086fb0b049ec0c970dd9cce1a5bf1c80a152da3a4cf8efcf.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/d79b415232657452086fb0b049ec0c970dd9cce1a5bf1c80a152da3a4cf8efcf.jpg)

![e794cd7db5a1874d5d63aafa344ad18a9a7be19c44db53bc33c5fe232e6b8435.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/e794cd7db5a1874d5d63aafa344ad18a9a7be19c44db53bc33c5fe232e6b8435.jpg)

![f5987ccc122fcaae82d09e50ef4aa9887aab540d582a4a884c060284fc676072.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/images/f5987ccc122fcaae82d09e50ef4aa9887aab540d582a4a884c060284fc676072.jpg)

### Tables

![49d0330e017d64bbb00a9deafdb7a42c4aa3fa294e8fbb22b7bd96b522d27fbd.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/49d0330e017d64bbb00a9deafdb7a42c4aa3fa294e8fbb22b7bd96b522d27fbd.jpg)

![64cea0773e94b37afc5fd9a163e1d3d381392f02717b33d36b27833b7ca70af7.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/64cea0773e94b37afc5fd9a163e1d3d381392f02717b33d36b27833b7ca70af7.jpg)

![9e8f99dd6f532c713c3a93dd2ba0e82beba2a562ea041fb18955b6ddf8f51a2e.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/9e8f99dd6f532c713c3a93dd2ba0e82beba2a562ea041fb18955b6ddf8f51a2e.jpg)

![a1128401ddf9169425b9b9c0f116d6d0265c011e78ef3767c93165e8421955ce.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/a1128401ddf9169425b9b9c0f116d6d0265c011e78ef3767c93165e8421955ce.jpg)

![bf453804f188e031042d9d3bcaac2df0f86c9e0c712b1ce30bde52b83416073a.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/bf453804f188e031042d9d3bcaac2df0f86c9e0c712b1ce30bde52b83416073a.jpg)

![d15ac7a727f81a260f8edfa688b3b3fb30da1851a11545aa7e94c66703f9350a.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/d15ac7a727f81a260f8edfa688b3b3fb30da1851a11545aa7e94c66703f9350a.jpg)

![d52c14b4f26eaa03858f2db4feaca8af0ce5a34970ce356ff040da7e85825d02.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/d52c14b4f26eaa03858f2db4feaca8af0ce5a34970ce356ff040da7e85825d02.jpg)

![db0bd04cd0fa7ca553fd41e99a524e17cd0a9d4b6549fd4074f4fc1e03f8a6de.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/db0bd04cd0fa7ca553fd41e99a524e17cd0a9d4b6549fd4074f4fc1e03f8a6de.jpg)

![f01dd37608da3d3c1f042643f9b455697c4ee7b148a264f408c9660df5845339.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/f01dd37608da3d3c1f042643f9b455697c4ee7b148a264f408c9660df5845339.jpg)

![f946f4a7140cbde8ce33ba277508e306f8fa665d042f39da1e1fefb879d208b3.jpg](../icml_results/5508_A%20Hypertoroidal%20Covering%20for%20Perfect%20Color%20Equivariance/tables/f946f4a7140cbde8ce33ba277508e306f8fa665d042f39da1e1fefb879d208b3.jpg)

<a id="paper-5461"></a>

## Masked Multi-path Contrast with Confidence-Gated Semantic Imputation for Incomplete Multi-view Clustering


### Images

![0fa0fe5cd2832342675243256a6983244492bae5742f16221191d7cf6112c8bf.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/0fa0fe5cd2832342675243256a6983244492bae5742f16221191d7cf6112c8bf.jpg)

![14f1a59f8d1ce3649a91c798112de14361989b975ea80e11eb6f5cd06481502e.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/14f1a59f8d1ce3649a91c798112de14361989b975ea80e11eb6f5cd06481502e.jpg)

![6039715917c979372da03a108ff946ad670f9deb0d0776125548f40202ce7c9a.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/6039715917c979372da03a108ff946ad670f9deb0d0776125548f40202ce7c9a.jpg)

![65f48aa1d103ba30530873df6386188195835f72433ff0ce7b62864bd5d2608d.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/65f48aa1d103ba30530873df6386188195835f72433ff0ce7b62864bd5d2608d.jpg)

![6926edc677036e0b5a4b3a9396c5276a974bd100c183b522b672ccfe0ccb4c35.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/6926edc677036e0b5a4b3a9396c5276a974bd100c183b522b672ccfe0ccb4c35.jpg)

![71458afa1a33745992cca07dd362808d1c8e43ab6f32591918bc26a2a8360bf0.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/71458afa1a33745992cca07dd362808d1c8e43ab6f32591918bc26a2a8360bf0.jpg)

![88ea04f35b520a3784764302c7bc0864d05c4348a2782c95e2954c07fb237303.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/88ea04f35b520a3784764302c7bc0864d05c4348a2782c95e2954c07fb237303.jpg)

![90df83d85fe75c44c724f5b0ddb400d5b60bfa82ff8c6fa91396294abe1ccde0.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/90df83d85fe75c44c724f5b0ddb400d5b60bfa82ff8c6fa91396294abe1ccde0.jpg)

![b9a075473082d636ae55030d1a4a3f93e7cac938c794c451c52dd103fbaccb9d.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/b9a075473082d636ae55030d1a4a3f93e7cac938c794c451c52dd103fbaccb9d.jpg)

![db2edcc5d6438d73c05fa17f47af37d3fc07cff491d95a1e54f284610bf9c81b.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/images/db2edcc5d6438d73c05fa17f47af37d3fc07cff491d95a1e54f284610bf9c81b.jpg)

### Tables

![30a46279bacb51b8ae61989a3a2af725f6639c5d10cc3725712b6aad375c2590.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/tables/30a46279bacb51b8ae61989a3a2af725f6639c5d10cc3725712b6aad375c2590.jpg)

![b0cbea375f6cf2a1db3eea38c38417853bbb96981f9f2d884e17bdd0a71e0273.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/tables/b0cbea375f6cf2a1db3eea38c38417853bbb96981f9f2d884e17bdd0a71e0273.jpg)

![d135e94cb6f64db65c105e95cf832fbafc8b473aff18eebaef60b61dc6d3dfcf.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/tables/d135e94cb6f64db65c105e95cf832fbafc8b473aff18eebaef60b61dc6d3dfcf.jpg)

![e52c6dfd8d165367a18d54c7c93ae7a3a7d06ab560895ad865143d84ce75f9ea.jpg](../icml_results/5509_Masked%20Multi-path%20Contrast%20with%20Confidence-Gated%20Semantic%20Imputation%20for%20Incomplete%20Multi-view%20Clust/tables/e52c6dfd8d165367a18d54c7c93ae7a3a7d06ab560895ad865143d84ce75f9ea.jpg)

<a id="paper-5462"></a>

## Do Vision and Text Cues Exhibit Evidential Coupling? UFO: A Benchmark for Compositional Multimodal Reasoning in Unified Models


### Images

![021376f4d348b373c300149675f06f392fcfebe5987573283c04a62b9f7c6a3c.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/021376f4d348b373c300149675f06f392fcfebe5987573283c04a62b9f7c6a3c.jpg)

![08c8fb0ceb976bed4b4f9b50f7a4abaa16a969822392539f8da808126baa42b0.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/08c8fb0ceb976bed4b4f9b50f7a4abaa16a969822392539f8da808126baa42b0.jpg)

![0e03f54dda711472a756c1e536d233f0a1cc543850c3dfd600d976e036857b07.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/0e03f54dda711472a756c1e536d233f0a1cc543850c3dfd600d976e036857b07.jpg)

![0eaa22709a5f7c46670527364caa753298c7e92dd2259580b620bce78ed5f82f.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/0eaa22709a5f7c46670527364caa753298c7e92dd2259580b620bce78ed5f82f.jpg)

![160dc74a807d4d56ab2581cb5cf9369c9c28f53b0082db68fdcaa20901d277d1.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/160dc74a807d4d56ab2581cb5cf9369c9c28f53b0082db68fdcaa20901d277d1.jpg)

![1e6c5e6a6b542cf3d4c8555e75ea4c8d4077057546ea5cc33206ad5a037d421a.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/1e6c5e6a6b542cf3d4c8555e75ea4c8d4077057546ea5cc33206ad5a037d421a.jpg)

![2e1078e13171611b4e682d9e1e2b77f6fa1486ea5c6f26516f3ead7bbdfeb20c.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/2e1078e13171611b4e682d9e1e2b77f6fa1486ea5c6f26516f3ead7bbdfeb20c.jpg)

![417e941033c28ac5b32807483ab621f85de13db2330b2699223f0a69dbe18f47.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/417e941033c28ac5b32807483ab621f85de13db2330b2699223f0a69dbe18f47.jpg)

![561a7712c797cdf202d7ab8b3f6b09b17c01ab1f8993ec47a832f1abe0bf89a4.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/561a7712c797cdf202d7ab8b3f6b09b17c01ab1f8993ec47a832f1abe0bf89a4.jpg)

![5b589c5bce4d0c6277c120d5c31d4845c013c9a24d1989cbcd1f5c30e90682ba.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/5b589c5bce4d0c6277c120d5c31d4845c013c9a24d1989cbcd1f5c30e90682ba.jpg)

![636c940c699d5f2e909c9207c2d9529ddd60ee92d3aa654f375a2ea3278d94e5.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/636c940c699d5f2e909c9207c2d9529ddd60ee92d3aa654f375a2ea3278d94e5.jpg)

![a23cd03fede4fdaae23c799481f6fc28c2db9e13ffadd4e6d37d5a127990b5da.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/a23cd03fede4fdaae23c799481f6fc28c2db9e13ffadd4e6d37d5a127990b5da.jpg)

![b03d7b4151edea1560439e7867977717c95288b5853027e7752f1cbc60271a0b.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/b03d7b4151edea1560439e7867977717c95288b5853027e7752f1cbc60271a0b.jpg)

![bc16c85d316cd2f7f2edfbd41430c18547008bc3c3291b7a0efc4d64756fb883.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/bc16c85d316cd2f7f2edfbd41430c18547008bc3c3291b7a0efc4d64756fb883.jpg)

![c0c4baa9ac659cd55c0aaab01b2764b57522b7d39052061798c77a44a7bf8352.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/c0c4baa9ac659cd55c0aaab01b2764b57522b7d39052061798c77a44a7bf8352.jpg)

![cf0ca5d68876c1f11a9954a564f843a47265ad723b332a4501e384307a6b40b5.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/cf0ca5d68876c1f11a9954a564f843a47265ad723b332a4501e384307a6b40b5.jpg)

![d81dd0d2d01ee169d46c0704cca725667d4626633c150ab5d7b83f764efee425.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/d81dd0d2d01ee169d46c0704cca725667d4626633c150ab5d7b83f764efee425.jpg)

![da90b38bd08643eb33d00fb6b35c096b654f8a1daf7890b4bc49c6b8a82cc246.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/da90b38bd08643eb33d00fb6b35c096b654f8a1daf7890b4bc49c6b8a82cc246.jpg)

![e7acdf997c3314c4834686e0f0edeaea40faef3acf284bf357cbcb7b75cb99ed.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/e7acdf997c3314c4834686e0f0edeaea40faef3acf284bf357cbcb7b75cb99ed.jpg)

![e9a2777b29da5dc35991fb1f577b90b59ceea6d4ddd465dec9f79a082c42b0b1.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/e9a2777b29da5dc35991fb1f577b90b59ceea6d4ddd465dec9f79a082c42b0b1.jpg)

![f7aa2bb1495f2fd0e6d54e9f09f4aa3e1501b54b97623dc8578084f4288f8cfb.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/f7aa2bb1495f2fd0e6d54e9f09f4aa3e1501b54b97623dc8578084f4288f8cfb.jpg)

![fd1df523e6b6ee5849cde1793b27868979011bedeaa5f1e8250c5f8726f02386.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/images/fd1df523e6b6ee5849cde1793b27868979011bedeaa5f1e8250c5f8726f02386.jpg)

### Tables

![1925a092eb32b66b4797d970d9cef30eaf92b3f8353bb88c40a06b941394cdc4.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/1925a092eb32b66b4797d970d9cef30eaf92b3f8353bb88c40a06b941394cdc4.jpg)

![4228e22c7d0175003b69cf326efff9c8c89736ce44c2594756536aa40aaa0ad4.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/4228e22c7d0175003b69cf326efff9c8c89736ce44c2594756536aa40aaa0ad4.jpg)

![67593a0001cf2b56dff8ed40f4c718055f76cad1bd171e54d3f88816f6becf14.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/67593a0001cf2b56dff8ed40f4c718055f76cad1bd171e54d3f88816f6becf14.jpg)

![7192c9ccf05ec5cbcd8497a9d190524971c95a28ff5db1fe6cb8a230256c7a0f.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/7192c9ccf05ec5cbcd8497a9d190524971c95a28ff5db1fe6cb8a230256c7a0f.jpg)

![ab1830f8a3890ae663f02170c157f96ac03a519b30a2b604298d4421156ffd60.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/ab1830f8a3890ae663f02170c157f96ac03a519b30a2b604298d4421156ffd60.jpg)

![c26c70fb522f924baec8deeab9d87cf2f5653bf88442a429dfcd7aecf4426c57.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/c26c70fb522f924baec8deeab9d87cf2f5653bf88442a429dfcd7aecf4426c57.jpg)

![c574a980e5af9a1c015a6a7f47e83cb1c572b54d7d3783b1925cfb9c7df67b22.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/c574a980e5af9a1c015a6a7f47e83cb1c572b54d7d3783b1925cfb9c7df67b22.jpg)

![ee2174c3074569248813e6726bd4767f286d3d7a56f668edcbc432848409f6ec.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/ee2174c3074569248813e6726bd4767f286d3d7a56f668edcbc432848409f6ec.jpg)

![f9ba853b8f6d218b20f4e89a07be766d6474128a29f0640e20d44f33bc763fb6.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/f9ba853b8f6d218b20f4e89a07be766d6474128a29f0640e20d44f33bc763fb6.jpg)

![fdde4306926127219a0ec1b1a824e73eeaa5bd0ad281c7f4443a3c56601ff9cd.jpg](../icml_results/5510_Do%20Vision%20and%20Text%20Cues%20Exhibit%20Evidential%20Coupling_%20UFO_%20A%20Benchmark%20for%20Compositional%20Multimodal%20R/tables/fdde4306926127219a0ec1b1a824e73eeaa5bd0ad281c7f4443a3c56601ff9cd.jpg)

<a id="paper-5463"></a>

## Prompt Tuning for CLIP on the Pretrained Manifold


### Images

![13849787b453543120bade7ede77f9d20b5d578be44d25113e165d5ad7914e5f.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/13849787b453543120bade7ede77f9d20b5d578be44d25113e165d5ad7914e5f.jpg)

![5cf31d1cdc1f2be5e09ca979e20aad6a294c33c1917aa7a3050b3ea92626a300.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/5cf31d1cdc1f2be5e09ca979e20aad6a294c33c1917aa7a3050b3ea92626a300.jpg)

![7407138667713f6b853b3ca07350fe0f2ef2959d24ad6e8684104c3f223a6c1c.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/7407138667713f6b853b3ca07350fe0f2ef2959d24ad6e8684104c3f223a6c1c.jpg)

![9cbf8d5758c0c05fd02de7d075eb5e350707aa50a82fc9ba25c74a1b065b4677.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/9cbf8d5758c0c05fd02de7d075eb5e350707aa50a82fc9ba25c74a1b065b4677.jpg)

![b0e6cdd0eb035540c8fb75af413951cf422f271dc80a649eef1225ce4e4b329a.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/b0e6cdd0eb035540c8fb75af413951cf422f271dc80a649eef1225ce4e4b329a.jpg)

![e4eb812e96d441bbe7a5b026152e7aeb37549a01e66a11ff6fce6a4c3fae318e.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/images/e4eb812e96d441bbe7a5b026152e7aeb37549a01e66a11ff6fce6a4c3fae318e.jpg)

### Tables

![00c8750c599df7bfca4a759ae842ddc432c951496c2ac2071d43c0bbe60f0e63.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/00c8750c599df7bfca4a759ae842ddc432c951496c2ac2071d43c0bbe60f0e63.jpg)

![08f5d2f7c33904f98150a7fa96c64391297d462dd09d55ae24081035d2454b65.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/08f5d2f7c33904f98150a7fa96c64391297d462dd09d55ae24081035d2454b65.jpg)

![0ce222ddf30d124dee19492571822f6dc546e1c7e211980748adf20af06feeba.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/0ce222ddf30d124dee19492571822f6dc546e1c7e211980748adf20af06feeba.jpg)

![0fa58d172e4a3695907c908fbd449a6967fcb6f45e5deb294f7421bb7a8b6853.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/0fa58d172e4a3695907c908fbd449a6967fcb6f45e5deb294f7421bb7a8b6853.jpg)

![18108bfc9ca30cc3c251ce972118a6e141deea5386e5cb6664dbdf7ccaf40efa.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/18108bfc9ca30cc3c251ce972118a6e141deea5386e5cb6664dbdf7ccaf40efa.jpg)

![2670c40838d414bab0725c08f7e1c55d1734e984440b6fc57e765f073f7260a0.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/2670c40838d414bab0725c08f7e1c55d1734e984440b6fc57e765f073f7260a0.jpg)

![5682fc24561acd3a50f919af121c8de97790d93f2b6ff215f5e2f22c9a894486.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/5682fc24561acd3a50f919af121c8de97790d93f2b6ff215f5e2f22c9a894486.jpg)

![666be4dcfcc8da5a2ad9a28049f744b5792fdf561b9b4f6aa0d8afc540d12b9c.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/666be4dcfcc8da5a2ad9a28049f744b5792fdf561b9b4f6aa0d8afc540d12b9c.jpg)

![7cd030db878b831c821eb22cf9c13344c4945cf4649a7c43309924d58ef3aba9.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/7cd030db878b831c821eb22cf9c13344c4945cf4649a7c43309924d58ef3aba9.jpg)

![7d513a165f8a2477d521155c1ee7025982e9c2326dd15c0c6d8a0566686fe61a.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/7d513a165f8a2477d521155c1ee7025982e9c2326dd15c0c6d8a0566686fe61a.jpg)

![7df0babbe95a217b1d49432420a2b7398a2cf35ac0a67b13973054db34b0e7b4.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/7df0babbe95a217b1d49432420a2b7398a2cf35ac0a67b13973054db34b0e7b4.jpg)

![b4552b8d61e092f29d6ebe4a3207340fadf4b64e205212b3da1a2455d9b8a6a8.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/b4552b8d61e092f29d6ebe4a3207340fadf4b64e205212b3da1a2455d9b8a6a8.jpg)

![bb7a26cf567940f4dd2878b4f6be2bae88dae4971fed60a8d782c909dcaff8c7.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/bb7a26cf567940f4dd2878b4f6be2bae88dae4971fed60a8d782c909dcaff8c7.jpg)

![c040ab35ed50e57ec7301005ddb47fa5f709bc4a7b207e3a2c3185abfc3e8b54.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/c040ab35ed50e57ec7301005ddb47fa5f709bc4a7b207e3a2c3185abfc3e8b54.jpg)

![d0780468d2daa3cb540d18783cfc6081d5cd4550695fbf95c19bce8183bd804a.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/d0780468d2daa3cb540d18783cfc6081d5cd4550695fbf95c19bce8183bd804a.jpg)

![d132747b4edb4051b418825055a6e5f37812db55de4b424ebbd6763ef49bfef1.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/d132747b4edb4051b418825055a6e5f37812db55de4b424ebbd6763ef49bfef1.jpg)

![d2a560e68179eb6cc9da7179b1ca2c454c76bc0544f59ebd0eb6f5e7abc87009.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/d2a560e68179eb6cc9da7179b1ca2c454c76bc0544f59ebd0eb6f5e7abc87009.jpg)

![e316ca6dd28395031e1d2da66476418db31f481e6853f7882385b26b8ad450a2.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/e316ca6dd28395031e1d2da66476418db31f481e6853f7882385b26b8ad450a2.jpg)

![e7d1259d03e4dbd988ce8a5b194024d5901728d98ff41e073ee5cde4b64bb67f.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/e7d1259d03e4dbd988ce8a5b194024d5901728d98ff41e073ee5cde4b64bb67f.jpg)

![f62fed9367069e8010785e5d9011a6dd7c737673c53026b1b07d16f6a036ff6e.jpg](../icml_results/5511_Prompt%20Tuning%20for%20CLIP%20on%20the%20Pretrained%20Manifold/tables/f62fed9367069e8010785e5d9011a6dd7c737673c53026b1b07d16f6a036ff6e.jpg)

<a id="paper-5464"></a>

## Merge to Remember: Sharpness-Aware Isotropic Merging for Continual Learning


### Images

![0b0f9f0493c7cf8596df88b0984eaaf002ccf81aeceace14100995cdd35b8cb5.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/0b0f9f0493c7cf8596df88b0984eaaf002ccf81aeceace14100995cdd35b8cb5.jpg)

![3f9d348633fc0cc52366fcb7ffbcbb309a9d58c5f71e6f019fb4fd9c64289981.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/3f9d348633fc0cc52366fcb7ffbcbb309a9d58c5f71e6f019fb4fd9c64289981.jpg)

![5645eaa25164aa69f8aa58e7fdac7c1a99d2b79f91544a8a0759d86db87de576.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/5645eaa25164aa69f8aa58e7fdac7c1a99d2b79f91544a8a0759d86db87de576.jpg)

![8b330fff3364b2c36454da4b5101a4c19486a707c3cbb8a007c9acbdfd620e49.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/8b330fff3364b2c36454da4b5101a4c19486a707c3cbb8a007c9acbdfd620e49.jpg)

![93a321041c5da7bffbcee6d346e3a4efb2332ba2e04672c9fb6baaccf11bd127.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/93a321041c5da7bffbcee6d346e3a4efb2332ba2e04672c9fb6baaccf11bd127.jpg)

![ce5806f2c8d584186734f3f437c1a6241547281836455099962b884220b410d6.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/ce5806f2c8d584186734f3f437c1a6241547281836455099962b884220b410d6.jpg)

![f9317b7444761b58b19d4cd2f91b9d27a29c20a8d3f60ddd4a8e865e8734ccae.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/images/f9317b7444761b58b19d4cd2f91b9d27a29c20a8d3f60ddd4a8e865e8734ccae.jpg)

### Tables

![0ba15bdfbe0fcb2fcac51c2d79bf5968b3a130fe9fd42e9210fa788f65a3f18a.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/0ba15bdfbe0fcb2fcac51c2d79bf5968b3a130fe9fd42e9210fa788f65a3f18a.jpg)

![0e190f0167ff16efe4cc75b988d9b0f8316996cf5c028b2e26b8056ecef965a2.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/0e190f0167ff16efe4cc75b988d9b0f8316996cf5c028b2e26b8056ecef965a2.jpg)

![2145ad09a544a07570bf8322d3d54af6a84444db7d7fc3461ba1f1f3146386c6.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/2145ad09a544a07570bf8322d3d54af6a84444db7d7fc3461ba1f1f3146386c6.jpg)

![33e448215ed611ace17c9d40f742b3ba4b78f7ab436fc4fb894ffc95f1c8719d.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/33e448215ed611ace17c9d40f742b3ba4b78f7ab436fc4fb894ffc95f1c8719d.jpg)

![436c8cd91a299a5d4987e83645f59b162895dbd13adcc7adcfe3fad696f800a9.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/436c8cd91a299a5d4987e83645f59b162895dbd13adcc7adcfe3fad696f800a9.jpg)

![45ca6a5aaf643ad50b364922be25a828a667cd07689b280fe1b1efdcebcf301b.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/45ca6a5aaf643ad50b364922be25a828a667cd07689b280fe1b1efdcebcf301b.jpg)

![570c942524ae0a0090cbc975dbcd90e8b2c538d4c77f024146dddb6e4036e302.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/570c942524ae0a0090cbc975dbcd90e8b2c538d4c77f024146dddb6e4036e302.jpg)

![6e68c0beb2acd1e5f800b3892f2c3c72781661b5c0c769b2eb8ad37ff1cf9f3f.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/6e68c0beb2acd1e5f800b3892f2c3c72781661b5c0c769b2eb8ad37ff1cf9f3f.jpg)

![73b8038bf2fce4e08585e9776bdbfa44a442f542c7f235994bba2a9b481e8a75.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/73b8038bf2fce4e08585e9776bdbfa44a442f542c7f235994bba2a9b481e8a75.jpg)

![7545c9065f200246a2d1646dd4290011492403ea538e6149809413604479fd61.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/7545c9065f200246a2d1646dd4290011492403ea538e6149809413604479fd61.jpg)

![7a28461e842f9de015dd2e433bb071e5bde95f757dd842d05abd77ec9fb1bf53.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/7a28461e842f9de015dd2e433bb071e5bde95f757dd842d05abd77ec9fb1bf53.jpg)

![8e7fb0ada25ec89523c966a6101b7e2f35b7e4725813f2b854cdbed1e464c488.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/8e7fb0ada25ec89523c966a6101b7e2f35b7e4725813f2b854cdbed1e464c488.jpg)

![9552b8bef3486519081e04a712ab86f4492c4357b3db8cb78511daabb64147b8.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/9552b8bef3486519081e04a712ab86f4492c4357b3db8cb78511daabb64147b8.jpg)

![9b70c4c3079b6c0f299768f27c0fdaaec0cc5d16538abe4e9a49b4a88ad7fa41.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/9b70c4c3079b6c0f299768f27c0fdaaec0cc5d16538abe4e9a49b4a88ad7fa41.jpg)

![aa4a5e9508f7f9f271898c41bc796312e0727cb6400e478b5e762fa47b86b60d.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/aa4a5e9508f7f9f271898c41bc796312e0727cb6400e478b5e762fa47b86b60d.jpg)

![ae071bcbba419f0ed1e267907d66ead53d66bb262f4b1bdb666ee5192e4959ee.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/ae071bcbba419f0ed1e267907d66ead53d66bb262f4b1bdb666ee5192e4959ee.jpg)

![b23dfce7c80a5e8d76c421296fac5598aab62207a17160b8778f22669b54a774.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/b23dfce7c80a5e8d76c421296fac5598aab62207a17160b8778f22669b54a774.jpg)

![b35954a95776d3ac137ce9b7a3253dfeb93625ff90739785a68f9e3ba1f69952.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/b35954a95776d3ac137ce9b7a3253dfeb93625ff90739785a68f9e3ba1f69952.jpg)

![c308159a0197912aebb9f823aaf95a872b04462a4a660751d06706b991e3e726.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/c308159a0197912aebb9f823aaf95a872b04462a4a660751d06706b991e3e726.jpg)

![e034ded20b503a153841877bf6a84212350cab1b719439ce972ca325cfa9518f.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/e034ded20b503a153841877bf6a84212350cab1b719439ce972ca325cfa9518f.jpg)

![e4b814289cdeb0a8e1d5e1e5b4d1da7fa63279bb59570b54731f077586f87628.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/e4b814289cdeb0a8e1d5e1e5b4d1da7fa63279bb59570b54731f077586f87628.jpg)

![f3a6e4c1369418815be513c27e6aa0dbb44aa7104de907f90e6865b5538382bb.jpg](../icml_results/5512_Merge%20to%20Remember_%20Sharpness-Aware%20Isotropic%20Merging%20for%20Continual%20Learning/tables/f3a6e4c1369418815be513c27e6aa0dbb44aa7104de907f90e6865b5538382bb.jpg)

<a id="paper-5465"></a>

## Factored Causal Representation Learning for Robust Reward Modeling in RLHF


### Images

![049c74bf0533a29ec45bbaa09077ba1bf5c8e8ad1ff83622061a8f82f00aa496.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/049c74bf0533a29ec45bbaa09077ba1bf5c8e8ad1ff83622061a8f82f00aa496.jpg)

![4aed55603995efb26826b090f102da15b78115afdad5d0eaefc2fe4aae33845c.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/4aed55603995efb26826b090f102da15b78115afdad5d0eaefc2fe4aae33845c.jpg)

![5d34b8979685f7ca961017adf6b25cd1572830900c132b76030e03938a72cb75.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/5d34b8979685f7ca961017adf6b25cd1572830900c132b76030e03938a72cb75.jpg)

![8e1de9a7a71808b228266d7946c72e0b6acac4e8aba2d167aa67f220e04e690c.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/8e1de9a7a71808b228266d7946c72e0b6acac4e8aba2d167aa67f220e04e690c.jpg)

![999b5f569ff9dbebaaea0df8a248f13ca1c7e2fa974f15a2656fb14ae178d018.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/999b5f569ff9dbebaaea0df8a248f13ca1c7e2fa974f15a2656fb14ae178d018.jpg)

![a614571c3376ef9401cdd4edd30fcedd4de2960134e7372bcbe0c5c09913b7b1.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/a614571c3376ef9401cdd4edd30fcedd4de2960134e7372bcbe0c5c09913b7b1.jpg)

![ac89d013ef99afc601fef3cb61dc296644719b74937efa2fd9a9fee46b331191.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/ac89d013ef99afc601fef3cb61dc296644719b74937efa2fd9a9fee46b331191.jpg)

![df215fa645cded1f5f3625615ce9aefd6a0dab909a65a1cfaa95f53fcd093170.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/images/df215fa645cded1f5f3625615ce9aefd6a0dab909a65a1cfaa95f53fcd093170.jpg)

### Tables

![1c1aaf5c711550f2a3034f48c981b50be7f256fe8ce9d209bb230d59b26901da.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/1c1aaf5c711550f2a3034f48c981b50be7f256fe8ce9d209bb230d59b26901da.jpg)

![1eb3e23b0a4ecbbfa01f92a2ab5c0120a8d54287fc405d12e48ee947f8ba9603.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/1eb3e23b0a4ecbbfa01f92a2ab5c0120a8d54287fc405d12e48ee947f8ba9603.jpg)

![2b65d887b4b2ad59896da3a49fb5690667be07236c26f96f567a1ad741499b0d.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/2b65d887b4b2ad59896da3a49fb5690667be07236c26f96f567a1ad741499b0d.jpg)

![2d972b43bfd0f982e7b1fa99ee7877d2e6c781928427c5cb1570a6c28276fecf.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/2d972b43bfd0f982e7b1fa99ee7877d2e6c781928427c5cb1570a6c28276fecf.jpg)

![39fb3d9d4f3e62500954478b4c124a9e2b9b0c8efb734bdfccd19154457773c3.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/39fb3d9d4f3e62500954478b4c124a9e2b9b0c8efb734bdfccd19154457773c3.jpg)

![6fe350953c20fd36aa526a30dd61fb6baf8a08d9f3fdecc8245963c59cfb56ca.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/6fe350953c20fd36aa526a30dd61fb6baf8a08d9f3fdecc8245963c59cfb56ca.jpg)

![759f591c58789917c098e5eb3310a34aae03014cb4c4fcc621673063aa2b17b3.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/759f591c58789917c098e5eb3310a34aae03014cb4c4fcc621673063aa2b17b3.jpg)

![75e8292f421b44401be21f7ad0bd8847e542983e3ff07182fe769e24fcd5939d.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/75e8292f421b44401be21f7ad0bd8847e542983e3ff07182fe769e24fcd5939d.jpg)

![914d1c25574ab65d2dbd86b22b8b4ace90054a4888c931a9031d806bd8d7db10.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/914d1c25574ab65d2dbd86b22b8b4ace90054a4888c931a9031d806bd8d7db10.jpg)

![92cbd533b460b7b9a04054d49e693dfbac92b451e646c6071fe6702d165c3363.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/92cbd533b460b7b9a04054d49e693dfbac92b451e646c6071fe6702d165c3363.jpg)

![98c5e00bb1e94db00fdd8c5721cda4bf9ca57a60d642e0d1569765e80a19d0db.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/98c5e00bb1e94db00fdd8c5721cda4bf9ca57a60d642e0d1569765e80a19d0db.jpg)

![a9acc4a4fc12178a4b2c65ccdbcdab517e6a8a880d02883a6242508dcc9dee1d.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/a9acc4a4fc12178a4b2c65ccdbcdab517e6a8a880d02883a6242508dcc9dee1d.jpg)

![aafb2d55c2c28d77e82b83a3c4d17fc85427538b97f481a40c2de818f0cb3b0d.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/aafb2d55c2c28d77e82b83a3c4d17fc85427538b97f481a40c2de818f0cb3b0d.jpg)

![c0ed0ee995da10f5c5a2c03fd0115ced68bfe0f907573232d5b6e15a2edd362f.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/c0ed0ee995da10f5c5a2c03fd0115ced68bfe0f907573232d5b6e15a2edd362f.jpg)

![d284706d2a8b9c91b5595b46639fd012970a1617d452670720be24865c936196.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/d284706d2a8b9c91b5595b46639fd012970a1617d452670720be24865c936196.jpg)

![f23ef87e525cd8b417c040b5f5c3793c34276313654b60863ccda8e97e03bd56.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/f23ef87e525cd8b417c040b5f5c3793c34276313654b60863ccda8e97e03bd56.jpg)

![ff24e851d3a55ea07b0d82ae108bf507a6705968583df5dfdd6af3103dc291be.jpg](../icml_results/5513_Factored%20Causal%20Representation%20Learning%20for%20Robust%20Reward%20Modeling%20in%20RLHF/tables/ff24e851d3a55ea07b0d82ae108bf507a6705968583df5dfdd6af3103dc291be.jpg)

<a id="paper-5466"></a>

## Decentralized Online Convex Optimization with Efficient Communication: Improved Algorithm and Lower Bounds


### Images

![0489d3700a87c281beb32d3f459732f81acdbf34a9a02290145a0b0bf658e09b.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/0489d3700a87c281beb32d3f459732f81acdbf34a9a02290145a0b0bf658e09b.jpg)

![2222377f72032e472580420d98778352e7323f7aa3f634cd38460880d5707378.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/2222377f72032e472580420d98778352e7323f7aa3f634cd38460880d5707378.jpg)

![4e68305430311a1e209373030bcc3a3377db44a65727cd331c7e0a0e0e0a94b3.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/4e68305430311a1e209373030bcc3a3377db44a65727cd331c7e0a0e0e0a94b3.jpg)

![6948e901f9159cd5fbd6979eb3e7fe5ffc405d6ddb3e8f009d1b34f641c714a6.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/6948e901f9159cd5fbd6979eb3e7fe5ffc405d6ddb3e8f009d1b34f641c714a6.jpg)

![894291559f52902309d29d8add9bdd9a2e6744c55afa6143008ff2fa1669b514.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/894291559f52902309d29d8add9bdd9a2e6744c55afa6143008ff2fa1669b514.jpg)

![b4846432144c42cd10ae90882e2717db45861098a717b19dc1cca89762278bdd.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/b4846432144c42cd10ae90882e2717db45861098a717b19dc1cca89762278bdd.jpg)

![bf8a64826c9e86d21568325d40da73d78141e5a879152fa33076145d9dcfafb9.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/bf8a64826c9e86d21568325d40da73d78141e5a879152fa33076145d9dcfafb9.jpg)

![bf9de34e8472ef5c0ddb379bac73d79ebbc2aa024b3b1a9149fa84d5386c8502.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/bf9de34e8472ef5c0ddb379bac73d79ebbc2aa024b3b1a9149fa84d5386c8502.jpg)

![f50653bfd7340e90fb487237ce5414f21e7baf46f72cf899fa73b1bcea125a82.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/images/f50653bfd7340e90fb487237ce5414f21e7baf46f72cf899fa73b1bcea125a82.jpg)

### Tables

![ad2513b502cee4e3aa0751a4bf92d9df6ceb3f521adfa3711eb1e037a9703c45.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/tables/ad2513b502cee4e3aa0751a4bf92d9df6ceb3f521adfa3711eb1e037a9703c45.jpg)

![e95ed65a3cc752c4010fc224404ffaaa6a89124bc5cb50104c73ee0cbdfa0cba.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/tables/e95ed65a3cc752c4010fc224404ffaaa6a89124bc5cb50104c73ee0cbdfa0cba.jpg)

![f6a1aae5f90ee0cea2595e996ddbc8a39c287f9d32687e1df440b476b866232e.jpg](../icml_results/5514_Decentralized%20Online%20Convex%20Optimization%20with%20Efficient%20Communication_%20Improved%20Algorithm%20and%20Lower%20/tables/f6a1aae5f90ee0cea2595e996ddbc8a39c287f9d32687e1df440b476b866232e.jpg)

<a id="paper-5467"></a>

## SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations


### Images

![56dfbfde2ec4706c1c67c2e9ec43de39fee55956ca5f35ac3b51dc0faa521056.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/images/56dfbfde2ec4706c1c67c2e9ec43de39fee55956ca5f35ac3b51dc0faa521056.jpg)

![7f0e7b83227fcb731071cc06b3f8facdc9d6cc8067bab4213d7a11feca8ae130.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/images/7f0e7b83227fcb731071cc06b3f8facdc9d6cc8067bab4213d7a11feca8ae130.jpg)

![b0f017936d18f868302459e4e24ef67239f25dbe651d729766373d0bb85090fb.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/images/b0f017936d18f868302459e4e24ef67239f25dbe651d729766373d0bb85090fb.jpg)

![ecfd362141fd1e8e96b7f039f22f5f7f8dfa56fbd86255373c7b34e64e068a23.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/images/ecfd362141fd1e8e96b7f039f22f5f7f8dfa56fbd86255373c7b34e64e068a23.jpg)

### Tables

![00c44d2b69065da71142a8cf9f35c5c3666f9daa4ee746f16fe81033afd127ab.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/00c44d2b69065da71142a8cf9f35c5c3666f9daa4ee746f16fe81033afd127ab.jpg)

![06cd671dc78e9e1172747c0e5bfbebb4ced7a4a1a16973d720779095b5d7d93e.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/06cd671dc78e9e1172747c0e5bfbebb4ced7a4a1a16973d720779095b5d7d93e.jpg)

![09cd4566b87a3f2dc0aa28b1ea52c123473a96245c623e04a8e749062fafe8d5.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/09cd4566b87a3f2dc0aa28b1ea52c123473a96245c623e04a8e749062fafe8d5.jpg)

![18bf8021c2c5475e1eb060e744339fe4027636eeced6ad93fbee955815ace59a.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/18bf8021c2c5475e1eb060e744339fe4027636eeced6ad93fbee955815ace59a.jpg)

![241366aad86db6a76512ad0c4eecd42590851fb1954d7550382700e5c963dc3b.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/241366aad86db6a76512ad0c4eecd42590851fb1954d7550382700e5c963dc3b.jpg)

![2b9dddeb9d89ce5e46b476cb0c38a6aa74526b89e46966c02b8ed2d8fc538d6a.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/2b9dddeb9d89ce5e46b476cb0c38a6aa74526b89e46966c02b8ed2d8fc538d6a.jpg)

![302a4da36490d0aba7950f09d3584eb61c924926397c8b12fdca868acb211269.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/302a4da36490d0aba7950f09d3584eb61c924926397c8b12fdca868acb211269.jpg)

![3272db96dd0586dbe7d9b7f9f48201ce52b210c38ebcc369614295a0294119aa.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/3272db96dd0586dbe7d9b7f9f48201ce52b210c38ebcc369614295a0294119aa.jpg)

![32d4c508b8a9f73ae4ff7676360b5507b8f38f6abeb742eeba67bfb80aa92f99.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/32d4c508b8a9f73ae4ff7676360b5507b8f38f6abeb742eeba67bfb80aa92f99.jpg)

![421fce4ef7d347271a9f583aa1913fe13170d905eeefb9ab53f6f95f063a3c98.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/421fce4ef7d347271a9f583aa1913fe13170d905eeefb9ab53f6f95f063a3c98.jpg)

![529baaa940dcdfc80a77ecfdae25c13b6f3b615085ececebb43c6f81231c7e3e.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/529baaa940dcdfc80a77ecfdae25c13b6f3b615085ececebb43c6f81231c7e3e.jpg)

![5430e75ad16b5e06e58ea6d3d39f57661aa3a8bc619c1692666b3d803dd20f07.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/5430e75ad16b5e06e58ea6d3d39f57661aa3a8bc619c1692666b3d803dd20f07.jpg)

![599b9d20679c4e46f6ce04da656fd92d8d025f3eac860cce796001410695b569.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/599b9d20679c4e46f6ce04da656fd92d8d025f3eac860cce796001410695b569.jpg)

![5d8ce84c3c51ea9a3f15f1d87f7db7af3ba6b466128da1843223501004f2a33c.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/5d8ce84c3c51ea9a3f15f1d87f7db7af3ba6b466128da1843223501004f2a33c.jpg)

![83752354f252eb81c57be72f5917bedac854a691da4fa11b0cb79a137f0eda86.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/83752354f252eb81c57be72f5917bedac854a691da4fa11b0cb79a137f0eda86.jpg)

![8fc18b05aefe61456d303253ef867e36e49dcda006a1e986e206133e461d956e.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/8fc18b05aefe61456d303253ef867e36e49dcda006a1e986e206133e461d956e.jpg)

![9b944e8665a345b48697e98946b63467c6383108d6dd5553aeaf69441c944f04.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/9b944e8665a345b48697e98946b63467c6383108d6dd5553aeaf69441c944f04.jpg)

![aeb55aaf875652248445f71df87680adc22e64b447e14041fdad71804f46a076.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/aeb55aaf875652248445f71df87680adc22e64b447e14041fdad71804f46a076.jpg)

![b27a392638c940f8a6e06a957d312a5913f49c03eefa073f71e643f97589fc9a.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/b27a392638c940f8a6e06a957d312a5913f49c03eefa073f71e643f97589fc9a.jpg)

![b4a16851cd51202dcf618af58209b0e3a17630063a6793007f5ad48026be8e57.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/b4a16851cd51202dcf618af58209b0e3a17630063a6793007f5ad48026be8e57.jpg)

![b7a1de7747c4cfefb76ad02390bbc43dff99caaf807032d49a368d425b1aac59.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/b7a1de7747c4cfefb76ad02390bbc43dff99caaf807032d49a368d425b1aac59.jpg)

![cae563a12d9fb9e60cd836c557c93962e5b1cbe676aed28c8991db47c0c1061b.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/cae563a12d9fb9e60cd836c557c93962e5b1cbe676aed28c8991db47c0c1061b.jpg)

![ccbd281b3899d4ca47d8e1583a0d199632021018d7798db602d20a23d973aa5c.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/ccbd281b3899d4ca47d8e1583a0d199632021018d7798db602d20a23d973aa5c.jpg)

![d0c0dc6e573bd0d6e22bb11a919caeef189ead2b0afa5c59248d08e0dd835a2c.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/d0c0dc6e573bd0d6e22bb11a919caeef189ead2b0afa5c59248d08e0dd835a2c.jpg)

![db5f62c1a0fed3a0e182a9f29363d4c915ea49fb14480a336a81b75e257000af.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/db5f62c1a0fed3a0e182a9f29363d4c915ea49fb14480a336a81b75e257000af.jpg)

![e28b571b7e06335028f30570b9b95a1a8b4b68d29f1824dd6d161b8ad30b6fa9.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/e28b571b7e06335028f30570b9b95a1a8b4b68d29f1824dd6d161b8ad30b6fa9.jpg)

![f15ebf4225857cbf24fe48bbdc49999860aa7e256d7eb54231fc1ab115cc44c3.jpg](../icml_results/5515_SPEAR_%20A%20Unified%20SSL%20Framework%20for%20Learning%20Speech%20and%20Audio%20Representations/tables/f15ebf4225857cbf24fe48bbdc49999860aa7e256d7eb54231fc1ab115cc44c3.jpg)

<a id="paper-5468"></a>

## SIPO: Stabilized and Improved Preference Optimization for Aligning Diffusion Models


### Images

![1b50c3dd804f9c2c54ade4c1539ebfb6de63e4f6e3cec55cf9cb4434f919fbc3.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/1b50c3dd804f9c2c54ade4c1539ebfb6de63e4f6e3cec55cf9cb4434f919fbc3.jpg)

![1e04c7bf524b122352de7b044bf92ecdceb2c0a8a9f0f80842d2e977f19ea7c7.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/1e04c7bf524b122352de7b044bf92ecdceb2c0a8a9f0f80842d2e977f19ea7c7.jpg)

![41aaa408b7829350b8a9443a9dd4826abf20d8bd05512a83b8ea9a703a2d2bfb.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/41aaa408b7829350b8a9443a9dd4826abf20d8bd05512a83b8ea9a703a2d2bfb.jpg)

![592f763995bf331753766ab8aeecaca5a1c9eaabdfa970e6dc74edd09a5415ab.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/592f763995bf331753766ab8aeecaca5a1c9eaabdfa970e6dc74edd09a5415ab.jpg)

![6889407c104c93fc51131440d3c62a35cad1f570967f0f6620116520a6da6f1f.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/6889407c104c93fc51131440d3c62a35cad1f570967f0f6620116520a6da6f1f.jpg)

![8b7b680d1558653a9e3f632864e9317fa32f469706bdf6f76f0bb1e3dab67105.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/8b7b680d1558653a9e3f632864e9317fa32f469706bdf6f76f0bb1e3dab67105.jpg)

![902cb684557d4cb39d4afcd235d5cd796f3419636f788a9db6ba4a6864e67029.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/902cb684557d4cb39d4afcd235d5cd796f3419636f788a9db6ba4a6864e67029.jpg)

![e70ba90f960e66f11f631cce48fdc1e1ceeb6594c0c890e13e36a87dc567502d.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/e70ba90f960e66f11f631cce48fdc1e1ceeb6594c0c890e13e36a87dc567502d.jpg)

![f2496b2ab4ade10fa81ae327f0b4ee7f0c2e9762e220eda46c1f1be5defe272d.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/images/f2496b2ab4ade10fa81ae327f0b4ee7f0c2e9762e220eda46c1f1be5defe272d.jpg)

### Tables

![17159044cd4d225f00590e956abd1f1923305609c699a482fd5a003875076e15.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/17159044cd4d225f00590e956abd1f1923305609c699a482fd5a003875076e15.jpg)

![22add882cfca0e37edd8d9972db036e6f84003a64a4ef531d38172cd08027526.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/22add882cfca0e37edd8d9972db036e6f84003a64a4ef531d38172cd08027526.jpg)

![44cedc3d593e304e1063dbf02edd0e90e2bd1cf1892b9af01a567f9df897461f.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/44cedc3d593e304e1063dbf02edd0e90e2bd1cf1892b9af01a567f9df897461f.jpg)

![50286edcc0f8002d41597407a8033c0ea957376bffb79defc8bffd08ea7b96a4.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/50286edcc0f8002d41597407a8033c0ea957376bffb79defc8bffd08ea7b96a4.jpg)

![656c4c65e426d2151cacf033887d85d3a74e53b649da7df71a533297893cc9d0.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/656c4c65e426d2151cacf033887d85d3a74e53b649da7df71a533297893cc9d0.jpg)

![992584526d99b94a70c3f0a3e526cd1b0a83e1c0c143fb6b1171b439ad48e268.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/992584526d99b94a70c3f0a3e526cd1b0a83e1c0c143fb6b1171b439ad48e268.jpg)

![a0a3ace3cebfeda6a5f7f044a8c273b8004e66de70d2c9df015fdccb6f310abc.jpg](../icml_results/5516_SIPO_%20Stabilized%20and%20Improved%20Preference%20Optimization%20for%20Aligning%20Diffusion%20Models/tables/a0a3ace3cebfeda6a5f7f044a8c273b8004e66de70d2c9df015fdccb6f310abc.jpg)

<a id="paper-5469"></a>

## Criterion-Conditional In-Context Learning: Evaluating Criterion-Shift Adaptation in Vision-Language Models


### Images

![0641fc3b17ae1231da1f5702056668856e72c365898fc4c444a7aa7797c68acd.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/0641fc3b17ae1231da1f5702056668856e72c365898fc4c444a7aa7797c68acd.jpg)

![0d889492289fb3e498a3887626d9c8c1069ba0b5c15b17dac4929a19f916da8a.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/0d889492289fb3e498a3887626d9c8c1069ba0b5c15b17dac4929a19f916da8a.jpg)

![3defbb3bddcbb099f04cad8c41cd8985625227f311c4aff64726187d74dad5a2.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/3defbb3bddcbb099f04cad8c41cd8985625227f311c4aff64726187d74dad5a2.jpg)

![627e0197d1794bcdecb4281ef64c337d4f777c9a1fa2c32ab5b078e1873f78e1.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/627e0197d1794bcdecb4281ef64c337d4f777c9a1fa2c32ab5b078e1873f78e1.jpg)

![86c40d1ab98d210eb53336271447bc297dd7e5a5f901024f7657d82936e67b0c.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/86c40d1ab98d210eb53336271447bc297dd7e5a5f901024f7657d82936e67b0c.jpg)

![882c34d6c61dfe054ac97609873ab6d9a1e062e144459dfc3a55630233cf78a2.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/882c34d6c61dfe054ac97609873ab6d9a1e062e144459dfc3a55630233cf78a2.jpg)

![a18f646660020d2ed5143da9a7c0f1b59991a4e69222e1c22471043244fc51c4.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/a18f646660020d2ed5143da9a7c0f1b59991a4e69222e1c22471043244fc51c4.jpg)

![b79efc77c637889705f19e0a95c72b4533e318d1992b504a449f03dbbc45493d.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/b79efc77c637889705f19e0a95c72b4533e318d1992b504a449f03dbbc45493d.jpg)

![c2b82e55e307e4ba47c690d369137971900d6109cd1ef30298cc86c2c3554075.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/c2b82e55e307e4ba47c690d369137971900d6109cd1ef30298cc86c2c3554075.jpg)

![c3b1a02190ca91b4969f57020ff7238ccd7b284913bfe0822da73fc2b0d42b1e.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/c3b1a02190ca91b4969f57020ff7238ccd7b284913bfe0822da73fc2b0d42b1e.jpg)

![d437c2f6f5f79f5c16c13664776924dc0aceed834652ad00c4053b0b0bfa7633.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/d437c2f6f5f79f5c16c13664776924dc0aceed834652ad00c4053b0b0bfa7633.jpg)

![de770ef3091e5f6b6196af3b9b6e9fb8af34b75efc9be266d37a16e5beb093f3.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/de770ef3091e5f6b6196af3b9b6e9fb8af34b75efc9be266d37a16e5beb093f3.jpg)

![f03151ea95c88c842c7543214b7ee1cd823bece9b1e2950347ad5536d76daa3a.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/images/f03151ea95c88c842c7543214b7ee1cd823bece9b1e2950347ad5536d76daa3a.jpg)

### Tables

![15fe1095ee12ab27edc1433627d4cebfabe66685365de6d48f0d2b47633ef3c7.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/15fe1095ee12ab27edc1433627d4cebfabe66685365de6d48f0d2b47633ef3c7.jpg)

![240a1f8dbdcf7d5fa6218933d4bf858fd40d3cf058a2450656341ef26026b8ff.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/240a1f8dbdcf7d5fa6218933d4bf858fd40d3cf058a2450656341ef26026b8ff.jpg)

![26f8fda96f02c4eb83f6afb677dfc8cde127c18ed5701e8a97de009f810d3493.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/26f8fda96f02c4eb83f6afb677dfc8cde127c18ed5701e8a97de009f810d3493.jpg)

![32f057afaa8295ab810ea0f21eae00b5c2d8d7f4a311ffd7823611ec4957390b.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/32f057afaa8295ab810ea0f21eae00b5c2d8d7f4a311ffd7823611ec4957390b.jpg)

![661276224c1ce861c52906a23136a824e63b651208dd1c345982411755f4747d.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/661276224c1ce861c52906a23136a824e63b651208dd1c345982411755f4747d.jpg)

![6f2169fbf2a03fd491ed1a778790446830a0b025f0f9b65aed59f922b7e12dcc.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/6f2169fbf2a03fd491ed1a778790446830a0b025f0f9b65aed59f922b7e12dcc.jpg)

![72d579c0c4a0edbc2d05f8597d2ce1ecc577641f13a2210f83f205dbd2282da2.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/72d579c0c4a0edbc2d05f8597d2ce1ecc577641f13a2210f83f205dbd2282da2.jpg)

![7cfa8ee3491fa6866b027b73cb6f5aa83a07d93981aeb8c0ea178f5f6e02e507.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/7cfa8ee3491fa6866b027b73cb6f5aa83a07d93981aeb8c0ea178f5f6e02e507.jpg)

![89d69b5302278e321468826a90f9208ae7b6075229e2a8e8d70df47445502282.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/89d69b5302278e321468826a90f9208ae7b6075229e2a8e8d70df47445502282.jpg)

![be78a58762df7983d3b562ce3fdc356a899cb09344666044e906dedbd28a1b2d.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/be78a58762df7983d3b562ce3fdc356a899cb09344666044e906dedbd28a1b2d.jpg)

![bf30787456cc0abb13496641804a8923ba35a1912177767d372794759f60757e.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/bf30787456cc0abb13496641804a8923ba35a1912177767d372794759f60757e.jpg)

![c2ed5a14db43ba2e29e292a2afa5061c470769dbadfb995dc4b4c3169a30ed32.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/c2ed5a14db43ba2e29e292a2afa5061c470769dbadfb995dc4b4c3169a30ed32.jpg)

![c3a52642e72ff5df3e3a7df878aacab753e84d6e94b50381112c6eebcc52e36e.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/c3a52642e72ff5df3e3a7df878aacab753e84d6e94b50381112c6eebcc52e36e.jpg)

![dd121d03112acb6a130098638ac422450b7218bef449c81f2ac512ebf3d7d7df.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/dd121d03112acb6a130098638ac422450b7218bef449c81f2ac512ebf3d7d7df.jpg)

![e1e4c504607ea36dd2b09454a557413bfe674409d8f56dd989533c17c6386a4b.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/e1e4c504607ea36dd2b09454a557413bfe674409d8f56dd989533c17c6386a4b.jpg)

![ff2a80fe55c3b7b3c37a84334e284e63cfde274602dd4e462802149dbdb33f23.jpg](../icml_results/5517_Criterion-Conditional%20In-Context%20Learning_%20Evaluating%20Criterion-Shift%20Adaptation%20in%20Vision-Language%20/tables/ff2a80fe55c3b7b3c37a84334e284e63cfde274602dd4e462802149dbdb33f23.jpg)

<a id="paper-5470"></a>

## DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing


### Images

![59f04f49448919f8fa9614ffaba296658145a37011e2544863b3b9b25e7136a9.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/59f04f49448919f8fa9614ffaba296658145a37011e2544863b3b9b25e7136a9.jpg)

![6b554f5ffebcca3a6262970dfdaf2018be1a4450a98899539e5d9db6693b9cfa.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/6b554f5ffebcca3a6262970dfdaf2018be1a4450a98899539e5d9db6693b9cfa.jpg)

![6e974c854ef2e5d56562218b8e7c32498ec93830c44fd6f84c34527b290f2012.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/6e974c854ef2e5d56562218b8e7c32498ec93830c44fd6f84c34527b290f2012.jpg)

![7be84079158d31ce85fd69065f1a01602180e63ea520e4ef8c6c1a5546e1de02.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/7be84079158d31ce85fd69065f1a01602180e63ea520e4ef8c6c1a5546e1de02.jpg)

![94d9388b4ee5cd2422d8cb31f4a93848e3b2c6d914fac60379f4fec4fe24fc48.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/94d9388b4ee5cd2422d8cb31f4a93848e3b2c6d914fac60379f4fec4fe24fc48.jpg)

![ad7930dd1221904565f6bd8b7516d582e749047495141dee7cde27d1a9728376.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/ad7930dd1221904565f6bd8b7516d582e749047495141dee7cde27d1a9728376.jpg)

![b030f9a2c51a17a801627008e8db719d297026a73c2d0bb22aa9e99b0544e98e.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/b030f9a2c51a17a801627008e8db719d297026a73c2d0bb22aa9e99b0544e98e.jpg)

![bca5471c471b4d8efd3ce313a573a473e7f275c5188acf940f4dbd8268ed4260.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/bca5471c471b4d8efd3ce313a573a473e7f275c5188acf940f4dbd8268ed4260.jpg)

![d716175872563fdad2ff8fef4e44ff2df743f9715d42cfc32d2ad212d449d0df.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/d716175872563fdad2ff8fef4e44ff2df743f9715d42cfc32d2ad212d449d0df.jpg)

![e140fa1efda36c53b0d708e1629d2341fe0db952f834369689424c211ddf656b.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/e140fa1efda36c53b0d708e1629d2341fe0db952f834369689424c211ddf656b.jpg)

![e5d52ac9d8d07e33df654322dc10a0ce4d856c680957bebbb1be96cda653c044.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/e5d52ac9d8d07e33df654322dc10a0ce4d856c680957bebbb1be96cda653c044.jpg)

![eac67f3e89687a675c8ae79f4e228f597d3ffbc9553cd75f6a96ad7f4ebb5883.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/images/eac67f3e89687a675c8ae79f4e228f597d3ffbc9553cd75f6a96ad7f4ebb5883.jpg)

### Tables

![22a7d4552bf6878e5bef8c2b2af07909f393c61e5e21ce7e816e7a972c5ecab3.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/tables/22a7d4552bf6878e5bef8c2b2af07909f393c61e5e21ce7e816e7a972c5ecab3.jpg)

![26781b55b1cd450b4b49b261a060b2d65778570e7a57723a8f6eb40935079b09.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/tables/26781b55b1cd450b4b49b261a060b2d65778570e7a57723a8f6eb40935079b09.jpg)

![512e60c61993067d520029d64d92d490f8522bc34d85e7bffe56ff8b5a2b8ec2.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/tables/512e60c61993067d520029d64d92d490f8522bc34d85e7bffe56ff8b5a2b8ec2.jpg)

![77ee5af6df302d31df066892cdba9f58bdb3a722cd6ccf9b4fed67643b912afb.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/tables/77ee5af6df302d31df066892cdba9f58bdb3a722cd6ccf9b4fed67643b912afb.jpg)

![812571d649654a4a0e5dc904d7bac300ec966661063b12d23e84636c43748489.jpg](../icml_results/5518_DirectEdit_%20Step-Level%20Accurate%20Inversion%20for%20Flow-Based%20Image%20Editing/tables/812571d649654a4a0e5dc904d7bac300ec966661063b12d23e84636c43748489.jpg)

<a id="paper-5471"></a>

## Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents


### Images

![071a88e88a0d1148df027344785e610588fc0c71ccc9faae48468e1d71425e25.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/071a88e88a0d1148df027344785e610588fc0c71ccc9faae48468e1d71425e25.jpg)

![2096ed4c4a62a946cdab3107c173b2c4d876e321c63fc8148e266f80c5e091fe.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/2096ed4c4a62a946cdab3107c173b2c4d876e321c63fc8148e266f80c5e091fe.jpg)

![37c401507b75ee0ab7cace5e8a40ccdb64a2ff83c602514e801d7ac0efd4188a.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/37c401507b75ee0ab7cace5e8a40ccdb64a2ff83c602514e801d7ac0efd4188a.jpg)

![43bdc71733a229c8012c007ad9cf4937e7b6a9e99fb4ba284910fd363f8404a0.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/43bdc71733a229c8012c007ad9cf4937e7b6a9e99fb4ba284910fd363f8404a0.jpg)

![4f916f69bb47a06720a5f8556ebedaf88a9a0faada418c37a5e54fcd99644b64.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/4f916f69bb47a06720a5f8556ebedaf88a9a0faada418c37a5e54fcd99644b64.jpg)

![79906252fbaf441d306bbe87c12e99c9a12fb2c1996a2f0544537bbc4c65aba0.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/79906252fbaf441d306bbe87c12e99c9a12fb2c1996a2f0544537bbc4c65aba0.jpg)

![87e3c16643e2217960af7d7a432f67fe8eda0383ec96fb67ab4fafa39462fbb1.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/87e3c16643e2217960af7d7a432f67fe8eda0383ec96fb67ab4fafa39462fbb1.jpg)

![9445285855230a802599378d36ab6eda8ca2e450974462be69f138a3689f4b34.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/9445285855230a802599378d36ab6eda8ca2e450974462be69f138a3689f4b34.jpg)

![a2a9f462022b0232e3affc9cd1d9037752b6d8d6c49c287a317ff5ee1550875d.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/a2a9f462022b0232e3affc9cd1d9037752b6d8d6c49c287a317ff5ee1550875d.jpg)

![e80e31cece5474fe4207439b304ab65cdc6c0f37992b288e7ef9f4c55943edd8.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/e80e31cece5474fe4207439b304ab65cdc6c0f37992b288e7ef9f4c55943edd8.jpg)

![f50c483d6145fc15abbbc2556becf7869b54ef44663dff2a13fca06ae2c1daa3.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/images/f50c483d6145fc15abbbc2556becf7869b54ef44663dff2a13fca06ae2c1daa3.jpg)

### Tables

![07334f8010011018ae6721829cc366a1eca6268f08868dd4361cc6e5824a2245.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/07334f8010011018ae6721829cc366a1eca6268f08868dd4361cc6e5824a2245.jpg)

![140e85868f9e4e69694d60d9bd3d536e13847a865a2628f11f69d80b7ff2c421.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/140e85868f9e4e69694d60d9bd3d536e13847a865a2628f11f69d80b7ff2c421.jpg)

![3f811ede9f8a87c8556fbf916d613ab4f3f19820451c5c3f641f4861ed8a17c5.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/3f811ede9f8a87c8556fbf916d613ab4f3f19820451c5c3f641f4861ed8a17c5.jpg)

![4b9e89750b85d079127985d4f59ff7e69b16284010b91ed3b8a2068747ccec7f.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/4b9e89750b85d079127985d4f59ff7e69b16284010b91ed3b8a2068747ccec7f.jpg)

![a7ec72ac7600f0bda2492e322a8d50cc804fddca7955f510bc977d80b925c6f3.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/a7ec72ac7600f0bda2492e322a8d50cc804fddca7955f510bc977d80b925c6f3.jpg)

![c4b03a8d4b6562a1c6981b555cb998dd84c731c08820e8148dd614d078938923.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/c4b03a8d4b6562a1c6981b555cb998dd84c731c08820e8148dd614d078938923.jpg)

![c5001d3ee3ef7aa22f2853cb200cd8e45231e148880238e80cb3cc1e5c27f4f5.jpg](../icml_results/5519_Think%20Fast%20and%20Slow_%20Step-Level%20Cognitive%20Depth%20Adaptation%20for%20LLM%20Agents/tables/c5001d3ee3ef7aa22f2853cb200cd8e45231e148880238e80cb3cc1e5c27f4f5.jpg)

<a id="paper-5472"></a>

## Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments


### Images

![090b63fa3770e0ef30396c0a957df44509f13e4fb49774696aaf2ca94ada4596.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/images/090b63fa3770e0ef30396c0a957df44509f13e4fb49774696aaf2ca94ada4596.jpg)

![8894a28d98c180c4de4f2deb41276b0b8a77307edf478c42d5836c14ccd8cebe.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/images/8894a28d98c180c4de4f2deb41276b0b8a77307edf478c42d5836c14ccd8cebe.jpg)

![9cf2dffe3e7dbc5d3208bf6b3226ed0d4ee01581a827bf731c6fbce5f76e02ab.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/images/9cf2dffe3e7dbc5d3208bf6b3226ed0d4ee01581a827bf731c6fbce5f76e02ab.jpg)

![facb43f6562059054b39b28775ecd44c754e53ce5606809fedd397e3e8671cc1.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/images/facb43f6562059054b39b28775ecd44c754e53ce5606809fedd397e3e8671cc1.jpg)

### Tables

![25608d612d5e591fb6cd55c1d5b51fd593e75b0ed83a53a953d1410ed7c7fe57.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/25608d612d5e591fb6cd55c1d5b51fd593e75b0ed83a53a953d1410ed7c7fe57.jpg)

![2e3a14742821475de1072f16b19bd8be0c29726e908907f997fe09c2d99c4244.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/2e3a14742821475de1072f16b19bd8be0c29726e908907f997fe09c2d99c4244.jpg)

![3515264fc8ec409389ca4392a41efb76ad61a540b19294a38b67c360ce454ff1.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/3515264fc8ec409389ca4392a41efb76ad61a540b19294a38b67c360ce454ff1.jpg)

![376f199708acd0d4fa018b69b3eb5ddfa29463d151a94c18c1d242f31fcec8b1.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/376f199708acd0d4fa018b69b3eb5ddfa29463d151a94c18c1d242f31fcec8b1.jpg)

![8e28b9cd2924f79955ea9e17e9619ab46f73474b3e6a925b37f3fd750d24164f.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/8e28b9cd2924f79955ea9e17e9619ab46f73474b3e6a925b37f3fd750d24164f.jpg)

![c2043969a14f873b2831c7700c019554d81dce7890aa98c793c8df675c115026.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/c2043969a14f873b2831c7700c019554d81dce7890aa98c793c8df675c115026.jpg)

![e01b336231851d19b28410e995f0e58171009af664bcd5735d5b89cbcaf5d914.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/e01b336231851d19b28410e995f0e58171009af664bcd5735d5b89cbcaf5d914.jpg)

![e82ce265e5a1b5d0aca1c9e647250873da7e0ea944defddf4b488ab692135d27.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/e82ce265e5a1b5d0aca1c9e647250873da7e0ea944defddf4b488ab692135d27.jpg)

![edfc93708c8e67518218bba1256c30b74a1081a1fd1d5150cba2d37562af95ee.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/edfc93708c8e67518218bba1256c30b74a1081a1fd1d5150cba2d37562af95ee.jpg)

![f4939384f8d3678d49bb6b2c6defca5a0fadbe070e6c2c02908e1d66956b5b8f.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/f4939384f8d3678d49bb6b2c6defca5a0fadbe070e6c2c02908e1d66956b5b8f.jpg)

![f4af667aadb1efcd3be8d634dd6ce4d931b9dca1aedb25b85a99b31910830b27.jpg](../icml_results/5520_Turning%20Drift%20into%20Constraint_%20Robust%20Reasoning%20Alignment%20in%20Non-Stationary%20Multi-Stream%20Environment/tables/f4af667aadb1efcd3be8d634dd6ce4d931b9dca1aedb25b85a99b31910830b27.jpg)

<a id="paper-5473"></a>

## SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval


### Images

![01d6aa2d06986391a90ac60a3023fd44020c686fbce217aee2fe48b3d193ec56.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/01d6aa2d06986391a90ac60a3023fd44020c686fbce217aee2fe48b3d193ec56.jpg)

![01ec3992547ac5f4556a3764d24dcb1a525d5bb747bcfc100c889aea541aa6b5.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/01ec3992547ac5f4556a3764d24dcb1a525d5bb747bcfc100c889aea541aa6b5.jpg)

![203985552b1bb97f2bfb0f78aca88034b0639ef7547e56aaf289bf041b7b17f1.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/203985552b1bb97f2bfb0f78aca88034b0639ef7547e56aaf289bf041b7b17f1.jpg)

![2362954cc0613ecbb7b26d9e5497dd7716ded18eee31bcb6ace10d890b29fba0.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/2362954cc0613ecbb7b26d9e5497dd7716ded18eee31bcb6ace10d890b29fba0.jpg)

![2d1cb2fe72ac09e4c7629d1cbe8eb040be18a66a07e0ea423205a0b5270b3f3b.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/2d1cb2fe72ac09e4c7629d1cbe8eb040be18a66a07e0ea423205a0b5270b3f3b.jpg)

![2dd8e8b0f334646c9634facf62bdb6607bd914dddb7d5994f2d2d147dabd7222.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/2dd8e8b0f334646c9634facf62bdb6607bd914dddb7d5994f2d2d147dabd7222.jpg)

![3efe0aa1b2b6ff0692ad2f554c068ae683120f4b8446473a65333b219d6150cb.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/3efe0aa1b2b6ff0692ad2f554c068ae683120f4b8446473a65333b219d6150cb.jpg)

![3f2adc307c2f04edd3bfb9afff74e88d8b8fd0193988c5602026ed43fe7a51dd.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/3f2adc307c2f04edd3bfb9afff74e88d8b8fd0193988c5602026ed43fe7a51dd.jpg)

![41729fad9d229f654c60c69c6bc95c81ba1a38775b56a77846322ce1cdec016e.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/41729fad9d229f654c60c69c6bc95c81ba1a38775b56a77846322ce1cdec016e.jpg)

![5023a824417936f5c9163c0bf7622c5922050e33014caf418e9157cf3c7d3ffe.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/5023a824417936f5c9163c0bf7622c5922050e33014caf418e9157cf3c7d3ffe.jpg)

![557f924689af61d823dad684109e1d0e3daae43b2ce40a2d6eb7953f0ebf880d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/557f924689af61d823dad684109e1d0e3daae43b2ce40a2d6eb7953f0ebf880d.jpg)

![6f0c5bffb2cb228f7b3213c4585561f1782dff35e26296d3eaf888ef3537bceb.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/6f0c5bffb2cb228f7b3213c4585561f1782dff35e26296d3eaf888ef3537bceb.jpg)

![70e0af33c5643841ea5af2da4dd09fdca79b410be23d4483734339beaf91d9f8.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/70e0af33c5643841ea5af2da4dd09fdca79b410be23d4483734339beaf91d9f8.jpg)

![73df90f89c55810d0f1ba16116fb3ba3639f80a92c167fe3aad5140a6ac4a825.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/73df90f89c55810d0f1ba16116fb3ba3639f80a92c167fe3aad5140a6ac4a825.jpg)

![75b69ce0a071765943c948a3fa3859b3aaf02fb86cd0ef4b4789619617a25a61.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/75b69ce0a071765943c948a3fa3859b3aaf02fb86cd0ef4b4789619617a25a61.jpg)

![770ab83119c82c78d395dc6630f310f8493534095184ea0159c318e03f9eb19b.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/770ab83119c82c78d395dc6630f310f8493534095184ea0159c318e03f9eb19b.jpg)

![8b90451d7445eddeec52e954cc365c64969c1218472680a16ebfdbf699ef9469.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/8b90451d7445eddeec52e954cc365c64969c1218472680a16ebfdbf699ef9469.jpg)

![8b9d83e31236067dc00e0696bfe1f60e7181cab52029fb2af0cba20065c7c22a.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/8b9d83e31236067dc00e0696bfe1f60e7181cab52029fb2af0cba20065c7c22a.jpg)

![95dfac9af89be5d3eab8c65027dd40f3f32b2fb9db3c040082ec2116db51c35f.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/95dfac9af89be5d3eab8c65027dd40f3f32b2fb9db3c040082ec2116db51c35f.jpg)

![a580e0ea016d82016f02232456319c1ca9cc656d6d0e22daa34905d102e237ce.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/a580e0ea016d82016f02232456319c1ca9cc656d6d0e22daa34905d102e237ce.jpg)

![b884bbe8515b71faa07c4f64d9575d8e343a9057f008335265859adfa470c725.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/b884bbe8515b71faa07c4f64d9575d8e343a9057f008335265859adfa470c725.jpg)

![c741fadd54cb57ead6a0b685a82047dc1bbb380e9b1b28dec5f5b7379c0cb853.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/c741fadd54cb57ead6a0b685a82047dc1bbb380e9b1b28dec5f5b7379c0cb853.jpg)

![c7b7187bb835d74bda74f85558a515173526204eb42eecffe530122af4956aa0.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/c7b7187bb835d74bda74f85558a515173526204eb42eecffe530122af4956aa0.jpg)

![dc2eec258ad69e255a7b8d28b915b612820f71f3a14d705015fbeb9c9dad65b7.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/dc2eec258ad69e255a7b8d28b915b612820f71f3a14d705015fbeb9c9dad65b7.jpg)

![ea311db60701c496ff567c83f1e91dd359166f7c3f2b377cbf59e55c555b071e.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/ea311db60701c496ff567c83f1e91dd359166f7c3f2b377cbf59e55c555b071e.jpg)

![eae4f5e92041bcca4a7d88968aca681ed3afc2bb10692515d9accdc4ad38408d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/eae4f5e92041bcca4a7d88968aca681ed3afc2bb10692515d9accdc4ad38408d.jpg)

![f747c9fa62fac7a3c46d1da05bea017ca2e4d71e0989b3eed179dc9d3fd66b73.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/f747c9fa62fac7a3c46d1da05bea017ca2e4d71e0989b3eed179dc9d3fd66b73.jpg)

![fad42ac2c43893a4562633b02b1fd9ef2c5899862cdad24bec01fc9c2e3c503a.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/fad42ac2c43893a4562633b02b1fd9ef2c5899862cdad24bec01fc9c2e3c503a.jpg)

![fbf6ec597e0bb9e68ef6650a6c8df77b075daaaf7357b01afdff140c4a6235d1.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/fbf6ec597e0bb9e68ef6650a6c8df77b075daaaf7357b01afdff140c4a6235d1.jpg)

![fca2909985ea9dbb21320f215f3b9e8c08221fe422dd31b4de3a06944c55063d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/images/fca2909985ea9dbb21320f215f3b9e8c08221fe422dd31b4de3a06944c55063d.jpg)

### Tables

![0855e94defc7e728c893498a1d9cbd6c1af5b21d7713587bab02e9c2f2635eec.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/0855e94defc7e728c893498a1d9cbd6c1af5b21d7713587bab02e9c2f2635eec.jpg)

![142b8af5458d703d856cbf124d6d98cc96abd3112d6ef9af7e96693a08791eaf.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/142b8af5458d703d856cbf124d6d98cc96abd3112d6ef9af7e96693a08791eaf.jpg)

![2958c5c252d483567cc97638c5698e9fb36761cc4b9f39e7fc3403389aaed55a.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/2958c5c252d483567cc97638c5698e9fb36761cc4b9f39e7fc3403389aaed55a.jpg)

![30ed66f1db71c8d5b5a8e7b4265d0fd32d50cfab6e4dd8f906864b1414d994a7.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/30ed66f1db71c8d5b5a8e7b4265d0fd32d50cfab6e4dd8f906864b1414d994a7.jpg)

![39c5e62165d3b9d0d2245073371012fd027de8726c76ef78900ecb96d99c0760.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/39c5e62165d3b9d0d2245073371012fd027de8726c76ef78900ecb96d99c0760.jpg)

![571a48e3cfbc0bd4d2eb52834bb506f383d0fb4f0aec42a414928e981215ce9c.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/571a48e3cfbc0bd4d2eb52834bb506f383d0fb4f0aec42a414928e981215ce9c.jpg)

![62d8cad36ad50f5a3b2f6d2e885dbafd76ee5a6e666f5ac6381d547c3493c42d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/62d8cad36ad50f5a3b2f6d2e885dbafd76ee5a6e666f5ac6381d547c3493c42d.jpg)

![63996bcd2469df459e2312d1050542e07fd92a0906b0da38e7496e4f6ba0ff8d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/63996bcd2469df459e2312d1050542e07fd92a0906b0da38e7496e4f6ba0ff8d.jpg)

![930b361e7491402d54b6bf707e3fae1517ee0167bdf1e2d28f45cc4d13b4936c.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/930b361e7491402d54b6bf707e3fae1517ee0167bdf1e2d28f45cc4d13b4936c.jpg)

![9c7e9fbf54634c67105c91afa862fc21ca3c20dc3c4759aea647a431fdd6fc85.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/9c7e9fbf54634c67105c91afa862fc21ca3c20dc3c4759aea647a431fdd6fc85.jpg)

![b6601921ade1a2897cb49d9d3febb2b194303b8db52d44b2fd0bab21aeb6979d.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/b6601921ade1a2897cb49d9d3febb2b194303b8db52d44b2fd0bab21aeb6979d.jpg)

![eb15ec15181681e4b1c66c874de1897a629a3aff92a2b091e0afb8753a4013ae.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/eb15ec15181681e4b1c66c874de1897a629a3aff92a2b091e0afb8753a4013ae.jpg)

![fd1bfae1460d2efdbb022d90bafe0bcbf6f905108bb12353323156e908c8e55f.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/fd1bfae1460d2efdbb022d90bafe0bcbf6f905108bb12353323156e908c8e55f.jpg)

![fdb6eb3f47c7227e1de39fd67b9c459f2cea03f88b4311dee9f79675c3f92daf.jpg](../icml_results/5521_SOLAR_%20Self-supervised%20Joint%20Learning%20for%20Symmetric%20Multimodal%20Retrieval/tables/fdb6eb3f47c7227e1de39fd67b9c459f2cea03f88b4311dee9f79675c3f92daf.jpg)

<a id="paper-5474"></a>

## Distinguishable Deletion: Unifying Knowledge Erasure and Refusal for Large Language Model Unlearning


### Images

![058d850b83720bf219df4ae41762121b7abe5d0df2dbf46e06ea0c4fe26937c7.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/058d850b83720bf219df4ae41762121b7abe5d0df2dbf46e06ea0c4fe26937c7.jpg)

![07bc8709a12bb24ea735837004391118ee475fe660b845a5d54d3514a8e71bf1.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/07bc8709a12bb24ea735837004391118ee475fe660b845a5d54d3514a8e71bf1.jpg)

![237c4a663ceee9ac97972549f25530c09becaa9bf60f6145def6639d2ed04982.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/237c4a663ceee9ac97972549f25530c09becaa9bf60f6145def6639d2ed04982.jpg)

![24054bb580a2864a618f5335a6193d932350d895665d33aedfd2b628a0bf1d4d.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/24054bb580a2864a618f5335a6193d932350d895665d33aedfd2b628a0bf1d4d.jpg)

![5c7048a1de2885276c445761efa5960f00a4ff80ac744114e35a782435bff12e.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/5c7048a1de2885276c445761efa5960f00a4ff80ac744114e35a782435bff12e.jpg)

![5cc93003d063b238898a05d1b8e3445d2be458c680a6cdb21372f91f391227cc.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/5cc93003d063b238898a05d1b8e3445d2be458c680a6cdb21372f91f391227cc.jpg)

![88830f55a2ff989683056f8467c8ad7c9133d427f86ffd48a16000794f8d3e2f.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/88830f55a2ff989683056f8467c8ad7c9133d427f86ffd48a16000794f8d3e2f.jpg)

![de420efa3a2b34c640a469c55b223b58bd88a6852902619942e80d7204a45dca.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/de420efa3a2b34c640a469c55b223b58bd88a6852902619942e80d7204a45dca.jpg)

![e5582bb55b4b15c8e2e722df09a69a20367a42cc216c2086826572523d8c27aa.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/e5582bb55b4b15c8e2e722df09a69a20367a42cc216c2086826572523d8c27aa.jpg)

![f1b673325403bf3fe35299673950227c87c13cbf3a3dd56e6ae2e8a9496b5a3a.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/images/f1b673325403bf3fe35299673950227c87c13cbf3a3dd56e6ae2e8a9496b5a3a.jpg)

### Tables

![08c887ecbb5e21488d1f03892cd5c005a1bf377dc0aef659e33ed7c3c925f948.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/08c887ecbb5e21488d1f03892cd5c005a1bf377dc0aef659e33ed7c3c925f948.jpg)

![0dfefd2e74c2730f06cfd6abfb102ef69dcdecee785e307d255cf89d2044daf1.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/0dfefd2e74c2730f06cfd6abfb102ef69dcdecee785e307d255cf89d2044daf1.jpg)

![0ea494fbabdb724ed8422de318a22a197ace4f2d74fc0ba54948fc60ba561e61.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/0ea494fbabdb724ed8422de318a22a197ace4f2d74fc0ba54948fc60ba561e61.jpg)

![18640b17e9bf54d2bed57bd52ba13bc5f7cf5a85a6e8ed360acd4ef1fe8e0448.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/18640b17e9bf54d2bed57bd52ba13bc5f7cf5a85a6e8ed360acd4ef1fe8e0448.jpg)

![2042fa104424d1ebea5f78955ad993588aae45807da7c6966f2c71093223b8f2.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/2042fa104424d1ebea5f78955ad993588aae45807da7c6966f2c71093223b8f2.jpg)

![243d36fb558c393aadf21eea7890ede8b2d3cc4eb5760a02ba1296b2570d7e79.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/243d36fb558c393aadf21eea7890ede8b2d3cc4eb5760a02ba1296b2570d7e79.jpg)

![2edd4f0596c32024243ef91d6162fab0aaae17a3f362abb9e17f4ce903d6ea7c.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/2edd4f0596c32024243ef91d6162fab0aaae17a3f362abb9e17f4ce903d6ea7c.jpg)

![2f1c19b93993f3e7fc79a26aeff899e68857f708719c9b8d412a2d2fdc06a10a.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/2f1c19b93993f3e7fc79a26aeff899e68857f708719c9b8d412a2d2fdc06a10a.jpg)

![31a5e3018fa647c957d3cd6cf661cf0d09bc51ed4a9b7dc4904518358ef3686c.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/31a5e3018fa647c957d3cd6cf661cf0d09bc51ed4a9b7dc4904518358ef3686c.jpg)

![5d3212ae23f6acc69cd34d0a7e10d461a0cfe36297c6a4120041f26f0d90fafa.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/5d3212ae23f6acc69cd34d0a7e10d461a0cfe36297c6a4120041f26f0d90fafa.jpg)

![676212db9cb67343a2d763aa3427809b245c289aa8a4a5bf34b8388e094f4a4b.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/676212db9cb67343a2d763aa3427809b245c289aa8a4a5bf34b8388e094f4a4b.jpg)

![7dda685e6732600f7fcecad28701e992c1306296bf8ba64a54cdf053ca631357.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/7dda685e6732600f7fcecad28701e992c1306296bf8ba64a54cdf053ca631357.jpg)

![982ae63827c3877c00f1fdff74ace751bc144f16ae95b8fec1b31559ae65398f.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/982ae63827c3877c00f1fdff74ace751bc144f16ae95b8fec1b31559ae65398f.jpg)

![9d0abd519dc2bd72935293b63a66099e1fcd34d3c9a005829aa826b1b60d2835.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/9d0abd519dc2bd72935293b63a66099e1fcd34d3c9a005829aa826b1b60d2835.jpg)

![b29d23db18c3ccd2a0e412c8fc835c5f8354651d08b4d20885bec3194a8c661b.jpg](../icml_results/5522_Distinguishable%20Deletion_%20Unifying%20Knowledge%20Erasure%20and%20Refusal%20for%20Large%20Language%20Model%20Unlearning/tables/b29d23db18c3ccd2a0e412c8fc835c5f8354651d08b4d20885bec3194a8c661b.jpg)

<a id="paper-5475"></a>

## Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization


### Images

![e05a424fdebc1309874762f8288997ccfe7aa2cbd18755ee8226257e705560c3.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/images/e05a424fdebc1309874762f8288997ccfe7aa2cbd18755ee8226257e705560c3.jpg)

![eb7a77ab7c99f7d6b55037a304c3fc910261775062f36d014bd1232a90c6c682.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/images/eb7a77ab7c99f7d6b55037a304c3fc910261775062f36d014bd1232a90c6c682.jpg)

![f927741e305edcf7be5fc3ce4ec216f0d76aede2460d13db1a51775f17d2baab.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/images/f927741e305edcf7be5fc3ce4ec216f0d76aede2460d13db1a51775f17d2baab.jpg)

### Tables

![03bb985679b8f447d8589d22141e9c92feb349f8b527256f76e73cb51abb0f80.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/tables/03bb985679b8f447d8589d22141e9c92feb349f8b527256f76e73cb51abb0f80.jpg)

![477a807cd5ac7342e90d46349f7a6cf85106d90b6b0e9b9301e3a9d1457933e7.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/tables/477a807cd5ac7342e90d46349f7a6cf85106d90b6b0e9b9301e3a9d1457933e7.jpg)

![5518a501b3f7ed2c0553290c1e996ba130f834da443ca3cd976194f1b2869c95.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/tables/5518a501b3f7ed2c0553290c1e996ba130f834da443ca3cd976194f1b2869c95.jpg)

![9b56d528eacaac5f8c38ba75d27f3add3d41eb7e911716fe69d3664ee8db957f.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/tables/9b56d528eacaac5f8c38ba75d27f3add3d41eb7e911716fe69d3664ee8db957f.jpg)

![f726108c35897c5c7df8972050b0cb4044cbbdf2284431b704d3a7edfd8904ee.jpg](../icml_results/5523_Support-Proximity%20Augmented%20Diffusion%20Estimation%20for%20Offline%20Black-Box%20Optimization/tables/f726108c35897c5c7df8972050b0cb4044cbbdf2284431b704d3a7edfd8904ee.jpg)

<a id="paper-5476"></a>

## Accelerated Dual Method for Distributed Optimization: An Inexact-Gradient View of Local Updates


### Images

![2633ab0c67aa23e606268de882d10ca52cb85d70b5d8e49d834d52940f4f3b24.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/2633ab0c67aa23e606268de882d10ca52cb85d70b5d8e49d834d52940f4f3b24.jpg)

![55606727808e4ecaf0973cdf06b828fbff7c6874812fff404a10889728b2fbaa.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/55606727808e4ecaf0973cdf06b828fbff7c6874812fff404a10889728b2fbaa.jpg)

![63753635e93fc8201e560754b65d3ef5de6a5cad8cd3bfd877e018a59a18fe3b.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/63753635e93fc8201e560754b65d3ef5de6a5cad8cd3bfd877e018a59a18fe3b.jpg)

![7c11f8d736766613c2cd49ce04b2938ca4460f7cb21059560024cb3f8231905a.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/7c11f8d736766613c2cd49ce04b2938ca4460f7cb21059560024cb3f8231905a.jpg)

![864fa67b7be64f064c05203eb5a86bccb7dcd3c6ee7070e52551bff7691e813d.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/864fa67b7be64f064c05203eb5a86bccb7dcd3c6ee7070e52551bff7691e813d.jpg)

![b456a5f277b9b258c2069994f5d11a3797dc5cdd1a38be6a735319b35f136367.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/images/b456a5f277b9b258c2069994f5d11a3797dc5cdd1a38be6a735319b35f136367.jpg)

### Tables

![34be7490172b4c59de81cf6251e3d53c7270a9c16d6172e457153a740f78dfca.jpg](../icml_results/5524_Accelerated%20Dual%20Method%20for%20Distributed%20Optimization_%20An%20Inexact-Gradient%20View%20of%20Local%20Updates/tables/34be7490172b4c59de81cf6251e3d53c7270a9c16d6172e457153a740f78dfca.jpg)

<a id="paper-5477"></a>

## Brain Networks Should Be Learned, Not Constructed


### Images

![14d41381fd53fd03aba7c8186401de620355286ef6a08a54c5b3f53a233c49ff.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/14d41381fd53fd03aba7c8186401de620355286ef6a08a54c5b3f53a233c49ff.jpg)

![2c4dc2c09cc35a58839d3c8ed4624ced04acc43a3c761788d5534a4349a03bd0.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/2c4dc2c09cc35a58839d3c8ed4624ced04acc43a3c761788d5534a4349a03bd0.jpg)

![3dab37bbc4351ef1df6952aac4672656a8c5f5d9a3103fa53377bdad2b60f936.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/3dab37bbc4351ef1df6952aac4672656a8c5f5d9a3103fa53377bdad2b60f936.jpg)

![4741f8541b5d4410e9b048ca158bd7c5dfbc31a39bcc76b1618e08f97fced2a4.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/4741f8541b5d4410e9b048ca158bd7c5dfbc31a39bcc76b1618e08f97fced2a4.jpg)

![4b13491126af0fcb2d97d7e76ac15b1706e3f2d9e56ee187cd7a07595917d2ae.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/4b13491126af0fcb2d97d7e76ac15b1706e3f2d9e56ee187cd7a07595917d2ae.jpg)

![5949c80e5a9361979f1b138665711cbbbb13dd05734753f961003a3cdb074751.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/5949c80e5a9361979f1b138665711cbbbb13dd05734753f961003a3cdb074751.jpg)

![60abfeacf54c4a7546e39ad1012220a436e3b4f872e9eff2c5f875cf27ace3b0.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/60abfeacf54c4a7546e39ad1012220a436e3b4f872e9eff2c5f875cf27ace3b0.jpg)

![7143a7bb9aed47bf560a4c17ccbd8fe4669b2da8e2d83e48077dfc48465e3d50.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/7143a7bb9aed47bf560a4c17ccbd8fe4669b2da8e2d83e48077dfc48465e3d50.jpg)

![7360d6abb87a8494caddc3dd69b2a2c27211c2cf2007ecf84aad5b08752d26df.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/7360d6abb87a8494caddc3dd69b2a2c27211c2cf2007ecf84aad5b08752d26df.jpg)

![77810a26792120365fbb5f124a28cc56027d71128f9176dfac694b75b581327b.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/77810a26792120365fbb5f124a28cc56027d71128f9176dfac694b75b581327b.jpg)

![8209edbb89ccc8c36fe09ae1294cacdbdb98954d5b4ecb16090c1c1e9678eac1.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/8209edbb89ccc8c36fe09ae1294cacdbdb98954d5b4ecb16090c1c1e9678eac1.jpg)

![9d1830280ffc4b158322130d973ae4b5e50097f2adf1e741b72bdf7da9829cfd.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/9d1830280ffc4b158322130d973ae4b5e50097f2adf1e741b72bdf7da9829cfd.jpg)

![a22c62b1d1c61caff8586bd20b78576b62df0be5b9d238351e0f5ccb76486455.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/a22c62b1d1c61caff8586bd20b78576b62df0be5b9d238351e0f5ccb76486455.jpg)

![aa73050d125996f371c5d180ce49cb1b50f90ac5adeaf969ae86edf260382b8e.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/aa73050d125996f371c5d180ce49cb1b50f90ac5adeaf969ae86edf260382b8e.jpg)

![bb6c74270b5088e2e1c4d64255427d9e0aaae717852b18ed354677f2c40d97b8.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/bb6c74270b5088e2e1c4d64255427d9e0aaae717852b18ed354677f2c40d97b8.jpg)

![d33e71d8c30188c65e893fa39354d8396d04ad19c7f0eeaab9c9f6fb149d640e.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/d33e71d8c30188c65e893fa39354d8396d04ad19c7f0eeaab9c9f6fb149d640e.jpg)

![e3ea5d58c64e8d13da7722f20d6f672ccdafc7ca1691a17c611ba724e26a9b42.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/e3ea5d58c64e8d13da7722f20d6f672ccdafc7ca1691a17c611ba724e26a9b42.jpg)

![ebe0cac21c3046e38adfb124cf27482801aaca199b0beb7da4957f4670519c92.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/ebe0cac21c3046e38adfb124cf27482801aaca199b0beb7da4957f4670519c92.jpg)

![f0e75429a0740a339b68edce62d835b0970d15a67dbb317c329b63beb9733a9c.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/f0e75429a0740a339b68edce62d835b0970d15a67dbb317c329b63beb9733a9c.jpg)

![f8c728a7b96f051471913095a2a0c1db096cb1444f5c81f4473054c50bf23def.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/images/f8c728a7b96f051471913095a2a0c1db096cb1444f5c81f4473054c50bf23def.jpg)

### Tables

![3054181205547ebbf5335a0fe7f22971f1141ff026a29b2537b76654d9d1cfe0.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/3054181205547ebbf5335a0fe7f22971f1141ff026a29b2537b76654d9d1cfe0.jpg)

![374bba080f892549b0daea972f3320d7b3c401435734310a2c15bab70b5ec4e7.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/374bba080f892549b0daea972f3320d7b3c401435734310a2c15bab70b5ec4e7.jpg)

![6ae60e5fd3c3cad730572f09b67723e8c413e1b9b436f1c8c8042f1ea1ef02cd.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/6ae60e5fd3c3cad730572f09b67723e8c413e1b9b436f1c8c8042f1ea1ef02cd.jpg)

![728c0fa70ec6773e85cf8b677da90d39a2722d043785bf9637394ad36eb869d6.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/728c0fa70ec6773e85cf8b677da90d39a2722d043785bf9637394ad36eb869d6.jpg)

![b7a5d7a4ceae90dd0ea253cd008eb7e5f0b0aa773af6371ba26dddd6ce2726ed.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/b7a5d7a4ceae90dd0ea253cd008eb7e5f0b0aa773af6371ba26dddd6ce2726ed.jpg)

![dcdfd25494e465bf215c3fdbea62e4f5626943823f7e4f8fb4b0105e167068fe.jpg](../icml_results/5525_Brain%20Networks%20Should%20Be%20Learned%2C%20Not%20Constructed/tables/dcdfd25494e465bf215c3fdbea62e4f5626943823f7e4f8fb4b0105e167068fe.jpg)

<a id="paper-5478"></a>

## Speech-Audio Compositional Attacks on Multimodal LLMs and Their Defense with SALMONN-Guard

### Images

![038ddb4927de64fd68fe78896359c02a9cb273722e3bfff4b37690f479ff84c3.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/038ddb4927de64fd68fe78896359c02a9cb273722e3bfff4b37690f479ff84c3.jpg)

![31b0ba5d4150cee5a781a068aadcfa335b812aa70a345e5edacdf681238060f0.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/31b0ba5d4150cee5a781a068aadcfa335b812aa70a345e5edacdf681238060f0.jpg)

![c8899ffdb7b308712b85d578998180268ef8823d36febacae70e84f2d6ef446f.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/c8899ffdb7b308712b85d578998180268ef8823d36febacae70e84f2d6ef446f.jpg)

![cb17792a7f2982d4c2983050b7e15a23968c6509b4d9fa8a3cff4b1578804cde.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/cb17792a7f2982d4c2983050b7e15a23968c6509b4d9fa8a3cff4b1578804cde.jpg)

![e4954973ab3e1934bb641a4525392356bdb6638fc2bb06cd6f143c6b7f8105c6.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/e4954973ab3e1934bb641a4525392356bdb6638fc2bb06cd6f143c6b7f8105c6.jpg)

![f408ec9a58e2c92649439ee1463f8f7fac1f91c38aaee74c5149fb801436c797.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/f408ec9a58e2c92649439ee1463f8f7fac1f91c38aaee74c5149fb801436c797.jpg)

![fc2c4f2760b62628a24c605189f0ff980aa7cbad970af3cdaf50525d00414073.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/images/fc2c4f2760b62628a24c605189f0ff980aa7cbad970af3cdaf50525d00414073.jpg)

### Tables

![0a301f01c69fd6fc06b29fac448d1a037bab0fcf9da3be3d68db9ffa08d1f5f9.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/0a301f01c69fd6fc06b29fac448d1a037bab0fcf9da3be3d68db9ffa08d1f5f9.jpg)

![189c98cfca5e6fb86657e340348f2fd5da03b8eee524e99bae7e46adb86677a5.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/189c98cfca5e6fb86657e340348f2fd5da03b8eee524e99bae7e46adb86677a5.jpg)

![2187082c4bcb6434034f34c02a0e515b1b214639e0bc7ef7eedac73ea6408824.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/2187082c4bcb6434034f34c02a0e515b1b214639e0bc7ef7eedac73ea6408824.jpg)

![266da9c013dddfb5761ab93ef7cefe86a7e9bfa60566a8c58bfa66ce948e8142.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/266da9c013dddfb5761ab93ef7cefe86a7e9bfa60566a8c58bfa66ce948e8142.jpg)

![3cdec41037ed47932331619f185d2360e4f666fc657c4e59b5a13553144ab1f3.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/3cdec41037ed47932331619f185d2360e4f666fc657c4e59b5a13553144ab1f3.jpg)

![631b13c02cd867c5a6255d2a5c0feafc22b22461487c3bde1ff5ba6de7b9750b.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/631b13c02cd867c5a6255d2a5c0feafc22b22461487c3bde1ff5ba6de7b9750b.jpg)

![bf8f2c0bceff9b84ba8da135121cf42ad865d6ee01eb5ad1c52eb4135e4e9fd1.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/bf8f2c0bceff9b84ba8da135121cf42ad865d6ee01eb5ad1c52eb4135e4e9fd1.jpg)

![e033f295f4a2f88a20a09aeac7ee55be5af25138277476dcfa99ed9c1d382dd8.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/e033f295f4a2f88a20a09aeac7ee55be5af25138277476dcfa99ed9c1d382dd8.jpg)

![f9014777d2b861b7b42b0f5a0560335410027ff0f3c1379a95ee859634592fe2.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/f9014777d2b861b7b42b0f5a0560335410027ff0f3c1379a95ee859634592fe2.jpg)

![fda2179aabce8ccface65b23a3fb91f853e305b152bf339ccd7960a713305f2e.jpg](../icml_results/5526_Speech-Audio%20Compositional%20Attacks%20on%20Multimodal%20LLMs%20and%20Their%20Defense%20with%20SALMONN-Guard/tables/fda2179aabce8ccface65b23a3fb91f853e305b152bf339ccd7960a713305f2e.jpg)
