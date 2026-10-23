# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 141 of 197

## 目录 (Table of Contents)

1. [Faster Query-Key Learning Sharpens Attention in Self-Attention Models](#paper-4621)
2. [Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads](#paper-4622)
3. [Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning](#paper-4623)
4. [MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems](#paper-4624)
5. [Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Model](#paper-4625)
6. [Operationalising the Superficial Alignment Hypothesis via Task Complexity](#paper-4626)
7. [Equalized Generative Treatment: Matching f-divergences for Fairness in Generative Models](#paper-4627)
8. [Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies](#paper-4628)
9. [GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond](#paper-4629)
10. [Multi-agent imitation learning with function approximation: linear Markov games and beyond](#paper-4630)
11. [Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks](#paper-4631)
12. [PACER: Acyclic Causal Discovery from Large-scale Interventional Data](#paper-4632)
13. [Optimal Domain-Aware Privacy Mechanisms for Synthetic Data Generation](#paper-4633)
14. [Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs](#paper-4634)
15. [Off-Policy Evaluation with Strategic Agents via Local Disclosure](#paper-4635)
16. [Riemannian Optimization for Fair Spectral Clustering](#paper-4636)
17. [$f$-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models](#paper-4637)
18. [New Bounds for Kernel Sums via Fast Spherical Embeddings](#paper-4638)
19. [A Probabilistic Framework for LLM-Based Model Discovery](#paper-4639)
20. [A Decision-Theoretic View of Test-Time Training: When, How Far, and Which Directions to Adapt](#paper-4640)
21. [In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning](#paper-4641)
22. [CAOS: Conformal Aggregation of One-Shot Predictors](#paper-4642)
23. [Hidden in Plain Sight – Class Competition Focuses Attribution Maps](#paper-4643)
24. [Mitigating Conversational Inertia in Multi-Turn Agents](#paper-4644)
25. [Saving Foundation Flow-Matching Priors for Inverse Problems](#paper-4645)
26. [Gradient Inversion Attacks Beyond SGD](#paper-4646)
27. [RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference](#paper-4647)
28. [Beyond Policy Training: Recursive Solution Search from Unannotated Videos](#paper-4648)
29. [Multi-view Consistent Latent Action Learning for World Modeling and Control](#paper-4649)
30. [Video-SVD: Efficient Video Diffusion via Orthogonal Basis Composition](#paper-4650)
31. [OpenMAG: A Comprehensive Benchmark for Multimodal-Attributed Graph](#paper-4651)
32. [A Solver-Free Training Method for Predict-then-Optimize](#paper-4652)
33. [Corrected Samplers for Discrete Flow Models](#paper-4653)

---


<a id="paper-4621"></a>

## Faster Query-Key Learning Sharpens Attention in Self-Attention Models

### Images

![05e268276297b792ce725b30142410bc5ea3b136b09061f3b1743a0938951251.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/05e268276297b792ce725b30142410bc5ea3b136b09061f3b1743a0938951251.jpg)

![116e2463d147b36a75f7ae9bf05956f1b283f34cc65def45b4c63b6da337803b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/116e2463d147b36a75f7ae9bf05956f1b283f34cc65def45b4c63b6da337803b.jpg)

![14817e186ed180ac9458b217a197e207870bfd952e7a39c390f2ab822e5f35a2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/14817e186ed180ac9458b217a197e207870bfd952e7a39c390f2ab822e5f35a2.jpg)

![14b2239d14e31cdb0c12090c58577e5adcb495b6152ad76164404dba2824837e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/14b2239d14e31cdb0c12090c58577e5adcb495b6152ad76164404dba2824837e.jpg)

![1819b78e5307a4ab2b927257498d1f57c6f89951f3313c51f6f798e94dd7e449.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/1819b78e5307a4ab2b927257498d1f57c6f89951f3313c51f6f798e94dd7e449.jpg)

![182ea1d2e77c850dc6afc84893ed7144b814f40bd7ff942c9eb740eb55103e0e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/182ea1d2e77c850dc6afc84893ed7144b814f40bd7ff942c9eb740eb55103e0e.jpg)

![1efb28c91541af639208b68c5cb97f3653d1b8da3dc247a7c6d6486b1f368a5e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/1efb28c91541af639208b68c5cb97f3653d1b8da3dc247a7c6d6486b1f368a5e.jpg)

![207e2baeb8c53294ba8a2ae1fc335f513d7eef1e95150045c821f5029b9c5a09.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/207e2baeb8c53294ba8a2ae1fc335f513d7eef1e95150045c821f5029b9c5a09.jpg)

![21c51ea0eadecda6557527d7c6601231f9fb75ab86fd28e6bc05856c7efc45ae.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/21c51ea0eadecda6557527d7c6601231f9fb75ab86fd28e6bc05856c7efc45ae.jpg)

![28d9143fd8f4667e6e9aa3889b0b776a56f41bc48afccb69f5e9745017d1f44b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/28d9143fd8f4667e6e9aa3889b0b776a56f41bc48afccb69f5e9745017d1f44b.jpg)

![3116089fe7121db001403d8fea2f5c01fc79dc764de40d29b8e6c72b18edb7bc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/3116089fe7121db001403d8fea2f5c01fc79dc764de40d29b8e6c72b18edb7bc.jpg)

![386b4ce2d24805afce0c2d543b3a9d74aa66161c0d6d08665569a5c0f458284f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/386b4ce2d24805afce0c2d543b3a9d74aa66161c0d6d08665569a5c0f458284f.jpg)

![412eb21a0443acb1277ed26915eb3d21ddd03434601e630f637779447725193f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/412eb21a0443acb1277ed26915eb3d21ddd03434601e630f637779447725193f.jpg)

![434bb4d7f8c5f22989c2df0c2aea8d62a1f3386fcd84e73f5bce01270767f82e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/434bb4d7f8c5f22989c2df0c2aea8d62a1f3386fcd84e73f5bce01270767f82e.jpg)

![47948776d2a690c515c8337fc10ad2532e43308cee1f4788fcc6b5066fea2fd7.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/47948776d2a690c515c8337fc10ad2532e43308cee1f4788fcc6b5066fea2fd7.jpg)

![47daee695571ca256f7be87df8386f080033da04bf401552340bfce8a2888d7c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/47daee695571ca256f7be87df8386f080033da04bf401552340bfce8a2888d7c.jpg)

![4a25baadf32eb3859553dff17ec7c566780223ac29fd4fcdad8a44e42495a0dc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/4a25baadf32eb3859553dff17ec7c566780223ac29fd4fcdad8a44e42495a0dc.jpg)

![517cb5336fdc093ec507b983f803ec6a8fd9f34d5dfeef62d87b78e4a3fcfa8b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/517cb5336fdc093ec507b983f803ec6a8fd9f34d5dfeef62d87b78e4a3fcfa8b.jpg)

![58a6ecb914e234370481f3ab932cd74c873ab6f1ffc530447fd700f644ea5cab.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/58a6ecb914e234370481f3ab932cd74c873ab6f1ffc530447fd700f644ea5cab.jpg)

![58cf14cc6f7915901c06604eaf28cea0b3f55a1e1f842d33907f1d9d1d43769c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/58cf14cc6f7915901c06604eaf28cea0b3f55a1e1f842d33907f1d9d1d43769c.jpg)

![779db65c27984745b01f1eaa9d5f91f4cc6e1bffff30a1071b1f4fb9b226bb78.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/779db65c27984745b01f1eaa9d5f91f4cc6e1bffff30a1071b1f4fb9b226bb78.jpg)

![79862ed3397b316a640d4aebcc0d915f2d5680d540872ace815cb42d989ba948.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/79862ed3397b316a640d4aebcc0d915f2d5680d540872ace815cb42d989ba948.jpg)

![83906a08e04fbe82319aba8d0636add8f6c0c6b2ab71479bfa4b2dffdc2c4150.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/83906a08e04fbe82319aba8d0636add8f6c0c6b2ab71479bfa4b2dffdc2c4150.jpg)

![883e35ae95d0c17f392edfcc1e891b0f5a0293f0c7fe2ed2c0efcc0915f7c6e1.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/883e35ae95d0c17f392edfcc1e891b0f5a0293f0c7fe2ed2c0efcc0915f7c6e1.jpg)

![89f58526011ef1cc257e54ff7bb3911773a0676b463e1bb1572764df07c29aad.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/89f58526011ef1cc257e54ff7bb3911773a0676b463e1bb1572764df07c29aad.jpg)

![8b7a9943d035d4e03e98eb0f2ca632970207d7634b313f43d3c6e18031bea613.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/8b7a9943d035d4e03e98eb0f2ca632970207d7634b313f43d3c6e18031bea613.jpg)

![8f95364aa7894ec1e64bc707e8fcb41014712e3f4d6a342afacfcfb0032fc763.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/8f95364aa7894ec1e64bc707e8fcb41014712e3f4d6a342afacfcfb0032fc763.jpg)

![9528e2c22674677335079b955e718020dd9880afa417574bed0499ed502476da.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9528e2c22674677335079b955e718020dd9880afa417574bed0499ed502476da.jpg)

![958c08eb955cf23a1e58b05dfd19b0e3cddcba8f0e2b2b911a91285537e6effe.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/958c08eb955cf23a1e58b05dfd19b0e3cddcba8f0e2b2b911a91285537e6effe.jpg)

![97cdd508ef3ae742bf77585550dd08920e906201e0f97fffb3a1e61a047a6c98.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/97cdd508ef3ae742bf77585550dd08920e906201e0f97fffb3a1e61a047a6c98.jpg)

![9bebb6cb423d6c38c4d57cff6494d18ad0d0e02349f46a8b2512b5a21b533071.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9bebb6cb423d6c38c4d57cff6494d18ad0d0e02349f46a8b2512b5a21b533071.jpg)

![9dd6da29a53832136e06e53ba54c32dae73ac4c49d67395b950a123832502e56.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9dd6da29a53832136e06e53ba54c32dae73ac4c49d67395b950a123832502e56.jpg)

![9efd768c9735ad9c7d1cf97363c63fc2308a081857b2419ebb6d9c4712829a67.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9efd768c9735ad9c7d1cf97363c63fc2308a081857b2419ebb6d9c4712829a67.jpg)

![9f7ada97dff6c9103dd0e7032e9a877dc29d82e83699cc74c9ffb30cb05ee251.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/9f7ada97dff6c9103dd0e7032e9a877dc29d82e83699cc74c9ffb30cb05ee251.jpg)

![a5cbe8333fd2eada57d7c4a5b9eb9cbd9b1fbbcee27f7dbe73a739ec1a3a3645.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/a5cbe8333fd2eada57d7c4a5b9eb9cbd9b1fbbcee27f7dbe73a739ec1a3a3645.jpg)

![b2507b852c080e1c1747092063b50fb7a96afd50daed8751d69292cdb809ae7b.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b2507b852c080e1c1747092063b50fb7a96afd50daed8751d69292cdb809ae7b.jpg)

![b609a60363a1fbe867954deb8690671ae868c4a503e93ebbd2d53fa41676e257.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b609a60363a1fbe867954deb8690671ae868c4a503e93ebbd2d53fa41676e257.jpg)

![b69019b07d97935fe9cd076b9a399ade7dc867b722420add06b2e53cc6315f82.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b69019b07d97935fe9cd076b9a399ade7dc867b722420add06b2e53cc6315f82.jpg)

![b7d07f12294e9cb74d7babb63e4f11814f4c86b3ddead8d6c983f46c2cfab1e1.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/b7d07f12294e9cb74d7babb63e4f11814f4c86b3ddead8d6c983f46c2cfab1e1.jpg)

![be6db83a23199499cd2fd93a9d76fff32f1988e020557ffcda1cca26d3f11c97.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/be6db83a23199499cd2fd93a9d76fff32f1988e020557ffcda1cca26d3f11c97.jpg)

![becb36f67a113027336022658f11b2878b41a984a1380ce23cd632b5d77cb2bb.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/becb36f67a113027336022658f11b2878b41a984a1380ce23cd632b5d77cb2bb.jpg)

![c2685697d3c65e92214721bdd124b3e80a1d80c9e2758ad760a0c8d6ea6cb813.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/c2685697d3c65e92214721bdd124b3e80a1d80c9e2758ad760a0c8d6ea6cb813.jpg)

![c277ee387b862764bd9035be1fcd84700583cecdda4b85ed323615669798bead.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/c277ee387b862764bd9035be1fcd84700583cecdda4b85ed323615669798bead.jpg)

![d137776b45c910422fbc4079473dbcc7631d75485605ebaa2422d3e81eddb3bc.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/d137776b45c910422fbc4079473dbcc7631d75485605ebaa2422d3e81eddb3bc.jpg)

![ec6d960479c36b17fbff246bc73c8c6a9badcb682150213eb7585ffe066464ec.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/ec6d960479c36b17fbff246bc73c8c6a9badcb682150213eb7585ffe066464ec.jpg)

![ee140d8a5e0a958fc5267556e0a12c7279205898bf2826c21a52a0c14b10b862.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/ee140d8a5e0a958fc5267556e0a12c7279205898bf2826c21a52a0c14b10b862.jpg)

![eebabbc75b8f0b5369bee17eace88dab2e6bc9a4884a99d49b04117fe8baed10.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/eebabbc75b8f0b5369bee17eace88dab2e6bc9a4884a99d49b04117fe8baed10.jpg)

![f09d547b747fb8280d3d2938258693891272548c665b0fb70ba4c03a293b9035.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/f09d547b747fb8280d3d2938258693891272548c665b0fb70ba4c03a293b9035.jpg)

![f643d73425c24d0a62a52860bab968b7c7aa4608dac42ed95bab89b9427aafeb.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/f643d73425c24d0a62a52860bab968b7c7aa4608dac42ed95bab89b9427aafeb.jpg)

![fae2d8c729751d9572bbeb469bc757154f5bf422945f2c6de3ade931c2ffd300.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/fae2d8c729751d9572bbeb469bc757154f5bf422945f2c6de3ade931c2ffd300.jpg)

![fbb04f6ae3c90a8665dae09fc753a92a91d9539220cbab4a56e1f176bb23762f.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/images/fbb04f6ae3c90a8665dae09fc753a92a91d9539220cbab4a56e1f176bb23762f.jpg)

### Tables

![3309f2f62ec5bb8a0dcd968049cf9858efd05cb4c8804ec147f6e294c51bdd9c.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/3309f2f62ec5bb8a0dcd968049cf9858efd05cb4c8804ec147f6e294c51bdd9c.jpg)

![4cdeceb90ffa9bef340b5e7a9a84bdf326bfc960790c05eaff4c3635c5ffce9e.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/4cdeceb90ffa9bef340b5e7a9a84bdf326bfc960790c05eaff4c3635c5ffce9e.jpg)

![6257229a1eac9aba3149438cc5fd9d8642c8a3c7248af1d1a3d4028d7d4612a2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/6257229a1eac9aba3149438cc5fd9d8642c8a3c7248af1d1a3d4028d7d4612a2.jpg)

![951b5e3e5568fce43933f8eb621a1944a34030b9da50bbe56f34df3b741e0849.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/951b5e3e5568fce43933f8eb621a1944a34030b9da50bbe56f34df3b741e0849.jpg)

![d01f4f65d7124fa3a1ffb42fa8579c139fb2d388788f4f9c0a6398a696466ea5.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/d01f4f65d7124fa3a1ffb42fa8579c139fb2d388788f4f9c0a6398a696466ea5.jpg)

![fbe888bd3e476401231187d7fbe1a5c7270f2030f6cb8b641d731f28857123c2.jpg](../icml_results/4667_Symbal_%20Detecting%20Systematic%20Misalignments%20in%20Model-Generated%20Captions/tables/fbe888bd3e476401231187d7fbe1a5c7270f2030f6cb8b641d731f28857123c2.jpg)

## Faster Query-Key Learning Sharpens Attention in Self-Attention Models


### Images

![2f09523c6fac286704b4a7f27b89c8d4147a61fa2c29056b16d3801fc2488459.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/2f09523c6fac286704b4a7f27b89c8d4147a61fa2c29056b16d3801fc2488459.jpg)

![343309475097aea7df5f77cc17d5624c9ddfd9fe1358f2d78591817477683fb1.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/343309475097aea7df5f77cc17d5624c9ddfd9fe1358f2d78591817477683fb1.jpg)

![47d507fe97a3045c577cdc5fe519764780fe8191e43b643b28b6e9b0c78fd79b.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/47d507fe97a3045c577cdc5fe519764780fe8191e43b643b28b6e9b0c78fd79b.jpg)

![72d20092d3f8787bc3a99755829d9adee37032beca44fff1ffcdb26a3d6e81db.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/72d20092d3f8787bc3a99755829d9adee37032beca44fff1ffcdb26a3d6e81db.jpg)

![84e91fb7df2a25abb421055317449fdd632b9e846933067a0974312dbe07aef0.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/84e91fb7df2a25abb421055317449fdd632b9e846933067a0974312dbe07aef0.jpg)

![aaaa5e1bdfd7b55d5c470cae9b94bcbddf84527e550e7e3ce10f4cd01d92eb10.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/aaaa5e1bdfd7b55d5c470cae9b94bcbddf84527e550e7e3ce10f4cd01d92eb10.jpg)

![d236555e7b3cbf1d369947ce957590ef616c724e113dfb16200a7de7a9907134.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/images/d236555e7b3cbf1d369947ce957590ef616c724e113dfb16200a7de7a9907134.jpg)

### Tables

![02910be68c237761fbebf60940a78c4a0027b8e692fb9e092c2ad6e14d0447e5.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/02910be68c237761fbebf60940a78c4a0027b8e692fb9e092c2ad6e14d0447e5.jpg)

![683bf1df93c254fbbc6844d437825f921808e81908ce5db4505bfcb5d6fee781.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/683bf1df93c254fbbc6844d437825f921808e81908ce5db4505bfcb5d6fee781.jpg)

![79564e0414d8d4aec23bdbc7bf649de926b3e9bc9656486193926eb3e30bd948.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/79564e0414d8d4aec23bdbc7bf649de926b3e9bc9656486193926eb3e30bd948.jpg)

![d24407ce2195f9f9b02550b1b1b62d50f5c4dfd301a63a86f7197272c5fdc77c.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/d24407ce2195f9f9b02550b1b1b62d50f5c4dfd301a63a86f7197272c5fdc77c.jpg)

![d359260b264b2ff9d4415b494f0321d1c3b2505d31d7e5a22da5b69a125f6e17.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/d359260b264b2ff9d4415b494f0321d1c3b2505d31d7e5a22da5b69a125f6e17.jpg)

![e4ce4ab8eb1c3b843bbbd016e330d0db127e687a8c48fc2898ca5f98cdf43126.jpg](../icml_results/4668_Faster%20Query-Key%20Learning%20Sharpens%20Attention%20in%20Self-Attention%20Models/tables/e4ce4ab8eb1c3b843bbbd016e330d0db127e687a8c48fc2898ca5f98cdf43126.jpg)

<a id="paper-4622"></a>

## Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads


### Images

![03540c76acfb2f07ec5a8be178d72eb4cf7f6a212ff3f01947b57f50bea1a76b.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/03540c76acfb2f07ec5a8be178d72eb4cf7f6a212ff3f01947b57f50bea1a76b.jpg)

![20396e0c5adfce21bfdefff22f446108f68c8444666d4464c249b4c5f666e2c6.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/20396e0c5adfce21bfdefff22f446108f68c8444666d4464c249b4c5f666e2c6.jpg)

![35c331852cdf66839e0a2b7c620bb361da82eab37e2746885d74f6855fb09464.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/35c331852cdf66839e0a2b7c620bb361da82eab37e2746885d74f6855fb09464.jpg)

![35f3b25896ba6945b764c9639f87e43298d0e21ceb0e5cccdcf0f45d45787d0a.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/35f3b25896ba6945b764c9639f87e43298d0e21ceb0e5cccdcf0f45d45787d0a.jpg)

![3bd2ea62a8c7f9a413ae94c121bc7ccc1f68ba2d104dbf8e67ec8ff59849bdc9.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/3bd2ea62a8c7f9a413ae94c121bc7ccc1f68ba2d104dbf8e67ec8ff59849bdc9.jpg)

![476f614ecf227aafabc0c904506fe72ce91cfa204ab7094aaf1a38b1e049b96e.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/476f614ecf227aafabc0c904506fe72ce91cfa204ab7094aaf1a38b1e049b96e.jpg)

![583fa12bf81072fe80f724b9c1db0701b5b40771c28e146eb06aa0e74c0e61f8.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/583fa12bf81072fe80f724b9c1db0701b5b40771c28e146eb06aa0e74c0e61f8.jpg)

![6f2776fb415089fb4c8a4b3ed5531bec32b343d4e7af6135e5c9423ae3f95980.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/6f2776fb415089fb4c8a4b3ed5531bec32b343d4e7af6135e5c9423ae3f95980.jpg)

![8cabbe86e55fb53b3d0395c9c3b6bb7a5206a4442ffd9f395fcd6f5f72c8a279.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/8cabbe86e55fb53b3d0395c9c3b6bb7a5206a4442ffd9f395fcd6f5f72c8a279.jpg)

![9e3ac35d44d59fef55fec8a7d5f8ef9020cd9bd195396b84f6178f04b24cdbdb.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/9e3ac35d44d59fef55fec8a7d5f8ef9020cd9bd195396b84f6178f04b24cdbdb.jpg)

![bb451d49fbc8988c07f60f01a32e26d8c3acdbea897638a4d5510d97556c6c56.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/bb451d49fbc8988c07f60f01a32e26d8c3acdbea897638a4d5510d97556c6c56.jpg)

![c265faae57ef17057f47c36a9d792391f545b0cb0f5ba5dd21aaa75a0e2a9d2f.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/c265faae57ef17057f47c36a9d792391f545b0cb0f5ba5dd21aaa75a0e2a9d2f.jpg)

![c4ed5ef8f223965182c1b1aec44cf867d05d5d63f212272c35d0bcc88c317e6b.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/c4ed5ef8f223965182c1b1aec44cf867d05d5d63f212272c35d0bcc88c317e6b.jpg)

![ccd7bf5e10bc65b2916263cea60043e7433ee8be8b0cc0f0ef946bbee9153d71.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/images/ccd7bf5e10bc65b2916263cea60043e7433ee8be8b0cc0f0ef946bbee9153d71.jpg)

### Tables

![2122747fc01a8d9401f8155d7b1bf67ff4d3ab0eaf31168ecb28baf9a8966a58.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/2122747fc01a8d9401f8155d7b1bf67ff4d3ab0eaf31168ecb28baf9a8966a58.jpg)

![34aa5317394db0a7fc096c49a919132b1257e6aeebc2ad22f1c46da7418d2d15.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/34aa5317394db0a7fc096c49a919132b1257e6aeebc2ad22f1c46da7418d2d15.jpg)

![373fe9e382191d0cae0b93b621bfdf04d94fac4e96b1daeb4488410e434ea003.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/373fe9e382191d0cae0b93b621bfdf04d94fac4e96b1daeb4488410e434ea003.jpg)

![3b50b6cb8f049fd6adde6363d97bb58774b6b7a942f76a7d6050cfa5c584e5dc.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/3b50b6cb8f049fd6adde6363d97bb58774b6b7a942f76a7d6050cfa5c584e5dc.jpg)

![3f91c5760f0e7c2daa5ea6529addb93370b5ecc178f11f3fca38752c775913e7.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/3f91c5760f0e7c2daa5ea6529addb93370b5ecc178f11f3fca38752c775913e7.jpg)

![4449ad7553d195d975cb67dcf529fce3bf608872151b7e6f2fe3e5aada97edae.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/4449ad7553d195d975cb67dcf529fce3bf608872151b7e6f2fe3e5aada97edae.jpg)

![54de629e354b34e0fa16015777c037dc4fccad9f03b181f58bab0a28c2ebc040.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/54de629e354b34e0fa16015777c037dc4fccad9f03b181f58bab0a28c2ebc040.jpg)

![7b87c75dc76270f53ae3dfe4811f2b16d61d0034889f1303fc61567818631f82.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/7b87c75dc76270f53ae3dfe4811f2b16d61d0034889f1303fc61567818631f82.jpg)

![7bbf1b8f018e2ad96d29216520a052b2acad5266fb01986682f78ff6fa12c93a.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/7bbf1b8f018e2ad96d29216520a052b2acad5266fb01986682f78ff6fa12c93a.jpg)

![823ea9c067cb779a5edb8f63dcdc617b5556d210a14681af7e1770f237244aa7.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/823ea9c067cb779a5edb8f63dcdc617b5556d210a14681af7e1770f237244aa7.jpg)

![84d1a7e4fc6f9163efe39ed1d48140605df6acd44bc49f7893daa34c760017c3.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/84d1a7e4fc6f9163efe39ed1d48140605df6acd44bc49f7893daa34c760017c3.jpg)

![914ca662ffce100069a9a3c0f9f3bdbcf1b76924da87dd1a28e58de9ea82fc5c.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/914ca662ffce100069a9a3c0f9f3bdbcf1b76924da87dd1a28e58de9ea82fc5c.jpg)

![93dd779a3efcbab1eb7bbce95c8550ea816672c4063f4d2ffd9ec1e50cdf4941.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/93dd779a3efcbab1eb7bbce95c8550ea816672c4063f4d2ffd9ec1e50cdf4941.jpg)

![98522706a873090e0261c20ff57ac664d8c306811a6583814860862d332b8d44.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/98522706a873090e0261c20ff57ac664d8c306811a6583814860862d332b8d44.jpg)

![9c2bdf043bb80f32f317c731a7642826df637e9c525b3987b6e5ed0a0f528de9.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/9c2bdf043bb80f32f317c731a7642826df637e9c525b3987b6e5ed0a0f528de9.jpg)

![a5dccdb8bdc43b35c2e9306aa96b20ee2526004f6871dca589f89ec57171cc48.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/a5dccdb8bdc43b35c2e9306aa96b20ee2526004f6871dca589f89ec57171cc48.jpg)

![ac9d0c612758db316fac425b522d26ee3dcfc9e5c24907771c687fbdbb7c4ae3.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/ac9d0c612758db316fac425b522d26ee3dcfc9e5c24907771c687fbdbb7c4ae3.jpg)

![aca729466d4c13107730edb12849f159ec8449ad36aedbe618e7dffeec79aab1.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/aca729466d4c13107730edb12849f159ec8449ad36aedbe618e7dffeec79aab1.jpg)

![b4a1245a5b80d49d306f9b8e9583b27ad1ac7eff5033adef32f28131eb1ef863.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/b4a1245a5b80d49d306f9b8e9583b27ad1ac7eff5033adef32f28131eb1ef863.jpg)

![b638697e563abe2251de4dd6682878cbe500d4ea1f76d4bfe1dc429484c6af02.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/b638697e563abe2251de4dd6682878cbe500d4ea1f76d4bfe1dc429484c6af02.jpg)

![defcdc716e215288eb28b4fd7934ddf9dec55c6cd22460c8adc1c07da4b3ef81.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/defcdc716e215288eb28b4fd7934ddf9dec55c6cd22460c8adc1c07da4b3ef81.jpg)

![e9f9344d1018ef8c06663e453eedfe6cee76c36033e23b5bc944205e2289cbc0.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/e9f9344d1018ef8c06663e453eedfe6cee76c36033e23b5bc944205e2289cbc0.jpg)

![ea31a3e24b88930a9300c6db97277a5de141476584bcd60b768f7fe70b691cba.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/ea31a3e24b88930a9300c6db97277a5de141476584bcd60b768f7fe70b691cba.jpg)

![f0ad7c11d79ed1718542464d26c4de67f5cfcd1ccb0a80458d1192e07d56e221.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/f0ad7c11d79ed1718542464d26c4de67f5cfcd1ccb0a80458d1192e07d56e221.jpg)

![f0f023c679ace0af00f5f34e77b6ecbf74f128fb730f4c1d1b8c1dc79e4e4e7e.jpg](../icml_results/4670_Efficient%20Hallucination%20Detection%20for%20LLMs%20Using%20Uncertainty-Aware%20Attention%20Heads/tables/f0f023c679ace0af00f5f34e77b6ecbf74f128fb730f4c1d1b8c1dc79e4e4e7e.jpg)

<a id="paper-4623"></a>

## Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning


### Images

![0a55307d43b548248cc651bea6052cb20a02d9f8f97062ab34560bc308c51ecd.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/0a55307d43b548248cc651bea6052cb20a02d9f8f97062ab34560bc308c51ecd.jpg)

![3725fbcfa34b4ea0747d664364cf2cff2f7041318619a85ba3bb1fb6fa254886.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/3725fbcfa34b4ea0747d664364cf2cff2f7041318619a85ba3bb1fb6fa254886.jpg)

![3aa554d70418938bd8cb96cf760bb32dae0d7f2a97cd297580dd93e299213511.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/3aa554d70418938bd8cb96cf760bb32dae0d7f2a97cd297580dd93e299213511.jpg)

![50545fd2e68dd0792806554bd1be2475479b8e9c877cc44b102e726bff7040fc.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/50545fd2e68dd0792806554bd1be2475479b8e9c877cc44b102e726bff7040fc.jpg)

![6d75360ea6da134964fe06a42778e5a9c8f0d3f8e9b2aee86eb11d343e53e5d7.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/6d75360ea6da134964fe06a42778e5a9c8f0d3f8e9b2aee86eb11d343e53e5d7.jpg)

![71cb4c8a48c47a9fac63d31f7389f3dc9e87a96f79c772bfe760e13aa37488ac.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/71cb4c8a48c47a9fac63d31f7389f3dc9e87a96f79c772bfe760e13aa37488ac.jpg)

![75b03d5256acdbdb0a8d5176e0aec4061b1084d697ee13c50c502f713221e10e.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/75b03d5256acdbdb0a8d5176e0aec4061b1084d697ee13c50c502f713221e10e.jpg)

![8a743fbb310c21fa65cd202f3ed88ba138e61a49a1e992cf76e25150941bde5f.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/8a743fbb310c21fa65cd202f3ed88ba138e61a49a1e992cf76e25150941bde5f.jpg)

![9c5791bc93f2f3e41a385e111db5ed15425ebc011e060a0c37fb1afae6093210.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/9c5791bc93f2f3e41a385e111db5ed15425ebc011e060a0c37fb1afae6093210.jpg)

![9e82884ef0861cb12f77ea1fdc4145b6438c37c4c36be28a005bf5a869d07bd0.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/9e82884ef0861cb12f77ea1fdc4145b6438c37c4c36be28a005bf5a869d07bd0.jpg)

![a42c3eeee8a85aed59f893c67136550836482ada27df85cdb0f4cf7b7266b2a8.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/a42c3eeee8a85aed59f893c67136550836482ada27df85cdb0f4cf7b7266b2a8.jpg)

![b0e81db730173097faa4698c0a8ceda2bdc8b504904797a7d7286a8b23bb6f87.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/b0e81db730173097faa4698c0a8ceda2bdc8b504904797a7d7286a8b23bb6f87.jpg)

![d003b8eb083d8b329df8bf9a6b2871670537091a1bab9ff37982cbde91842036.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/d003b8eb083d8b329df8bf9a6b2871670537091a1bab9ff37982cbde91842036.jpg)

![d9f2611c69fb3303486cd4f55afb89f22b527bc1be7e0ad308246fbb259002ad.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/d9f2611c69fb3303486cd4f55afb89f22b527bc1be7e0ad308246fbb259002ad.jpg)

![e9e3c0995c5fd38b011e063b242687d22663e617e0049b9ee6d532c0810ff8da.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/e9e3c0995c5fd38b011e063b242687d22663e617e0049b9ee6d532c0810ff8da.jpg)

![ecdfc4ac9a9ec5bfc312c03e6f6a33efdbb9b09435af01c01c1e7e9ec84af6bc.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/ecdfc4ac9a9ec5bfc312c03e6f6a33efdbb9b09435af01c01c1e7e9ec84af6bc.jpg)

![f09fc294e257a232c77114e4b7ad2df70b46dba215063d190bcb90d0de96d8b3.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/images/f09fc294e257a232c77114e4b7ad2df70b46dba215063d190bcb90d0de96d8b3.jpg)

### Tables

![040c5beb3651e3415084973a56d0b44b3baf713f98b2521a5868ac4ead9b90ae.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/040c5beb3651e3415084973a56d0b44b3baf713f98b2521a5868ac4ead9b90ae.jpg)

![051ad64ca9f1435ee13f57de052b08bc742b977f38c4f304ad5f858b80e6c34a.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/051ad64ca9f1435ee13f57de052b08bc742b977f38c4f304ad5f858b80e6c34a.jpg)

![0df8523557b5262512ec1c6dcde1130bab5449baac241a1ce166204b69f2d685.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/0df8523557b5262512ec1c6dcde1130bab5449baac241a1ce166204b69f2d685.jpg)

![0ec2f489b2cf8fc448f8bbca27a605dcaf9f6b3fefab695b447258a6e12da660.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/0ec2f489b2cf8fc448f8bbca27a605dcaf9f6b3fefab695b447258a6e12da660.jpg)

![11231bbc456f26e9832bf00a3aa680bb4d16e72e565cc2d9e33a81aa7bedab61.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/11231bbc456f26e9832bf00a3aa680bb4d16e72e565cc2d9e33a81aa7bedab61.jpg)

![120b6d3db0baee91e4a34f8bb584198721d9143aa2c6ed00ad17e80289a48874.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/120b6d3db0baee91e4a34f8bb584198721d9143aa2c6ed00ad17e80289a48874.jpg)

![14dd82c7b3b93b2f3161a77c6a13eeac469e8edb372ff6d53adc9987080009f9.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/14dd82c7b3b93b2f3161a77c6a13eeac469e8edb372ff6d53adc9987080009f9.jpg)

![18db34839363d1a2d9435ee866d72cc3f050bf5e8ca0fb0db92a29612c864c7e.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/18db34839363d1a2d9435ee866d72cc3f050bf5e8ca0fb0db92a29612c864c7e.jpg)

![239f50c935f61110156d9bab88fa8b3d8025e0b0b6a1d117ae9e463d76dd96dc.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/239f50c935f61110156d9bab88fa8b3d8025e0b0b6a1d117ae9e463d76dd96dc.jpg)

![3d288d124625fd6ac20a79a49581267b0d912c9ca8b782eed4617e5210c6987a.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/3d288d124625fd6ac20a79a49581267b0d912c9ca8b782eed4617e5210c6987a.jpg)

![4a535855655901eb62ccdacb817f93ca6b67a8af0d9178a9f59b8dfe1bd91f31.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/4a535855655901eb62ccdacb817f93ca6b67a8af0d9178a9f59b8dfe1bd91f31.jpg)

![5769d92194bceb6cb377b60a71a3b5e657f64162f292cb9a88acd126611d25e5.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/5769d92194bceb6cb377b60a71a3b5e657f64162f292cb9a88acd126611d25e5.jpg)

![765302c1e0cdabdabb98940ba174eec1d961378af24e2db590c58a72f24095d4.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/765302c1e0cdabdabb98940ba174eec1d961378af24e2db590c58a72f24095d4.jpg)

![7dbdaea2f2cd7e11d659ced57d023fa0ffec08c953c233ccb9f88e909a653e2b.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/7dbdaea2f2cd7e11d659ced57d023fa0ffec08c953c233ccb9f88e909a653e2b.jpg)

![96200394885102cab06e4683faa0ac1bb1b03aec89b512c336b2fc2deaef95d9.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/96200394885102cab06e4683faa0ac1bb1b03aec89b512c336b2fc2deaef95d9.jpg)

![aaa7ddb9ea349e47b7ce98b6dfa677ffdadad9aa329c3f4b45300fe58241e7e7.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/aaa7ddb9ea349e47b7ce98b6dfa677ffdadad9aa329c3f4b45300fe58241e7e7.jpg)

![ae11105ab518299662e63fcc679886169512e09dc0b6ca9eb1e1428fc61c74fd.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/ae11105ab518299662e63fcc679886169512e09dc0b6ca9eb1e1428fc61c74fd.jpg)

![b7e5d10a15553ccc26d4b5925d436235c068da607a888f1b79d0cd192dbc0221.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/b7e5d10a15553ccc26d4b5925d436235c068da607a888f1b79d0cd192dbc0221.jpg)

![c9335c435e222ca4d49b9cb5aed550cfb38b0525c613793a11d3afd03d60eff1.jpg](../icml_results/4671_Formal%20Concept%20Lattices%20are%20Good%20Semantic%20Scaffolds%20for%20Concept-Based%20Learning/tables/c9335c435e222ca4d49b9cb5aed550cfb38b0525c613793a11d3afd03d60eff1.jpg)

<a id="paper-4624"></a>

## MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems


### Images

![08f28458e9b5c257411da791a4f3f33818dead64120068900a81d2535e5e12e2.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/08f28458e9b5c257411da791a4f3f33818dead64120068900a81d2535e5e12e2.jpg)

![0dcce97627178a248c6ed831a7975e156618b4646ac7434cbc8382c01de58d7e.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/0dcce97627178a248c6ed831a7975e156618b4646ac7434cbc8382c01de58d7e.jpg)

![17e05f7265a4dac6e6dc0035e53c41eba58ed3d4e45a941e3b570106312ee41a.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/17e05f7265a4dac6e6dc0035e53c41eba58ed3d4e45a941e3b570106312ee41a.jpg)

![1f94cce1346ac63bede4f9f2e378834c128e5390a67c1b8c71a4bfef032ce4e5.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/1f94cce1346ac63bede4f9f2e378834c128e5390a67c1b8c71a4bfef032ce4e5.jpg)

![2f35d63d1dfee8492900b011fb73a6f0d912615e1ffd7ad7db2ad59f7d2d5a36.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/2f35d63d1dfee8492900b011fb73a6f0d912615e1ffd7ad7db2ad59f7d2d5a36.jpg)

![42103d8dd126909775fb14b2b61a2096e542fb55311279dc23005bba04ecf6fc.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/42103d8dd126909775fb14b2b61a2096e542fb55311279dc23005bba04ecf6fc.jpg)

![5103cfe46189a3bd7e71646138c78f4b4045f7cabf0fe3e2bab8232689057624.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/5103cfe46189a3bd7e71646138c78f4b4045f7cabf0fe3e2bab8232689057624.jpg)

![579b469042cb671250398b4905d96af64c51ef4710913234839a806d555f31a6.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/579b469042cb671250398b4905d96af64c51ef4710913234839a806d555f31a6.jpg)

![98110ab6c2123c07a0b911e2d2b2d3fb800b5a619a370aedcc8677a9cf7a84ff.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/98110ab6c2123c07a0b911e2d2b2d3fb800b5a619a370aedcc8677a9cf7a84ff.jpg)

![a55bf37803fc59dcc8a3353eb9023f0bc2aac6bc3e59a4f0c8f33ee5800883aa.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/a55bf37803fc59dcc8a3353eb9023f0bc2aac6bc3e59a4f0c8f33ee5800883aa.jpg)

![aecc03688bf61fb5ed1f2e3579b9f4836e1c346ed543f18f785f5418f50f51b8.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/aecc03688bf61fb5ed1f2e3579b9f4836e1c346ed543f18f785f5418f50f51b8.jpg)

![b40be0329e0c152c3082b2bc83555d505df501d6aec0566d0d1b23a643f4328b.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/b40be0329e0c152c3082b2bc83555d505df501d6aec0566d0d1b23a643f4328b.jpg)

![e7dbb916fc4e1b82aed92db10b5943b009606f5064d7a3d16a0afafcfb794c96.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/e7dbb916fc4e1b82aed92db10b5943b009606f5064d7a3d16a0afafcfb794c96.jpg)

![ecb344b845b5b3c497024c07ec90d1d7ed95f0befcd045f1701bac96009c4456.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/ecb344b845b5b3c497024c07ec90d1d7ed95f0befcd045f1701bac96009c4456.jpg)

![fc4458414e13dcaac0c3c06528688c3f4ba94a3ba77db1b256e6dcb6d35c668f.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/images/fc4458414e13dcaac0c3c06528688c3f4ba94a3ba77db1b256e6dcb6d35c668f.jpg)

### Tables

![0b688a2fc572e14c3056ccce08e2b697303cb9f78efca21b4207a5336cbc4d40.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/0b688a2fc572e14c3056ccce08e2b697303cb9f78efca21b4207a5336cbc4d40.jpg)

![29e519c2b7bc0a6749c7f2423a8761c284944b42cb117660456242a6687b2d83.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/29e519c2b7bc0a6749c7f2423a8761c284944b42cb117660456242a6687b2d83.jpg)

![4078c06e229e698580898ec4f2f9e42e3394d42b2fbdf0ef0688c4fa9d01837d.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/4078c06e229e698580898ec4f2f9e42e3394d42b2fbdf0ef0688c4fa9d01837d.jpg)

![8d989fb0efb34e584c893d23084ab1f9cd15267d72d924d25307fe953b847467.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/8d989fb0efb34e584c893d23084ab1f9cd15267d72d924d25307fe953b847467.jpg)

![8e14567d2aa1f14800157839578d95d0b9a7b38dd746653f2badf12d69b0f9da.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/8e14567d2aa1f14800157839578d95d0b9a7b38dd746653f2badf12d69b0f9da.jpg)

![9ced808ac97cc55ea4a237c369c2959eb3cf2078ca57e038e4354489d5ed453e.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/9ced808ac97cc55ea4a237c369c2959eb3cf2078ca57e038e4354489d5ed453e.jpg)

![c9d218a6634432a9242536ae531d84fd810a1c08664e2db49947caaa9714754a.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/c9d218a6634432a9242536ae531d84fd810a1c08664e2db49947caaa9714754a.jpg)

![cfb9dcca4638f8327000d1ba1be17268874f836fa5b16f10ab38e49b350e4f17.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/cfb9dcca4638f8327000d1ba1be17268874f836fa5b16f10ab38e49b350e4f17.jpg)

![e0b290a274d51bd71a63de3e12475ef186cffd4d33dc63af6f2dd4a86ace03c2.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/e0b290a274d51bd71a63de3e12475ef186cffd4d33dc63af6f2dd4a86ace03c2.jpg)

![ef5d824dda0cd98e79f9ebd31c2c99d478fea9816cae30f298473df37eae94d5.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/ef5d824dda0cd98e79f9ebd31c2c99d478fea9816cae30f298473df37eae94d5.jpg)

![f05b32444cd385e05e89a5532335017c1cb0714531ddd25585cf8ac003084d1e.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/f05b32444cd385e05e89a5532335017c1cb0714531ddd25585cf8ac003084d1e.jpg)

![f2d90952e19361b5cb8998580d498ddb70a48c600137220d87d7526d9b7158c5.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/f2d90952e19361b5cb8998580d498ddb70a48c600137220d87d7526d9b7158c5.jpg)

![fc0490acbd82d80996a34d9aab2fb17b8eb5728db9707f4f9b8cdb4165efdc3c.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/fc0490acbd82d80996a34d9aab2fb17b8eb5728db9707f4f9b8cdb4165efdc3c.jpg)

![ffbd1777c57f0ce6d75add386f33fb7af37697aa8ed635f9c35460720437f3c2.jpg](../icml_results/4672_MAS-ProVe_%20Understanding%20the%20Process%20Verification%20of%20Multi-Agent%20Systems/tables/ffbd1777c57f0ce6d75add386f33fb7af37697aa8ed635f9c35460720437f3c2.jpg)

<a id="paper-4625"></a>

## Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Model


### Images

![017700567de002f1c3879aed27537f2ed1d9b30fb6a59842447edb43e71ebfd0.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/017700567de002f1c3879aed27537f2ed1d9b30fb6a59842447edb43e71ebfd0.jpg)

![16eaa4d56ff494d4371cec65dbc868280ecc9921ea2064e13c6d35445239be37.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/16eaa4d56ff494d4371cec65dbc868280ecc9921ea2064e13c6d35445239be37.jpg)

![2347cadc4cdcf06e225796b2f602d9547f12f1c40e1ad2c49c5c8a2988aa4727.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/2347cadc4cdcf06e225796b2f602d9547f12f1c40e1ad2c49c5c8a2988aa4727.jpg)

![406cfee51a70e5f11725a30a08eb751d2ac0222bd44a6cac68020e64ac9ac4d0.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/406cfee51a70e5f11725a30a08eb751d2ac0222bd44a6cac68020e64ac9ac4d0.jpg)

![4bdf6c3f4842a3a904e57d7918d000d5fb1f4a7f2672ad024702749ec9a40044.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/4bdf6c3f4842a3a904e57d7918d000d5fb1f4a7f2672ad024702749ec9a40044.jpg)

![555e86d1b4f9f32584a2da592e6134e4862a39139c74eecc239847d6db69fe2a.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/555e86d1b4f9f32584a2da592e6134e4862a39139c74eecc239847d6db69fe2a.jpg)

![5cac94ef4431405b065fa97e6e7d4820388858d60af1e9ecc683c24b6e4f7420.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/5cac94ef4431405b065fa97e6e7d4820388858d60af1e9ecc683c24b6e4f7420.jpg)

![6e020886c46bcac5068223f310e84fe9ceb997d041f642cf80fd6f1d37302fca.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/6e020886c46bcac5068223f310e84fe9ceb997d041f642cf80fd6f1d37302fca.jpg)

![7fa8e5b1ab50eea20b54b95edd41f13638a1c9f275eb900b1853a36cf943213a.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/7fa8e5b1ab50eea20b54b95edd41f13638a1c9f275eb900b1853a36cf943213a.jpg)

![7fd101bb33bf2514beec55bd6fafdc9a3fe799a6ab6a8e2e70f0968dbb24749d.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/7fd101bb33bf2514beec55bd6fafdc9a3fe799a6ab6a8e2e70f0968dbb24749d.jpg)

![980f61695badcb6967b1fdbded1332515bd1ada32fb05521e39ead3026178d37.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/980f61695badcb6967b1fdbded1332515bd1ada32fb05521e39ead3026178d37.jpg)

![a7d313f7b3730ef828fa07ce1dee9bf7ef26f9cd84cb2bb9024e661891f97d6b.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/a7d313f7b3730ef828fa07ce1dee9bf7ef26f9cd84cb2bb9024e661891f97d6b.jpg)

![e57aaec825b93b6476a6fd6c0b2564bc1b35a2b34a92e1eb2e30f4d04fb7da17.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/e57aaec825b93b6476a6fd6c0b2564bc1b35a2b34a92e1eb2e30f4d04fb7da17.jpg)

![ec0dc7a8ec83dd2aabc5a5bf7a4b3f1d77965a479c8da4ce142688764a89824b.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/ec0dc7a8ec83dd2aabc5a5bf7a4b3f1d77965a479c8da4ce142688764a89824b.jpg)

![f042d6114ff901be94132cb5609f6f3b3dfd39d81f61dbd7952a5de0ca12f0ce.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/f042d6114ff901be94132cb5609f6f3b3dfd39d81f61dbd7952a5de0ca12f0ce.jpg)

![f32fa1434209497cc88560bc9bd43e8f071930c69450c7a5902ab1d9113f61fb.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/images/f32fa1434209497cc88560bc9bd43e8f071930c69450c7a5902ab1d9113f61fb.jpg)

### Tables

![03615fff5ade8fcc00ef7e67a7e1cd075e9320740e10ffb672f52d932e8ea9cd.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/03615fff5ade8fcc00ef7e67a7e1cd075e9320740e10ffb672f52d932e8ea9cd.jpg)

![090e200af95065cb4a5f4c04f1ab106b97576a1cf37f32082d29ec855386c92a.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/090e200af95065cb4a5f4c04f1ab106b97576a1cf37f32082d29ec855386c92a.jpg)

![1039bbfab5262f78ee4f240f9859d99f0f7ee4f8c9c8645cbd99790c84bef2e8.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/1039bbfab5262f78ee4f240f9859d99f0f7ee4f8c9c8645cbd99790c84bef2e8.jpg)

![116c5384c08b9b8936466b6be215fcb7c571f5fe6ad8e341e3832998abe93156.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/116c5384c08b9b8936466b6be215fcb7c571f5fe6ad8e341e3832998abe93156.jpg)

![1e223ef697271757bf702f65e71714e374225b26bac82eacff6ca801cd4cc474.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/1e223ef697271757bf702f65e71714e374225b26bac82eacff6ca801cd4cc474.jpg)

![23473b0534a9294acd48aaaee80ab59250427a50c42add8223ae6089e3387095.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/23473b0534a9294acd48aaaee80ab59250427a50c42add8223ae6089e3387095.jpg)

![27b8fea7bcaea2f6da19b581659316067fcb17668f2e010c8a1e98aad1dadeb1.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/27b8fea7bcaea2f6da19b581659316067fcb17668f2e010c8a1e98aad1dadeb1.jpg)

![39c2bc135d3b651b8e29b1f9da8a95f162047190e7dc8eeede265220fa55d278.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/39c2bc135d3b651b8e29b1f9da8a95f162047190e7dc8eeede265220fa55d278.jpg)

![3d8999376e4ffc15bdca2f035347d5ba562fe9d3786f81bd230545f61ccd76ab.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/3d8999376e4ffc15bdca2f035347d5ba562fe9d3786f81bd230545f61ccd76ab.jpg)

![405b773a1ba4017f1b3a5fe317ce6d066fdc036c72dbb6fa12ef876ec09d9173.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/405b773a1ba4017f1b3a5fe317ce6d066fdc036c72dbb6fa12ef876ec09d9173.jpg)

![6e82a4eb6824116bdbf73e14480c4220bf563c1c343e9a9d8649e6560162c1d5.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/6e82a4eb6824116bdbf73e14480c4220bf563c1c343e9a9d8649e6560162c1d5.jpg)

![701be7a3f678a4f7d6324fb2826e3c418295544bda3141f73be166825479a307.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/701be7a3f678a4f7d6324fb2826e3c418295544bda3141f73be166825479a307.jpg)

![78f26496b582c7d5470c507a16eafd0c400e0a6a65f89d4227108aa6fa9a3323.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/78f26496b582c7d5470c507a16eafd0c400e0a6a65f89d4227108aa6fa9a3323.jpg)

![86ade2912d260b7cea86a740392e2b74a1c2a06695c94d60fdbb30e8876a73bf.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/86ade2912d260b7cea86a740392e2b74a1c2a06695c94d60fdbb30e8876a73bf.jpg)

![8967e9cf3f5733b7d7a2c47bc3b4f9c7f4c4cc0554c250deb9987b510e931654.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/8967e9cf3f5733b7d7a2c47bc3b4f9c7f4c4cc0554c250deb9987b510e931654.jpg)

![a845c6d50fc0af63eebea59ece7941c07fe46d592644e8c37d91eef660ad57ab.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/a845c6d50fc0af63eebea59ece7941c07fe46d592644e8c37d91eef660ad57ab.jpg)

![ca6e9c92f7af5191b40ae9240019d93b14a3a6d56d470d47cc940c075fd4e058.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/ca6e9c92f7af5191b40ae9240019d93b14a3a6d56d470d47cc940c075fd4e058.jpg)

![ce12f08f26ec4963b60822cba7c8305f4cf4ef288b7c809931d460a4c310dbf5.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/ce12f08f26ec4963b60822cba7c8305f4cf4ef288b7c809931d460a4c310dbf5.jpg)

![ec8ffe77951f175b697b15e0797be4d284da5f01ad11f3e89c4d5c922a2a8514.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/ec8ffe77951f175b697b15e0797be4d284da5f01ad11f3e89c4d5c922a2a8514.jpg)

![f1c7ba0265ba8ed13ad754faa6d157e71505032600ef01a12a47e13b8e2dd5c0.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/f1c7ba0265ba8ed13ad754faa6d157e71505032600ef01a12a47e13b8e2dd5c0.jpg)

![f2cb6e5bf1a2fc997197a4714d37a7f0816eef2b7b52c77d0326bda805910172.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/f2cb6e5bf1a2fc997197a4714d37a7f0816eef2b7b52c77d0326bda805910172.jpg)

![f581d06ea2c9fb562b079f3dc3fcebd242271d3c76052f9c6faa5aa4d1879337.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/f581d06ea2c9fb562b079f3dc3fcebd242271d3c76052f9c6faa5aa4d1879337.jpg)

![f9002038c99d39f2553e5aba2bd373838851a085ee70de0c8ac47f728e546c89.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/f9002038c99d39f2553e5aba2bd373838851a085ee70de0c8ac47f728e546c89.jpg)

![ff1a80b2b1db6206a6970c2e1c0df41c2fbba221d9050403fdf12ad6da2db70f.jpg](../icml_results/4673_Mitigating%20Surgical%20Data%20Imbalance%20with%20Dual-Prediction%20Video%20Diffusion%20Model/tables/ff1a80b2b1db6206a6970c2e1c0df41c2fbba221d9050403fdf12ad6da2db70f.jpg)

<a id="paper-4626"></a>

## Operationalising the Superficial Alignment Hypothesis via Task Complexity


### Images

![10fe2b8f19f99b6e52567bd62ffd20bc9c2057ed4921e377de2d9de983c8abb7.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/10fe2b8f19f99b6e52567bd62ffd20bc9c2057ed4921e377de2d9de983c8abb7.jpg)

![272065204d52e32a601045827e33e310ca2dc9db165659355489d9dfa9138a75.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/272065204d52e32a601045827e33e310ca2dc9db165659355489d9dfa9138a75.jpg)

![2bbd6ccbd4ca5afef9e6b4b8239d3d1c8acba6eb8959ec9259f69c498d0bdf9c.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/2bbd6ccbd4ca5afef9e6b4b8239d3d1c8acba6eb8959ec9259f69c498d0bdf9c.jpg)

![481577b4dbe96211889f820432c6b3e36269702c11019612033d1effdd882fd3.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/481577b4dbe96211889f820432c6b3e36269702c11019612033d1effdd882fd3.jpg)

![64be91b20bc6e3e48f15ac12eb3f5b79ab83e414c527a92644a5db0e7d4549d0.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/64be91b20bc6e3e48f15ac12eb3f5b79ab83e414c527a92644a5db0e7d4549d0.jpg)

![8fb5281ed24fd15f3fbbd210b2fffcb5d5cef1849875c5347c49352c809211d2.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/8fb5281ed24fd15f3fbbd210b2fffcb5d5cef1849875c5347c49352c809211d2.jpg)

![9586acb4ec4da58385e2aad7235df3b11d77f56a102d10bd268ec685110102a9.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/9586acb4ec4da58385e2aad7235df3b11d77f56a102d10bd268ec685110102a9.jpg)

![d74029c6da1110601d53a3d1e2951dd6d7f1bf9c51b4aa018543af6b7907e712.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/d74029c6da1110601d53a3d1e2951dd6d7f1bf9c51b4aa018543af6b7907e712.jpg)

![d7b1a5854d4025d641601b63aa9348dd511ced52ed7fb8be1571fd9facb6982b.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/d7b1a5854d4025d641601b63aa9348dd511ced52ed7fb8be1571fd9facb6982b.jpg)

![dd2d352f6b5cee0393c7cf31fff87895955e9e4db67d53ae9c543cc3a748a049.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/dd2d352f6b5cee0393c7cf31fff87895955e9e4db67d53ae9c543cc3a748a049.jpg)

![eea19cd967c81d22ea8f43feaff794beaccc3133c5296a15eeae6908397b225d.jpg](../icml_results/4674_Operationalising%20the%20Superficial%20Alignment%20Hypothesis%20via%20Task%20Complexity/images/eea19cd967c81d22ea8f43feaff794beaccc3133c5296a15eeae6908397b225d.jpg)

<a id="paper-4627"></a>

## Equalized Generative Treatment: Matching f-divergences for Fairness in Generative Models


### Images

![006ec82c5874c779feecbda3a785bba8ec3a625204531b048d3dc37e3e760f58.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/006ec82c5874c779feecbda3a785bba8ec3a625204531b048d3dc37e3e760f58.jpg)

![02012c5a08b7a5e8e3700329b4eeb0b3189427a3f7960c3d50b24ea8e0e96445.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/02012c5a08b7a5e8e3700329b4eeb0b3189427a3f7960c3d50b24ea8e0e96445.jpg)

![0f1ace1c2bbfeddedb98e2642870b6a556f86d9b040d43c9269ee52cb84f996f.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/0f1ace1c2bbfeddedb98e2642870b6a556f86d9b040d43c9269ee52cb84f996f.jpg)

![1cd43029f9a161d65957cc9e3019cc34699d5f6d5e6f3e05e28acfc8c3f3128a.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/1cd43029f9a161d65957cc9e3019cc34699d5f6d5e6f3e05e28acfc8c3f3128a.jpg)

![26ee13b617bfba015b5a0f15c72676b00aa5617fb0287bb846505a6a5ba7ed67.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/26ee13b617bfba015b5a0f15c72676b00aa5617fb0287bb846505a6a5ba7ed67.jpg)

![4da499777fc7b9b6b11e5746105ca18e812e671182cebdf95f2ecb400a3a1452.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/4da499777fc7b9b6b11e5746105ca18e812e671182cebdf95f2ecb400a3a1452.jpg)

![81812a2efaa8a9e0df3c4b34a1781ca6fdaf0156d797c6feb976a760da387f6a.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/81812a2efaa8a9e0df3c4b34a1781ca6fdaf0156d797c6feb976a760da387f6a.jpg)

![ce6380d2a2650beaad37fa48092d5b4467a5260e829bf0e86e85af1b301743a9.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/images/ce6380d2a2650beaad37fa48092d5b4467a5260e829bf0e86e85af1b301743a9.jpg)

### Tables

![06d4296bc8376f08f6ec7ee8c5fd21b93a5cc71ee34d738d1f9899c9ec4df484.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/06d4296bc8376f08f6ec7ee8c5fd21b93a5cc71ee34d738d1f9899c9ec4df484.jpg)

![24a207a21fc286a33a0457b856fead81d6f72c298777ba49e87ea55ec58deab8.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/24a207a21fc286a33a0457b856fead81d6f72c298777ba49e87ea55ec58deab8.jpg)

![35d7bc5973c78211cc8c8a99b6c4e36e80e356f6d794e6be82b4014f76059f10.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/35d7bc5973c78211cc8c8a99b6c4e36e80e356f6d794e6be82b4014f76059f10.jpg)

![3b3fbda9627f6116d7de6750d1dc07afefec5f421bcebaf16c9d497694ed5e65.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/3b3fbda9627f6116d7de6750d1dc07afefec5f421bcebaf16c9d497694ed5e65.jpg)

![4c52ec5c7ab862a3eb226a4863b4777bba1a2a579a5895535ece54c5a0ee1555.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/4c52ec5c7ab862a3eb226a4863b4777bba1a2a579a5895535ece54c5a0ee1555.jpg)

![4f883c83020e5f4629d02b2c0795bfb19058a4db4687c225d0fb8166ca1d4330.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/4f883c83020e5f4629d02b2c0795bfb19058a4db4687c225d0fb8166ca1d4330.jpg)

![5b2605493cd9877c1a75741e410393aba7a4b83ce5e4abcdbc3a6e1c9751871e.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/5b2605493cd9877c1a75741e410393aba7a4b83ce5e4abcdbc3a6e1c9751871e.jpg)

![6ef9f3582b49427c9a908c190ea38604344575b0cfb8a5c55ee2ea84390e59b9.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/6ef9f3582b49427c9a908c190ea38604344575b0cfb8a5c55ee2ea84390e59b9.jpg)

![9485adb1e3174fcf19ea664366e92e17641a15f7d7e37427b5e14485c89f4806.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/9485adb1e3174fcf19ea664366e92e17641a15f7d7e37427b5e14485c89f4806.jpg)

![a6268e38ad5ef13ad1e123bc31cffe747e8e5b90709bae63b4a3dc5d5e528616.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/a6268e38ad5ef13ad1e123bc31cffe747e8e5b90709bae63b4a3dc5d5e528616.jpg)

![b4456d606123800ce5359193f16fc0d230cf9fcf8632635dcb96cc4e32c114ba.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/b4456d606123800ce5359193f16fc0d230cf9fcf8632635dcb96cc4e32c114ba.jpg)

![bd3b889f63727889721381485e8a5f9cc56270a9fb143d2c02a21ade046a439a.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/bd3b889f63727889721381485e8a5f9cc56270a9fb143d2c02a21ade046a439a.jpg)

![be0df5c1dba27e3bb99bab2473ff018023f7810198f25254179f3f46d2f3f04c.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/be0df5c1dba27e3bb99bab2473ff018023f7810198f25254179f3f46d2f3f04c.jpg)

![c866edf23daf8638129c8f35c500ec2bc1775fcdaa586647136027600fd6f465.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/c866edf23daf8638129c8f35c500ec2bc1775fcdaa586647136027600fd6f465.jpg)

![ec475bcf39ba2984a1b8eaa57bd2110e058588ed857d181bdb281ab5d2282a91.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/ec475bcf39ba2984a1b8eaa57bd2110e058588ed857d181bdb281ab5d2282a91.jpg)

![f84b892a0870559273b3623468b6f502a6d04e5ace16ddd7a280ad0e76e0ed51.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/f84b892a0870559273b3623468b6f502a6d04e5ace16ddd7a280ad0e76e0ed51.jpg)

![fa2d21ff9f2d16a8b49c88b7592082bf30dc942484b199686ca3647620c20e4e.jpg](../icml_results/4675_Equalized%20Generative%20Treatment_%20Matching%20f-divergences%20for%20Fairness%20in%20Generative%20Models/tables/fa2d21ff9f2d16a8b49c88b7592082bf30dc942484b199686ca3647620c20e4e.jpg)

<a id="paper-4628"></a>

## Keep Everyone Happy: Online Fair Division of Numerous Items with Few Copies


### Images

![35adcf35ef349997a57da01c1207eaa3705eaabf8022f43d5a41ce20beb26ea2.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/35adcf35ef349997a57da01c1207eaa3705eaabf8022f43d5a41ce20beb26ea2.jpg)

![47a26d95005e6caf94f0de4ade6ff71877d55805944eb6e69824f7b5d4012820.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/47a26d95005e6caf94f0de4ade6ff71877d55805944eb6e69824f7b5d4012820.jpg)

![500d0f39ea2a99c6c897144a024985fdf1eb0b808c5fb4d82cf53f22e6782f09.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/500d0f39ea2a99c6c897144a024985fdf1eb0b808c5fb4d82cf53f22e6782f09.jpg)

![996cca1c191033853d3fa3b4e37c669bb78a6975921387cd67a001f18f9c4b97.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/996cca1c191033853d3fa3b4e37c669bb78a6975921387cd67a001f18f9c4b97.jpg)

![cbef5e5d482aba708de9b31cf0a73f605e9cf008c52a8117c51cf9e9d173fdcc.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/cbef5e5d482aba708de9b31cf0a73f605e9cf008c52a8117c51cf9e9d173fdcc.jpg)

![d304e5a160a5295eb11952caebca4d4f18a801a5d83c26bb42542b4d7861e06e.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/images/d304e5a160a5295eb11952caebca4d4f18a801a5d83c26bb42542b4d7861e06e.jpg)

### Tables

![91aa1c97780708335eb7c72ebae05f24f948adc169e9245ba505e6a97527caa5.jpg](../icml_results/4676_Keep%20Everyone%20Happy_%20Online%20Fair%20Division%20of%20Numerous%20Items%20with%20Few%20Copies/tables/91aa1c97780708335eb7c72ebae05f24f948adc169e9245ba505e6a97527caa5.jpg)

<a id="paper-4629"></a>

## GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond


### Images

![0049c9b965aedc24ffea2da098907817c5c453f863f4bb1466b6777a5b71ca68.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/0049c9b965aedc24ffea2da098907817c5c453f863f4bb1466b6777a5b71ca68.jpg)

![04ec9efecb82ca9c13508d0a0ab7700af78fff13c828a921659187b3974645f8.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/04ec9efecb82ca9c13508d0a0ab7700af78fff13c828a921659187b3974645f8.jpg)

![116de22b4c93b55b4ac0a0e6c81537d46255936f43b7be971ca664a91f42274b.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/116de22b4c93b55b4ac0a0e6c81537d46255936f43b7be971ca664a91f42274b.jpg)

![131a14c91671c18fb0551a4bffe1342ed8b400eeeea2970223682815736072a0.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/131a14c91671c18fb0551a4bffe1342ed8b400eeeea2970223682815736072a0.jpg)

![42af1e6aa2e64591b47819dad1834de98dd5b3e27a51aee99aa399edd4f2ed3f.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/42af1e6aa2e64591b47819dad1834de98dd5b3e27a51aee99aa399edd4f2ed3f.jpg)

![478ad1b17ba5feb286ba979650e98c01ade5e7efb453efdec461b7435fd87ba1.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/478ad1b17ba5feb286ba979650e98c01ade5e7efb453efdec461b7435fd87ba1.jpg)

![58f41a14566b0e2a47509b9cecd9770b3d6d8ced42bf8cc22205214a95cb3880.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/58f41a14566b0e2a47509b9cecd9770b3d6d8ced42bf8cc22205214a95cb3880.jpg)

![b18ce5519029904ff11c948b99acaebc77c4d006d16b0696fe6a1c078ba25d57.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/images/b18ce5519029904ff11c948b99acaebc77c4d006d16b0696fe6a1c078ba25d57.jpg)

### Tables

![09926381987b76c0aa1ee92fbcef90a2e8f953bc3b6b6c4f2c336bf27333c9ae.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/09926381987b76c0aa1ee92fbcef90a2e8f953bc3b6b6c4f2c336bf27333c9ae.jpg)

![16c620a158fd12ff3e139aee11b10dc6f468917a626fe99c7c13872c4028000d.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/16c620a158fd12ff3e139aee11b10dc6f468917a626fe99c7c13872c4028000d.jpg)

![16d167300cc92d3874a1e9ad2f29bbe9d5a8a78d9cbe5209711c629f7afa30df.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/16d167300cc92d3874a1e9ad2f29bbe9d5a8a78d9cbe5209711c629f7afa30df.jpg)

![179b5a164a9cc9b580cd5ba634d9dfa17bd6237c6d559940ec9183d033df32bb.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/179b5a164a9cc9b580cd5ba634d9dfa17bd6237c6d559940ec9183d033df32bb.jpg)

![1be24bc6e0da6499baa088093b7e70cd09b50d155a1f6252ba63fa58ecc7970f.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/1be24bc6e0da6499baa088093b7e70cd09b50d155a1f6252ba63fa58ecc7970f.jpg)

![1d6f0455c41aa2308798af7b0484c22663c4a33277c344a22848a7d2e772033a.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/1d6f0455c41aa2308798af7b0484c22663c4a33277c344a22848a7d2e772033a.jpg)

![20c6ae00eeb02f75b06f054df1e6251037097c563e57fedca490142ac5f2845e.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/20c6ae00eeb02f75b06f054df1e6251037097c563e57fedca490142ac5f2845e.jpg)

![2c96f5dbeb07b4a3cd02bf7b367c4282e49ebd0d3fd30daf4adf63e7a0919b55.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/2c96f5dbeb07b4a3cd02bf7b367c4282e49ebd0d3fd30daf4adf63e7a0919b55.jpg)

![33f56bc3ef09b7b1c5259aa3a842597eff6352a7893abf2d8909b92d9d05337e.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/33f56bc3ef09b7b1c5259aa3a842597eff6352a7893abf2d8909b92d9d05337e.jpg)

![35a61dc3e4dc344fadd2ad16193aa0ae5508cb61719de4064beca1c473830a8d.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/35a61dc3e4dc344fadd2ad16193aa0ae5508cb61719de4064beca1c473830a8d.jpg)

![380e8b5fda156a7e09d5f3712d5d9c51c55632e0bd823c2dc740a863f396623c.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/380e8b5fda156a7e09d5f3712d5d9c51c55632e0bd823c2dc740a863f396623c.jpg)

![3bfc4b694bc7b82840a88c7dcf1a19a41871adbb59a2607fc155773c99c11459.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/3bfc4b694bc7b82840a88c7dcf1a19a41871adbb59a2607fc155773c99c11459.jpg)

![44bd6dd9a984140279a4b7296451dd8e2478402119d1b98ec310305e6580b602.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/44bd6dd9a984140279a4b7296451dd8e2478402119d1b98ec310305e6580b602.jpg)

![45967e06390fb31fa2668819f3ea90439f7018e3f8c93c4dcf0e5441bc6dc633.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/45967e06390fb31fa2668819f3ea90439f7018e3f8c93c4dcf0e5441bc6dc633.jpg)

![68cafa9c259ce5069eb921df2b6dfa4ee4adac21871c0b68e4e8a94484de99de.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/68cafa9c259ce5069eb921df2b6dfa4ee4adac21871c0b68e4e8a94484de99de.jpg)

![6f70a26737cc80eddb42e51bfcb49a1cacb4ad7d11d54a2e660c52a642cd3bab.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/6f70a26737cc80eddb42e51bfcb49a1cacb4ad7d11d54a2e660c52a642cd3bab.jpg)

![7b24e82cdeed1404b5d09e177d8a0080247fbe8c5b25d9433418c8a5438eea90.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/7b24e82cdeed1404b5d09e177d8a0080247fbe8c5b25d9433418c8a5438eea90.jpg)

![8f944784aa65c2ac347d7f2f299cce909c43548cd1f8e973fd4c98527cbb92ff.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/8f944784aa65c2ac347d7f2f299cce909c43548cd1f8e973fd4c98527cbb92ff.jpg)

![9544929a6df18f10ff2476388308b1c74bae61ae608c2a7f0127cf112856870e.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/9544929a6df18f10ff2476388308b1c74bae61ae608c2a7f0127cf112856870e.jpg)

![9703dd7a92cd6b602729e6539dfba64f69b2c781b541e073bc9e7f2bbab3f586.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/9703dd7a92cd6b602729e6539dfba64f69b2c781b541e073bc9e7f2bbab3f586.jpg)

![9c1f0c840a9c49bcfd1954acb131f4e911c2a68f72c47f0008bde0912081b1a7.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/9c1f0c840a9c49bcfd1954acb131f4e911c2a68f72c47f0008bde0912081b1a7.jpg)

![ae958c6adaecea75fc5d6f552f8e3d41d18790864a2081379d77aa23ebd23b16.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/ae958c6adaecea75fc5d6f552f8e3d41d18790864a2081379d77aa23ebd23b16.jpg)

![b6a3a55a5bca441bb81ca3017db64c1d794ef09fe74c44e112949367c70ddd69.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/b6a3a55a5bca441bb81ca3017db64c1d794ef09fe74c44e112949367c70ddd69.jpg)

![bed189dfb12368b2e7220f00b9254d4e2e4f300a1961de48f638176899e7f2a8.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/bed189dfb12368b2e7220f00b9254d4e2e4f300a1961de48f638176899e7f2a8.jpg)

![c68fd94c81d0369eccd2528239b99c2a66e12020ac65ef5dd3cea017ba593de7.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/c68fd94c81d0369eccd2528239b99c2a66e12020ac65ef5dd3cea017ba593de7.jpg)

![cf4bb2fc10dd6a51955855fd1dd4574299656544ec268e41b475302b25402a5d.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/cf4bb2fc10dd6a51955855fd1dd4574299656544ec268e41b475302b25402a5d.jpg)

![d263cb1d525a94c2945015d19298c764d6a8eb1f37d8419d5b409f4c1121618c.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/d263cb1d525a94c2945015d19298c764d6a8eb1f37d8419d5b409f4c1121618c.jpg)

![da0655bbe9ccc9b0fa7e5d05df8deff1955f96f8c984486e9e859cf4a9cc1c42.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/da0655bbe9ccc9b0fa7e5d05df8deff1955f96f8c984486e9e859cf4a9cc1c42.jpg)

![ddbce2af0f0a86e7e7662bf2492cc433bad1ae1c681f10582341f3a4c2fba986.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/ddbce2af0f0a86e7e7662bf2492cc433bad1ae1c681f10582341f3a4c2fba986.jpg)

![e6c85995748e4bcae41435f14e241cc16b7b14114e974e4f7e8dd1844c7e95c7.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/e6c85995748e4bcae41435f14e241cc16b7b14114e974e4f7e8dd1844c7e95c7.jpg)

![eaf45cc57019e61ce6fd16fa8fb80dcde2f546ec9cb01d86d2a4d6cee0422636.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/eaf45cc57019e61ce6fd16fa8fb80dcde2f546ec9cb01d86d2a4d6cee0422636.jpg)

![ee148bd34bdad05c1933108b279159393f8b7c71ce1a8fefd81c2deb4b8cd51a.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/ee148bd34bdad05c1933108b279159393f8b7c71ce1a8fefd81c2deb4b8cd51a.jpg)

![efba206c1f2d73cb1038aa432808d871fa51612af289dc8fa029ed9d37adaa7b.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/efba206c1f2d73cb1038aa432808d871fa51612af289dc8fa029ed9d37adaa7b.jpg)

![f7a5d8be9dd5dba7c5592eee0f06d4af0f5b93e0c4dffd91f16117e3fdf68f7a.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/f7a5d8be9dd5dba7c5592eee0f06d4af0f5b93e0c4dffd91f16117e3fdf68f7a.jpg)

![ff218603d5add8b6cfc2c9530d0ba45adc4f9d43cf26be431383aaab6fdbf44f.jpg](../icml_results/4677_GFFMERGE_%20Efficient%20Merging%20of%20Graph%20Neural%20Force%20Fields%20and%20Beyond/tables/ff218603d5add8b6cfc2c9530d0ba45adc4f9d43cf26be431383aaab6fdbf44f.jpg)

<a id="paper-4630"></a>

## Multi-agent imitation learning with function approximation: linear Markov games and beyond


### Images

![14155f30da67f443bdb9bafbeef2bd375ad12a8aadd7f92c7824397eee3b066f.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/images/14155f30da67f443bdb9bafbeef2bd375ad12a8aadd7f92c7824397eee3b066f.jpg)

![d1642aaef9bff96f84424894d5e757d28c3a7a41d730114b108e77eba2c968d1.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/images/d1642aaef9bff96f84424894d5e757d28c3a7a41d730114b108e77eba2c968d1.jpg)

### Tables

![0f52ae99b0e5b0b59995fa00999bce71a487c5d9268a03e7ee59a2e303390a91.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/tables/0f52ae99b0e5b0b59995fa00999bce71a487c5d9268a03e7ee59a2e303390a91.jpg)

![2a47b6cc92a3bd8ea6e520fadfb950cb5e20e7deaee3ac157537d9e1ae6993ad.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/tables/2a47b6cc92a3bd8ea6e520fadfb950cb5e20e7deaee3ac157537d9e1ae6993ad.jpg)

![507dd7f086aa63003a3001e629a00b323bc4ed33072637d87dbb9c349c9ad79f.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/tables/507dd7f086aa63003a3001e629a00b323bc4ed33072637d87dbb9c349c9ad79f.jpg)

![ee94691f0b952dacbc29ff0b1aa5b66ce45d6f33607f51a1f8da13e5f79448fd.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/tables/ee94691f0b952dacbc29ff0b1aa5b66ce45d6f33607f51a1f8da13e5f79448fd.jpg)

![fda94746cfac17f1f5946e5eede13cc15ed662d542c380e9685721b485e39a2f.jpg](../icml_results/4678_Multi-agent%20imitation%20learning%20with%20function%20approximation_%20linear%20Markov%20games%20and%20beyond/tables/fda94746cfac17f1f5946e5eede13cc15ed662d542c380e9685721b485e39a2f.jpg)

<a id="paper-4631"></a>

## Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks


### Images

![4e6bb3d23dc108cb0a031bb0cc1b1280888fe6ded615b6c4158d68f43efb398c.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/4e6bb3d23dc108cb0a031bb0cc1b1280888fe6ded615b6c4158d68f43efb398c.jpg)

![5eef75bcab8acb093a2642bf8b6cbd0e0169cc2e750be2ed2a59cd5810638e98.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/5eef75bcab8acb093a2642bf8b6cbd0e0169cc2e750be2ed2a59cd5810638e98.jpg)

![6ff6a2fd09102f321424eb3fe42f6445cfea0a8c835d637e34c56c353fca9de9.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/6ff6a2fd09102f321424eb3fe42f6445cfea0a8c835d637e34c56c353fca9de9.jpg)

![710db2ac389d0de4e3b5095da12af06ffb0486326cd54c8142e5842cc76ff93d.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/710db2ac389d0de4e3b5095da12af06ffb0486326cd54c8142e5842cc76ff93d.jpg)

![a51acd7906ae1f8f98a61969bf45c1c2e0a9f6d8f186293f8c3145a7abf17b6d.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/a51acd7906ae1f8f98a61969bf45c1c2e0a9f6d8f186293f8c3145a7abf17b6d.jpg)

![b7d6536753ac294bc4979611bfee5172bf97892bb005145653d8fc64c7819c08.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/b7d6536753ac294bc4979611bfee5172bf97892bb005145653d8fc64c7819c08.jpg)

![cdf49d5f076090a664f5751c792a72ed4fa76a162ab0224d86c37b948ed65ee4.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/images/cdf49d5f076090a664f5751c792a72ed4fa76a162ab0224d86c37b948ed65ee4.jpg)

### Tables

![206d53d602082bbb4c74f290cff09ef841e6d1c2c44a6a93cedccb60510d2746.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/206d53d602082bbb4c74f290cff09ef841e6d1c2c44a6a93cedccb60510d2746.jpg)

![23b8da705c72d13964f991c9a8a3074158a96006fcd9e321b8661fba76b90906.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/23b8da705c72d13964f991c9a8a3074158a96006fcd9e321b8661fba76b90906.jpg)

![56502696722ecbfd75e94aca4609c4750ee021320551c409da220379bcbb8762.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/56502696722ecbfd75e94aca4609c4750ee021320551c409da220379bcbb8762.jpg)

![6dd8d05a0eb3cdcad043023b1fdaa2e46f229d1312d96a4ea7aa7f3035d796cb.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/6dd8d05a0eb3cdcad043023b1fdaa2e46f229d1312d96a4ea7aa7f3035d796cb.jpg)

![6f43055a0dbf3555be1088421fc5528e61ade8b7a349ed8c86a0dd1101d70627.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/6f43055a0dbf3555be1088421fc5528e61ade8b7a349ed8c86a0dd1101d70627.jpg)

![8d485521624fb7f1869a328a5c6e2f8d36b9b0c0b9cfe3c656518945ab02065b.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/8d485521624fb7f1869a328a5c6e2f8d36b9b0c0b9cfe3c656518945ab02065b.jpg)

![b103c574e8d903e54ff98d9abf215822c575400376792a395e0f1689103149f0.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/b103c574e8d903e54ff98d9abf215822c575400376792a395e0f1689103149f0.jpg)

![e16dc5e7ee933188797d97c2500154de83f7152adaf910fb48631e7aa79b34c9.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/e16dc5e7ee933188797d97c2500154de83f7152adaf910fb48631e7aa79b34c9.jpg)

![f45fab540b007ef9b40c621f23b6ca776c2014c86537314ea124c4bf0e63db9d.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/f45fab540b007ef9b40c621f23b6ca776c2014c86537314ea124c4bf0e63db9d.jpg)

![f9fe5175e86fd6f4e236f29319a95cea45ea927688925b9f6cbf9aa0ef95bf5c.jpg](../icml_results/4679_Asking%20What%20Matters_%20Reward-Driven%20Clarification%20for%20Software%20Engineering%20Tasks/tables/f9fe5175e86fd6f4e236f29319a95cea45ea927688925b9f6cbf9aa0ef95bf5c.jpg)

<a id="paper-4632"></a>

## PACER: Acyclic Causal Discovery from Large-scale Interventional Data


### Images

![089671142749ed0b67e3e2c8259c0a923b01e2660173ffe84d8edd3a6665cd94.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/089671142749ed0b67e3e2c8259c0a923b01e2660173ffe84d8edd3a6665cd94.jpg)

![0c71b94ed545c2479cf0a834d4a8ccb970648bf11f05fb8b03fc9d8e14edb0f2.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/0c71b94ed545c2479cf0a834d4a8ccb970648bf11f05fb8b03fc9d8e14edb0f2.jpg)

![2c0dffffb316a5c3a40a51d6bb6f67d796a6c68486848aecefbd8beda153f498.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/2c0dffffb316a5c3a40a51d6bb6f67d796a6c68486848aecefbd8beda153f498.jpg)

![34b499e8c92ebfe12424f44c65da41a4ea673ac1a85a952bf4913abfdeb64848.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/34b499e8c92ebfe12424f44c65da41a4ea673ac1a85a952bf4913abfdeb64848.jpg)

![3726ee04644e7091f0c29b6e465443542ce41289f721ca8c69e117a7b1ed4b1f.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/3726ee04644e7091f0c29b6e465443542ce41289f721ca8c69e117a7b1ed4b1f.jpg)

![401b916cd664649527026bbee8f6170612ce602b415933c4fbc818e71b582570.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/401b916cd664649527026bbee8f6170612ce602b415933c4fbc818e71b582570.jpg)

![54f1f052c584502e7e61844009b6aec2328c3c216921ad079972d64ef1fca30b.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/54f1f052c584502e7e61844009b6aec2328c3c216921ad079972d64ef1fca30b.jpg)

![63bd24871e87f10bfc720a37cb420333cfa4f21bf93cfef7884bb84d0002b6ab.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/63bd24871e87f10bfc720a37cb420333cfa4f21bf93cfef7884bb84d0002b6ab.jpg)

![6712302752630771caf7494a744e4d6a1e9d6798a66728567defc6f57aeca3e2.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/6712302752630771caf7494a744e4d6a1e9d6798a66728567defc6f57aeca3e2.jpg)

![6b3d6fe17d08d82e711ab5d91ebab80bac7a843cf5719b0b2830f757981a7e1f.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/6b3d6fe17d08d82e711ab5d91ebab80bac7a843cf5719b0b2830f757981a7e1f.jpg)

![9a9a46827aa850f5d8807c0c46bc850b2c3136a5a24919a6e24e4d855176c77c.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/9a9a46827aa850f5d8807c0c46bc850b2c3136a5a24919a6e24e4d855176c77c.jpg)

![9aa021f6c3bee94bdfeedc0cb6f1887358ae016395fc55b190d691e9ee0cfb01.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/9aa021f6c3bee94bdfeedc0cb6f1887358ae016395fc55b190d691e9ee0cfb01.jpg)

![9de0ab3754bbf75288e64df04c1a3f0f56d7402eea1db25fd34d2b5df5a3bb71.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/9de0ab3754bbf75288e64df04c1a3f0f56d7402eea1db25fd34d2b5df5a3bb71.jpg)

![ae8c2b9ac4f23515447f5f8b23ed841be9789c8ca0774b8b1237504b2f809206.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/ae8c2b9ac4f23515447f5f8b23ed841be9789c8ca0774b8b1237504b2f809206.jpg)

![b7873fb2dca5b7bc58603789fc113e367f89a2e6f5319b065ab8c87fa869a80f.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/b7873fb2dca5b7bc58603789fc113e367f89a2e6f5319b065ab8c87fa869a80f.jpg)

![bfdfae71b3314e992ab3e6095863b7093729df0f9e9e6269f3b7365a0f97c274.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/bfdfae71b3314e992ab3e6095863b7093729df0f9e9e6269f3b7365a0f97c274.jpg)

![d41b7abd7671b526e4d08592421f028bda3494c1c0c56bcea68a0f3bab5b5b64.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/d41b7abd7671b526e4d08592421f028bda3494c1c0c56bcea68a0f3bab5b5b64.jpg)

![d61ff4167e64a928f4ea13a7dc0f490dbd7f886edef210a7e3043a6f845a3738.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/d61ff4167e64a928f4ea13a7dc0f490dbd7f886edef210a7e3043a6f845a3738.jpg)

![e2c548f4962cd0b7f300c70ef1ffea81fbc73e932c2ff9d7672c81c6fc2e84f8.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/e2c548f4962cd0b7f300c70ef1ffea81fbc73e932c2ff9d7672c81c6fc2e84f8.jpg)

![ef1019803072bcd7032c381a6af30bf6065b288001b7648f18d2010003dd91d4.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/ef1019803072bcd7032c381a6af30bf6065b288001b7648f18d2010003dd91d4.jpg)

![fffd5a847d5a6646720509e66b3d24ae8211ea1c762dcd42bf15e0f448b23347.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/images/fffd5a847d5a6646720509e66b3d24ae8211ea1c762dcd42bf15e0f448b23347.jpg)

### Tables

![0da9c4717a373d641041aa8f1d5be48548d0454ca883a500fd5a520bf0bd4dbd.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/0da9c4717a373d641041aa8f1d5be48548d0454ca883a500fd5a520bf0bd4dbd.jpg)

![232992b3342f0c9bf778de5d28f079bbcb8c965067aa6716cf78c99f9fc83a12.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/232992b3342f0c9bf778de5d28f079bbcb8c965067aa6716cf78c99f9fc83a12.jpg)

![2a719439e131ad1bb572076c45bc874b8e45c5213018278f16336be1023c6116.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/2a719439e131ad1bb572076c45bc874b8e45c5213018278f16336be1023c6116.jpg)

![408382b2d465a7cf470d471d145e6d50008e50881548827f5c0b90b9879c59c0.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/408382b2d465a7cf470d471d145e6d50008e50881548827f5c0b90b9879c59c0.jpg)

![5116acf819b115f14a635728fe5899927adba9cee9867da868a2d50a92952644.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/5116acf819b115f14a635728fe5899927adba9cee9867da868a2d50a92952644.jpg)

![862ad958b51c22f909ebd4c22b9ab607545cbb228c17ef777fed9e7aeef5d4e5.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/862ad958b51c22f909ebd4c22b9ab607545cbb228c17ef777fed9e7aeef5d4e5.jpg)

![890316a69a3e4ebf4a50f1f5d909cd759b3ad43780fcda134384bf9e5f25c879.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/890316a69a3e4ebf4a50f1f5d909cd759b3ad43780fcda134384bf9e5f25c879.jpg)

![a5bd1c37cc2abc77871578486ed1918566632a59fb14cb82ab45eeba10c261f1.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/a5bd1c37cc2abc77871578486ed1918566632a59fb14cb82ab45eeba10c261f1.jpg)

![d315eb8bdcaac59d1320d32b06ff113b0049ff6abeab81593f15f7dcc402b058.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/d315eb8bdcaac59d1320d32b06ff113b0049ff6abeab81593f15f7dcc402b058.jpg)

![f09823d918ba4cc8eef96a0a0453ae1c208695b212778e8caa036f8ca5349e65.jpg](../icml_results/4680_PACER_%20Acyclic%20Causal%20Discovery%20from%20Large-scale%20Interventional%20Data/tables/f09823d918ba4cc8eef96a0a0453ae1c208695b212778e8caa036f8ca5349e65.jpg)

<a id="paper-4633"></a>

## Optimal Domain-Aware Privacy Mechanisms for Synthetic Data Generation


### Images

![0ad3ed2ec191edb7b27d8e08f3956d1a805502096023ee8c9a3a49dfce0dbcd3.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/0ad3ed2ec191edb7b27d8e08f3956d1a805502096023ee8c9a3a49dfce0dbcd3.jpg)

![27585806b4171c8722d6613bdc6d3aa77ed32f9001942f3bbdb82ee8abce945b.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/27585806b4171c8722d6613bdc6d3aa77ed32f9001942f3bbdb82ee8abce945b.jpg)

![3b0192c74702eaae8d98f32ba5c9f18a4d9b14198b1d3a0ec789c9ce49b47de5.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/3b0192c74702eaae8d98f32ba5c9f18a4d9b14198b1d3a0ec789c9ce49b47de5.jpg)

![62b7e19a3f2a64ba41ffef7f1a417dcd7555c8d378de63fd67b25db0fec73763.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/62b7e19a3f2a64ba41ffef7f1a417dcd7555c8d378de63fd67b25db0fec73763.jpg)

![9c06662d3915636e924256307a04ba534fad2409b055cfa657e0536ab553ef82.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/9c06662d3915636e924256307a04ba534fad2409b055cfa657e0536ab553ef82.jpg)

![c3126b24f6082a982ad7e7b5b9dc871f2281fa07cbd71dec411477c6a60440e4.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/c3126b24f6082a982ad7e7b5b9dc871f2281fa07cbd71dec411477c6a60440e4.jpg)

![ccaef28787d172bf4ae2285016254d839ff5dda0f647f6efd9d1398321205ab8.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/images/ccaef28787d172bf4ae2285016254d839ff5dda0f647f6efd9d1398321205ab8.jpg)

### Tables

![2c1fcf0d95a4e204d1f632175dd6682abce68bba7805433a148c13ac9e7f06d1.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/tables/2c1fcf0d95a4e204d1f632175dd6682abce68bba7805433a148c13ac9e7f06d1.jpg)

![2cfd89115be4ee1407e23db05f14d45f06daf34e304593d37c78ded69d432236.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/tables/2cfd89115be4ee1407e23db05f14d45f06daf34e304593d37c78ded69d432236.jpg)

![de78e10f72ddfa53afe1b2d63fd1def83909cefb90f6de2a6c44f263c0cb5a82.jpg](../icml_results/4681_Optimal%20Domain-Aware%20Privacy%20Mechanisms%20for%20Synthetic%20Data%20Generation/tables/de78e10f72ddfa53afe1b2d63fd1def83909cefb90f6de2a6c44f263c0cb5a82.jpg)

<a id="paper-4634"></a>

## Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs


### Images

![23d52508680478ec94d62220bb8a9ad6a6f4c08ff36f5d6b7cdc93b454988b8a.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/23d52508680478ec94d62220bb8a9ad6a6f4c08ff36f5d6b7cdc93b454988b8a.jpg)

![2d1557daddfd655afb08db5aa34843ad3fd14eaf4a5f69afe894257ab92af91e.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/2d1557daddfd655afb08db5aa34843ad3fd14eaf4a5f69afe894257ab92af91e.jpg)

![32b3a971701abe30d948fbffebcf2c0618f62a33b963216f2d1ec53133fb037d.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/32b3a971701abe30d948fbffebcf2c0618f62a33b963216f2d1ec53133fb037d.jpg)

![4635ce49cd0ffd51e069558fb05b36a0f3ad967d467bd96fb3c75e6ab0251fea.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/4635ce49cd0ffd51e069558fb05b36a0f3ad967d467bd96fb3c75e6ab0251fea.jpg)

![67a1d4c7df891964f6fbc70cb90d3b3b9a895be0e303818871bec876a6ac3190.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/67a1d4c7df891964f6fbc70cb90d3b3b9a895be0e303818871bec876a6ac3190.jpg)

![830436b7a74d4e8a04c44cadcb6e7c820d62044806aededfb67aadafecc16f0c.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/830436b7a74d4e8a04c44cadcb6e7c820d62044806aededfb67aadafecc16f0c.jpg)

![94927208e9b2bfcafbfc689a3ab41d154af49c79fe72b4392cf05fa3d7f26dae.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/94927208e9b2bfcafbfc689a3ab41d154af49c79fe72b4392cf05fa3d7f26dae.jpg)

![990b6ffecfa13ccba07e00c9392d402d1bb6b509979603bd7650a7dc5a741953.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/990b6ffecfa13ccba07e00c9392d402d1bb6b509979603bd7650a7dc5a741953.jpg)

![b1ff5ffd2b79277213b9ba569b82b56151c3ee30b64b91f2414bb4851aa0d743.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/b1ff5ffd2b79277213b9ba569b82b56151c3ee30b64b91f2414bb4851aa0d743.jpg)

![b6c53c85d14b9cefad16c75f9bc19506e2bc46857c39ace268748c491b3e5779.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/b6c53c85d14b9cefad16c75f9bc19506e2bc46857c39ace268748c491b3e5779.jpg)

![d3e3ba2891393c4c7b3240abbbbf950fe5604c1ff94969e350ec1d0a7b2dab81.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/d3e3ba2891393c4c7b3240abbbbf950fe5604c1ff94969e350ec1d0a7b2dab81.jpg)

![f0bebae13a908e21b1c269729d3193aceeac4b66d590db54c3edbd655f9b71f1.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/f0bebae13a908e21b1c269729d3193aceeac4b66d590db54c3edbd655f9b71f1.jpg)

![f4c45c48847eb7e4b1dea6bfaf7302c3e556da62783afe5b2ae264e28c3e6d62.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/images/f4c45c48847eb7e4b1dea6bfaf7302c3e556da62783afe5b2ae264e28c3e6d62.jpg)

### Tables

![1a848bc3958b76d0da83f67d0dae08d5bf6a05f391ae6a994df33ce90ef5b286.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/1a848bc3958b76d0da83f67d0dae08d5bf6a05f391ae6a994df33ce90ef5b286.jpg)

![1ba05334da2cc712fe0bec2b785beb49165dd2c1818c878c01d3fc4e92b63fa0.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/1ba05334da2cc712fe0bec2b785beb49165dd2c1818c878c01d3fc4e92b63fa0.jpg)

![1d80ebd1fc2589eeaa0713696261030b75ddad1affa2a1424e3f8be102ae01ac.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/1d80ebd1fc2589eeaa0713696261030b75ddad1affa2a1424e3f8be102ae01ac.jpg)

![1f73a2c3388817e9acc6a5dc2704f2429b12906eb519480371af40f76b2a3e9a.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/1f73a2c3388817e9acc6a5dc2704f2429b12906eb519480371af40f76b2a3e9a.jpg)

![386e2801a4a771f5bba647f92a6e5da0cf7419d7cee7581d403b920992d82e00.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/386e2801a4a771f5bba647f92a6e5da0cf7419d7cee7581d403b920992d82e00.jpg)

![62dc7d412875b92e431b746cc1ea43fc6194676b2356088f1759a98398c8bc54.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/62dc7d412875b92e431b746cc1ea43fc6194676b2356088f1759a98398c8bc54.jpg)

![b9d01b0b7c0b0040395ce4dc1af7419bd864a79d8261d6174e6b4a1cc242bf02.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/b9d01b0b7c0b0040395ce4dc1af7419bd864a79d8261d6174e6b4a1cc242bf02.jpg)

![be5a25c245939d58e41b41e3f576ab4820425885d0bafdd0e9d3616dd7fc1bd5.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/be5a25c245939d58e41b41e3f576ab4820425885d0bafdd0e9d3616dd7fc1bd5.jpg)

![d347f705a88ebab28329d1c3a4139369017290b0d152be4f4823801a93ff1b7b.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/d347f705a88ebab28329d1c3a4139369017290b0d152be4f4823801a93ff1b7b.jpg)

![db29fb619df17ec29a82bae3974ee062caf84fdae910200484b5e2840c89683c.jpg](../icml_results/4682_H%C3%B6lder%2B%2B_%20Improving%20the%20Quality-Coherence%20Trade-off%20in%20Multimodal%20VAEs/tables/db29fb619df17ec29a82bae3974ee062caf84fdae910200484b5e2840c89683c.jpg)

<a id="paper-4635"></a>

## Off-Policy Evaluation with Strategic Agents via Local Disclosure


### Images

![1ef74c3686604bea9f490b435f15e4e0210bf521d60b2ff9f8e1c502420e9487.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/1ef74c3686604bea9f490b435f15e4e0210bf521d60b2ff9f8e1c502420e9487.jpg)

![20a7d9177957a99a8b7aa51ad3182befe2e88e22dd106274df25a7d53da05288.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/20a7d9177957a99a8b7aa51ad3182befe2e88e22dd106274df25a7d53da05288.jpg)

![3893f252cf949fc3ff35a0be0ea9a46850bd66c2c0c961d386bd35baf9c51ce4.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/3893f252cf949fc3ff35a0be0ea9a46850bd66c2c0c961d386bd35baf9c51ce4.jpg)

![5e7acc2972943fb1d8caae066e34920c45e5bacfd41ab3a7f3286ae7b83a4cd2.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/5e7acc2972943fb1d8caae066e34920c45e5bacfd41ab3a7f3286ae7b83a4cd2.jpg)

![8cd6ffdbe798346fdf784ac40e6199c57db3f4cfa7cd2d39bd9361a1e941f91b.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/8cd6ffdbe798346fdf784ac40e6199c57db3f4cfa7cd2d39bd9361a1e941f91b.jpg)

![8ebd8e6c69b872102e217ee546be696d66ea1754671937cf787a55f1bbfd8522.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/8ebd8e6c69b872102e217ee546be696d66ea1754671937cf787a55f1bbfd8522.jpg)

![f87a0a5832ae05b5ed96d4971fb066aed56b701fd4189427aaeac54fdcdc06b6.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/images/f87a0a5832ae05b5ed96d4971fb066aed56b701fd4189427aaeac54fdcdc06b6.jpg)

### Tables

![1b6be5292a1df072dd8307f557e5d5a9622e0d495b67efb70ce1b6dac9cdfcf5.jpg](../icml_results/4683_Off-Policy%20Evaluation%20with%20Strategic%20Agents%20via%20Local%20Disclosure/tables/1b6be5292a1df072dd8307f557e5d5a9622e0d495b67efb70ce1b6dac9cdfcf5.jpg)

<a id="paper-4636"></a>

## Riemannian Optimization for Fair Spectral Clustering


### Images

![096f32a18ceb7230d4c197017237b00db27efc49422136ed08488bfbb7aa3f5f.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/096f32a18ceb7230d4c197017237b00db27efc49422136ed08488bfbb7aa3f5f.jpg)

![72bd228b905781b294094faf137c28bee0460f8111859e8b52f1c7a5660a6b85.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/72bd228b905781b294094faf137c28bee0460f8111859e8b52f1c7a5660a6b85.jpg)

![744d147f367ddd503908645e0ac62c83f655851f496a4a125e6de4ca641842e6.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/744d147f367ddd503908645e0ac62c83f655851f496a4a125e6de4ca641842e6.jpg)

![d4a6dfe2f8cb87979e665667ddcdb8803d5bcc8ea14e23ac931c5eaf1a472056.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/d4a6dfe2f8cb87979e665667ddcdb8803d5bcc8ea14e23ac931c5eaf1a472056.jpg)

![ef15193272ce3586c40f463363cce297ced39866d88da4262df8076b1ad9b24d.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/ef15193272ce3586c40f463363cce297ced39866d88da4262df8076b1ad9b24d.jpg)

![fb0cbfdd8024927bf44c37fb88f8213fedd3c4cdaa037294dcd699cd9959b3de.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/images/fb0cbfdd8024927bf44c37fb88f8213fedd3c4cdaa037294dcd699cd9959b3de.jpg)

### Tables

![1f00712f9698d06bd4545534029a869dccdd0ff44c8982046bc1a08762019c74.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/1f00712f9698d06bd4545534029a869dccdd0ff44c8982046bc1a08762019c74.jpg)

![247b3ece91e5575b3578c46d2ca555d4fc6444191d7cc33e391093e2008b9252.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/247b3ece91e5575b3578c46d2ca555d4fc6444191d7cc33e391093e2008b9252.jpg)

![310f4218a4605117c2a1f06583a722942e0a76f2f87e99b7eed4b40066f860d4.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/310f4218a4605117c2a1f06583a722942e0a76f2f87e99b7eed4b40066f860d4.jpg)

![42b03fafa4973d752676e40ba4e22343f2d7fd4fad3f83361917ad4edd4a58b2.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/42b03fafa4973d752676e40ba4e22343f2d7fd4fad3f83361917ad4edd4a58b2.jpg)

![54df738ceb5914384459bb6d69150cd3f85e7cc6a8ec400d8f2a99e279309da0.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/54df738ceb5914384459bb6d69150cd3f85e7cc6a8ec400d8f2a99e279309da0.jpg)

![6d5c22435f9821a24c62126a0d142e40bfe6d095d58cb2fe748e839e642ae819.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/6d5c22435f9821a24c62126a0d142e40bfe6d095d58cb2fe748e839e642ae819.jpg)

![88f960a84602720e307cb9b3e522d378805e32b309972a71109431e62ed8a177.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/88f960a84602720e307cb9b3e522d378805e32b309972a71109431e62ed8a177.jpg)

![b0b3ca5a0680896ebfc7a6048ea158a0293d776d74dfa9803ffdfe8ff21c72a3.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/b0b3ca5a0680896ebfc7a6048ea158a0293d776d74dfa9803ffdfe8ff21c72a3.jpg)

![b7e104f32664003c8c40666392c5ed67f51f7169f460a6fd1c43dffcca760b80.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/b7e104f32664003c8c40666392c5ed67f51f7169f460a6fd1c43dffcca760b80.jpg)

![da7c53732dd385e42525b0fa880d43871a29ece469ecc9b7f61c3d56974c8b3d.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/da7c53732dd385e42525b0fa880d43871a29ece469ecc9b7f61c3d56974c8b3d.jpg)

![dc9c05b5fed3483511dc7d48ed892d96c660e56d770de0a44bf0b3a1b63b2bd3.jpg](../icml_results/4684_Riemannian%20Optimization%20for%20Fair%20Spectral%20Clustering/tables/dc9c05b5fed3483511dc7d48ed892d96c660e56d770de0a44bf0b3a1b63b2bd3.jpg)

<a id="paper-4637"></a>

## $f$-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models


### Images

![0ac4bf721301047901bbf52fd0306492fc7d7fac1f93ce303b2890506126b2f7.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/0ac4bf721301047901bbf52fd0306492fc7d7fac1f93ce303b2890506126b2f7.jpg)

![797db33cfe85140359551ea4631b2655d5dee2bdcb81ccdfc1cfb09ce773e5d2.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/797db33cfe85140359551ea4631b2655d5dee2bdcb81ccdfc1cfb09ce773e5d2.jpg)

![83c5fe991713088860d5afc3175ddd8af4a0204a20ae35db27b38388cf7c1dbd.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/83c5fe991713088860d5afc3175ddd8af4a0204a20ae35db27b38388cf7c1dbd.jpg)

![8a8b82ba7f2cf1a3e4390c5a4cb63ef102f83b7e7c35c0168f85a2dcf5d7ef7e.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/8a8b82ba7f2cf1a3e4390c5a4cb63ef102f83b7e7c35c0168f85a2dcf5d7ef7e.jpg)

![9041f2cf24f96c2f799bd5a6509e0914dfe3f3de2d27d86ff791c3133ac49480.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/9041f2cf24f96c2f799bd5a6509e0914dfe3f3de2d27d86ff791c3133ac49480.jpg)

![aae89a55e3f3087a8442dbd3d3ec51420c0c89ec334718c02db0424a07b79e94.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/aae89a55e3f3087a8442dbd3d3ec51420c0c89ec334718c02db0424a07b79e94.jpg)

![b62d093036d495a3a26afa11461b46286fbf3c1cedd7bbce422fa2105241a84b.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/images/b62d093036d495a3a26afa11461b46286fbf3c1cedd7bbce422fa2105241a84b.jpg)

### Tables

![005e7e62c24f5798a0c1277e283b7fccfbacd2e5137729110b2ca8685180b47f.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/005e7e62c24f5798a0c1277e283b7fccfbacd2e5137729110b2ca8685180b47f.jpg)

![09ce7f2f85dc0314119bbce17858d27ae4cdb10c03a7e10db224d15c275d87fc.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/09ce7f2f85dc0314119bbce17858d27ae4cdb10c03a7e10db224d15c275d87fc.jpg)

![16f51c9c0b5b194c6bcdee1e6edbc40834fcc62118d2126adf07b9518b780790.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/16f51c9c0b5b194c6bcdee1e6edbc40834fcc62118d2126adf07b9518b780790.jpg)

![27098dc29734309ac6d51042eaa9a178747876db45bd61a5289c5aee42d590fa.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/27098dc29734309ac6d51042eaa9a178747876db45bd61a5289c5aee42d590fa.jpg)

![6dd61e55602c54265d5ae26086ce504f8ded7d1b7b36f1b34af8855cf5a68bbb.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/6dd61e55602c54265d5ae26086ce504f8ded7d1b7b36f1b34af8855cf5a68bbb.jpg)

![7ba3c317316b5bafbe443628bbc3e743637fd884b54f3243890064760ee56a57.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/7ba3c317316b5bafbe443628bbc3e743637fd884b54f3243890064760ee56a57.jpg)

![85f2f5fd63663fe69fb361927a1cb2b478353cfc4d614564873e9f28c64b7b27.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/85f2f5fd63663fe69fb361927a1cb2b478353cfc4d614564873e9f28c64b7b27.jpg)

![aa07c9f192c0bb2cdde0f7dfcb921656cb264b4245eb8666c5238487b9df535a.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/aa07c9f192c0bb2cdde0f7dfcb921656cb264b4245eb8666c5238487b9df535a.jpg)

![b511bed580c0d0329c80550a5fc00e25e22aed52e6fc9ff27a31a89542aea615.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/b511bed580c0d0329c80550a5fc00e25e22aed52e6fc9ff27a31a89542aea615.jpg)

![be6c25bd14b90a982cf40e75e18495ba04f8bfa86929db583edcf7631b270ad8.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/be6c25bd14b90a982cf40e75e18495ba04f8bfa86929db583edcf7631b270ad8.jpg)

![bf184e713c5f627f2a7087d5f8cfef67d385021cac5de06298ef1f0b19cd9833.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/bf184e713c5f627f2a7087d5f8cfef67d385021cac5de06298ef1f0b19cd9833.jpg)

![e6a5a76e86d0fd77ee18aca0b681d2937c670a47a4a94f538fcc86ddb91021ae.jpg](../icml_results/4685_%24f%24-Divergence%20Self-Play%20for%20Tabular%20Anomaly%20Detection%20via%20Large%20Language%20Models/tables/e6a5a76e86d0fd77ee18aca0b681d2937c670a47a4a94f538fcc86ddb91021ae.jpg)

<a id="paper-4638"></a>

## New Bounds for Kernel Sums via Fast Spherical Embeddings


### Images

![716a9cad1d539753c39049e7492351a2ec5a8f10f0bbfcf902fb2868bbb016e1.jpg](../icml_results/4686_New%20Bounds%20for%20Kernel%20Sums%20via%20Fast%20Spherical%20Embeddings/images/716a9cad1d539753c39049e7492351a2ec5a8f10f0bbfcf902fb2868bbb016e1.jpg)

### Tables

![190679e738fa7476bdb1480eaa249a89b0d8c07edcde7b570b7548a18ca6d71b.jpg](../icml_results/4686_New%20Bounds%20for%20Kernel%20Sums%20via%20Fast%20Spherical%20Embeddings/tables/190679e738fa7476bdb1480eaa249a89b0d8c07edcde7b570b7548a18ca6d71b.jpg)

<a id="paper-4639"></a>

## A Probabilistic Framework for LLM-Based Model Discovery


### Images

![0147614451dbe8aa1bd4a122c38764eca2fc320bb056ef16ab8b64129fa3a781.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/0147614451dbe8aa1bd4a122c38764eca2fc320bb056ef16ab8b64129fa3a781.jpg)

![1e36062066f084b2da95b2b21fa8584c5a8486921cb8027cd03489814625dd32.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/1e36062066f084b2da95b2b21fa8584c5a8486921cb8027cd03489814625dd32.jpg)

![3501c09231848216e2db2e500adfa908b9abede286535e5a8bf7bd1dc2ab39a7.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/3501c09231848216e2db2e500adfa908b9abede286535e5a8bf7bd1dc2ab39a7.jpg)

![4208f5787542e5a8c0f83d209a80ee198e10773eb6844c396aab91fb271b72f2.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/4208f5787542e5a8c0f83d209a80ee198e10773eb6844c396aab91fb271b72f2.jpg)

![43a882d9299b00cb567fca77d50779330ac2901769d72ffc72302efc6fc3bc8c.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/43a882d9299b00cb567fca77d50779330ac2901769d72ffc72302efc6fc3bc8c.jpg)

![4802dab4aa9888521fdb74744042c7baabbef5d915e2ad608136c0498be90b04.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/4802dab4aa9888521fdb74744042c7baabbef5d915e2ad608136c0498be90b04.jpg)

![4add908593c94e93bfc00e715f6050298ae91b29f597ad9366ef84f0b8de1c98.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/4add908593c94e93bfc00e715f6050298ae91b29f597ad9366ef84f0b8de1c98.jpg)

![5218d79e0f3aed009c49df1a13574e9b3397f8dd8aa3e83a4ff2de23c54fdde6.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/5218d79e0f3aed009c49df1a13574e9b3397f8dd8aa3e83a4ff2de23c54fdde6.jpg)

![576941c79ab47e851e4986d662aebc833a3f27dbf3aa43667c9ca142ea8dfe2e.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/576941c79ab47e851e4986d662aebc833a3f27dbf3aa43667c9ca142ea8dfe2e.jpg)

![60bfec039b1b73a6d35d975e1377d764e79d562720bf6f9f63579ea5f52273fb.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/60bfec039b1b73a6d35d975e1377d764e79d562720bf6f9f63579ea5f52273fb.jpg)

![76eedac96eca49e32bb9eca8b6f7c73d903749cd8c3551398badac3732266240.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/76eedac96eca49e32bb9eca8b6f7c73d903749cd8c3551398badac3732266240.jpg)

![845a0f1f7b33dd235b7d8b23825b620827ea26c34a774167edf6a230a377e4e6.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/845a0f1f7b33dd235b7d8b23825b620827ea26c34a774167edf6a230a377e4e6.jpg)

![85136bf9c519d0eb0cdf277af3e41699a9c82c93b9cb0b56d5a4f3b83915e47a.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/85136bf9c519d0eb0cdf277af3e41699a9c82c93b9cb0b56d5a4f3b83915e47a.jpg)

![8e94bb5d59d829dc4511f2adfe36d9988de1ba66c6a0de20633c2b089f0f0eb0.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/8e94bb5d59d829dc4511f2adfe36d9988de1ba66c6a0de20633c2b089f0f0eb0.jpg)

![954056b1d7e04c1c107d4e65c5ad3ac63c55a8361bb7853fb9117264b36e3146.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/954056b1d7e04c1c107d4e65c5ad3ac63c55a8361bb7853fb9117264b36e3146.jpg)

![a6fa1f2aea6d5783aa4ba1ed9d25e02221e816488bf28d070ddd370bfd96fcd2.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/a6fa1f2aea6d5783aa4ba1ed9d25e02221e816488bf28d070ddd370bfd96fcd2.jpg)

![c7158624c1265abcadddde85ea53398fffe45b9c97a823709200aad768963ee1.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/c7158624c1265abcadddde85ea53398fffe45b9c97a823709200aad768963ee1.jpg)

![d9dba20a649b61f53ea307063908021e8367c4b6ec8d39c094b1884379cfd995.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/d9dba20a649b61f53ea307063908021e8367c4b6ec8d39c094b1884379cfd995.jpg)

![ed44e49a780811dfec8f605a21077520318f2504c9478d3f4ae180dd3db7df3f.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/ed44e49a780811dfec8f605a21077520318f2504c9478d3f4ae180dd3db7df3f.jpg)

![efbbcb99fca9ba2281285c7197c5dbe5da3f8e0449a19fabd927b4231b39b66b.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/images/efbbcb99fca9ba2281285c7197c5dbe5da3f8e0449a19fabd927b4231b39b66b.jpg)

### Tables

![115922d1ccaf081335b02da5407e3ec2bc12052aafcc8d3a84ad518b5745e65f.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/115922d1ccaf081335b02da5407e3ec2bc12052aafcc8d3a84ad518b5745e65f.jpg)

![1e86bfa5c053036bb8ab143ffbff1fe79970e64c7c23be0b49e2c91044b59087.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/1e86bfa5c053036bb8ab143ffbff1fe79970e64c7c23be0b49e2c91044b59087.jpg)

![4cb1aded5f695c5367bb65d86ff993cf0d023d7c6a6307ccedf19117d6dca21c.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/4cb1aded5f695c5367bb65d86ff993cf0d023d7c6a6307ccedf19117d6dca21c.jpg)

![5882bcb6e25be0b2d34792ad911542eebc64c1825a669677063f7a28e35826f4.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/5882bcb6e25be0b2d34792ad911542eebc64c1825a669677063f7a28e35826f4.jpg)

![8bbc5b686f8f17a72e5b2f66527828ec9559c89c915133ccec446f520a4e4d85.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/8bbc5b686f8f17a72e5b2f66527828ec9559c89c915133ccec446f520a4e4d85.jpg)

![a976ef07d5f81601cca4a9546fbf5ab95a026bf3e6e073d671e85c3486960a16.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/a976ef07d5f81601cca4a9546fbf5ab95a026bf3e6e073d671e85c3486960a16.jpg)

![e4633858c89fee6ad9e534494534f1151150e7d57de8832dbbe10aaa63aeff42.jpg](../icml_results/4687_A%20Probabilistic%20Framework%20for%20LLM-Based%20Model%20Discovery/tables/e4633858c89fee6ad9e534494534f1151150e7d57de8832dbbe10aaa63aeff42.jpg)

<a id="paper-4640"></a>

## A Decision-Theoretic View of Test-Time Training: When, How Far, and Which Directions to Adapt


### Images

![1843b7cbc5a76a71a6ce6f64bcaf23241307d1317b97b121abad680dd4f77186.jpg](../icml_results/4688_A%20Decision-Theoretic%20View%20of%20Test-Time%20Training_%20When%2C%20How%20Far%2C%20and%20Which%20Directions%20to%20Adapt/images/1843b7cbc5a76a71a6ce6f64bcaf23241307d1317b97b121abad680dd4f77186.jpg)

### Tables

![184bd76863684e81adc9fd8fd134a89ca87673dd1fbfb99a3f58d7b338fc6d53.jpg](../icml_results/4688_A%20Decision-Theoretic%20View%20of%20Test-Time%20Training_%20When%2C%20How%20Far%2C%20and%20Which%20Directions%20to%20Adapt/tables/184bd76863684e81adc9fd8fd134a89ca87673dd1fbfb99a3f58d7b338fc6d53.jpg)

![87904ee845e8c7d2fb710e245e45835bb5a29878c82dc617c37b27529ed0946b.jpg](../icml_results/4688_A%20Decision-Theoretic%20View%20of%20Test-Time%20Training_%20When%2C%20How%20Far%2C%20and%20Which%20Directions%20to%20Adapt/tables/87904ee845e8c7d2fb710e245e45835bb5a29878c82dc617c37b27529ed0946b.jpg)

![a6254d5ef3731293af29e2766a45f6f7db2c11f4d55d1345409b6ae2888886be.jpg](../icml_results/4688_A%20Decision-Theoretic%20View%20of%20Test-Time%20Training_%20When%2C%20How%20Far%2C%20and%20Which%20Directions%20to%20Adapt/tables/a6254d5ef3731293af29e2766a45f6f7db2c11f4d55d1345409b6ae2888886be.jpg)

![d15b4d8d3a9a6cea8d48d26234667a1f8af57b55e0a01eab8c09417dafb644a9.jpg](../icml_results/4688_A%20Decision-Theoretic%20View%20of%20Test-Time%20Training_%20When%2C%20How%20Far%2C%20and%20Which%20Directions%20to%20Adapt/tables/d15b4d8d3a9a6cea8d48d26234667a1f8af57b55e0a01eab8c09417dafb644a9.jpg)

<a id="paper-4641"></a>

## In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning


### Images

![76ae7872b674af8313c9fa991d5696e19ccae3d792d8004e1314604c5b8d8d13.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/images/76ae7872b674af8313c9fa991d5696e19ccae3d792d8004e1314604c5b8d8d13.jpg)

![908c2ffe2d544a707198a920b2f81c823e4b0be21284d9731d60818b5f554257.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/images/908c2ffe2d544a707198a920b2f81c823e4b0be21284d9731d60818b5f554257.jpg)

![af11380dd8d880089baa0bb8e93c36cbea3e9b3739133469ea64bfc0d780b165.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/images/af11380dd8d880089baa0bb8e93c36cbea3e9b3739133469ea64bfc0d780b165.jpg)

![f743369171d1ea156548f5c0844510b156b7829ede820cab2e476ed229fc4879.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/images/f743369171d1ea156548f5c0844510b156b7829ede820cab2e476ed229fc4879.jpg)

### Tables

![047fdeb25980e5c0df2818faf84998c03bf056a1df0c17251784bfffe3818400.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/tables/047fdeb25980e5c0df2818faf84998c03bf056a1df0c17251784bfffe3818400.jpg)

![a5fcdc8283c8f6377215b94cf0facb350e6c1756ca537ee5532f76412386418e.jpg](../icml_results/4689_In-Context%20Learning%20Is%20Provably%20Bayesian%20Inference_%20A%20Generalization%20Theory%20for%20Meta-Learning/tables/a5fcdc8283c8f6377215b94cf0facb350e6c1756ca537ee5532f76412386418e.jpg)

<a id="paper-4642"></a>

## CAOS: Conformal Aggregation of One-Shot Predictors


### Images

![2a4aa86a6974d9da90f56903fef59a1c377c6a25a6585c627694016bd7556199.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/2a4aa86a6974d9da90f56903fef59a1c377c6a25a6585c627694016bd7556199.jpg)

![4318759a67f49751b89a1561b7f4cfdfd4d6ecce2e2e4a08df4c1c55f26b6653.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/4318759a67f49751b89a1561b7f4cfdfd4d6ecce2e2e4a08df4c1c55f26b6653.jpg)

![523b011c152811578729041f8e01be17eff1015085a2e58c09b672b7ea6dc62a.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/523b011c152811578729041f8e01be17eff1015085a2e58c09b672b7ea6dc62a.jpg)

![5f0c987bc97455bfc8522b5858d38924d8710af1063a70b7cba60b3e5015e62d.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/5f0c987bc97455bfc8522b5858d38924d8710af1063a70b7cba60b3e5015e62d.jpg)

![707becd7824e97b95069500996013e83e163a904f5c1cff3ddc5ca289ee47acc.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/707becd7824e97b95069500996013e83e163a904f5c1cff3ddc5ca289ee47acc.jpg)

![990725b76df9a9213f4975fdb391f9a171130a211f1e1e489516a13ef1b75b24.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/990725b76df9a9213f4975fdb391f9a171130a211f1e1e489516a13ef1b75b24.jpg)

![ac88d5ebd331849bf4782785cb5d2fb9405114ad3c143f2a62b8770d37746bd4.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/ac88d5ebd331849bf4782785cb5d2fb9405114ad3c143f2a62b8770d37746bd4.jpg)

![cb9558ceadc2493ea480f7933e71b99f6a918cf5ea2a09a02bcb9651353406b2.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/cb9558ceadc2493ea480f7933e71b99f6a918cf5ea2a09a02bcb9651353406b2.jpg)

![f37774b5fef503688df15eb6227c1762f77f20814ee2ed01e01042422c599e46.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/images/f37774b5fef503688df15eb6227c1762f77f20814ee2ed01e01042422c599e46.jpg)

### Tables

![37fe1543895419061c0c1e50140b9ae11e6e0a8b4168475c86b8a5d8ef2606c4.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/tables/37fe1543895419061c0c1e50140b9ae11e6e0a8b4168475c86b8a5d8ef2606c4.jpg)

![b583e8e79aa082682f5e28b0da1aa6d24a6d3ab35feda422649f21cab29bc4f5.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/tables/b583e8e79aa082682f5e28b0da1aa6d24a6d3ab35feda422649f21cab29bc4f5.jpg)

![fdadfaa41a39165e907298bd8f88995c0321e0f72acbeec346a3ffef6dbaca6e.jpg](../icml_results/4690_CAOS_%20Conformal%20Aggregation%20of%20One-Shot%20Predictors/tables/fdadfaa41a39165e907298bd8f88995c0321e0f72acbeec346a3ffef6dbaca6e.jpg)

<a id="paper-4643"></a>

## Hidden in Plain Sight – Class Competition Focuses Attribution Maps


### Images

![0a0f048ef321bbb6f77c8b8f70f0e8223e6d7b2763c662608c2fc8ef6e6c7757.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/0a0f048ef321bbb6f77c8b8f70f0e8223e6d7b2763c662608c2fc8ef6e6c7757.jpg)

![1015a4beef90b9f242db3be30d2de52170db0ebd997df1928596fd5e9123384f.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/1015a4beef90b9f242db3be30d2de52170db0ebd997df1928596fd5e9123384f.jpg)

![1329b5632f538409d65256e3dac149acd3670fcb4e1c0befc71c9c5809646466.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/1329b5632f538409d65256e3dac149acd3670fcb4e1c0befc71c9c5809646466.jpg)

![1864fd3123cbc44432ee51cc19f98c8a44acafa7caeb0fb5a7ac705d6d3dc278.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/1864fd3123cbc44432ee51cc19f98c8a44acafa7caeb0fb5a7ac705d6d3dc278.jpg)

![1ed164b76e72b31334aba2fb19c2a10919ab67e8369636a3102eb1f1f7fcd265.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/1ed164b76e72b31334aba2fb19c2a10919ab67e8369636a3102eb1f1f7fcd265.jpg)

![2ef6522e5eec462c9cf31f0586f734e4939809eb5013f5fabd22b7a4ca1325be.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/2ef6522e5eec462c9cf31f0586f734e4939809eb5013f5fabd22b7a4ca1325be.jpg)

![4191e2b83e225c56c09faf7e61425c6251a3b6e7904c48e8dde8aa10e78e07c7.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/4191e2b83e225c56c09faf7e61425c6251a3b6e7904c48e8dde8aa10e78e07c7.jpg)

![43b4ff8331ec54b950afc309b1e514d60f41ffc5d319dc1eacd6918e57be3464.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/43b4ff8331ec54b950afc309b1e514d60f41ffc5d319dc1eacd6918e57be3464.jpg)

![461da759c15234f5584e9f68646dddcab1cdb2e553a463410784c7861b0551cd.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/461da759c15234f5584e9f68646dddcab1cdb2e553a463410784c7861b0551cd.jpg)

![48ae5bacffea095eb1eb42cf4b9af40a6aba39781b30b60a3de32661a47f70d0.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/48ae5bacffea095eb1eb42cf4b9af40a6aba39781b30b60a3de32661a47f70d0.jpg)

![4bb1d6862a112c298ce8b27193be97c6885e27dd59035ddfc7938f78b655d03a.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/4bb1d6862a112c298ce8b27193be97c6885e27dd59035ddfc7938f78b655d03a.jpg)

![4dfe306bcf4708424e748b738e82fd13ba0bb2fb32a939a25b91765d5bc45830.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/4dfe306bcf4708424e748b738e82fd13ba0bb2fb32a939a25b91765d5bc45830.jpg)

![5504f530d3ab6b49d4747461cbfb3ed07e0e0358b1e9702b14ec2ae01bea30b8.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/5504f530d3ab6b49d4747461cbfb3ed07e0e0358b1e9702b14ec2ae01bea30b8.jpg)

![5ba06d0fbd31380893cd0af99f6d7b0364baba16b4328aae94a645efb82ef958.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/5ba06d0fbd31380893cd0af99f6d7b0364baba16b4328aae94a645efb82ef958.jpg)

![7b9d8777acd654946f9cc095ce2ac970a228efdcf6eed84fa71d18ea9854fac6.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/7b9d8777acd654946f9cc095ce2ac970a228efdcf6eed84fa71d18ea9854fac6.jpg)

![8f250a418299623fddab150cd1ffca6ea34042787c99c5482d1f76af4b31c8eb.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/8f250a418299623fddab150cd1ffca6ea34042787c99c5482d1f76af4b31c8eb.jpg)

![8fa188409525084dabb118bf9f25680611ff865cf3c5d7d0407a68c69c945dfd.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/8fa188409525084dabb118bf9f25680611ff865cf3c5d7d0407a68c69c945dfd.jpg)

![96d5134be4496385371caf6fa220bb99aa31899cb51e1565cc80525879d0cc26.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/96d5134be4496385371caf6fa220bb99aa31899cb51e1565cc80525879d0cc26.jpg)

![9791e7033562c86cb149700a322c6d897a4c215b732bf8311ca3e60690d0797b.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/9791e7033562c86cb149700a322c6d897a4c215b732bf8311ca3e60690d0797b.jpg)

![c2ffd4c9e457a6fcff538c45e227942a2e5716d1f0048ebd4c4804db74d86e46.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/c2ffd4c9e457a6fcff538c45e227942a2e5716d1f0048ebd4c4804db74d86e46.jpg)

![d2c2f969a1d23811d90004e6edbf545db4b4f9216a526c1f681cae600cf78ea3.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/d2c2f969a1d23811d90004e6edbf545db4b4f9216a526c1f681cae600cf78ea3.jpg)

![dd48057776ac16ff91a7c7fc47f7de387af024b1f99224543c85b70aa98f9a4c.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/dd48057776ac16ff91a7c7fc47f7de387af024b1f99224543c85b70aa98f9a4c.jpg)

![ddd31f1a52c3fe2e9666177c205fd43c2c79be54944c644c8cf2072344f3174a.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/ddd31f1a52c3fe2e9666177c205fd43c2c79be54944c644c8cf2072344f3174a.jpg)

![ebf045057702367b0a9817b87da8d53588674b17974503e1ab84bf51b1db97ba.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/ebf045057702367b0a9817b87da8d53588674b17974503e1ab84bf51b1db97ba.jpg)

![f10cbad9cba9b49206eabaf925d4fc19c0b27cc4c40a0fa34eac35871fcae4ef.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/f10cbad9cba9b49206eabaf925d4fc19c0b27cc4c40a0fa34eac35871fcae4ef.jpg)

![f66226afad74e66bf51a4c182c4403a82680a0ec664f0b0a4e83fe67fe16289a.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/f66226afad74e66bf51a4c182c4403a82680a0ec664f0b0a4e83fe67fe16289a.jpg)

![fd6e3e8542a512ef4c505fe2ff8be6f8ce179d43f9ef8c7f94e55b48ed5c2259.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/images/fd6e3e8542a512ef4c505fe2ff8be6f8ce179d43f9ef8c7f94e55b48ed5c2259.jpg)

### Tables

![2627379ab56667ad70a4b55773425faf7ac2cbd30498e435a5041b5b36e2e17f.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/2627379ab56667ad70a4b55773425faf7ac2cbd30498e435a5041b5b36e2e17f.jpg)

![2b9e467e5abf4be62a9584983bf372e6def02841e4054c936db3e553881dfb82.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/2b9e467e5abf4be62a9584983bf372e6def02841e4054c936db3e553881dfb82.jpg)

![317118501edd93719bea7f70bbf16a7f5b9636f8ce019c7213239a81a4876ae6.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/317118501edd93719bea7f70bbf16a7f5b9636f8ce019c7213239a81a4876ae6.jpg)

![99799cd405fcce6c204d9e15fa9a7d7ba47606e3d44764b03d2cc83d78fc579e.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/99799cd405fcce6c204d9e15fa9a7d7ba47606e3d44764b03d2cc83d78fc579e.jpg)

![a282ae2635bb46f5e8e4b409c048ab99934868a1dc20bb557e30fd879cf8b0bb.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/a282ae2635bb46f5e8e4b409c048ab99934868a1dc20bb557e30fd879cf8b0bb.jpg)

![ad6d2b3e63345ccce7f374bdb894f802cef11953abe0ded86ffb966e8760d731.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/ad6d2b3e63345ccce7f374bdb894f802cef11953abe0ded86ffb966e8760d731.jpg)

![cbf9ff4010b7934d5af1e9c45230cb607cf83134a9eff381b3fb7f5fa1e4779c.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/cbf9ff4010b7934d5af1e9c45230cb607cf83134a9eff381b3fb7f5fa1e4779c.jpg)

![cc193dbbb92d89cf50af785f1513b2779c4360f35d92cd927172da0db9d15dcb.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/cc193dbbb92d89cf50af785f1513b2779c4360f35d92cd927172da0db9d15dcb.jpg)

![cc43015d22c79d8c0ae78d1327487751a10a3c8eb37590c1966f27d631029bbc.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/cc43015d22c79d8c0ae78d1327487751a10a3c8eb37590c1966f27d631029bbc.jpg)

![d0579c3eefbb4a765a8b2f72ea984bdcefef672a63d84aa1514b24cfea792648.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/d0579c3eefbb4a765a8b2f72ea984bdcefef672a63d84aa1514b24cfea792648.jpg)

![d1e8f65d1b04b1e5033ef2b30330dbaecf9a51a8c869a7d62152edc27c41abe9.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/d1e8f65d1b04b1e5033ef2b30330dbaecf9a51a8c869a7d62152edc27c41abe9.jpg)

![d8683be45b133ebd94a82d5f446df9dc12f4e191913eb360a5d8df8eb7b46f0e.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/d8683be45b133ebd94a82d5f446df9dc12f4e191913eb360a5d8df8eb7b46f0e.jpg)

![dcfa3e9e5ad59ad07b50355e4e455de0b4c7e26582d17d6b95b0079fe0d221b0.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/dcfa3e9e5ad59ad07b50355e4e455de0b4c7e26582d17d6b95b0079fe0d221b0.jpg)

![f3b052d5c387c27dcf2c2a02fdad5dca93778463aac2ce46b3634729af0667b1.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/f3b052d5c387c27dcf2c2a02fdad5dca93778463aac2ce46b3634729af0667b1.jpg)

![ff0479b5862dd343282ea2111358c49aed14ca00565e6b9d25fbb83e320b37c2.jpg](../icml_results/4691_Hidden%20in%20Plain%20Sight%20%E2%80%93%20Class%20Competition%20Focuses%20Attribution%20Maps/tables/ff0479b5862dd343282ea2111358c49aed14ca00565e6b9d25fbb83e320b37c2.jpg)

<a id="paper-4644"></a>

## Mitigating Conversational Inertia in Multi-Turn Agents


### Images

![0d6684cb6bcf777083dd450c70c397ce9f7a8b8f14a7775dae902aa3df54d730.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/0d6684cb6bcf777083dd450c70c397ce9f7a8b8f14a7775dae902aa3df54d730.jpg)

![19dbb784d38f3c17e3dd414f114319e09b22cae21744fad5521339e5a88e2149.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/19dbb784d38f3c17e3dd414f114319e09b22cae21744fad5521339e5a88e2149.jpg)

![233127df669354c6285bf3a3f1cfc3167a11154296e63d4940cc92e8b9f8bf35.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/233127df669354c6285bf3a3f1cfc3167a11154296e63d4940cc92e8b9f8bf35.jpg)

![31d2fadeeb5d805d1fdaea08c6a6146fa9ec3aaf3e7b44adf1cda6360f2d953c.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/31d2fadeeb5d805d1fdaea08c6a6146fa9ec3aaf3e7b44adf1cda6360f2d953c.jpg)

![3790f9e5792464a36cf17b0f1f976b5bcc557b6c593560414a4e33a8a355cbbb.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/3790f9e5792464a36cf17b0f1f976b5bcc557b6c593560414a4e33a8a355cbbb.jpg)

![4c052ad3a8f42576d1474c40edcc3f87b47b4261cb947791a4cc002a531a1b15.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/4c052ad3a8f42576d1474c40edcc3f87b47b4261cb947791a4cc002a531a1b15.jpg)

![5b2d7d66b047dc00bc8d5f668c43006677e57dee5ab3f40bb700b2301b995739.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/5b2d7d66b047dc00bc8d5f668c43006677e57dee5ab3f40bb700b2301b995739.jpg)

![5cc37c8f06022fcb812f72562458d2b7277f9b4db6e98167dc733daa6d84d078.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/5cc37c8f06022fcb812f72562458d2b7277f9b4db6e98167dc733daa6d84d078.jpg)

![60addd62a905e2680f442894a663b6efe8b3959131a3ed2732f544b4df12c872.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/60addd62a905e2680f442894a663b6efe8b3959131a3ed2732f544b4df12c872.jpg)

![62c3390a782a9209e015774fed92046e9148dd38fed0157026aca7746d2c086e.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/62c3390a782a9209e015774fed92046e9148dd38fed0157026aca7746d2c086e.jpg)

![7885c537f89c6c1068281eb4c10732cfe1a5c99db2e06b48fc01c18f4ad4618a.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/7885c537f89c6c1068281eb4c10732cfe1a5c99db2e06b48fc01c18f4ad4618a.jpg)

![82b1c47b0b1a7a925df9b6c7f6f2d7d9412235056050c87a9cf236da3fe518a6.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/82b1c47b0b1a7a925df9b6c7f6f2d7d9412235056050c87a9cf236da3fe518a6.jpg)

![90dc8af2e896c82cab04aaa9012050afa7483bea44a708f8b15e5bf7ba20930f.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/90dc8af2e896c82cab04aaa9012050afa7483bea44a708f8b15e5bf7ba20930f.jpg)

![91944f437ddbfcd2ac6e2563363cc8f38752c951773ad8964f76be3bf7f6fba1.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/91944f437ddbfcd2ac6e2563363cc8f38752c951773ad8964f76be3bf7f6fba1.jpg)

![a88e499f19bc892d469fd0458d3b1017d297f421937ee7186c7228f36fb24c58.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/a88e499f19bc892d469fd0458d3b1017d297f421937ee7186c7228f36fb24c58.jpg)

![bb17f9d9e06442a15ec598970c14e577c2039b807f5cd2ffd4491819d66420dd.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/bb17f9d9e06442a15ec598970c14e577c2039b807f5cd2ffd4491819d66420dd.jpg)

![cca1b525671db07157c51fd01f8020926a5c6240d45451212ac8e7ef8bbe13c9.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/cca1b525671db07157c51fd01f8020926a5c6240d45451212ac8e7ef8bbe13c9.jpg)

![ce781789377011b4cfaabfa306688bd78750afb57f3bd6cf72478e7c6098c208.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/ce781789377011b4cfaabfa306688bd78750afb57f3bd6cf72478e7c6098c208.jpg)

![d94d57b3fe5a3608aac8f99890947f2597493ef6da1ef40faece4e6fa773d384.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/d94d57b3fe5a3608aac8f99890947f2597493ef6da1ef40faece4e6fa773d384.jpg)

![f5f4febc011dc604991e1f61cf025db0a8fcd6c3dd8e70c6d64522541a61b57c.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/images/f5f4febc011dc604991e1f61cf025db0a8fcd6c3dd8e70c6d64522541a61b57c.jpg)

### Tables

![048edcf94ebbfe79ecd71a54690d3a55bf9481d1d063dc5b5f814772b973d71b.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/048edcf94ebbfe79ecd71a54690d3a55bf9481d1d063dc5b5f814772b973d71b.jpg)

![158673e926522000ded97795d3ca0207d3aa12db5f698473110746acf9f140ab.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/158673e926522000ded97795d3ca0207d3aa12db5f698473110746acf9f140ab.jpg)

![20d8312974c7e2cfed67fa42d4b7ba45593754a4f1dd72353d69dc0ee309c487.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/20d8312974c7e2cfed67fa42d4b7ba45593754a4f1dd72353d69dc0ee309c487.jpg)

![2fc2668e10ac50b2e201cf43724c9c22d5d63f7432d36725582663d224260bf0.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/2fc2668e10ac50b2e201cf43724c9c22d5d63f7432d36725582663d224260bf0.jpg)

![34f43d3ee5cc1188582783f5a7f929672bdece658dd89c787d161b13f09a9ea1.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/34f43d3ee5cc1188582783f5a7f929672bdece658dd89c787d161b13f09a9ea1.jpg)

![3f391e9dbd851dd252de518701139ac59b3763553a76ac853db7f19fe8144c0c.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/3f391e9dbd851dd252de518701139ac59b3763553a76ac853db7f19fe8144c0c.jpg)

![4866f0bc5abfdc8d17f009a80da4fa441e4bdbf0f2183f8ec14f3dc5a2117c7e.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/4866f0bc5abfdc8d17f009a80da4fa441e4bdbf0f2183f8ec14f3dc5a2117c7e.jpg)

![5a31cb67385407dc60617bcfd1ddc2e9815a9cd5eadb66b87681d10598f33560.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/5a31cb67385407dc60617bcfd1ddc2e9815a9cd5eadb66b87681d10598f33560.jpg)

![6973347b4ab77ffea5909e5ee1ba84e58991e964cf18e1019b70df072694852d.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/6973347b4ab77ffea5909e5ee1ba84e58991e964cf18e1019b70df072694852d.jpg)

![6f3bbd167150a90a99160dcc6d3a64d7cdb21a318a4309f0d1c74f55ae523922.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/6f3bbd167150a90a99160dcc6d3a64d7cdb21a318a4309f0d1c74f55ae523922.jpg)

![7cbcf583860a1953264fcd0e39ed442e08c2e844614ff838c5aac1144959e9bf.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/7cbcf583860a1953264fcd0e39ed442e08c2e844614ff838c5aac1144959e9bf.jpg)

![a284527130be5732286291bcbd847deeb73fbc2065dec97f9642b4e12e53003e.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/a284527130be5732286291bcbd847deeb73fbc2065dec97f9642b4e12e53003e.jpg)

![b1e8bdc80ae2bda2918209319285314fc9f11623a0e7d385153a7dc9a2e2211f.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/b1e8bdc80ae2bda2918209319285314fc9f11623a0e7d385153a7dc9a2e2211f.jpg)

![bf210f0a3262c0dace6e470ae266788b80d57e7626be85e49957da3f5f2d0768.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/bf210f0a3262c0dace6e470ae266788b80d57e7626be85e49957da3f5f2d0768.jpg)

![e5ab9a28582b7f796a0071a41434f2db2c9c667b183fb14954cf10ef0d795fb4.jpg](../icml_results/4692_Mitigating%20Conversational%20Inertia%20in%20Multi-Turn%20Agents/tables/e5ab9a28582b7f796a0071a41434f2db2c9c667b183fb14954cf10ef0d795fb4.jpg)

<a id="paper-4645"></a>

## Saving Foundation Flow-Matching Priors for Inverse Problems


### Images

![13fecbbeeea0663cbe0aedd0faaadf0423a7cc8eb9681f3c80bb048e2b64887c.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/13fecbbeeea0663cbe0aedd0faaadf0423a7cc8eb9681f3c80bb048e2b64887c.jpg)

![5ce6e652e5f48580c9ccf69165a32101468ecd0e37dd365491a1b1b3ee8706d7.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/5ce6e652e5f48580c9ccf69165a32101468ecd0e37dd365491a1b1b3ee8706d7.jpg)

![694a08f6727f223ef2c24f669a7c7d2a731c1c465fe8e7c67864a2011cb05ab6.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/694a08f6727f223ef2c24f669a7c7d2a731c1c465fe8e7c67864a2011cb05ab6.jpg)

![7b58967bf418dc3889215faca14ab2a0669fb83da48b7622fb99710b974f309a.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/7b58967bf418dc3889215faca14ab2a0669fb83da48b7622fb99710b974f309a.jpg)

![918d3b266631b29a817c3ebf294e10d6875fad563db5a18ea474e1cc9fa9b9b5.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/918d3b266631b29a817c3ebf294e10d6875fad563db5a18ea474e1cc9fa9b9b5.jpg)

![9a42a93f3e2d3e7271d79a02230a39967aa7ae69324b68d533aad1f40e548ed9.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/9a42a93f3e2d3e7271d79a02230a39967aa7ae69324b68d533aad1f40e548ed9.jpg)

![a51deea4def7c98e4860278c8b558416618702c995402f80535792743a2331c0.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/a51deea4def7c98e4860278c8b558416618702c995402f80535792743a2331c0.jpg)

![b36aff4e99d58388b11e4d7ea0a8f3bc39c0a8b19e36b6fb49a4abf53d06ef93.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/b36aff4e99d58388b11e4d7ea0a8f3bc39c0a8b19e36b6fb49a4abf53d06ef93.jpg)

![b4bb7cbb585b90a9ad076ad4aeba16bac3e99583062d36aa8d27321295727db9.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/b4bb7cbb585b90a9ad076ad4aeba16bac3e99583062d36aa8d27321295727db9.jpg)

![dbf1ac4faa729c6ae70869c2ebf810cbbad93d34cea6c572d26ec0d01a636f4b.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/dbf1ac4faa729c6ae70869c2ebf810cbbad93d34cea6c572d26ec0d01a636f4b.jpg)

![e6b0d57e92fcefe2aef36919079c859a0b9e091ab5eadebb7c13433f3448333c.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/e6b0d57e92fcefe2aef36919079c859a0b9e091ab5eadebb7c13433f3448333c.jpg)

![f9f51856404bb59ab4a7bcb3ab53c732a5bb9a877d331fe875df761674e6401b.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/images/f9f51856404bb59ab4a7bcb3ab53c732a5bb9a877d331fe875df761674e6401b.jpg)

### Tables

![1242caf28796442b9980923c6d2c374e1029320799cec4a7708f1a01bb5560fc.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/1242caf28796442b9980923c6d2c374e1029320799cec4a7708f1a01bb5560fc.jpg)

![3a9a85e341b5bc9c94c4064f98034f3ddf3d9cbaaf006d69562d7b0cb0e3dd0d.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/3a9a85e341b5bc9c94c4064f98034f3ddf3d9cbaaf006d69562d7b0cb0e3dd0d.jpg)

![57aa26140893fe7db194081a84dc5b7fe110867d694f3ff750d412b8fa4480a3.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/57aa26140893fe7db194081a84dc5b7fe110867d694f3ff750d412b8fa4480a3.jpg)

![6a3f9feb5e38d0559a7895fea67390da6924378cfc2c71b4b9b5f929577c91dc.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/6a3f9feb5e38d0559a7895fea67390da6924378cfc2c71b4b9b5f929577c91dc.jpg)

![7e9904f781f09c5cfe4d8d217564241c9ea311a5779d5fb29e07e09683d17915.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/7e9904f781f09c5cfe4d8d217564241c9ea311a5779d5fb29e07e09683d17915.jpg)

![82c0fbb6660ca685ea0f8dc5ab5a5f7475bb86f6be4d246ce443e1107eefb3a0.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/82c0fbb6660ca685ea0f8dc5ab5a5f7475bb86f6be4d246ce443e1107eefb3a0.jpg)

![8ac9a0b3b37874e2c4728006bf352224b2e8624823397c47dc18a72748c7cf2e.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/8ac9a0b3b37874e2c4728006bf352224b2e8624823397c47dc18a72748c7cf2e.jpg)

![9798272291ec25c10ca0f1766e73db2a1b968034fba2dab14894309ff339fc32.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/9798272291ec25c10ca0f1766e73db2a1b968034fba2dab14894309ff339fc32.jpg)

![9a1032e59ffef21fb33edaa0b7130a95def7b2c3166401e0d4c4817e68a95abd.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/9a1032e59ffef21fb33edaa0b7130a95def7b2c3166401e0d4c4817e68a95abd.jpg)

![9dedb5bd8a0ee820dbeec873214dc097f2edc6e9e640c7dbb952717ea6f42801.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/9dedb5bd8a0ee820dbeec873214dc097f2edc6e9e640c7dbb952717ea6f42801.jpg)

![9fda83925d3ef019c7edcbbdabcd5000d2b5da3eb53f511985ca52b377e3b1f4.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/9fda83925d3ef019c7edcbbdabcd5000d2b5da3eb53f511985ca52b377e3b1f4.jpg)

![b2f18e68109885348fd758a5616799290c3874e45e9346c570b95bc24934e6f0.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/b2f18e68109885348fd758a5616799290c3874e45e9346c570b95bc24934e6f0.jpg)

![c2c8a0d4f91f60fd833d6a43972a59623e9cd4a3d0d8efc53419dc2346f1264d.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/c2c8a0d4f91f60fd833d6a43972a59623e9cd4a3d0d8efc53419dc2346f1264d.jpg)

![c99d6db21e86a8c0b25bac03807c7c857a46ce1bb47ae02e273ad0c87f213358.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/c99d6db21e86a8c0b25bac03807c7c857a46ce1bb47ae02e273ad0c87f213358.jpg)

![ca5694e6d44f1fa285e429c2e557b6c9c2bd5cb399a64fdd200e05267b61e648.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/ca5694e6d44f1fa285e429c2e557b6c9c2bd5cb399a64fdd200e05267b61e648.jpg)

![d8f5c38b708b3ac1658e85bdb3c50d0e14a7b2ccc3cc0c1059909b3e9c3619f7.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/d8f5c38b708b3ac1658e85bdb3c50d0e14a7b2ccc3cc0c1059909b3e9c3619f7.jpg)

![dd210f354cd2056985d5707812ebd0efcf5e7fc5b7ed8c2d3f12dab7c94adcb0.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/dd210f354cd2056985d5707812ebd0efcf5e7fc5b7ed8c2d3f12dab7c94adcb0.jpg)

![f22136698dc1e1d42319f3f63abcb28ec106fb6b685614b87385045bb7f89558.jpg](../icml_results/4693_Saving%20Foundation%20Flow-Matching%20Priors%20for%20Inverse%20Problems/tables/f22136698dc1e1d42319f3f63abcb28ec106fb6b685614b87385045bb7f89558.jpg)

<a id="paper-4646"></a>

## Gradient Inversion Attacks Beyond SGD


### Images

![081bb6665c381be60fbb00dceed52b9135482c16ad4394fdb3b4a9d53936952e.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/081bb6665c381be60fbb00dceed52b9135482c16ad4394fdb3b4a9d53936952e.jpg)

![1ba4ce6191dcc6ee2430142c26d47e9d26d0cac1a83eb5d641f768c8f0b6717a.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/1ba4ce6191dcc6ee2430142c26d47e9d26d0cac1a83eb5d641f768c8f0b6717a.jpg)

![32b0192a75536401b129decfc1951469c23c75f4642e4c741cf3073f73cfbc54.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/32b0192a75536401b129decfc1951469c23c75f4642e4c741cf3073f73cfbc54.jpg)

![679d25ae61e0648d76c2890875b5d29bb5432210428947a5349145f500d7fbc2.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/679d25ae61e0648d76c2890875b5d29bb5432210428947a5349145f500d7fbc2.jpg)

![88b8ab2979725d5e1bbf442093477281f7143ce1e1433c12f78fe857c74497c6.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/88b8ab2979725d5e1bbf442093477281f7143ce1e1433c12f78fe857c74497c6.jpg)

![9fb03e4ec1ab454005e3cac4d6b741c876889829fcbdab80fa1f413655c762c5.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/9fb03e4ec1ab454005e3cac4d6b741c876889829fcbdab80fa1f413655c762c5.jpg)

![a366790695ea828a28a54d3cd45daee2bed24794ed6af52b27b79b36bc80f316.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/a366790695ea828a28a54d3cd45daee2bed24794ed6af52b27b79b36bc80f316.jpg)

![bc074fa5601bd52b7260b4c98f9bcd405c5c438da4889990a3357238084b453f.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/bc074fa5601bd52b7260b4c98f9bcd405c5c438da4889990a3357238084b453f.jpg)

![c01215aa57977aae22295b7e4ffebfdf6bbc2a8477796cc79b8bb336b94105e4.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/c01215aa57977aae22295b7e4ffebfdf6bbc2a8477796cc79b8bb336b94105e4.jpg)

![d120f153e9ae3d0c72d68897e20ad932c50da3ad6ba9b76f546a2d4a261d82fd.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/d120f153e9ae3d0c72d68897e20ad932c50da3ad6ba9b76f546a2d4a261d82fd.jpg)

![fc8f9fed79f7b1fd8898ec9d1a1dde57e1b0be31ba605be95d0078a15d4575a9.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/images/fc8f9fed79f7b1fd8898ec9d1a1dde57e1b0be31ba605be95d0078a15d4575a9.jpg)

### Tables

![253101907037691857bf6765352d75a4fc07886b9d2110a42a1dfe34935c1d0b.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/253101907037691857bf6765352d75a4fc07886b9d2110a42a1dfe34935c1d0b.jpg)

![3ebd75526b8264daea365c6e1d9db452cd54b7420729b3d2cd047e48b5cfaff7.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/3ebd75526b8264daea365c6e1d9db452cd54b7420729b3d2cd047e48b5cfaff7.jpg)

![44bef8a62e01e00801017175cade71ef88375e11e6b66d4098131cfd7854b14b.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/44bef8a62e01e00801017175cade71ef88375e11e6b66d4098131cfd7854b14b.jpg)

![7c44abaf186809310773c70d5f3e7552279bed080621ed17596d3e5d71290d16.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/7c44abaf186809310773c70d5f3e7552279bed080621ed17596d3e5d71290d16.jpg)

![88ad1ce6785fabfca7b62ea54edf94827b6574746015899336142c311be3601e.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/88ad1ce6785fabfca7b62ea54edf94827b6574746015899336142c311be3601e.jpg)

![a4460bfeefefeb284d645a094a909e52f6397245a5326e734f5d9b6532cc976e.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/a4460bfeefefeb284d645a094a909e52f6397245a5326e734f5d9b6532cc976e.jpg)

![ab16996d113df366f3cf1e4c6f0981f042122182cf8c9fc8c34fa89361fc6dc8.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/ab16996d113df366f3cf1e4c6f0981f042122182cf8c9fc8c34fa89361fc6dc8.jpg)

![c3d8622011d9cd46a031224e469d5028a0dd1710c1f38f14b854689b086cc028.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/c3d8622011d9cd46a031224e469d5028a0dd1710c1f38f14b854689b086cc028.jpg)

![d73a52108647f3ec947851c2f1f9cb73eb562ecfe591ea91604bc6ce7ee1ad77.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/d73a52108647f3ec947851c2f1f9cb73eb562ecfe591ea91604bc6ce7ee1ad77.jpg)

![e1b4fbc972e4a8feabb016e6620669a0fa53ae0e47982c59f2cf16f10c24b62c.jpg](../icml_results/4694_Gradient%20Inversion%20Attacks%20Beyond%20SGD/tables/e1b4fbc972e4a8feabb016e6620669a0fa53ae0e47982c59f2cf16f10c24b62c.jpg)

<a id="paper-4647"></a>

## RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference


### Images

![329e67eae828d8e245705288bcce7daec782ed616cd8ba4095aa24693a8f4117.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/329e67eae828d8e245705288bcce7daec782ed616cd8ba4095aa24693a8f4117.jpg)

![34b9c1fbb81a764a0d5d2329f28cf4a32d88c247a2826ded4baa10d8f692f293.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/34b9c1fbb81a764a0d5d2329f28cf4a32d88c247a2826ded4baa10d8f692f293.jpg)

![36649b1ddf96987961286feb908f95939eae4c99a9b96bfb1793324701878d52.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/36649b1ddf96987961286feb908f95939eae4c99a9b96bfb1793324701878d52.jpg)

![3800436219b966bb8310c2f6de940ef98201fe4215a21a809e66776382017ee0.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/3800436219b966bb8310c2f6de940ef98201fe4215a21a809e66776382017ee0.jpg)

![9cf04e848a0f7ec497baba79d26983562ab6b4bd6a5e6652bd93dfb3a368d972.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/9cf04e848a0f7ec497baba79d26983562ab6b4bd6a5e6652bd93dfb3a368d972.jpg)

![ae43591bbcc1bf683986da5f47583cd4dc7ef43e9cc12e0b8a4290d22988f620.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/ae43591bbcc1bf683986da5f47583cd4dc7ef43e9cc12e0b8a4290d22988f620.jpg)

![b13356ae5b2e1643bc60ecb48329b8cbc110d7dfaae96fbd29722c0be47b1ec7.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/images/b13356ae5b2e1643bc60ecb48329b8cbc110d7dfaae96fbd29722c0be47b1ec7.jpg)

### Tables

![01b5a0ec54c6ef0a9a8f5a6047e1e6ae4c7e5723f75af65a1e50757d0f39dfec.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/01b5a0ec54c6ef0a9a8f5a6047e1e6ae4c7e5723f75af65a1e50757d0f39dfec.jpg)

![0b43b5c18eb3a9e0c46a9cc2b8444e3bdf99fbf0f5e0952e678cd12725be6bcc.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/0b43b5c18eb3a9e0c46a9cc2b8444e3bdf99fbf0f5e0952e678cd12725be6bcc.jpg)

![1ef3d2f70a8b49215978e6e44fa4d4b016aa424cbc3bd2483beda8c44dff3964.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/1ef3d2f70a8b49215978e6e44fa4d4b016aa424cbc3bd2483beda8c44dff3964.jpg)

![37f596d5bebb2603f6c36b3dca8d51d3a57bf48738cef63fb03d53d65886647c.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/37f596d5bebb2603f6c36b3dca8d51d3a57bf48738cef63fb03d53d65886647c.jpg)

![694cee0ea31d9acabf59b772c12ba6ef5c2a3573eede69b703ea834ee8605e56.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/694cee0ea31d9acabf59b772c12ba6ef5c2a3573eede69b703ea834ee8605e56.jpg)

![844a93a796563aa7aa188d4e4c0f1c8acd3c9a8a7bda44d09f3a407149d79596.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/844a93a796563aa7aa188d4e4c0f1c8acd3c9a8a7bda44d09f3a407149d79596.jpg)

![952fd2f590ddd512c7f0f7d7b221c5e23f31b8c8bdcf9e91060f9be428d7a030.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/952fd2f590ddd512c7f0f7d7b221c5e23f31b8c8bdcf9e91060f9be428d7a030.jpg)

![9c7f9120de6b5aac8c5bb97971e408c3ced4791de0c1619a758d29de3b168afa.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/9c7f9120de6b5aac8c5bb97971e408c3ced4791de0c1619a758d29de3b168afa.jpg)

![c8c2b2e00b80866d19d8aa73280356a6165ba5fd5ee28665ac8bd9fdfcb54380.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/c8c2b2e00b80866d19d8aa73280356a6165ba5fd5ee28665ac8bd9fdfcb54380.jpg)

![d63ceb2a8826fcf76a75baa65e0910c874e068b8d7b45d8b9ee4ae4d733af56f.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/d63ceb2a8826fcf76a75baa65e0910c874e068b8d7b45d8b9ee4ae4d733af56f.jpg)

![e1064c64fe8557bf10cd00ae42d34238f9f20ef78d04a5733f1a9fb40b3e723a.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/e1064c64fe8557bf10cd00ae42d34238f9f20ef78d04a5733f1a9fb40b3e723a.jpg)

![e91c6507303750cf00e8a92af5260665a25ba388a37e25e7d59cf02cc80db73b.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/e91c6507303750cf00e8a92af5260665a25ba388a37e25e7d59cf02cc80db73b.jpg)

![ecc481fcf91e9a6d4980b96119484e8daa9ac8b00b64e1eb53d17e40f7796f5f.jpg](../icml_results/4695_RTPrune_%20Reading-Twice%20Inspired%20Token%20Pruning%20for%20Efficient%20DeepSeek-OCR%20Inference/tables/ecc481fcf91e9a6d4980b96119484e8daa9ac8b00b64e1eb53d17e40f7796f5f.jpg)

<a id="paper-4648"></a>

## Beyond Policy Training: Recursive Solution Search from Unannotated Videos


### Images

![24ef4f1098f377a655cf8779ec4701606e259f1752bdfb60cec27b54a9cd47b4.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/24ef4f1098f377a655cf8779ec4701606e259f1752bdfb60cec27b54a9cd47b4.jpg)

![3213eff1ce483b4b9ff1f33c17ae14d601f028a1141ea2eda8b008925d9e0e9e.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/3213eff1ce483b4b9ff1f33c17ae14d601f028a1141ea2eda8b008925d9e0e9e.jpg)

![33e9d0f4368b77495c8b7be44706c8b08ac82f4523da09c4b3d5b5a1ac33eddf.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/33e9d0f4368b77495c8b7be44706c8b08ac82f4523da09c4b3d5b5a1ac33eddf.jpg)

![5bfec68c6b815bfc3b2afec87862730fe0c74caf44d97878389ccf05b1dd4037.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/5bfec68c6b815bfc3b2afec87862730fe0c74caf44d97878389ccf05b1dd4037.jpg)

![73b019dd0976a74fbcfa85f374d580f07a2fe1ec2a1c21a14363c8cbff62bece.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/73b019dd0976a74fbcfa85f374d580f07a2fe1ec2a1c21a14363c8cbff62bece.jpg)

![79f9289691b3946689bd0d71a355ad879a15cabd171eb1983f7886cebdbfa6b1.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/79f9289691b3946689bd0d71a355ad879a15cabd171eb1983f7886cebdbfa6b1.jpg)

![d1470f66b1ab8440be6ca4baab4c1a578100d6734f627a316e2f7eb766fbd252.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/d1470f66b1ab8440be6ca4baab4c1a578100d6734f627a316e2f7eb766fbd252.jpg)

![f0998ea5cd25107132e8ef263d0aa09d8ac3ee2bca5eb78dd526c46c8aa9e874.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/f0998ea5cd25107132e8ef263d0aa09d8ac3ee2bca5eb78dd526c46c8aa9e874.jpg)

![f25bcee180cce87ed3b8b3fe4c5baf51505124dc6b0dfef69c54a1a0fc2a4114.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/f25bcee180cce87ed3b8b3fe4c5baf51505124dc6b0dfef69c54a1a0fc2a4114.jpg)

![f9092a054d4148e79d94aaaf5853d9c352b856f18742787c9ccde814be1787c8.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/f9092a054d4148e79d94aaaf5853d9c352b856f18742787c9ccde814be1787c8.jpg)

![fc9d804e83b81ef2b3407b5962bd4b4a5718512c4d949f6aad2851b51e6ee2fb.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/images/fc9d804e83b81ef2b3407b5962bd4b4a5718512c4d949f6aad2851b51e6ee2fb.jpg)

### Tables

![0f0f18c7f177bdd25766e51ee7a93cb9fdb6489dac52e1f87562df1c64e1bcea.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/0f0f18c7f177bdd25766e51ee7a93cb9fdb6489dac52e1f87562df1c64e1bcea.jpg)

![1cf1d86dd8fc208e2a802ff44712a06ca00a03aa17210319255b090c177b0119.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/1cf1d86dd8fc208e2a802ff44712a06ca00a03aa17210319255b090c177b0119.jpg)

![39a03fd11a1de64fc46658c8312e27faef8814cff958af9a2f86dc8646c00675.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/39a03fd11a1de64fc46658c8312e27faef8814cff958af9a2f86dc8646c00675.jpg)

![55dde676dcb115f045c777da3e4ff79ea32a54c0c487150d3585f68f17148a94.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/55dde676dcb115f045c777da3e4ff79ea32a54c0c487150d3585f68f17148a94.jpg)

![6c996635ee39c57ecfbe3909106a1604eb9208e9edde3a11bb830af930c846b7.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/6c996635ee39c57ecfbe3909106a1604eb9208e9edde3a11bb830af930c846b7.jpg)

![a539bfe4f734319696b30ef82f49aa5e4d8f6d4b907b16237abecaaa597815f5.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/a539bfe4f734319696b30ef82f49aa5e4d8f6d4b907b16237abecaaa597815f5.jpg)

![ec57f884b152677162694a7b13cca00fc08c89e2ac4b943389e6bdd2df8223d5.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/ec57f884b152677162694a7b13cca00fc08c89e2ac4b943389e6bdd2df8223d5.jpg)

![fff44255a6ea22b4c61c6d5f2e47ab54d7cc1d523bede40879af49c8699580c6.jpg](../icml_results/4696_Beyond%20Policy%20Training_%20Recursive%20Solution%20Search%20from%20Unannotated%20Videos/tables/fff44255a6ea22b4c61c6d5f2e47ab54d7cc1d523bede40879af49c8699580c6.jpg)

<a id="paper-4649"></a>

## Multi-view Consistent Latent Action Learning for World Modeling and Control


### Images

![016576ad68f2c9a8a4f492a4bacf25c209d815fe6073b4731d5b035a657456ef.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/016576ad68f2c9a8a4f492a4bacf25c209d815fe6073b4731d5b035a657456ef.jpg)

![0ecbe79612020ac1ad1aef8682e89f09cd515452c00842020ef68784563bdf6e.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/0ecbe79612020ac1ad1aef8682e89f09cd515452c00842020ef68784563bdf6e.jpg)

![20f5d93bda497f8cbb1c19c0cf9e49cb22584bb722ad4251660e991236a9a314.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/20f5d93bda497f8cbb1c19c0cf9e49cb22584bb722ad4251660e991236a9a314.jpg)

![2e2295ebdf5683055ae5367b21eb86254cdf63eb90039cc8381963ffc314d56c.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/2e2295ebdf5683055ae5367b21eb86254cdf63eb90039cc8381963ffc314d56c.jpg)

![3460368c6aea4e601ef2f9bd06812705919a0a1b41caea188d7f14308ad056ee.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/3460368c6aea4e601ef2f9bd06812705919a0a1b41caea188d7f14308ad056ee.jpg)

![3a8f2e244cad262d442ef058835c5359239648a14f3d1d5d087ab1d747781eb2.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/3a8f2e244cad262d442ef058835c5359239648a14f3d1d5d087ab1d747781eb2.jpg)

![4a537b10d7e342c987572c3fa80257637540a1053952ecce219bbbb269cbfa71.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/4a537b10d7e342c987572c3fa80257637540a1053952ecce219bbbb269cbfa71.jpg)

![4e0593a5dc4a88f6d247ce442d7b1cc2d79fed586bdad6c530a742c241e28215.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/4e0593a5dc4a88f6d247ce442d7b1cc2d79fed586bdad6c530a742c241e28215.jpg)

![5858db7f3c67ff8fd3bb3ece9213197ba372d6a7f87208d3fc208424c087b6f5.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/5858db7f3c67ff8fd3bb3ece9213197ba372d6a7f87208d3fc208424c087b6f5.jpg)

![793dd9febe6e8c37649dae67e65327b93d90e7602a4f1424f75350b4d4177f9c.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/793dd9febe6e8c37649dae67e65327b93d90e7602a4f1424f75350b4d4177f9c.jpg)

![7a348da712c5342c013a357df009e8e4da707ae62857d1a8b8f4ecc5ec825d19.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/7a348da712c5342c013a357df009e8e4da707ae62857d1a8b8f4ecc5ec825d19.jpg)

![89f042b286484620b8ead9e4130085730332ed4dbf136addf0a790923b435fe2.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/89f042b286484620b8ead9e4130085730332ed4dbf136addf0a790923b435fe2.jpg)

![b2fdaa62722796a026571e6323fd5569640a3c72e0452af7f3ff1f9393e4b68e.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/b2fdaa62722796a026571e6323fd5569640a3c72e0452af7f3ff1f9393e4b68e.jpg)

![d441edfce14e62a7f6683b45a4d9e3a618c608ea251a7743dd7ca3c1ff17ef34.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/d441edfce14e62a7f6683b45a4d9e3a618c608ea251a7743dd7ca3c1ff17ef34.jpg)

![d93a786bd684d9ac4260bae1b5200ec1c41d9d52dbbc0e4f975648cc443c35c0.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/images/d93a786bd684d9ac4260bae1b5200ec1c41d9d52dbbc0e4f975648cc443c35c0.jpg)

### Tables

![1dce1954d64541f54dcf787a9f76af2c5d6839f54e274dab39b893aac84bb4f2.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/1dce1954d64541f54dcf787a9f76af2c5d6839f54e274dab39b893aac84bb4f2.jpg)

![52a5281c5a81597889a894ca1d6fd8f49e1bfd5e0cb70640b4ce0c784b3eac52.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/52a5281c5a81597889a894ca1d6fd8f49e1bfd5e0cb70640b4ce0c784b3eac52.jpg)

![75ba2cffc51aab0ff085c8621d54a2dcd86a0f2d8407f26d21c2530324b92243.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/75ba2cffc51aab0ff085c8621d54a2dcd86a0f2d8407f26d21c2530324b92243.jpg)

![7ca58cd46ff84a3a1d1caf01217a35966e63d074fb601fcdcd8e3014e228392c.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/7ca58cd46ff84a3a1d1caf01217a35966e63d074fb601fcdcd8e3014e228392c.jpg)

![8b703bb30865e4f27de313dc04cf13fefb639a021f2a405d5d9a907be85923ca.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/8b703bb30865e4f27de313dc04cf13fefb639a021f2a405d5d9a907be85923ca.jpg)

![d18972b6be87eb3f83a44900c682eea7e3cd8e756185fda7edd601900fe69a9a.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/d18972b6be87eb3f83a44900c682eea7e3cd8e756185fda7edd601900fe69a9a.jpg)

![f5f84f8b4896cae195ff54a500530c2ebf44783658b6873932197270f968ff47.jpg](../icml_results/4697_Multi-view%20Consistent%20Latent%20Action%20Learning%20for%20World%20Modeling%20and%20Control/tables/f5f84f8b4896cae195ff54a500530c2ebf44783658b6873932197270f968ff47.jpg)

<a id="paper-4650"></a>

## Video-SVD: Efficient Video Diffusion via Orthogonal Basis Composition


### Images

![07aa3c77e34e1140592524b98effa0c4993f3764129e5d0378bad500eb976279.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/07aa3c77e34e1140592524b98effa0c4993f3764129e5d0378bad500eb976279.jpg)

![0cb2c238744f64039cf924b71350e43ff02d3394daa7100a98dd4cf79de49a37.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/0cb2c238744f64039cf924b71350e43ff02d3394daa7100a98dd4cf79de49a37.jpg)

![25d74244f857509e197c36cf1141b05856d79b7091abbe6fd160116e0f5f0fc8.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/25d74244f857509e197c36cf1141b05856d79b7091abbe6fd160116e0f5f0fc8.jpg)

![390d4a22404722bfc28299c99338071ecb4bc57577c9f4389b4d4922481e6962.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/390d4a22404722bfc28299c99338071ecb4bc57577c9f4389b4d4922481e6962.jpg)

![4710011eb6cd86d61f65cebb65b4534f1d41c22a7923d5ac64e2ef5aa0876ea2.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/4710011eb6cd86d61f65cebb65b4534f1d41c22a7923d5ac64e2ef5aa0876ea2.jpg)

![5416e7657c8901f5dbd8078f4d534278e85f733c02471dadba55419dc0ef9a91.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/5416e7657c8901f5dbd8078f4d534278e85f733c02471dadba55419dc0ef9a91.jpg)

![8cdc87d144e67607a9de146c2991d94a903c431e9f5dfa74f1d321e462b733cc.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/8cdc87d144e67607a9de146c2991d94a903c431e9f5dfa74f1d321e462b733cc.jpg)

![8eacaeeedfcd5500de1e0d433f13bc96e6d52e7dd04f2f7be977bec7534df54d.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/8eacaeeedfcd5500de1e0d433f13bc96e6d52e7dd04f2f7be977bec7534df54d.jpg)

![9458e5b5229aa05f5e01da14268acd86e3d3ae9d99d659a9014c5a6d5b877f39.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/9458e5b5229aa05f5e01da14268acd86e3d3ae9d99d659a9014c5a6d5b877f39.jpg)

![a556cdc38ccba60d672312932ca5b1ff70a9206b27131cb6b5ee39ea658f8eed.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/a556cdc38ccba60d672312932ca5b1ff70a9206b27131cb6b5ee39ea658f8eed.jpg)

![c9b24a9eb1ed3ee5f71b64da291eab02a0d1e35782a15b2f7fb9ef8e6a5a7c69.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/c9b24a9eb1ed3ee5f71b64da291eab02a0d1e35782a15b2f7fb9ef8e6a5a7c69.jpg)

![d88e0fbe2b06374b76276098e6741025bd70cc7281bf476589600dba3ac9b5da.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/d88e0fbe2b06374b76276098e6741025bd70cc7281bf476589600dba3ac9b5da.jpg)

![fc5cf87f1d3a568d405ed45e0a6fa8456d3d40eaa34571914dd4732897c1c980.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/images/fc5cf87f1d3a568d405ed45e0a6fa8456d3d40eaa34571914dd4732897c1c980.jpg)

### Tables

![23625cf6ad311808fcd4b1de1e37ace67ac51b90f684dfabfa331c2a1d804ef4.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/23625cf6ad311808fcd4b1de1e37ace67ac51b90f684dfabfa331c2a1d804ef4.jpg)

![3b1d249fc595d8d19b28e8d84556faf3063f3c30098323f9b7be0fad1dd824b8.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/3b1d249fc595d8d19b28e8d84556faf3063f3c30098323f9b7be0fad1dd824b8.jpg)

![4f5421dd5ed0e9c9d64800529d6084922ed3bdbbe72f1efe883dc7d8de202254.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/4f5421dd5ed0e9c9d64800529d6084922ed3bdbbe72f1efe883dc7d8de202254.jpg)

![918f50a4e810ea13a8b8372687ab3546f5a24e003a916f2e1c02286e6004705f.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/918f50a4e810ea13a8b8372687ab3546f5a24e003a916f2e1c02286e6004705f.jpg)

![96a9426bf0a8223efbb1ccdf68abda3f128c26d911618746f2411885a0987ef1.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/96a9426bf0a8223efbb1ccdf68abda3f128c26d911618746f2411885a0987ef1.jpg)

![c13d87340cd5fc9c0bfe5b4f11ff92209acd3ad4cb5508f458fe1e441c2f8efb.jpg](../icml_results/4698_Video-SVD_%20Efficient%20Video%20Diffusion%20via%20Orthogonal%20Basis%20Composition/tables/c13d87340cd5fc9c0bfe5b4f11ff92209acd3ad4cb5508f458fe1e441c2f8efb.jpg)

<a id="paper-4651"></a>

## OpenMAG: A Comprehensive Benchmark for Multimodal-Attributed Graph


### Images

![017d2e15857851bd445cbd02353fea4561a1825caeb22ef63e26dbf1dd4a4060.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/017d2e15857851bd445cbd02353fea4561a1825caeb22ef63e26dbf1dd4a4060.jpg)

![44e6cb6c45806c2cbbb75bbf93355366533319e24c7cd088eee660edb7485228.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/44e6cb6c45806c2cbbb75bbf93355366533319e24c7cd088eee660edb7485228.jpg)

![4eae09645e9b92b409c47982c4bf643a31036b857e634d2ed19b94e3d56c6ce6.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/4eae09645e9b92b409c47982c4bf643a31036b857e634d2ed19b94e3d56c6ce6.jpg)

![8d1af8a4bb491bd00ec6fca290652dbdbd78f968b9e645a25193ccc2dc3a56f2.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/8d1af8a4bb491bd00ec6fca290652dbdbd78f968b9e645a25193ccc2dc3a56f2.jpg)

![d7c294feb993d62dfa338b814d3154659d303115e5b35c2626373e173daf99b1.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/d7c294feb993d62dfa338b814d3154659d303115e5b35c2626373e173daf99b1.jpg)

![de166cab87c8a57e1d5928dbf0377e608cb32e432df884b70c47dfef8e97dce5.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/de166cab87c8a57e1d5928dbf0377e608cb32e432df884b70c47dfef8e97dce5.jpg)

![e03168515f14cc9d3d5d835e00d1c891791bdd45c7d2702f31b92f07ab05f946.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/images/e03168515f14cc9d3d5d835e00d1c891791bdd45c7d2702f31b92f07ab05f946.jpg)

### Tables

![35fa78bb4406f16ea42f90d2113928568a7d5990afee0739671ae0735ae35f70.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/35fa78bb4406f16ea42f90d2113928568a7d5990afee0739671ae0735ae35f70.jpg)

![46820ef6efa10836712b61038af9e92bfaf3f0c12bad6c8824fbf2d50f62e9ec.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/46820ef6efa10836712b61038af9e92bfaf3f0c12bad6c8824fbf2d50f62e9ec.jpg)

![6dfca3efac3db352ddca51629704119fa7183cc09673b3f36e160314c3bc3b6d.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/6dfca3efac3db352ddca51629704119fa7183cc09673b3f36e160314c3bc3b6d.jpg)

![c5f296f4d894942b53bb5768c5e478ef80f58390ac867265cd0f7f625ca3d087.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/c5f296f4d894942b53bb5768c5e478ef80f58390ac867265cd0f7f625ca3d087.jpg)

![d6a0a3a705dd161d06b94ce867bcd403188523ea2f58010bfbe88ca46ed2bf29.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/d6a0a3a705dd161d06b94ce867bcd403188523ea2f58010bfbe88ca46ed2bf29.jpg)

![dbd4e619de5b378b0189ce04c68ecf4411c68a2c262757df89bb896598b45d2e.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/dbd4e619de5b378b0189ce04c68ecf4411c68a2c262757df89bb896598b45d2e.jpg)

![e2bb15b67bb4ae0bb09d9bbbb23c097953e6f1860322c2de7ca8836eaa37b1cd.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/e2bb15b67bb4ae0bb09d9bbbb23c097953e6f1860322c2de7ca8836eaa37b1cd.jpg)

![ec361ecb8b6f2b9d152a9fd2f101160169d958961279c461a9e44ca5ddd5dfac.jpg](../icml_results/4699_OpenMAG_%20A%20Comprehensive%20Benchmark%20for%20Multimodal-Attributed%20Graph/tables/ec361ecb8b6f2b9d152a9fd2f101160169d958961279c461a9e44ca5ddd5dfac.jpg)

<a id="paper-4652"></a>

## A Solver-Free Training Method for Predict-then-Optimize


### Images

![0a31308e0c4cef2ab8572ea672b1903ea6e49e758ac34a4327ddf49522b7873a.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/0a31308e0c4cef2ab8572ea672b1903ea6e49e758ac34a4327ddf49522b7873a.jpg)

![4f65a0c5811fa247a6f5b3a5ed27623876244c529d1e92202a852ce401f92b87.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/4f65a0c5811fa247a6f5b3a5ed27623876244c529d1e92202a852ce401f92b87.jpg)

![636db0a957b57d6d3561e32e07b5a51f61069d171a453aff96be936f95c9f16e.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/636db0a957b57d6d3561e32e07b5a51f61069d171a453aff96be936f95c9f16e.jpg)

![91786a317c0af93235227c832b52c8da109db4f95f566aad098c646a6052cd15.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/91786a317c0af93235227c832b52c8da109db4f95f566aad098c646a6052cd15.jpg)

![91df343eb9b1f0f2a21308e96cbfdee8e99b1dd934e0a6e67cc23743fb96d93e.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/91df343eb9b1f0f2a21308e96cbfdee8e99b1dd934e0a6e67cc23743fb96d93e.jpg)

![a9608f0722e267ab225308b1954efb57dedbeb8a0df62d7fc6ef4cc04ce74af9.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/a9608f0722e267ab225308b1954efb57dedbeb8a0df62d7fc6ef4cc04ce74af9.jpg)

![ec92e15e90aa65bf3c2ee233b4e7bb505b1cc4eccbd3c88923dddcb1b6b6c9e4.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/images/ec92e15e90aa65bf3c2ee233b4e7bb505b1cc4eccbd3c88923dddcb1b6b6c9e4.jpg)

### Tables

![12658de90f36492721971798d9e2be61aca968ee131979336f1da4abf409364a.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/tables/12658de90f36492721971798d9e2be61aca968ee131979336f1da4abf409364a.jpg)

![143dfabdea6ec0287bae89dc34ccc0d44156fa53dfc9f2dee9c86f7cbe07d7b2.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/tables/143dfabdea6ec0287bae89dc34ccc0d44156fa53dfc9f2dee9c86f7cbe07d7b2.jpg)

![23dfc1e7a852c94551f97334b158eb80bbe159d4ffe2acb6d5229f46c9491475.jpg](../icml_results/4700_A%20Solver-Free%20Training%20Method%20for%20Predict-then-Optimize/tables/23dfc1e7a852c94551f97334b158eb80bbe159d4ffe2acb6d5229f46c9491475.jpg)

<a id="paper-4653"></a>

## Corrected Samplers for Discrete Flow Models

### Images

![086ae64318e4993ca443f0acf411e00a453d0ebaf4e61b843199a3a7976fb27a.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/086ae64318e4993ca443f0acf411e00a453d0ebaf4e61b843199a3a7976fb27a.jpg)

![0f1045947ffc5fe1afc02dc4d1a3c923b85778aea00aa56e4bdc83469fe884a0.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/0f1045947ffc5fe1afc02dc4d1a3c923b85778aea00aa56e4bdc83469fe884a0.jpg)

![1d21f5bca881a5bbdaada0515a8275de338efa4f5ec455dd417718cfb5f09209.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/1d21f5bca881a5bbdaada0515a8275de338efa4f5ec455dd417718cfb5f09209.jpg)

![2ea05416ca177eaab3d51f0045197c5555d8687aeb1d0a535b1495c11a53792b.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/2ea05416ca177eaab3d51f0045197c5555d8687aeb1d0a535b1495c11a53792b.jpg)

![30a7883b83e3bd5a7a83d15dc16a4e353bac42325b31a08a4d562a7b1dfdb39e.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/30a7883b83e3bd5a7a83d15dc16a4e353bac42325b31a08a4d562a7b1dfdb39e.jpg)

![768a57f897581ff4376f04c5348bb7a0681c142b2a093512b86725933058bbbd.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/images/768a57f897581ff4376f04c5348bb7a0681c142b2a093512b86725933058bbbd.jpg)

### Tables

![27e4222b4f3b30fecafe664463a99925f31a1867a2b1cc898ebea534ab5c6aa2.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/tables/27e4222b4f3b30fecafe664463a99925f31a1867a2b1cc898ebea534ab5c6aa2.jpg)

![63936655a7a94e0c93997b22462f0b838889d46c8fa3625f2fe2d7e2491b7cb3.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/tables/63936655a7a94e0c93997b22462f0b838889d46c8fa3625f2fe2d7e2491b7cb3.jpg)

![80efbd7aa808f9612987b9ae3a00510247795a9d95e42276be8beb195856006e.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/tables/80efbd7aa808f9612987b9ae3a00510247795a9d95e42276be8beb195856006e.jpg)

![8d2aaa69e7cdb4f8798dc173ec8cc0a5d150da808666e51e11a61543cb4be321.jpg](../icml_results/4701_Corrected%20Samplers%20for%20Discrete%20Flow%20Models/tables/8d2aaa69e7cdb4f8798dc173ec8cc0a5d150da808666e51e11a61543cb4be321.jpg)
