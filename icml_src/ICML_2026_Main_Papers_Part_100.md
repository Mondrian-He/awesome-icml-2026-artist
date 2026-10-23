# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 100 of 197

## 目录 (Table of Contents)

1. [Learning a Generative Meta-Model of LLM Activations](#paper-3268)
2. [Vision-Language-Action Pretraining from Large-Scale Human Videos](#paper-3269)
3. [Graph of States: Solving Abductive Tasks with Large Language Models](#paper-3270)
4. [AsyncSpade: Efficient Test-Time Scaling with Asynchronous Sparse Decoding](#paper-3271)
5. [D$^2$O: A Dual Debiasing Operator for Training-Free Test-Time Adaptation of Vision–Language Models](#paper-3272)
6. [TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward](#paper-3273)
7. [AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](#paper-3274)
8. [Physically-Guided Data-Space Rectified Flow for Precipitation Nowcasting](#paper-3275)
9. [TarGATE: Target-Aware Data Selection via Token-Attenuation Gates](#paper-3276)
10. [Vision Language Models Cannot Reason About Physical Transformation](#paper-3277)
11. [DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs](#paper-3278)
12. [IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning](#paper-3279)
13. [Task-and-Model-Aware Fractal-Consistency for Efficient LLM Reasoning](#paper-3280)
14. [TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins](#paper-3281)
15. [SimGFM: Simplifying Discrete Flow Matching for Graph Generation](#paper-3282)
16. [CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](#paper-3283)
17. [Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](#paper-3284)
18. [DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning](#paper-3285)
19. [A Risk Decomposition Framework for Pre-hoc Fine-tuning Prediction](#paper-3286)
20. [Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations](#paper-3287)
21. [ACO-MoE-LoRA: Evolving-while-Training for Adapting Segment Anything Model 2 to Specialized Domains](#paper-3288)
22. [Online Change Point Detection for Multivariate Inhomogeneous Poisson Processes Time Series](#paper-3289)
23. [Utility Boundary of Dataset Distillation: Scaling and Coverage Laws](#paper-3290)
24. [STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](#paper-3291)
25. [ProConMV: Provenance-Enabled Conceptual Framework for Interpretable Multi-View Diabetic Retinopathy Diagnosis](#paper-3292)
26. [Closing the Loop: Universal Repository Representation with RPG-Encoder](#paper-3293)
27. [Causal Matrix Completion under Multiple Treatments via Mixed Synthetic Nearest Neighbors](#paper-3294)
28. [MotionMAR: Multi-scale Auto-Regressive Human Motion Reconstruction from Sparse Observations](#paper-3295)
29. [SABER: Continual Learning with Representation Conflict Management](#paper-3296)
30. [SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](#paper-3297)
31. [UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios](#paper-3298)
32. [MICE-Bench: A Challenging and Comprehensive Benchmark for Multi-Reference Image Creation and Editing](#paper-3299)
33. [4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere](#paper-3300)

---


<a id="paper-3268"></a>

## Learning a Generative Meta-Model of LLM Activations

### Images

![3460c8214b570df8af543b54b2c05bbd12bba11e96de2518cd102d8dba4348b2.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/3460c8214b570df8af543b54b2c05bbd12bba11e96de2518cd102d8dba4348b2.jpg)

![4841b8e5715210c5e479ebf1fa7d477796a558ea6cc4df483d8ee414541fbd4d.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/4841b8e5715210c5e479ebf1fa7d477796a558ea6cc4df483d8ee414541fbd4d.jpg)

![60e3f24ff7a8dd9501b03d55614b401f79e14d22263a06074bdd2dedda27950f.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/60e3f24ff7a8dd9501b03d55614b401f79e14d22263a06074bdd2dedda27950f.jpg)

![6c50e121f68b38619d63e70de1946a2ed88e3a86d29fae2bfe688f24b9ab69d3.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/6c50e121f68b38619d63e70de1946a2ed88e3a86d29fae2bfe688f24b9ab69d3.jpg)

![7d31695e0e912df2852a43df03d4b3e3e31f40f5fa4e1e8fb3d281cca7625553.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/7d31695e0e912df2852a43df03d4b3e3e31f40f5fa4e1e8fb3d281cca7625553.jpg)

![944573af906012106158330b02baa671f843ec01fb19d474ac974c2e250b4484.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/944573af906012106158330b02baa671f843ec01fb19d474ac974c2e250b4484.jpg)

![97960dae6148e0274a1deea56ce91afab573bb7c71c5432e869cedc6ac05b641.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/97960dae6148e0274a1deea56ce91afab573bb7c71c5432e869cedc6ac05b641.jpg)

![c4d12a1f53fa690704292fd05955d4367106f2600b7cea2a1c8bd24500e89d53.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/c4d12a1f53fa690704292fd05955d4367106f2600b7cea2a1c8bd24500e89d53.jpg)

![df276b0ee734e2b8c8414210f690db9f41f80a0dde88e9c56e3b65813923561e.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/df276b0ee734e2b8c8414210f690db9f41f80a0dde88e9c56e3b65813923561e.jpg)

![e039c5bb8db6b6fd9102ce8e5e0b3164b161f11a7baa49ed126cc31bbe53d2cc.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/e039c5bb8db6b6fd9102ce8e5e0b3164b161f11a7baa49ed126cc31bbe53d2cc.jpg)

![e4279a375dc537fe6e66ea62999dbe8263f755f4e041d52f66e21a04e6fcad3e.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/e4279a375dc537fe6e66ea62999dbe8263f755f4e041d52f66e21a04e6fcad3e.jpg)

![f87e318808a19e557a72aee8da47e61bc809a05baba821dccc77a1c33b7c31fa.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/images/f87e318808a19e557a72aee8da47e61bc809a05baba821dccc77a1c33b7c31fa.jpg)

### Tables

![0d92acb78132b00f65547c8f5a0847372345f83c58d1794cf780f95e2a7fce6f.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/0d92acb78132b00f65547c8f5a0847372345f83c58d1794cf780f95e2a7fce6f.jpg)

![245ee958ec9ac726e62bb3d43eae77522f8e6cf89b5abddd94e93b889fa6c4df.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/245ee958ec9ac726e62bb3d43eae77522f8e6cf89b5abddd94e93b889fa6c4df.jpg)

![5a4fbe0757eb1a038bd71ee982a4ec63c8c4b9b1c5163cd239259daa4c5f9a8e.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/5a4fbe0757eb1a038bd71ee982a4ec63c8c4b9b1c5163cd239259daa4c5f9a8e.jpg)

![d38b6540346680e2470d316438a44cad6a359ef9daf3c2a92973ead3bb753c93.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/d38b6540346680e2470d316438a44cad6a359ef9daf3c2a92973ead3bb753c93.jpg)

![e3d8194f4b875610836b24239f05b8664712921775eb7c5de01280597a142687.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/e3d8194f4b875610836b24239f05b8664712921775eb7c5de01280597a142687.jpg)

![fe525a3bbae7363fa705bcdc8d130095d5500994e1a78d2809f797ccb848d3bf.jpg](../icml_results/3306_Graph-R1_%20Towards%20Agentic%20GraphRAG%20Framework%20via%20End-to-end%20Reinforcement%20Learning/tables/fe525a3bbae7363fa705bcdc8d130095d5500994e1a78d2809f797ccb848d3bf.jpg)

## Learning a Generative Meta-Model of LLM Activations


### Images

![02f1942f0cd4debf46696ceb1b76db56d8cd116d0ed88a0d1eff3455330ccff3.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/02f1942f0cd4debf46696ceb1b76db56d8cd116d0ed88a0d1eff3455330ccff3.jpg)

![03a73b9a282ac9f8559753b24746b2145fc542ca584394c28f89648dd23138d4.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/03a73b9a282ac9f8559753b24746b2145fc542ca584394c28f89648dd23138d4.jpg)

![148bd7af7556c1d547d83ac1e13ba4cc3dd3ea9551268ca58f39c4aec95b1444.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/148bd7af7556c1d547d83ac1e13ba4cc3dd3ea9551268ca58f39c4aec95b1444.jpg)

![1e632236b1d1babfacbba53e8ca900b48891aeaf5f8cd9cc69458ef33097c3cd.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/1e632236b1d1babfacbba53e8ca900b48891aeaf5f8cd9cc69458ef33097c3cd.jpg)

![254bc54344687109dca6df6655b34a89ed0e939a6237fed8b3efaa80a1cc7968.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/254bc54344687109dca6df6655b34a89ed0e939a6237fed8b3efaa80a1cc7968.jpg)

![2f595808506449661825734c56d07126ea520b79c87dba99aaa2137752b80898.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/2f595808506449661825734c56d07126ea520b79c87dba99aaa2137752b80898.jpg)

![32989956f2e772c34a0f1a709e8eefeb668cf00381913be980979bcbf01c2e83.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/32989956f2e772c34a0f1a709e8eefeb668cf00381913be980979bcbf01c2e83.jpg)

![5e5c5e6c554f4bc2a72a5ffe9633b730090672ef7320466e9213d3b8f7ebab00.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/5e5c5e6c554f4bc2a72a5ffe9633b730090672ef7320466e9213d3b8f7ebab00.jpg)

![65f1c03b935bcc70359b15985fb4ef15373d517c44fb9ea81919c949b88cf76e.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/65f1c03b935bcc70359b15985fb4ef15373d517c44fb9ea81919c949b88cf76e.jpg)

![77ed3d3dd633b8039d0aad53fc84cafc3e9d8b69fe584a301932f1255a2a5d79.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/77ed3d3dd633b8039d0aad53fc84cafc3e9d8b69fe584a301932f1255a2a5d79.jpg)

![805cc3b054350dec33d53ff85ddff0bb48779fd7af813d87bd3328813c1da515.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/805cc3b054350dec33d53ff85ddff0bb48779fd7af813d87bd3328813c1da515.jpg)

![9e448f0fdaf0cff89603e88c2747683e78bafca3019a1f62faf6e2ab9021c617.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/9e448f0fdaf0cff89603e88c2747683e78bafca3019a1f62faf6e2ab9021c617.jpg)

![a628889688e21bd4f1027a40d47041ce0014131f435e371af12976f035e9d8a2.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/a628889688e21bd4f1027a40d47041ce0014131f435e371af12976f035e9d8a2.jpg)

![bcca12c100d68c34f323a8c4d09dc0fac74d9e829891c0761508379618de0f59.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/bcca12c100d68c34f323a8c4d09dc0fac74d9e829891c0761508379618de0f59.jpg)

![c8d878a645e8e833c2403b240a80d7433ccdc0dc77d66f91c88857ad584b2de4.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/c8d878a645e8e833c2403b240a80d7433ccdc0dc77d66f91c88857ad584b2de4.jpg)

![d0e03e1e9bb2bf382975e5a15d272301a3ea407df80f1fc03c27c0699f0362cf.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/d0e03e1e9bb2bf382975e5a15d272301a3ea407df80f1fc03c27c0699f0362cf.jpg)

![f4a3dffbe7705e7f91e01f36035e990da997612da2a9a6dc2c2cc52111dd15b5.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/f4a3dffbe7705e7f91e01f36035e990da997612da2a9a6dc2c2cc52111dd15b5.jpg)

![fca584942f6730b260beb3cf45ad9867ef39e3e6ecf71184a7e2d8ded7ad9dee.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/images/fca584942f6730b260beb3cf45ad9867ef39e3e6ecf71184a7e2d8ded7ad9dee.jpg)

### Tables

![1036af2a2573c82c6865d7f0f03042b69bc8725ee55155025dd1fa63e99aa96e.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/1036af2a2573c82c6865d7f0f03042b69bc8725ee55155025dd1fa63e99aa96e.jpg)

![20987a939afc3812138ae4f5a6e7eda0d3ba82943616b1b4ed955c9beac3bb2a.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/20987a939afc3812138ae4f5a6e7eda0d3ba82943616b1b4ed955c9beac3bb2a.jpg)

![86e772047b25b5944fd75d29f9dd5227c809e86955c1f6ad8f770f5198893573.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/86e772047b25b5944fd75d29f9dd5227c809e86955c1f6ad8f770f5198893573.jpg)

![9d7ae25944bfbd78272705a899455025fab84efd180053c41ebccd6eb896d498.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/9d7ae25944bfbd78272705a899455025fab84efd180053c41ebccd6eb896d498.jpg)

![a014e7bda3f1ce30e2bb992031c912ca1c2c5eda2300619f49e4afec60b544a2.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/a014e7bda3f1ce30e2bb992031c912ca1c2c5eda2300619f49e4afec60b544a2.jpg)

![a0c7a3c3be1d68e78063fb27b9b262dd6d629ed62495ad90f8e35c62efea36f4.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/a0c7a3c3be1d68e78063fb27b9b262dd6d629ed62495ad90f8e35c62efea36f4.jpg)

![a772c35c0834c7a2afc61f9ba9a6e249010b485485b58ff0fada4afcf6b5b91c.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/a772c35c0834c7a2afc61f9ba9a6e249010b485485b58ff0fada4afcf6b5b91c.jpg)

![b8ff5b6a9522676e9283b7bec07de9d28b9dd81605005cacea1081f3e1ea5efe.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/b8ff5b6a9522676e9283b7bec07de9d28b9dd81605005cacea1081f3e1ea5efe.jpg)

![bfac604500814bc6c1be04e86a0f30c849fe1134973bc0bf88223635a7c678a1.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/bfac604500814bc6c1be04e86a0f30c849fe1134973bc0bf88223635a7c678a1.jpg)

![cd455d9ab99cf73c5e23b5ca4c4658c0a299bf3346e046364cf2826e8d18fc13.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/cd455d9ab99cf73c5e23b5ca4c4658c0a299bf3346e046364cf2826e8d18fc13.jpg)

![d52704189f898198e4256aa0f023737826db3cd6fd52d32f589b55190da04b59.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/d52704189f898198e4256aa0f023737826db3cd6fd52d32f589b55190da04b59.jpg)

![da883c5d7a0eb64cb75cf97ff8a2cbc2d70e5cfdb013ffa2f5044e0a37c04f18.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/da883c5d7a0eb64cb75cf97ff8a2cbc2d70e5cfdb013ffa2f5044e0a37c04f18.jpg)

![dfb3f4c765bfe529120145cc8ce106f671544f523c8d1172b664ac0a227c5a4a.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/dfb3f4c765bfe529120145cc8ce106f671544f523c8d1172b664ac0a227c5a4a.jpg)

![e6a0a5fc0c689bb428401a9b8262d78609f25b42d74970bc818cfd73aba7d56f.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/e6a0a5fc0c689bb428401a9b8262d78609f25b42d74970bc818cfd73aba7d56f.jpg)

![f91cd467c58d2d60e8d3ad7c39bd07db63e4690a81e96697cc71139eadea3730.jpg](../icml_results/3307_Learning%20a%20Generative%20Meta-Model%20of%20LLM%20Activations/tables/f91cd467c58d2d60e8d3ad7c39bd07db63e4690a81e96697cc71139eadea3730.jpg)

<a id="paper-3269"></a>

## Vision-Language-Action Pretraining from Large-Scale Human Videos


### Images

![123a5a7143ffb3d42da2796e3f8b027aabf718405e95a77fb91392493f4b0a96.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/123a5a7143ffb3d42da2796e3f8b027aabf718405e95a77fb91392493f4b0a96.jpg)

![1521932241cf5b19681bd47d26db2ad1613e17e010cb5c5c04fa1bf584552faa.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/1521932241cf5b19681bd47d26db2ad1613e17e010cb5c5c04fa1bf584552faa.jpg)

![15db8a0934c440a91901b3efa9d26c7a3e295b05a09faf2624e381cc1b9be954.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/15db8a0934c440a91901b3efa9d26c7a3e295b05a09faf2624e381cc1b9be954.jpg)

![1cb55b1deabb86cb43baf05a1ca75c61869ffc3d8108725e75fc6284a259dec0.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/1cb55b1deabb86cb43baf05a1ca75c61869ffc3d8108725e75fc6284a259dec0.jpg)

![2d4296ef4270c1919bc1683fb04ccd8e4cf9f72e64e382cce38b519c9b870cf3.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/2d4296ef4270c1919bc1683fb04ccd8e4cf9f72e64e382cce38b519c9b870cf3.jpg)

![327c5f11e5ddc1f7b02066ea22d7ac579e40ffeba30d04e731a0c48354cde5da.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/327c5f11e5ddc1f7b02066ea22d7ac579e40ffeba30d04e731a0c48354cde5da.jpg)

![333ad83d0dc61324fa2b0ae3228652fde923d34336b59df7aedf4814486ae95e.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/333ad83d0dc61324fa2b0ae3228652fde923d34336b59df7aedf4814486ae95e.jpg)

![3af6a1500e1998c04ca9ea62e8cecf2cf36b37784f3d162023752e03ccab059f.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/3af6a1500e1998c04ca9ea62e8cecf2cf36b37784f3d162023752e03ccab059f.jpg)

![3ee94f810be62389e8a6763ad63801c96b8523570bbdf4dca9d5cda476ae379e.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/3ee94f810be62389e8a6763ad63801c96b8523570bbdf4dca9d5cda476ae379e.jpg)

![5ac9bb5716686260315e7e97aa02fd5d111d1a560280f0b5dc99669b6f19ca01.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/5ac9bb5716686260315e7e97aa02fd5d111d1a560280f0b5dc99669b6f19ca01.jpg)

![6166d8852148b79f923f2a11b57af482f65d497cdf32a8a9fd342c7c6017c4d2.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/6166d8852148b79f923f2a11b57af482f65d497cdf32a8a9fd342c7c6017c4d2.jpg)

![77a2606048c0104eb1926cd9f31470c4f7158504e6aea90aad9621f7fb0221ae.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/77a2606048c0104eb1926cd9f31470c4f7158504e6aea90aad9621f7fb0221ae.jpg)

![a2380c7ba7fdcfcf6866b502bc1d4e08a5e5b27a30df22425a3c476cf50b6fdc.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/a2380c7ba7fdcfcf6866b502bc1d4e08a5e5b27a30df22425a3c476cf50b6fdc.jpg)

![ac16a8e516ed46b917871714aaccfd7d77316beb296fb92a3adb2b92a9114f59.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/ac16a8e516ed46b917871714aaccfd7d77316beb296fb92a3adb2b92a9114f59.jpg)

![b05284f2bd42409030a0f867418ffeca787e9b3c1f13eadaf2ff30db1760488c.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/b05284f2bd42409030a0f867418ffeca787e9b3c1f13eadaf2ff30db1760488c.jpg)

![bbad25accc7d297e10403412523d8dbfe4856214d063063c618e4e966503ed0d.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/bbad25accc7d297e10403412523d8dbfe4856214d063063c618e4e966503ed0d.jpg)

![d3e52d7aeb5d09422cc0c6157c9e5c115ba87b66771adc4e99c4c5327073db9c.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/d3e52d7aeb5d09422cc0c6157c9e5c115ba87b66771adc4e99c4c5327073db9c.jpg)

![e4a640b10a95a39fccbf4fe2d2e3e98583b27efeabdcf841c6227f68682bf44d.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/e4a640b10a95a39fccbf4fe2d2e3e98583b27efeabdcf841c6227f68682bf44d.jpg)

![f8eb60f943068d831246fdba17254d02ac3f36a811a0caf95bd7638749b65301.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/images/f8eb60f943068d831246fdba17254d02ac3f36a811a0caf95bd7638749b65301.jpg)

### Tables

![1770164fef9c0aa30bae62c63ad4c4d7bcd728fa447b4d6bfde1c93474ce9dd7.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/1770164fef9c0aa30bae62c63ad4c4d7bcd728fa447b4d6bfde1c93474ce9dd7.jpg)

![1840f326b69e4bd4ac46b1c24bfa3362a6b5aec9cb10521777b0ec01f93a0c23.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/1840f326b69e4bd4ac46b1c24bfa3362a6b5aec9cb10521777b0ec01f93a0c23.jpg)

![3cd70995a27c795a32d91d537eaf5384be0f601ed31f8c088efc3a8f91776d60.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/3cd70995a27c795a32d91d537eaf5384be0f601ed31f8c088efc3a8f91776d60.jpg)

![43892c720a9e11ba01198d31d154194550d0a4bcbc4d803ccb1b6190011250a0.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/43892c720a9e11ba01198d31d154194550d0a4bcbc4d803ccb1b6190011250a0.jpg)

![4bd88f350c7627d9e6c1ba87ff27f428c5f4cfcc763a827c4497c211bb54402b.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/4bd88f350c7627d9e6c1ba87ff27f428c5f4cfcc763a827c4497c211bb54402b.jpg)

![65a50038529bd61d20dda176d5595227350c980d76ca5c62e36b03a1da0f1425.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/65a50038529bd61d20dda176d5595227350c980d76ca5c62e36b03a1da0f1425.jpg)

![6f518687e559984bba960ed6f9cea53eba5ee3959e8a5e943e90d9fe870768b8.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/6f518687e559984bba960ed6f9cea53eba5ee3959e8a5e943e90d9fe870768b8.jpg)

![782ce44ed54cfea9fe40d150975cd620276873cc1332330c77250ac370bb6834.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/782ce44ed54cfea9fe40d150975cd620276873cc1332330c77250ac370bb6834.jpg)

![7adf23bd35f50c24eaaeb3ceaf9b3b1e8010e120a4528ec571edaa19c1da4ad8.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/7adf23bd35f50c24eaaeb3ceaf9b3b1e8010e120a4528ec571edaa19c1da4ad8.jpg)

![967c67e40a4090b3519f6b23fec3abafeda1dc485e975fe8947ab2dd4b021bb7.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/967c67e40a4090b3519f6b23fec3abafeda1dc485e975fe8947ab2dd4b021bb7.jpg)

![9d1c4d9c49aa235f802f7c84034e7338a97962dc61d2b637ab687b463e15cdc7.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/9d1c4d9c49aa235f802f7c84034e7338a97962dc61d2b637ab687b463e15cdc7.jpg)

![9e935808c82867becd16a54278a0dd0c12a07c0e62476aee6806aaf4a9e54135.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/9e935808c82867becd16a54278a0dd0c12a07c0e62476aee6806aaf4a9e54135.jpg)

![e8da4fbd8edf66126bce18c0d297230ed205e641a5c53ea8e010902cdd99d797.jpg](../icml_results/3308_Vision-Language-Action%20Pretraining%20from%20Large-Scale%20Human%20Videos/tables/e8da4fbd8edf66126bce18c0d297230ed205e641a5c53ea8e010902cdd99d797.jpg)

<a id="paper-3270"></a>

## Graph of States: Solving Abductive Tasks with Large Language Models


### Images

![100e9db9f5ed8a40630cb180c11e2b05f99b6b80483f4272159173bd043d55d5.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/100e9db9f5ed8a40630cb180c11e2b05f99b6b80483f4272159173bd043d55d5.jpg)

![34b3b0470ed1804f1cd82edb68454e7f159000b249904ee3f9c5da86a955f479.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/34b3b0470ed1804f1cd82edb68454e7f159000b249904ee3f9c5da86a955f479.jpg)

![5f4752cdb09463776153dddd67ba1606c09daf3e9d19d8a2addda924e28e80a7.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/5f4752cdb09463776153dddd67ba1606c09daf3e9d19d8a2addda924e28e80a7.jpg)

![800218b490e980bb3db11e21db271c530f36e0e9c51c55dd1e501dbccaee1183.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/800218b490e980bb3db11e21db271c530f36e0e9c51c55dd1e501dbccaee1183.jpg)

![8c3775f60b573fcf6f4686bf87eccce9f95620f7696026a59d6eb20639069278.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/8c3775f60b573fcf6f4686bf87eccce9f95620f7696026a59d6eb20639069278.jpg)

![9902fd0ab49180f909ed59f166cf2c2ae888c85085dbff5e74ab80c13c3f1db6.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/9902fd0ab49180f909ed59f166cf2c2ae888c85085dbff5e74ab80c13c3f1db6.jpg)

![b2f9ce489e6c993009f5e974287a184a0132fa0be72e97bb1cddb0e82fb92679.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/b2f9ce489e6c993009f5e974287a184a0132fa0be72e97bb1cddb0e82fb92679.jpg)

![bf091c944b9f55a1b2a97dd960c82b01d31785615a53384eb96243af2bac8873.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/bf091c944b9f55a1b2a97dd960c82b01d31785615a53384eb96243af2bac8873.jpg)

![d785bc4d98fdd5a34e280d146aecbc3f7ec1437ee2bdd3dbb5af787845b942a7.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/d785bc4d98fdd5a34e280d146aecbc3f7ec1437ee2bdd3dbb5af787845b942a7.jpg)

![f4ac06223692a7dc4768c694ad0d56ff4581f598bd06485f933ebe296ecdcd78.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/f4ac06223692a7dc4768c694ad0d56ff4581f598bd06485f933ebe296ecdcd78.jpg)

![fb865fa85e6b56cbcbc2a4d01d9c800de1a85a550f7a6b2a07105b3bd5c383ee.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/images/fb865fa85e6b56cbcbc2a4d01d9c800de1a85a550f7a6b2a07105b3bd5c383ee.jpg)

### Tables

![105612055644d864ba0f56e5cd04f875b7d0917d2b5fad819da0175147f12e76.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/105612055644d864ba0f56e5cd04f875b7d0917d2b5fad819da0175147f12e76.jpg)

![4548c3f4811b1e9b9a709c4de551b1a6b457f022b73238cb74b2ed019e9fab71.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/4548c3f4811b1e9b9a709c4de551b1a6b457f022b73238cb74b2ed019e9fab71.jpg)

![5da9c98642ce8cae143a7c5dc4d4d80959896f21d1a554d90df621697074d4bd.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/5da9c98642ce8cae143a7c5dc4d4d80959896f21d1a554d90df621697074d4bd.jpg)

![6c08a3b966a68b87b8bd2fd9efdabb6f3c09c0232e203b45ec0da43d94ec1752.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/6c08a3b966a68b87b8bd2fd9efdabb6f3c09c0232e203b45ec0da43d94ec1752.jpg)

![a8b6e1c1866aebdba69bd9a27a6375058190651120973f18e80963f80f9d6e32.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/a8b6e1c1866aebdba69bd9a27a6375058190651120973f18e80963f80f9d6e32.jpg)

![d6de35a65bbb072b5d8d01df81ac1659582c0cee288e4759f520d0259de50036.jpg](../icml_results/3309_Graph%20of%20States_%20Solving%20Abductive%20Tasks%20with%20Large%20Language%20Models/tables/d6de35a65bbb072b5d8d01df81ac1659582c0cee288e4759f520d0259de50036.jpg)

<a id="paper-3271"></a>

## AsyncSpade: Efficient Test-Time Scaling with Asynchronous Sparse Decoding


### Images

![29c7c334aec49c93a88b36df1143e3ba1d5d77ef7b0394af2b941c13c3cb90e2.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/29c7c334aec49c93a88b36df1143e3ba1d5d77ef7b0394af2b941c13c3cb90e2.jpg)

![34e45a6e589a3be3cb4c925831010a213131118a4a4a1414423b7fd5c1a92d48.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/34e45a6e589a3be3cb4c925831010a213131118a4a4a1414423b7fd5c1a92d48.jpg)

![369ce5fd4e4a86ef78abdc5cc47dda3da66039d3f11f89e2e2e2788e67a8809e.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/369ce5fd4e4a86ef78abdc5cc47dda3da66039d3f11f89e2e2e2788e67a8809e.jpg)

![6264cbd61601d45bcec6fbaef23dd1c258280f8701b3ec32c900fb9fd1942a29.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/6264cbd61601d45bcec6fbaef23dd1c258280f8701b3ec32c900fb9fd1942a29.jpg)

![652ae153022d52871e5e4ce9ca47b2244e909afe034d51164b82ec5e84906dac.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/652ae153022d52871e5e4ce9ca47b2244e909afe034d51164b82ec5e84906dac.jpg)

![72b114b75eb4e07d2a54549e9aa4b6e9bd4e827b5251d521202536ce1e845ea2.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/72b114b75eb4e07d2a54549e9aa4b6e9bd4e827b5251d521202536ce1e845ea2.jpg)

![847d2814e1e2f635dddf2f2ef0c6cf793c1317a05f0002b046059378ea4d7bf7.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/847d2814e1e2f635dddf2f2ef0c6cf793c1317a05f0002b046059378ea4d7bf7.jpg)

![bd8cd98ba38785bf65f44402c35b36aafba672996c429d4cde505b4e851721cb.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/bd8cd98ba38785bf65f44402c35b36aafba672996c429d4cde505b4e851721cb.jpg)

![da30854331340fa7c4f2438e3c35364c49658b7ff224427b6b49ccbf55d67971.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/images/da30854331340fa7c4f2438e3c35364c49658b7ff224427b6b49ccbf55d67971.jpg)

### Tables

![2ca423bbad77ff8e1b6b1905bfd39c8c14917d9af35dccd7e4d0e86fdfba4e79.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/2ca423bbad77ff8e1b6b1905bfd39c8c14917d9af35dccd7e4d0e86fdfba4e79.jpg)

![537e928e5d360da27b22a8624d0298a03203a9268e3330d60243e5352572be76.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/537e928e5d360da27b22a8624d0298a03203a9268e3330d60243e5352572be76.jpg)

![7c072dcf96d9a01cca0ddf916f7ee0bbf8d794b24afd27c741111b2fda88a5c8.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/7c072dcf96d9a01cca0ddf916f7ee0bbf8d794b24afd27c741111b2fda88a5c8.jpg)

![7c179d4dbf17649e5cc3d3bf52ee73f68f9eb0682ba6a0223aec219f0fcd753c.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/7c179d4dbf17649e5cc3d3bf52ee73f68f9eb0682ba6a0223aec219f0fcd753c.jpg)

![a3a3fe555473894929ef15edb1b8b379ac5f2556ca75abae0af1f590c00f9664.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/a3a3fe555473894929ef15edb1b8b379ac5f2556ca75abae0af1f590c00f9664.jpg)

![ab5e7a78fcb02de9028c843dd16ce1fcf05bc9cde0e621fda481734136686433.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/ab5e7a78fcb02de9028c843dd16ce1fcf05bc9cde0e621fda481734136686433.jpg)

![ad85c90cb4e76a579d56c18253565c42dc4d5af70aece409e23c5dd39f4c87af.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/ad85c90cb4e76a579d56c18253565c42dc4d5af70aece409e23c5dd39f4c87af.jpg)

![c9aac427a8309279a6c02c2aca86d5224773b628b7ab23c4055676fa6430a33f.jpg](../icml_results/3310_AsyncSpade_%20Efficient%20Test-Time%20Scaling%20with%20Asynchronous%20Sparse%20Decoding/tables/c9aac427a8309279a6c02c2aca86d5224773b628b7ab23c4055676fa6430a33f.jpg)

<a id="paper-3272"></a>

## D$^2$O: A Dual Debiasing Operator for Training-Free Test-Time Adaptation of Vision–Language Models


### Images

![1e1d1aa0b4baf42c17e0281dbe79b0c77810c46849e052c25258a66fb3cb7665.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/1e1d1aa0b4baf42c17e0281dbe79b0c77810c46849e052c25258a66fb3cb7665.jpg)

![25eddcc74a6480fbce4eef379f990b0b5895892fafd6156694ec28bcb3cea09b.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/25eddcc74a6480fbce4eef379f990b0b5895892fafd6156694ec28bcb3cea09b.jpg)

![38e9fe75aceb1006b4bbaad3e4bcbeba3190c43d89a58c9a693cf45e5eeb0661.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/38e9fe75aceb1006b4bbaad3e4bcbeba3190c43d89a58c9a693cf45e5eeb0661.jpg)

![42da40dd39b517215ced6c0dee9041dc7f2b43baecdc3265a1bda0a731fdcf03.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/42da40dd39b517215ced6c0dee9041dc7f2b43baecdc3265a1bda0a731fdcf03.jpg)

![44f550c15b5c5a52f1d286b45641b014aa3f0f935ff14c609bde7150e2fcb1e1.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/44f550c15b5c5a52f1d286b45641b014aa3f0f935ff14c609bde7150e2fcb1e1.jpg)

![4980ae4ca3533ad9aadada839583a6d355354cddf270811391cbd52a42105dbc.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/4980ae4ca3533ad9aadada839583a6d355354cddf270811391cbd52a42105dbc.jpg)

![80210e162f019d7647d8e8dc34db628df8266afe5b2bbe1e75a6ab86a33ccb07.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/80210e162f019d7647d8e8dc34db628df8266afe5b2bbe1e75a6ab86a33ccb07.jpg)

![804efc57406898deb82e4ca04294448771c02d378baf2f7749340ec6e8d93338.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/804efc57406898deb82e4ca04294448771c02d378baf2f7749340ec6e8d93338.jpg)

![8d8f0831cd5526cc3820835e51b0cb5e05100cf28922d12b44a6d33ab505798e.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/8d8f0831cd5526cc3820835e51b0cb5e05100cf28922d12b44a6d33ab505798e.jpg)

![fe792e2ea202d6febc574db59e7cd4f2c38036c14d6b0e9c4fa8ff5a3ddb9bcf.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/fe792e2ea202d6febc574db59e7cd4f2c38036c14d6b0e9c4fa8ff5a3ddb9bcf.jpg)

![ffa80f9c4fbd54b4c511e88f605967a7a7ce40d40b97738f3c7bf1132ac03f10.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/images/ffa80f9c4fbd54b4c511e88f605967a7a7ce40d40b97738f3c7bf1132ac03f10.jpg)

### Tables

![0798349ca8fcaa7c0b8bb2038fea5bf2e967e3538b47a7f4a595030749300810.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/0798349ca8fcaa7c0b8bb2038fea5bf2e967e3538b47a7f4a595030749300810.jpg)

![1ae4dee947601665109798ebb82460f9f2c453e13b5fb2517ff04f2c4b27b97a.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/1ae4dee947601665109798ebb82460f9f2c453e13b5fb2517ff04f2c4b27b97a.jpg)

![23f9a5d65a920f39c402f7bbf81b8a17a30a95f0528ddb31e285e70c7e3179e6.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/23f9a5d65a920f39c402f7bbf81b8a17a30a95f0528ddb31e285e70c7e3179e6.jpg)

![276e9ba870a08a112566ca846b64d575ceb3b428d0174cf537e20aa593c80fa9.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/276e9ba870a08a112566ca846b64d575ceb3b428d0174cf537e20aa593c80fa9.jpg)

![286298171eedd0f8f28ee6a79d290788438b6223f2f5f7a29cd57921c6d57f66.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/286298171eedd0f8f28ee6a79d290788438b6223f2f5f7a29cd57921c6d57f66.jpg)

![2ceace2420dad1183a44156a9f68ba358cbe503ed6ca77ac7ceab89460efea46.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/2ceace2420dad1183a44156a9f68ba358cbe503ed6ca77ac7ceab89460efea46.jpg)

![3ff437a67c5d9d053b77b62b4b06e8eafce363fb67b141bdc28eaf385124b4a6.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/3ff437a67c5d9d053b77b62b4b06e8eafce363fb67b141bdc28eaf385124b4a6.jpg)

![4581c354801641e1f8fd4edc40dac16d0a1304cab384c48a6e3ffae049b92bfe.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/4581c354801641e1f8fd4edc40dac16d0a1304cab384c48a6e3ffae049b92bfe.jpg)

![4821e89736d5866bf44f7081978ef3552dd71931a3be80e2e5c2cf3d64bc0c0a.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/4821e89736d5866bf44f7081978ef3552dd71931a3be80e2e5c2cf3d64bc0c0a.jpg)

![4e627c8d8ac637495d4fb8c89279909d83170c2d14f77c26b366a7b0e6bee13d.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/4e627c8d8ac637495d4fb8c89279909d83170c2d14f77c26b366a7b0e6bee13d.jpg)

![64b74640c7275687fef19c7b1af639d3fd5ec815fb6baaf855a446b447fd4cca.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/64b74640c7275687fef19c7b1af639d3fd5ec815fb6baaf855a446b447fd4cca.jpg)

![6b0fa9b68068f42e11ad2ad7fd2d67d0a3bf8956c46915ef3eddc190ab30402f.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/6b0fa9b68068f42e11ad2ad7fd2d67d0a3bf8956c46915ef3eddc190ab30402f.jpg)

![7ef24e3e5ced8c52718fed24e3a6c818da27467127e50f0186405100272abe53.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/7ef24e3e5ced8c52718fed24e3a6c818da27467127e50f0186405100272abe53.jpg)

![a70d564fa1e2906f1d91e55931af55f2668d57989a1f74eb6006a69a916b426e.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/a70d564fa1e2906f1d91e55931af55f2668d57989a1f74eb6006a69a916b426e.jpg)

![abba1312ee753db97e5f66aaed80ffead20f8911df7ba80eee68cdc2d4c79c36.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/abba1312ee753db97e5f66aaed80ffead20f8911df7ba80eee68cdc2d4c79c36.jpg)

![abdc0fea84016ec195977dab910b1226dac1945d00812b04aece5253c561054d.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/abdc0fea84016ec195977dab910b1226dac1945d00812b04aece5253c561054d.jpg)

![b2266dc1ae495cb7248110aa763ba9a10a457508d0d357d2cb8873ebda8b241a.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/b2266dc1ae495cb7248110aa763ba9a10a457508d0d357d2cb8873ebda8b241a.jpg)

![be8ade83fa87c2cd752a68b0a4dcc6859be9d72d309d3097e7e59af60e35e3af.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/be8ade83fa87c2cd752a68b0a4dcc6859be9d72d309d3097e7e59af60e35e3af.jpg)

![c06f431863f61c5b5558cd785505c97b3c54f5ff83bd9c1e644bf0d5af368100.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/c06f431863f61c5b5558cd785505c97b3c54f5ff83bd9c1e644bf0d5af368100.jpg)

![cfb28ff5787539c1ab51eaddcc73c5a2f5d8b3c993d4904619c976b1c9db09b9.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/cfb28ff5787539c1ab51eaddcc73c5a2f5d8b3c993d4904619c976b1c9db09b9.jpg)

![dadc066f724b469082c73e8904bcd53a2017c4ca69406765489c7cb005bf25f5.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/dadc066f724b469082c73e8904bcd53a2017c4ca69406765489c7cb005bf25f5.jpg)

![dda2e237f203322e1f61d5873548a54f1657a4343bcd9fa7e4e739dbdfd71410.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/dda2e237f203322e1f61d5873548a54f1657a4343bcd9fa7e4e739dbdfd71410.jpg)

![e228d30360b35900241fb52879bed696c9a74eefb958a9b51a300fbca6f3183f.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/e228d30360b35900241fb52879bed696c9a74eefb958a9b51a300fbca6f3183f.jpg)

![fed58f3c8ce7a093de2c8171091bae980292c4b3bb7183bb3819842677498b42.jpg](../icml_results/3311_D%24%5E2%24O_%20A%20Dual%20Debiasing%20Operator%20for%20Training-Free%20Test-Time%20Adaptation%20of%20Vision%E2%80%93Language%20Models/tables/fed58f3c8ce7a093de2c8171091bae980292c4b3bb7183bb3819842677498b42.jpg)

<a id="paper-3273"></a>

## TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward


### Images

![04f7844f755d20a6b93b22f0347933f1e0a0f5fe1687b50d87cdba88f8cca9d7.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/04f7844f755d20a6b93b22f0347933f1e0a0f5fe1687b50d87cdba88f8cca9d7.jpg)

![30c45b0512a1c2c81b16e067d93242b83316380c7d6d9d8c136e871cd713b769.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/30c45b0512a1c2c81b16e067d93242b83316380c7d6d9d8c136e871cd713b769.jpg)

![374c11ee14fd828a42931b148128acb4597a08112b41254b3bc0bfb9c0e286c7.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/374c11ee14fd828a42931b148128acb4597a08112b41254b3bc0bfb9c0e286c7.jpg)

![9b2e355b301c847225b2bcbd38bd0e966061aee1076738e063d9bb68a05118d3.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/9b2e355b301c847225b2bcbd38bd0e966061aee1076738e063d9bb68a05118d3.jpg)

![9fc0912511cf829b698762d01c3a3390f9fbc85123a4be08be9d931be26b9562.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/9fc0912511cf829b698762d01c3a3390f9fbc85123a4be08be9d931be26b9562.jpg)

![af76f72564165be1017c506365fe3b9e105d6fa434fe7fd23dd7f3a03ef78fab.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/af76f72564165be1017c506365fe3b9e105d6fa434fe7fd23dd7f3a03ef78fab.jpg)

![dcde24d6c74509c699f5c2717844a76525a54ac15a6fe1e6ea5838a20726488e.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/dcde24d6c74509c699f5c2717844a76525a54ac15a6fe1e6ea5838a20726488e.jpg)

![ec23d15943b18c4753a8e580417c917f382900d0844e4f5fb9c8e8e5d6579673.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/ec23d15943b18c4753a8e580417c917f382900d0844e4f5fb9c8e8e5d6579673.jpg)

![f39164c15a1017b72796d18350463aa4dd4ef61f426f3f6b94d81082c40d26af.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/images/f39164c15a1017b72796d18350463aa4dd4ef61f426f3f6b94d81082c40d26af.jpg)

### Tables

![440f5adc6dc22dcc29434c19525a611412d1ce5152371c7d778d4901aa21a858.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/tables/440f5adc6dc22dcc29434c19525a611412d1ce5152371c7d778d4901aa21a858.jpg)

![44d7bd20c3f2589e2ef5fc7e04dbe5eb64d2bcc2d5ac628a1ada52c966ddcea6.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/tables/44d7bd20c3f2589e2ef5fc7e04dbe5eb64d2bcc2d5ac628a1ada52c966ddcea6.jpg)

![ff582f69893fa4002944f4f5b418f618a35d930180171e203a5219ef167cb203.jpg](../icml_results/3312_TDM-R1_%20Reinforcing%20Few-Step%20Diffusion%20Models%20with%20Non-Differentiable%20Reward/tables/ff582f69893fa4002944f4f5b418f618a35d930180171e203a5219ef167cb203.jpg)

<a id="paper-3274"></a>

## AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters


### Images

![14a20120cf4797041989ac06d33967fd15f60beb338cd3be0b91a12c3a9d06dc.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/14a20120cf4797041989ac06d33967fd15f60beb338cd3be0b91a12c3a9d06dc.jpg)

![182078dc7291d6f40f4f524e1e16f42a53b265c759d9d8d55ec5fe9d4cb9faa7.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/182078dc7291d6f40f4f524e1e16f42a53b265c759d9d8d55ec5fe9d4cb9faa7.jpg)

![1edd75f925b7665c7387a8ad9aa17fa42023ff92137018c11200f4c9d543a61b.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/1edd75f925b7665c7387a8ad9aa17fa42023ff92137018c11200f4c9d543a61b.jpg)

![2eae4435bb1669c579e5cd3001599660b32bc11a863c021f23d21a5f9ce0b6e2.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/2eae4435bb1669c579e5cd3001599660b32bc11a863c021f23d21a5f9ce0b6e2.jpg)

![3efcc697c32838701492e811c1f9e1725c1f01f2534024d76c4f59e59341140a.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/3efcc697c32838701492e811c1f9e1725c1f01f2534024d76c4f59e59341140a.jpg)

![43b423cb5a1b44c32493713d5f44853ae333dc5a7e561743a613924fc8536c5b.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/43b423cb5a1b44c32493713d5f44853ae333dc5a7e561743a613924fc8536c5b.jpg)

![44430f565be2c2a8f55d5e6754a8015066f00578d1d7fa6730cbab19d66caa34.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/44430f565be2c2a8f55d5e6754a8015066f00578d1d7fa6730cbab19d66caa34.jpg)

![6985170c71ca27ed75f8dd8b9d248eab5ca54a34ce2681a5372b8185e4c9b1fe.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/6985170c71ca27ed75f8dd8b9d248eab5ca54a34ce2681a5372b8185e4c9b1fe.jpg)

![786da345be552aadefe1aba3f8969004dc0dc4100e59ede16bcbf2f0ace0a351.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/786da345be552aadefe1aba3f8969004dc0dc4100e59ede16bcbf2f0ace0a351.jpg)

![916cc065f26e1200b912aaf097ecd4cc17de6778d72e56cffeb9ebb56d152c94.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/916cc065f26e1200b912aaf097ecd4cc17de6778d72e56cffeb9ebb56d152c94.jpg)

![9a80207962c85494e7b93cae35dba2561d6169f60fbf8b87f22b1e522f49cb24.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/9a80207962c85494e7b93cae35dba2561d6169f60fbf8b87f22b1e522f49cb24.jpg)

![a13305b67cc24566d3f26770b86eb73ce796abdb4f9ad903933d95c95ff9aefa.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/a13305b67cc24566d3f26770b86eb73ce796abdb4f9ad903933d95c95ff9aefa.jpg)

![a34ca098c63c24f2540cc8deecfc2870664d3e943be89f830ba1bff310ca7595.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/a34ca098c63c24f2540cc8deecfc2870664d3e943be89f830ba1bff310ca7595.jpg)

![bb3df7355d1f5e0f6d98e37ae803edb840fa93c7a0e8765ed5968e79725e13b7.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/bb3df7355d1f5e0f6d98e37ae803edb840fa93c7a0e8765ed5968e79725e13b7.jpg)

![c2199be7788e538b93f7f343125bdf91ba72e73c194b2319b3b5ba36ae06253d.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/c2199be7788e538b93f7f343125bdf91ba72e73c194b2319b3b5ba36ae06253d.jpg)

![cf832d39f102ee382728abccea0ced2e5fa4496861a09b7749b54bf4bc57b85e.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/cf832d39f102ee382728abccea0ced2e5fa4496861a09b7749b54bf4bc57b85e.jpg)

![d5fcd3c1b76ee5dca44b598f93c959dd013b666d11c775b1eb1a78047d6d2a2e.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/d5fcd3c1b76ee5dca44b598f93c959dd013b666d11c775b1eb1a78047d6d2a2e.jpg)

![df5e9fce24566958bdc5e66b23ffc0363eb5edfd354d065e3c0cda4237e18bd6.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/df5e9fce24566958bdc5e66b23ffc0363eb5edfd354d065e3c0cda4237e18bd6.jpg)

![f0d96e5ef9f877a3e62793a0014c60a90b1535953eb410652f7c02fdd0fc57a8.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/images/f0d96e5ef9f877a3e62793a0014c60a90b1535953eb410652f7c02fdd0fc57a8.jpg)

### Tables

![0c0d4f1ee24decba51dfdeb9d7e1b5ff49b110d1eb90922c6ad0cef8bf9ae338.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/0c0d4f1ee24decba51dfdeb9d7e1b5ff49b110d1eb90922c6ad0cef8bf9ae338.jpg)

![17962d6239306a563aa29d5ae54b093b9547afc82512523901295814966118fa.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/17962d6239306a563aa29d5ae54b093b9547afc82512523901295814966118fa.jpg)

![3b1d91e46ae760a3f8c699e9aff91dd98e496c64ff4fd67be2c49d1a6086006c.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/3b1d91e46ae760a3f8c699e9aff91dd98e496c64ff4fd67be2c49d1a6086006c.jpg)

![3dd0e4cb3cbf8aef879e4df8a89c6bc846cecc4fea34f6d8b4d0563033fb05b2.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/3dd0e4cb3cbf8aef879e4df8a89c6bc846cecc4fea34f6d8b4d0563033fb05b2.jpg)

![4095d5ed40d7ffe144d0b186f70432a04ed4a45de270af4d2022957ccb6ff76f.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/4095d5ed40d7ffe144d0b186f70432a04ed4a45de270af4d2022957ccb6ff76f.jpg)

![457a243f0ea4497233d5c1cfe716775348ee5bef0b5275f264d3eac70fe8bb6e.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/457a243f0ea4497233d5c1cfe716775348ee5bef0b5275f264d3eac70fe8bb6e.jpg)

![882b29e20ad4e9d225ea6c69d2b1bbe5c3ae42ffc7362d2ac8b8f9b9f558173e.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/882b29e20ad4e9d225ea6c69d2b1bbe5c3ae42ffc7362d2ac8b8f9b9f558173e.jpg)

![91dd9d051de6a18141e058cc713c8039c88bd2764a5823a0fd1a521623c9129f.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/91dd9d051de6a18141e058cc713c8039c88bd2764a5823a0fd1a521623c9129f.jpg)

![a3822b4c4b3459a194cf3e71fdcd32731d5cc1759ceb4c08a509727e901946a7.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/a3822b4c4b3459a194cf3e71fdcd32731d5cc1759ceb4c08a509727e901946a7.jpg)

![a676e4ed41df77adf072999738432519ac81e8e145c495ad0697c274f0ed4d37.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/a676e4ed41df77adf072999738432519ac81e8e145c495ad0697c274f0ed4d37.jpg)

![b248d5e472bf5cbd8f4d700832d09f9b0e2c50dab115fde9afff273aacada941.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/b248d5e472bf5cbd8f4d700832d09f9b0e2c50dab115fde9afff273aacada941.jpg)

![b36fb2028280d4dd27906a25572cd452d1652f4cf38ccfce276902af71676805.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/b36fb2028280d4dd27906a25572cd452d1652f4cf38ccfce276902af71676805.jpg)

![be1d32560f4b836bf2eea6e14afc334f90a7569b754fd55492b48209d4ebf247.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/be1d32560f4b836bf2eea6e14afc334f90a7569b754fd55492b48209d4ebf247.jpg)

![d8157d4eb53ebd8d8b09c21a01269785b9b2e4e157ae65681f477adbff36df7d.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/d8157d4eb53ebd8d8b09c21a01269785b9b2e4e157ae65681f477adbff36df7d.jpg)

![df245535dd56b1f595ef942efcbf777d7c418e73c7f42f252a2c4bb31fb9e01b.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/df245535dd56b1f595ef942efcbf777d7c418e73c7f42f252a2c4bb31fb9e01b.jpg)

![f48f3a531262978bb7f7581a2ec0450c61f6f9f414f9f857c48ce32ce795acdb.jpg](../icml_results/3313_AtelierEval_%20Agentic%20Evaluation%20of%20Humans%20%26%20LLMs%20as%20Text-to-Image%20Prompters/tables/f48f3a531262978bb7f7581a2ec0450c61f6f9f414f9f857c48ce32ce795acdb.jpg)

<a id="paper-3275"></a>

## Physically-Guided Data-Space Rectified Flow for Precipitation Nowcasting


### Images

![0c9f807542168ececa88b8dc773164dbfa260ce46a167c0ad76e63300d275322.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/0c9f807542168ececa88b8dc773164dbfa260ce46a167c0ad76e63300d275322.jpg)

![1b0756d71d21c52f489f417ff127fd45d46ea14e91bff74c7fe2c5c6e8715305.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/1b0756d71d21c52f489f417ff127fd45d46ea14e91bff74c7fe2c5c6e8715305.jpg)

![261c20e69033bb89f2c929bc8b41035f2c21189519e38650a13b80cc91575729.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/261c20e69033bb89f2c929bc8b41035f2c21189519e38650a13b80cc91575729.jpg)

![27c0e8346e2daa07d8c0e7749a7cf246184bb3beac923a60ca6cafaa268bc4a8.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/27c0e8346e2daa07d8c0e7749a7cf246184bb3beac923a60ca6cafaa268bc4a8.jpg)

![2d7dae4d27292486f07387041869e2386841345d207922936ea6c672bf2ef645.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/2d7dae4d27292486f07387041869e2386841345d207922936ea6c672bf2ef645.jpg)

![320a632b4a254029d939369a2c64a6fced225871043986bc34943574157adecd.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/320a632b4a254029d939369a2c64a6fced225871043986bc34943574157adecd.jpg)

![35c1caa6bac8c37439af9446c60aadcf7e67ba7e5c8fa95167946e26ae62f6d1.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/35c1caa6bac8c37439af9446c60aadcf7e67ba7e5c8fa95167946e26ae62f6d1.jpg)

![3dd5e1380a47bbeeff4a300cbc4a8717f469608c88bb25be691125fb519c580b.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/3dd5e1380a47bbeeff4a300cbc4a8717f469608c88bb25be691125fb519c580b.jpg)

![52f5d7387a5c961641c4716f407c71c38216e49e1770088ebcab2ac7c64d8cdb.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/52f5d7387a5c961641c4716f407c71c38216e49e1770088ebcab2ac7c64d8cdb.jpg)

![623e85c01207a243f3c556e591bd2519c54e0b034e464e9ec8caf281b5760d46.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/623e85c01207a243f3c556e591bd2519c54e0b034e464e9ec8caf281b5760d46.jpg)

![639cc080f6b0ec8dfe38b36c3409562d5fcde184ed8a0f695cd773732c7f6d5e.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/639cc080f6b0ec8dfe38b36c3409562d5fcde184ed8a0f695cd773732c7f6d5e.jpg)

![6c033a07b78c1b12e5e20b22d1f28a7454f9235b2f599465742f97904e656445.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/6c033a07b78c1b12e5e20b22d1f28a7454f9235b2f599465742f97904e656445.jpg)

![6d2d34537b8f40989ef2f8f2dbb17f7b511188776d1f86eeeabaaf5110b2f562.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/6d2d34537b8f40989ef2f8f2dbb17f7b511188776d1f86eeeabaaf5110b2f562.jpg)

![802774d46081fee7899afeede212b79f69c99b48c24a559503acb7f9bfcf53e3.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/802774d46081fee7899afeede212b79f69c99b48c24a559503acb7f9bfcf53e3.jpg)

![83096e89c1186958cd97dd5639fb0d48e3f0bd523d4d04467b02dc0c75712816.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/83096e89c1186958cd97dd5639fb0d48e3f0bd523d4d04467b02dc0c75712816.jpg)

![8ec084b68d51ed16105d935ba6d36cc3ba5a2de60bb21070fcfd150f325d9e5d.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/8ec084b68d51ed16105d935ba6d36cc3ba5a2de60bb21070fcfd150f325d9e5d.jpg)

![a95e9a46f53134059aa859376f03e74626473b0c3e56f1d9c28df1c8322820a8.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/a95e9a46f53134059aa859376f03e74626473b0c3e56f1d9c28df1c8322820a8.jpg)

![b05cb9a03e5204f671aa0dcb494db1f40a0d5b28ddffa0901e11a6bdaa3f766d.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/b05cb9a03e5204f671aa0dcb494db1f40a0d5b28ddffa0901e11a6bdaa3f766d.jpg)

![d03d0d0d1c253293c2ca8abcd76416bc51102c8ecaefc1f73e5fb7511d0d0d4e.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/d03d0d0d1c253293c2ca8abcd76416bc51102c8ecaefc1f73e5fb7511d0d0d4e.jpg)

![d1cf682947ba95958c4b2c5684f2c75517abc3c474e6ad5eb249c5acdef74781.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/d1cf682947ba95958c4b2c5684f2c75517abc3c474e6ad5eb249c5acdef74781.jpg)

![e2362ce822bcaee6fc67341af6c19eb90b9a43bb415d4184defa6f42eb84abed.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/e2362ce822bcaee6fc67341af6c19eb90b9a43bb415d4184defa6f42eb84abed.jpg)

![ee609774e508f7f9e6ed25886a2bc539d6d65ea9e9581a10afb8a7cbb654d664.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/ee609774e508f7f9e6ed25886a2bc539d6d65ea9e9581a10afb8a7cbb654d664.jpg)

![f0c91dd96cd8e74ba99ed61d54c71a64a1ae2cd4b3070bb2eb50f70d76abf328.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/images/f0c91dd96cd8e74ba99ed61d54c71a64a1ae2cd4b3070bb2eb50f70d76abf328.jpg)

### Tables

![01c28a560e669aeb66ad926422923b4e6ff6df60b588565fe93b31926b40d7ec.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/01c28a560e669aeb66ad926422923b4e6ff6df60b588565fe93b31926b40d7ec.jpg)

![0df4b61b394493b0430595ee61a28db81edf73488b80b1c9c4b40ece43ab7b17.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/0df4b61b394493b0430595ee61a28db81edf73488b80b1c9c4b40ece43ab7b17.jpg)

![11ec9706408e01e92c4ecf71cc264bca20aa87e90838381e0db7344e69ccbcc3.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/11ec9706408e01e92c4ecf71cc264bca20aa87e90838381e0db7344e69ccbcc3.jpg)

![2ac6700d8cde357080d69cf86dd288bdb6b0fb1547f3cea5e46773eb364323fb.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/2ac6700d8cde357080d69cf86dd288bdb6b0fb1547f3cea5e46773eb364323fb.jpg)

![3550fe0569d1f7802c10fd7d6dd3c4c621e165917a6e0dbc4f27631acbf29d34.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/3550fe0569d1f7802c10fd7d6dd3c4c621e165917a6e0dbc4f27631acbf29d34.jpg)

![4328c4f87ff0ec53fe9a075d3c3afed2bfe2eb3b0ef49729e73ab9cb7d001b46.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/4328c4f87ff0ec53fe9a075d3c3afed2bfe2eb3b0ef49729e73ab9cb7d001b46.jpg)

![5cfa2bbc90925fa61f02660544fef80a313ea5bcb2be18fbbc38c91033082197.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/5cfa2bbc90925fa61f02660544fef80a313ea5bcb2be18fbbc38c91033082197.jpg)

![7ab614d8a2cb078361105f528f30f74a506188b54eb62efcd7bb47bc7928858f.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/7ab614d8a2cb078361105f528f30f74a506188b54eb62efcd7bb47bc7928858f.jpg)

![8f1e696f751dbdf519256a0ab291d71c20c26faf538edb7643b4d34d9d370649.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/8f1e696f751dbdf519256a0ab291d71c20c26faf538edb7643b4d34d9d370649.jpg)

![e327b882d1f4dd6291794e6a1f9b737be063d8d6fca1b3dd11b3914584885d0d.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/e327b882d1f4dd6291794e6a1f9b737be063d8d6fca1b3dd11b3914584885d0d.jpg)

![ec95d762f27144d7a9f3217699f9b91be9c3d02d315dcaca2ce95de2143bd3e2.jpg](../icml_results/3314_Physically-Guided%20Data-Space%20Rectified%20Flow%20for%20Precipitation%20Nowcasting/tables/ec95d762f27144d7a9f3217699f9b91be9c3d02d315dcaca2ce95de2143bd3e2.jpg)

<a id="paper-3276"></a>

## TarGATE: Target-Aware Data Selection via Token-Attenuation Gates


### Images

![2a5ed2054788cf847d8fbed065338809f1ad60901367396016fe0c9eec08144b.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/2a5ed2054788cf847d8fbed065338809f1ad60901367396016fe0c9eec08144b.jpg)

![2e3ff0c6e60e2c86ce29bf67b06cbff0dcba5d015009836753b702b4b42ee453.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/2e3ff0c6e60e2c86ce29bf67b06cbff0dcba5d015009836753b702b4b42ee453.jpg)

![371ed890416680f0a2cfc933ed8dc2dd95371e6c7eb88e2be5ed46ec7e4be5f0.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/371ed890416680f0a2cfc933ed8dc2dd95371e6c7eb88e2be5ed46ec7e4be5f0.jpg)

![8c28c447bc6e78eb0f4abafea947b09986361a19b456b493654903e43edc49f2.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/8c28c447bc6e78eb0f4abafea947b09986361a19b456b493654903e43edc49f2.jpg)

![ae10b717a6995e964ccf555a04b3296579314a110b5c7db50141bcdf78c9d085.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/ae10b717a6995e964ccf555a04b3296579314a110b5c7db50141bcdf78c9d085.jpg)

![b0dbd5fecab0f8464b7478123349ff9521fcec5597e07c42f78d8b52038d0ca2.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/b0dbd5fecab0f8464b7478123349ff9521fcec5597e07c42f78d8b52038d0ca2.jpg)

![b99d385d97b4d2b05e17ffdb4eb1c67473118d29cff4ba50ddebf695db1717cf.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/b99d385d97b4d2b05e17ffdb4eb1c67473118d29cff4ba50ddebf695db1717cf.jpg)

![c5e108532d208cc372b0f13a75a7937af20f6c47fe62b23ac1c350d7be321c5e.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/c5e108532d208cc372b0f13a75a7937af20f6c47fe62b23ac1c350d7be321c5e.jpg)

![c88450e9213f3c585d0278a1c0e66bc81fdbb64fd0782eb1330b4929ee9b80d9.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/c88450e9213f3c585d0278a1c0e66bc81fdbb64fd0782eb1330b4929ee9b80d9.jpg)

![cf603cb292ed2e0c019a8d836470850c41bb27e9a8d2c9ff9de5eea2529e39d6.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/cf603cb292ed2e0c019a8d836470850c41bb27e9a8d2c9ff9de5eea2529e39d6.jpg)

![d0fbd06f9c50ee8e3a0c429d49f42b78fbeb6bc888ca72dcbab04b8be60c4834.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/d0fbd06f9c50ee8e3a0c429d49f42b78fbeb6bc888ca72dcbab04b8be60c4834.jpg)

![d851e0f53ed2f4832a70a07c8afeef9b449d29280f308ee117a225c5227949e3.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/d851e0f53ed2f4832a70a07c8afeef9b449d29280f308ee117a225c5227949e3.jpg)

![edc0d55d635738bce5233d06fc4e16cab7920735017dd0ba6ecc57ebe756e33e.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/images/edc0d55d635738bce5233d06fc4e16cab7920735017dd0ba6ecc57ebe756e33e.jpg)

### Tables

![19cd1e8ae93557f39046e03742d089207d7b5a79ebe367c773ddc75ab99f0754.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/19cd1e8ae93557f39046e03742d089207d7b5a79ebe367c773ddc75ab99f0754.jpg)

![59972c092028973dc998e5d058b61ef2f969e1686f78829436ea3be68af6b676.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/59972c092028973dc998e5d058b61ef2f969e1686f78829436ea3be68af6b676.jpg)

![5fa2bc8218e5d11874a5ae4245cefa20f0fb2a286a248327b5028c6d28ec5b20.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/5fa2bc8218e5d11874a5ae4245cefa20f0fb2a286a248327b5028c6d28ec5b20.jpg)

![7484067d17c769630ce060f500d9bf87c0b5244ddecc2f8958c3b4b548481d57.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/7484067d17c769630ce060f500d9bf87c0b5244ddecc2f8958c3b4b548481d57.jpg)

![dcb3c3db6b808aa4c62bd924411d5b38c7a2a8649581ddd00afa36b72eb0d90f.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/dcb3c3db6b808aa4c62bd924411d5b38c7a2a8649581ddd00afa36b72eb0d90f.jpg)

![f541e2b8d3d702e2214dd5d55449390c2447ca6ff160612d7ab8560055075f2d.jpg](../icml_results/3315_TarGATE_%20Target-Aware%20Data%20Selection%20via%20Token-Attenuation%20Gates/tables/f541e2b8d3d702e2214dd5d55449390c2447ca6ff160612d7ab8560055075f2d.jpg)

<a id="paper-3277"></a>

## Vision Language Models Cannot Reason About Physical Transformation


### Images

![0a2a582a0a1663af435ef7308ce44e060747a2855027ffe2f2d131e189eb22f3.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/0a2a582a0a1663af435ef7308ce44e060747a2855027ffe2f2d131e189eb22f3.jpg)

![1324d03d1eaffd167dfe6a0c3ee0d6f0ac2f17ae78f672317096cdfad1e07743.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/1324d03d1eaffd167dfe6a0c3ee0d6f0ac2f17ae78f672317096cdfad1e07743.jpg)

![1483be0e26479ecf755729cfc2f6c938f2ccddecce6cdfbc8ec9599777a09710.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/1483be0e26479ecf755729cfc2f6c938f2ccddecce6cdfbc8ec9599777a09710.jpg)

![1889bcfa52334028303bdbfcb42c8f73609425ca7fc8a86fc7f3dcaced79538c.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/1889bcfa52334028303bdbfcb42c8f73609425ca7fc8a86fc7f3dcaced79538c.jpg)

![279448fe6e7db5f21bcefe06ad936c0dc9ad5d7831ded68198ce936ee8256189.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/279448fe6e7db5f21bcefe06ad936c0dc9ad5d7831ded68198ce936ee8256189.jpg)

![32f4bcea822f2e0e276bcb8cd2c38aa338fd077430b53be20f92bcb985c33cea.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/32f4bcea822f2e0e276bcb8cd2c38aa338fd077430b53be20f92bcb985c33cea.jpg)

![59258ec368a26c277eac9d127aa741d68c14c9ac5dd4746c603a51456b7754e5.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/59258ec368a26c277eac9d127aa741d68c14c9ac5dd4746c603a51456b7754e5.jpg)

![67c7318b12112c5d748d265896ab779e5606857c63fa2c715b5c62d5b29c5bda.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/67c7318b12112c5d748d265896ab779e5606857c63fa2c715b5c62d5b29c5bda.jpg)

![69d32074a8626170003cc7642205e385b346ce8317988ca75c7233fa1064093b.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/69d32074a8626170003cc7642205e385b346ce8317988ca75c7233fa1064093b.jpg)

![754c8835ad65c645eb170f82f1dcc82ef9d79e419631c6023a48ca52d06aad47.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/754c8835ad65c645eb170f82f1dcc82ef9d79e419631c6023a48ca52d06aad47.jpg)

![7f2bfbca8357a3bcaeedb3491fa5b42d0f2ebc945f867d27c0d26a93cd9f176e.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/7f2bfbca8357a3bcaeedb3491fa5b42d0f2ebc945f867d27c0d26a93cd9f176e.jpg)

![8092555dd9fbd59d5e556aef1c9654d1046527ac708d18b549fd55c83ac77847.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/8092555dd9fbd59d5e556aef1c9654d1046527ac708d18b549fd55c83ac77847.jpg)

![900920026422931053a115a4efdf43286c2775aae8f19fb0949fa2cd6a65404d.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/900920026422931053a115a4efdf43286c2775aae8f19fb0949fa2cd6a65404d.jpg)

![a5352c21cd4cb201fb48b8ac52a83f7d88c54853c8dcd9c2da7b27368be2a370.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/a5352c21cd4cb201fb48b8ac52a83f7d88c54853c8dcd9c2da7b27368be2a370.jpg)

![bee1849a9bb53f35ee8b6b0f76d0967a077162a2300fcbcf7bdd037f6382dca3.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/bee1849a9bb53f35ee8b6b0f76d0967a077162a2300fcbcf7bdd037f6382dca3.jpg)

![cd0c0277a08b5551f5e9e34a41f6b19e012c7bd4c8b7f0437d29bed92ec5f05c.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/cd0c0277a08b5551f5e9e34a41f6b19e012c7bd4c8b7f0437d29bed92ec5f05c.jpg)

![d76327513325316713fa03df17a8726c9128574eb99866b777d7b3a10449e760.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/images/d76327513325316713fa03df17a8726c9128574eb99866b777d7b3a10449e760.jpg)

### Tables

![01cd14a4c6d2b121c88d003d431a18313d1823e6bb4e3dfdc2b020c819d85d2a.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/01cd14a4c6d2b121c88d003d431a18313d1823e6bb4e3dfdc2b020c819d85d2a.jpg)

![2c8ce7ab5638d3c80565d947f75dcc3d568239b789a20503a4320f1c17188864.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/2c8ce7ab5638d3c80565d947f75dcc3d568239b789a20503a4320f1c17188864.jpg)

![640365995cde0c2bb9105618e54a6386bf2f960ad1114eee0916f6c95dcf18b4.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/640365995cde0c2bb9105618e54a6386bf2f960ad1114eee0916f6c95dcf18b4.jpg)

![a84a39880950b6417cab28c7376daae43a3c4e0a7f1eae0df78c651c059ff32d.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/a84a39880950b6417cab28c7376daae43a3c4e0a7f1eae0df78c651c059ff32d.jpg)

![cba5a645cfe3cc952f1d51bcbf164245d9bb95afa1eb4fe501350152039f923b.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/cba5a645cfe3cc952f1d51bcbf164245d9bb95afa1eb4fe501350152039f923b.jpg)

![fc90146b8d63c273585797d837720edacd0e5f3c3269d5389a9ced0bb37eb317.jpg](../icml_results/3316_Vision%20Language%20Models%20Cannot%20Reason%20About%20Physical%20Transformation/tables/fc90146b8d63c273585797d837720edacd0e5f3c3269d5389a9ced0bb37eb317.jpg)

<a id="paper-3278"></a>

## DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs


### Images

![024487112d82509baf5747813678e86945280c191794f6fa3d80cbab0863a672.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/024487112d82509baf5747813678e86945280c191794f6fa3d80cbab0863a672.jpg)

![0f6e0e437164f4f215c83547361799b87a3e593f7bf673f0e5f9c48131435a3b.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/0f6e0e437164f4f215c83547361799b87a3e593f7bf673f0e5f9c48131435a3b.jpg)

![1041f4154c749658edbd61d7165773ee0cd31b4cc5d2549b73d184cbc8fcbb47.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/1041f4154c749658edbd61d7165773ee0cd31b4cc5d2549b73d184cbc8fcbb47.jpg)

![404ea6aba69277cb238aa059896590728879b8c679bb9e3713aa7cc54148f23e.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/404ea6aba69277cb238aa059896590728879b8c679bb9e3713aa7cc54148f23e.jpg)

![4cc30f7d05a59c82e39b4072b2f75e2a60ddb1796bc2bb737a2e520b4166f019.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/4cc30f7d05a59c82e39b4072b2f75e2a60ddb1796bc2bb737a2e520b4166f019.jpg)

![5af04b2e178f5eac35dc9dd987bc18d9d83c2ef1efadcb4c95eeb899e1e3deef.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/5af04b2e178f5eac35dc9dd987bc18d9d83c2ef1efadcb4c95eeb899e1e3deef.jpg)

![94937292a39d527409a31680c1d2bdf73a8f17fda26412e91b18bfc84a57bfbb.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/94937292a39d527409a31680c1d2bdf73a8f17fda26412e91b18bfc84a57bfbb.jpg)

![9f6abfd33baa3d9a244c827f9e93dce3d0a1d7f9387f2179ca397ed619d4b6db.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/9f6abfd33baa3d9a244c827f9e93dce3d0a1d7f9387f2179ca397ed619d4b6db.jpg)

![c17908e2d94e02f1fbc8fb8324680ffc1178a78eabd42c0166b0633cdcab59a6.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/images/c17908e2d94e02f1fbc8fb8324680ffc1178a78eabd42c0166b0633cdcab59a6.jpg)

### Tables

![af7daac6b9f02c0ec8735918a441368ecdfea2a3a07baa014a01cb7258cb1858.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/tables/af7daac6b9f02c0ec8735918a441368ecdfea2a3a07baa014a01cb7258cb1858.jpg)

![b38bed442679be44b1e8a8c189dcf365c357c0362e87282f023c37ca61f74714.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/tables/b38bed442679be44b1e8a8c189dcf365c357c0362e87282f023c37ca61f74714.jpg)

![c32bc06acdc7af6545759fa4a94ac1ab1b2d55a06c7324723cae658a90db9047.jpg](../icml_results/3317_DSB_%20Dynamic%20Sliding%20Block%20Scheduling%20for%20Diffusion%20LLMs/tables/c32bc06acdc7af6545759fa4a94ac1ab1b2d55a06c7324723cae658a90db9047.jpg)

<a id="paper-3279"></a>

## IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning


### Images

![26907fe0141a28ca56c585a522a1a33a2518fa7b9f581840da34779d4c51c03c.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/26907fe0141a28ca56c585a522a1a33a2518fa7b9f581840da34779d4c51c03c.jpg)

![38c848af352d317a3d87189d2662908fcdbb80eeb30d208a4221c98c37c1d183.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/38c848af352d317a3d87189d2662908fcdbb80eeb30d208a4221c98c37c1d183.jpg)

![417fed372ebe640da9244f16f572083331a9a020de018fbe929ee698cee1e8f4.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/417fed372ebe640da9244f16f572083331a9a020de018fbe929ee698cee1e8f4.jpg)

![51a36366c704f0ee583e24cb2f84c74c92f4ef95da9429ddb9ea2c57e874a862.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/51a36366c704f0ee583e24cb2f84c74c92f4ef95da9429ddb9ea2c57e874a862.jpg)

![53dfeb41c6085b4f99261ea19321dcde8358bbea0e9909d7c87a6f73ecbf652c.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/53dfeb41c6085b4f99261ea19321dcde8358bbea0e9909d7c87a6f73ecbf652c.jpg)

![563b721dbf67beb9326409a229e443c23eeed66b0f16e731f9f55ba545153058.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/563b721dbf67beb9326409a229e443c23eeed66b0f16e731f9f55ba545153058.jpg)

![6eaaa4b84ae462bea416e413054249636727582a2431ad8b1d19cc2b81afaf43.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/6eaaa4b84ae462bea416e413054249636727582a2431ad8b1d19cc2b81afaf43.jpg)

![73a14793821a1a9a1bc43a851746188518929322d2d42610aaa4ab57b86e20a3.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/73a14793821a1a9a1bc43a851746188518929322d2d42610aaa4ab57b86e20a3.jpg)

![7fc19897f79b33783ff5cfab429e023ecd23d782b5aee58f9bed718266113820.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/7fc19897f79b33783ff5cfab429e023ecd23d782b5aee58f9bed718266113820.jpg)

![961ca729c3ef567a14987e46b5c7124bbaa44057a9faa1cd52bddcd6b997ec3e.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/961ca729c3ef567a14987e46b5c7124bbaa44057a9faa1cd52bddcd6b997ec3e.jpg)

![cd1f1676440a160f922607c177bb4fea3496cc3bc7074c9ec75142e0033e35fa.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/images/cd1f1676440a160f922607c177bb4fea3496cc3bc7074c9ec75142e0033e35fa.jpg)

### Tables

![172f029a676b53ea93b9925a0717007912a49e9b2eea1da8159c80d788ded8d0.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/172f029a676b53ea93b9925a0717007912a49e9b2eea1da8159c80d788ded8d0.jpg)

![20aa584495d5befc84fe65d9c5d9229a58f295050e7631f055348ff245a421aa.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/20aa584495d5befc84fe65d9c5d9229a58f295050e7631f055348ff245a421aa.jpg)

![31fe95a2dd2bd63f96fa038994ace92a608954d1da76004d2ad57965ca19d038.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/31fe95a2dd2bd63f96fa038994ace92a608954d1da76004d2ad57965ca19d038.jpg)

![491acef24274342aa1eef397011cfa443e4c9b3563dfdbf40200bc24c114b4fd.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/491acef24274342aa1eef397011cfa443e4c9b3563dfdbf40200bc24c114b4fd.jpg)

![4d35a3e973a749465f979c16f4b8f6ef28ee2f6499fa99e9a332d799c91df670.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/4d35a3e973a749465f979c16f4b8f6ef28ee2f6499fa99e9a332d799c91df670.jpg)

![4e12a07b6663befedf26fcc5da3ec0f3d8c82c97c607afc0ba4f9254e187cf45.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/4e12a07b6663befedf26fcc5da3ec0f3d8c82c97c607afc0ba4f9254e187cf45.jpg)

![697aabea1e56353eaa832f1fa7dc7841d826836eee4b978d402cfb412c6bba36.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/697aabea1e56353eaa832f1fa7dc7841d826836eee4b978d402cfb412c6bba36.jpg)

![d046f4fe17bd97dd7371e8e7d846a64cbeeff989c4ba95f659fa67f4e1f19509.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/d046f4fe17bd97dd7371e8e7d846a64cbeeff989c4ba95f659fa67f4e1f19509.jpg)

![d1f636cb8d35cd49f722aaad3994a094a1d61df7daf77481c435ad2d214eb6eb.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/d1f636cb8d35cd49f722aaad3994a094a1d61df7daf77481c435ad2d214eb6eb.jpg)

![e8df1ab0e0b291a44fe3a9a644f6bd25bbd21ce95b355cb4a76620f2625c0c11.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/e8df1ab0e0b291a44fe3a9a644f6bd25bbd21ce95b355cb4a76620f2625c0c11.jpg)

![ecddf7a2fda4a374fa69c4812f723a4fd5148450ace123a687656ff7ac663cd3.jpg](../icml_results/3318_IntentRL_%20Training%20Proactive%20User-intent%20Agents%20for%20Open-ended%20Deep%20Research%20via%20Reinforcement%20Learn/tables/ecddf7a2fda4a374fa69c4812f723a4fd5148450ace123a687656ff7ac663cd3.jpg)

<a id="paper-3280"></a>

## Task-and-Model-Aware Fractal-Consistency for Efficient LLM Reasoning


### Images

![381b1016254f932f0bb82f8ff834b7f9f86ee44cb99460f9a92e7fcb8b0901fe.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/381b1016254f932f0bb82f8ff834b7f9f86ee44cb99460f9a92e7fcb8b0901fe.jpg)

![42ff6c8f3ca7fa97edfc66595f8050fb8c95259bd5d4df80e11db5a0b88b9fdb.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/42ff6c8f3ca7fa97edfc66595f8050fb8c95259bd5d4df80e11db5a0b88b9fdb.jpg)

![5e3494af9d3ccbaa16f169b6b05feac2146b8e657972441be47a95a6480457fd.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/5e3494af9d3ccbaa16f169b6b05feac2146b8e657972441be47a95a6480457fd.jpg)

![69c930cb323de7147d58ce9707fb60c9d35ae4b22eb2f8aa3682fd4f3c373c23.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/69c930cb323de7147d58ce9707fb60c9d35ae4b22eb2f8aa3682fd4f3c373c23.jpg)

![a15866d5c50897d885ed54cb7ab0b53e37c7d5e5edb30e48e1deb5af8340b9a0.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/a15866d5c50897d885ed54cb7ab0b53e37c7d5e5edb30e48e1deb5af8340b9a0.jpg)

![b4fd57b4956beb595e8d18ae6b935695fd7eb00ea2d1a02f0c5ec89b946e0c8f.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/images/b4fd57b4956beb595e8d18ae6b935695fd7eb00ea2d1a02f0c5ec89b946e0c8f.jpg)

### Tables

![239d2375393e6af0a28250d4b90c9e56cd3d9994988cfa4c0f26cf706e641dcf.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/239d2375393e6af0a28250d4b90c9e56cd3d9994988cfa4c0f26cf706e641dcf.jpg)

![5b481e46b9e44946e21646500579e2e9f736d4b765f1f544d3102f845de0f043.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/5b481e46b9e44946e21646500579e2e9f736d4b765f1f544d3102f845de0f043.jpg)

![7cac1a40a1a39454fdb672a1026c87c83abd1096b60207f10971a1ac4a532d55.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/7cac1a40a1a39454fdb672a1026c87c83abd1096b60207f10971a1ac4a532d55.jpg)

![8ce8162d7b2eafd57f9166c815331f6d6cfe33cfe803277f9c1c47df265f4b13.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/8ce8162d7b2eafd57f9166c815331f6d6cfe33cfe803277f9c1c47df265f4b13.jpg)

![9b017b84f99da1924ab7ebfce27c7595811fdfb26204e1806c8daa59a72cab14.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/9b017b84f99da1924ab7ebfce27c7595811fdfb26204e1806c8daa59a72cab14.jpg)

![e903a2a14805b0489dfc748cfcabbc2e633ecf81ddc67d5cb3ac43c2eeffec1b.jpg](../icml_results/3319_Task-and-Model-Aware%20Fractal-Consistency%20for%20Efficient%20LLM%20Reasoning/tables/e903a2a14805b0489dfc748cfcabbc2e633ecf81ddc67d5cb3ac43c2eeffec1b.jpg)

<a id="paper-3281"></a>

## TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins


### Images

![31ef82d9f8a4f62cdaa9d7cc68140c13e28e8aadd208c351c4c8490955ebb804.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/31ef82d9f8a4f62cdaa9d7cc68140c13e28e8aadd208c351c4c8490955ebb804.jpg)

![64cbd0881469b241212a7a662a9a46a73a2b9a972deedd39f4a2c3c778a69a1f.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/64cbd0881469b241212a7a662a9a46a73a2b9a972deedd39f4a2c3c778a69a1f.jpg)

![69fbb76124018b79c1d46cc99d7e71bf8f7255d2a472357027cc0f546d82d2c7.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/69fbb76124018b79c1d46cc99d7e71bf8f7255d2a472357027cc0f546d82d2c7.jpg)

![83136f8e66bd80cf22c07bc1cfcbdd17bf200bf480087c23891304aaf8137a29.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/83136f8e66bd80cf22c07bc1cfcbdd17bf200bf480087c23891304aaf8137a29.jpg)

![87d6014425f284e4d734c819e117f6e33816839d5b19d1a7a9e3cb6f36b9bd36.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/87d6014425f284e4d734c819e117f6e33816839d5b19d1a7a9e3cb6f36b9bd36.jpg)

![a25dfef48e7b5be235b2c04b8a6f50faf852f8b9e5298f6c2f60535adf0a491e.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/a25dfef48e7b5be235b2c04b8a6f50faf852f8b9e5298f6c2f60535adf0a491e.jpg)

![b29c6b5ade37a395ec992bc9a3e7fcdd1068623d1c29e4b17931e833cb05422b.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/b29c6b5ade37a395ec992bc9a3e7fcdd1068623d1c29e4b17931e833cb05422b.jpg)

![c0b307363e975d410cba102c632618206254ed6eac72ff2cea9f3d2b70671f86.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/c0b307363e975d410cba102c632618206254ed6eac72ff2cea9f3d2b70671f86.jpg)

![f1bec7a60ad1ab5fa0c55f6b126af2a04afbf240c0fdedf4ad888f3fedc36516.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/f1bec7a60ad1ab5fa0c55f6b126af2a04afbf240c0fdedf4ad888f3fedc36516.jpg)

![f647b3dbdba3e4b4027a66b8711bf594f3fe7e8642b9c829d72a0c4ef1ce1226.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/images/f647b3dbdba3e4b4027a66b8711bf594f3fe7e8642b9c829d72a0c4ef1ce1226.jpg)

### Tables

![04e298baa73b87fd2c9968fa54b165b3d933841b97ad85cc1725e5dba6b92929.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/04e298baa73b87fd2c9968fa54b165b3d933841b97ad85cc1725e5dba6b92929.jpg)

![0f3f572cd70e1a6cfc8ae40dcb32e1e19d212002576794d148273e96aa1d24f3.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/0f3f572cd70e1a6cfc8ae40dcb32e1e19d212002576794d148273e96aa1d24f3.jpg)

![14305208b47f055b14ed4f9289d0807e99d80c748ed69ee34e53372eb8b3fb62.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/14305208b47f055b14ed4f9289d0807e99d80c748ed69ee34e53372eb8b3fb62.jpg)

![3e09ff92203fdbb7df3e5067641443bf68bbd1b9515ecfa61264c2994c45a34c.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/3e09ff92203fdbb7df3e5067641443bf68bbd1b9515ecfa61264c2994c45a34c.jpg)

![554062e1713b009cd064c7dafe6d8436635d1367abf85c22a2d4a42b2f8f8289.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/554062e1713b009cd064c7dafe6d8436635d1367abf85c22a2d4a42b2f8f8289.jpg)

![a5ecaca8fb511b5dbeb6467fb2e02d2e80481e8d6899d62bce1498ad760aed3e.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/a5ecaca8fb511b5dbeb6467fb2e02d2e80481e8d6899d62bce1498ad760aed3e.jpg)

![b07fc7efed201cd0a6bd254d42f4e3e91e7c1b5cd53888ea3b9dcacdc80d3006.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/b07fc7efed201cd0a6bd254d42f4e3e91e7c1b5cd53888ea3b9dcacdc80d3006.jpg)

![b7b62dcd341777b39a30bdc75be7543af4986c6a89200385c95223ef8b1c81da.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/b7b62dcd341777b39a30bdc75be7543af4986c6a89200385c95223ef8b1c81da.jpg)

![e3e747dcfa7cd2bf353b09686248381f799fc3d5c2d45662279160164ca6d355.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/e3e747dcfa7cd2bf353b09686248381f799fc3d5c2d45662279160164ca6d355.jpg)

![f3908032bdc40480ed1ac442f24a7bcb01237d7031f70c0499c2826071ff32e1.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/f3908032bdc40480ed1ac442f24a7bcb01237d7031f70c0499c2826071ff32e1.jpg)

![f9b9b79ee6c113b824c25dd7b6a05916ee8ec7aa089620c8fbf4bc8e26f6d4ad.jpg](../icml_results/3320_TuneAhead_%20Predicting%20Fine-tuning%20Performance%20Before%20Full%20Training%20Begins/tables/f9b9b79ee6c113b824c25dd7b6a05916ee8ec7aa089620c8fbf4bc8e26f6d4ad.jpg)

<a id="paper-3282"></a>

## SimGFM: Simplifying Discrete Flow Matching for Graph Generation


### Images

![22c77069636db3c936bf86b7e60fcec4cf6b574982f967c53ab3de80a6aef95b.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/22c77069636db3c936bf86b7e60fcec4cf6b574982f967c53ab3de80a6aef95b.jpg)

![2af893c009811651134a6a6e03cedddeee2df9dacb37ec3011912de8ce198bbb.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/2af893c009811651134a6a6e03cedddeee2df9dacb37ec3011912de8ce198bbb.jpg)

![504ee2ea9b91a9dd7a9370f920afcc229246c7b5baf2cb095e96c2b99f73fec4.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/504ee2ea9b91a9dd7a9370f920afcc229246c7b5baf2cb095e96c2b99f73fec4.jpg)

![7615f3bcbade802567001e91a7ad4818e00f46051eec5db5068ec0af8e45ed02.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/7615f3bcbade802567001e91a7ad4818e00f46051eec5db5068ec0af8e45ed02.jpg)

![888cd4c660cc90e36a21187ac24135e5d0a05d3d7d7e6ad43adca5473a3424fa.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/888cd4c660cc90e36a21187ac24135e5d0a05d3d7d7e6ad43adca5473a3424fa.jpg)

![b66ec5a9d151ed97d0b00d9ff150104472b7685d62d8023aac921ad0eb73428c.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/b66ec5a9d151ed97d0b00d9ff150104472b7685d62d8023aac921ad0eb73428c.jpg)

![c1f0d2a4708b76fd4d7b3d06261b22c9959ae128ea7bb02e2ed86995d00d44c5.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/c1f0d2a4708b76fd4d7b3d06261b22c9959ae128ea7bb02e2ed86995d00d44c5.jpg)

![ec94b1f0e561526479685f2b5f86624c2bec2fe7da7a40aaf90f85dad13e8d5f.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/images/ec94b1f0e561526479685f2b5f86624c2bec2fe7da7a40aaf90f85dad13e8d5f.jpg)

### Tables

![2b9ee1e8946d409190f023eaccf88c3927cbe160abfd83d0df0fc7d53fe2e350.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/2b9ee1e8946d409190f023eaccf88c3927cbe160abfd83d0df0fc7d53fe2e350.jpg)

![36797a6a9ef8dfaefeb3b8b4038bfeedf3add118d592852bd53d7fd0ab6b1b1a.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/36797a6a9ef8dfaefeb3b8b4038bfeedf3add118d592852bd53d7fd0ab6b1b1a.jpg)

![393431bfd3e88457c0aceda23a1ebd4c59e33dcf8ce9d1c6aaa180a5c12bc16c.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/393431bfd3e88457c0aceda23a1ebd4c59e33dcf8ce9d1c6aaa180a5c12bc16c.jpg)

![416dfadef86701585cf855baa5d21bb8b496cfcd82424a25110f1169bff2a47f.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/416dfadef86701585cf855baa5d21bb8b496cfcd82424a25110f1169bff2a47f.jpg)

![43647a67010ed2a65ee9f0a8bbc11991b3db647c25fab10062ce4e4f266285c6.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/43647a67010ed2a65ee9f0a8bbc11991b3db647c25fab10062ce4e4f266285c6.jpg)

![465c438d703f30a13a62e836aeaa098dcf4da505c514fc16d9beb98b520e6a18.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/465c438d703f30a13a62e836aeaa098dcf4da505c514fc16d9beb98b520e6a18.jpg)

![5826d9b85b8afdecd5a9c554e113b537968101bfdab33648317144ef6cf110d1.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/5826d9b85b8afdecd5a9c554e113b537968101bfdab33648317144ef6cf110d1.jpg)

![5921f19e62194ab21b7f9ef2c41c371b96e2eac3df656a89013455db6c9f1e01.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/5921f19e62194ab21b7f9ef2c41c371b96e2eac3df656a89013455db6c9f1e01.jpg)

![5cae5ce70ad750216379c0e0cc48bf1475c6ade87efc2abd9831ba7cc52707b3.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/5cae5ce70ad750216379c0e0cc48bf1475c6ade87efc2abd9831ba7cc52707b3.jpg)

![708650f15efc4c5d8765e40533e1fdf748f625ea72a4eb2c279d94280fb1dfb3.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/708650f15efc4c5d8765e40533e1fdf748f625ea72a4eb2c279d94280fb1dfb3.jpg)

![743ab3c21eef88ffc4be8f4d90b38b9070425b16ab508b7e8c0a205ae91d2ba3.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/743ab3c21eef88ffc4be8f4d90b38b9070425b16ab508b7e8c0a205ae91d2ba3.jpg)

![89ff87179930b3bf41c98229e81ff0080edbefadbb8b6129559f99049b1bc9b2.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/89ff87179930b3bf41c98229e81ff0080edbefadbb8b6129559f99049b1bc9b2.jpg)

![8d79e5fe29632d2059ea7589d42bce504a26060655f4253659941c9060c36324.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/8d79e5fe29632d2059ea7589d42bce504a26060655f4253659941c9060c36324.jpg)

![b77912197a0459365e2bc0d31d4293327a62a1273113991058489f55ba9dee77.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/b77912197a0459365e2bc0d31d4293327a62a1273113991058489f55ba9dee77.jpg)

![dbfcb1e6c19122e3379d3ef7f7e97b1c46c2ed27e8d5bca26169d042b801232f.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/dbfcb1e6c19122e3379d3ef7f7e97b1c46c2ed27e8d5bca26169d042b801232f.jpg)

![e707ea49ea29ca9ff12593b7899ec23668cb966f67b194fd41c654bb4d65e5e3.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/e707ea49ea29ca9ff12593b7899ec23668cb966f67b194fd41c654bb4d65e5e3.jpg)

![fd2e57be8645aca306f1054454effdb2634e8ae3b7807e6de8645813bbd608e5.jpg](../icml_results/3321_SimGFM_%20Simplifying%20Discrete%20Flow%20Matching%20for%20Graph%20Generation/tables/fd2e57be8645aca306f1054454effdb2634e8ae3b7807e6de8645813bbd608e5.jpg)

<a id="paper-3283"></a>

## CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding


### Images

![182da21e0bc80e860104a9edb352506cf3be2a452d7b8221410c0d421375e2cd.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/182da21e0bc80e860104a9edb352506cf3be2a452d7b8221410c0d421375e2cd.jpg)

![18a8625753f048ebe46d1bfdf0f816203a9a01a63610116a48c5bf93fe9d0d80.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/18a8625753f048ebe46d1bfdf0f816203a9a01a63610116a48c5bf93fe9d0d80.jpg)

![19f4bd729f891269be557b8c03f1e0045ca3cf3ccbf3f027f2a10bcb252c270e.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/19f4bd729f891269be557b8c03f1e0045ca3cf3ccbf3f027f2a10bcb252c270e.jpg)

![1c6b7e4399106799b6ac157e364ebc5859dad0c6cdc8f76b45ace3efda31cecf.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/1c6b7e4399106799b6ac157e364ebc5859dad0c6cdc8f76b45ace3efda31cecf.jpg)

![3e0c5fbcc437e497f23919a5bebf1adc73e8cc2049628ccb4767e83eb31e9126.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/3e0c5fbcc437e497f23919a5bebf1adc73e8cc2049628ccb4767e83eb31e9126.jpg)

![578fd83c13ab624d49621dff81bbb49cd0dd0800bb9a1b7ce1242faf916501a2.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/578fd83c13ab624d49621dff81bbb49cd0dd0800bb9a1b7ce1242faf916501a2.jpg)

![6144d411cd705670573be3e34eb2e779c4837cf6d6f1eafe728327938c135e51.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/6144d411cd705670573be3e34eb2e779c4837cf6d6f1eafe728327938c135e51.jpg)

![a675eee1d483d66f53d44cc35832277a646e1cea0358d982e2a8d5daade8302d.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/a675eee1d483d66f53d44cc35832277a646e1cea0358d982e2a8d5daade8302d.jpg)

![bebf37cf7ae3e87d90d3f2c8599c4a127e36dcb90d1b8970182823cfb2657286.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/bebf37cf7ae3e87d90d3f2c8599c4a127e36dcb90d1b8970182823cfb2657286.jpg)

![c7d0a2c9aa39d791321b0f288c7fd0e6dc49cbc53800a2c15d14a67b115aa445.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/c7d0a2c9aa39d791321b0f288c7fd0e6dc49cbc53800a2c15d14a67b115aa445.jpg)

![c83d6c327a6f921378cbe38765e1ab61dde88e008ca0494ef44aca71130d35c0.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/c83d6c327a6f921378cbe38765e1ab61dde88e008ca0494ef44aca71130d35c0.jpg)

![c91a4327770df1dde56cc91ce537890033f95a04b6aa6c5ef6de62b7b91126c4.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/c91a4327770df1dde56cc91ce537890033f95a04b6aa6c5ef6de62b7b91126c4.jpg)

![d902e6a06677d21861a31db0b3f2d467646dca2ecb9f0be2bf9099089af7a954.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/d902e6a06677d21861a31db0b3f2d467646dca2ecb9f0be2bf9099089af7a954.jpg)

![f585b3f72bbbde87b9b34570b8529b6000d0347612f73e58d77642f55842a052.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/f585b3f72bbbde87b9b34570b8529b6000d0347612f73e58d77642f55842a052.jpg)

![f9b94994d8de9febf3a8c793c9772951a77620ad4729244dc5aa354a165b28e3.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/images/f9b94994d8de9febf3a8c793c9772951a77620ad4729244dc5aa354a165b28e3.jpg)

### Tables

![04fc7d6cbf166410962814bac8bd10140e2a18360a560731e4291735ef9f2467.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/04fc7d6cbf166410962814bac8bd10140e2a18360a560731e4291735ef9f2467.jpg)

![0ded7413416650d902a20d299a6bc0d6e6fea54a8c3ffb28e102a08f09dd5007.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/0ded7413416650d902a20d299a6bc0d6e6fea54a8c3ffb28e102a08f09dd5007.jpg)

![16900958c5382f9b88c2943a5cfa5a1b8b5781cdffb2e1193c6b92e8159f8288.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/16900958c5382f9b88c2943a5cfa5a1b8b5781cdffb2e1193c6b92e8159f8288.jpg)

![32d804e480e45c8de6c0d1c246f0364c39004539d9275074c84e7ee55e5d7bc9.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/32d804e480e45c8de6c0d1c246f0364c39004539d9275074c84e7ee55e5d7bc9.jpg)

![3e3a0ba8f1ad834d1701b140cc593cc2ed32a26d53916888791c4c9170091352.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/3e3a0ba8f1ad834d1701b140cc593cc2ed32a26d53916888791c4c9170091352.jpg)

![504f7d693814ad0b075394e683a1769fce89c8c71510117911d583c0e73ec086.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/504f7d693814ad0b075394e683a1769fce89c8c71510117911d583c0e73ec086.jpg)

![6143ccb21232ccbd4654ef86e278eb612179fe47a73243bd785af73613263e86.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/6143ccb21232ccbd4654ef86e278eb612179fe47a73243bd785af73613263e86.jpg)

![8f8e82e7abf7fc9e4cedfc37011e073008f2326386d87be330f7b206f858da1a.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/8f8e82e7abf7fc9e4cedfc37011e073008f2326386d87be330f7b206f858da1a.jpg)

![ace9a54802bcf708eab7e88c1e6f82e4a4859a5cc61baac9745dcd66c6fa105b.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/ace9a54802bcf708eab7e88c1e6f82e4a4859a5cc61baac9745dcd66c6fa105b.jpg)

![bfcc0f11dabf8d4181c9f4ce84d138894ecffc173140253ba82a250fe167377e.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/bfcc0f11dabf8d4181c9f4ce84d138894ecffc173140253ba82a250fe167377e.jpg)

![dc22bd59b145b65823ba6abc566763ffbaaa76a491fbee9a8afc3dbd67f38610.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/dc22bd59b145b65823ba6abc566763ffbaaa76a491fbee9a8afc3dbd67f38610.jpg)

![ecf529f2efdb479411a295721afe310d6246b99b5f4abd50a2413da73c28d1dc.jpg](../icml_results/3322_CentaurEval_%20Benchmarking%20Human-in-the-Loop%20Value%20in%20Agentic%20Coding/tables/ecf529f2efdb479411a295721afe310d6246b99b5f4abd50a2413da73c28d1dc.jpg)

<a id="paper-3284"></a>

## Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization


### Images

![0c7c60e2add4cc0f975ac3be9e1e7e16bb7c0f358322f6a894e11a65ad10afbd.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/0c7c60e2add4cc0f975ac3be9e1e7e16bb7c0f358322f6a894e11a65ad10afbd.jpg)

![15865e41e5c5418f87e4fb0c421334e2866ce6a108b26dca1d9865cc465a3dc4.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/15865e41e5c5418f87e4fb0c421334e2866ce6a108b26dca1d9865cc465a3dc4.jpg)

![2b7aecabf9f165cf793a80b176fc1fba363b3e61dfb862185e2e3a6d863a17f9.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/2b7aecabf9f165cf793a80b176fc1fba363b3e61dfb862185e2e3a6d863a17f9.jpg)

![33a7d479add18f8843bb5e0063fa31e4a5f1057cf1ac19590d9f148b9af7bb36.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/33a7d479add18f8843bb5e0063fa31e4a5f1057cf1ac19590d9f148b9af7bb36.jpg)

![3fc9f832e2db8e20a0108ff29ba8e50d704602bdcfad0d68dd25a67507f70c09.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/3fc9f832e2db8e20a0108ff29ba8e50d704602bdcfad0d68dd25a67507f70c09.jpg)

![470a59ab57ba6775d51eecfd0929561bd6622a01c25a93a5608065aab48b7836.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/470a59ab57ba6775d51eecfd0929561bd6622a01c25a93a5608065aab48b7836.jpg)

![66476bf9fbf73be19aafc88fefc949acb2787d17db3cf24f60319bf98b765bac.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/66476bf9fbf73be19aafc88fefc949acb2787d17db3cf24f60319bf98b765bac.jpg)

![68cc967542e21e21ac471a03de8090e472d3ba29eb0888188004579442ef404d.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/68cc967542e21e21ac471a03de8090e472d3ba29eb0888188004579442ef404d.jpg)

![6fe44f2dd9b4ac4514a54dd05e61e1f42f1686659d0640e889dc7372225117f1.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/6fe44f2dd9b4ac4514a54dd05e61e1f42f1686659d0640e889dc7372225117f1.jpg)

![7471aaf2dfc1bd442a668596b398c9a68a8b9a229435fd83045c6a724fca4845.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/7471aaf2dfc1bd442a668596b398c9a68a8b9a229435fd83045c6a724fca4845.jpg)

![78de6177e20e8cf12f6237fc689f1a26e90c45b35ea9338057474b7d571f1b76.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/78de6177e20e8cf12f6237fc689f1a26e90c45b35ea9338057474b7d571f1b76.jpg)

![7de7ecee2e9d7a88867865d9e02cdaf35064ba2a793d71b7d87e5a7e09842535.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/7de7ecee2e9d7a88867865d9e02cdaf35064ba2a793d71b7d87e5a7e09842535.jpg)

![7e4f2da2b92bee8ce6554d7a2b00cc7a7f8afb66d121f47fc730775d0a5012b1.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/7e4f2da2b92bee8ce6554d7a2b00cc7a7f8afb66d121f47fc730775d0a5012b1.jpg)

![80e20ba510ea72b47bf5f7bbf36a7bcd59b229179c1e8d5a4262691c86c53893.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/80e20ba510ea72b47bf5f7bbf36a7bcd59b229179c1e8d5a4262691c86c53893.jpg)

![8107b4e59688b6bf20ba40f1fd21196770bbba1bad65e5e14afbb9e1cc0d6d2b.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/8107b4e59688b6bf20ba40f1fd21196770bbba1bad65e5e14afbb9e1cc0d6d2b.jpg)

![a3bd942c285a90b5dff66216ee2aa7c9b59025979e439ace9c6051014bec8a06.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/a3bd942c285a90b5dff66216ee2aa7c9b59025979e439ace9c6051014bec8a06.jpg)

![bb0c7521aa76daa4718acc5ff0554464b1f31dbfbd3056a176cc74c7eec6aa74.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/bb0c7521aa76daa4718acc5ff0554464b1f31dbfbd3056a176cc74c7eec6aa74.jpg)

![bb23abee31d7afb349ed53f6c18f891124a7513c6056fa44e4d16a79bdd380ae.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/bb23abee31d7afb349ed53f6c18f891124a7513c6056fa44e4d16a79bdd380ae.jpg)

![cebc0334ef83fafcf189cfcb8143e4ec6c8b50fce00a0f3fdfb24b95148143a8.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/cebc0334ef83fafcf189cfcb8143e4ec6c8b50fce00a0f3fdfb24b95148143a8.jpg)

![e27080f8e541042b700cff308dd1e4bb560cf0174b132f65179f94f3cf322ecd.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/e27080f8e541042b700cff308dd1e4bb560cf0174b132f65179f94f3cf322ecd.jpg)

![f0616956b7b7443a807c2d531bed427b864c5f9940c10340ba3154694703187d.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/f0616956b7b7443a807c2d531bed427b864c5f9940c10340ba3154694703187d.jpg)

![fcb1758d9a67ad4d0050b7dbc5ab52df4c40293ce3752639645f554dd22bb67c.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/images/fcb1758d9a67ad4d0050b7dbc5ab52df4c40293ce3752639645f554dd22bb67c.jpg)

### Tables

![0057937ed51a55e3182b44d6410b034facc9733ba067562dc6850d901baf7c70.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/0057937ed51a55e3182b44d6410b034facc9733ba067562dc6850d901baf7c70.jpg)

![0d7fe6a9ecd164772f4fdf955e668e1487b971e748baad79348cd856cf3dba7e.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/0d7fe6a9ecd164772f4fdf955e668e1487b971e748baad79348cd856cf3dba7e.jpg)

![583ebea6c113a5e951830a0c518df99a5da3ab4be793abdd4036a4e3850c4ed5.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/583ebea6c113a5e951830a0c518df99a5da3ab4be793abdd4036a4e3850c4ed5.jpg)

![6570e905a9dbec43c8f2574a6a9f880742148df9624405ee784f34e8f5d07b65.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/6570e905a9dbec43c8f2574a6a9f880742148df9624405ee784f34e8f5d07b65.jpg)

![757d16d58530499a690d0849dcf155d712318bac19fd4651d03e0d279bb7a839.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/757d16d58530499a690d0849dcf155d712318bac19fd4651d03e0d279bb7a839.jpg)

![9f8de4497f58e7fdf5bdd4d02bb7244f17d10f662039d962aa4c382c9094b7d5.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/9f8de4497f58e7fdf5bdd4d02bb7244f17d10f662039d962aa4c382c9094b7d5.jpg)

![a0e6e5edab52947bc59a8e0c20fecf69a04b3b2ec330bcf47783f70c9f9705ef.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/a0e6e5edab52947bc59a8e0c20fecf69a04b3b2ec330bcf47783f70c9f9705ef.jpg)

![a81f115aeb9501b83546917e5c6cabeac20bae48791f8356acb458fa03465488.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/a81f115aeb9501b83546917e5c6cabeac20bae48791f8356acb458fa03465488.jpg)

![a8a868504879d0dbb973ec1478acc5b2f48931fffefb43c5e2b3e81b5c1fb0d0.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/a8a868504879d0dbb973ec1478acc5b2f48931fffefb43c5e2b3e81b5c1fb0d0.jpg)

![f7da25d4a385dda5e4adb96a0b6dcd2efa287de0b91760c38d517c826b9304cd.jpg](../icml_results/3323_Principled%20RL%20for%20Flow%20Matching%20Emerges%20from%20the%20Chunk-level%20Policy%20Optimization/tables/f7da25d4a385dda5e4adb96a0b6dcd2efa287de0b91760c38d517c826b9304cd.jpg)

<a id="paper-3285"></a>

## DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning


### Images

![26ef7ebfcd3b4723f06dde2e6ad9ca447fb1004d7759191a5577759041d6932f.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/26ef7ebfcd3b4723f06dde2e6ad9ca447fb1004d7759191a5577759041d6932f.jpg)

![4ca5cb5c0e4aa7e89d4e4e70566d5c1786d6100d16e0660af8182a60c3fef1bf.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/4ca5cb5c0e4aa7e89d4e4e70566d5c1786d6100d16e0660af8182a60c3fef1bf.jpg)

![57aa426f59779a18c119e6486479d7e0de175df5c52016e84316e818ce8ead93.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/57aa426f59779a18c119e6486479d7e0de175df5c52016e84316e818ce8ead93.jpg)

![823a58b3778b0e5569eb95032eb9338ee0be0571d79473432db7001028aa9a7a.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/823a58b3778b0e5569eb95032eb9338ee0be0571d79473432db7001028aa9a7a.jpg)

![b5e4b5daad16738543fe0eb4deddf4a0b4ffd2ec0fa27369112ee0aa7f9d93a4.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/b5e4b5daad16738543fe0eb4deddf4a0b4ffd2ec0fa27369112ee0aa7f9d93a4.jpg)

![cb188e1fb34c4170246434ebe2821a291a460fea6559f075e8e83b9b9abc9f9b.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/cb188e1fb34c4170246434ebe2821a291a460fea6559f075e8e83b9b9abc9f9b.jpg)

![f767154fb5a49c952e43431bde0c6cc9f63259a58f2e96903e32a3c74a72413d.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/f767154fb5a49c952e43431bde0c6cc9f63259a58f2e96903e32a3c74a72413d.jpg)

![fcf45cabdd27f6b4b9632b36d6ba647b6a42ad43c6990a59f62d5876664c37fb.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/fcf45cabdd27f6b4b9632b36d6ba647b6a42ad43c6990a59f62d5876664c37fb.jpg)

![ffcb83fd7d349d7443f539adb9f2d1cce00de6e0ffeb358edac8f2ba3f755059.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/images/ffcb83fd7d349d7443f539adb9f2d1cce00de6e0ffeb358edac8f2ba3f755059.jpg)

### Tables

![2bbf7934f55af616ac1ceb1de6bdf78cb124392898951929599a26b44dcb3514.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/tables/2bbf7934f55af616ac1ceb1de6bdf78cb124392898951929599a26b44dcb3514.jpg)

![97cd502f303d8400a56bd190d8ba5133917a875479c323d2309257a6a01865dd.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/tables/97cd502f303d8400a56bd190d8ba5133917a875479c323d2309257a6a01865dd.jpg)

![adb6ffa39ad88d3617060e65b71cdf4f7b0bf7fa6773946ca6b1622c2926ed18.jpg](../icml_results/3324_DF-ExpEnse_%20Diffusion%20Filtered%20Exploration%20for%20Sample%20Efficient%20Finetuning/tables/adb6ffa39ad88d3617060e65b71cdf4f7b0bf7fa6773946ca6b1622c2926ed18.jpg)

<a id="paper-3286"></a>

## A Risk Decomposition Framework for Pre-hoc Fine-tuning Prediction


### Images

![00657d4ed4ae71592ed7679d6485cd8d3ca43ffb0cc7c9b53b0f4001214140ef.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/00657d4ed4ae71592ed7679d6485cd8d3ca43ffb0cc7c9b53b0f4001214140ef.jpg)

![0fc10702a8b777e29ea0fc58cb8ea940cfc00589f11c4a43a9d7b4eacb6864aa.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/0fc10702a8b777e29ea0fc58cb8ea940cfc00589f11c4a43a9d7b4eacb6864aa.jpg)

![196b28f03eb37a4306d51a71de1a6c516a42f86ed773295f1ec58d09c297c219.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/196b28f03eb37a4306d51a71de1a6c516a42f86ed773295f1ec58d09c297c219.jpg)

![2b7d73f7ce1d6b59e3c5303b320a7df90a4a88b8a1536ac14acd2fa552f301b5.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/2b7d73f7ce1d6b59e3c5303b320a7df90a4a88b8a1536ac14acd2fa552f301b5.jpg)

![968fb4a3af7852e1f3536e4e9ec542e899cc4840a0c7cbb532ae2644b70852b5.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/968fb4a3af7852e1f3536e4e9ec542e899cc4840a0c7cbb532ae2644b70852b5.jpg)

![be83ffc3453f41fb4f8135c2f8fb80b8a3c4c28a2bf41a4615182f5a9ab53006.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/be83ffc3453f41fb4f8135c2f8fb80b8a3c4c28a2bf41a4615182f5a9ab53006.jpg)

![d3057c83985d101b89019d6c4f16219f1263b611ed3f806533f7cdda6c031ca0.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/d3057c83985d101b89019d6c4f16219f1263b611ed3f806533f7cdda6c031ca0.jpg)

![f44c21c6d2fea1dfe1aa43cc08c6b668502ba0855e67cba32f6d29d1ee3218c2.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/images/f44c21c6d2fea1dfe1aa43cc08c6b668502ba0855e67cba32f6d29d1ee3218c2.jpg)

### Tables

![75ca6be9755db76baabeb1703851e503e4e55d13e3e61a20b9c40f5a6d14cbab.jpg](../icml_results/3325_A%20Risk%20Decomposition%20Framework%20for%20Pre-hoc%20Fine-tuning%20Prediction/tables/75ca6be9755db76baabeb1703851e503e4e55d13e3e61a20b9c40f5a6d14cbab.jpg)

<a id="paper-3287"></a>

## Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations


### Images

![04bdf7c1513e58e128df974dcdbe65493ccd073f20f6d374b3b430cdd4ec8cd4.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/04bdf7c1513e58e128df974dcdbe65493ccd073f20f6d374b3b430cdd4ec8cd4.jpg)

![103f8679997f14d04079ae5d0fa79d8807002d74544be07ecb927d83f0a58ec8.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/103f8679997f14d04079ae5d0fa79d8807002d74544be07ecb927d83f0a58ec8.jpg)

![11f96283dcf74863488f6cdf345074fff3cfd6bf78848a48905e5254371e06c9.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/11f96283dcf74863488f6cdf345074fff3cfd6bf78848a48905e5254371e06c9.jpg)

![53f6d265680f3f21b3ac06cb7b8c980cc7898a72ab426a677b2110d721cc03cc.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/53f6d265680f3f21b3ac06cb7b8c980cc7898a72ab426a677b2110d721cc03cc.jpg)

![d4bb31394dc01b678671c60e74a05981e89a9047b3a0218a7ccc0d33def38db2.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/d4bb31394dc01b678671c60e74a05981e89a9047b3a0218a7ccc0d33def38db2.jpg)

![e948e5572d3634c53d698c7db0ac77449fc97bb4e8edb4f5177a126774c78e25.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/images/e948e5572d3634c53d698c7db0ac77449fc97bb4e8edb4f5177a126774c78e25.jpg)

### Tables

![15683c372ab20c2d1abbb3f04256b0052f9717dfdcfbf8fc9c1bd31d6d8bd651.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/tables/15683c372ab20c2d1abbb3f04256b0052f9717dfdcfbf8fc9c1bd31d6d8bd651.jpg)

![90e41a3fa7c393c263bc21514709fdd9c24cfc6ece73ad2d590b95f05be21f93.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/tables/90e41a3fa7c393c263bc21514709fdd9c24cfc6ece73ad2d590b95f05be21f93.jpg)

![990819d2a5876021b75f132110db963ad57128907e6e2da70c8e0ef0ea040ab1.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/tables/990819d2a5876021b75f132110db963ad57128907e6e2da70c8e0ef0ea040ab1.jpg)

![b167ee8b5851dd77441bb850efebb961c35a440466616a9ce2574ab73f1acb20.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/tables/b167ee8b5851dd77441bb850efebb961c35a440466616a9ce2574ab73f1acb20.jpg)

![fa17ed5b230e946efd1611a1106debfef0d7aa853a0752d4c462a35854a41928.jpg](../icml_results/3326_Contrastive%20Reasoning%20Alignment_%20Reinforcement%20Learning%20from%20Hidden%20Representations/tables/fa17ed5b230e946efd1611a1106debfef0d7aa853a0752d4c462a35854a41928.jpg)

<a id="paper-3288"></a>

## ACO-MoE-LoRA: Evolving-while-Training for Adapting Segment Anything Model 2 to Specialized Domains


### Images

![36e118868f111c16f7f27b88c2e73b726f69921cb5a1c01e5f3eb21d5f04b2eb.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/36e118868f111c16f7f27b88c2e73b726f69921cb5a1c01e5f3eb21d5f04b2eb.jpg)

![3bebff0ae7329c800140cfbab2ac7072d5a6cdbc0428bb2091f2475dafed2219.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/3bebff0ae7329c800140cfbab2ac7072d5a6cdbc0428bb2091f2475dafed2219.jpg)

![81642d3605c8c74d2fcb5c359f301f57b1a6926d007e3719830ca3a3c4850363.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/81642d3605c8c74d2fcb5c359f301f57b1a6926d007e3719830ca3a3c4850363.jpg)

![819a04b798ceac386c2c4d5c590f1aee184d103ec695b4183f0205a68fb1c986.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/819a04b798ceac386c2c4d5c590f1aee184d103ec695b4183f0205a68fb1c986.jpg)

![a5d300171ba837e91ec37fa25c85726db06a50a9797a947e8c466758788f270c.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/a5d300171ba837e91ec37fa25c85726db06a50a9797a947e8c466758788f270c.jpg)

![b599f41c3fe7ab9d8651d363ea38d6f913e1cae2f39872311af7041f53a0bd09.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/b599f41c3fe7ab9d8651d363ea38d6f913e1cae2f39872311af7041f53a0bd09.jpg)

![b8dac50e72673ef3859b4b6599ed021833d583f8eda95186d326b9acb0aa07cd.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/b8dac50e72673ef3859b4b6599ed021833d583f8eda95186d326b9acb0aa07cd.jpg)

![bf00ef5c2486893730ed33cb168e92e2c09eef842148c0c028911558e67b3ca9.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/bf00ef5c2486893730ed33cb168e92e2c09eef842148c0c028911558e67b3ca9.jpg)

![fb535f38654e878ecdc6a44c14668fa91c994aacfeed9e5d95f298999da50ec5.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/images/fb535f38654e878ecdc6a44c14668fa91c994aacfeed9e5d95f298999da50ec5.jpg)

### Tables

![0943c05f90a553e342b6ee200b9ad5a04f07aa46733e335baa2c0dea1434cce1.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/0943c05f90a553e342b6ee200b9ad5a04f07aa46733e335baa2c0dea1434cce1.jpg)

![2c140807b92bdaa16e40dbe69f1603ee4eef3fc95997cecf5a6410afef259514.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/2c140807b92bdaa16e40dbe69f1603ee4eef3fc95997cecf5a6410afef259514.jpg)

![42d35ec7c8e5b4ca71fabe4c919d232d9f02c73f7cfe1232c640646a4474e491.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/42d35ec7c8e5b4ca71fabe4c919d232d9f02c73f7cfe1232c640646a4474e491.jpg)

![5afce6582310f969e05cdcf7b01fa5fa6f461f72cc1431902bcb9bcc00b4245e.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/5afce6582310f969e05cdcf7b01fa5fa6f461f72cc1431902bcb9bcc00b4245e.jpg)

![6b5400a88503f1cd60d4dad0d58ada633409b1b738ba05400010589a8c8b3f96.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/6b5400a88503f1cd60d4dad0d58ada633409b1b738ba05400010589a8c8b3f96.jpg)

![7ab2d736282f46e25697b0edd9145ab8f8f317671486a6da69229f06d629acc4.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/7ab2d736282f46e25697b0edd9145ab8f8f317671486a6da69229f06d629acc4.jpg)

![934d9a618891169632b8e6fe80b5ab248fb23883e23915a714b8420743ce05e8.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/934d9a618891169632b8e6fe80b5ab248fb23883e23915a714b8420743ce05e8.jpg)

![958d3f10d1d87318ca846526239e9e41ca073bdddd9d96aca5f775392e02d636.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/958d3f10d1d87318ca846526239e9e41ca073bdddd9d96aca5f775392e02d636.jpg)

![9bc59e3b9d697cd393c10f32766c58faa25717850f90f6e5b6c1dba7dd8be7df.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/9bc59e3b9d697cd393c10f32766c58faa25717850f90f6e5b6c1dba7dd8be7df.jpg)

![c7e9fdf7cf802d63b42325aa8219b3585dfbecedbebbddf75cc86da679a6cbfe.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/c7e9fdf7cf802d63b42325aa8219b3585dfbecedbebbddf75cc86da679a6cbfe.jpg)

![cbb6e6ceffbe089e0d86cdf3818adb19a7db7105d46faee619c14ba0cf666362.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/cbb6e6ceffbe089e0d86cdf3818adb19a7db7105d46faee619c14ba0cf666362.jpg)

![de223562eecd90ae745425b260e0df463c399e456cd3061f5b9978bdd804bd00.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/de223562eecd90ae745425b260e0df463c399e456cd3061f5b9978bdd804bd00.jpg)

![fabe7ca77ca4a4ff3fb5e74eb0755c5059fc42bcd5429fdde263f5c445fbc57e.jpg](../icml_results/3327_ACO-MoE-LoRA_%20Evolving-while-Training%20for%20Adapting%20Segment%20Anything%20Model%202%20to%20Specialized%20Domains/tables/fabe7ca77ca4a4ff3fb5e74eb0755c5059fc42bcd5429fdde263f5c445fbc57e.jpg)

<a id="paper-3289"></a>

## Online Change Point Detection for Multivariate Inhomogeneous Poisson Processes Time Series


### Images

![15dfdd31865fb90179cfca175152cbefffe7f2864a3d1d598ce5f31438377633.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/15dfdd31865fb90179cfca175152cbefffe7f2864a3d1d598ce5f31438377633.jpg)

![61dcb73bfd6173926de6356babb44da1efccf6f6f16ec05e858cdf053ba145af.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/61dcb73bfd6173926de6356babb44da1efccf6f6f16ec05e858cdf053ba145af.jpg)

![6a74f6cbb08c1561e720cea95f1c82c6d56c29c780d70126463ac8da84f4768a.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/6a74f6cbb08c1561e720cea95f1c82c6d56c29c780d70126463ac8da84f4768a.jpg)

![a9ac72f3e03af35114064147d08713db8e375c94498836897603e5c6ba307027.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/a9ac72f3e03af35114064147d08713db8e375c94498836897603e5c6ba307027.jpg)

![b090782a2b95769cb57b844a763bf24fdbf6a21799d5949e9ae8fc9ce520e344.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/b090782a2b95769cb57b844a763bf24fdbf6a21799d5949e9ae8fc9ce520e344.jpg)

![d366258b8b623d530f23345ab42e594e0a1a0f2b33863ae48e709252be9b46ae.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/images/d366258b8b623d530f23345ab42e594e0a1a0f2b33863ae48e709252be9b46ae.jpg)

### Tables

![08c5d29968b365602dac2381dcb1c52153539a28c88ed5f5b43bcde9419cd28e.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/tables/08c5d29968b365602dac2381dcb1c52153539a28c88ed5f5b43bcde9419cd28e.jpg)

![2cf2feecaa5852999a0399efe919d5620c10e3f7612f85fb621aee14a7dc6c70.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/tables/2cf2feecaa5852999a0399efe919d5620c10e3f7612f85fb621aee14a7dc6c70.jpg)

![be537915e5904fab2faae2fa40733671117f6c452632e9152ca94aa148cd918b.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/tables/be537915e5904fab2faae2fa40733671117f6c452632e9152ca94aa148cd918b.jpg)

![de9dd39fafee881a35f6b2858447f9ff72a07c2779bb572589290d55371b2cf4.jpg](../icml_results/3328_Online%20Change%20Point%20Detection%20for%20Multivariate%20Inhomogeneous%20Poisson%20Processes%20Time%20Series/tables/de9dd39fafee881a35f6b2858447f9ff72a07c2779bb572589290d55371b2cf4.jpg)

<a id="paper-3290"></a>

## Utility Boundary of Dataset Distillation: Scaling and Coverage Laws


### Images

![098cbdca0845fab40518a15af9229a83e15c90beb57b325706479993d8ab5a9b.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/098cbdca0845fab40518a15af9229a83e15c90beb57b325706479993d8ab5a9b.jpg)

![1f38d114c15a3e696e2f29a8efe7abb090887b9462151c2cf07d3cb1b42f0100.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/1f38d114c15a3e696e2f29a8efe7abb090887b9462151c2cf07d3cb1b42f0100.jpg)

![24a7856f79045d7a24ed70477d63f7a8aa39ac520735bfa016e122f913055e34.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/24a7856f79045d7a24ed70477d63f7a8aa39ac520735bfa016e122f913055e34.jpg)

![26b688b68482e6cd2b9150c6f562a9c1fba29ba06e102bc83fb803a30f4876c4.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/26b688b68482e6cd2b9150c6f562a9c1fba29ba06e102bc83fb803a30f4876c4.jpg)

![308a105443929bfd7d8d0d41e75c6efdb6c6b74ad2a19753ed121cc000c234d6.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/308a105443929bfd7d8d0d41e75c6efdb6c6b74ad2a19753ed121cc000c234d6.jpg)

![766cb0ebd1666acb5e064a63ca6967571abb6c4c47fea02a2be01941bffb04ca.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/766cb0ebd1666acb5e064a63ca6967571abb6c4c47fea02a2be01941bffb04ca.jpg)

![84371338ef9aa30fb69e55e6b30df9c767c4885de4f03bdb2117b8b662b29de7.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/84371338ef9aa30fb69e55e6b30df9c767c4885de4f03bdb2117b8b662b29de7.jpg)

![861a98ae076de385392978e020df5cfb3a45cf30000fbc8ece55bf96c9c52f3e.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/861a98ae076de385392978e020df5cfb3a45cf30000fbc8ece55bf96c9c52f3e.jpg)

![8706378857c83f3f57d36b8ab278aa7253c28f55e987b2ef35b6c7fd8963a094.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/8706378857c83f3f57d36b8ab278aa7253c28f55e987b2ef35b6c7fd8963a094.jpg)

![9bd8c759d4919501603d7498038c1066daade429984320b4a9587cccfe68371b.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/9bd8c759d4919501603d7498038c1066daade429984320b4a9587cccfe68371b.jpg)

![a00deb0eef49396e66c3430d2dfdea7f634ffd0aed2fb9e6b37774af5b299b10.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/a00deb0eef49396e66c3430d2dfdea7f634ffd0aed2fb9e6b37774af5b299b10.jpg)

![a4e8dfb36b55f79cbeed0fa024d7ccfecdb9f21483556a6e929943b009ff28ef.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/a4e8dfb36b55f79cbeed0fa024d7ccfecdb9f21483556a6e929943b009ff28ef.jpg)

![af03f19c96af661eda93c3b2ea338d9b9e8df89c7ae17c49765da9f634fc2818.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/af03f19c96af661eda93c3b2ea338d9b9e8df89c7ae17c49765da9f634fc2818.jpg)

![b6f91754dbeabe8aa2893a47820e24bf07f8a8e28df7ceaf11735bd5b08505fa.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/b6f91754dbeabe8aa2893a47820e24bf07f8a8e28df7ceaf11735bd5b08505fa.jpg)

![bb478ce4f08ae3ad610417223ab71b429ffd2e936fd7599cb521ef09c12fcbab.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/bb478ce4f08ae3ad610417223ab71b429ffd2e936fd7599cb521ef09c12fcbab.jpg)

![e1b1aae1ac35ada0ed3b4f97c208ae3b0583e746aa6927f3478269767ecc1938.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/e1b1aae1ac35ada0ed3b4f97c208ae3b0583e746aa6927f3478269767ecc1938.jpg)

![eba8813bde9f1d8658a6af5e4858c5755c0a7209436b8e6323452085172ff85f.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/eba8813bde9f1d8658a6af5e4858c5755c0a7209436b8e6323452085172ff85f.jpg)

![ecc331c49c109543241d4555f709538906393569fcb864dbede66dc2bee66412.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/images/ecc331c49c109543241d4555f709538906393569fcb864dbede66dc2bee66412.jpg)

### Tables

![15967ba4c89b532aeee46dca51d452cef8e3a710f5df0ede5a541ffc21e00775.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/15967ba4c89b532aeee46dca51d452cef8e3a710f5df0ede5a541ffc21e00775.jpg)

![2df8055ca15282dc78952320a3bd6ab2387635dce07591f355975dbcbc319d79.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/2df8055ca15282dc78952320a3bd6ab2387635dce07591f355975dbcbc319d79.jpg)

![4104f4bcc5749995cfbfc68de6a2c9063653993b6b87cb5a3c20b2710607b13d.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/4104f4bcc5749995cfbfc68de6a2c9063653993b6b87cb5a3c20b2710607b13d.jpg)

![42a1f63f4874af40084b3fe3f4e184483c0e0e2622f25350da7022b28681b02b.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/42a1f63f4874af40084b3fe3f4e184483c0e0e2622f25350da7022b28681b02b.jpg)

![53a39a4c76c01df0c980d2500d3d753fde3a49e849aef8e0de361da492856cd0.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/53a39a4c76c01df0c980d2500d3d753fde3a49e849aef8e0de361da492856cd0.jpg)

![5f1b6bb2e781e61857d6dcf26dfaafde95d07be0264f489b8d4154ff805eb749.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/5f1b6bb2e781e61857d6dcf26dfaafde95d07be0264f489b8d4154ff805eb749.jpg)

![60197a9d94cea57a09e2328bcd6fb0c3dc9e3e8ec62b3bfd5967e266ff3e74d9.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/60197a9d94cea57a09e2328bcd6fb0c3dc9e3e8ec62b3bfd5967e266ff3e74d9.jpg)

![79866af0d8308e6ce85021d06b3e21b2d97038c39fd4d04e8c5395d7e5a5019c.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/79866af0d8308e6ce85021d06b3e21b2d97038c39fd4d04e8c5395d7e5a5019c.jpg)

![839f47ed3a9079b4aa3eb79bd473b9a01ab59a8f82c6c5df9f32daa0dd4780ae.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/839f47ed3a9079b4aa3eb79bd473b9a01ab59a8f82c6c5df9f32daa0dd4780ae.jpg)

![8f75cc79ebd0e67a848cb536d12e513f0786003cbaa4e07f9c37ac0f06f5f12e.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/8f75cc79ebd0e67a848cb536d12e513f0786003cbaa4e07f9c37ac0f06f5f12e.jpg)

![ab2b36f975ed7a562fbc0c36ab68a1474511f7e53c24f232927d14a751be6e6f.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/ab2b36f975ed7a562fbc0c36ab68a1474511f7e53c24f232927d14a751be6e6f.jpg)

![b7d0cb11f9429d61996e5f6186868bd4550bf193b51c31895049e869feb7a9da.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/b7d0cb11f9429d61996e5f6186868bd4550bf193b51c31895049e869feb7a9da.jpg)

![cd1518411a877687b2e8b9a7e322d12352752390c5df2419b91498df8c0f262e.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/cd1518411a877687b2e8b9a7e322d12352752390c5df2419b91498df8c0f262e.jpg)

![d5ffeb4217a426248a8140ba1d0200f1b6643f8e24ca98045db149a6dab2e59a.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/d5ffeb4217a426248a8140ba1d0200f1b6643f8e24ca98045db149a6dab2e59a.jpg)

![d818f94ce5fffbba616a65cddf89675492d3da4aab17e53a81511d00b0a1839a.jpg](../icml_results/3329_Utility%20Boundary%20of%20Dataset%20Distillation_%20Scaling%20and%20Coverage%20Laws/tables/d818f94ce5fffbba616a65cddf89675492d3da4aab17e53a81511d00b0a1839a.jpg)

<a id="paper-3291"></a>

## STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System


### Images

![02ac8f1f7a06042e16639b6604c325bbf3ff56f2f6be12a1a2ed4af57b9104f2.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/02ac8f1f7a06042e16639b6604c325bbf3ff56f2f6be12a1a2ed4af57b9104f2.jpg)

![098a199e709289adfbbf01cb1e61a8416715b3fbe28ed6b2c9d1de9285d69fb0.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/098a199e709289adfbbf01cb1e61a8416715b3fbe28ed6b2c9d1de9285d69fb0.jpg)

![09e24516be552bac748c8859b84bd071710ebbf8e3af543ab6d82ff6c0b7416c.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/09e24516be552bac748c8859b84bd071710ebbf8e3af543ab6d82ff6c0b7416c.jpg)

![1756a61647ea0a9a096391a682e2ee845abbb2883b94615bdef10919ec5c6d70.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/1756a61647ea0a9a096391a682e2ee845abbb2883b94615bdef10919ec5c6d70.jpg)

![1da987b7b45eb2036be22a6514fc2347c9019184d21b59cf27ee176c02b17d57.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/1da987b7b45eb2036be22a6514fc2347c9019184d21b59cf27ee176c02b17d57.jpg)

![60cb215064d297f79bf914045939eb1320da38ca8083fba2edf7e90047b0496f.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/60cb215064d297f79bf914045939eb1320da38ca8083fba2edf7e90047b0496f.jpg)

![78b64df6bb9aa24643a67c896d1be5434be26d4825f3e6bea05b673275c03b26.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/78b64df6bb9aa24643a67c896d1be5434be26d4825f3e6bea05b673275c03b26.jpg)

![a445a32ed57ed7285b40792785e0791c377ed94f54bef5e0d1e90991489980e5.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/a445a32ed57ed7285b40792785e0791c377ed94f54bef5e0d1e90991489980e5.jpg)

![ae6e566d3d6a8933837bad1e147ec8173ad657f9514c6f5a0b7521801d732570.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/ae6e566d3d6a8933837bad1e147ec8173ad657f9514c6f5a0b7521801d732570.jpg)

![c43e9fb9fb469db60d6a248e21372c25381000b42fc1fb04f6a7a6b822a0ea7b.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/c43e9fb9fb469db60d6a248e21372c25381000b42fc1fb04f6a7a6b822a0ea7b.jpg)

![de94f73d21b75e250058b0b11b1b2aaeda4f8c10ab16b3ba85a91c5626af0f5a.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/de94f73d21b75e250058b0b11b1b2aaeda4f8c10ab16b3ba85a91c5626af0f5a.jpg)

![e1f5c0edad57c065cfd0b6fb4d381eb6b605802d17ccfd3f43439689e8b8c601.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/images/e1f5c0edad57c065cfd0b6fb4d381eb6b605802d17ccfd3f43439689e8b8c601.jpg)

### Tables

![4858ffe28292dfdb3f6db13c9ac9c92570472805417bdaac098ae5c0dadf0903.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/tables/4858ffe28292dfdb3f6db13c9ac9c92570472805417bdaac098ae5c0dadf0903.jpg)

![eca948857be1cb7b9ba8ce9fa8940d89827055c2740d0d32fea10775f187bc48.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/tables/eca948857be1cb7b9ba8ce9fa8940d89827055c2740d0d32fea10775f187bc48.jpg)

![f7acc8df0181790bde24def020f703965c3b053b30ce9bf5bab52bb47dbf5e2d.jpg](../icml_results/3330_STABLE_%20Simulation-Ready%20Tabletop%20Layout%20Generation%20via%20a%20Semantics%E2%80%93Physics%20Dual%20System/tables/f7acc8df0181790bde24def020f703965c3b053b30ce9bf5bab52bb47dbf5e2d.jpg)

<a id="paper-3292"></a>

## ProConMV: Provenance-Enabled Conceptual Framework for Interpretable Multi-View Diabetic Retinopathy Diagnosis


### Images

![03cb863afb238479014e6190760e2371350dd865e2a3405fcfaa606b58b73063.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/03cb863afb238479014e6190760e2371350dd865e2a3405fcfaa606b58b73063.jpg)

![273e410801942fbd3f9030b2fe6962a75c7a12171fc3621b769d77ab01deaf66.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/273e410801942fbd3f9030b2fe6962a75c7a12171fc3621b769d77ab01deaf66.jpg)

![338dac16bdc58a0bacdd196f0e8868634a5c4c47b6c2772e0e545adaf1d5b695.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/338dac16bdc58a0bacdd196f0e8868634a5c4c47b6c2772e0e545adaf1d5b695.jpg)

![3f84b4a4a11eabf532bb924cea8a2e951e11f692ba614bc11d8dcf8326a0d829.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/3f84b4a4a11eabf532bb924cea8a2e951e11f692ba614bc11d8dcf8326a0d829.jpg)

![6a7e1d0b0ea06ca50cd2dbd57a12f5fbdd8519ec2b681b86bfb9932c781518d7.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/6a7e1d0b0ea06ca50cd2dbd57a12f5fbdd8519ec2b681b86bfb9932c781518d7.jpg)

![78e48ee42b1d56d4cea6caf5208b2669d5b5335fd21fa1068f2ab4d27386bc4f.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/78e48ee42b1d56d4cea6caf5208b2669d5b5335fd21fa1068f2ab4d27386bc4f.jpg)

![8cd44bcaba84461a850a3f0bed2622ef59d11913b62af6f8f7735565587bc0d2.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/8cd44bcaba84461a850a3f0bed2622ef59d11913b62af6f8f7735565587bc0d2.jpg)

![978bb9f0f4853f6bcc27b5d122a55715ed54d571ef3a7feb1708e767c91e471c.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/978bb9f0f4853f6bcc27b5d122a55715ed54d571ef3a7feb1708e767c91e471c.jpg)

![b6d7a103cc62aee15506b11e6e10635fdd15de09a2196c66557a2073aedf883d.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/b6d7a103cc62aee15506b11e6e10635fdd15de09a2196c66557a2073aedf883d.jpg)

![be1681b91669a7d968aac6ff850f7a71ae1cd783b1397fdbeddd5f8eeb8e9b0f.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/be1681b91669a7d968aac6ff850f7a71ae1cd783b1397fdbeddd5f8eeb8e9b0f.jpg)

![d05502edb77820fa8b6b5b70572b18d8a1a546f935683b05a8efbdd8bb2372f7.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/d05502edb77820fa8b6b5b70572b18d8a1a546f935683b05a8efbdd8bb2372f7.jpg)

![d36738062c663ff50e22555968947c26909d604d1daeb2602fc5842643e25194.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/d36738062c663ff50e22555968947c26909d604d1daeb2602fc5842643e25194.jpg)

![d6098ae50bffef56a7d553b47199d0a141f81bffb2b7f3917ca4a879f49504ae.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/d6098ae50bffef56a7d553b47199d0a141f81bffb2b7f3917ca4a879f49504ae.jpg)

![f5bafbf2e3104656b3ac70c0cbeedce00f141ba34ac99040584c66b2c845bce2.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/images/f5bafbf2e3104656b3ac70c0cbeedce00f141ba34ac99040584c66b2c845bce2.jpg)

### Tables

![24fffafb0c04eb26f665e8f0bbf54eaf126f958a9439df039bbb4fbdd43b9a5b.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/tables/24fffafb0c04eb26f665e8f0bbf54eaf126f958a9439df039bbb4fbdd43b9a5b.jpg)

![bfde2bc388b4aa364c5a3fb5707e37d9c863073c5d477ee933b938c99c54b8f6.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/tables/bfde2bc388b4aa364c5a3fb5707e37d9c863073c5d477ee933b938c99c54b8f6.jpg)

![c3fe6864127e0c5ae3f8349daf569efb115718712872c856a19d7520cfa826f3.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/tables/c3fe6864127e0c5ae3f8349daf569efb115718712872c856a19d7520cfa826f3.jpg)

![c7ff9e7f6d1ec0ed64957d5b2663db36adbf5c7928d317e05e6d80ada5689eee.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/tables/c7ff9e7f6d1ec0ed64957d5b2663db36adbf5c7928d317e05e6d80ada5689eee.jpg)

![efe10ec33c3c7072058aee32df8c445f4a4580ead787e543e458d375871dd2ef.jpg](../icml_results/3331_ProConMV_%20Provenance-Enabled%20Conceptual%20Framework%20for%20Interpretable%20Multi-View%20Diabetic%20Retinopathy%20/tables/efe10ec33c3c7072058aee32df8c445f4a4580ead787e543e458d375871dd2ef.jpg)

<a id="paper-3293"></a>

## Closing the Loop: Universal Repository Representation with RPG-Encoder


### Images

![192e19dccd1d01e149096beaf1af33a3d488f2ece8a95aa28ccc61ecab4f340d.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/192e19dccd1d01e149096beaf1af33a3d488f2ece8a95aa28ccc61ecab4f340d.jpg)

![2334cc4a2acb8bde9c966b985c6dd1115715200feb48aa22391b42ffe8a7f15d.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/2334cc4a2acb8bde9c966b985c6dd1115715200feb48aa22391b42ffe8a7f15d.jpg)

![26f8d2ab28701341808d9521a37b97c6f3f3575178709501f2525dd1fcf606ce.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/26f8d2ab28701341808d9521a37b97c6f3f3575178709501f2525dd1fcf606ce.jpg)

![3bfcbed99059024a9c22351e64536e0efb4e33bdcdff7d8fac9fdcf6303be704.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/3bfcbed99059024a9c22351e64536e0efb4e33bdcdff7d8fac9fdcf6303be704.jpg)

![44b79129217ff2fd3cdc0d0f266304556194317ce4957f5c1883dceed640c211.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/44b79129217ff2fd3cdc0d0f266304556194317ce4957f5c1883dceed640c211.jpg)

![88c1ee15e0b66d2259c6f01c4688c724f57afc35b793e86380f1259e6b39758b.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/88c1ee15e0b66d2259c6f01c4688c724f57afc35b793e86380f1259e6b39758b.jpg)

![8fd8b80569e5e1dbdbea49ffd569dddde3638fbc7e9bdc30102a43ad87543a22.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/8fd8b80569e5e1dbdbea49ffd569dddde3638fbc7e9bdc30102a43ad87543a22.jpg)

![975b08399b722d7a5f3cedde2a0da4c934579cba9e4122cc69cb95995898608a.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/975b08399b722d7a5f3cedde2a0da4c934579cba9e4122cc69cb95995898608a.jpg)

![a6439bdc77a8e0290f0240afb2d91aaa5fd9365c1104ce7354cbd48fa876aceb.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/a6439bdc77a8e0290f0240afb2d91aaa5fd9365c1104ce7354cbd48fa876aceb.jpg)

![d245dff96338bd489bd053728abe6c1a43dfdaf023cb1f07c6d7b745ce109116.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/d245dff96338bd489bd053728abe6c1a43dfdaf023cb1f07c6d7b745ce109116.jpg)

![d682fa0d9f0f391b0720ecd1b5686d6b51c867d66121a5e490c69b061aa58521.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/d682fa0d9f0f391b0720ecd1b5686d6b51c867d66121a5e490c69b061aa58521.jpg)

![e23a9c00f9924337561e9041f5ffe5045fec26f67d8241392c8643c51e0bcd98.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/e23a9c00f9924337561e9041f5ffe5045fec26f67d8241392c8643c51e0bcd98.jpg)

![ee6b97ebf3d24e675d1a31c95c7eeaa0cf6ae190386f2b22feaed271a1817750.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/ee6b97ebf3d24e675d1a31c95c7eeaa0cf6ae190386f2b22feaed271a1817750.jpg)

![f20bf1fad1fdffe1dd9a40d077e32ba8451195e3f207e44625bd2e6b9bfaeb56.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/images/f20bf1fad1fdffe1dd9a40d077e32ba8451195e3f207e44625bd2e6b9bfaeb56.jpg)

### Tables

![0ecf5367b9e626a17aae5323cc87afade3fd00cdf3b683cfe1cc1eba99dbb823.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/0ecf5367b9e626a17aae5323cc87afade3fd00cdf3b683cfe1cc1eba99dbb823.jpg)

![1b3c13fd7abfd00daad384d9883fffc43defa295e7ed55331ab56a75866e236c.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/1b3c13fd7abfd00daad384d9883fffc43defa295e7ed55331ab56a75866e236c.jpg)

![3e760bf37a56c0e83241f917135ee36f277b6923f71f5c5c23741ccd17c128cb.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/3e760bf37a56c0e83241f917135ee36f277b6923f71f5c5c23741ccd17c128cb.jpg)

![44bc29a3b7f383ef7b901dc9b0b46bdd4fcc12d5854100e708701dcb1964df26.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/44bc29a3b7f383ef7b901dc9b0b46bdd4fcc12d5854100e708701dcb1964df26.jpg)

![55168b9cff7ed1d1526007f21d51bd603bc17111be7b4248a2ec9e5f69c83de6.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/55168b9cff7ed1d1526007f21d51bd603bc17111be7b4248a2ec9e5f69c83de6.jpg)

![5a76278ea7fefd3af314a70bd037e6981a1070509f4e22366e6610b2a3f1742e.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/5a76278ea7fefd3af314a70bd037e6981a1070509f4e22366e6610b2a3f1742e.jpg)

![6fd9f6ceb603b5c9709d4a7e07f1c38fb01117b12c856246d0a7a64508243372.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/6fd9f6ceb603b5c9709d4a7e07f1c38fb01117b12c856246d0a7a64508243372.jpg)

![714970055610a771af32993cba89f86dea3b084dc4fb7bd9d8b15d67e3adf300.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/714970055610a771af32993cba89f86dea3b084dc4fb7bd9d8b15d67e3adf300.jpg)

![91a5ae635838520c4506e23704948f2675f8a2242a4bef1f115850aec4d1a7e9.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/91a5ae635838520c4506e23704948f2675f8a2242a4bef1f115850aec4d1a7e9.jpg)

![94657612a988803214c57dd374df7ef956c21b58f3e46ff206d2ca0a78d9cf33.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/94657612a988803214c57dd374df7ef956c21b58f3e46ff206d2ca0a78d9cf33.jpg)

![a0f9253f9f8d88eaa1fa4250e357a246c4998c8477743a361723223b514e4193.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/a0f9253f9f8d88eaa1fa4250e357a246c4998c8477743a361723223b514e4193.jpg)

![b85776d26694ca5d119d73cf7b9b7c49417ce225cb8a5a206551591d31048b8d.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/b85776d26694ca5d119d73cf7b9b7c49417ce225cb8a5a206551591d31048b8d.jpg)

![bc2fc1073e9ac598c54b754fa51b6829dceeb853d7755c082f64f44f5c0bd089.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/bc2fc1073e9ac598c54b754fa51b6829dceeb853d7755c082f64f44f5c0bd089.jpg)

![c289e813aa4a3136fd9cdae23d98cb7ec37fe73991a12f8f0370326735981c14.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/c289e813aa4a3136fd9cdae23d98cb7ec37fe73991a12f8f0370326735981c14.jpg)

![c9a993d04d57ed89c5e0c2297273cc95fe5e4b5670e56c551ce7f7e24988996c.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/c9a993d04d57ed89c5e0c2297273cc95fe5e4b5670e56c551ce7f7e24988996c.jpg)

![ca99686d07d62a0b11869b5c2972890e66020c686d0b23f18026993d9f142e33.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/ca99686d07d62a0b11869b5c2972890e66020c686d0b23f18026993d9f142e33.jpg)

![cd2dbb0c566e9d2438da4be86e3b5b6f9c4d4804449851f104b8bae9fe2259ac.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/cd2dbb0c566e9d2438da4be86e3b5b6f9c4d4804449851f104b8bae9fe2259ac.jpg)

![f3f5793b8eb4d0aa3af02a01d95cf49287b94d4161d779b9c20f75b27305ab6d.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/f3f5793b8eb4d0aa3af02a01d95cf49287b94d4161d779b9c20f75b27305ab6d.jpg)

![f9ab6565319e91a0974a2bb9981a79b7e97f5f06cb1a35adc620d34709c36976.jpg](../icml_results/3332_Closing%20the%20Loop_%20Universal%20Repository%20Representation%20with%20RPG-Encoder/tables/f9ab6565319e91a0974a2bb9981a79b7e97f5f06cb1a35adc620d34709c36976.jpg)

<a id="paper-3294"></a>

## Causal Matrix Completion under Multiple Treatments via Mixed Synthetic Nearest Neighbors


### Images

![a697efb32433d4e2b578b3fa543c6780f27a477678d4e2d5b77cd3a1cf851fc5.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/images/a697efb32433d4e2b578b3fa543c6780f27a477678d4e2d5b77cd3a1cf851fc5.jpg)

![c24de7b2a7b2da3cca1e43291e275efe6e336b8da058dbc1a5029244b9bd3c3f.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/images/c24de7b2a7b2da3cca1e43291e275efe6e336b8da058dbc1a5029244b9bd3c3f.jpg)

### Tables

![0f33181248a0c1bdc0aba69cdfabc01a995fe7d9a99abf8b64e6446260e8ef96.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/0f33181248a0c1bdc0aba69cdfabc01a995fe7d9a99abf8b64e6446260e8ef96.jpg)

![16e9541d937c44cc1792c6c08225b10d2b6ce04ae5327b7fa2763ba4dcb97966.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/16e9541d937c44cc1792c6c08225b10d2b6ce04ae5327b7fa2763ba4dcb97966.jpg)

![33ad80c46aa4fc721cce821ef48cefad3491a8b51012f8437e7df0dfcde79d82.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/33ad80c46aa4fc721cce821ef48cefad3491a8b51012f8437e7df0dfcde79d82.jpg)

![4af364ca47da881b0f511a1174476fd41aca28378ec556cc0513045ec5b349fa.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/4af364ca47da881b0f511a1174476fd41aca28378ec556cc0513045ec5b349fa.jpg)

![51b3e643102ddc4fbf19e60a0593e3163ecd7592e54ab3d2b8c6bc49b1e9b24a.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/51b3e643102ddc4fbf19e60a0593e3163ecd7592e54ab3d2b8c6bc49b1e9b24a.jpg)

![69774704d9e01cc8647d1e1feee018ed2b9f3f0da451cdf52750d19fa970c37e.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/69774704d9e01cc8647d1e1feee018ed2b9f3f0da451cdf52750d19fa970c37e.jpg)

![772fa86374c112048dc90e8fe3af671a147983d7385ffd93dd0dee2cbc9b4dcd.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/772fa86374c112048dc90e8fe3af671a147983d7385ffd93dd0dee2cbc9b4dcd.jpg)

![7d5f4c346fdc7d964d33f2fef5a836cf4931d31a65f82e6d99912db1237ce263.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/7d5f4c346fdc7d964d33f2fef5a836cf4931d31a65f82e6d99912db1237ce263.jpg)

![93e558347618e098374b32520fcef8bb63887fcc96a0f191f66f48ef62b4c6df.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/93e558347618e098374b32520fcef8bb63887fcc96a0f191f66f48ef62b4c6df.jpg)

![efc7cd5fc2011c8d2f892f3c5b35d65741ea8cdd8927a480252eb7e921e73161.jpg](../icml_results/3333_Causal%20Matrix%20Completion%20under%20Multiple%20Treatments%20via%20Mixed%20Synthetic%20Nearest%20Neighbors/tables/efc7cd5fc2011c8d2f892f3c5b35d65741ea8cdd8927a480252eb7e921e73161.jpg)

<a id="paper-3295"></a>

## MotionMAR: Multi-scale Auto-Regressive Human Motion Reconstruction from Sparse Observations


### Images

![28fc9ed014facb4e7e75e3f03df13c034a567c73dd691c7fefdbfcec8470dc59.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/28fc9ed014facb4e7e75e3f03df13c034a567c73dd691c7fefdbfcec8470dc59.jpg)

![475ea314a4006e3d907c68cc76f6367777dd6120471a9297267db4c924dd2e0c.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/475ea314a4006e3d907c68cc76f6367777dd6120471a9297267db4c924dd2e0c.jpg)

![5028c7fd58c536e09a29e6bc6c543f6e091dd5ce70264894d61fe9ae7f9e4246.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/5028c7fd58c536e09a29e6bc6c543f6e091dd5ce70264894d61fe9ae7f9e4246.jpg)

![5a8d20094dedb57cdc6782c0169226e7ba99333f9c0c343cfd1443d0ba549e7a.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/5a8d20094dedb57cdc6782c0169226e7ba99333f9c0c343cfd1443d0ba549e7a.jpg)

![78ffcd3f24c0f034a654cca83cc3a668cf70c477bf4ea4c836eab98f050db029.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/78ffcd3f24c0f034a654cca83cc3a668cf70c477bf4ea4c836eab98f050db029.jpg)

![876169598f2f75bb01d5ea35e5e845b590253df20fc628d894c2450f3248166f.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/876169598f2f75bb01d5ea35e5e845b590253df20fc628d894c2450f3248166f.jpg)

![8c7a45442fc4a385464b9dde929c40e2c0bfe6acea758a8daccb989e66883f57.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/8c7a45442fc4a385464b9dde929c40e2c0bfe6acea758a8daccb989e66883f57.jpg)

![a7d0305c0056eb091b27019918d3c866538c96ed0297f3bda7e44124807f984b.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/images/a7d0305c0056eb091b27019918d3c866538c96ed0297f3bda7e44124807f984b.jpg)

### Tables

![1cefba96625f550f2ec398341d5b233aba8370fa77a105e62ff1cc0fe49a527c.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/1cefba96625f550f2ec398341d5b233aba8370fa77a105e62ff1cc0fe49a527c.jpg)

![419c6718edda391226c9116b20f5f7cb60a3ad4d7cb6afebbf964c6c35587085.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/419c6718edda391226c9116b20f5f7cb60a3ad4d7cb6afebbf964c6c35587085.jpg)

![508f8787f465ceef502de04bb069cf64880ee6f739692e04354c15ccb64074c5.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/508f8787f465ceef502de04bb069cf64880ee6f739692e04354c15ccb64074c5.jpg)

![54a8e993cedae6a3d82faa4657cd0853b48ef154de8c7910b656f5b5492d3c5d.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/54a8e993cedae6a3d82faa4657cd0853b48ef154de8c7910b656f5b5492d3c5d.jpg)

![73c72804dd88b6412f69680a100b69c1ea5e0a44c7b8f85bc4e9efc3614cd43d.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/73c72804dd88b6412f69680a100b69c1ea5e0a44c7b8f85bc4e9efc3614cd43d.jpg)

![75825918c5caa17be3daba95b66f0f136450ac7816e2da2fcf6c7e5d4bcea85a.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/75825918c5caa17be3daba95b66f0f136450ac7816e2da2fcf6c7e5d4bcea85a.jpg)

![78731db73907c68d8ddfb9691a334ed445222e977cd70b188ef06c9781b0e770.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/78731db73907c68d8ddfb9691a334ed445222e977cd70b188ef06c9781b0e770.jpg)

![7db3e90d663cf2b746b9570e341cda70dd2597d0efb4b013de6dd68ae9e4d545.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/7db3e90d663cf2b746b9570e341cda70dd2597d0efb4b013de6dd68ae9e4d545.jpg)

![b1c8b57a6466597b20cd89bb037e98215bd8da2b3171de5766ff75ee6be2d38b.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/b1c8b57a6466597b20cd89bb037e98215bd8da2b3171de5766ff75ee6be2d38b.jpg)

![d43eb39c6d437adc6c4519cf2e259e9f26931dc406e64052cc64dbf8fde3e395.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/d43eb39c6d437adc6c4519cf2e259e9f26931dc406e64052cc64dbf8fde3e395.jpg)

![e00c9bfc069f64e634492b3e455c3efdfca3adeccdd3645d75041e2facbd062a.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/e00c9bfc069f64e634492b3e455c3efdfca3adeccdd3645d75041e2facbd062a.jpg)

![e5ec1d13522a5c35dabd72008f4660508cedf0b2fa6ffe137768f53551ec0da2.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/e5ec1d13522a5c35dabd72008f4660508cedf0b2fa6ffe137768f53551ec0da2.jpg)

![e68f63269e2d46a285a43e4d965cc245301b3240c28c158dd8b68650aed4f2de.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/e68f63269e2d46a285a43e4d965cc245301b3240c28c158dd8b68650aed4f2de.jpg)

![faca56f144a5eacc28362f85d6b5403560ff896cf5ad8381b9f0822a37c9a491.jpg](../icml_results/3334_MotionMAR_%20Multi-scale%20Auto-Regressive%20Human%20Motion%20Reconstruction%20from%20Sparse%20Observations/tables/faca56f144a5eacc28362f85d6b5403560ff896cf5ad8381b9f0822a37c9a491.jpg)

<a id="paper-3296"></a>

## SABER: Continual Learning with Representation Conflict Management


### Images

![08bb9262f0d698d97cb0694c7cbcb5afc379243b857b35c837ed1d62f2d94563.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/images/08bb9262f0d698d97cb0694c7cbcb5afc379243b857b35c837ed1d62f2d94563.jpg)

![35d3287846dfa475de74c7ee5415467d0bbdc33c8748f5fc1420f5fffa0a7040.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/images/35d3287846dfa475de74c7ee5415467d0bbdc33c8748f5fc1420f5fffa0a7040.jpg)

![5f2895b3a2244bb97820a0204b5d4a76e39720b95f907750b0a9b84c64bbfe33.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/images/5f2895b3a2244bb97820a0204b5d4a76e39720b95f907750b0a9b84c64bbfe33.jpg)

![94541ee6a22b15613b2167fe7f2dd356b473e7b88c66150820a7dc3d143bcb15.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/images/94541ee6a22b15613b2167fe7f2dd356b473e7b88c66150820a7dc3d143bcb15.jpg)

![ce96a752928d091f78e630eedbb0197333678f695c8c99e588ea6e88595b7111.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/images/ce96a752928d091f78e630eedbb0197333678f695c8c99e588ea6e88595b7111.jpg)

### Tables

![073d1f890c306cf27124075dca1e2672a6e47c354d13db5d5cd34f533e2e677f.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/073d1f890c306cf27124075dca1e2672a6e47c354d13db5d5cd34f533e2e677f.jpg)

![0e07cbc72dd46be76e959504f09d8826aa6de0501c3354fa509ff7bee7d4e382.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/0e07cbc72dd46be76e959504f09d8826aa6de0501c3354fa509ff7bee7d4e382.jpg)

![32ebca58f10dd057699b2a7ddb97b3b873c9022ab52ccf53b253c3585f057b3d.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/32ebca58f10dd057699b2a7ddb97b3b873c9022ab52ccf53b253c3585f057b3d.jpg)

![3e16c0f8d32c64d0c62b5664a393f6775cb5106768d60c8670fc6c1ad9f736a8.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/3e16c0f8d32c64d0c62b5664a393f6775cb5106768d60c8670fc6c1ad9f736a8.jpg)

![6c7c9840558ef1cee4523fff3c30931e277c4a51b4a5d24c65c81bea8b388880.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/6c7c9840558ef1cee4523fff3c30931e277c4a51b4a5d24c65c81bea8b388880.jpg)

![8fc3b6c0f8b9b91e0692e1074151f74b14b5ea95971b8f013446a06daa4990f5.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/8fc3b6c0f8b9b91e0692e1074151f74b14b5ea95971b8f013446a06daa4990f5.jpg)

![c1d41de96597ce69fc6e2edcc79a6cac604bb548c98bcaf0a4ac100120c72ffc.jpg](../icml_results/3335_SABER_%20Continual%20Learning%20with%20Representation%20Conflict%20Management/tables/c1d41de96597ce69fc6e2edcc79a6cac604bb548c98bcaf0a4ac100120c72ffc.jpg)

<a id="paper-3297"></a>

## SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning


### Images

![1bb9d4ea4a4b1148248fe927d8fbd029994b04aba4babc81d544eee9a43a7256.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/1bb9d4ea4a4b1148248fe927d8fbd029994b04aba4babc81d544eee9a43a7256.jpg)

![256048ef2b419bd8f842fc60f64786fccf963cef94624464199178026c52db53.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/256048ef2b419bd8f842fc60f64786fccf963cef94624464199178026c52db53.jpg)

![479aa9cb651ab0ed123443e1f3686133ea8bb13805d6ab9ea130d353963399de.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/479aa9cb651ab0ed123443e1f3686133ea8bb13805d6ab9ea130d353963399de.jpg)

![496dce05e8349277e0acf83cb37c0a877711816d75845708278d099f7c029721.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/496dce05e8349277e0acf83cb37c0a877711816d75845708278d099f7c029721.jpg)

![5c1bc0546c4435221e538f7b0b8d7204acceef20d536dc82f36bf24c600db91f.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/5c1bc0546c4435221e538f7b0b8d7204acceef20d536dc82f36bf24c600db91f.jpg)

![71868dbc33d4a41507d9e76b7bace6cec00208478470d8045944848bb17e739c.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/71868dbc33d4a41507d9e76b7bace6cec00208478470d8045944848bb17e739c.jpg)

![73ab0eda4e18b785bd53d87268e59491e0b695f564ede3345d3d9a61b505d0ee.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/73ab0eda4e18b785bd53d87268e59491e0b695f564ede3345d3d9a61b505d0ee.jpg)

![8cd92be31cb06b4f3508da1f9449d3102788c52b6ff6451ce4154d1e954c650d.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/8cd92be31cb06b4f3508da1f9449d3102788c52b6ff6451ce4154d1e954c650d.jpg)

![8cef0f5a3dc72a97f585629284a68db81e8b2112c5243d906cec10f8402eff6c.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/8cef0f5a3dc72a97f585629284a68db81e8b2112c5243d906cec10f8402eff6c.jpg)

![a8accce83f8939721e5fb86a60666af470054fb5859b8c84822bbab140264f6d.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/a8accce83f8939721e5fb86a60666af470054fb5859b8c84822bbab140264f6d.jpg)

![adea4d6a988c78f59daf1439d18910917311711d139b4e887a118842aeb29f0d.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/adea4d6a988c78f59daf1439d18910917311711d139b4e887a118842aeb29f0d.jpg)

![de3fbc29a5d58aec096eb91e8fb83c28581b6959133ae899e7ab298ab1c1a342.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/de3fbc29a5d58aec096eb91e8fb83c28581b6959133ae899e7ab298ab1c1a342.jpg)

![e3d2b4ced600fa56575cddac21ab40827a982a78458628636ba8675cd3689273.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/e3d2b4ced600fa56575cddac21ab40827a982a78458628636ba8675cd3689273.jpg)

![e851c6e1474ae79b15edeb322d88aa421ec41161e03ff2e2aad4255515744ba1.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/e851c6e1474ae79b15edeb322d88aa421ec41161e03ff2e2aad4255515744ba1.jpg)

![fbe55d38b9ae7b9cd9eee42495c2c7e7936f24fea8bdf18caea351d1db5bff9d.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/images/fbe55d38b9ae7b9cd9eee42495c2c7e7936f24fea8bdf18caea351d1db5bff9d.jpg)

### Tables

![033c785eae015b039f6f7f42e8fdb9a3498bbc064f1b4ce9322fdad32f08561a.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/033c785eae015b039f6f7f42e8fdb9a3498bbc064f1b4ce9322fdad32f08561a.jpg)

![0d4ea76b616a4fdb88596dcd6f8a725d9ac6fcfffc8d9e201cc491f8e37109c1.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/0d4ea76b616a4fdb88596dcd6f8a725d9ac6fcfffc8d9e201cc491f8e37109c1.jpg)

![294ecd99144c0c3984b9fba3c4c991e0d175b7793e7e381a64b141a53297d54c.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/294ecd99144c0c3984b9fba3c4c991e0d175b7793e7e381a64b141a53297d54c.jpg)

![2ed2b03b98da73f7c825a1d589edeadfc9c10ea0a93a191077753c25016e84e6.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/2ed2b03b98da73f7c825a1d589edeadfc9c10ea0a93a191077753c25016e84e6.jpg)

![37ceb6127e71c97d3d541f0c360ef2b78036d2a60600ec60abb4edea1fdcfd14.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/37ceb6127e71c97d3d541f0c360ef2b78036d2a60600ec60abb4edea1fdcfd14.jpg)

![47a57c8983866624573a6d9808fdfb9973e0ef18892372c318dd4f43badcab5d.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/47a57c8983866624573a6d9808fdfb9973e0ef18892372c318dd4f43badcab5d.jpg)

![5734eb950d93c9e980d611ecf9ef8fe21e1f2f6965fb082c7356cc4a96ad806e.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/5734eb950d93c9e980d611ecf9ef8fe21e1f2f6965fb082c7356cc4a96ad806e.jpg)

![5f5015f08f4184a55ff33a7cbcde96391f6746d652fffc4d0df2045395aa3b34.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/5f5015f08f4184a55ff33a7cbcde96391f6746d652fffc4d0df2045395aa3b34.jpg)

![631dc0ee630184c7fd875ae4a8198bead855e8155b8ecd509b03921078e937d1.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/631dc0ee630184c7fd875ae4a8198bead855e8155b8ecd509b03921078e937d1.jpg)

![772f12f104ce03ae9c0766a426319949d8c44ce1d5d6c0a677c727f353e18e8c.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/772f12f104ce03ae9c0766a426319949d8c44ce1d5d6c0a677c727f353e18e8c.jpg)

![838ef7929b284a8d53ba82d7c987acb52313ee637cec6e9e509524dd5274f0d7.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/838ef7929b284a8d53ba82d7c987acb52313ee637cec6e9e509524dd5274f0d7.jpg)

![83f2a6a76b47b42bcea4ce58123ef30c5fa495ac59d2154da476a0924c2f0239.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/83f2a6a76b47b42bcea4ce58123ef30c5fa495ac59d2154da476a0924c2f0239.jpg)

![8febf0df525e633c8bbd24ee1feaa29797759e4535918dde107002e861394cd1.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/8febf0df525e633c8bbd24ee1feaa29797759e4535918dde107002e861394cd1.jpg)

![9b89e0e1a26d06d4d3d59f6399e9ff07ddba90435ba5adc395659ed2cd6f5511.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/9b89e0e1a26d06d4d3d59f6399e9ff07ddba90435ba5adc395659ed2cd6f5511.jpg)

![9f8d9e493f051cbb745264a91296a061b4cbc105eb55f3c7a0b6c3f4964baf2a.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/9f8d9e493f051cbb745264a91296a061b4cbc105eb55f3c7a0b6c3f4964baf2a.jpg)

![a240dde82f3bdf5dec9bf2296f5ed950ba249c12f5b0230a4b5a95f916cc6cdc.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/a240dde82f3bdf5dec9bf2296f5ed950ba249c12f5b0230a4b5a95f916cc6cdc.jpg)

![b0df3fc3faef1d68e2dd7b1151044d2b2db640b8f3739dbdc9d5c6317f3ea0df.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/b0df3fc3faef1d68e2dd7b1151044d2b2db640b8f3739dbdc9d5c6317f3ea0df.jpg)

![b4e001a370d772d526a1905f6e913b46616f2cc46bf17d018cb40de8b17103f5.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/b4e001a370d772d526a1905f6e913b46616f2cc46bf17d018cb40de8b17103f5.jpg)

![b9712462db7b4ef5e354290908a2e43d30fc83d05b40126915d1fd32221f38f8.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/b9712462db7b4ef5e354290908a2e43d30fc83d05b40126915d1fd32221f38f8.jpg)

![d2ef6dcf0a64daea951d0f39a0ceb65eaefc6c5967f7118eb05e81a4dce9a471.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/d2ef6dcf0a64daea951d0f39a0ceb65eaefc6c5967f7118eb05e81a4dce9a471.jpg)

![d5d82b3dcc5efb1b2e43865ee950a278550c2fa46b34d526b06b8e5a9a76eca7.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/d5d82b3dcc5efb1b2e43865ee950a278550c2fa46b34d526b06b8e5a9a76eca7.jpg)

![ddcca026d598c7972d28e22072e456a4c052b689a6549500c5fe5ee54eb426d0.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/ddcca026d598c7972d28e22072e456a4c052b689a6549500c5fe5ee54eb426d0.jpg)

![ec3a993a5bad17fed2521ff2f23039dc24c2729359232703dd40c25f68608989.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/ec3a993a5bad17fed2521ff2f23039dc24c2729359232703dd40c25f68608989.jpg)

![efddba765f96f438adde0a090484a282cc4a1c3731e928441e512ec8a565dca0.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/efddba765f96f438adde0a090484a282cc4a1c3731e928441e512ec8a565dca0.jpg)

![f298098b1a168fc760a18f1d2e8bba287178e74cd8b63ac7318999bd2c07b767.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/f298098b1a168fc760a18f1d2e8bba287178e74cd8b63ac7318999bd2c07b767.jpg)

![f5145219776616561baf48fb12709f5f08d9fedc1e9f3ef760a575327bcc5550.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/f5145219776616561baf48fb12709f5f08d9fedc1e9f3ef760a575327bcc5550.jpg)

![f6343698f6dc36581e9a10346fef76cd43b56160f7aea910a7aaa6c24f725195.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/f6343698f6dc36581e9a10346fef76cd43b56160f7aea910a7aaa6c24f725195.jpg)

![fb6cba9e38bfdab15ee677a238fff5fddd372248e79ac9c830aae8f0169d84dc.jpg](../icml_results/3336_SPHERE_%20Mitigating%20the%20Loss%20of%20Spectral%20Plasticity%20in%20Mixture-of-Experts%20for%20Deep%20Reinforcement%20Lear/tables/fb6cba9e38bfdab15ee677a238fff5fddd372248e79ac9c830aae8f0169d84dc.jpg)

<a id="paper-3298"></a>

## UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios


### Images

![01a4ae5c48352033ca19619e377d654b2fae08022c48adb30247e94af39f3663.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/01a4ae5c48352033ca19619e377d654b2fae08022c48adb30247e94af39f3663.jpg)

![1ac4160a0897e0f3d164eb933133fe91a460af6acfab06b6539ef085afb824e0.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/1ac4160a0897e0f3d164eb933133fe91a460af6acfab06b6539ef085afb824e0.jpg)

![2ab3dc5267c32eba9eadce366dc415a8e8b3f64e7da69f35edd1d2869d61f98d.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/2ab3dc5267c32eba9eadce366dc415a8e8b3f64e7da69f35edd1d2869d61f98d.jpg)

![303e5cb94c2e34fe7aa50df3fba1de4188c39a2cbdc007df713d0ab009da4605.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/303e5cb94c2e34fe7aa50df3fba1de4188c39a2cbdc007df713d0ab009da4605.jpg)

![32e0282b1601013b95a1cdf5f65c7b874a86b49c9d5406de754c675b5b1b4c3d.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/32e0282b1601013b95a1cdf5f65c7b874a86b49c9d5406de754c675b5b1b4c3d.jpg)

![3be4521c7fc9b8b06432c7635b26dabc01b89c5068ff2b5868ff4274b3a6b395.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/3be4521c7fc9b8b06432c7635b26dabc01b89c5068ff2b5868ff4274b3a6b395.jpg)

![4843c1767213e3685cb289d39ff450f7922ea2c14faf45dbdad8d774c64898f3.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/4843c1767213e3685cb289d39ff450f7922ea2c14faf45dbdad8d774c64898f3.jpg)

![521fb767370c09f429c67862c0c67c8c64086ef0f776b970d587258cd24b5a8e.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/521fb767370c09f429c67862c0c67c8c64086ef0f776b970d587258cd24b5a8e.jpg)

![54653057776dafe052734cc3969ff151ee538e3214017c3fbe6346bda639d203.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/54653057776dafe052734cc3969ff151ee538e3214017c3fbe6346bda639d203.jpg)

![5dcb9062e61defbea019ef8bcd8424437f5a1fdcc2b762a9d7c372c30712376c.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/5dcb9062e61defbea019ef8bcd8424437f5a1fdcc2b762a9d7c372c30712376c.jpg)

![6a2db56ea1d67035f514a5290b20bc04260ea1d281767f042944d6f1078a2a9c.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/6a2db56ea1d67035f514a5290b20bc04260ea1d281767f042944d6f1078a2a9c.jpg)

![6eed7c8178fed828f16496e219f43f7a28c78b40493f31486336efcfaf5505ed.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/6eed7c8178fed828f16496e219f43f7a28c78b40493f31486336efcfaf5505ed.jpg)

![80ebd0c90b7dcb1c5f6dff14e4346d76bbbb3f1620cadd9e08b6abdc77cc884d.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/80ebd0c90b7dcb1c5f6dff14e4346d76bbbb3f1620cadd9e08b6abdc77cc884d.jpg)

![81efee1f451c2062fe3a9a3138d46905e1e94f6db6a29a1b7f9957898944db83.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/81efee1f451c2062fe3a9a3138d46905e1e94f6db6a29a1b7f9957898944db83.jpg)

![842bddb037c393752483b08dfbae3069e79fd08ed403fa29c67a776cf458d9a4.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/842bddb037c393752483b08dfbae3069e79fd08ed403fa29c67a776cf458d9a4.jpg)

![aa52e30a623995c988f9a84780279f20a52cddfd3683b663b8fa9506902c2b62.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/aa52e30a623995c988f9a84780279f20a52cddfd3683b663b8fa9506902c2b62.jpg)

![c2c0cc2e09284d4bae366e99ef4562dfc75932f2e6c03a30fc2a9f25a7a1472f.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/c2c0cc2e09284d4bae366e99ef4562dfc75932f2e6c03a30fc2a9f25a7a1472f.jpg)

![f6c15ab9e9fedb02501fda7829c67d0423b26a658c5ec05cd29684b4d2e69b9b.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/images/f6c15ab9e9fedb02501fda7829c67d0423b26a658c5ec05cd29684b4d2e69b9b.jpg)

### Tables

![1be2ae9b7f95d94b149f9485919766b36b1ab033c69f5a1944f09a60762b3a6d.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/1be2ae9b7f95d94b149f9485919766b36b1ab033c69f5a1944f09a60762b3a6d.jpg)

![319857a409dc4431885607114ca7a12c2e511a244262a874f112b144706540ef.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/319857a409dc4431885607114ca7a12c2e511a244262a874f112b144706540ef.jpg)

![411185ea1caa8c1482ed9bccbab527059175aa09b713f7eaea9bb5f1769dc7c8.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/411185ea1caa8c1482ed9bccbab527059175aa09b713f7eaea9bb5f1769dc7c8.jpg)

![51085121059a3989d56fdee29cbc0e8aaa3feca8c9487dc6f738dd98b51e08d6.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/51085121059a3989d56fdee29cbc0e8aaa3feca8c9487dc6f738dd98b51e08d6.jpg)

![5dbedf31bfa5007089f85dbcd66ed27535c46421124ccc0fcf7189bef179a95e.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/5dbedf31bfa5007089f85dbcd66ed27535c46421124ccc0fcf7189bef179a95e.jpg)

![6ee5dfaa1ecfcc4cd853393808b8ac690223f22171b0c27fbb5c5e7731e174a7.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/6ee5dfaa1ecfcc4cd853393808b8ac690223f22171b0c27fbb5c5e7731e174a7.jpg)

![96ae6613d1f84ca3d7f933ff1aacb0398a7216864ccbe95bf227291b2e8507eb.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/96ae6613d1f84ca3d7f933ff1aacb0398a7216864ccbe95bf227291b2e8507eb.jpg)

![d365d760dcefa2029e55bcb165c5b159e0ef7560d360b7842c111f7331efef99.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/d365d760dcefa2029e55bcb165c5b159e0ef7560d360b7842c111f7331efef99.jpg)

![ef0d3d848d3d528c172bc3eecfe435ce506bc8d83aae8eb8924921bc8507523f.jpg](../icml_results/3337_UltraHorizon_%20Benchmarking%20Agent%20Capabilities%20in%20Ultra%20Long-Horizon%20Scenarios/tables/ef0d3d848d3d528c172bc3eecfe435ce506bc8d83aae8eb8924921bc8507523f.jpg)

<a id="paper-3299"></a>

## MICE-Bench: A Challenging and Comprehensive Benchmark for Multi-Reference Image Creation and Editing


### Images

![009354adcf47c14f5eca1aef66042c534fdf88509c721727406c9631266d92d2.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/009354adcf47c14f5eca1aef66042c534fdf88509c721727406c9631266d92d2.jpg)

![0792b2e2054ff76139d7c0751e0760e93f2ece960d684a81e9a9969b7eb60e09.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/0792b2e2054ff76139d7c0751e0760e93f2ece960d684a81e9a9969b7eb60e09.jpg)

![2745da6988fa2d1ea9432ba5fb3d0b8b773cae2418fa041d410999798b09a199.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/2745da6988fa2d1ea9432ba5fb3d0b8b773cae2418fa041d410999798b09a199.jpg)

![2e95fc91ba6e84c06a4870378596955e626efe0048ca27b487fcd75f31b3d2d7.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/2e95fc91ba6e84c06a4870378596955e626efe0048ca27b487fcd75f31b3d2d7.jpg)

![35e753278a81e64226a0b8b1ffec8ded15af21351a49f5794b357a5be6a52b71.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/35e753278a81e64226a0b8b1ffec8ded15af21351a49f5794b357a5be6a52b71.jpg)

![379959a4ce42b30d6ae52efecd2c11a62f2430441c1be1e878e560befe412718.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/379959a4ce42b30d6ae52efecd2c11a62f2430441c1be1e878e560befe412718.jpg)

![42b48544809fceaa2db388346954d05a02f83a34f391cc3f271cace52bdb4fd3.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/42b48544809fceaa2db388346954d05a02f83a34f391cc3f271cace52bdb4fd3.jpg)

![4e019b11a4942ab992db1628bbb19b592c182453a691676fa269517a207de9b3.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/4e019b11a4942ab992db1628bbb19b592c182453a691676fa269517a207de9b3.jpg)

![4fe6216dc733fac453941574fa2551bf7fffb769d69e6ca5e2988e651bfb4a96.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/4fe6216dc733fac453941574fa2551bf7fffb769d69e6ca5e2988e651bfb4a96.jpg)

![508aeae85be5ec36adf446df46e0a1e247a0b2cf3bc67ba9b556c3fde124807b.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/508aeae85be5ec36adf446df46e0a1e247a0b2cf3bc67ba9b556c3fde124807b.jpg)

![5e27410cd11bbcdc5db9a6006da1c439f59f31eef9c426c7eaf5f8a533f6844b.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/5e27410cd11bbcdc5db9a6006da1c439f59f31eef9c426c7eaf5f8a533f6844b.jpg)

![601ed7c721afc756b55fd4890ee944ff1595e3fe250faccec76bc7dbc262d9ac.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/601ed7c721afc756b55fd4890ee944ff1595e3fe250faccec76bc7dbc262d9ac.jpg)

![631720fdfdc702617c129922d848f691a98e7361f00111eb7c03c463489de115.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/631720fdfdc702617c129922d848f691a98e7361f00111eb7c03c463489de115.jpg)

![765e300f4b9ede97cda33b6e1992b64d693a25d91e2fd47bded316216d961926.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/765e300f4b9ede97cda33b6e1992b64d693a25d91e2fd47bded316216d961926.jpg)

![79d87da74cc328afdf4412a4659403bf037b5ab443d9a4684527671747ba836b.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/79d87da74cc328afdf4412a4659403bf037b5ab443d9a4684527671747ba836b.jpg)

![7c17cfd6203566bd6a807a56538d4b1fdd3aa3c7a08b32fbe618ffd0933a82ea.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/7c17cfd6203566bd6a807a56538d4b1fdd3aa3c7a08b32fbe618ffd0933a82ea.jpg)

![8b679f3edd0da58c75cf28febcaf1fe2823eb4055f495f02ff5f0915c52f0684.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/8b679f3edd0da58c75cf28febcaf1fe2823eb4055f495f02ff5f0915c52f0684.jpg)

![95b4b87c06935a1f3b7708d763e539672a1dff149c948741e6e8ac4e58a34660.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/95b4b87c06935a1f3b7708d763e539672a1dff149c948741e6e8ac4e58a34660.jpg)

![9eccc960c1818b41e5e7890cfc4f3b2caee9bec5dba6f4c21cacc4b41aadf434.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/9eccc960c1818b41e5e7890cfc4f3b2caee9bec5dba6f4c21cacc4b41aadf434.jpg)

![a48c51c2e710a3a6ced9241f79800d20d0a29b9e3065e6eab0ecd54776648154.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/a48c51c2e710a3a6ced9241f79800d20d0a29b9e3065e6eab0ecd54776648154.jpg)

![b2bf8a5e2ad0ff10d68c8521a0c2805c3833c65a7f2e0b98279eda6afbf6d7ab.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/b2bf8a5e2ad0ff10d68c8521a0c2805c3833c65a7f2e0b98279eda6afbf6d7ab.jpg)

![b7b9d51bcee17fe1fc983db57094e7543f5c5f1b91b3a4fb41f176b0f61eb03d.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/b7b9d51bcee17fe1fc983db57094e7543f5c5f1b91b3a4fb41f176b0f61eb03d.jpg)

![bad375dc07a539ffe6345f3da804512502b6953380b183a044a2764bdfdcc06f.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/bad375dc07a539ffe6345f3da804512502b6953380b183a044a2764bdfdcc06f.jpg)

![bd3ff9937ba7e4abc6ee61dfdd869ac6541b4cd911c41c72eca1b8a997723371.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/bd3ff9937ba7e4abc6ee61dfdd869ac6541b4cd911c41c72eca1b8a997723371.jpg)

![c897c0c8f8d88c2ef041d5f0e2a32358872a889125da1915f09abba6eb310613.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/c897c0c8f8d88c2ef041d5f0e2a32358872a889125da1915f09abba6eb310613.jpg)

![c9c9008abc397339f6fff1464639d726be7f54e0bbcd7c2e59882291e66433f3.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/c9c9008abc397339f6fff1464639d726be7f54e0bbcd7c2e59882291e66433f3.jpg)

![cdfaa4ad296ac8d0c6985c51f151a736793fba5ea211f920a483e71c1131ec9a.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/cdfaa4ad296ac8d0c6985c51f151a736793fba5ea211f920a483e71c1131ec9a.jpg)

![d12a3cd7255f94afe1b05900de3d46e8550c75428d80ad68e49d2cd10249a235.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/d12a3cd7255f94afe1b05900de3d46e8550c75428d80ad68e49d2cd10249a235.jpg)

![e07da421bbe5e9eb253de81d21a96a0f0e2e90d48192e37ace0d1fca54c4a9dc.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/e07da421bbe5e9eb253de81d21a96a0f0e2e90d48192e37ace0d1fca54c4a9dc.jpg)

![eda200c22e08c2cc3e66d2704f2a3851cd0e9f35d598fa752430fad2907fc98e.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/eda200c22e08c2cc3e66d2704f2a3851cd0e9f35d598fa752430fad2907fc98e.jpg)

![edd48870b6c350c7f0aea7cc513c3bd18de6a1abeaba652861d0aa94ea3522bb.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/edd48870b6c350c7f0aea7cc513c3bd18de6a1abeaba652861d0aa94ea3522bb.jpg)

![f24c1cc06204c263a875d5ae3daf9bde3809568c48b755bf47612278cbbe24c5.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/f24c1cc06204c263a875d5ae3daf9bde3809568c48b755bf47612278cbbe24c5.jpg)

![fe4bf4c9b0945d125c4c52617c7a653015cb6a5c5ac899060f04e45d892d00cd.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/images/fe4bf4c9b0945d125c4c52617c7a653015cb6a5c5ac899060f04e45d892d00cd.jpg)

### Tables

![0767cb2ca08460eaa57fc0ddfe8cee14990297302d2eadad597bffc4db0aebfa.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/0767cb2ca08460eaa57fc0ddfe8cee14990297302d2eadad597bffc4db0aebfa.jpg)

![49e1cd8e5c39daec491301e66057b1212a2f11d281c6ce2f2addad0ea39e0c60.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/49e1cd8e5c39daec491301e66057b1212a2f11d281c6ce2f2addad0ea39e0c60.jpg)

![91bf03d446fb5d58382ee39c7be744910dfc7be704a77c2c95e0b7b688ffb3ea.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/91bf03d446fb5d58382ee39c7be744910dfc7be704a77c2c95e0b7b688ffb3ea.jpg)

![9ffbb911a58b3b6206122e883a3a483570a84f98fde2bf969a1dc81cfa4c3a03.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/9ffbb911a58b3b6206122e883a3a483570a84f98fde2bf969a1dc81cfa4c3a03.jpg)

![af53fad6f670d6ce5d5fde6e707b9208044064a410e792d2641aa5aeebf01d93.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/af53fad6f670d6ce5d5fde6e707b9208044064a410e792d2641aa5aeebf01d93.jpg)

![c53014bd9a4b743e15fac529403dd74de50868c8f6becddbb3af24bdb5919586.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/c53014bd9a4b743e15fac529403dd74de50868c8f6becddbb3af24bdb5919586.jpg)

![e39518d292cc2b687dd33abe6dd22ff9fe4f419c6c6572ffa38da6a5b79a4649.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/e39518d292cc2b687dd33abe6dd22ff9fe4f419c6c6572ffa38da6a5b79a4649.jpg)

![e7e7baae3eff4727272ec48352c6fcc549380b698a0ec1a6c076a55ee11f66ac.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/e7e7baae3eff4727272ec48352c6fcc549380b698a0ec1a6c076a55ee11f66ac.jpg)

![f081bf05d11f6687660bb85ef79eb3926787ce768d0f6406bb3d57d8dd2b448c.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/f081bf05d11f6687660bb85ef79eb3926787ce768d0f6406bb3d57d8dd2b448c.jpg)

![f4a9dadfa906c891179e6e830be7b9fee5f49c75eab6686494e77e77e0d7cf4c.jpg](../icml_results/3338_MICE-Bench_%20A%20Challenging%20and%20Comprehensive%20Benchmark%20for%20Multi-Reference%20Image%20Creation%20and%20Editing/tables/f4a9dadfa906c891179e6e830be7b9fee5f49c75eab6686494e77e77e0d7cf4c.jpg)

<a id="paper-3300"></a>

## 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere

### Images

![0b6825c7ae81b49de8d6aac7e6f452fa1bc8c75d0e55ef5aa4d36465495fe509.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/0b6825c7ae81b49de8d6aac7e6f452fa1bc8c75d0e55ef5aa4d36465495fe509.jpg)

![3fd49ad55a165bd69d70398fae14d7eddeef926d5d082deeb445d3edaeb6f21a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/3fd49ad55a165bd69d70398fae14d7eddeef926d5d082deeb445d3edaeb6f21a.jpg)

![4a5ddb8b846bb26a08e18674aa61e8ece2c88611801ae40afa33f19fb4c64566.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/4a5ddb8b846bb26a08e18674aa61e8ece2c88611801ae40afa33f19fb4c64566.jpg)

![5b7b0ab20d89d55dc71a7f5f22f8a2257fb36a91b1c23adebf2d3618a870ae59.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/5b7b0ab20d89d55dc71a7f5f22f8a2257fb36a91b1c23adebf2d3618a870ae59.jpg)

![9a9b431cb08bfa3dc0af0e53a42c636877344f22206a7937203b94f4f41e736c.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/9a9b431cb08bfa3dc0af0e53a42c636877344f22206a7937203b94f4f41e736c.jpg)

![f3c1542adde908b0bb59a4633f577138b76b95f7ab8058e65bf9eda03524a849.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/f3c1542adde908b0bb59a4633f577138b76b95f7ab8058e65bf9eda03524a849.jpg)

![f764fab6664bb25aa7737c1a7133555b011ef39026f4fe5c49ca8d02eb7eb744.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/images/f764fab6664bb25aa7737c1a7133555b011ef39026f4fe5c49ca8d02eb7eb744.jpg)

### Tables

![1ad7cdbad9e105d2634505f53f1de3d476fc6a45172522c9b59390049a1afd47.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/1ad7cdbad9e105d2634505f53f1de3d476fc6a45172522c9b59390049a1afd47.jpg)

![228cf6825b113a0b97d505e7e1ceeec1b230fbe6aa640d53fa85711692b4185a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/228cf6825b113a0b97d505e7e1ceeec1b230fbe6aa640d53fa85711692b4185a.jpg)

![4244d6fdba656eb7a350544c65dffb38e88c4a02734522994dda60f487c059dc.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/4244d6fdba656eb7a350544c65dffb38e88c4a02734522994dda60f487c059dc.jpg)

![42e1ccfe9482efbf1d2545abf25548df0958347f47733f37b95d56f969f261cd.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/42e1ccfe9482efbf1d2545abf25548df0958347f47733f37b95d56f969f261cd.jpg)

![67523dfd0606df80d9be1c702f24dff8ed62c463905b38df303b7c15c21bd33a.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/67523dfd0606df80d9be1c702f24dff8ed62c463905b38df303b7c15c21bd33a.jpg)

![94d136faac8c4045a1545ed5f29cc95dacc9f73f35122eb8bbbfb8b4bb39ca0b.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/94d136faac8c4045a1545ed5f29cc95dacc9f73f35122eb8bbbfb8b4bb39ca0b.jpg)

![9ecffef678901d0fe2377d2ec7e0ea6b9b473207ed570b39be6d7f35c8fff01f.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/9ecffef678901d0fe2377d2ec7e0ea6b9b473207ed570b39be6d7f35c8fff01f.jpg)

![b8556d35b78d2e5c43a3f59109cf33991566b18d015a4b4634fa3ba7bda46581.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/b8556d35b78d2e5c43a3f59109cf33991566b18d015a4b4634fa3ba7bda46581.jpg)

![cc8003cf7727d5fc01fe9e67681a88cbf8cc9b03a5446a3c3d533daec69d7414.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/cc8003cf7727d5fc01fe9e67681a88cbf8cc9b03a5446a3c3d533daec69d7414.jpg)

![d2559fbfd6d92c0fd1f733d734f69137bb0d5f263fa9f1eca092b5ee90f9ae88.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/d2559fbfd6d92c0fd1f733d734f69137bb0d5f263fa9f1eca092b5ee90f9ae88.jpg)

![d9691c05e099a0c117a3c7145bba1e040a18d81136384f4b4eaad5716b43ba8e.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/d9691c05e099a0c117a3c7145bba1e040a18d81136384f4b4eaad5716b43ba8e.jpg)

![ea5ec5959b11d8368646c9572615263fae44cb3ae002fcb0c77c00852bf69df4.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/ea5ec5959b11d8368646c9572615263fae44cb3ae002fcb0c77c00852bf69df4.jpg)

![f202a1a6b95014a7b46620bfdfbfadf8ca0558949dcf84e3165b4a8838c0ccea.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/f202a1a6b95014a7b46620bfdfbfadf8ca0558949dcf84e3165b4a8838c0ccea.jpg)

![fffe3b8f4c893945aeaf337f0c074e91022e61ca7fffa6d3cbfdb6cf8dd842e3.jpg](../icml_results/3339_4RC_%204D%20Reconstruction%20via%20Conditional%20Querying%20Anytime%20and%20Anywhere/tables/fffe3b8f4c893945aeaf337f0c074e91022e61ca7fffa6d3cbfdb6cf8dd842e3.jpg)
