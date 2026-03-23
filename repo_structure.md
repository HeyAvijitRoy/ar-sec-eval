# ar-sec-eval - Repository Structure

This document lists all files currently present in the repository in tree format.

```text
ar-sec-eval/
|-- .dockerignore
|-- .gitignore
|-- adv1_collision_attack.py
|-- adv1_gan_attack.py
|-- adv2_evasion_attack.py
|-- adv2_few_pixels_attack.py
|-- adv3_evaluation.ipynb
|-- adv3_robustness_check.py
|-- adv4_evaluation.ipynb
|-- adv4_information_extraction.py
|-- ar-works/
|   |-- AR-Phase1.md
|   |-- project_status.md
|   `-- T1.AR-AttackEffiencyAnalysis.md
|-- datasets/
|   |-- imagenette.py
|   `-- stanford_dogs.py
|-- Dockerfile
|-- evidence/
|   |-- env_versions.txt
|   `-- model_hashes.sha256
|-- experiments/
|   |-- hash_self_consistency.py
|   |-- make_inputs_sample.py
|   |-- run_nhash_evasion.py
|   |-- run_nhash_evasion_sweep.py
|   |-- RUNBOOK.md
|   `-- sanity_check.py
|-- fdeph_eval/
|   |-- analysis/
|   |   |-- attack_efficiency_analysis.ipynb
|   |   |-- figures/
|   |   |   |-- nhash_distance_vs_steps.png
|   |   |   |-- nhash_distance_vs_time.png
|   |   |   |-- nhash_success_rate_vs_steps.png
|   |   |   |-- nhash_success_rate_vs_time.png
|   |   |   `-- nhash_time_to_success_hist.png
|   |   |-- plotting.py
|   |   `-- tables/
|   |       |-- nhash_summary_stats.csv
|   |       |-- nhash_threshold_sweep.csv
|   |       `-- nhash_time_to_success.csv
|   |-- attacks/
|   |   `-- nhash_evasion_steps.py
|   `-- utils/
|       `-- structured_logger.py
|-- images/
|   |-- collision_example.png
|   |-- hash_change_example.png
|   |-- neural_hash_architecture.png
|   `-- transformations.png
|-- LICENSE
|-- logs/
|   |-- attack_steps_nhash_evasion___.csv
|   |-- attack_steps_nhash_evasion_mt50.csv
|   |-- attack_steps_nhash_evasion_mt500.csv
|   |-- attack_steps_nhash_evasion_mt500_T0.08.csv
|   |-- attack_steps_nhash_evasion_mt500_T0.10.csv
|   |-- attack_steps_nhash_evasion_mt500_T0.12.csv
|   |-- attack_steps_nhash_evasion_pilot50.csv
|   `-- attack_steps_nhash_evasion_pilot50___.csv
|-- losses/
|   |-- hinge_loss.py
|   |-- mse_loss.py
|   `-- quality_losses.py
|-- metrics/
|   `-- hamming_distance.py
|-- models/
|   `-- neuralhash.py
|-- README.md
|-- repo_files.list.txt
|-- repo_structure.md
|-- repo_structure.tree.txt
|-- requirements.txt
|-- rootless.Dockerfile
`-- utils/
    |-- compute_dataset_hashes.py
    |-- hashing.py
    |-- image_converter.py
    |-- image_processing.py
    |-- load_generator.py
    |-- logger.py
    |-- metrics.py
    |-- onnx2pytorch.py
    |-- training.py
    `-- transforms.py
```