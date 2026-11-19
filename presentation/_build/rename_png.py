import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "png")
M = {"chart_p_v0": "02_v0_synthetic_accuracy", "chart_p_e3": "03a_lme_retrieval_recall10_CI", "chart_p_e3abl": "03b_lme_ablations_vs_full_graph",
     "chart_p_e3q": "03c_lme_by_question_type", "chart_p_e3b": "04a_reranker_primary_delta", "chart_p_e3bgrid": "04b_reranker_all_configs_test",
     "chart_p_e2": "05_cnorm_error_signal_auroc", "chart_p_lc02": "06a_locomo_rrf_confirmation", "chart_p_lc03": "06b_locomo_multihop_all_retrievers",
     "chart_p_lc03f": "06c_locomo_multihop_paired_holm", "chart_p_qa": "07a_qa_f1_by_arm", "chart_p_qad": "07b_qa_paired_differences",
     "chart_p_led": "08_theory_ledger_counts", "chart_p_tl": "09_research_process_predictions"}
for f in os.listdir(P):
    b, e = os.path.splitext(f)
    if b in M: os.replace(os.path.join(P, f), os.path.join(P, "chart_" + M[b] + e))
    elif b.startswith("results_section_s"): os.replace(os.path.join(P, f), os.path.join(P, "section_" + b[-2:] + e))
