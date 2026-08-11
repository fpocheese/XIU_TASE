## Reviewer comment

> Then, it would be helpful to add an ablation investigation demonstrating
> the independent contribution of the trust-aware mechanism, GRU temporal
> encoder, and attention-residual backbone within ART-MAPPO architecture.

## Response

Thank you for this constructive suggestion. We added a controlled
single-component ablation study of the complete ART-MAPPO, removing,
respectively, (i) trust-modulated tactical exploration, (ii) the GRU temporal
encoder, and (iii) the attention-residual feature extractor. All variants use
the same three-dimensional engagement dynamics, reward, target assignment,
action bounds, optimizer, rollout budget, and three training seeds. For the
third ablation, the removed structured backbone is replaced by a
capacity-matched plain MLP, rather than by a smaller network.

The new experiment separates training-level and deployment-level evidence.
The training comparison reports episode return, critic loss, and policy
entropy as three-seed means with standard-deviation bands. The test comparison
disables all guided exploration and evaluates the frozen learned actors on
exactly 100 paired Monte Carlo episodes per variant and case. We report
target-coverage success, all-defender interception, strict cooperative
success under the 0.5-s group threshold, and the four terminal metrics
\(E_{\mathrm{co\text{-}time}}, E_n, E_{\mathrm{miss}}, E_t\).

This distinction is important for interpreting the trust-aware component.
Consistent with the manuscript definition, trust is used only during training
to modulate tactical exploration according to standardized episodic returns;
it is not an additional deployment-time controller. Its independent
contribution is therefore assessed primarily through reward acquisition,
inter-seed stability, and the entropy trajectory, with frozen-policy test
performance reported as the downstream effect of the resulting training
distribution. In contrast, the GRU and attention-residual backbone operate in
both training and deployment, so their contributions are discussed from both
optimization and final interception perspectives.

% AUDITED_QUANTITATIVE_RESULTS_TO_BE_INSERTED_FROM_FORMAL_CSV

We have added the experiment protocol, the training curves, the 100-episode
test results, and a corresponding discussion to the revised material. We also
provide the raw per-update and per-episode CSV files to make the comparison
fully traceable.
