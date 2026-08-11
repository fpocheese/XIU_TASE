# GRU sensor-delay development screen

This directory is a post-formal, frozen-policy diagnostic and is not included
in the primary 100-episode ablation table.

- Models: the selected Full and No-GRU seed-8301 Case-1/Case-2 actors.
- Evaluation seeds: 99001--99012, paired across methods and delays.
- Delay points: 1, 3, 5, and 10 simulation steps (50, 150, 250, 500 ms).
- Episodes: 12 per method/case/delay condition.
- Training/back-propagation/optimizer updates: none.

The screen was added to test whether the Full model's much lower Critic Loss
would become a frozen-policy advantage under longer observation delay. It did
not provide such evidence. In Case 1, Full cooperative success was
41.7%, 25.0%, 16.7%, and 0%, whereas No-GRU obtained
83.3%, 75.0%, 50.0%, and 0%. In Case 2 both methods degraded rapidly and
neither achieved strict cooperation. Therefore no 100-episode follow-up was
launched and this diagnostic is retained as a negative development result,
not promoted as a paper result.
