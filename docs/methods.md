# How epistemic diversity is measured

Every measure here comes from existing work on creativity, ecology or text generation. Language models are never asked whether an answer is creative or good. Embedding models are used as fixed rulers to place answers in a space of meaning, and each semantic measure is checked against at least two embedding models and one measure that uses no model at all.

The unit of comparison is a group of answers to the same pair: humans, AI with no persona, and AI with each of the four personas. Groups are compared per direction (a → b, b → a) and pooled.

All measures are in `analysis/diversity.py`.

## For each answer

**Rarity.** How far an answer sits from what other people wrote for the same pair: one minus its mean cosine similarity to the human answers (leaving itself out). This is the continuous form of the originality score used in divergent thinking tasks, where an idea counts as original when few people give it (Silvia et al. 2008), measured in meaning space as in SemDis (Beaty and Johnson 2021). AI answers are scored against the human crowd, so the question becomes how often AI lands where people rarely go.

## For each group

**Mean pairwise distance.** The average cosine distance between all answers in the group. The same idea as the Divergent Association Task (Olson et al. 2021), applied to bridges instead of single words. The players' own warm-up gives a DAT baseline for each person.

**Vendi score.** The effective number of distinct answers in the group, from the eigenvalues of the similarity matrix (Friedman and Dieng 2023). Ten answers that all say the same thing score close to 1; ten unrelated answers score close to 10. It does not need clusters or a threshold.

**Idea families (Hill numbers).** Answers are grouped into families by agglomerative clustering at cosine distance 0.4, the same setting as False Epistemic Redundancy. The number of families is then reported as Hill numbers (Hill 1973; Jost 2006): q = 0 counts families, q = 1 is the exponential of Shannon entropy, q = 2 is the inverse Simpson index. Higher q gives less weight to rare families, so the three together show whether diversity lives in the tail.

**Distinct-n.** The share of word unigrams and bigrams that are unique across the group (Li et al. 2016). Uses no model, so it guards against an embedding model that happens to see two different ideas as close.

**Stepping stones.** When players or models list the ideas they passed through, the mean Jaccard overlap between their sets of stones, and the stones that appear most often. Shared central stones are the hubs seen in human wayfinding on Wikipedia (West and Leskovec 2012). The question is whether AI routes through fewer, more central hubs.

**Direction shift.** The cosine distance between the centre of the a → b answers and the centre of the b → a answers for the same pair. A large shift means the starting point shapes the path.

## Comparing groups

Diversity measures depend on group size, so groups are compared at equal size by repeated subsampling, with bootstrap intervals over answers and over pairs. The main comparisons:

1. Humans vs AI with no persona: does AI cover less of the space?
2. AI with no persona vs each persona: do personas widen it, and in which measure?
3. Players who asked for the AI bridge, before vs after: does seeing it pull the next answers toward it? This is the collective homogenisation effect found by Doshi and Hauser (2024) and Anderson et al. (2024).

## Human judgement

Rarity alone rewards nonsense. A blind sample of answers (human and AI mixed, source hidden) is rated by people for how surprising and how fitting each bridge is, following the consensual assessment technique (Amabile 1982). Creativity is then read as rare and fitting together.

## Reading the answers

Players connect ideas in at least three styles so far: a story or sentence, a chain of words, and filling the gap so that the two ideas start and end the sentence (for example "a bee → *visits flowers next to* → a mirror", an invented illustration). With `--with-ends`, each answer is measured together with its two ideas, so the third style reads as a full sentence. Results are reported both ways.

The welcome text changed once, on 14 Nov 2026 at about 00:30 Paris time (see `instructions_versions` in `data/pairs.json`). Answers are tagged by version through their timestamp, and AI generations use the version they are compared against.

## References

- Amabile, T. M. (1982). Social psychology of creativity: A consensual assessment technique. *Journal of Personality and Social Psychology*, 43(5), 997-1013.
- Anderson, B. R., Shah, J. H., and Kreminski, M. (2024). Homogenization effects of large language models on human creative ideation. *Creativity and Cognition 2024*.
- Beaty, R. E., and Johnson, D. R. (2021). Automating creativity assessment with SemDis. *Behavior Research Methods*, 53, 757-780.
- Doshi, A. R., and Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. *Science Advances*, 10(28).
- Friedman, D., and Dieng, A. B. (2023). The Vendi score: A diversity evaluation metric for machine learning. *Transactions on Machine Learning Research*.
- Hill, M. O. (1973). Diversity and evenness: A unifying notation and its consequences. *Ecology*, 54(2), 427-432.
- Jost, L. (2006). Entropy and diversity. *Oikos*, 113(2), 363-375.
- Li, J., Galley, M., Brockett, C., Gao, J., and Dolan, B. (2016). A diversity-promoting objective function for neural conversation models. *NAACL 2016*.
- Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., and Webb, M. E. (2021). Naming unrelated words predicts creativity. *PNAS*, 118(25).
- Silvia, P. J., et al. (2008). Assessing creativity with divergent thinking tasks. *Psychology of Aesthetics, Creativity, and the Arts*, 2(2), 68-85.
- West, R., and Leskovec, J. (2012). Human wayfinding in information networks. *WWW 2012*.
