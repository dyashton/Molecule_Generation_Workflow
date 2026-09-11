# Literature criteria for CO₂ binding affinity (MGW re-rank)

**Caveat:** MGW does not predict CO₂ ΔG. Molecules come from the physchem + PFC heuristic reward. This document defines a **post-hoc literature checklist** used to re-rank those candidates for *highest binding-affinity proxy*, not synthesizability, regeneration energy, or process capacity.

Affinity-first ranking **does not penalize** high MW, high SCScore, many amines, azacrowns, or polar LogP — those are realism constraints, not affinity.

---

## A) Favor (affinity — soft score up)

| Criterion | Why | Operational check |
|-----------|-----|-------------------|
| Aliphatic 1°/2° amines (unhindered) | Direct N→C attack on CO₂ → carbamate / carbamic acid (MEA-type chemisorption). Steric hindrance at α-C weakens N–CO₂ bonding. | Count aliphatic 1°/2° amine N that are not α-branched (`[NX3;H2,H1]` on carbon, exclude hindered α). |
| High nucleophilicity / basicity at N | Stronger Brønsted bases bind CO₂ more tightly; EWGs next to N hurt. | Prefer alkyl amines; no α/β EWG required for score; Mordred basicity / MGW `raw_physchem` used only as tie-breaker. |
| Guanidine / amidine superbases | Organic superbases form very stable ionic / adduct products with CO₂. | SMARTS guanidine `N=C(N)N`, amidine `N=C(N)C`. |
| Multi-N cooperation (≥2 basic N) | One N forms carbamate while another accepts H⁺ (intersite pairing); more reactive sites per molecule. | Aliphatic amine count ≥ 2 (piperazine / polyamines). |
| Nearby H-bond donors (OH/NH near N) | Stabilizes zwitterion / carbamate (alkanolamine chemistry). | Aliphatic OH within short graph distance of an amine N. |
| Accessible lone pair (amine ≫ amide / aniline) | Amides and aromatic amines are poor CO₂ nucleophiles. | Score aliphatic amine / guanidine / amidine only; amide N is not a binding site. |

**Literature anchors (conceptual):** alkanolamine carbamate chemistry and steric/electronic SAR; guanidine/amidine superbases and switchable solvents; multi-amine cooperative capture. Industrial “easy-regeneration” hindered / tertiary amines are *weaker* affinity motifs and are not rewarded here.

---

## B) Avoid (instability — hard reject)

| Feature | Why |
|---------|-----|
| Sanitize / valence failure | Not a closed-shell isolable organic. |
| Peroxides / peracids (`[OX2][OX2]`) | Shock / heat sensitive. |
| Azides / diazo | Explosive / highly reactive. |
| Acyl / sulfonyl / phosphoryl halides; N/O/S–halogen | Hydrolytically reactive. |
| Isocyanates / isothiocyanates | Self-react / polymerize. |
| Epoxides / aziridines (3-membered rings) | Strain-driven ring opening. |
| Nitrate / nitrite esters, nitramines | Energetic. |
| Multi-fragment / salts | Keep single neutral organic (same as MGW hard gate). |

Soft liabilities that are **not** hard rejects on this pass: ester, amide, alkyne, furan, high MW, SCScore, rotatable bonds. They may appear in notes but must not exclude high-affinity polyamines.

---

## Ranking note

The script `scripts/rerank_co2_literature_top5.py` pools gate-ok MGW CO₂ molecules, hard-rejects section B, scores section A, then greedily picks five diverse SMILES (Morgan Tanimoto &lt; 0.7). Output: `co2_literature_top5.csv` (does not overwrite the presentation “presentable” `co2_success_top5.csv`).
