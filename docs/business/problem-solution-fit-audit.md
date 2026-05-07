# Problem-Solution Fit Audit — SkateLab

> Date: 2026-05-07
> Framework: 4-question PS Fit audit across Problem, Solution, and Fit phases.
> Sources: custdev-results.md, vision.md, positioning.md, landscape.md

---

## Executive Summary

| Phase | Rating | Key Risk |
|-------|--------|----------|
| 1. Problem Validation | **MODERATE** | N=3, no quantitative data |
| 2. Solution Validation | **MODERATE** | Built but untested with real users |
| 3. Fit Validation | **MODERATE** | Differentiation strong; 10x value theoretical |
| **Overall Verdict** | **ITERATE** | Close 3 critical gaps before investor conversations |

---

## Phase 1: Problem Validation

### Q1. Is this a real problem? (frequency, severity, evidence)

**Rating: MODERATE**

**Evidence FOR:**
- All 3 CustDev respondents (2 coaches, 1 athlete) confirm the problem occurs **every training session**.
- Time cost: 5–20 min/session on technique review (custdev-results.md, H1).
- Subjective evaluation causes real friction: "video doesn't always give an accurate picture," "arguments between coach and athlete" (custdev-results.md, H2).
- Existing tools (video, Kinovea) are partial solutions that leave the problem unsolved (custdev-results.md, Key Insight #5).

**Evidence AGAINST / GAPS:**
- **N=3.** The document itself warns: "N=3, неравномерность (2 тренера + 1 спортсмен), нет прототипа. Выводы качественные, не статистические."
- Coaches **undervalue their own time**. Respondent 1: "Для тренера — нет" (no monetary loss). Problem severity is higher for athletes than coaches.
- No frequency data beyond "every session" — no log studies, no time diaries.

---

### Q2. Is it painful enough? (willingness to pay, time/money cost)

**Rating: MODERATE (athletes) / WEAK (coaches)**

**Evidence FOR:**
- Athlete (Master of Sport) willing to pay **up to 10,000 ₽ one-time and 500–600 ₽/month** (custdev-results.md, Respondent 3).
- Financial impact calculation: athlete loses **15,000–35,000 ₽/month** from training inefficiency at 84,000 ₽/month base cost. ROI at 1,000 ₽/month is **15–35x** (custdev-results.md, Financial Impact).
- Athletes directly bear financial cost: "1 hour ~ 3,500 ₽", "slow correction = more training time".

**Evidence AGAINST / GAPS:**
- **Coaches are conditional payers.** Both coaches say they will pay **only if effectiveness is proven** (custdev-results.md, H3). They do not buy "technology"; they buy results. This creates a chicken-and-egg problem.
- Athlete WTP ceiling (500–600 ₽/mo ≈ **$6–7**) is at or below the proposed SaaS pricing (Entry 490 ₽, Pro 990 ₽). The Pro tier exceeds stated athlete WTP.
- **Coach pricing (1,500–3,500 ₽/mo) is completely unvalidated** by CustDev. No coach named a number.
- No actual transactions. Willingness to pay ≠ willingness to buy.

---

## Phase 2: Solution Validation

### Q3. Is our solution viable? (technical feasibility, does it solve root cause)

**Rating: MODERATE**

**Evidence FOR:**
- MVP is 100% complete with 65+ files, 279+ tests, ~12s pipeline for 14.5s video (vision.md, Current Status).
- Architecture is sound: FastAPI + SQLAlchemy + arq + Valkey + R2 + Vast.ai Serverless GPU dispatch.
- Core differentiators built: OOFSkate proxy features (no blade detection needed), CoM trajectory, anatomical Re-ID, Russian-language UI and recommendations, choreography planner with CSP solver (vision.md, Unique Differentiators).
- GPU-only inference enforced; CPU path forbidden.

**Evidence AGAINST / GAPS:**
- **No user testing with prototype.** CustDev "Next Steps" explicitly lists "Тестирование с прототипом (UX + доверие)" as unchecked.
- **No accuracy validation.** The pipeline outputs metrics (airtime, height, knee angles, rotation, landing quality), but there is **no benchmark against ground truth** (e.g., Omega-level reference, force plates, or expert coach consensus). We do not know if 3px shift from CorrectiveLens matters because we do not know absolute accuracy.
- **No efficacy proof.** The root cause of slow progress is "lack of objective data," but we have not proven that SkateLab's specific data accelerates progress. It is plausible, not validated.
- Tracking is known to degrade in real-world scenarios (black clothing, multiple people on ice). Memory notes: "Tracking degrades (skeleton jumps wrong person) → data-driven analysis." This means the product is not yet robust for uncontrolled training environments.

---

## Phase 3: Fit Validation

### Q4. Is it better than alternatives? (10x test, differentiation)

**Rating: MODERATE**

**Evidence FOR (Differentiation STRONG):**
- SkateLab is the **only** product combining: video ML analysis + Russian language + phone video input + 3D biomechanics + automatic recommendations + choreography planner (landscape.md, Differentiation Matrix).
- Direct competitors are either inaccessible (Omega: federations only, 14 cameras), research projects (JudgeAI-LutzEdge: 4 GitHub stars), paper-only (YourSkatingCoach), or lack SaaS/recommendations (Pose2Sim, Sports2D).
- Indirect competitors (Coach's Eye, HUDL) are manual annotation tools. SkateLab is automated ML pipeline — fundamentally different value proposition.
- Russian language is a genuine moat for Russia, Kazakhstan, Belarus, Latvia markets (landscape.md, Competitive Moats #2).
- Unique data moat: SkatingVerse (28K videos), AthletePose3D (1.3M frames).

**Evidence AGAINST / GAPS (10x test WEAK):**
- The **15–35x ROI** for athletes is a spreadsheet calculation, not a measured outcome. It assumes that "objective data" automatically eliminates 15–25% inefficiency. Unproven.
- Compared to **Coach's Eye ($5/month)** and **Kinovea (free)**, SkateLab must prove its automation delivers measurably better results, not just faster video review. A coach using free tools + 20 min/session may still prefer that to paying 1,500–3,500 ₽/mo for automation if the automation's recommendations are not trusted.
- The **"12-second analysis"** claim is a technical benchmark, not a validated user value. Does a coach trust a 12-second report enough to change their coaching decision? Unknown.
- Omega's existence proves the problem is real at the elite level, but it also sets a high accuracy bar that SkateLab has not yet been benchmarked against.

---

## Overall Verdict: ITERATE

SkateLab should **iterate, not build at scale or pivot.**

The problem is real, the solution is technically feasible, and differentiation is strong on paper. However, **no critical hypothesis has been de-risked with real usage or measured outcomes.** The business knowledge base contains excellent research and a working MVP, but the next layer of validation — "does it work for real users in real rinks, and do they pay?" — is entirely missing.

Investors will ask: "How many coaches have paid?" and "Can you prove the metrics are accurate?" Current answer to both is zero evidence.

---

## Critical Gaps to Close Before Investor Conversations

1. **CustDev sample size (N=3 → N=30+).**
   - Need quantitative survey or structured interviews with 30+ respondents across coaches, athletes, parents, and academies.
   - Current data is qualitative and non-representative (2 coaches, 1 athlete, all Russian-speaking, high-level).

2. **No prototype user testing.**
   - CustDev was conducted **without a prototype.** The "Next Steps" list explicitly flags this as incomplete.
   - Need usability + trust testing: do coaches believe the numbers? Do they act on recommendations?

3. **No accuracy or efficacy validation.**
   - ML pipeline outputs metrics, but there is **no validation study** comparing SkateLab outputs to expert annotations, force plates, or reference systems.
   - Need a benchmark: e.g., measure jump height vs. known reference, or compare SkateLab GOE proxy to actual ISU scores.

4. **Coach WTP is conditional, not committed.**
   - Both coaches said "if proven effective." No coach named a monthly price point.
   - Pricing for Coach tier (1,500–3,500 ₽/mo) and Academy tier (5,000–25,000 ₽/mo) is **purely hypothetical.**

5. **No B2B validation.**
   - Schools/academies are listed as a key segment with the highest revenue tier, but zero interviews have been conducted with them.
   - The B2B sales cycle, decision-maker, and budget owner are unknown.

---

## Recommended Experiments to Validate Weak Areas

| # | Experiment | Hypothesis | Success Metric | Owner |
|---|------------|------------|----------------|-------|
| 1 | **Paid Pilot (10–15 users)** | Coaches/athletes will pay and retain if they see progress | 60%+ monthly retention after 30 days; 3+ referrals | Product |
| 2 | **Accuracy Benchmark** | SkateLab metrics are within 10% of expert/coach consensus on 20 videos | MAE < 10% vs. ground truth for height, airtime, rotation | ML |
| 3 | **Progress A/B Study** | Athletes using SkateLab improve faster than control group | Measurable technique improvement (coach-rated) in 4 weeks vs. control | Product + ML |
| 4 | **Coach Pricing Sensitivity Test** | Coaches have a WTP ceiling between 1,000–3,000 ₽/mo | 30%+ conversion at a specific price point in a Van Westendorp survey | Business |
| 5 | **B2B Discovery Calls (5 academies)** | Academies have budget for team analytics dashboards | 2+ LOIs or pilot commitments from academies | Business |
| 6 | **Trust/UX Prototype Test** | 12-second report format builds trust, not skepticism | 7+ SUS score; 80%+ "would use this in next training session" | UX |

---

## Bottom Line

SkateLab has **strong technical differentiation** and **a plausible value proposition**, but it is currently a **solution in search of validated demand.** The business knowledge base is rich on product and competitive analysis, but thin on market proof. Before raising capital or scaling, the team must close the evidence gap between "MVP works in code" and "users pay because it makes them better skaters."
