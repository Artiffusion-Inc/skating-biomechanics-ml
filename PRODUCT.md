# Product

## Register

product

## Users

**Primary:** Figure skaters at all levels (beginner to competitive) who train independently or with a coach.
**Secondary:** Coaches who manage multiple students, track their progress, and provide remote feedback.

**Context:** Users record training videos on ice, upload them, and receive biomechanical analysis. Coaches review student analytics even when physically distant. Both roles are interchangeable: a coach can also be a student under another coach.

**Job to be done:**
- For skaters: get fast, accurate feedback on technique (shoulder alignment, body posture, jump metrics) without waiting for a coach.
- For coaches: monitor student progress remotely, track jump height, rotation, and landing quality over time.

## Product Purpose

AI-powered figure skating coach that analyzes video in near real-time, tracks progress, and connects skaters with coaches. Like Strava for figure skating: lower the entry barrier to the sport and accelerate progression for both athletes and coaches.

**Key outcomes:**
- Video upload → biomechanical analysis (pose, metrics, recommendations) delivered fast.
- Persistent progress tracking with dashboards and PRs.
- Coach-student relationship platform with shared session visibility.
- Future: choreography generation, H-Sense blade-angle tracking (accelerometer add-on).

## Brand Personality

**Three words:** ледяной, спортивный, технологичный

**Voice:** Direct, confident, no fluff. Athletic precision meets modern tech. Not clinical — energetic enough to motivate training, restrained enough to feel professional.

**Emotional goals:**
- For skaters: empowerment — "I can see exactly what to fix."
- For coaches: control — "I know my students' progress without being there."

**Reference:** Strava for UX organization (activity-centric, clear hierarchy), but with a colder, sharper aesthetic. Nike was an initial placeholder — discard the direct Nike visual mimicry, keep the athletic discipline.

## Anti-references

- **Slow / laggy interfaces** — speed is a core value proposition. Any perceived sluggishness kills trust.
- **Generic SaaS cream** — Linear clones, soft rounded everything, warm beige palettes.
- **Crypto neon** — aggressive gradients, glowing accents, dark-mode-by-default edginess.
- **Health-tech softness** — pastel colors, gentle curves, overly reassuring copy.
- **Toy-like / game UI** — playful illustrations, cartoonish icons, gamification with badges and confetti.
- **Robotized / hyper-technical** — raw data dumps, terminal aesthetics, engineering-first dashboards.
- **Direct Strava clone** — take UX structure inspiration, never the orange heatmap, never the social feed layout.

## Design Principles

1. **Speed is a feature** — every interaction must feel snappy. Load states are informative, never blocking. Perceived performance matters as much as actual.
2. **Precision over decoration** — clean lines, disciplined spacing, no ornamental flourishes. The tool serves the athlete, not the other way around.
3. **Data at a glance** — metrics and progress should be scannable in < 2 seconds. No buried insights.
4. **Ice as identity, not cliché** — the cold aesthetic (sharp blues, white, graphite) reflects the sport without falling into generic "winter" tropes.
5. **Unified roles, clear context** — coach and student views share the same visual language but signal context shifts clearly (via subtle surface changes, not jarring theme switches).

## Accessibility & Inclusion

- **WCAG 2.1 AA** as baseline.
- **Reduced motion:** Respect `prefers-reduced-motion`. No bouncing, no elastic transitions.
- **Colorblind-safe metrics:** Score indicators (good/mid/bad) must not rely on color alone — use weight, iconography, or labels alongside OKLCH hues.
- **Touch targets:** Minimum 44×44px on mobile (WCAG AAA), 48×48px on bottom dock navigation.
