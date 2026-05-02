---
name: Skating Biomechanics ML
description: AI-powered figure skating coach — video analysis, progress tracking, and coach-student platform
colors:
  ice-deep: "oklch(0.42 0.12 240)"
  ice-glow: "oklch(0.72 0.12 240)"
  ice-surface: "oklch(0.97 0.01 240)"
  ice-dark: "oklch(0.18 0.03 240)"
  background: "oklch(1 0 0)"
  foreground: "oklch(0.145 0 0)"
  card: "oklch(1 0 0)"
  card-foreground: "oklch(0.145 0 0)"
  muted: "oklch(0.967 0 0)"
  muted-foreground: "oklch(0.55 0 0)"
  border: "oklch(0.815 0 0)"
  primary: "oklch(0.145 0 0)"
  primary-foreground: "oklch(1 0 0)"
  accent: "oklch(0.905 0 0)"
  accent-foreground: "oklch(0.145 0 0)"
  destructive: "oklch(0.541 0.22 25)"
  link: "oklch(0.541 0.23 264)"
  score-good: "oklch(0.723 0.219 149)"
  score-mid: "oklch(0.795 0.184 86)"
  score-bad: "oklch(0.577 0.245 27)"
  accent-gold: "oklch(0.795 0.184 86)"
typography:
  display:
    fontFamily: '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
    fontSize: "clamp(3rem, 6vw, 6rem)"
    fontWeight: 500
    lineHeight: 0.9
    letterSpacing: "-0.02em"
  headline:
    fontFamily: '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
    fontSize: "2rem"
    fontWeight: 500
    lineHeight: 1.2
  title:
    fontFamily: '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.75
  label:
    fontFamily: '"Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: "0.05em"
rounded:
  sm: "0.5rem"
  md: "1.25rem"
  lg: "1.875rem"
  xl: "1.875rem"
  2xl: "2.25rem"
spacing:
  1: "0.25rem"
  2: "0.5rem"
  3: "0.75rem"
  4: "1rem"
  5: "1.25rem"
  6: "1.5rem"
  8: "2rem"
  10: "2.5rem"
  12: "3rem"
  16: "4rem"
  20: "5rem"
  24: "6rem"
components:
  button-primary:
    backgroundColor: "{colors.ice-deep}"
    textColor: "{colors.background}"
    rounded: "{rounded.lg}"
    padding: "0.75rem 1.5rem"
    typography: "{typography.label}"
  button-primary-hover:
    backgroundColor: "{colors.ice-glow}"
    textColor: "{colors.foreground}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: "0.75rem 1.5rem"
  card-default:
    backgroundColor: "{colors.card}"
    textColor: "{colors.card-foreground}"
    rounded: "{rounded.md}"
    padding: "1rem"
---

# Design System: Skating Biomechanics ML

## 1. Overview

**Creative North Star: "The Ice Lab"**

A system that treats the interface as an analytical environment: cold, precise, and fast. The aesthetic draws from the physical reality of figure skating (ice, blade, speed) without falling into seasonal clichés. No snowflakes, no winter pastels, no soft romantic gradients. Instead: sharp blues, graphite neutrals, and data-forward layouts that let athletes and coaches see what matters in under two seconds.

The platform (dashboards, analytics, session reviews) dominates the register. The landing page is the laboratory's facade: it may use atmosphere and motion, but the product inside is restrained and functional. Every visual choice traces back to the principle that speed is a feature — both in performance and in cognitive parsing.

**Key Characteristics:**
- Restrained product surfaces with a single cold accent (ice-deep) used sparingly
- Tonal layering for elevation; shadows only on floating overlays
- Pill-shaped interactive elements for touch targets, with sharp state feedback
- Weight-500 typography for all interactive and data text; weight-400 for long prose only
- Metric-forward layouts: scores, charts, and progress indicators are scannable at a glance
- Flat cards with hairline borders; no lift shadows on static containers

## 2. Colors

The palette is built on a graphite-neutral foundation with a single cold blue accent family. The "ice" identity is expressed through hue, not decoration.

### Primary
- **Ice Deep** (oklch(0.42 0.12 240)): Primary actions, active navigation, filled buttons, links on light backgrounds. Used on ≤ 10% of any product screen.
- **Ice Glow** (oklch(0.72 0.12 240)): Hover states, secondary highlights, landing gradients, primary on dark backgrounds.

### Neutral
- **Background** (oklch(1 0 0)): Primary canvas on light mode. Never pure #fff; this is the perceptual white of the system.
- **Foreground** (oklch(0.145 0 0)): Primary text, headings, icons. Near-black with neutral undertone.
- **Card** (oklch(1 0 0)): Surface for grouped content. Matches background; differentiation via border, not color shift.
- **Muted** (oklch(0.967 0 0)): Secondary backgrounds, hover states, skeleton fills.
- **Muted Foreground** (oklch(0.55 0 0)): Descriptive copy, metadata, timestamps, placeholders.
- **Border** (oklch(0.815 0 0)): Hairline dividers, input borders, card outlines.

### Semantic
- **Score Good** (oklch(0.723 0.219 149)): Positive metric indicators. Always paired with a checkmark or label; never color-only.
- **Score Mid** (oklch(0.795 0.184 86)): Neutral / warning metric range.
- **Score Bad** (oklch(0.577 0.245 27)): Critical metric indicators. Always paired with an icon or label.
- **Destructive** (oklch(0.541 0.22 25)): Errors, destructive actions, deletion confirmations.
- **Link** (oklch(0.541 0.23 264)): Inline text links. Distinct from Ice Deep.

### Dark Mode
- **Background** (oklch(0.185 0 0)): Graphite canvas. Not pure black.
- **Foreground** (oklch(0.98 0 0)): Ice-white text.
- **Card** (oklch(0.225 0 0)): Elevated dark surface.
- **Muted** (oklch(0.285 0 0)): Secondary dark backgrounds.
- **Ice Glow** becomes the primary accent on dark; **Ice Deep** becomes secondary.

### Named Rules
**The One Ice Rule.** The ice-blue accent is the only non-neutral hue in the product UI. No secondary brand colors, no category colors, no avatar color cycles. Landing pages may use the extended ice family (surface, glow, deep) more liberally, but the platform stays disciplined.

**The No-Winter-Cliché Rule.** No snowflake icons, no frosted decorative borders, no "frozen glass" effects on product screens. The ice identity is expressed through the cold hue and sharp precision, not literal winter imagery.

## 3. Typography

**Display / Body / Label Font:** Inter Variable (with "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif fallback)

**Character:** A neutral-grotesque with technical precision. Inter's tight metrics and clear distinction between similar glyphs (1/l/I) make it ideal for data-dense athlete dashboards. No emotional serif contrast; the personality comes from weight and spacing, not typeface choice.

### Hierarchy
- **Display** (500, clamp(3rem, 6vw, 6rem), line-height 0.9, letter-spacing -0.02em): Landing hero headlines only. Uppercase optional. Never used in product UI below 24px.
- **Headline** (500, 2rem/32px, line-height 1.2): Page titles, section headers. Max 65–75ch width for multi-line.
- **Title** (600, 1.125rem/18px, line-height 1.3): Card titles, list item headers, metric labels.
- **Body** (400, 1rem/16px, line-height 1.75): Long-form descriptions, explanations. Max 75ch.
- **Body Medium** (500, 1rem/16px, line-height 1.75): Emphasized body, buttons, navigation links, captions. The workhorse weight of the system.
- **Label** (500, 0.875rem/14px, line-height 1.5, letter-spacing 0.05em): Tags, timestamps, small metadata. Uppercase for section labels only.

### Named Rules
**The 500-Dominance Rule.** Weight 500 (Medium) is the default for every interactive element, label, button, and data point. Weight 400 is reserved for multi-sentence prose only. This gives the interface a confident, assertive feel without shouting.

## 4. Elevation

The system uses tonal layering, not shadow-based elevation. Surfaces are flat at rest. Depth is communicated through background color shifts: background → card → popover. 

Shadows appear only as a response to state: dropdown menus, popover panels, and active tab pills receive a subtle ambient shadow to separate them from the page. Static cards, lists, and containers never cast shadows.

### Shadow Vocabulary
- **Ambient Low** (`box-shadow: 0 1px 3px rgba(0,0,0,0.08)`): Active tab pill, selected filter chip. Barely perceptible lift.
- **Ambient Medium** (`box-shadow: 0 4px 12px rgba(0,0,0,0.10)`): Dropdown menus, select popovers, floating panels.
- **Ambient High** (`box-shadow: 0 8px 24px rgba(0,0,0,0.12)`): Modals (rare), camera recorder overlay, choreography floating toolbars.

### Named Rules
**The Flat-By-Default Rule.** Every container is flat at rest. If a card has a shadow, it's wrong. Elevation is earned through interaction or overlay context, not decoration.

## 5. Components

### Buttons
- **Shape:** Pill-shaped (rounded-lg, 1.875rem / 30px). Minimum height 40px (default), 44px (lg), 36px (sm). Touch-target compliant.
- **Primary:** Background Ice Deep (oklch(0.42 0.12 240)), text Background (white), padding 0.75rem 1.5rem. Weight 500.
- **Hover:** Background shifts to Ice Glow (oklch(0.72 0.12 240)), text becomes Foreground (dark). Transition 200ms ease.
- **Active:** Scale 0.98 transform, no shadow. Sharp tactile feedback.
- **Outline:** Transparent background, 1px Border color, text Foreground. Hover: Muted background + Foreground text.
- **Ghost:** Transparent background, no border. Hover: Muted background. For tertiary actions.
- **Destructive:** Background Destructive/10, text Destructive. Hover: Destructive/20.

### Cards / Containers
- **Corner Style:** rounded-md (1.25rem / 20px) for standard cards. rounded-sm (0.5rem / 8px) for inline containers and chips.
- **Background:** Card token (white in light, graphite in dark).
- **Shadow Strategy:** None at rest. Flat-By-Default.
- **Border:** 1px solid Border token. Hairline, not decorative.
- **Internal Padding:** 1rem (16px) standard, 0.75rem (12px) for compact data cards.

### Inputs / Fields
- **Style:** Background Muted, 1px Border, rounded-md (1.25rem) for search; rounded-sm (0.5rem) for form inputs.
- **Focus:** Border shifts to Foreground (dark), 2px ring in Ice Glow. No shadow.
- **Error:** Border Destructive, text Destructive. Ring Destructive/20.
- **Placeholder:** Muted Foreground.

### Chips / Tags
- **Style:** Background Muted, text Muted Foreground, rounded-sm (0.5rem). No border.
- **Selected:** Background Ice Deep, text white. Transition 150ms.
- **Filter variant:** Same as selected but with a dot indicator.

### Navigation
- **App Nav (desktop):** Horizontal tabs, weight 500, 1rem size. Active tab: bottom border 2px Ice Deep. Hover: Muted background pill.
- **Bottom Dock (mobile):** Fixed bottom bar, 48px touch targets, active icon filled in Ice Deep.
- **Landing Nav:** Minimal horizontal, transparent over hero. White text on dark overlay.

### Skeleton / Loading
- **Style:** Background Muted, animate-pulse. No border-radius larger than rounded-md. No skeleton cards with shadows.

### Signature Component: Metric Card
A data-dense card used throughout the platform (session list, progress dashboard, profile).
- **Layout:** Top row = metric label (Label style) + score badge (right-aligned). Bottom = giant metric value (Display scale at 3rem clamp) + sparkline or delta indicator.
- **Background:** Card token, 1px Border.
- **Score badge:** Small pill, background Score-Good/Mid/Bad, white text. Always includes the numeric score, never color-only.

## 6. Do's and Don'ts

### Do:
- **Do** use Ice Deep for ≤ 10% of any product screen surface. Its rarity creates focus.
- **Do** use weight 500 for every button, link, label, and data value. The interface should feel assertive.
- **Do** use hairline borders (1px Border token) to separate cards and sections. No dividers wider than 1px.
- **Do** respect prefers-reduced-motion. Replace animations with instant state changes.
- **Do** pair score colors with icons or numeric labels for colorblind accessibility.
- **Do** use pill-shaped buttons (30px radius) for all primary and secondary actions. Touch targets must be ≥ 44px.
- **Do** keep body text lines under 75ch. Scannability over density.

### Don't:
- **Don't** use shadow on static cards or containers. The Flat-By-Default rule is absolute.
- **Don't** use border-left or border-right greater than 1px as a colored accent stripe on cards, lists, or alerts. Use full borders or background tints instead.
- **Don't** use gradient text (background-clip: text). Use a single solid color. Emphasis via weight or size.
- **Don't** use glassmorphism (backdrop-filter blur) on product screens. Landing hero may use subtle frost; platform never does.
- **Don't** use the hero-metric template (big number + small label + gradient). Metrics belong in context, not as isolated hero elements.
- **Don't** create identical card grids with icon + heading + text repeated endlessly. Vary density and layout rhythm.
- **Don't** use modals as a first solution. Exhaust inline progressive disclosure first.
- **Don't** make the platform feel like a SaaS cream clone (Linear-style warm neutrals, soft shadows, beige accents).
- **Don't** make it feel like a crypto interface (neon accents, dark mode with purple gradients, glass cards).
- **Don't** make it feel like health-tech software (pastel colors, gentle curves, reassuring soft copy).
- **Don't** make it feel like a toy or game (cartoonish icons, badge confetti, gamification with levels and unlocks).
- **Don't** make it feel like a raw engineering dashboard (monospace dumps, terminal aesthetics, unstyled data tables).
- **Don't** use em dashes in copy. Use commas, colons, semicolons, or periods.
