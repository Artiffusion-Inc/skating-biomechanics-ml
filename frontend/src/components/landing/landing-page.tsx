import { HeroSection } from "./hero-section"
import { FeaturesSection } from "./features-section"
import { MetricsSection } from "./metrics-section"
import { DemoSection } from "./demo-section"
import { CTASection } from "./cta-section"

export function LandingPage() {
  return (
    <div className="landing-page">
      <HeroSection />
      <FeaturesSection />
      <MetricsSection />
      <DemoSection />
      <CTASection />
    </div>
  )
}
