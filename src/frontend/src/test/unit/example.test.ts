import { describe, expect, it, vi } from "vitest"

// Simple test to verify vitest is working
describe("Vitest setup", () => {
  it("should run tests", () => {
    expect(1 + 1).toBe(2)
  })

  it("should have mocked next/navigation", () => {
    // The mock is in setup.ts - just verify it doesn't crash
    expect(() => {
      require("next/navigation")
    }).not.toThrow()
  })
})
