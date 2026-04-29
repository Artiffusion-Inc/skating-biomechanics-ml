import { describe, expect, it } from "vitest"
import { cn } from "../utils"

describe("cn", () => {
  it("returns a single class string", () => {
    expect(cn("btn")).toBe("btn")
  })

  it("merges multiple class strings", () => {
    expect(cn("btn", "btn-primary")).toBe("btn btn-primary")
  })

  it("filters out falsy values", () => {
    expect(cn("btn", false && "hidden", null, undefined, "active")).toBe("btn active")
  })

  it("handles conditional classes with objects", () => {
    expect(cn("btn", { "btn-lg": true, "btn-sm": false })).toBe("btn btn-lg")
  })

  it("deduplicates conflicting tailwind classes", () => {
    expect(cn("px-2", "px-4")).toBe("px-4")
  })

  it("returns empty string for no inputs", () => {
    expect(cn()).toBe("")
  })
})
