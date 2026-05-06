import { describe, expect, it } from "vitest"
import { render, screen, fireEvent } from "./test-utils"
import { ErrorBoundary } from "@/components/error-boundary"
import { useState } from "react"

function ThrowOnce({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error("Test error")
  }
  return <div data-testid="safe">Safe content</div>
}

function Wrapper() {
  const [shouldThrow, setShouldThrow] = useState(true)
  return (
    <div>
      <button type="button" data-testid="fix" onClick={() => setShouldThrow(false)}>
        Fix
      </button>
      <ErrorBoundary>
        <ThrowOnce shouldThrow={shouldThrow} />
      </ErrorBoundary>
    </div>
  )
}

describe("ErrorBoundary", () => {
  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <div data-testid="child">Hello</div>
      </ErrorBoundary>,
    )
    expect(screen.getByTestId("child")).toHaveTextContent("Hello")
  })

  it("renders fallback UI when child throws", () => {
    const originalError = console.error
    try {
      console.error = () => {}
      render(
        <ErrorBoundary>
          <ThrowOnce shouldThrow={true} />
        </ErrorBoundary>,
      )
      expect(screen.getByText("Something went wrong")).toBeInTheDocument()
      expect(screen.getByText("Test error")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument()
    } finally {
      console.error = originalError
    }
  })

  it("renders custom fallback when child throws and fallback is provided", () => {
    const originalError = console.error
    try {
      console.error = () => {}
      render(
        <ErrorBoundary fallback={<div data-testid="custom-fallback">Custom error</div>}>
          <ThrowOnce shouldThrow={true} />
        </ErrorBoundary>,
      )
      expect(screen.getByTestId("custom-fallback")).toHaveTextContent("Custom error")
      expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument()
    } finally {
      console.error = originalError
    }
  })

  it("resets and shows children after clicking try again", () => {
    const originalError = console.error
    try {
      console.error = () => {}
      render(<Wrapper />)
      expect(screen.getByText("Something went wrong")).toBeInTheDocument()

      fireEvent.click(screen.getByTestId("fix"))
      fireEvent.click(screen.getByRole("button", { name: /try again/i }))
      expect(screen.getByTestId("safe")).toHaveTextContent("Safe content")
    } finally {
      console.error = originalError
    }
  })
})
