import { describe, it, expect } from "vitest"
import { zipSync, strToU8 } from "fflate"
import { parseZip, isZipFile, isVideoFile } from "../zip-parser"

function createTestZip(files: { name: string; content: string | Uint8Array }[]): File {
  const entries: Record<string, Uint8Array> = {}
  for (const f of files) {
    entries[f.name] = typeof f.content === "string" ? strToU8(f.content) : f.content
  }
  const zipped = zipSync(entries)
  return new File([zipped as BlobPart], "test.zip", { type: "application/zip" })
}

describe("zip-parser", () => {
  it("detects ZIP file by extension", () => {
    expect(isZipFile(new File([], "test.zip", { type: "application/zip" }))).toBe(true)
    expect(isZipFile(new File([], "test.mp4"))).toBe(false)
  })

  it("detects video file by extension and MIME", () => {
    expect(isVideoFile(new File([], "test.mp4"))).toBe(true)
    expect(isVideoFile(new File([], "test.MOV"))).toBe(true)
    expect(isVideoFile(new File([], "test.zip"))).toBe(false)
    expect(isVideoFile(new File([], "test", { type: "video/mp4" }))).toBe(true)
  })

  it("extracts video, IMU, and manifest from ZIP", async () => {
    const zip = createTestZip([
      { name: "capture_20260507.mp4", content: "fake-video" },
      { name: "capture_20260507_left.pb", content: new Uint8Array([1, 2, 3]) },
      { name: "capture_20260507_right.pb", content: new Uint8Array([4, 5, 6]) },
      { name: "capture_20260507.json", content: JSON.stringify({ version: "1.0", videoFps: 60 }) },
    ])
    const result = await parseZip(zip)

    expect(result.video).not.toBeNull()
    expect(result.videoName).toBe("capture_20260507.mp4")
    expect(result.imuLeft).not.toBeNull()
    expect(result.imuRight).not.toBeNull()
    expect(result.manifest).toEqual({ version: "1.0", videoFps: 60 })
    expect(result.manifestVersion).toBe("1.0")
  })

  it("returns nulls for missing components", async () => {
    const zip = createTestZip([{ name: "video.mp4", content: "only-video" }])
    const result = await parseZip(zip)

    expect(result.video).not.toBeNull()
    expect(result.imuLeft).toBeNull()
    expect(result.imuRight).toBeNull()
    expect(result.manifest).toBeNull()
  })

  it("returns null video when ZIP has no video", async () => {
    const zip = createTestZip([{ name: "data.json", content: "{}" }])
    const result = await parseZip(zip)

    expect(result.video).toBeNull()
  })

  it("skips __MACOSX and dotfiles", async () => {
    const zip = createTestZip([
      { name: "__MACOSX/._capture.mp4", content: "junk" },
      { name: ".DS_Store", content: "junk" },
      { name: "capture.mp4", content: "real-video" },
    ])
    const result = await parseZip(zip)

    expect(result.videoName).toBe("capture.mp4")
  })
})
