import { unzip, type Unzipped } from "fflate"

export interface ZipContents {
  video: File | null
  imuLeft: Uint8Array | null
  imuRight: Uint8Array | null
  manifest: { [key: string]: unknown } | null
  videoName: string | null
  manifestVersion: string | null
}

function unzipAsync(file: File): Promise<Unzipped> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      unzip(new Uint8Array(reader.result as ArrayBuffer), (err, data) => {
        if (err) reject(err)
        else resolve(data)
      })
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsArrayBuffer(file)
  })
}

function basename(path: string): string {
  return path.split("/").pop() ?? path
}

export async function parseZip(file: File): Promise<ZipContents> {
  const entries = await unzipAsync(file)

  let video: File | null = null
  let videoName: string | null = null
  let imuLeft: Uint8Array | null = null
  let imuRight: Uint8Array | null = null
  let manifest: { [key: string]: unknown } | null = null
  let manifestVersion: string | null = null

  for (const [path, data] of Object.entries(entries)) {
    const name = basename(path)

    // Skip macOS metadata
    if (path.startsWith("__MACOSX") || name.startsWith(".")) continue

    const ext = name.split(".").pop()?.toLowerCase() ?? ""
    if (ext === "mp4" || ext === "mov" || ext === "webm") {
      const blob = new Blob([new Uint8Array(data) as BlobPart], { type: `video/${ext}` })
      video = new File([blob], name, { type: `video/${ext}` })
      videoName = name
    } else if (name.endsWith("_left.pb")) {
      imuLeft = data
    } else if (name.endsWith("_right.pb")) {
      imuRight = data
    } else if (ext === "json") {
      try {
        const parsed = JSON.parse(new TextDecoder().decode(data)) as { [key: string]: unknown }
        manifest = parsed
        manifestVersion = typeof parsed.version === "string" ? parsed.version : null
      } catch {
        // Not a valid JSON manifest, skip
      }
    }
  }

  return { video, imuLeft, imuRight, manifest, videoName, manifestVersion }
}

export function isZipFile(file: File): boolean {
  return file.name.toLowerCase().endsWith(".zip") || file.type === "application/zip"
}

export function isVideoFile(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? ""
  if (["mp4", "mov", "webm", "mkv"].includes(ext)) return true
  if (file.type.startsWith("video/")) return true
  return false
}
