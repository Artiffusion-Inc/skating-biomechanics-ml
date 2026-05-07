"use client"

import { FileVideo, FileArchive, Activity, ClipboardCheck } from "lucide-react"
import { useTranslations } from "@/i18n"
import type { ZipContents } from "@/lib/zip-parser"

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

export function FilePreview({
  file,
  zipContents,
  previewUrl,
  onRemove,
  onUpload,
}: {
  file: File
  zipContents: ZipContents | null
  previewUrl: string | null
  onRemove: () => void
  onUpload: () => void
}) {
  const t = useTranslations("upload")
  const isZip = zipContents !== null

  return (
    <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
      {/* File header */}
      <div className="flex items-center gap-3">
        {isZip ? (
          <FileArchive className="h-6 w-6 text-primary" />
        ) : (
          <FileVideo className="h-6 w-6 text-primary" />
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{file.name}</p>
          <p className="text-xs text-muted-foreground">{formatSize(file.size)}</p>
        </div>
      </div>

      {/* ZIP contents summary */}
      {isZip && (
        <div className="space-y-2 rounded-xl border border-border p-4">
          {zipContents.video && (
            <div className="flex items-center gap-2 text-sm">
              <FileVideo className="h-4 w-4 text-muted-foreground" />
              <span>{t("videoFound")}</span>
            </div>
          )}
          {(zipContents.imuLeft || zipContents.imuRight) && (
            <div className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span>
                {t("imuInfo", { sensorCount: zipContents.imuLeft && zipContents.imuRight ? 2 : 1 })}
              </span>
            </div>
          )}
          <div className="flex items-center gap-2 text-sm">
            <ClipboardCheck className="h-4 w-4 text-muted-foreground" />
            <span>{zipContents.manifest ? t("manifestOk") : t("manifestMissing")}</span>
          </div>
        </div>
      )}

      {/* Video preview for standalone MP4 */}
      {!isZip && previewUrl && (
        <div
          className="overflow-hidden rounded-2xl"
          style={{ backgroundColor: "oklch(var(--background))" }}
        >
          {/* biome-ignore lint/a11y/useMediaCaption: user upload, no captions */}
          <video
            src={previewUrl}
            controls
            playsInline
            className="aspect-video w-full object-contain"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onRemove}
          className="flex flex-1 items-center justify-center gap-2 rounded-2xl border border-border px-4 py-3 font-medium text-muted-foreground transition-colors hover:bg-accent"
        >
          {t("remove")}
        </button>
        <button
          type="button"
          onClick={onUpload}
          className="flex flex-[2] items-center justify-center gap-2 rounded-2xl bg-primary px-4 py-3 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          {t("startUpload")}
        </button>
      </div>
    </div>
  )
}
