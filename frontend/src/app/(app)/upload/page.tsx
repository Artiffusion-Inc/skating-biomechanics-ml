"use client"

import { useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { Upload } from "lucide-react"
import { useCreateSession } from "@/lib/api/sessions"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ChunkedUploader } from "@/components/upload/chunked-uploader"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
  }

  const handleUploaded = async (key: string) => {
    try {
      await createSession.mutateAsync({ element_type: "auto" })
      toast.success("Видео загружено, анализ начат")
      router.push("/feed")
    } catch {
      toast.error("Ошибка создания сессии")
    }
  }

  if (file) {
    return (
      <div className="mx-auto max-w-lg space-y-4">
        <p className="text-sm text-muted-foreground">Загрузка видео...</p>
        <ChunkedUploader file={file} onUploaded={handleUploaded} />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <CameraRecorder onRecorded={(blob) => handleFile(new File([blob], `recording_${Date.now()}.webm`, { type: blob.type }))} />

      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs text-muted-foreground">или</span>
        <div className="h-px flex-1 bg-border" />
      </div>

      <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="mx-auto flex items-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-accent/50"
      >
        <Upload className="h-4 w-4" />
        Загрузить файл
      </button>
    </div>
  )
}
