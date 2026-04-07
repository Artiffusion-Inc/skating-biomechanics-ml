import { Route, Routes } from "react-router-dom"
import AnalyzePage from "./pages/AnalyzePage"
import UploadPage from "./pages/UploadPage"

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-border border-b px-6 py-3">
        <h1 className="font-semibold text-lg">AI Тренер — Фигурное катание</h1>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
