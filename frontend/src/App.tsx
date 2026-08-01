import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { Dashboard } from "@/pages/Dashboard";
import { Upload } from "@/pages/Upload";
import { History } from "@/pages/History";
import { AnalysisDetail } from "@/pages/AnalysisDetail";

// Data Mode (createBrowserRouter) — the unaffected mode per React Router's
// own advisories for the RSC/Framework-mode-only CSRF issues; we don't use
// server actions or RSC here, this is a client-only SPA talking to a
// separate FastAPI backend.
const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <AppShell>
        <Dashboard />
      </AppShell>
    ),
  },
  {
    path: "/upload",
    element: (
      <AppShell>
        <Upload />
      </AppShell>
    ),
  },
  {
    path: "/history",
    element: (
      <AppShell>
        <History />
      </AppShell>
    ),
  },
  {
    path: "/analysis/:id",
    element: (
      <AppShell>
        <AnalysisDetail />
      </AppShell>
    ),
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
