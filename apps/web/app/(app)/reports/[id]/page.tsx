import type { Report } from "@/components/report/ReportView";
import { serverApi } from "@/lib/server-api";
import { ReportPageClient } from "./ReportPageClient";

// Server-rendered when the report exists (fast first paint); otherwise the client finishes the
// session and loads it.
export default async function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const initial = await serverApi<Report>(`/reports/${encodeURIComponent(id)}`);
  return <ReportPageClient id={id} initial={initial} />;
}
