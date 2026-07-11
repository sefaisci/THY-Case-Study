import { useQuery } from "@tanstack/react-query";
import { ChevronDown, CircleDollarSign, Cpu, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { getUsage } from "../../api/usage";
import type {
  ChatMessageResponse,
  ChatTurnResponse,
  UsageSummaryResponse,
  UsageTotals,
} from "../../api/contracts";
import { queryKeys } from "../../app/queryKeys";
import { formatInteger, formatKnownCost } from "../../lib/format";

interface UsageDisclosureProps {
  username: string;
  sessionId: string;
  message: ChatMessageResponse;
  turn?: ChatTurnResponse;
}

function UsageScope({ label, usage }: { label: string; usage: UsageTotals | null }) {
  return (
    <div className="usage-scope">
      <span>{label}</span>
      {usage ? (
        <>
          <strong>{formatInteger(usage.total_tokens)} tokens</strong>
          <small>{formatKnownCost(usage.cost_usd)} known cost</small>
        </>
      ) : (
        <small>Not available</small>
      )}
    </div>
  );
}

export function UsageDisclosure({
  username,
  sessionId,
  message,
  turn,
}: UsageDisclosureProps) {
  const [open, setOpen] = useState(false);
  const usageQuery = useQuery({
    queryKey: queryKeys.usage.summary(username, {
      sessionId,
      messageId: message.id,
    }),
    queryFn: ({ signal }) =>
      getUsage(username, { sessionId, messageId: message.id }, signal),
    enabled: open,
    staleTime: 5 * 60_000,
  });

  const immediate: UsageSummaryResponse | undefined = turn
    ? {
        request: turn.request_usage,
        session: turn.session_usage,
        total: turn.total_usage,
        records: [],
      }
    : undefined;
  const usage = usageQuery.data ?? immediate;
  const compact = usage?.request;

  return (
    <section className={`usage-disclosure${open ? " usage-disclosure--open" : ""}`}>
      <button
        type="button"
        className="usage-disclosure__trigger"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <Cpu size={14} aria-hidden="true" />
        <span>{message.model ?? "Provider model"}</span>
        <i aria-hidden="true" />
        <strong>{compact ? `${formatInteger(compact.total_tokens)} tokens` : "Usage details"}</strong>
        {compact ? (
          <>
            <i aria-hidden="true" />
            <span>{formatKnownCost(compact.cost_usd)} known cost</span>
          </>
        ) : null}
        <ChevronDown size={15} aria-hidden="true" />
      </button>

      {open ? (
        <div className="usage-details">
          <div className="usage-details__header">
            <div>
              <span>Model</span>
              <strong>{message.model ?? "Not available"}</strong>
            </div>
            <div>
              <span>Reasoning effort</span>
              <strong>{message.reasoning_effort ?? "Not applicable"}</strong>
            </div>
          </div>

          {usageQuery.isError && !usage ? (
            <p className="inline-error">Usage details could not be loaded.</p>
          ) : (
            <>
              <div className="usage-scope-grid">
                <UsageScope label="Request" usage={usage?.request ?? null} />
                <UsageScope label="Session" usage={usage?.session ?? null} />
                <UsageScope label="Workspace" usage={usage?.total ?? null} />
              </div>
              {usage?.request && usage.request.unpriced_record_count > 0 ? (
                <p className="usage-warning">
                  <TriangleAlert size={14} aria-hidden="true" />
                  Known cost excludes {usage.request.unpriced_record_count} unpriced record(s).
                </p>
              ) : null}
              {usageQuery.isFetching ? <p className="usage-loading">Loading stage details…</p> : null}
              {usageQuery.data && usageQuery.data.records.length > 0 ? (
                <div className="usage-stage-list">
                  <div className="usage-stage-list__title">
                    <CircleDollarSign size={15} aria-hidden="true" />
                    Stage breakdown
                  </div>
                  {usageQuery.data.records.map((record) => (
                    <div key={record.id} className="usage-stage-row">
                      <div>
                        <strong>{record.stage.replaceAll("_", " ")}</strong>
                        <span>{record.model ?? record.provider ?? "No provider usage"}</span>
                      </div>
                      <span>{formatInteger(record.total_tokens)} tokens</span>
                      <span>
                        {record.cost_usd === null ? "Unpriced" : formatKnownCost(record.cost_usd)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}
