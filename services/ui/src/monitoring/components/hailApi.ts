import { hailApiFetch } from '../../shared/hailApiFetch';

export interface CloudBillingResponse {
  compute_cost_breakdown?: { source: string; cost: string }[];
  cost_by_service?: { service: string; cost: string }[];
  cost_by_sku_label?: { source: string | null; sku_description: string; service_description: string; cost: string }[];
}

export interface BillingRow {
  billing_project: string;
  resource: string;
  cost: number;
}

export interface HailApiBaseUrls {
  monitoring: string;
  batch: string;
}

// A small, hand-written mirror of just the endpoints this dashboard calls — not the services'
// full API surface. A future PR could generate the full thing (for every service) from the
// OpenAPI specs each service already publishes at /openapi.yaml; this is deliberately scoped
// to current usage rather than speculatively covering endpoints nothing calls yet.
export function createHailApi({ monitoring, batch }: HailApiBaseUrls) {
  return {
    monitoring: {
      billing: {
        get: (timePeriod: string) =>
          hailApiFetch<CloudBillingResponse>(`${monitoring}/api/v1alpha/billing?time_period=${encodeURIComponent(timePeriod)}`),
      },
    },
    batch: {
      billingBreakdown: {
        get: (start: string, end: string) =>
          hailApiFetch<BillingRow[]>(`${batch}/api/v1alpha/billing_breakdown?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`),
      },
    },
  };
}

export type HailApi = ReturnType<typeof createHailApi>;
