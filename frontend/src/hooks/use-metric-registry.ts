// src/hooks/use-metric-registry.ts
import { useQuery } from "@tanstack/react-query"
import { apiFetch } from "@/lib/api-client"
import { z } from "zod"

const RegistrySchema = z.record(
  z.string(),
  z.object({
    name: z.string(),
    label_ru: z.string(),
    unit: z.string(),
    format: z.string(),
    direction: z.enum(["higher", "lower"]),
    element_types: z.array(z.string()),
    ideal_range: z.tuple([z.number(), z.number()]),
  }),
)

export type MetricRegistry = z.infer<typeof RegistrySchema>

export function useMetricRegistry() {
  return useQuery({
    queryKey: ["metric-registry"],
    queryFn: () => apiFetch("/metrics/registry", RegistrySchema),
    staleTime: Infinity,
  })
}
