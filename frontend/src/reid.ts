export type ReIDEncoder = { id: string; name: string; family: string; origin: string; available: boolean; fingerprint?: string; runtime?: string; embedding_dimension?: number; supports_training?: boolean; pair_mode?: "vlm_fusion" | "visual_cosine"; availability_reason?: string | null; adaptation?: string };
export type MatchingSettings = { mode: "automatic" | "review"; threshold: number; margin: number; new_threshold: number; color_weight: number };
export type ReIDSite = { id: string; name: string; active_encoder: string; matching?: MatchingSettings | null; cameras: { id: string; name: string }[]; transitions: { source: string; destination: string; min_seconds: number; max_seconds: number }[] };
export type ReIDJob = { id: string; name: string; kind: "experiment" | "training" | "evaluation" | "promotion" | "benchmark" | "benchmark_export" | "comparison"; site_id: string; state: string; finished_at?: string; stage: string; progress: number; error?: string; refresh_required?: boolean; deletion_pending?: boolean; deletion_error?: string; artifacts: Record<string, string>; config: { encoders?: string[]; encoder_id?: string; benchmark_id?: string }; metrics: Record<string, unknown> };
export type DeletionPreview = { experiment_id: string; name: string; observations: number; files: number; identities_removed: string[]; identities_preserved: string[]; experiments_to_refresh: string[]; blockers: { kind: string; id: string; name: string }[] };
export type ReIDScore = { appearance_similarity?: number; color_similarity?: number | null; combined_score?: number; color_weight?: number; effective_color_weight?: number; color_methods?: { query: string; reference: string }; scoring_version?: string; reference_crop_index?: number };
export type ReIDTrack = { id: string; experiment_id: string; camera_id: string; media_type: "image"; class_name: "car" | "truck"; local_id: number; start: number; end: number; start_frame: number; end_frame: number; identity: string | null; excluded: boolean; global_id: string | null; reviewed: boolean; crops: { frame: number; timestamp: number }[]; assignments: Record<string, ReIDScore & { global_id: string | null; status: string; similarity: number | null; provenance?: string; reason?: string; reference_track_id?: string | null; candidates: (ReIDScore & { global_id: string; similarity: number; reference_track_id?: string })[]; neighbors?: { track_id: string; similarity: number }[] }> };
export type ReIDDataset = { id: string; site_id: string; fingerprint: string; splits: Record<string, string[]> };
export type RetrievalMetrics = { rank1: number | null; rank5: number | null; mAP: number | null; eligible_queries: number; excluded_queries: number };
export type ReIDEvaluation = { id: string; site_id: string; dataset_id: string; encoder_id: string; split: string; metrics: { cross_camera: RetrievalMetrics; same_camera: RetrievalMetrics; combined_cross_camera?: RetrievalMetrics; combined_same_camera?: RetrievalMetrics; scoring?: { color_weight: number; fingerprint: string }; false_merges: number; missed_matches: number; calibration: { precision: number } | null } };
export type BinaryBenchmarkMetrics = { threshold: number; pairs: number; tp: number; fp: number; tn: number; fn: number; precision: number; recall: number; f1: number; accuracy: number };
export type IdentificationMetrics = { eligible_queries: number; excluded_queries: number; accuracy: number; rank1: number; mAP: number; macro_precision: number; macro_recall: number; macro_f1: number };
export type BenchmarkIdentityMetrics = { identity: string; support: number; included_in_macro: boolean; tp: number; fp: number; tn: number; fn: number; precision: number; recall: number; f1: number };
export type BenchmarkEncoderResult = { encoder_id: string; encoder_name: string; encoder_fingerprint: string; embedding_dimension: number; configured_threshold: number; pairwise_strict: BinaryBenchmarkMetrics; pairwise_all_images: BinaryBenchmarkMetrics; best_f1_exploratory: BinaryBenchmarkMetrics; identification: IdentificationMetrics; per_identity: BenchmarkIdentityMetrics[]; confusion: { actual_identity: string; predicted_identity: string; count: number }[] };
export type BenchmarkComparisonExport = { job_id: string | null; state: "not_prepared" | "queued" | "running" | "completed" | "failed" | "cancelled" | "stale"; progress: number; error: string | null; comparison_count: number; limit: number; ready: boolean; source_result_fingerprint: string };
export type ReIDBenchmark = { id: string; job_id: string; name: string; site_id: string; threshold: number; state: string; counts: { identities: number; images: number; source_groups: number; all_pairs_per_encoder?: number; strict_pairs_per_encoder?: number; eligible_identification_queries?: number }; encoders?: BenchmarkEncoderResult[]; comparison_export?: BenchmarkComparisonExport; created_at: string; updated_at: string };
export type BenchmarkComparison = { left_id: string; left_path: string; left_filename: string; left_image_url: string; left_identity: string; left_source_group: string; right_id: string; right_path: string; right_filename: string; right_image_url: string; right_identity: string; right_source_group: string; encoder_id: string; encoder_name: string; encoder_fingerprint: string; cosine_similarity: number; actual_same_identity: boolean; strict_eligible: boolean; exclusion_reason: string | null; predicted_same_configured: boolean; predicted_same_best_f1: boolean; best_f1_threshold: number };
export type BenchmarkComparisonPage = { benchmark_id: string; items: BenchmarkComparison[]; total: number; offset: number; limit: number; available_encoders: { id: string; name: string; fingerprint: string | null }[]; available_identities: string[] };
export type BenchmarkComparisonFilters = { encoder_id?: string; identity?: string; pair_type?: "all" | "same" | "different"; eligibility?: "all" | "strict" | "excluded"; offset?: number; limit?: number };
export type ReIDComparison = { experiment_id: string; query_observation_id: string; query_crop_index: number; query_filename: string; query_image_url: string; query_camera_id: string; query_capture_time: number | string | null; query_class_name: string; reference_observation_id: string; reference_crop_index: number; reference_filename: string; reference_image_url: string; reference_camera_id: string; reference_capture_time: number | string | null; reference_class_name: string; candidate_vehicle_id: string; encoder_id: string; encoder_name: string; encoder_fingerprint: string; cosine_similarity: number; cosine_rank: number; color_similarity: number | null; combined_score: number; effective_color_weight: number; context_score: number; detected_vehicle_id: string | null; assigned_vehicle_id: string | null; decision_status: string; is_winning_identity: boolean; is_assigned_identity: boolean; is_decision_reference: boolean };
export type ReIDComparisonPage = { experiment_id: string; items: ReIDComparison[]; total: number; offset: number; limit: number; available_encoders: { id: string; name: string; fingerprint: string | null }[]; available_vehicle_ids: string[]; available_observations: { id: string; filename: string }[] };
export type ReIDComparisonFilters = { observation_id?: string; encoder_id?: string; candidate_vehicle_id?: string; decision_flag?: "all" | "winning" | "assigned" | "decision-reference"; offset?: number; limit?: number };
export type PairWeights = { appearance: number; color: number; shape: number; semantic: number };
export type PairSimilarity = { appearance_cosine_raw: number; appearance: number; color: number | null; shape: number | null; semantic: number; combined: number };
export type PairSemanticAttribute = { label: string; confidence: number; distribution: Record<string, number> };
export type PairCriteriaAttribute = { image_1: PairSemanticAttribute; image_2: PairSemanticAttribute; consistency_score: number };
export type PairCriteriaReport = { version: string; fingerprint: string; criteria_similarity_score: number; qualitative_band: "high" | "moderate" | "low"; status: string; criteria_components: Record<string, number | null>; effective_weights: Record<string, number>; appearance_embedding: Record<string, number | PairCriteriaAttribute | null>; color_extractor: { dominant_colors: { image_1: string[]; image_2: string[]; shared: string[] }; color_distribution: Record<string, PairCriteriaAttribute>; color_variance: number | null; hue_shift: number | null }; shape_extractor: Record<string, number | PairCriteriaAttribute | null>; plate_logo_type_viewpoint: Record<string, PairCriteriaAttribute | unknown> };
export type PairResultSummary = { encoder: { id: string; name: string; family: string; fingerprint: string; embedding_dimension: number }; encoder_only: boolean; similarities: PairSimilarity; availability: Record<keyof PairWeights, boolean>; scoring: { weights: PairWeights; effective_weights: PairWeights; threshold: number; fingerprint: string }; decision: { same_vehicle: boolean; threshold: number; score_source: string; status: string }; criteria_report?: PairCriteriaReport; criteria_similarity_score?: number; images: { side: "first" | "second"; filename: string; norms: Record<keyof PairWeights, number | null>; color_method: string; semantic_attributes: Record<string, PairSemanticAttribute> }[] };
export type PairEncoderId = "coca" | "coca_visual" | "coca_l14" | "coca_l14_visual" | "siglip2" | "siglip2_visual";
export type PairComparison = { id: string; job_id: string; name: string; site_id: string; encoder_id: PairEncoderId; threshold: number; weights: PairWeights; images: { side: "first" | "second"; filename: string; image_url: string }[]; result?: PairResultSummary; state: string; stage: string; progress: number; error?: string | null; created_at: string; updated_at: string };

async function request<T>(path: string, body?: unknown, method = "GET"): Promise<T> {
  const response = await fetch("/api/reid" + path, { method, ...(body === undefined ? {} : body instanceof FormData ? { body } : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail));
  }
  return response.json() as Promise<T>;
}

export const reid = {
  deletionPreview: (id: string) => request<DeletionPreview>("/experiments/" + id + "/deletion-preview"),
  deleteExperiment: (id: string) => request<{ deleted: boolean }>("/experiments/" + id, undefined, "DELETE"),
  comparisons: (id: string, filters: ReIDComparisonFilters = {}) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => { if (value !== undefined && value !== "") query.set(key, String(value)); });
    return request<ReIDComparisonPage>("/experiments/" + id + "/comparisons" + (query.size ? "?" + query : ""));
  },
  status: () => request<{ environment_ready: boolean; encoders: ReIDEncoder[] }>("/status"),
  sites: () => request<ReIDSite[]>("/sites"),
  createSite: (body: unknown) => request<ReIDSite>("/sites", body, "POST"),
  updateSite: (id: string, body: unknown) => request<ReIDSite>("/sites/" + id, body, "PUT"),
  identities: (id: string) => request<{ global_id: string }[]>("/sites/" + id + "/identities"),
  jobs: () => request<ReIDJob[]>("/jobs"),
  pairComparisons: () => request<PairComparison[]>("/pair-comparisons"),
  pairComparison: (id: string) => request<PairComparison>("/pair-comparisons/" + id),
  createPairComparison: (files: File[], config: unknown) => { const body = new FormData(); files.forEach(file => body.append("files", file)); body.append("config", JSON.stringify(config)); return request<ReIDJob>("/pair-comparisons", body, "POST"); },
  updatePairScoring: (id: string, threshold: number, weights: PairWeights) => request<PairComparison>("/pair-comparisons/" + id + "/scoring", { threshold, weights }, "PUT"),
  deletePairComparison: (id: string) => request<{ deleted: boolean }>("/pair-comparisons/" + id, undefined, "DELETE"),
  upload: (files: File[], config: unknown) => { const body = new FormData(); files.forEach(file => body.append("files", file)); body.append("config", JSON.stringify(config)); return request<ReIDJob>("/experiments", body, "POST"); },
  cancel: (id: string) => request<ReIDJob>("/jobs/" + id + "/cancel", undefined, "POST"),
  retry: (id: string, encoders?: string[]) => request<ReIDJob>("/jobs/" + id + "/retry", encoders ? { encoders } : undefined, "POST"),
  tracks: (id: string) => request<ReIDTrack[]>("/experiments/" + id + "/tracks"),
  annotate: (id: string, identity: string | null, excluded: boolean) => request<ReIDTrack>("/tracks/" + id + "/annotation", { identity, excluded }, "PUT"),
  review: (id: string, globalId?: string) => request<ReIDTrack>("/tracks/" + id + "/review", { global_id: globalId || null, new_identity: !globalId }, "PUT"),
  datasets: () => request<ReIDDataset[]>("/datasets"),
  snapshot: (siteId: string, experimentIds: string[]) => request<ReIDDataset>("/datasets", { site_id: siteId, experiment_ids: experimentIds }, "POST"),
  evaluations: () => request<ReIDEvaluation[]>("/evaluations"),
  benchmarks: () => request<ReIDBenchmark[]>("/benchmarks"),
  benchmark: (id: string) => request<ReIDBenchmark>("/benchmarks/" + id),
  benchmarkComparisons: (id: string, filters: BenchmarkComparisonFilters = {}) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => { if (value !== undefined && value !== "") query.set(key, String(value)); });
    return request<BenchmarkComparisonPage>("/benchmarks/" + id + "/comparisons" + (query.size ? "?" + query : ""));
  },
  createBenchmark: (files: File[], config: unknown) => { const body = new FormData(); files.forEach(file => body.append("files", file)); body.append("config", JSON.stringify(config)); return request<ReIDJob>("/benchmarks", body, "POST"); },
  updateBenchmarkThreshold: (id: string, threshold: number) => request<ReIDBenchmark>("/benchmarks/" + id + "/threshold", { threshold }, "PUT"),
  prepareBenchmarkComparisonImages: (id: string) => request<ReIDJob>("/benchmarks/" + id + "/comparison-images", undefined, "POST"),
  deleteBenchmark: (id: string) => request<{ deleted: boolean }>("/benchmarks/" + id, undefined, "DELETE"),
  evaluate: (datasetId: string, encoderId: string, split: string) => request<ReIDJob>("/evaluations", { dataset_id: datasetId, encoder_id: encoderId, split }, "POST"),
  train: (datasetId: string, encoderId: string, epochs: number) => request<ReIDJob>("/training", { dataset_id: datasetId, encoder_id: encoderId, epochs }, "POST"),
  promote: (siteId: string, encoderId: string) => request<ReIDJob>("/sites/" + siteId + "/encoder", { encoder_id: encoderId }, "POST"),
  importEncoder: (file: File) => { const body = new FormData(); body.append("file", file); return request<ReIDEncoder>("/encoders/import", body, "POST"); },
  exportTraining: async (datasetId: string, encoderId: string, epochs: number) => {
    const response = await fetch("/api/reid/training/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id: datasetId, encoder_id: encoderId, epochs }) });
    if (!response.ok) { const error = await response.json(); throw new Error(error.detail); }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href = url; link.download = "iris-reid-training.zip"; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
};
