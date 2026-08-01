/**
 * Display metadata for each detector signal name (as they appear in
 * detector_breakdown.signals). Keep in sync with the backend signal names:
 * backend/app/services/fusion_engine.py::FORENSIC_SIGNAL_NAMES /
 * SPORTS_SIGNAL_NAMES, plus "deep_learning".
 */

export type SignalGroup = "ai" | "forensic" | "sports";

interface SignalMeta {
  label: string;
  group: SignalGroup;
}

export const SIGNAL_META: Record<string, SignalMeta> = {
  deep_learning: { label: "AI Deepfake Classifier", group: "ai" },
  frequency_analysis: { label: "Frequency Analysis", group: "forensic" },
  compression_analysis: { label: "Compression Analysis", group: "forensic" },
  lighting_analysis: { label: "Lighting Consistency", group: "forensic" },
  landmark_instability: { label: "Facial Landmark Stability", group: "forensic" },
  optical_flow_analysis: { label: "Motion Consistency", group: "forensic" },
  temporal_consistency: { label: "Temporal Consistency", group: "forensic" },
  jersey_color_consistency: { label: "Jersey Consistency", group: "sports" },
  scene_consistency: { label: "Scene Consistency", group: "sports" },
  broadcast_overlay_analysis: { label: "Broadcast Overlay", group: "sports" },
  crowd_texture_analysis: { label: "Crowd Texture", group: "sports" },
};

export function signalLabel(name: string): string {
  return SIGNAL_META[name]?.label ?? name;
}

export const GROUP_LABELS: Record<SignalGroup, string> = {
  ai: "AI Model",
  forensic: "Forensic Analysis",
  sports: "Sports Intelligence",
};
