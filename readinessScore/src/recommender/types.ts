export type SubmissionStatus =
  | "ACCEPTED"
  | "WRONG_ANSWER"
  | "TIME_LIMIT_EXCEEDED"
  | "MEMORY_LIMIT_EXCEEDED"
  | "RUNTIME_ERROR"
  | "COMPILE_ERROR";

export type ModelDifficulty = "Easy" | "Medium" | "Hard";

export type RecommendationReason = "NEXT_STEP" | "PRACTICE_WEAK_PATTERN";

export interface RawProblemModel {
  slugs: string[];
  titles: string[];
  difficulty: ModelDifficulty[];
  tags: string[][];
  neighbors: Array<Array<[number, number]>>;
}

export interface HistoryEntry {
  slug: string;
  status: SubmissionStatus;
}

export interface RecommendOptions {
  n: number;
  decay?: number;
  excludeSlugs?: Set<string>;
  maxDifficulty?: number; // 0=Easy, 1=Medium, 2=Hard. If omitted, computed from history.
}

export interface RecommendationItem {
  slug: string;
  title: string;
  difficulty: ModelDifficulty;
  tags: string[];
  score: number;
  reasonCode: RecommendationReason;
}

export interface RecommendMeta {
  historyCount: number;
  usableEntries: number;
  candidatesConsidered: number;
  ceiling: number;
  decay: number;
}

export interface RecommendResult {
  recommendations: RecommendationItem[];
  meta: RecommendMeta;
}