// algoforge-ai/src/recommender/recommend.ts
import { problemModelLoader } from "./loader";
import type {
  HistoryEntry,
  ModelDifficulty,
  RecommendOptions,
  RecommendResult,
  RecommendationItem,
  RecommendationReason,
} from "./types";

const DIFFICULTY_ORDER: Record<ModelDifficulty, number> = {
  Easy: 0,
  Medium: 1,
  Hard: 2,
};

const STATUS_WEIGHT: Record<string, number> = {
  ACCEPTED: 1.0,
  WRONG_ANSWER: 1.5,
  TIME_LIMIT_EXCEEDED: 1.5,
  MEMORY_LIMIT_EXCEEDED: 1.5,
  RUNTIME_ERROR: 1.2,
  COMPILE_ERROR: 0.5,
};

const STRUGGLE_STATUSES: ReadonlySet<string> = new Set([
  "WRONG_ANSWER",
  "TIME_LIMIT_EXCEEDED",
  "MEMORY_LIMIT_EXCEEDED",
  "RUNTIME_ERROR",
]);

const TAG_BONUS = 0.3;
const DEFAULT_DECAY = 0.85;

const REASON = {
  PRACTICE_WEAK_PATTERN: "PRACTICE_WEAK_PATTERN" as RecommendationReason,
  NEXT_STEP: "NEXT_STEP" as RecommendationReason,
};

export function recommend(
  history: HistoryEntry[],
  options: RecommendOptions,
): RecommendResult {
  const { model, index } = problemModelLoader.load();
  const { slugs, titles, difficulty, tags, neighbors } = model;

  const { n, decay = DEFAULT_DECAY, excludeSlugs, maxDifficulty } = options;

  const scores = new Map<number, number>();
  const bestContributor = new Map<number, { amount: number; struggled: boolean }>();
  const tagWeight = new Map<string, number>();
  let usableEntries = 0;

  history.forEach((entry, rank) => {
    const i = index.get(entry.slug);
    if (i === undefined) return;
    usableEntries++;

    const w = Math.pow(decay, rank) * (STATUS_WEIGHT[entry.status] ?? 1.0);
    const struggled = STRUGGLE_STATUSES.has(entry.status);

    for (const [j, sim] of neighbors[i]) {
      const amount = w * sim;
      scores.set(j, (scores.get(j) ?? 0) + amount);
      const best = bestContributor.get(j);
      if (!best || amount > best.amount) bestContributor.set(j, { amount, struggled });
    }

    for (const tag of tags[i]) {
      tagWeight.set(tag, (tagWeight.get(tag) ?? 0) + w);
    }
  });

  if (usableEntries === 0) {
    return {
      recommendations: [],
      meta: {
        historyCount: history.length,
        usableEntries: 0,
        candidatesConsidered: 0,
        ceiling: 1, // python default when nothing solved
        decay,
      },
    };
  }

  const totalTagWeight = [...tagWeight.values()].reduce((a, b) => a + b, 0) || 1;

  // Ceiling: caller-provided (server's experience-aware) OR fall back to python logic
  let ceiling: number;
  if (typeof maxDifficulty === "number") {
    ceiling = maxDifficulty;
  } else {
    const solvedRanks = history
      .filter((h) => h.status === "ACCEPTED")
      .map((h) => {
        const i = index.get(h.slug);
        return i !== undefined ? DIFFICULTY_ORDER[difficulty[i]] : -1;
      })
      .filter((r) => r >= 0);
    ceiling = solvedRanks.length > 0 ? Math.max(...solvedRanks) + 1 : 1;
  }

  const seen = new Set<string>(history.map((h) => h.slug));
  if (excludeSlugs) for (const s of excludeSlugs) seen.add(s);

  const ranked: Array<{ j: number; score: number; reasonCode: RecommendationReason }> = [];

  for (const [j, baseScore] of scores) {
    const slug = slugs[j];
    if (seen.has(slug)) continue;
    if (DIFFICULTY_ORDER[difficulty[j]] > ceiling) continue;

    const overlap = tags[j].reduce((sum, t) => sum + (tagWeight.get(t) ?? 0), 0);
    const score = baseScore + (TAG_BONUS * overlap) / totalTagWeight;
    const reasonCode = bestContributor.get(j)?.struggled
      ? REASON.PRACTICE_WEAK_PATTERN
      : REASON.NEXT_STEP;

    ranked.push({ j, score, reasonCode });
  }

  ranked.sort((a, b) => b.score - a.score);
  const top = ranked.slice(0, n);

  const recommendations: RecommendationItem[] = top.map((r) => ({
    slug: slugs[r.j],
    title: titles[r.j],
    difficulty: difficulty[r.j],
    tags: tags[r.j],
    score: Math.round(r.score * 1000) / 1000,
    reasonCode: r.reasonCode,
  }));

  return {
    recommendations,
    meta: {
      historyCount: history.length,
      usableEntries,
      candidatesConsidered: scores.size,
      ceiling,
      decay,
    },
  };
}