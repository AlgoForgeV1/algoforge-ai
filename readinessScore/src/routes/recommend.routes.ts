import { Router } from "express";
import { z } from "zod";
import { recommend } from "../recommender/recommend";
import type { SubmissionStatus } from "../recommender/types";

const VALID_STATUSES: SubmissionStatus[] = [
  "ACCEPTED",
  "WRONG_ANSWER",
  "TIME_LIMIT_EXCEEDED",
  "MEMORY_LIMIT_EXCEEDED",
  "RUNTIME_ERROR",
  "COMPILE_ERROR",
];

/** "Wrong Answer" → "WRONG_ANSWER", "Accepted" → "ACCEPTED". */
const statusSchema = z
  .string()
  .trim()
  .transform((raw) => raw.toUpperCase().replace(/\s+/g, "_"))
  .refine(
    (s): s is SubmissionStatus => (VALID_STATUSES as string[]).includes(s),
    {
      message: `status must be one of: ${VALID_STATUSES.join(", ")}`,
    },
  );

const historyEntrySchema = z.object({
  slug: z
    .string()
    .trim()
    .min(1)
    .max(200)
    .regex(/^[a-z0-9-]+$/, "slug must be lowercase-hyphenated"),
  status: statusSchema,
});

const recommendRequestSchema = z.object({
  history: z.array(historyEntrySchema).max(200).default([]),
  n: z.number().int().min(1).max(50).default(8),
  decay: z.number().gt(0).lt(1).default(0.85),
  excludeSlugs: z.array(z.string()).optional(),
  maxDifficulty: z.number().int().min(0).max(2).optional(),
});

export const recommendRouter = Router();

recommendRouter.post("/", (req, res, next) => {
  try {
    const body = recommendRequestSchema.parse(req.body);
    const result = recommend(body.history, {
      n: body.n,
      decay: body.decay,
      excludeSlugs: body.excludeSlugs ? new Set(body.excludeSlugs) : undefined,
      maxDifficulty: body.maxDifficulty,
    });
    res.status(200).json({ success: true, data: result });
  } catch (err) {
    next(err);
  }
});
