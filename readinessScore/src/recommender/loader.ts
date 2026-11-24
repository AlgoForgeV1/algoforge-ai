import { readFileSync } from "node:fs";
import { env } from "../config/env";
import { logger } from "../config/logger";
import type { RawProblemModel } from "./types";

class ProblemModelLoader {
  private data: RawProblemModel | null = null;
  private index: Map<string, number> | null = null;

  load(): { model: RawProblemModel; index: Map<string, number> } {
    if (this.data && this.index) {
      return { model: this.data, index: this.index };
    }

    const raw = JSON.parse(readFileSync(env.PROBLEM_MODEL_PATH, "utf-8")) as RawProblemModel;
    const n = raw.slugs.length;
    const checks: Array<[string, number]> = [
      ["titles", raw.titles.length],
      ["difficulty", raw.difficulty.length],
      ["tags", raw.tags.length],
      ["neighbors", raw.neighbors.length],
    ];
    for (const [name, len] of checks) {
      if (len !== n) {
        throw new Error(
          `problem_model.json: ${name}.length=${len} does not match slugs.length=${n}`,
        );
      }
    }

    this.data = raw;
    this.index = new Map(raw.slugs.map((s, i) => [s, i]));
    logger.info(
      { problems: n, path: env.PROBLEM_MODEL_PATH },
      "Problem model loaded",
    );
    return { model: this.data, index: this.index };
  }
}

export const problemModelLoader = new ProblemModelLoader();