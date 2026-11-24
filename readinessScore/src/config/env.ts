import dotenv from "dotenv";
import path from "node:path";
import { z } from "zod";

dotenv.config();

const schema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.coerce.number().default(5001),
  PROBLEM_MODEL_PATH: z.string().default("./data/problem_model.json"),
  LOG_LEVEL: z.string().default("info"),
});

const parsed = schema.parse(process.env);

export const env = {
  ...parsed,
  PROBLEM_MODEL_PATH: path.resolve(process.cwd(), parsed.PROBLEM_MODEL_PATH),
};