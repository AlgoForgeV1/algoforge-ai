import express from "express";
import { logger } from "./config/logger";
import { env } from "./config/env";
import { recommendRouter } from "./routes/recommend.routes";
import { problemModelLoader } from "./recommender/loader";

const app = express();

app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    success: true,
    service: "AlgoForge AI Recommender",
    status: "UP",
    env: env.NODE_ENV,
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

app.use("/recommend", recommendRouter);

// 404
app.use((_req, res) => {
  res.status(404).json({ success: false, message: "Not found." });
});

// error handler
app.use(
  (err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    if (err instanceof Error && "issues" in err) {
      res.status(400).json({
        success: false,
        message: "Validation failed.",
        errors: (err as { issues: unknown }).issues,
      });
      return;
    }
    logger.error({ err }, "Unhandled request error");
    res.status(500).json({ success: false, message: "Internal server error." });
  },
);

// Fail fast if the model can't be loaded
problemModelLoader.load();

export default app;