/**
 * Utility functions for metrics calculation and formatting
 */

export interface ConfusionMatrix {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
}

/**
 * Calculate metrics from confusion matrix
 */
export function calculateMetrics(matrix: ConfusionMatrix): {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
} {
  const total = matrix.truePositives + matrix.trueNegatives + matrix.falsePositives + matrix.falseNegatives;
  
  if (total === 0) {
    return {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1Score: 0,
      falsePositiveRate: 0,
      falseNegativeRate: 0,
    };
  }

  const accuracy = (matrix.truePositives + matrix.trueNegatives) / total;
  const precision = matrix.truePositives + matrix.falsePositives > 0
    ? matrix.truePositives / (matrix.truePositives + matrix.falsePositives)
    : 0;
  const recall = matrix.truePositives + matrix.falseNegatives > 0
    ? matrix.truePositives / (matrix.truePositives + matrix.falseNegatives)
    : 0;
  const f1Score = precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;
  const falsePositiveRate = matrix.falsePositives + matrix.trueNegatives > 0
    ? matrix.falsePositives / (matrix.falsePositives + matrix.trueNegatives)
    : 0;
  const falseNegativeRate = matrix.falseNegatives + matrix.truePositives > 0
    ? matrix.falseNegatives / (matrix.falseNegatives + matrix.truePositives)
    : 0;

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    falsePositiveRate,
    falseNegativeRate,
  };
}

/**
 * Format metrics for logging
 */
export function formatMetrics(metrics: {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
}): string {
  return `Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%, ` +
         `Precision: ${(metrics.precision * 100).toFixed(2)}%, ` +
         `Recall: ${(metrics.recall * 100).toFixed(2)}%, ` +
         `F1: ${(metrics.f1Score * 100).toFixed(2)}%, ` +
         `FPR: ${(metrics.falsePositiveRate * 100).toFixed(2)}%, ` +
         `FNR: ${(metrics.falseNegativeRate * 100).toFixed(2)}%`;
}
