import { MLServiceResponse } from '../types';
import { Threat, ThreatSeverity, ThreatType } from '../models/detection.model';

export class DecisionEngineService {
  private readonly PHISHING_THRESHOLD = 0.7;
  private readonly HIGH_CONFIDENCE_THRESHOLD = 0.85;
  
  makeDecision(mlResponse: MLServiceResponse, input: any): Threat {
    const scores = this.calculateScores(mlResponse);
    const ensembleScore = this.calculateEnsembleScore(scores);
    const severity = this.determineSeverity(ensembleScore, scores);
    const threatType = this.determineThreatType(mlResponse, scores);
    
    const isThreat = ensembleScore >= this.PHISHING_THRESHOLD;
    const confidence = this.calculateConfidence(scores);
    
    return {
      isThreat,
      confidence,
      severity,
      threatType,
      scores: {
        ensemble: ensembleScore,
        nlp: scores.nlp,
        url: scores.url,
        visual: scores.visual
      },
      indicators: this.extractIndicators(mlResponse),
      metadata: {
        processingTimeMs: mlResponse.processingTimeMs,
        timestamp: new Date().toISOString()
      }
    };
  }
  
  private calculateScores(mlResponse: MLServiceResponse): {
    nlp: number;
    url: number;
    visual: number;
  } {
    const scores = {
      nlp: 0,
      url: 0,
      visual: 0
    };
    
    // NLP score
    if (mlResponse.nlp) {
      scores.nlp = mlResponse.nlp.phishing_probability || 0;
      
      // Boost score based on urgency and AI-generated content
      if (mlResponse.nlp.urgency_score && mlResponse.nlp.urgency_score > 70) {
        scores.nlp += 0.1;
      }
      if (mlResponse.nlp.ai_generated_probability && mlResponse.nlp.ai_generated_probability > 0.7) {
        scores.nlp += 0.15;
      }
      scores.nlp = Math.min(1.0, scores.nlp);
    }
    
    // URL score
    if (mlResponse.url) {
      const urlData = Array.isArray(mlResponse.url) ? mlResponse.url[0] : mlResponse.url;

      // Use direct phishing_probability if available (from heuristic ensemble)
      if (urlData.phishing_probability !== undefined) {
        scores.url = urlData.phishing_probability;
      } else {
        // Combine multiple URL analysis results
        if (urlData.is_suspicious) {
          scores.url += 0.4;
        }
        if (urlData.domain_analysis?.is_suspicious) {
          scores.url += 0.3;
        }
        if (urlData.whois_analysis?.is_suspicious) {
          scores.url += 0.2;
        }
        if (urlData.redirect_analysis?.is_suspicious) {
          scores.url += 0.25;
        }
        if (urlData.homoglyph_analysis?.is_suspicious) {
          scores.url += 0.25;
        }
        scores.url = Math.min(1.0, scores.url);
      }

      // Boost URL score with obfuscation signals
      if (urlData.obfuscation_analysis?.is_obfuscated) {
        const obfScore = urlData.obfuscation_analysis.obfuscation_score || 0;
        scores.url = Math.min(1.0, scores.url + obfScore * 0.3);
      }
    }
    
    // Visual score
    if (mlResponse.visual) {
      const visualData = mlResponse.visual;
      
      // Use direct phishing_probability if available
      if (visualData.phishing_probability !== undefined) {
        scores.visual = visualData.phishing_probability;
      } else {
        if (visualData.form_analysis?.is_suspicious) {
          scores.visual += 0.4;
        }
        if (visualData.brand_prediction?.is_brand_impersonation) {
          scores.visual += 0.3;
        }
        if (visualData.similarity_analysis?.is_similar && 
            visualData.similarity_analysis?.similarity_score < 0.9) {
          scores.visual += 0.2; // Similar but not identical = suspicious
        }
        scores.visual = Math.min(1.0, scores.visual);
      }
    }
    
    return scores;
  }
  
  private calculateEnsembleScore(scores: { nlp: number; url: number; visual: number }): number {
    // Weighted ensemble
    const weights = {
      nlp: 0.4,
      url: 0.4,
      visual: 0.2
    };
    
    // Only include scores that are available
    let totalWeight = 0;
    let weightedSum = 0;
    
    if (scores.nlp > 0) {
      weightedSum += scores.nlp * weights.nlp;
      totalWeight += weights.nlp;
    }
    
    if (scores.url > 0) {
      weightedSum += scores.url * weights.url;
      totalWeight += weights.url;
    }
    
    if (scores.visual > 0) {
      weightedSum += scores.visual * weights.visual;
      totalWeight += weights.visual;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }
  
  private determineSeverity(
    ensembleScore: number,
    scores: { nlp: number; url: number; visual: number }
  ): ThreatSeverity {
    if (ensembleScore >= 0.9) {
      return 'critical';
    } else if (ensembleScore >= 0.75) {
      return 'high';
    } else if (ensembleScore >= 0.6) {
      return 'medium';
    } else {
      return 'low';
    }
  }
  
  private determineThreatType(
    mlResponse: MLServiceResponse,
    scores: { nlp: number; url: number; visual: number }
  ): ThreatType {
    if (scores.visual > 0.7) {
      return 'brand_impersonation';
    } else if (scores.url > 0.7) {
      return 'url_spoofing';
    } else if (mlResponse.nlp?.ai_generated_probability && mlResponse.nlp.ai_generated_probability > 0.7) {
      return 'ai_generated';
    } else {
      return 'email_phishing';
    }
  }
  
  private calculateConfidence(scores: { nlp: number; url: number; visual: number }): number {
    // Confidence is based on agreement between models
    const nonZeroScores = Object.values(scores).filter(s => s > 0);
    if (nonZeroScores.length === 0) return 0;
    
    const avgScore = nonZeroScores.reduce((a, b) => a + b, 0) / nonZeroScores.length;
    const variance = nonZeroScores.reduce((sum, score) => {
      return sum + Math.pow(score - avgScore, 2);
    }, 0) / nonZeroScores.length;
    
    // Lower variance = higher confidence
    const confidence = Math.max(0, 1 - variance);
    return confidence;
  }
  
  private extractIndicators(mlResponse: MLServiceResponse): string[] {
    const indicators: string[] = [];
    
    if (mlResponse.nlp?.urgency_score && mlResponse.nlp.urgency_score > 70) {
      indicators.push('high_urgency_language');
    }
    if (mlResponse.nlp?.ai_generated_probability && mlResponse.nlp.ai_generated_probability > 0.7) {
      indicators.push('ai_generated_content');
    }
    if (mlResponse.url) {
      const urlData = Array.isArray(mlResponse.url) ? mlResponse.url[0] : mlResponse.url;
      if (urlData.homoglyph_analysis?.is_suspicious) {
        indicators.push('homoglyph_attack');
      }
      if (urlData.redirect_analysis?.redirect_count && urlData.redirect_analysis.redirect_count > 3) {
        indicators.push('excessive_redirects');
      }
      if (urlData.obfuscation_analysis?.is_obfuscated) {
        indicators.push('url_obfuscation');
        const techniques = urlData.obfuscation_analysis.techniques_detected || [];
        for (const technique of techniques) {
          indicators.push(`obfuscation_${technique}`);
        }
      }
    }
    if (mlResponse.visual?.form_analysis?.is_suspicious) {
      indicators.push('credential_harvesting_form');
    }
    if (mlResponse.visual?.brand_prediction?.is_brand_impersonation) {
      indicators.push('brand_impersonation');
    }
    
    return indicators;
  }
}
