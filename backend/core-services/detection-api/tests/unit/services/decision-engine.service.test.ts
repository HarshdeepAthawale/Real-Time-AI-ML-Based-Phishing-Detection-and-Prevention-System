import { DecisionEngineService } from '../../../src/services/decision-engine.service';
import {
  mockMLServiceResponse,
  mockMLServiceResponseEmail,
  mockNLPResponse,
  mockURLResponse,
  mockVisualResponse,
  mockNLPResponseLow,
  mockURLResponseLegitimate,
  mockVisualResponseLegitimate,
} from '../../fixtures/mock-responses';
import { ThreatSeverity, ThreatType } from '../../../src/models/detection.model';

describe('DecisionEngineService', () => {
  let decisionEngine: DecisionEngineService;

  beforeEach(() => {
    decisionEngine = new DecisionEngineService();
  });

  describe('makeDecision', () => {
    it('should make decision with all ML responses', () => {
      const result = decisionEngine.makeDecision(mockMLServiceResponse, {});

      expect(result.isThreat).toBe(true);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.severity).toBe('high');
      expect(result.scores.ensemble).toBeGreaterThan(0.7);
      expect(result.indicators.length).toBeGreaterThan(0);
    });

    it('should identify threat when ensemble score >= 0.7', () => {
      const result = decisionEngine.makeDecision(mockMLServiceResponse, {});

      expect(result.isThreat).toBe(true);
      expect(result.scores.ensemble).toBeGreaterThanOrEqual(0.7);
    });

    it('should not identify threat when ensemble score < 0.7', () => {
      const lowThreatResponse = {
        nlp: mockNLPResponseLow,
        url: mockURLResponseLegitimate,
        visual: mockVisualResponseLegitimate,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(lowThreatResponse, {});

      expect(result.isThreat).toBe(false);
      expect(result.scores.ensemble).toBeLessThan(0.7);
    });

    it('should include processing time in metadata', () => {
      const result = decisionEngine.makeDecision(mockMLServiceResponse, {});

      expect(result.metadata.processingTimeMs).toBe(150);
      expect(result.metadata.timestamp).toBeDefined();
    });
  });

  describe('score calculation', () => {
    it('should calculate NLP score with urgency boost', () => {
      const nlpWithUrgency = {
        ...mockNLPResponse,
        urgency_score: 80,
      };

      const response = {
        nlp: nlpWithUrgency,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.nlp).toBeGreaterThan(mockNLPResponse.phishing_probability);
    });

    it('should calculate NLP score with AI-generated boost', () => {
      const nlpWithAI = {
        ...mockNLPResponse,
        ai_generated_probability: 0.8,
      };

      const response = {
        nlp: nlpWithAI,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.nlp).toBeGreaterThan(mockNLPResponse.phishing_probability);
    });

    it('should calculate URL score from phishing_probability', () => {
      const response = {
        nlp: null,
        url: mockURLResponse,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.url).toBe(mockURLResponse.phishing_probability);
    });

    it('should calculate URL score from analysis components', () => {
      const urlWithoutProbability = {
        ...mockURLResponse,
        phishing_probability: undefined,
      };

      const response = {
        nlp: null,
        url: urlWithoutProbability,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.url).toBeGreaterThan(0);
    });

    it('should calculate visual score from phishing_probability', () => {
      const response = {
        nlp: null,
        url: null,
        visual: mockVisualResponse,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.visual).toBe(mockVisualResponse.phishing_probability);
    });

    it('should handle array URL responses', () => {
      const response = {
        nlp: null,
        url: [mockURLResponse],
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.url).toBeGreaterThan(0);
    });
  });

  describe('ensemble score calculation', () => {
    it('should calculate weighted ensemble score', () => {
      const result = decisionEngine.makeDecision(mockMLServiceResponse, {});

      const expected = 
        (result.scores.nlp * 0.4) +
        (result.scores.url * 0.4) +
        (result.scores.visual * 0.2);

      expect(result.scores.ensemble).toBeCloseTo(expected, 2);
    });

    it('should handle missing service responses', () => {
      const response = {
        nlp: mockNLPResponse,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.ensemble).toBe(result.scores.nlp);
    });

    it('should return 0 when no services respond', () => {
      const response = {
        nlp: null,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(response, {});
      expect(result.scores.ensemble).toBe(0);
    });
  });

  describe('severity determination', () => {
    it('should return critical for score >= 0.9', () => {
      const highScoreResponse = {
        nlp: { ...mockNLPResponse, phishing_probability: 0.95 },
        url: { ...mockURLResponse, phishing_probability: 0.95 },
        visual: { ...mockVisualResponse, phishing_probability: 0.95 },
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(highScoreResponse, {});
      expect(result.severity).toBe('critical');
    });

    it('should return high for score >= 0.75', () => {
      const result = decisionEngine.makeDecision(mockMLServiceResponse, {});
      expect(result.severity).toBe('high');
    });

    it('should return medium for score >= 0.6', () => {
      const mediumResponse = {
        nlp: { ...mockNLPResponse, phishing_probability: 0.65 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(mediumResponse, {});
      expect(result.severity).toBe('medium');
    });

    it('should return low for score < 0.6', () => {
      const lowResponse = {
        nlp: mockNLPResponseLow,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(lowResponse, {});
      expect(result.severity).toBe('low');
    });
  });

  describe('threat type determination', () => {
    it('should return brand_impersonation when visual score > 0.7', () => {
      const visualThreatResponse = {
        nlp: null,
        url: null,
        visual: { ...mockVisualResponse, phishing_probability: 0.8 },
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(visualThreatResponse, {});
      expect(result.threatType).toBe('brand_impersonation');
    });

    it('should return url_spoofing when URL score > 0.7', () => {
      const urlThreatResponse = {
        nlp: null,
        url: { ...mockURLResponse, phishing_probability: 0.8 },
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(urlThreatResponse, {});
      expect(result.threatType).toBe('url_spoofing');
    });

    it('should return ai_generated when AI probability > 0.7', () => {
      const aiResponse = {
        nlp: { ...mockNLPResponse, ai_generated_probability: 0.8 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(aiResponse, {});
      expect(result.threatType).toBe('ai_generated');
    });

    it('should return email_phishing as default', () => {
      const defaultResponse = {
        nlp: mockNLPResponse,
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(defaultResponse, {});
      expect(result.threatType).toBe('email_phishing');
    });
  });

  describe('confidence calculation', () => {
    it('should calculate high confidence when scores agree', () => {
      const agreeingResponse = {
        nlp: { ...mockNLPResponse, phishing_probability: 0.8 },
        url: { ...mockURLResponse, phishing_probability: 0.8 },
        visual: { ...mockVisualResponse, phishing_probability: 0.8 },
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(agreeingResponse, {});
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('should calculate low confidence when scores disagree', () => {
      const disagreeingResponse = {
        nlp: { ...mockNLPResponse, phishing_probability: 0.9 },
        url: { ...mockURLResponse, phishing_probability: 0.1 },
        visual: { ...mockVisualResponse, phishing_probability: 0.9 },
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(disagreeingResponse, {});
      expect(result.confidence).toBeLessThan(0.5);
    });
  });

  describe('indicator extraction', () => {
    it('should extract high urgency language indicator', () => {
      const urgentResponse = {
        nlp: { ...mockNLPResponse, urgency_score: 80 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(urgentResponse, {});
      expect(result.indicators).toContain('high_urgency_language');
    });

    it('should extract AI generated content indicator', () => {
      const aiResponse = {
        nlp: { ...mockNLPResponse, ai_generated_probability: 0.8 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(aiResponse, {});
      expect(result.indicators).toContain('ai_generated_content');
    });

    it('should extract homoglyph attack indicator', () => {
      const homoglyphResponse = {
        nlp: null,
        url: mockURLResponse,
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(homoglyphResponse, {});
      expect(result.indicators).toContain('homoglyph_attack');
    });

    it('should extract excessive redirects indicator', () => {
      const redirectResponse = {
        nlp: null,
        url: {
          ...mockURLResponse,
          redirect_analysis: { is_suspicious: true, redirect_count: 5 },
        },
        visual: null,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(redirectResponse, {});
      expect(result.indicators).toContain('excessive_redirects');
    });

    it('should extract brand impersonation indicator', () => {
      const brandResponse = {
        nlp: null,
        url: null,
        visual: mockVisualResponse,
        processingTimeMs: 100,
      };

      const result = decisionEngine.makeDecision(brandResponse, {});
      expect(result.indicators).toContain('brand_impersonation');
    });
  });
});
