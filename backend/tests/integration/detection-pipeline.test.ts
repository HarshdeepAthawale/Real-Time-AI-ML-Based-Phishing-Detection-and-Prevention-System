/**
 * REAL Integration Tests - Detection Pipeline
 * 
 * These tests make ACTUAL API calls to running services.
 * NO MOCKS - tests the real system end-to-end.
 * 
 * Prerequisites:
 * - All services must be running (docker-compose up)
 * - Database must be initialized
 * - Valid API key must exist
 */

import axios from 'axios';
import * as dotenv from 'dotenv';

dotenv.config();

// Real service URLs
const DETECTION_API = process.env.DETECTION_API_URL || 'http://localhost:3001';
const THREAT_INTEL_API = process.env.THREAT_INTEL_URL || 'http://localhost:3002';
const NLP_SERVICE = process.env.NLP_SERVICE_URL || 'http://localhost:8001';
const URL_SERVICE = process.env.URL_SERVICE_URL || 'http://localhost:8002';

// Real API key (must be created via setup script)
const API_KEY = process.env.TEST_API_KEY;

describe('Detection Pipeline - Real Integration Tests', () => {
  
  beforeAll(async () => {
    // Wait for services to be ready
    console.log('⏳ Waiting for services to be ready...');
    await waitForServices();
    console.log('✅ All services ready');
  });

  describe('Health Checks', () => {
    test('Detection API should be healthy', async () => {
      const response = await axios.get(`${DETECTION_API}/health`);
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('healthy');
    });

    test('Threat Intel API should be healthy', async () => {
      const response = await axios.get(`${THREAT_INTEL_API}/health`);
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('healthy');
    });

    test('NLP Service should be healthy', async () => {
      const response = await axios.get(`${NLP_SERVICE}/health`);
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('healthy');
    });

    test('URL Service should be healthy', async () => {
      const response = await axios.get(`${URL_SERVICE}/health`);
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('healthy');
    });
  });

  describe('URL Detection - Real Analysis', () => {
    test('Should detect benign URL (google.com)', async () => {
      const response = await axios.post(
        `${DETECTION_API}/api/v1/detect/url`,
        {
          url: 'https://www.google.com',
          organizationId: 'test-org'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('is_threat');
      expect(response.data).toHaveProperty('confidence');
      expect(response.data).toHaveProperty('scores');
      expect(response.data.is_threat).toBe(false);
      expect(response.data.confidence).toBeLessThan(0.5);
      
      // Verify real ML scores
      expect(response.data.scores).toHaveProperty('nlp');
      expect(response.data.scores).toHaveProperty('url');
      expect(typeof response.data.scores.nlp).toBe('number');
      expect(typeof response.data.scores.url).toBe('number');
    }, 30000);

    test('Should detect suspicious patterns in URL', async () => {
      const response = await axios.post(
        `${DETECTION_API}/api/v1/detect/url`,
        {
          url: 'http://192.168.1.1/urgent-verify-account.php?token=abc123',
          organizationId: 'test-org'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data.is_threat).toBe(true);
      expect(response.data.confidence).toBeGreaterThan(0.5);
      
      // Should have real indicators
      expect(response.data.indicators).toBeDefined();
      expect(Array.isArray(response.data.indicators)).toBe(true);
      expect(response.data.indicators.length).toBeGreaterThan(0);
    }, 30000);

    test('Should analyze real domain with DNS/SSL/WHOIS', async () => {
      const response = await axios.post(
        `${URL_SERVICE}/analyze`,
        {
          url: 'https://github.com'
        }
      );

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('dns');
      expect(response.data).toHaveProperty('ssl');
      expect(response.data).toHaveProperty('whois');
      
      // Verify real DNS records
      expect(response.data.dns.has_dns).toBe(true);
      expect(response.data.dns.a_records.length).toBeGreaterThan(0);
      
      // Verify real SSL certificate
      expect(response.data.ssl.has_ssl).toBe(true);
      expect(response.data.ssl.certificate_valid).toBe(true);
      
      // Verify real WHOIS data
      expect(response.data.whois.has_whois).toBe(true);
      expect(response.data.whois.domain_age_days).toBeGreaterThan(0);
    }, 30000);
  });

  describe('Email Detection - Real Analysis', () => {
    test('Should detect phishing email content', async () => {
      const phishingEmail = `
URGENT: Your account will be suspended!

Dear valued customer,

We detected suspicious activity on your account. 
You must verify your identity immediately or your account will be locked.

Click here to verify: http://verify-account-now.tk/login.php

Act now! This link expires in 24 hours.

Thank you,
Security Team
      `;

      const response = await axios.post(
        `${DETECTION_API}/api/v1/detect/email`,
        {
          emailContent: phishingEmail,
          organizationId: 'test-org'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data.is_threat).toBe(true);
      expect(response.data.confidence).toBeGreaterThan(0.7);
      
      // Should detect urgency keywords
      const indicators = response.data.indicators.join(' ');
      expect(indicators.toLowerCase()).toMatch(/urgent|suspicious|verify|immediately/);
    }, 30000);

    test('Should handle benign email', async () => {
      const benignEmail = `
Hello,

This is a reminder about our team meeting tomorrow at 2 PM.
Please review the agenda before the meeting.

Best regards,
Team Lead
      `;

      const response = await axios.post(
        `${DETECTION_API}/api/v1/detect/email`,
        {
          emailContent: benignEmail,
          organizationId: 'test-org'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data.is_threat).toBe(false);
      expect(response.data.confidence).toBeLessThan(0.5);
    }, 30000);
  });

  describe('Threat Intelligence - Real Lookups', () => {
    test('Should check URL against threat feeds', async () => {
      const response = await axios.post(
        `${THREAT_INTEL_API}/api/v1/intelligence/check/url`,
        {
          url: 'https://example.com'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('is_threat');
      expect(response.data).toHaveProperty('sources');
      expect(Array.isArray(response.data.sources)).toBe(true);
    }, 30000);

    test('Should provide threat intel statistics', async () => {
      const response = await axios.get(
        `${THREAT_INTEL_API}/api/v1/intelligence/stats`,
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('feeds');
      expect(response.data).toHaveProperty('bloom_filters');
      expect(response.data.bloom_filters).toHaveProperty('filters');
    }, 30000);
  });

  describe('End-to-End Detection Flow', () => {
    test('Complete detection pipeline with all services', async () => {
      const testURL = 'http://suspicious-phishing-test.example.com';
      
      // Step 1: Submit for detection
      const detectionResponse = await axios.post(
        `${DETECTION_API}/api/v1/detect/url`,
        {
          url: testURL,
          organizationId: 'test-org'
        },
        {
          headers: { 'X-API-Key': API_KEY }
        }
      );

      expect(detectionResponse.status).toBe(200);
      
      // Step 2: Verify result structure
      const result = detectionResponse.data;
      expect(result).toHaveProperty('detection_id');
      expect(result).toHaveProperty('is_threat');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('scores');
      expect(result).toHaveProperty('metadata');
      
      // Step 3: Verify all ML models were called
      expect(result.scores).toHaveProperty('nlp');
      expect(result.scores).toHaveProperty('url');
      
      // Step 4: Verify processing time is reasonable
      expect(result.metadata.processing_time_ms).toBeLessThan(10000);
      
      // Step 5: Verify threat intel was checked
      expect(result.metadata).toHaveProperty('threat_intel_checked');
      
      console.log('✅ Complete pipeline test passed:', {
        url: testURL,
        is_threat: result.is_threat,
        confidence: result.confidence,
        processing_time: result.metadata.processing_time_ms
      });
    }, 30000);
  });

  describe('Performance Tests', () => {
    test('Should handle concurrent requests', async () => {
      const urls = [
        'https://google.com',
        'https://github.com',
        'https://microsoft.com',
        'https://amazon.com',
        'https://apple.com'
      ];

      const startTime = Date.now();
      
      const promises = urls.map(url =>
        axios.post(
          `${DETECTION_API}/api/v1/detect/url`,
          { url, organizationId: 'test-org' },
          { headers: { 'X-API-Key': API_KEY } }
        )
      );

      const responses = await Promise.all(promises);
      const endTime = Date.now();
      
      // All should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
      });

      // Should complete in reasonable time (parallel processing)
      const totalTime = endTime - startTime;
      console.log(`✅ Processed ${urls.length} URLs concurrently in ${totalTime}ms`);
      
      expect(totalTime).toBeLessThan(15000);
    }, 30000);

    test('Should use caching for repeated URLs', async () => {
      const url = 'https://example.com';
      
      // First request (should be slower)
      const start1 = Date.now();
      const response1 = await axios.post(
        `${DETECTION_API}/api/v1/detect/url`,
        { url, organizationId: 'test-org' },
        { headers: { 'X-API-Key': API_KEY } }
      );
      const time1 = Date.now() - start1;

      // Second request (should be cached)
      const start2 = Date.now();
      const response2 = await axios.post(
        `${DETECTION_API}/api/v1/detect/url`,
        { url, organizationId: 'test-org' },
        { headers: { 'X-API-Key': API_KEY } }
      );
      const time2 = Date.now() - start2;

      expect(response1.status).toBe(200);
      expect(response2.status).toBe(200);
      
      // Results should be consistent
      expect(response1.data.is_threat).toBe(response2.data.is_threat);
      
      // Second request should be faster (cached)
      console.log(`⚡ Cache performance: First=${time1}ms, Second=${time2}ms`);
      expect(time2).toBeLessThan(time1);
    }, 30000);
  });
});

/**
 * Wait for all services to be ready
 */
async function waitForServices(maxWaitTime = 60000): Promise<void> {
  const services = [
    { name: 'Detection API', url: `${DETECTION_API}/health` },
    { name: 'Threat Intel', url: `${THREAT_INTEL_API}/health` },
    { name: 'NLP Service', url: `${NLP_SERVICE}/health` },
    { name: 'URL Service', url: `${URL_SERVICE}/health` }
  ];

  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitTime) {
    try {
      const checks = await Promise.all(
        services.map(service =>
          axios.get(service.url, { timeout: 2000 })
            .then(() => ({ ...service, ready: true }))
            .catch(() => ({ ...service, ready: false }))
        )
      );

      const allReady = checks.every(check => check.ready);
      
      if (allReady) {
        return;
      }

      const notReady = checks.filter(c => !c.ready).map(c => c.name);
      console.log(`⏳ Waiting for: ${notReady.join(', ')}`);
      
      await new Promise(resolve => setTimeout(resolve, 2000));
    } catch (error) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  throw new Error('Services did not become ready in time');
}

/**
 * Test Configuration
 */
const testConfig = {
  timeout: 30000,
  verbose: true
};

// Only run if API key is configured
if (!API_KEY) {
  console.error('❌ TEST_API_KEY environment variable not set');
  console.log('Create an API key using the setup script first:');
  console.log('  cd backend/shared/scripts');
  console.log('  ts-node create-initial-setup.ts ...');
  process.exit(1);
}
