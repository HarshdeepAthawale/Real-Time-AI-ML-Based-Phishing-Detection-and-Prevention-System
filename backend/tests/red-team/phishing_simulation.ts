/**
 * Red-Team Phishing Simulation Harness
 *
 * Injects known phishing and legitimate samples into the detection API
 * and measures detection accuracy, false positive rate, and latency.
 *
 * Usage:
 *   npx ts-node backend/tests/red-team/phishing_simulation.ts
 *
 * Environment:
 *   DETECTION_API_URL - Detection API base URL (default: http://localhost:3002)
 *   API_KEY - Valid API key for the detection service
 */

import axios, { AxiosInstance } from 'axios';

const API_URL = process.env.DETECTION_API_URL || 'http://localhost:3002';
const API_KEY = process.env.API_KEY || 'test-api-key';

interface TestCase {
  name: string;
  type: 'email' | 'url' | 'text' | 'sms';
  input: Record<string, any>;
  expectedThreat: boolean;
  category: string;
}

// ---------- Test Cases ----------

const TEST_CASES: TestCase[] = [
  // === PHISHING EMAILS (should be detected) ===
  {
    name: 'Classic credential phishing',
    type: 'email',
    input: {
      emailContent: 'Subject: URGENT Account Verification Required\nFrom: security@paypa1.com\n\nDear Customer,\nYour PayPal account has been limited due to unusual activity. Click here immediately to verify your identity: http://paypa1-verify.xyz/login\nFailure to verify within 24 hours will result in permanent account suspension.',
    },
    expectedThreat: true,
    category: 'credential_phishing',
  },
  {
    name: 'CEO fraud / BEC attack',
    type: 'email',
    input: {
      emailContent: 'Subject: Urgent Request\nFrom: ceo@company.com\n\nHi,\nI need you to purchase 10 Amazon gift cards worth $100 each for a client meeting today. Please send me the redemption codes via email ASAP. This is urgent and confidential.\n\nThanks,\nJohn CEO',
    },
    expectedThreat: true,
    category: 'bec_attack',
  },
  {
    name: 'Invoice phishing with attachment lure',
    type: 'email',
    input: {
      emailContent: 'Subject: Invoice #INV-2024-38291 - Payment Overdue\nFrom: billing@quickbooks-invoices.xyz\n\nYour invoice is 30 days overdue. Total amount: $4,892.00\nClick to view and pay immediately to avoid late fees and collection: http://quickbooks-pay.xyz/invoice/38291\nThis is your FINAL NOTICE.',
    },
    expectedThreat: true,
    category: 'invoice_phishing',
  },

  // === PHISHING URLs (should be detected) ===
  {
    name: 'Typosquatting domain',
    type: 'url',
    input: { url: 'https://www.paypa1.com/signin/verify' },
    expectedThreat: true,
    category: 'typosquatting',
  },
  {
    name: 'IP address phishing',
    type: 'url',
    input: { url: 'http://192.168.1.100/login.php?redirect=bank' },
    expectedThreat: true,
    category: 'ip_phishing',
  },
  {
    name: 'Obfuscated URL',
    type: 'url',
    input: { url: 'https://bit.ly/3xR9kP2' },
    expectedThreat: true,
    category: 'url_shortener',
  },

  // === PHISHING SMS (should be detected) ===
  {
    name: 'SMS package scam',
    type: 'sms',
    input: {
      message: 'USPS: Your package is being held at customs. Pay $3.99 to release. Track: http://usps-customs.xyz/pay',
      sender: '+1234567890',
    },
    expectedThreat: true,
    category: 'sms_phishing',
  },
  {
    name: 'SMS bank alert scam',
    type: 'sms',
    input: {
      message: 'ALERT: Suspicious transaction of $2,847 on your account. If unauthorized, verify at bankofamerica-alerts.com/verify',
      sender: '+1987654321',
    },
    expectedThreat: true,
    category: 'sms_phishing',
  },

  // === PHISHING TEXT (should be detected) ===
  {
    name: 'AI-generated phishing text',
    type: 'text',
    input: {
      text: 'As part of our ongoing commitment to maintaining the highest standards of cybersecurity, we are requesting that all account holders verify their credentials through our newly upgraded authentication system. This process is essential to ensuring the continued protection of your personal information.',
    },
    expectedThreat: true,
    category: 'ai_generated',
  },

  // === LEGITIMATE EMAILS (should NOT be flagged) ===
  {
    name: 'Normal business email',
    type: 'email',
    input: {
      emailContent: 'Subject: Q4 Budget Review Meeting\nFrom: sarah@company.com\n\nHi team,\nJust a reminder that the Q4 budget review meeting is tomorrow at 2 PM in the main conference room. Please bring your department projections.\n\nThanks,\nSarah',
    },
    expectedThreat: false,
    category: 'legitimate_business',
  },
  {
    name: 'Normal shipping notification',
    type: 'email',
    input: {
      emailContent: 'Subject: Your order has shipped!\nFrom: noreply@amazon.com\n\nYour order #112-3456789 has shipped and will arrive by Friday. Track your package at ups.com with tracking number 1Z999AA10123456784.',
    },
    expectedThreat: false,
    category: 'legitimate_notification',
  },

  // === LEGITIMATE URLs (should NOT be flagged) ===
  {
    name: 'Google login page',
    type: 'url',
    input: { url: 'https://accounts.google.com/signin' },
    expectedThreat: false,
    category: 'legitimate_url',
  },
  {
    name: 'GitHub repository',
    type: 'url',
    input: { url: 'https://github.com/microsoft/vscode' },
    expectedThreat: false,
    category: 'legitimate_url',
  },

  // === LEGITIMATE TEXT (should NOT be flagged) ===
  {
    name: 'Normal text message',
    type: 'text',
    input: {
      text: 'Hey, the meeting got pushed to 3 PM. Conference room B instead. See you there!',
    },
    expectedThreat: false,
    category: 'legitimate_text',
  },
  {
    name: 'Normal SMS',
    type: 'sms',
    input: {
      message: 'Your Uber ride is arriving in 3 minutes. Driver: Mike in a blue Toyota Camry.',
      sender: '+1555000123',
    },
    expectedThreat: false,
    category: 'legitimate_sms',
  },
];

// ---------- Execution ----------

interface TestResult {
  name: string;
  type: string;
  category: string;
  expectedThreat: boolean;
  actualThreat: boolean;
  correct: boolean;
  confidence: number;
  severity: string;
  latencyMs: number;
  error?: string;
}

async function runSimulation(): Promise<void> {
  const client: AxiosInstance = axios.create({
    baseURL: `${API_URL}/api/v1/detect`,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    timeout: 30000,
  });

  console.log('='.repeat(70));
  console.log('RED-TEAM PHISHING SIMULATION');
  console.log('='.repeat(70));
  console.log(`API: ${API_URL}`);
  console.log(`Test Cases: ${TEST_CASES.length}`);
  console.log('');

  const results: TestResult[] = [];

  for (const testCase of TEST_CASES) {
    const start = Date.now();
    let result: TestResult;

    try {
      const endpoint = testCase.type === 'sms' ? '/sms' : `/${testCase.type}`;
      const response = await client.post(endpoint, testCase.input);
      const latency = Date.now() - start;

      const isThreat = response.data.isThreat || false;
      const confidence = response.data.confidence || 0;
      const severity = response.data.severity || 'none';

      result = {
        name: testCase.name,
        type: testCase.type,
        category: testCase.category,
        expectedThreat: testCase.expectedThreat,
        actualThreat: isThreat,
        correct: isThreat === testCase.expectedThreat,
        confidence,
        severity,
        latencyMs: latency,
      };
    } catch (error: any) {
      result = {
        name: testCase.name,
        type: testCase.type,
        category: testCase.category,
        expectedThreat: testCase.expectedThreat,
        actualThreat: false,
        correct: false,
        confidence: 0,
        severity: 'error',
        latencyMs: Date.now() - start,
        error: error.message,
      };
    }

    results.push(result);

    const status = result.error ? 'ERR' : result.correct ? 'OK ' : 'FAIL';
    const emoji = result.error ? 'x' : result.correct ? '+' : '-';
    console.log(
      `  [${status}] ${testCase.name} | Expected=${testCase.expectedThreat ? 'THREAT' : 'SAFE  '} Got=${result.actualThreat ? 'THREAT' : 'SAFE  '} | Conf=${result.confidence.toFixed(2)} Sev=${result.severity} | ${result.latencyMs}ms`
    );
  }

  // ---------- Summary ----------
  console.log('\n' + '='.repeat(70));
  console.log('SIMULATION RESULTS');
  console.log('='.repeat(70));

  const total = results.length;
  const correct = results.filter((r) => r.correct).length;
  const errors = results.filter((r) => r.error).length;

  // Phishing samples
  const phishingSamples = results.filter((r) => r.expectedThreat);
  const truePositives = phishingSamples.filter((r) => r.actualThreat && r.correct).length;
  const falseNegatives = phishingSamples.filter((r) => !r.actualThreat).length;

  // Legitimate samples
  const legitimateSamples = results.filter((r) => !r.expectedThreat);
  const trueNegatives = legitimateSamples.filter((r) => !r.actualThreat && r.correct).length;
  const falsePositives = legitimateSamples.filter((r) => r.actualThreat).length;

  const tpr = phishingSamples.length > 0 ? truePositives / phishingSamples.length : 0;
  const fpr = legitimateSamples.length > 0 ? falsePositives / legitimateSamples.length : 0;
  const accuracy = total > 0 ? correct / total : 0;

  const latencies = results.filter((r) => !r.error).map((r) => r.latencyMs).sort((a, b) => a - b);
  const p50 = latencies[Math.floor(latencies.length * 0.5)] || 0;
  const p95 = latencies[Math.floor(latencies.length * 0.95)] || 0;
  const p99 = latencies[Math.floor(latencies.length * 0.99)] || 0;

  console.log(`  Total:          ${total}`);
  console.log(`  Correct:        ${correct} (${(accuracy * 100).toFixed(1)}%)`);
  console.log(`  Errors:         ${errors}`);
  console.log(`  True Positives: ${truePositives}`);
  console.log(`  True Negatives: ${trueNegatives}`);
  console.log(`  False Positives:${falsePositives}`);
  console.log(`  False Negatives:${falseNegatives}`);
  console.log('');
  console.log(`  TPR (recall):   ${(tpr * 100).toFixed(1)}% (target: >= 95%)`);
  console.log(`  FPR:            ${(fpr * 100).toFixed(1)}% (target: <= 2%)`);
  console.log(`  Accuracy:       ${(accuracy * 100).toFixed(1)}%`);
  console.log('');
  console.log(`  Latency p50:    ${p50}ms`);
  console.log(`  Latency p95:    ${p95}ms (target: <= 100ms)`);
  console.log(`  Latency p99:    ${p99}ms`);
  console.log('='.repeat(70));

  // Pass/fail assessment
  const tprPassed = tpr >= 0.95;
  const fprPassed = fpr <= 0.02;
  const latencyPassed = p95 <= 100;

  console.log(`\n  TPR Target:     ${tprPassed ? 'PASS' : 'FAIL'}`);
  console.log(`  FPR Target:     ${fprPassed ? 'PASS' : 'FAIL'}`);
  console.log(`  Latency Target: ${latencyPassed ? 'PASS' : 'FAIL'}`);

  const overallPass = tprPassed && fprPassed;
  console.log(`\n  Overall:        ${overallPass ? 'PASS' : 'FAIL'}`);
  console.log('');

  // Exit with appropriate code
  process.exit(overallPass ? 0 : 1);
}

runSimulation().catch((error) => {
  console.error('Simulation failed:', error);
  process.exit(1);
});
