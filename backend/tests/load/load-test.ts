/**
 * REAL Load Tests - Performance Testing
 * 
 * Tests the system under realistic load conditions.
 * NO MOCKS - actual performance testing.
 * 
 * Usage:
 *   npm run test:load
 *   npm run test:load -- --users 100 --duration 60
 */

import axios from 'axios';
import * as dotenv from 'dotenv';

dotenv.config();

interface LoadTestConfig {
  baseURL: string;
  apiKey: string;
  concurrentUsers: number;
  testDuration: number; // seconds
  requestsPerUser: number;
}

interface LoadTestResults {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_response_time: number;
  min_response_time: number;
  max_response_time: number;
  p95_response_time: number;
  p99_response_time: number;
  requests_per_second: number;
  errors: Array<{ type: string; count: number }>;
}

class LoadTester {
  private config: LoadTestConfig;
  private results: {
    response_times: number[];
    errors: Map<string, number>;
    start_time: number;
    end_time: number;
    successful: number;
    failed: number;
  };

  constructor(config: LoadTestConfig) {
    this.config = config;
    this.results = {
      response_times: [],
      errors: new Map(),
      start_time: 0,
      end_time: 0,
      successful: 0,
      failed: 0
    };
  }

  /**
   * Run load test
   */
  async run(): Promise<LoadTestResults> {
    console.log('🚀 Starting Load Test');
    console.log(`   Concurrent Users: ${this.config.concurrentUsers}`);
    console.log(`   Duration: ${this.config.testDuration}s`);
    console.log(`   Target: ${this.config.baseURL}`);
    console.log('');

    this.results.start_time = Date.now();

    // Create user sessions
    const users = [];
    for (let i = 0; i < this.config.concurrentUsers; i++) {
      users.push(this.simulateUser(i));
    }

    // Wait for all users to complete
    await Promise.all(users);

    this.results.end_time = Date.now();

    return this.calculateResults();
  }

  /**
   * Simulate a single user making requests
   */
  private async simulateUser(userId: number): Promise<void> {
    const testURLs = [
      'https://google.com',
      'https://github.com',
      'https://microsoft.com',
      'https://amazon.com',
      'https://facebook.com',
      'http://suspicious-test-phishing.example.com',
      'http://192.168.1.1/verify.php',
      'https://bit.ly/test123'
    ];

    const endTime = Date.now() + (this.config.testDuration * 1000);

    while (Date.now() < endTime) {
      try {
        // Random URL from test set
        const url = testURLs[Math.floor(Math.random() * testURLs.length)];
        
        const startTime = Date.now();
        
        const response = await axios.post(
          `${this.config.baseURL}/api/v1/detect/url`,
          {
            url,
            organizationId: `user-${userId}`
          },
          {
            headers: { 'X-API-Key': this.config.apiKey },
            timeout: 30000
          }
        );

        const responseTime = Date.now() - startTime;
        this.results.response_times.push(responseTime);
        this.results.successful++;

        // Random delay between requests (1-3 seconds)
        const delay = 1000 + Math.random() * 2000;
        await new Promise(resolve => setTimeout(resolve, delay));

      } catch (error: any) {
        this.results.failed++;
        
        const errorType = error.response?.status 
          ? `HTTP_${error.response.status}` 
          : error.code || 'UNKNOWN';
        
        this.results.errors.set(
          errorType,
          (this.results.errors.get(errorType) || 0) + 1
        );

        // Small delay on error
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
  }

  /**
   * Calculate final results
   */
  private calculateResults(): LoadTestResults {
    const times = this.results.response_times.sort((a, b) => a - b);
    const totalRequests = this.results.successful + this.results.failed;
    const duration = (this.results.end_time - this.results.start_time) / 1000;

    return {
      total_requests: totalRequests,
      successful_requests: this.results.successful,
      failed_requests: this.results.failed,
      avg_response_time: times.length > 0 
        ? times.reduce((a, b) => a + b, 0) / times.length 
        : 0,
      min_response_time: times.length > 0 ? times[0] : 0,
      max_response_time: times.length > 0 ? times[times.length - 1] : 0,
      p95_response_time: times.length > 0 
        ? times[Math.floor(times.length * 0.95)] 
        : 0,
      p99_response_time: times.length > 0 
        ? times[Math.floor(times.length * 0.99)] 
        : 0,
      requests_per_second: totalRequests / duration,
      errors: Array.from(this.results.errors.entries()).map(([type, count]) => ({
        type,
        count
      }))
    };
  }

  /**
   * Print results
   */
  static printResults(results: LoadTestResults): void {
    console.log('\n📊 Load Test Results');
    console.log('═'.repeat(60));
    console.log('');
    
    console.log('Request Statistics:');
    console.log(`  Total Requests:     ${results.total_requests}`);
    console.log(`  Successful:         ${results.successful_requests} (${(results.successful_requests / results.total_requests * 100).toFixed(1)}%)`);
    console.log(`  Failed:             ${results.failed_requests} (${(results.failed_requests / results.total_requests * 100).toFixed(1)}%)`);
    console.log(`  Requests/sec:       ${results.requests_per_second.toFixed(2)}`);
    console.log('');

    console.log('Response Time (ms):');
    console.log(`  Average:            ${results.avg_response_time.toFixed(0)}ms`);
    console.log(`  Min:                ${results.min_response_time}ms`);
    console.log(`  Max:                ${results.max_response_time}ms`);
    console.log(`  P95:                ${results.p95_response_time}ms`);
    console.log(`  P99:                ${results.p99_response_time}ms`);
    console.log('');

    if (results.errors.length > 0) {
      console.log('Errors:');
      results.errors.forEach(error => {
        console.log(`  ${error.type}: ${error.count}`);
      });
      console.log('');
    }

    // Performance evaluation
    console.log('Performance Evaluation:');
    if (results.avg_response_time < 3000) {
      console.log('  ✅ EXCELLENT - Average response time under 3s');
    } else if (results.avg_response_time < 5000) {
      console.log('  ✅ GOOD - Average response time under 5s');
    } else if (results.avg_response_time < 10000) {
      console.log('  ⚠️  ACCEPTABLE - Average response time under 10s');
    } else {
      console.log('  ❌ POOR - Average response time over 10s');
    }

    const successRate = (results.successful_requests / results.total_requests) * 100;
    if (successRate >= 99) {
      console.log('  ✅ EXCELLENT - Success rate >= 99%');
    } else if (successRate >= 95) {
      console.log('  ✅ GOOD - Success rate >= 95%');
    } else if (successRate >= 90) {
      console.log('  ⚠️  ACCEPTABLE - Success rate >= 90%');
    } else {
      console.log('  ❌ POOR - Success rate < 90%');
    }

    console.log('');
    console.log('═'.repeat(60));
  }
}

/**
 * Run load test scenarios
 */
async function runLoadTests(): Promise<void> {
  const baseURL = process.env.DETECTION_API_URL || 'http://localhost:3001';
  const apiKey = process.env.TEST_API_KEY;

  if (!apiKey) {
    console.error('❌ TEST_API_KEY environment variable not set');
    process.exit(1);
  }

  // Scenario 1: Light Load
  console.log('\n🔹 Scenario 1: Light Load');
  const lightLoad = new LoadTester({
    baseURL,
    apiKey,
    concurrentUsers: 5,
    testDuration: 30,
    requestsPerUser: 10
  });
  const lightResults = await lightLoad.run();
  LoadTester.printResults(lightResults);

  // Wait between scenarios
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Scenario 2: Medium Load
  console.log('\n🔹 Scenario 2: Medium Load');
  const mediumLoad = new LoadTester({
    baseURL,
    apiKey,
    concurrentUsers: 20,
    testDuration: 60,
    requestsPerUser: 20
  });
  const mediumResults = await mediumLoad.run();
  LoadTester.printResults(mediumResults);

  // Wait between scenarios
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Scenario 3: Heavy Load (optional, comment out if needed)
  console.log('\n🔹 Scenario 3: Heavy Load');
  const heavyLoad = new LoadTester({
    baseURL,
    apiKey,
    concurrentUsers: 50,
    testDuration: 120,
    requestsPerUser: 30
  });
  const heavyResults = await heavyLoad.run();
  LoadTester.printResults(heavyResults);

  // Summary
  console.log('\n📈 Overall Summary');
  console.log('═'.repeat(60));
  console.log(`Light Load:  ${lightResults.avg_response_time.toFixed(0)}ms avg, ${lightResults.requests_per_second.toFixed(1)} req/s`);
  console.log(`Medium Load: ${mediumResults.avg_response_time.toFixed(0)}ms avg, ${mediumResults.requests_per_second.toFixed(1)} req/s`);
  console.log(`Heavy Load:  ${heavyResults.avg_response_time.toFixed(0)}ms avg, ${heavyResults.requests_per_second.toFixed(1)} req/s`);
  console.log('');
}

// Run tests
if (require.main === module) {
  runLoadTests()
    .then(() => {
      console.log('✅ Load tests completed');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Load tests failed:', error);
      process.exit(1);
    });
}

export { LoadTester, LoadTestConfig, LoadTestResults };
