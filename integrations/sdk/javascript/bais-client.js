/**
 * BAIS JavaScript/TypeScript SDK Client
 * 
 * Simple client for integrating BAIS governance into any JavaScript application.
 * 
 * Usage:
 *   import { BAISClient } from 'bais-client';
 *   
 *   const client = new BAISClient({ apiUrl: 'https://api.bais.invitas.ai' });
 *   
 *   const result = await client.audit({
 *     query: 'What is 2+2?',
 *     response: 'The answer is 4.',
 *     domain: 'general'
 *   });
 *   
 *   if (result.approved) {
 *     console.log('Response approved');
 *   }
 */

class BAISClient {
  /**
   * Initialize BAIS client.
   * 
   * @param {Object} options - Configuration options
   * @param {string} options.apiUrl - Base URL of BAIS API
   * @param {string} [options.apiKey] - Optional API key for authentication
   * @param {number} [options.timeout] - Request timeout in milliseconds (default: 30000)
   */
  constructor(options = {}) {
    this.apiUrl = (options.apiUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = options.apiKey || null;
    this.timeout = options.timeout || 30000;
  }

  /**
   * Audit an LLM response for bias, errors, and quality issues.
   * 
   * @param {Object} params - Audit parameters
   * @param {string} params.query - The user's original query
   * @param {string} params.response - The LLM's response to audit
   * @param {string} [params.domain] - Domain context (general, medical, financial, legal)
   * @param {Array} [params.documents] - Optional source documents for grounding verification
   * @returns {Promise<AuditResult>} Audit result with approval status and detected issues
   */
  async audit({ query, response, domain = 'general', documents = [] }) {
    const result = await this._post('/governance/audit', {
      query,
      response,
      domain,
      documents
    });

    return {
      approved: result.decision === 'approved',
      confidence: (result.accuracy || 0) / 100,
      issues: result.issues || [],
      warnings: result.warnings || [],
      recommendation: result.recommendation || '',
      shouldRegenerate: result.should_regenerate || false,
      improvedResponse: result.improved_response || null
    };
  }

  /**
   * Verify a completion claim against evidence.
   * 
   * @param {Object} params - Verification parameters
   * @param {string} params.claim - The claim to verify
   * @param {Array<string>} params.evidence - List of evidence items
   * @returns {Promise<VerificationResult>} Verification result with validity and clinical status
   */
  async verifyCompletion({ claim, evidence }) {
    const result = await this._post('/governance/verify', {
      claim,
      evidence
    });

    return {
      valid: result.valid || false,
      confidence: result.confidence || 0,
      violations: result.violations || [],
      clinicalStatus: result.clinical_status || 'unknown'
    };
  }

  /**
   * Pre-check a query for manipulation or injection attempts.
   * 
   * @param {string} query - The user query to check
   * @returns {Promise<Object>} Result with safe, riskLevel, and any issues
   */
  async checkQuery(query) {
    return await this._post('/governance/check_query', { query });
  }

  /**
   * Improve a response based on detected issues.
   * 
   * @param {Object} params - Improvement parameters
   * @param {string} params.response - The response to improve
   * @param {Array<string>} params.issues - List of issues to address
   * @returns {Promise<string>} Improved response text
   */
  async improveResponse({ response, issues }) {
    const result = await this._post('/governance/improve', {
      response,
      issues
    });
    return result.improved_response || response;
  }

  /**
   * Get governance statistics.
   * 
   * @returns {Promise<Object>} Statistics object
   */
  async getStatistics() {
    return await this._get('/governance/statistics');
  }

  /**
   * Make POST request.
   * @private
   */
  async _post(endpoint, payload) {
    const url = `${this.apiUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json'
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Make GET request.
   * @private
   */
  async _get(endpoint) {
    const url = `${this.apiUrl}${endpoint}`;
    const headers = {};

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers,
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

// TypeScript type definitions (as JSDoc)
/**
 * @typedef {Object} AuditResult
 * @property {boolean} approved - Whether the response was approved
 * @property {number} confidence - Confidence score (0-1)
 * @property {string[]} issues - List of detected issues
 * @property {string[]} warnings - List of warnings
 * @property {string} recommendation - Recommendation message
 * @property {boolean} shouldRegenerate - Whether response should be regenerated
 * @property {string|null} improvedResponse - Improved response if available
 */

/**
 * @typedef {Object} VerificationResult
 * @property {boolean} valid - Whether the claim is valid
 * @property {number} confidence - Confidence score (0-1)
 * @property {string[]} violations - List of violations found
 * @property {string} clinicalStatus - Clinical status (truly_working, incomplete, stubbed, etc.)
 */

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { BAISClient };
}

if (typeof window !== 'undefined') {
  window.BAISClient = BAISClient;
}

export { BAISClient };

