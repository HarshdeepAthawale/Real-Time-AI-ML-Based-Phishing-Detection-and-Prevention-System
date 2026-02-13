// Extension API configuration
let API_URL = 'http://localhost:3003/api/v1/extension';
let API_KEY = '';
let BLOCK_THREATS = true;
let BLOCK_SEVERITY = 'high'; // 'critical', 'high', 'medium', 'low'

// Temporary allowlist (URLs user chose to proceed to)
const allowlistedUrls = new Set();

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
  console.log('Phishing Detection Extension installed');
  loadSettings();
});

// Load settings from storage
function loadSettings() {
  chrome.storage.local.get(['apiUrl', 'apiKey', 'blockThreats', 'blockSeverity'], (result) => {
    if (result.apiUrl) API_URL = result.apiUrl;
    if (result.apiKey) API_KEY = result.apiKey;
    if (result.blockThreats !== undefined) BLOCK_THREATS = result.blockThreats;
    if (result.blockSeverity) BLOCK_SEVERITY = result.blockSeverity;
  });
}

// Listen for settings changes
chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === 'local') {
    if (changes.apiUrl) API_URL = changes.apiUrl.newValue;
    if (changes.apiKey) API_KEY = changes.apiKey.newValue;
    if (changes.blockThreats) BLOCK_THREATS = changes.blockThreats.newValue;
    if (changes.blockSeverity) BLOCK_SEVERITY = changes.blockSeverity.newValue;
  }
});

// Severity levels for comparison
const SEVERITY_LEVELS = { 'low': 0, 'medium': 1, 'high': 2, 'critical': 3 };

function shouldBlock(severity) {
  if (!BLOCK_THREATS) return false;
  const threatLevel = SEVERITY_LEVELS[severity] || 0;
  const thresholdLevel = SEVERITY_LEVELS[BLOCK_SEVERITY] || 2;
  return threatLevel >= thresholdLevel;
}

// Check URL when tab is updated
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Skip browser internal URLs
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') ||
        tab.url.startsWith('moz-extension://') || tab.url.startsWith('edge://') ||
        tab.url.startsWith('about:')) {
      return;
    }

    // Skip allowlisted URLs
    if (allowlistedUrls.has(tab.url)) {
      allowlistedUrls.delete(tab.url); // One-time allowlist
      return;
    }

    await checkURL(tab.url, tabId);
  }
});

// Check URL function
async function checkURL(url, tabId) {
  if (!API_KEY) {
    console.warn('API key not configured');
    return;
  }

  try {
    // Check local cache first
    const cacheKey = `url_cache_${btoa(url).substring(0, 50)}`;
    const cached = await chrome.storage.local.get([cacheKey]);

    if (cached[cacheKey] && cached[cacheKey].timestamp > Date.now() - 1800000) {
      const result = cached[cacheKey];
      updateBadge(result.isThreat, result.severity, tabId);

      // Block if threat detected and blocking enabled
      if (result.isThreat && shouldBlock(result.severity)) {
        redirectToBlockedPage(tabId, url, result);
      }
      return;
    }

    const response = await fetch(`${API_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': chrome.runtime.id
      },
      body: JSON.stringify({ url, privacyMode: false })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    // Cache result
    await chrome.storage.local.set({
      [cacheKey]: { ...result, timestamp: Date.now() }
    });

    // Update badge
    updateBadge(result.isThreat, result.severity, tabId);

    // Store threat info for popup
    if (result.isThreat) {
      await chrome.storage.local.set({ [`threat_${tabId}`]: result });

      // Block if threat detected and blocking enabled
      if (shouldBlock(result.severity)) {
        redirectToBlockedPage(tabId, url, result);
      }
    } else {
      await chrome.storage.local.remove([`threat_${tabId}`]);
    }

    console.log('URL check completed', {
      url: url.substring(0, 100),
      isThreat: result.isThreat,
      severity: result.severity
    });
  } catch (error) {
    console.error('URL check failed', error);
  }
}

// Redirect to blocked page with threat details
function redirectToBlockedPage(tabId, url, threatInfo) {
  const params = new URLSearchParams({
    url: url,
    type: threatInfo.threatType || 'phishing',
    severity: threatInfo.severity || 'high',
    confidence: String(threatInfo.confidence || 0),
    indicators: (threatInfo.indicators || []).join(',')
  });

  const blockedPageUrl = chrome.runtime.getURL(`blocked.html?${params.toString()}`);
  chrome.tabs.update(tabId, { url: blockedPageUrl });
}

// Update badge based on threat status
function updateBadge(isThreat, severity, tabId) {
  if (isThreat) {
    const colors = {
      'critical': '#cc0000',
      'high': '#ff4444',
      'medium': '#ff8800',
      'low': '#ffcc00'
    };

    chrome.action.setBadgeText({ text: '!', tabId });
    chrome.action.setBadgeBackgroundColor({
      color: colors[severity] || '#ff0000',
      tabId
    });
  } else {
    chrome.action.setBadgeText({ text: '', tabId });
  }
}

// Handle messages from content script and blocked page
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'checkForm') {
    checkForm(request.data, sender.tab.id).then(sendResponse);
    return true;
  }

  if (request.type === 'reportThreat') {
    reportThreat(request.data).then(sendResponse);
    return true;
  }

  if (request.type === 'getThreatInfo') {
    const tabId = sender.tab?.id;
    if (tabId) {
      chrome.storage.local.get([`threat_${tabId}`], (result) => {
        sendResponse(result[`threat_${tabId}`] || null);
      });
      return true;
    }
    sendResponse(null);
  }

  if (request.type === 'allowlistUrl') {
    allowlistedUrls.add(request.url);
    sendResponse({ success: true });
  }

  if (request.type === 'scanEmail') {
    scanEmail(request.data).then(sendResponse);
    return true;
  }
});

// Check form for phishing indicators
async function checkForm(formData, tabId) {
  if (!API_KEY) return { isSuspicious: false };

  try {
    const url = formData.action || '';
    if (!url) return { isSuspicious: false };

    const response = await fetch(`${API_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': chrome.runtime.id
      },
      body: JSON.stringify({ url, privacyMode: true })
    });

    if (!response.ok) return { isSuspicious: false };

    const result = await response.json();
    return { isSuspicious: result.isThreat || false, severity: result.severity };
  } catch (error) {
    console.error('Form check failed', error);
    return { isSuspicious: false };
  }
}

// Scan email content
async function scanEmail(emailData) {
  if (!API_KEY) return { success: false, error: 'API key not configured' };

  try {
    const response = await fetch(`${API_URL}/scan-email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': chrome.runtime.id
      },
      body: JSON.stringify({
        subject: emailData.subject,
        body: emailData.body,
        sender: emailData.sender,
        headers: emailData.headers
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Email scan failed', error);
    return { success: false, error: error.message };
  }
}

// Report threat
async function reportThreat(data) {
  if (!API_KEY) return { success: false, error: 'API key not configured' };

  try {
    const response = await fetch(`${API_URL}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': chrome.runtime.id
      },
      body: JSON.stringify({
        url: data.url,
        reason: data.reason || 'User reported',
        description: data.description || '',
        severity: data.severity || 'medium'
      })
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const result = await response.json();
    return { success: true, message: result.message };
  } catch (error) {
    console.error('Report threat failed', error);
    return { success: false, error: error.message };
  }
}

// Periodic cache cleanup
chrome.alarms.create('cacheCleanup', { periodInMinutes: 30 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'cacheCleanup') {
    cleanupCache();
  }
});

async function cleanupCache() {
  const allItems = await chrome.storage.local.get(null);
  const keysToRemove = [];
  const now = Date.now();

  for (const [key, value] of Object.entries(allItems)) {
    if (key.startsWith('url_cache_') && value.timestamp && value.timestamp < now - 3600000) {
      keysToRemove.push(key);
    }
  }

  if (keysToRemove.length > 0) {
    await chrome.storage.local.remove(keysToRemove);
    console.log(`Cleaned up ${keysToRemove.length} cached URL results`);
  }
}
