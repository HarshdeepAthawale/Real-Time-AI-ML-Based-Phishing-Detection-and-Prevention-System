// Extension API configuration
let API_URL = ''; // Will be set from options
let API_KEY = ''; // Will be set from options

// Initialize extension
browser.runtime.onInstalled.addListener(() => {
  console.log('Phishing Detection Extension installed');
  
  // Load settings
  browser.storage.local.get(['apiUrl', 'apiKey']).then((result) => {
    if (result.apiUrl) {
      API_URL = result.apiUrl;
    }
    if (result.apiKey) {
      API_KEY = result.apiKey;
    }
  });
});

// Listen for settings changes
browser.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === 'local') {
    if (changes.apiUrl) {
      API_URL = changes.apiUrl.newValue;
    }
    if (changes.apiKey) {
      API_KEY = changes.apiKey.newValue;
    }
  }
});

// Check URL when tab is updated
browser.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Skip moz:// and extension:// URLs
    if (tab.url.startsWith('about:') || tab.url.startsWith('moz-extension://') ||
        tab.url.startsWith('chrome://') || tab.url.startsWith('edge://')) {
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
    const cached = await browser.storage.local.get([cacheKey]);
    
    if (cached[cacheKey] && cached[cacheKey].timestamp > Date.now() - 1800000) {
      // Use cached result (30 minutes)
      const result = cached[cacheKey];
      updateBadge(result.isThreat, tabId);
      return;
    }
    
    const response = await fetch(`${API_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': browser.runtime.id
      },
      body: JSON.stringify({
        url: url,
        privacyMode: false
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Cache result
    await browser.storage.local.set({
      [cacheKey]: {
        ...result,
        timestamp: Date.now()
      }
    });
    
    // Update badge
    updateBadge(result.isThreat, tabId);
    
    // Store threat info for popup
    if (result.isThreat) {
      await browser.storage.local.set({
        [`threat_${tabId}`]: result
      });
    } else {
      await browser.storage.local.remove([`threat_${tabId}`]);
    }
    
    console.log('URL check completed', { url: url.substring(0, 100), isThreat: result.isThreat });
  } catch (error) {
    console.error('URL check failed', error);
  }
}

// Update badge based on threat status
function updateBadge(isThreat, tabId) {
  if (isThreat) {
    browser.browserAction.setBadgeText({
      text: '!',
      tabId: tabId
    });
    browser.browserAction.setBadgeBackgroundColor({
      color: '#ff0000',
      tabId: tabId
    });
  } else {
    browser.browserAction.setBadgeText({
      text: '',
      tabId: tabId
    });
  }
}

// Handle messages from content script
browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'checkForm') {
    checkForm(request.data, sender.tab.id).then(sendResponse);
    return true; // Keep channel open for async response
  }
  
  if (request.type === 'reportThreat') {
    reportThreat(request.data).then(sendResponse);
    return true;
  }
});

// Check form for phishing indicators
async function checkForm(formData, tabId) {
  if (!API_KEY) {
    return { isSuspicious: false };
  }
  
  try {
    const url = formData.action || '';
    if (!url) {
      return { isSuspicious: false };
    }
    
    const response = await fetch(`${API_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': browser.runtime.id
      },
      body: JSON.stringify({
        url: url,
        privacyMode: true
      })
    });
    
    if (!response.ok) {
      return { isSuspicious: false };
    }
    
    const result = await response.json();
    return { isSuspicious: result.isThreat || false };
  } catch (error) {
    console.error('Form check failed', error);
    return { isSuspicious: false };
  }
}

// Report threat
async function reportThreat(data) {
  if (!API_KEY) {
    return { success: false, error: 'API key not configured' };
  }
  
  try {
    const response = await fetch(`${API_URL}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
        'X-Extension-Id': browser.runtime.id
      },
      body: JSON.stringify({
        url: data.url,
        reason: data.reason || 'User reported',
        description: data.description || '',
        severity: data.severity || 'medium'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    return { success: true, message: result.message };
  } catch (error) {
    console.error('Report threat failed', error);
    return { success: false, error: error.message };
  }
}
