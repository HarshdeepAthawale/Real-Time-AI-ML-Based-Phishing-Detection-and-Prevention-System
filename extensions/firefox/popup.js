// Popup script for Firefox

document.addEventListener('DOMContentLoaded', async () => {
  // Get current tab
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  
  if (!tab || !tab.url) {
    document.getElementById('status').textContent = 'Unable to get page URL';
    return;
  }
  
  // Skip about:// and extension:// URLs
  if (tab.url.startsWith('about:') || tab.url.startsWith('moz-extension://') ||
      tab.url.startsWith('chrome://') || tab.url.startsWith('edge://')) {
    document.getElementById('status').textContent = 'Not applicable to this page';
    return;
  }
  
  // Check for threat info
  const threatKey = `threat_${tab.id}`;
  const result = await browser.storage.local.get([threatKey]);
  
  if (result[threatKey]) {
    const threat = result[threatKey];
    showThreat(threat);
  } else {
    showSafe();
  }
  
  // Settings link
  document.getElementById('settingsLink').addEventListener('click', (e) => {
    e.preventDefault();
    browser.runtime.openOptionsPage();
  });
  
  // Report button
  document.getElementById('reportButton').addEventListener('click', async () => {
    const response = await browser.runtime.sendMessage({
      type: 'reportThreat',
      data: {
        url: tab.url,
        reason: 'User reported via extension',
        severity: 'high'
      }
    });
    
    if (response && response.success) {
      alert('Threat reported successfully. Thank you!');
      document.getElementById('reportButton').disabled = true;
      document.getElementById('reportButton').textContent = 'Reported';
    } else {
      alert('Failed to report threat: ' + (response.error || 'Unknown error'));
    }
  });
});

function showThreat(threat) {
  const statusEl = document.getElementById('status');
  statusEl.className = 'status threat';
  statusEl.textContent = '⚠️ Threat Detected';
  
  const threatInfo = document.getElementById('threatInfo');
  threatInfo.style.display = 'block';
  
  document.getElementById('threatMessage').textContent = 
    threat.warningMessage || 'This page may be a phishing attempt.';
  
  const confidence = threat.confidence ? Math.round(threat.confidence * 100) : 0;
  document.getElementById('threatConfidence').textContent = 
    `Confidence: ${confidence}%`;
}

function showSafe() {
  const statusEl = document.getElementById('status');
  statusEl.className = 'status safe';
  statusEl.textContent = '✓ Page appears safe';
  
  document.getElementById('threatInfo').style.display = 'none';
}
