// Options page script for Firefox

document.addEventListener('DOMContentLoaded', async () => {
  // Load saved settings
  const result = await browser.storage.local.get(['apiUrl', 'apiKey', 'blockThreats']);
  
  if (result.apiUrl) {
    document.getElementById('apiUrl').value = result.apiUrl;
  }
  if (result.apiKey) {
    document.getElementById('apiKey').value = result.apiKey;
  }
  if (result.blockThreats) {
    document.getElementById('blockThreats').checked = result.blockThreats;
  }
  
  // Form submission
  document.getElementById('settingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const apiUrl = document.getElementById('apiUrl').value.trim();
    const apiKey = document.getElementById('apiKey').value.trim();
    const blockThreats = document.getElementById('blockThreats').checked;
    
    // Validate URL
    try {
      new URL(apiUrl);
    } catch (error) {
      showStatus('Invalid API URL format', 'error');
      return;
    }
    
    // Save settings
    await browser.storage.local.set({
      apiUrl: apiUrl,
      apiKey: apiKey,
      blockThreats: blockThreats
    });
    
    showStatus('Settings saved successfully!', 'success');
    
    // Test API connection
    try {
      const response = await fetch(`${apiUrl}/health`, {
        method: 'GET'
      });
      
      if (response.ok) {
        showStatus('Settings saved and API connection successful!', 'success');
      } else {
        showStatus('Settings saved, but API connection failed. Please check your configuration.', 'error');
      }
    } catch (error) {
      showStatus('Settings saved, but unable to connect to API. Please check your configuration.', 'error');
    }
  });
});

function showStatus(message, type) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = message;
  statusEl.className = `status ${type}`;
  
  setTimeout(() => {
    statusEl.className = 'status';
    statusEl.textContent = '';
  }, 5000);
}
