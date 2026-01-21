// Content script to detect phishing forms and inject warnings

(function() {
  'use strict';
  
  // Check for suspicious forms
  function checkForms() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
      const passwordFields = form.querySelectorAll('input[type="password"]');
      if (passwordFields.length > 0) {
        // Check if form is suspicious
        checkForm(form);
      }
    });
  }
  
  // Check individual form
  async function checkForm(form) {
    const formData = {
      action: form.action,
      method: form.method,
      fields: Array.from(form.querySelectorAll('input')).map(input => ({
        type: input.type,
        name: input.name
      }))
    };
    
    // Send to background script for analysis
    chrome.runtime.sendMessage({
      type: 'checkForm',
      data: formData
    }, (response) => {
      if (response && response.isSuspicious) {
        showWarning(form);
      }
    });
  }
  
  // Show warning banner
  function showWarning(form) {
    // Check if warning already exists
    if (form.parentNode.querySelector('.phishing-warning')) {
      return;
    }
    
    const warning = document.createElement('div');
    warning.className = 'phishing-warning';
    warning.style.cssText = `
      background: #ff0000;
      color: white;
      padding: 15px;
      margin: 10px 0;
      border-radius: 4px;
      font-family: Arial, sans-serif;
      font-size: 14px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      z-index: 10000;
      position: relative;
    `;
    
    warning.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px;">⚠️</span>
        <div style="flex: 1;">
          <strong>Warning: This form may be a phishing attempt.</strong>
          <div style="font-size: 12px; margin-top: 5px;">
            Proceed with caution. Do not enter sensitive information unless you are certain this is a legitimate website.
          </div>
        </div>
        <button id="phishing-warning-dismiss" style="
          background: white;
          color: #ff0000;
          border: none;
          padding: 5px 10px;
          border-radius: 3px;
          cursor: pointer;
          font-weight: bold;
        ">Dismiss</button>
      </div>
    `;
    
    form.parentNode.insertBefore(warning, form);
    
    // Dismiss button handler
    warning.querySelector('#phishing-warning-dismiss').addEventListener('click', () => {
      warning.remove();
    });
  }
  
  // Check for page-level threats
  function checkPageThreat() {
    chrome.runtime.sendMessage({
      type: 'getThreatInfo',
      url: window.location.href
    }, (response) => {
      if (response && response.isThreat) {
        showPageWarning(response);
      }
    });
  }
  
  // Show page-level warning
  function showPageWarning(threatInfo) {
    // Check if warning already exists
    if (document.querySelector('.phishing-page-warning')) {
      return;
    }
    
    const warning = document.createElement('div');
    warning.className = 'phishing-page-warning';
    warning.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: #ff0000;
      color: white;
      padding: 15px;
      text-align: center;
      font-family: Arial, sans-serif;
      font-size: 14px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      z-index: 999999;
    `;
    
    const confidence = threatInfo.confidence ? Math.round(threatInfo.confidence * 100) : 0;
    
    warning.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
        <span style="font-size: 24px;">⚠️</span>
        <div>
          <strong>PHISHING WARNING</strong>
          <div style="font-size: 12px; margin-top: 5px;">
            This page has been flagged as a potential phishing site (${confidence}% confidence).
            <button id="phishing-page-dismiss" style="
              background: white;
              color: #ff0000;
              border: none;
              padding: 5px 10px;
              border-radius: 3px;
              cursor: pointer;
              font-weight: bold;
              margin-left: 10px;
            ">Dismiss</button>
          </div>
        </div>
      </div>
    `;
    
    document.body.insertBefore(warning, document.body.firstChild);
    
    // Adjust body margin to account for warning
    document.body.style.paddingTop = '60px';
    
    // Dismiss button handler
    warning.querySelector('#phishing-page-dismiss').addEventListener('click', () => {
      warning.remove();
      document.body.style.paddingTop = '';
    });
  }
  
  // Run checks when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      checkForms();
      checkPageThreat();
    });
  } else {
    checkForms();
    checkPageThreat();
  }
  
  // Re-check forms when new forms are added dynamically
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) { // Element node
          if (node.tagName === 'FORM' || node.querySelector('form')) {
            checkForms();
          }
        }
      });
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
})();
