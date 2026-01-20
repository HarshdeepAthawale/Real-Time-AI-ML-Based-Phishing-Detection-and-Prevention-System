from playwright.async_api import async_playwright, Page, Browser
from typing import Dict, Optional
import asyncio
import base64
from io import BytesIO

class PageRenderer:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
    
    async def initialize(self):
        """Initialize browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
    
    async def render_page(self, url: str, wait_time: int = 3000) -> Dict:
        """Render page and capture screenshot"""
        if not self.browser:
            await self.initialize()
        
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        
        try:
            # Navigate to URL
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for page to stabilize
            await asyncio.sleep(wait_time / 1000)
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            
            # Get DOM
            dom_content = await page.content()
            
            # Get page metrics
            metrics = await page.evaluate("""
                () => {
                    return {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        scrollHeight: document.documentElement.scrollHeight,
                        elementCount: document.querySelectorAll('*').length
                    }
                }
            """)
            
            return {
                "screenshot": base64.b64encode(screenshot_bytes).decode(),
                "dom": dom_content,
                "status_code": response.status if response else None,
                "metrics": metrics,
                "url": url
            }
        except Exception as e:
            return {
                "error": str(e),
                "url": url
            }
        finally:
            await page.close()
            await context.close()
    
    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
