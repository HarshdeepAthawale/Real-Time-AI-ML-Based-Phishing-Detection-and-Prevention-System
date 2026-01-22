"""Playwright-based page renderer"""
import asyncio
import base64
from typing import Dict, Optional
from playwright.async_api import async_playwright, Browser, Page
from src.config import settings
from src.utils.logger import logger


class PageRenderer:
    """Render pages using Playwright headless browser"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.timeout = settings.browser_timeout
    
    async def initialize(self):
        """Initialize browser"""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            logger.info("Playwright browser initialized")
    
    async def render_page(self, url: str, wait_time: int = 3000) -> Dict:
        """
        Render page and capture screenshot
        
        Args:
            url: URL to render
            wait_time: Time to wait for page to load (ms)
            
        Returns:
            Dictionary with screenshot and DOM data
        """
        if not self.browser:
            await self.initialize()
        
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        
        try:
            # Navigate to URL
            response = await page.goto(url, wait_until='networkidle', timeout=self.timeout)
            
            # Wait for page to stabilize
            await asyncio.sleep(wait_time / 1000)
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot(full_page=True, quality=settings.screenshot_quality)
            
            # Get DOM content
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
                "screenshot_bytes": screenshot_bytes,
                "dom": dom_content,
                "status_code": response.status if response else None,
                "metrics": metrics,
                "url": url,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error rendering page {url}: {e}")
            return {
                "error": str(e),
                "url": url,
                "success": False
            }
        
        finally:
            await page.close()
            await context.close()
    
    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")
        if self.playwright:
            await self.playwright.stop()
            logger.info("Playwright stopped")


# Global instance
page_renderer = PageRenderer()
