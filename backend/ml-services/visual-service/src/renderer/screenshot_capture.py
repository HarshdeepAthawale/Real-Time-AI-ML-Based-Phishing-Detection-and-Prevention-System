from playwright.async_api import Page
from typing import Dict, Optional
import base64
import asyncio

class ScreenshotCapture:
    """Utility class for capturing screenshots with various options"""
    
    @staticmethod
    async def capture_full_page(page: Page) -> bytes:
        """Capture full page screenshot"""
        return await page.screenshot(full_page=True)
    
    @staticmethod
    async def capture_viewport(page: Page) -> bytes:
        """Capture viewport screenshot"""
        return await page.screenshot(full_page=False)
    
    @staticmethod
    async def capture_element(page: Page, selector: str) -> Optional[bytes]:
        """Capture screenshot of specific element"""
        try:
            element = await page.query_selector(selector)
            if element:
                return await element.screenshot()
            return None
        except Exception:
            return None
    
    @staticmethod
    async def capture_with_options(
        page: Page,
        full_page: bool = True,
        quality: int = 90,
        clip: Optional[Dict] = None
    ) -> bytes:
        """Capture screenshot with custom options"""
        options = {
            "full_page": full_page,
            "quality": quality
        }
        if clip:
            options["clip"] = clip
        
        return await page.screenshot(**options)
    
    @staticmethod
    def encode_to_base64(screenshot_bytes: bytes) -> str:
        """Encode screenshot bytes to base64 string"""
        return base64.b64encode(screenshot_bytes).decode()
