"""
Tests for page renderer components
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import base64
from PIL import Image
import io

from src.renderer.page_renderer import PageRenderer
from src.renderer.screenshot_capture import ScreenshotCapture

@pytest.mark.asyncio
async def test_page_renderer_initialization():
    """Test PageRenderer initialization"""
    renderer = PageRenderer()
    
    assert renderer is not None
    assert renderer.browser is None
    assert renderer.playwright is None

@pytest.mark.asyncio
@patch('src.renderer.page_renderer.async_playwright')
async def test_page_renderer_initialize(mock_playwright):
    """Test PageRenderer initialization with mocked Playwright"""
    # Mock Playwright
    mock_playwright_instance = AsyncMock()
    mock_browser = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
    
    renderer = PageRenderer()
    await renderer.initialize()
    
    assert renderer.playwright is not None
    assert renderer.browser is not None

@pytest.mark.asyncio
@patch('src.renderer.page_renderer.async_playwright')
async def test_page_renderer_render_page(mock_playwright):
    """Test page rendering with mocked Playwright"""
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    screenshot_bytes = img_bytes.getvalue()
    
    # Mock Playwright components
    mock_playwright_instance = AsyncMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    
    # Setup mocks
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_context.new_page = AsyncMock(return_value=mock_page)
    
    # Mock page methods
    mock_response = Mock()
    mock_response.status = 200
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.screenshot = AsyncMock(return_value=screenshot_bytes)
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    mock_page.evaluate = AsyncMock(return_value={
        "width": 1920,
        "height": 1080,
        "scrollHeight": 1080,
        "elementCount": 10
    })
    mock_page.close = AsyncMock()
    mock_context.close = AsyncMock()
    
    renderer = PageRenderer()
    renderer.playwright = mock_playwright_instance
    renderer.browser = mock_browser
    
    result = await renderer.render_page("https://example.com", wait_time=1000)
    
    assert "screenshot" in result
    assert "dom" in result
    assert "status_code" in result
    assert "metrics" in result
    assert "url" in result
    
    assert result["url"] == "https://example.com"
    assert result["status_code"] == 200
    assert isinstance(result["screenshot"], str)  # Base64 encoded
    assert result["dom"] == "<html><body>Test</body></html>"
    
    # Verify mocks were called
    mock_page.goto.assert_called_once()
    mock_page.screenshot.assert_called_once()
    mock_page.content.assert_called_once()
    mock_page.evaluate.assert_called_once()

@pytest.mark.asyncio
@patch('src.renderer.page_renderer.async_playwright')
async def test_page_renderer_render_page_error(mock_playwright):
    """Test page rendering with error"""
    # Mock Playwright components
    mock_playwright_instance = AsyncMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_context.new_page = AsyncMock(return_value=mock_page)
    
    # Mock error
    mock_page.goto = AsyncMock(side_effect=Exception("Network error"))
    mock_page.close = AsyncMock()
    mock_context.close = AsyncMock()
    
    renderer = PageRenderer()
    renderer.playwright = mock_playwright_instance
    renderer.browser = mock_browser
    
    result = await renderer.render_page("https://invalid-url.com")
    
    assert "error" in result
    assert "url" in result
    assert result["url"] == "https://invalid-url.com"

@pytest.mark.asyncio
@patch('src.renderer.page_renderer.async_playwright')
async def test_page_renderer_auto_initialize(mock_playwright):
    """Test that render_page auto-initializes if browser not initialized"""
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    screenshot_bytes = img_bytes.getvalue()
    
    # Mock Playwright components
    mock_playwright_instance = AsyncMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_context.new_page = AsyncMock(return_value=mock_page)
    
    mock_response = Mock()
    mock_response.status = 200
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.screenshot = AsyncMock(return_value=screenshot_bytes)
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.evaluate = AsyncMock(return_value={})
    mock_page.close = AsyncMock()
    mock_context.close = AsyncMock()
    
    renderer = PageRenderer()
    # Don't initialize, let render_page do it
    
    result = await renderer.render_page("https://example.com")
    
    # Should have initialized
    assert renderer.browser is not None
    assert "screenshot" in result

@pytest.mark.asyncio
async def test_page_renderer_close():
    """Test PageRenderer close method"""
    renderer = PageRenderer()
    
    # Mock browser and playwright
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_playwright.stop = AsyncMock()
    
    renderer.browser = mock_browser
    renderer.playwright = mock_playwright
    
    await renderer.close()
    
    mock_browser.close.assert_called_once()
    mock_playwright.stop.assert_called_once()

@pytest.mark.asyncio
async def test_page_renderer_close_no_browser():
    """Test PageRenderer close when browser not initialized"""
    renderer = PageRenderer()
    
    # Should not raise error
    await renderer.close()

def test_screenshot_capture_encode_to_base64():
    """Test screenshot encoding to base64"""
    test_bytes = b"fake_screenshot_data"
    encoded = ScreenshotCapture.encode_to_base64(test_bytes)
    
    assert isinstance(encoded, str)
    # Verify it's valid base64
    decoded = base64.b64decode(encoded)
    assert decoded == test_bytes

@pytest.mark.asyncio
async def test_screenshot_capture_full_page():
    """Test full page screenshot capture"""
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")
    
    result = await ScreenshotCapture.capture_full_page(mock_page)
    
    assert result == b"screenshot_data"
    mock_page.screenshot.assert_called_once_with(full_page=True)

@pytest.mark.asyncio
async def test_screenshot_capture_viewport():
    """Test viewport screenshot capture"""
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")
    
    result = await ScreenshotCapture.capture_viewport(mock_page)
    
    assert result == b"screenshot_data"
    mock_page.screenshot.assert_called_once_with(full_page=False)

@pytest.mark.asyncio
async def test_screenshot_capture_element():
    """Test element screenshot capture"""
    mock_page = AsyncMock()
    mock_element = AsyncMock()
    mock_element.screenshot = AsyncMock(return_value=b"element_screenshot")
    mock_page.query_selector = AsyncMock(return_value=mock_element)
    
    result = await ScreenshotCapture.capture_element(mock_page, "#test-element")
    
    assert result == b"element_screenshot"
    mock_page.query_selector.assert_called_once_with("#test-element")

@pytest.mark.asyncio
async def test_screenshot_capture_element_not_found():
    """Test element screenshot when element not found"""
    mock_page = AsyncMock()
    mock_page.query_selector = AsyncMock(return_value=None)
    
    result = await ScreenshotCapture.capture_element(mock_page, "#nonexistent")
    
    assert result is None

@pytest.mark.asyncio
async def test_screenshot_capture_with_options():
    """Test screenshot capture with custom options"""
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")
    
    result = await ScreenshotCapture.capture_with_options(
        mock_page,
        full_page=True,
        quality=90,
        clip={"x": 0, "y": 0, "width": 100, "height": 100}
    )
    
    assert result == b"screenshot_data"
    mock_page.screenshot.assert_called_once()
    call_kwargs = mock_page.screenshot.call_args[1]
    assert call_kwargs["full_page"] is True
    assert call_kwargs["quality"] == 90
    assert "clip" in call_kwargs
