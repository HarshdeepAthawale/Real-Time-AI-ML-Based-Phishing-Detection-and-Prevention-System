import email
from email.header import decode_header
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EmailParser:
    def parse(self, raw_email: str) -> Dict:
        """Parse email content"""
        try:
            msg = email.message_from_string(raw_email)
        except Exception as e:
            logger.error(f"Failed to parse email: {e}")
            return {
                "subject": "",
                "from": "",
                "to": "",
                "body_text": "",
                "body_html": None,
                "headers": {},
                "error": str(e)
            }
        
        # Decode headers
        subject = self._decode_header(msg.get('Subject', ''))
        from_addr = self._decode_header(msg.get('From', ''))
        to_addrs = self._decode_header(msg.get('To', ''))
        
        # Extract body
        body_text, body_html = self._extract_body(msg)
        
        # Extract headers
        headers = {}
        for key, value in msg.items():
            headers[key] = self._decode_header(value)
        
        return {
            "subject": subject,
            "from": from_addr,
            "to": to_addrs,
            "body_text": body_text,
            "body_html": body_html,
            "headers": headers
        }
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if not header:
            return ""
        
        try:
            decoded_parts = decode_header(header)
            decoded_string = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    try:
                        decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
                    except (UnicodeDecodeError, LookupError):
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += part
            return decoded_string
        except Exception as e:
            logger.warning(f"Failed to decode header: {e}")
            return str(header)
    
    def _extract_body(self, msg) -> tuple[str, Optional[str]]:
        """Extract text and HTML body"""
        body_text = ""
        body_html = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = payload.decode('utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"Failed to decode text/plain body: {e}")
                elif content_type == "text/html":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_html = payload.decode('utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"Failed to decode text/html body: {e}")
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    if content_type == "text/plain":
                        body_text = payload.decode('utf-8', errors='ignore')
                    elif content_type == "text/html":
                        body_html = payload.decode('utf-8', errors='ignore')
                except Exception as e:
                    logger.warning(f"Failed to decode body: {e}")
        
        return body_text, body_html
    
    def extract_links_from_html(self, html_content: str) -> List[str]:
        """Extract links from HTML content"""
        import re
        if not html_content:
            return []
        
        # Simple regex to find href attributes
        link_pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(link_pattern, html_content, re.IGNORECASE)
        return links
