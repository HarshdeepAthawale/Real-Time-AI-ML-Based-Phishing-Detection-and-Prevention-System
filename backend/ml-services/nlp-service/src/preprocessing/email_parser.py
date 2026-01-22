"""Email parsing utilities"""
import email
from email.header import decode_header
from typing import Dict, Optional, Tuple


class EmailParser:
    """Parse email content and headers"""
    
    def parse(self, raw_email: str) -> Dict:
        """
        Parse raw email content
        
        Args:
            raw_email: Raw email string
            
        Returns:
            Dictionary with parsed email components
        """
        try:
            msg = email.message_from_string(raw_email)
            
            # Decode headers
            subject = self._decode_header(msg.get('Subject', ''))
            from_addr = self._decode_header(msg.get('From', ''))
            to_addrs = self._decode_header(msg.get('To', ''))
            reply_to = self._decode_header(msg.get('Reply-To', ''))
            
            # Extract body
            body_text, body_html = self._extract_body(msg)
            
            # Extract all headers
            headers = {}
            for key, value in msg.items():
                headers[key] = self._decode_header(value)
            
            return {
                "subject": subject,
                "from": from_addr,
                "to": to_addrs,
                "reply_to": reply_to,
                "body_text": body_text,
                "body_html": body_html,
                "headers": headers,
                "has_attachments": self._has_attachments(msg)
            }
        
        except Exception as e:
            return {
                "subject": "",
                "from": "",
                "to": "",
                "reply_to": "",
                "body_text": raw_email,  # Fallback to raw content
                "body_html": None,
                "headers": {},
                "has_attachments": False,
                "parse_error": str(e)
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
                    decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
                else:
                    decoded_string += str(part)
            
            return decoded_string
        
        except Exception:
            return str(header)
    
    def _extract_body(self, msg) -> Tuple[str, Optional[str]]:
        """Extract text and HTML body"""
        body_text = ""
        body_html = None
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = payload.decode('utf-8', errors='ignore')
                    
                    elif content_type == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_html = payload.decode('utf-8', errors='ignore')
            
            else:
                content_type = msg.get_content_type()
                payload = msg.get_payload(decode=True)
                
                if payload:
                    if content_type == "text/plain":
                        body_text = payload.decode('utf-8', errors='ignore')
                    elif content_type == "text/html":
                        body_html = payload.decode('utf-8', errors='ignore')
        
        except Exception:
            pass
        
        return body_text, body_html
    
    def _has_attachments(self, msg) -> bool:
        """Check if email has attachments"""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition'):
                        return True
            return False
        except Exception:
            return False
