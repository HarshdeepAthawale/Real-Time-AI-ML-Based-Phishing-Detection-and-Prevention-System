-- Email messages (metadata only, full content in MongoDB)
CREATE TABLE email_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    message_id VARCHAR(500) UNIQUE, -- Email Message-ID header
    from_email VARCHAR(255),
    to_emails TEXT[], -- Array of recipient emails
    subject TEXT,
    received_at TIMESTAMP,
    analyzed_at TIMESTAMP,
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Email headers (parsed headers)
CREATE TABLE email_headers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email_message_id UUID REFERENCES email_messages(id) ON DELETE CASCADE,
    header_name VARCHAR(255) NOT NULL,
    header_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_email_messages_org ON email_messages(organization_id);
CREATE INDEX idx_email_messages_message_id ON email_messages(message_id);
CREATE INDEX idx_email_messages_received_at ON email_messages(received_at DESC);
CREATE INDEX idx_email_headers_email ON email_headers(email_message_id);
