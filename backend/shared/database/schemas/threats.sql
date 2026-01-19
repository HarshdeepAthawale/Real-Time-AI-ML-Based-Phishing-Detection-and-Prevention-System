-- Threats table (master threat records)
CREATE TABLE threats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    threat_type VARCHAR(50) NOT NULL, -- email_phishing, url_spoofing, domain_hijacking, ai_generated
    severity VARCHAR(20) NOT NULL, -- critical, high, medium, low
    status VARCHAR(20) DEFAULT 'detected', -- detected, blocked, resolved, false_positive
    confidence_score DECIMAL(5,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
    source VARCHAR(50), -- email, url, domain, file
    source_value TEXT, -- The actual email/URL/domain that triggered detection
    title VARCHAR(500),
    description TEXT,
    metadata JSONB DEFAULT '{}', -- Additional context
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detections table (individual detection events)
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    detection_type VARCHAR(50) NOT NULL, -- nlp, url, visual, ioc, ensemble
    model_version VARCHAR(50),
    input_data JSONB NOT NULL, -- What was analyzed
    analysis_result JSONB NOT NULL, -- ML model output
    confidence_score DECIMAL(5,2) NOT NULL,
    processing_time_ms INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Threat Indicators table (IOCs associated with threats)
CREATE TABLE threat_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threat_id UUID REFERENCES threats(id) ON DELETE CASCADE,
    indicator_type VARCHAR(50) NOT NULL, -- url, domain, ip, email, hash, filename
    indicator_value TEXT NOT NULL,
    source VARCHAR(50), -- detection, threat_intel, user_report
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback on detections
CREATE TABLE detection_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    feedback_type VARCHAR(20) NOT NULL, -- true_positive, false_positive, false_negative
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_threats_organization ON threats(organization_id);
CREATE INDEX idx_threats_detected_at ON threats(detected_at DESC);
CREATE INDEX idx_threats_status ON threats(status);
CREATE INDEX idx_detections_threat ON detections(threat_id);
CREATE INDEX idx_detections_organization ON detections(organization_id);
CREATE INDEX idx_detections_detected_at ON detections(detected_at DESC);
CREATE INDEX idx_threat_indicators_threat ON threat_indicators(threat_id);
CREATE INDEX idx_threat_indicators_value ON threat_indicators(indicator_type, indicator_value);
CREATE INDEX idx_detection_feedback_detection ON detection_feedback(detection_id);
