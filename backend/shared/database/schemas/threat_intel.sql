-- Threat intelligence feeds
CREATE TABLE threat_intelligence_feeds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    feed_type VARCHAR(50) NOT NULL, -- misp, otx, custom, user_submitted
    api_endpoint TEXT,
    api_key_encrypted TEXT,
    sync_interval_minutes INTEGER DEFAULT 60,
    last_sync_at TIMESTAMP,
    last_sync_status VARCHAR(20), -- success, failed, partial
    last_sync_error TEXT,
    is_active BOOLEAN DEFAULT true,
    reliability_score DECIMAL(5,2) DEFAULT 50,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IOCs (Indicators of Compromise)
CREATE TABLE iocs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feed_id UUID REFERENCES threat_intelligence_feeds(id) ON DELETE SET NULL,
    ioc_type VARCHAR(50) NOT NULL, -- url, domain, ip, email, hash_md5, hash_sha1, hash_sha256, filename
    ioc_value TEXT NOT NULL,
    ioc_value_hash VARCHAR(64), -- For fast lookups
    threat_type VARCHAR(100),
    severity VARCHAR(20),
    confidence DECIMAL(5,2),
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    source_reports INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ioc_type, ioc_value_hash)
);

-- IOC matches (when our system detects an IOC)
CREATE TABLE ioc_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ioc_id UUID REFERENCES iocs(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_iocs_type_value ON iocs(ioc_type, ioc_value_hash);
CREATE INDEX idx_iocs_feed ON iocs(feed_id);
CREATE INDEX idx_iocs_severity ON iocs(severity);
CREATE INDEX idx_ioc_matches_ioc ON ioc_matches(ioc_id);
CREATE INDEX idx_ioc_matches_detection ON ioc_matches(detection_id);
