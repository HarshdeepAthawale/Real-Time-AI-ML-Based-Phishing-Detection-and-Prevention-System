-- =========================================
-- Real-Time Phishing Detection System
-- Database Initialization Script
-- =========================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable JSONB functions
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- =========================================
-- Organizations Table
-- =========================================
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    api_key_hash VARCHAR(255),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_organizations_domain ON organizations(domain);
CREATE INDEX idx_organizations_active ON organizations(is_active);

-- =========================================
-- Users Table
-- =========================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_org_id ON users(organization_id);
CREATE INDEX idx_users_role ON users(role);

-- =========================================
-- API Keys Table
-- =========================================
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    permissions JSONB DEFAULT '[]',
    rate_limit INTEGER DEFAULT 1000,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_org_id ON api_keys(organization_id);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);

-- =========================================
-- Threats Table (Main Detection Results)
-- =========================================
CREATE TABLE IF NOT EXISTS threats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    threat_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'detected',
    confidence_score DECIMAL(5, 2) NOT NULL,
    source VARCHAR(50),
    source_value TEXT,
    title VARCHAR(500),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_threats_org_id ON threats(organization_id);
CREATE INDEX idx_threats_detected_at ON threats(detected_at DESC);
CREATE INDEX idx_threats_type ON threats(threat_type);
CREATE INDEX idx_threats_severity ON threats(severity);
CREATE INDEX idx_threats_status ON threats(status);
CREATE INDEX idx_threats_confidence ON threats(confidence_score DESC);
CREATE INDEX idx_threats_org_detected ON threats(organization_id, detected_at DESC);

-- GIN index for JSONB metadata searching
CREATE INDEX idx_threats_metadata ON threats USING GIN (metadata);

-- =========================================
-- Detections Table (ML Analysis Records)
-- =========================================
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_id UUID REFERENCES threats(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    detection_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    input_data JSONB NOT NULL,
    analysis_result JSONB,
    confidence_score DECIMAL(5, 2),
    processing_time_ms INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_detections_threat_id ON detections(threat_id);
CREATE INDEX idx_detections_org_id ON detections(organization_id);
CREATE INDEX idx_detections_type ON detections(detection_type);
CREATE INDEX idx_detections_detected_at ON detections(detected_at DESC);

-- =========================================
-- Threat Indicators (IOCs)
-- =========================================
CREATE TABLE IF NOT EXISTS threat_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    threat_id UUID REFERENCES threats(id) ON DELETE CASCADE,
    indicator_type VARCHAR(50) NOT NULL,
    indicator_value TEXT NOT NULL,
    confidence DECIMAL(5, 2),
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_indicators_threat_id ON threat_indicators(threat_id);
CREATE INDEX idx_indicators_type ON threat_indicators(indicator_type);
CREATE INDEX idx_indicators_value ON threat_indicators(indicator_value);
CREATE INDEX idx_indicators_first_seen ON threat_indicators(first_seen_at DESC);

-- =========================================
-- IOCs (Indicators of Compromise) from Threat Intel
-- =========================================
CREATE TABLE IF NOT EXISTS iocs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ioc_type VARCHAR(50) NOT NULL,
    ioc_value TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    threat_type VARCHAR(50),
    confidence DECIMAL(5, 2),
    severity VARCHAR(20),
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_iocs_type ON iocs(ioc_type);
CREATE INDEX idx_iocs_value ON iocs(ioc_value);
CREATE INDEX idx_iocs_source ON iocs(source);
CREATE INDEX idx_iocs_active ON iocs(is_active);
CREATE INDEX idx_iocs_type_value ON iocs(ioc_type, ioc_value);
CREATE INDEX idx_iocs_tags ON iocs USING GIN (tags);

-- Unique constraint to prevent duplicates
CREATE UNIQUE INDEX idx_iocs_unique ON iocs(ioc_type, ioc_value, source);

-- =========================================
-- IOC Matches (Detected IOCs in Analysis)
-- =========================================
CREATE TABLE IF NOT EXISTS ioc_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    ioc_id UUID REFERENCES iocs(id) ON DELETE CASCADE,
    matched_value TEXT NOT NULL,
    match_type VARCHAR(50),
    confidence DECIMAL(5, 2),
    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ioc_matches_detection_id ON ioc_matches(detection_id);
CREATE INDEX idx_ioc_matches_ioc_id ON ioc_matches(ioc_id);
CREATE INDEX idx_ioc_matches_matched_at ON ioc_matches(matched_at DESC);

-- =========================================
-- Threat Intelligence Feeds
-- =========================================
CREATE TABLE IF NOT EXISTS threat_intelligence_feeds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    provider VARCHAR(100) NOT NULL,
    feed_type VARCHAR(50),
    url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    sync_frequency_minutes INTEGER DEFAULT 60,
    last_sync_at TIMESTAMP,
    last_sync_status VARCHAR(50),
    last_sync_error TEXT,
    iocs_imported INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feeds_provider ON threat_intelligence_feeds(provider);
CREATE INDEX idx_feeds_active ON threat_intelligence_feeds(is_active);
CREATE INDEX idx_feeds_last_sync ON threat_intelligence_feeds(last_sync_at DESC);

-- =========================================
-- Sandbox Analyses
-- =========================================
CREATE TABLE IF NOT EXISTS sandbox_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    sandbox_provider VARCHAR(50) NOT NULL,
    submission_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    analysis_result JSONB,
    behavioral_indicators JSONB,
    network_connections JSONB,
    processes JSONB,
    risk_score DECIMAL(5, 2),
    malware_family VARCHAR(100),
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sandbox_detection_id ON sandbox_analyses(detection_id);
CREATE INDEX idx_sandbox_org_id ON sandbox_analyses(organization_id);
CREATE INDEX idx_sandbox_status ON sandbox_analyses(status);
CREATE INDEX idx_sandbox_submitted_at ON sandbox_analyses(submitted_at DESC);
CREATE INDEX idx_sandbox_risk_score ON sandbox_analyses(risk_score DESC);

-- =========================================
-- Detection Feedback (for model improvement)
-- =========================================
CREATE TABLE IF NOT EXISTS detection_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL,
    is_correct BOOLEAN,
    actual_threat_type VARCHAR(50),
    comments TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_detection_id ON detection_feedback(detection_id);
CREATE INDEX idx_feedback_org_id ON detection_feedback(organization_id);
CREATE INDEX idx_feedback_type ON detection_feedback(feedback_type);
CREATE INDEX idx_feedback_created_at ON detection_feedback(created_at DESC);

-- =========================================
-- Training Jobs (ML Model Training)
-- =========================================
CREATE TABLE IF NOT EXISTS training_jobs (
    id VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    training_config JSONB DEFAULT '{}',
    dataset_path_s3 TEXT,
    model_path_s3 TEXT,
    metrics JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_model_type ON training_jobs(model_type);
CREATE INDEX idx_training_jobs_started_at ON training_jobs(started_at DESC);

-- =========================================
-- Dashboard Views for Performance
-- =========================================

-- Recent Threats View
CREATE OR REPLACE VIEW recent_threats AS
SELECT 
    t.id,
    t.organization_id,
    t.threat_type,
    t.severity,
    t.confidence_score,
    t.source,
    t.source_value,
    t.detected_at,
    COUNT(d.id) as detection_count,
    JSONB_AGG(DISTINCT d.detection_type) as detection_types
FROM threats t
LEFT JOIN detections d ON t.id = d.threat_id
WHERE t.detected_at > NOW() - INTERVAL '30 days'
GROUP BY t.id
ORDER BY t.detected_at DESC;

-- Threat Statistics View
CREATE OR REPLACE VIEW threat_statistics AS
SELECT 
    organization_id,
    DATE(detected_at) as date,
    threat_type,
    severity,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence
FROM threats
WHERE detected_at > NOW() - INTERVAL '90 days'
GROUP BY organization_id, DATE(detected_at), threat_type, severity
ORDER BY date DESC;

-- IOC Statistics View
CREATE OR REPLACE VIEW ioc_statistics AS
SELECT 
    source,
    ioc_type,
    COUNT(*) as total_iocs,
    COUNT(CASE WHEN is_active THEN 1 END) as active_iocs,
    MAX(last_seen_at) as last_updated
FROM iocs
GROUP BY source, ioc_type
ORDER BY total_iocs DESC;

-- =========================================
-- Functions for Maintenance
-- =========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_threats_updated_at BEFORE UPDATE ON threats FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_iocs_updated_at BEFORE UPDATE ON iocs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_feeds_updated_at BEFORE UPDATE ON threat_intelligence_feeds FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sandbox_updated_at BEFORE UPDATE ON sandbox_analyses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON training_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =========================================
-- Partitioning for threats table (by month)
-- =========================================
-- Note: Uncomment this section if you need partitioning for large scale

-- CREATE TABLE threats_2024_01 PARTITION OF threats FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- CREATE TABLE threats_2024_02 PARTITION OF threats FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- Add more partitions as needed

-- =========================================
-- Initial Threat Intelligence Feed Configuration
-- =========================================
-- These feed entries are created as INACTIVE by default.
-- Enable and configure them via environment variables and API.
-- No real data is created - just the feed registry structure.

INSERT INTO threat_intelligence_feeds (name, provider, feed_type, is_active) VALUES
    ('MISP Feed', 'MISP', 'events', false),
    ('AlienVault OTX', 'OTX', 'pulses', false),
    ('PhishTank', 'PhishTank', 'urls', false),
    ('URLhaus', 'URLhaus', 'urls', false),
    ('VirusTotal', 'VirusTotal', 'api', false)
ON CONFLICT DO NOTHING;

-- =========================================
-- Grant Permissions
-- =========================================

-- Grant all privileges to postgres user (already the owner)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- =========================================
-- Completion Message
-- =========================================

DO $$
BEGIN
    RAISE NOTICE '✓ Database schema initialized successfully';
    RAISE NOTICE '✓ Tables created: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public');
    RAISE NOTICE '✓ Indexes created for performance optimization';
    RAISE NOTICE '✓ Threat intelligence feed registry initialized (inactive by default)';
    RAISE NOTICE '→ Next: Create your organization, users, and API keys via the API or migrations';
END $$;
