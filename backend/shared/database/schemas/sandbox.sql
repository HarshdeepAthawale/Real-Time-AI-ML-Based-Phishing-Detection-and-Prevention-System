-- Sandbox analysis jobs
CREATE TABLE sandbox_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL, -- url, file
    target_url TEXT,
    target_file_hash VARCHAR(64),
    sandbox_provider VARCHAR(50), -- cuckoo, anyrun, custom
    sandbox_job_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_data JSONB,
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sandbox_analyses_org ON sandbox_analyses(organization_id);
CREATE INDEX idx_sandbox_analyses_status ON sandbox_analyses(status);
CREATE INDEX idx_sandbox_analyses_submitted_at ON sandbox_analyses(submitted_at DESC);
