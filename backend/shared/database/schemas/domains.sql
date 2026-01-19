-- Domains table
CREATE TABLE domains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(255) UNIQUE NOT NULL,
    tld VARCHAR(50),
    subdomain VARCHAR(255),
    registered_domain VARCHAR(255), -- Base domain without subdomain
    reputation_score DECIMAL(5,2) DEFAULT 50 CHECK (reputation_score >= 0 AND reputation_score <= 100),
    is_malicious BOOLEAN DEFAULT false,
    is_suspicious BOOLEAN DEFAULT false,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed_at TIMESTAMP,
    whois_data JSONB,
    dns_records JSONB,
    ssl_certificate_data JSONB,
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- URLs table
CREATE TABLE urls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    full_url TEXT NOT NULL,
    url_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 hash
    scheme VARCHAR(10), -- http, https
    path TEXT,
    query_params JSONB,
    fragment TEXT,
    redirect_chain JSONB, -- Array of redirect URLs
    redirect_count INTEGER DEFAULT 0,
    is_malicious BOOLEAN DEFAULT false,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed_at TIMESTAMP,
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Domain relationships (for graph analysis)
CREATE TABLE domain_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    target_domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- redirects_to, shares_ip, shares_registrar, similar_name
    strength DECIMAL(5,2) DEFAULT 1.0, -- Relationship strength score
    metadata JSONB DEFAULT '{}',
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_domain_id, target_domain_id, relationship_type)
);

CREATE INDEX idx_domains_domain ON domains(domain);
CREATE INDEX idx_domains_registered_domain ON domains(registered_domain);
CREATE INDEX idx_domains_reputation ON domains(reputation_score);
CREATE INDEX idx_urls_domain ON urls(domain_id);
CREATE INDEX idx_urls_hash ON urls(url_hash);
CREATE INDEX idx_urls_malicious ON urls(is_malicious);
CREATE INDEX idx_domain_relationships_source ON domain_relationships(source_domain_id);
CREATE INDEX idx_domain_relationships_target ON domain_relationships(target_domain_id);
