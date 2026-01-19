-- ML Models table
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL, -- nlp, url_gnn, visual_cnn, adversarial
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    model_path_s3 TEXT, -- S3 path to model file
    model_size_bytes BIGINT,
    framework VARCHAR(50), -- pytorch, tensorflow, onnx
    input_schema JSONB, -- Expected input format
    output_schema JSONB, -- Expected output format
    metrics JSONB DEFAULT '{}', -- accuracy, precision, recall, f1
    training_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT false,
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_type, version)
);

-- Model versions (tracking all versions)
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    model_path_s3 TEXT,
    metrics JSONB DEFAULT '{}',
    training_job_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training jobs
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    training_config JSONB NOT NULL,
    dataset_path_s3 TEXT,
    dataset_size INTEGER,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metrics JSONB DEFAULT '{}',
    error_message TEXT,
    logs_s3_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance monitoring
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    avg_inference_time_ms DECIMAL(10,2),
    accuracy DECIMAL(5,2),
    precision DECIMAL(5,2),
    recall DECIMAL(5,2),
    f1_score DECIMAL(5,2),
    false_positive_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, date)
);

CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_active ON ml_models(is_active);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_model_type ON training_jobs(model_type);
CREATE INDEX idx_model_performance_model ON model_performance(model_id);
CREATE INDEX idx_model_performance_date ON model_performance(date DESC);
