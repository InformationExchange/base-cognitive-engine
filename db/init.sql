-- BAIS Cognitive Governance Engine
-- PostgreSQL Database Schema
-- Version: 1.0.0

-- =============================================================================
-- EXTENSIONS
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- TENANTS (Multi-Tenant Support)
-- =============================================================================
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    api_key VARCHAR(64) UNIQUE NOT NULL DEFAULT encode(gen_random_bytes(32), 'hex'),
    api_key_hash VARCHAR(64) NOT NULL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deleted')),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_tenants_api_key ON tenants(api_key);
CREATE INDEX idx_tenants_status ON tenants(status);

-- =============================================================================
-- LLM PROVIDER CONFIGURATIONS (Per-Tenant)
-- =============================================================================
CREATE TABLE tenant_llm_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    api_key_encrypted BYTEA,
    model_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    priority INT DEFAULT 0,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, provider)
);

CREATE INDEX idx_tenant_llm_tenant ON tenant_llm_configs(tenant_id);

-- =============================================================================
-- AUDIT RECORDS (Case Management)
-- =============================================================================
CREATE TABLE audit_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    case_id VARCHAR(50) NOT NULL,
    transaction_id VARCHAR(50) NOT NULL,
    
    -- Request/Response
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    domain VARCHAR(50) DEFAULT 'general',
    
    -- Governance Decision
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('approved', 'rejected', 'flagged', 'improved')),
    confidence DECIMAL(5,4),
    accuracy DECIMAL(5,2),
    
    -- Clinical Status
    clinical_status VARCHAR(20) CHECK (clinical_status IN (
        'truly_working', 'incomplete', 'stubbed', 'simulated', 'fallback', 'failover', 'unknown'
    )),
    
    -- Detected Issues
    issues JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    biases_detected JSONB DEFAULT '[]',
    
    -- Metadata
    llm_provider VARCHAR(50),
    model_name VARCHAR(100),
    processing_time_ms INT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_tenant ON audit_records(tenant_id);
CREATE INDEX idx_audit_case ON audit_records(case_id);
CREATE INDEX idx_audit_transaction ON audit_records(transaction_id);
CREATE INDEX idx_audit_decision ON audit_records(decision);
CREATE INDEX idx_audit_created ON audit_records(created_at);
CREATE INDEX idx_audit_clinical ON audit_records(clinical_status);

-- =============================================================================
-- LEARNING DATA (Pattern Learning)
-- =============================================================================
CREATE TABLE learned_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    
    -- Pattern Info
    pattern_type VARCHAR(50) NOT NULL,
    pattern_name VARCHAR(100) NOT NULL,
    domain VARCHAR(50),
    
    -- Effectiveness
    true_positives INT DEFAULT 0,
    false_positives INT DEFAULT 0,
    true_negatives INT DEFAULT 0,
    false_negatives INT DEFAULT 0,
    confidence_weight DECIMAL(5,4) DEFAULT 1.0,
    
    -- LLM-Specific
    llm_provider VARCHAR(50),
    model_name VARCHAR(100),
    
    -- Transfer Learning
    is_universal BOOLEAN DEFAULT false,
    transfer_score DECIMAL(5,4) DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_patterns_tenant ON learned_patterns(tenant_id);
CREATE INDEX idx_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX idx_patterns_domain ON learned_patterns(domain);
CREATE INDEX idx_patterns_llm ON learned_patterns(llm_provider, model_name);

-- =============================================================================
-- BIAS PROFILES (LLM-Aware Learning)
-- =============================================================================
CREATE TABLE llm_bias_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- LLM Identity
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    -- Big 5 Profile
    openness DECIMAL(5,4),
    conscientiousness DECIMAL(5,4),
    extraversion DECIMAL(5,4),
    agreeableness DECIMAL(5,4),
    neuroticism DECIMAL(5,4),
    
    -- Bias Tendencies
    bias_profile JSONB DEFAULT '{}',
    common_issues JSONB DEFAULT '[]',
    
    -- Effectiveness
    total_audits INT DEFAULT 0,
    issues_detected INT DEFAULT 0,
    false_positive_rate DECIMAL(5,4),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(provider, model_name, model_version)
);

CREATE INDEX idx_bias_provider ON llm_bias_profiles(provider);
CREATE INDEX idx_bias_model ON llm_bias_profiles(model_name);

-- =============================================================================
-- USAGE METRICS (Billing/Analytics)
-- =============================================================================
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Time Period
    date DATE NOT NULL,
    hour INT CHECK (hour >= 0 AND hour < 24),
    
    -- Counts
    api_calls INT DEFAULT 0,
    audits_performed INT DEFAULT 0,
    verifications INT DEFAULT 0,
    improvements INT DEFAULT 0,
    
    -- LLM Usage
    llm_calls INT DEFAULT 0,
    llm_tokens_input INT DEFAULT 0,
    llm_tokens_output INT DEFAULT 0,
    
    -- Performance
    avg_latency_ms DECIMAL(10,2),
    p95_latency_ms DECIMAL(10,2),
    p99_latency_ms DECIMAL(10,2),
    
    -- Results
    approved_count INT DEFAULT 0,
    rejected_count INT DEFAULT 0,
    flagged_count INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, date, hour)
);

CREATE INDEX idx_usage_tenant ON usage_metrics(tenant_id);
CREATE INDEX idx_usage_date ON usage_metrics(date);

-- =============================================================================
-- FEEDBACK RECORDS (Learning Loop)
-- =============================================================================
CREATE TABLE feedback_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_record_id UUID REFERENCES audit_records(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    
    -- Feedback
    feedback_type VARCHAR(20) NOT NULL CHECK (feedback_type IN ('correct', 'incorrect', 'partial', 'unknown')),
    user_correction TEXT,
    user_notes TEXT,
    
    -- Impact
    was_false_positive BOOLEAN DEFAULT false,
    was_false_negative BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_feedback_audit ON feedback_records(audit_record_id);
CREATE INDEX idx_feedback_tenant ON feedback_records(tenant_id);
CREATE INDEX idx_feedback_type ON feedback_records(feedback_type);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_tenant_llm_configs_updated_at
    BEFORE UPDATE ON tenant_llm_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_learned_patterns_updated_at
    BEFORE UPDATE ON learned_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_llm_bias_profiles_updated_at
    BEFORE UPDATE ON llm_bias_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- DEFAULT TENANT (For Development)
-- =============================================================================
INSERT INTO tenants (name, api_key, api_key_hash, settings) VALUES (
    'Development',
    'dev-api-key-for-testing-only',
    encode(sha256('dev-api-key-for-testing-only'::bytea), 'hex'),
    '{"rate_limit": 1000, "tier": "development"}'
);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Tenant Usage Summary
CREATE VIEW tenant_usage_summary AS
SELECT 
    t.id as tenant_id,
    t.name as tenant_name,
    COALESCE(SUM(um.api_calls), 0) as total_api_calls,
    COALESCE(SUM(um.audits_performed), 0) as total_audits,
    COALESCE(SUM(um.llm_tokens_input + um.llm_tokens_output), 0) as total_llm_tokens,
    COALESCE(AVG(um.avg_latency_ms), 0) as avg_latency_ms
FROM tenants t
LEFT JOIN usage_metrics um ON t.id = um.tenant_id
GROUP BY t.id, t.name;

-- Pattern Effectiveness Summary
CREATE VIEW pattern_effectiveness AS
SELECT 
    pattern_type,
    domain,
    COUNT(*) as pattern_count,
    AVG(confidence_weight) as avg_confidence,
    SUM(true_positives) as total_tp,
    SUM(false_positives) as total_fp,
    CASE 
        WHEN SUM(true_positives) + SUM(false_positives) > 0 
        THEN SUM(true_positives)::DECIMAL / (SUM(true_positives) + SUM(false_positives))
        ELSE 0 
    END as precision
FROM learned_patterns
GROUP BY pattern_type, domain;

-- =============================================================================
-- GRANTS (For Application User)
-- =============================================================================
-- In production, create a separate application user with limited privileges
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO bais_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bais_app;

