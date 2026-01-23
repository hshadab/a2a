-- Threat Intelligence Network Database Schema
-- PostgreSQL 15+

-- Enable trigram extension for similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Classifications table - stores all URL classifications with proofs
CREATE TABLE IF NOT EXISTS classifications (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    classification TEXT NOT NULL CHECK (classification IN ('PHISHING', 'SAFE', 'SUSPICIOUS')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Proof data
    proof_hash TEXT NOT NULL,
    model_commitment TEXT NOT NULL,
    input_commitment TEXT NOT NULL,
    output_commitment TEXT NOT NULL,

    -- Features and context
    features JSONB,
    context_used JSONB,

    -- Provenance
    source TEXT,
    batch_id UUID,
    analyst_paid_usdc FLOAT,
    policy_proof_hash TEXT,
    classified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint on URL
    CONSTRAINT unique_url UNIQUE (url)
);

-- Indexes for classifications
CREATE INDEX IF NOT EXISTS idx_classifications_url ON classifications(url);
CREATE INDEX IF NOT EXISTS idx_classifications_domain ON classifications(domain);
CREATE INDEX IF NOT EXISTS idx_classifications_batch ON classifications(batch_id);
CREATE INDEX IF NOT EXISTS idx_classifications_time ON classifications(classified_at DESC);
CREATE INDEX IF NOT EXISTS idx_classifications_class ON classifications(classification);

-- Batches table - tracks each processing batch
CREATE TABLE IF NOT EXISTS batches (
    id UUID PRIMARY KEY,
    url_count INT NOT NULL,
    source TEXT,
    policy_decision TEXT NOT NULL CHECK (policy_decision IN ('AUTHORIZED', 'DENIED')),
    policy_proof_hash TEXT NOT NULL,
    policy_paid_usdc FLOAT DEFAULT 0,
    total_analyst_paid_usdc FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_batches_time ON batches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batches_completed ON batches(completed_at);

-- Domain statistics - aggregated stats per domain
CREATE TABLE IF NOT EXISTS domain_stats (
    domain TEXT PRIMARY KEY,
    times_seen INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    safe_count INT DEFAULT 0,
    suspicious_count INT DEFAULT 0,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE
);

-- Trigram index for similarity search
CREATE INDEX IF NOT EXISTS idx_domain_stats_trgm ON domain_stats USING gin (domain gin_trgm_ops);

-- Registrar statistics
CREATE TABLE IF NOT EXISTS registrar_stats (
    registrar TEXT PRIMARY KEY,
    domains_seen INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE
);

-- IP statistics
CREATE TABLE IF NOT EXISTS ip_stats (
    ip TEXT PRIMARY KEY,
    domains_hosted INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE
);

-- Treasury tracking
CREATE TABLE IF NOT EXISTS treasury (
    id SERIAL PRIMARY KEY,
    balance_usdc FLOAT DEFAULT 1000.0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Initialize treasury with default balance
INSERT INTO treasury (balance_usdc) VALUES (1000.0) ON CONFLICT DO NOTHING;

-- Proofs table - stores full proof data for verification
CREATE TABLE IF NOT EXISTS proofs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proof_hash TEXT UNIQUE NOT NULL,
    proof_type TEXT NOT NULL CHECK (proof_type IN ('AUTHORIZATION', 'CLASSIFICATION', 'BATCH')),
    proof_data BYTEA,
    model_commitment TEXT NOT NULL,
    input_commitment TEXT NOT NULL,
    output_commitment TEXT NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    verify_time_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_proofs_hash ON proofs(proof_hash);
CREATE INDEX IF NOT EXISTS idx_proofs_type ON proofs(proof_type);

-- Payments table - tracks all x402 payments
CREATE TABLE IF NOT EXISTS payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tx_hash TEXT UNIQUE,
    amount_usdc FLOAT NOT NULL,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    memo TEXT,
    chain_id INT DEFAULT 8453,
    block_number BIGINT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_payments_tx ON payments(tx_hash);
CREATE INDEX IF NOT EXISTS idx_payments_recipient ON payments(recipient);

-- View for network statistics
CREATE OR REPLACE VIEW network_stats_view AS
SELECT
    (SELECT COUNT(*) FROM classifications) as total_urls,
    (SELECT COUNT(*) FROM classifications WHERE classification = 'PHISHING') as phishing_count,
    (SELECT COUNT(*) FROM classifications WHERE classification = 'SAFE') as safe_count,
    (SELECT COUNT(*) FROM classifications WHERE classification = 'SUSPICIOUS') as suspicious_count,
    (SELECT COUNT(*) FROM batches) as total_batches,
    (SELECT COUNT(*) FROM proofs WHERE verified = true) as verified_proofs,
    (SELECT COALESCE(SUM(policy_paid_usdc), 0) FROM batches) as total_policy_paid,
    (SELECT COALESCE(SUM(total_analyst_paid_usdc), 0) FROM batches) as total_analyst_paid,
    (SELECT balance_usdc FROM treasury ORDER BY id DESC LIMIT 1) as treasury_balance,
    (SELECT MIN(created_at) FROM batches) as running_since;
