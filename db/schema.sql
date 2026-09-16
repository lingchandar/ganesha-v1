-- ============================================================================
-- GANESHA V1 — Production PostgreSQL Database Schema
-- Version: 1.0.0 (Hardened Final)
-- Execute: psql -U ganesha_admin -d ganesha_v1 -f db/schema.sql
-- ============================================================================

-- 1. Eligible Universe Master Table
CREATE TABLE IF NOT EXISTS nse_eligible_universe (
    instrument_token BIGINT PRIMARY KEY,
    ticker_symbol VARCHAR(25) NOT NULL UNIQUE,
    company_name VARCHAR(150) NOT NULL,
    sector_name VARCHAR(75) NOT NULL,
    industry_name VARCHAR(100),
    is_active_swing BOOLEAN DEFAULT TRUE,
    market_cap_crores NUMERIC(14, 2),
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Point-in-Time Historical Daily OHLCV & Delivery Data
CREATE TABLE IF NOT EXISTS historical_daily_candles (
    id BIGSERIAL PRIMARY KEY,
    instrument_token BIGINT NOT NULL REFERENCES nse_eligible_universe(instrument_token) ON DELETE CASCADE,
    candle_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price NUMERIC(12, 2) NOT NULL,
    high_price NUMERIC(12, 2) NOT NULL,
    low_price NUMERIC(12, 2) NOT NULL,
    close_price NUMERIC(12, 2) NOT NULL,
    volume_traded BIGINT NOT NULL,
    delivery_volume BIGINT,
    delivery_percentage NUMERIC(6, 2),
    is_corporate_action_adjusted BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_token_candle_time UNIQUE (instrument_token, candle_timestamp)
);

-- 3. Corporate Actions Registry for CATB Enforcement
CREATE TABLE IF NOT EXISTS corporate_actions_calendar (
    id BIGSERIAL PRIMARY KEY,
    ticker_symbol VARCHAR(25) NOT NULL,
    action_type VARCHAR(50) NOT NULL,  -- SPLIT, BONUS, DIVIDEND, RIGHTS
    ex_date DATE NOT NULL,
    record_date DATE,
    adjustment_factor NUMERIC(10, 4) DEFAULT 1.0000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. NSE Derivative Expiry Calendar for Expiry-Week Volatility Filtering
CREATE TABLE IF NOT EXISTS nse_derivative_expiry_calendar (
    id BIGSERIAL PRIMARY KEY,
    expiry_date DATE NOT NULL UNIQUE,
    contract_month VARCHAR(10) NOT NULL,
    is_monthly_expiry BOOLEAN DEFAULT TRUE,
    three_day_buffer_start DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Trade Journal & Audit Registry
CREATE TABLE IF NOT EXISTS ganesha_trade_journal (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_date DATE NOT NULL,
    ticker_symbol VARCHAR(25) NOT NULL,
    setup_type VARCHAR(50) NOT NULL,
    market_regime VARCHAR(30) NOT NULL,
    sector_name VARCHAR(75) NOT NULL,
    recommended_entry NUMERIC(12, 2) NOT NULL,
    stop_loss NUMERIC(12, 2) NOT NULL,
    target_price NUMERIC(12, 2) NOT NULL,
    risk_reward_ratio NUMERIC(5, 2) NOT NULL,
    calibrated_win_probability NUMERIC(5, 4) NOT NULL,
    allocated_position_shares INT NOT NULL,
    capital_at_risk_inr NUMERIC(12, 2) NOT NULL,
    gemini_audit_reasoning TEXT NOT NULL,
    tool_call_payload JSONB NOT NULL,
    trade_status VARCHAR(30) DEFAULT 'WATCHLIST_PENDING',
    realized_r_multiple NUMERIC(6, 2),
    exit_date DATE,
    exit_reason VARCHAR(50)
);

-- ============================================================================
-- Performance Indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_candles_perf_scan
ON historical_daily_candles (instrument_token, candle_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_universe_sector_group
ON nse_eligible_universe (sector_name);

CREATE INDEX IF NOT EXISTS idx_catb_lookup
ON corporate_actions_calendar (ticker_symbol, ex_date);

CREATE INDEX IF NOT EXISTS idx_expiry_lookup
ON nse_derivative_expiry_calendar (expiry_date, three_day_buffer_start);

CREATE INDEX IF NOT EXISTS idx_journal_status_date
ON ganesha_trade_journal (scan_date, trade_status);
