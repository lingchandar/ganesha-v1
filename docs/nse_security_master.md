# NSE Security Master Ingestion

Ganesha ingests the official NSE CM-MII daily security master before applying
any swing-trading eligibility rules.

## Source

NSE daily CM-MII security file:

`https://nsearchives.nseindia.com/content/equities/NSE_CM_security_DDMMYYYY.csv.gz`

The filename date is the NSE snapshot date.

## Pipeline

```
NSE daily security master
        |
        v
raw snapshot + SHA-256
        |
        v
validated normalized rows
        |
        v
NSE listed universe
        |
        v
equity-series filter
        |
        v
liquidity / swing eligibility
```

Do not hard-code NIFTY 100 as the production universe.

## Historical backfill

Backfill one official daily snapshot per NSE trading session, retain the raw
source file/hash, write `universe_membership_snapshot`, then compact changes
into `universe_membership_history`. This prevents survivorship bias.

The backfill must use actual NSE trading-session dates. Missing source files
must be recorded as missing; they must not be silently replaced with today's
universe.

## Important

FYERS instrument tokens are not assumed to be NSE security-master identifiers.
The live broker instrument mapping should be maintained separately.
