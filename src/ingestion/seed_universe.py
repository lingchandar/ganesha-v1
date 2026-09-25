"""
GANESHA V1 — NIFTY 100 Universe Master Seeder (Module 1, 4)

Populates the nse_eligible_universe table with the 100 constituents of
NIFTY 50 and NIFTY NEXT 50 with sectoral classifications and Fyers symbol formatting.
"""
from datetime import datetime, timezone
from sqlalchemy import text
from loguru import logger

from src.core.database import get_db_session

# Authentic Nifty 100 constituents mapping (Symbol, Name, Sector)
NIFTY_100_CONSTITUENTS = [
    # ── Financial Services / Banking ──
    ("HDFCBANK", "HDFC Bank Ltd.", "FINANCIAL_SERVICES", "Private Bank"),
    ("ICICIBANK", "ICICI Bank Ltd.", "FINANCIAL_SERVICES", "Private Bank"),
    ("SBIN", "State Bank of India", "FINANCIAL_SERVICES", "Public Bank"),
    ("KOTAKBANK", "Kotak Mahindra Bank Ltd.", "FINANCIAL_SERVICES", "Private Bank"),
    ("AXISBANK", "Axis Bank Ltd.", "FINANCIAL_SERVICES", "Private Bank"),
    ("BAJFINANCE", "Bajaj Finance Ltd.", "FINANCIAL_SERVICES", "NBFC"),
    ("BAJAJFINSV", "Bajaj Finserv Ltd.", "FINANCIAL_SERVICES", "NBFC"),
    ("HDFCLIFE", "HDFC Life Insurance Co.", "FINANCIAL_SERVICES", "Insurance"),
    ("SBILIFE", "SBI Life Insurance Co.", "FINANCIAL_SERVICES", "Insurance"),
    ("CHOLAFIN", "Cholamandalam Investment", "FINANCIAL_SERVICES", "NBFC"),
    ("SHRIRAMFIN", "Shriram Finance Ltd.", "FINANCIAL_SERVICES", "NBFC"),
    ("PFC", "Power Finance Corporation", "FINANCIAL_SERVICES", "NBFC"),
    ("RECLTD", "REC Ltd.", "FINANCIAL_SERVICES", "NBFC"),
    ("BANKBARODA", "Bank of Baroda", "FINANCIAL_SERVICES", "Public Bank"),
    ("PNB", "Punjab National Bank", "FINANCIAL_SERVICES", "Public Bank"),
    ("CANBK", "Canara Bank", "FINANCIAL_SERVICES", "Public Bank"),
    ("INDUSINDBK", "IndusInd Bank Ltd.", "FINANCIAL_SERVICES", "Private Bank"),

    # ── Information Technology ──
    ("TCS", "Tata Consultancy Services Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("INFY", "Infosys Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("HCLTECH", "HCL Technologies Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("WIPRO", "Wipro Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("TECHM", "Tech Mahindra Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("LTIM", "LTIMindtree Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("PERSISTENT", "Persistent Systems Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("COFORGE", "Coforge Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),
    ("MPHASIS", "Mphasis Ltd.", "INFORMATION_TECHNOLOGY", "IT Services"),

    # ── Oil, Gas & Consumable Fuels ──
    ("RELIANCE", "Reliance Industries Ltd.", "OIL_AND_GAS", "Refineries & Marketing"),
    ("ONGC", "Oil & Natural Gas Corp Ltd.", "OIL_AND_GAS", "Oil Exploration"),
    ("BPCL", "Bharat Petroleum Corp Ltd.", "OIL_AND_GAS", "Refineries & Marketing"),
    ("IOC", "Indian Oil Corporation Ltd.", "OIL_AND_GAS", "Refineries & Marketing"),
    ("GAIL", "GAIL (India) Ltd.", "OIL_AND_GAS", "Gas Transmission"),
    ("COALINDIA", "Coal India Ltd.", "OIL_AND_GAS", "Coal"),
    ("PETRONET", "Petronet LNG Ltd.", "OIL_AND_GAS", "Gas"),

    # ── Automobiles & Auto Components ──
    ("TATAMOTORS", "Tata Motors Ltd.", "AUTOMOBILE", "Commercial & Passenger Vehicles"),
    ("M&M", "Mahindra & Mahindra Ltd.", "AUTOMOBILE", "Passenger Vehicles & Tractors"),
    ("MARUTI", "Maruti Suzuki India Ltd.", "AUTOMOBILE", "Passenger Vehicles"),
    ("BAJAJ-AUTO", "Bajaj Auto Ltd.", "AUTOMOBILE", "2/3 Wheelers"),
    ("EICHERMOT", "Eicher Motors Ltd.", "AUTOMOBILE", "2 Wheelers"),
    ("HEROMOTOCO", "Hero MotoCorp Ltd.", "AUTOMOBILE", "2 Wheelers"),
    ("TVSMOTOR", "TVS Motor Company Ltd.", "AUTOMOBILE", "2 Wheelers"),
    ("BHARATFORG", "Bharat Forge Ltd.", "AUTOMOBILE", "Auto Components"),
    ("BOSCHLTD", "Bosch Ltd.", "AUTOMOBILE", "Auto Components"),
    ("SAMVARDHANA", "Samvardhana Motherson Intl.", "AUTOMOBILE", "Auto Components"),

    # ── Fast Moving Consumer Goods (FMCG) ──
    ("ITC", "ITC Ltd.", "FMCG", "Diversified FMCG"),
    ("HINDUNILVR", "Hindustan Unilever Ltd.", "FMCG", "Diversified FMCG"),
    ("NESTLEIND", "Nestle India Ltd.", "FMCG", "Packaged Foods"),
    ("BRITANNIA", "Britannia Industries Ltd.", "FMCG", "Packaged Foods"),
    ("TATACONSUM", "Tata Consumer Products Ltd.", "FMCG", "Tea & Coffee"),
    ("VBL", "Varun Beverages Ltd.", "FMCG", "Beverages"),
    ("DABUR", "Dabur India Ltd.", "FMCG", "Personal Care"),
    ("GODREJCP", "Godrej Consumer Products", "FMCG", "Personal Care"),
    ("MARICO", "Marico Ltd.", "FMCG", "Edible Oils & Hair Care"),
    ("COLPAL", "Colgate-Palmolive (India) Ltd.", "FMCG", "Oral Care"),

    # ── Healthcare & Pharmaceuticals ──
    ("SUNPHARMA", "Sun Pharmaceutical Industries", "HEALTHCARE", "Pharmaceuticals"),
    ("CIPLA", "Cipla Ltd.", "HEALTHCARE", "Pharmaceuticals"),
    ("DRREDDY", "Dr. Reddy's Laboratories", "HEALTHCARE", "Pharmaceuticals"),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise", "HEALTHCARE", "Hospital Chains"),
    ("DIVISLAB", "Divi's Laboratories Ltd.", "HEALTHCARE", "Active Pharmaceutical Ingredients"),
    ("TORNTPHARM", "Torrent Pharmaceuticals Ltd.", "HEALTHCARE", "Pharmaceuticals"),
    ("ZYDUSLIFE", "Zydus Lifesciences Ltd.", "HEALTHCARE", "Pharmaceuticals"),
    ("LUPIN", "Lupin Ltd.", "HEALTHCARE", "Pharmaceuticals"),
    ("MAXHEALTH", "Max Healthcare Institute", "HEALTHCARE", "Hospital Chains"),
    ("MANKIND", "Mankind Pharma Ltd.", "HEALTHCARE", "Pharmaceuticals"),

    # ── Metals & Mining ──
    ("TATASTEEL", "Tata Steel Ltd.", "METALS", "Steel"),
    ("JSWSTEEL", "JSW Steel Ltd.", "METALS", "Steel"),
    ("HINDALCO", "Hindalco Industries Ltd.", "METALS", "Aluminium"),
    ("VEDL", "Vedanta Ltd.", "METALS", "Diversified Metals"),
    ("JINDALSTEL", "Jindal Steel & Power Ltd.", "METALS", "Steel"),
    ("NMDC", "NMDC Ltd.", "METALS", "Iron Ore"),
    ("NATIONALUM", "National Aluminium Co Ltd.", "METALS", "Aluminium"),

    # ── Construction, Engineering & Capital Goods ──
    ("LT", "Larsen & Toubro Ltd.", "CAPITAL_GOODS", "Infrastructure & Engineering"),
    ("SIEMENS", "Siemens Ltd.", "CAPITAL_GOODS", "Industrial Equipment"),
    ("ABB", "ABB India Ltd.", "CAPITAL_GOODS", "Industrial Equipment"),
    ("HAL", "Hindustan Aeronautics Ltd.", "CAPITAL_GOODS", "Aerospace & Defence"),
    ("BEL", "Bharat Electronics Ltd.", "CAPITAL_GOODS", "Defence Electronics"),
    ("BHEL", "Bharat Heavy Electricals Ltd.", "CAPITAL_GOODS", "Heavy Electricals"),
    ("CUMMINSIND", "Cummins India Ltd.", "CAPITAL_GOODS", "Engines"),

    # ── Power & Utilities ──
    ("NTPC", "NTPC Ltd.", "POWER", "Power Generation"),
    ("POWERGRID", "Power Grid Corporation", "POWER", "Power Transmission"),
    ("TATAPOWER", "Tata Power Co Ltd.", "POWER", "Integrated Power"),
    ("ADANIGREEN", "Adani Green Energy Ltd.", "POWER", "Renewable Energy"),
    ("ADANIPOWER", "Adani Power Ltd.", "POWER", "Power Generation"),

    # ── Telecommunication & Services ──
    ("BHARTIARTL", "Bharti Airtel Ltd.", "TELECOM", "Telecom Services"),
    ("IDEA", "Vodafone Idea Ltd.", "TELECOM", "Telecom Services"),

    # ── Consumer Services, Retail & Travel ──
    ("TITAN", "Titan Company Ltd.", "CONSUMER_SERVICES", "Gems, Jewellery & Watches"),
    ("TRENT", "Trent Ltd.", "CONSUMER_SERVICES", "Apparel Retail"),
    ("DMART", "Avenue Supermarts Ltd.", "CONSUMER_SERVICES", "Hypermarkets"),
    ("ZOMATO", "Zomato Ltd.", "CONSUMER_SERVICES", "Food Delivery & Quick Commerce"),
    ("INDIGO", "InterGlobe Aviation Ltd.", "CONSUMER_SERVICES", "Airlines"),
    ("INDIANHOTE", "The Indian Hotels Company", "CONSUMER_SERVICES", "Hotels"),

    # ── Cement & Construction Materials ──
    ("ULTRACEMCO", "UltraTech Cement Ltd.", "CEMENT", "Cement"),
    ("GRASIM", "Grasim Industries Ltd.", "CEMENT", "Diversified"),
    ("AMBUJACEM", "Ambuja Cements Ltd.", "CEMENT", "Cement"),
    ("SHREECEM", "Shree Cement Ltd.", "CEMENT", "Cement"),

    # ── Chemicals ──
    ("PIDILITIND", "Pidilite Industries Ltd.", "CHEMICALS", "Specialty Chemicals"),
    ("SRF", "SRF Ltd.", "CHEMICALS", "Specialty Chemicals"),
    ("SOLARINDS", "Solar Industries India Ltd.", "CHEMICALS", "Industrial Explosives"),

    # ── Realty / Real Estate ──
    ("DLF", "DLF Ltd.", "REALTY", "Real Estate"),
    ("GODREJPROP", "Godrej Properties Ltd.", "REALTY", "Real Estate"),
    ("LODHA", "Macrotech Developers Ltd.", "REALTY", "Real Estate"),
]


def seed_nifty100_universe():
    """Seed the database with all 100 constituent equities."""
    logger.info("Seeding PostgreSQL database with authentic NIFTY 100 equities...")

    with get_db_session() as db:
        token_counter = 10001
        inserted = 0

        for symbol, name, sector, industry in NIFTY_100_CONSTITUENTS:
            fyers_ticker = f"NSE:{symbol}-EQ"
            db.execute(
                text("""
                    INSERT INTO nse_eligible_universe 
                        (instrument_token, ticker_symbol, company_name, sector_name, industry_name, is_active_swing, last_updated_at)
                    VALUES 
                        (:token, :ticker, :name, :sector, :industry, TRUE, :now)
                    ON CONFLICT (ticker_symbol) 
                    DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector_name = EXCLUDED.sector_name,
                        industry_name = EXCLUDED.industry_name,
                        is_active_swing = TRUE,
                        last_updated_at = EXCLUDED.last_updated_at
                """),
                {
                    "token": token_counter,
                    "ticker": fyers_ticker,
                    "name": name,
                    "sector": sector,
                    "industry": industry,
                    "now": datetime.now(timezone.utc),
                }
            )
            token_counter += 1
            inserted += 1

    logger.success(f"Successfully seeded {inserted} NIFTY 100 equities into nse_eligible_universe table!")


if __name__ == "__main__":
    seed_nifty100_universe()
