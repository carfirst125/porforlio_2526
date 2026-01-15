from sqlalchemy import Column, create_engine, engine, String, SmallInteger, Float, DateTime, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Any, Optional, Union, List
from sqlalchemy import text

import logging

# Thiết lập logger
logging.basicConfig(
    level=logging.INFO,  # mức log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

Base = declarative_base()
Connection = Union[engine.Engine, str]

class FmCcyRate(Base):
    __tablename__ = 'fm_ccy_rate'
    __table_args__ = {'schema': 'eoc'}

    # Primary key columns
    ccy = Column(String(3), primary_key=True)
    branch = Column(String(6), primary_key=True)
    rate_type = Column(String(3), primary_key=True)
    
    # Other columns
    effective_date = Column(DateTime(6), nullable=False)
    last_change_date = Column(DateTime(6), nullable=False)
    quote_type = Column(String(1), nullable=False)
    ccy_rate = Column(Float, nullable=False)
    buy_rate = Column(Float, nullable=False)
    sell_rate = Column(Float, nullable=False)
    central_bank_rate = Column(Float, nullable=True)
    buy_spread = Column(Float, nullable=True)
    sell_spread = Column(Float, nullable=True)
    ctrl_spread_usd = Column(Float, nullable=True)
    cdc_id = Column(BigInteger, nullable=False)
    cdc_timestamp = Column(DateTime(6), nullable=True)

    def to_dict(self) -> dict:
        """
        Convert rate object to dictionary
        """
        return {
            "ccy": self.ccy,
            "branch": self.branch,
            "rate_type": self.rate_type,
            "effective_date": self.effective_date.isoformat(),
            "last_change_date": self.last_change_date.isoformat(),
            "quote_type": self.quote_type,
            "ccy_rate": self.ccy_rate,
            "buy_rate": self.buy_rate,
            "sell_rate": self.sell_rate,
            "central_bank_rate": self.central_bank_rate,
            "buy_spread": self.buy_spread,
            "sell_spread": self.sell_spread,
            "ctrl_spread_usd": self.ctrl_spread_usd,
            "cdc_id": self.cdc_id,
            "cdc_timestamp": self.cdc_timestamp.isoformat() if self.cdc_timestamp else None
        }
    
class SavingCoreRate(Base):
    __tablename__ = 'tbl_saving_core_rate'
    __table_args__ = {'schema': 'eocetl'}

    # Primary key column
    id = Column(String, primary_key=True)  # UUID stored as string
    
    # Other columns
    branch = Column(String(6))
    ccy = Column(String(3))
    term = Column(String(5))
    day_basis = Column(SmallInteger)  # int2
    balance = Column(BigInteger)      # int8
    int_type = Column(String(3))
    cr_int_freq = Column(String(5))
    actual_rate = Column(Float)       # float8
    effect_date = Column(DateTime(6))
    last_change_date = Column(DateTime(6))
    last_update = Column(DateTime(6))
    product_type = Column(String(3))
    int_basis = Column(String(3))

    def to_dict(self) -> dict:
        """
        Convert saving rate object to dictionary
        """
        return {
            "id": self.id,
            "branch": self.branch,
            "ccy": self.ccy,
            "term": self.term,
            "day_basis": self.day_basis,
            "balance": self.balance,
            "int_type": self.int_type,
            "cr_int_freq": self.cr_int_freq,
            "actual_rate": self.actual_rate,
            "effect_date": self.effect_date.isoformat() if self.effect_date else None,
            "last_change_date": self.last_change_date.isoformat() if self.last_change_date else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "product_type": self.product_type,
            "int_basis": self.int_basis
        }

class FunctionsStore:
    """Store for managing function tables."""
    
    connection_string: str
    # Constants
    CLIENT_TYPE_CA_NHAN = "CN"
    SAVING_TYPE_TRUC_TUYEN = "ONLINE"
    SAVING_TYPE_TAI_QUAY = "COUNTER"
    CURRENCY_USD = "USD"
    CURRENCY_EUR = "EUR"
    CURRENCY_AUD = "AUD"
    CURRENCY_VND = "VND"
    
    def __init__(self, *, connection: Connection, engine_args: Optional[dict[str, Any]] = None, create_tables: bool = True):
        if isinstance(connection, str):
            self._engine = create_engine(url=connection, **(engine_args or {}))
        elif isinstance(connection, engine.Engine):
            self._engine = connection
        else:
            raise ValueError("connection should be a connection string or an instance of " "sqlalchemy.engine.Engine")
        self._session_maker = sessionmaker(bind=self._engine)
        self.create_tables = create_tables
        self.__post_init__()


    def __post_init__(self):
        """Initialize the store."""
        if self.create_tables:
            self.create_tables_if_not_exists()


    def create_tables_if_not_exists(self):
        """Create the tables if doesn't exist."""
        with self._session_maker() as session:
            session.execute(text("CREATE SCHEMA IF NOT EXISTS eoc"))
            session.execute(text("CREATE SCHEMA IF NOT EXISTS eocetl"))
            session.commit()
            Base.metadata.create_all(session.get_bind())
         

    def get_exchange_rate(self, 
                 from_ccy: str, 
                 to_ccy: str, 
                 rate_type: str, 
                 effective_date: Optional[datetime] = None) -> Optional[List[dict]]:
        """
        Get exchange rate based on from currency, to currency, rate type, branch and optional effective date
        """

        with self._session_maker() as session:
            try:
                # Build the base query with DISTINCT ON
                query = session.query(
                    FmCcyRate.ccy,
                    FmCcyRate.effective_date,
                    FmCcyRate.rate_type,
                    FmCcyRate.buy_rate,
                    FmCcyRate.sell_rate
                ).distinct(
                    FmCcyRate.effective_date,
                    FmCcyRate.ccy,
                    FmCcyRate.rate_type
                ).filter(
                    and_(
                        FmCcyRate.ccy.in_([from_ccy, to_ccy]),
                        FmCcyRate.rate_type.in_(['CA1', 'CA3', 'TRS']),
                        FmCcyRate.branch == '001',
                        FmCcyRate.rate_type.like(f"{rate_type}%")
                    )
                )
                
                # Add effective_date filter if provided
                if effective_date:
                    query = query.filter(FmCcyRate.effective_date <= effective_date)
                
                # Order the results
                query = query.order_by(
                    FmCcyRate.ccy,
                    FmCcyRate.rate_type
                )
                
                exchange_rates = query.all()

                if not exchange_rates:
                    return []

                # Format the results
                formatted_rates = []
                for rate in exchange_rates:
                    rate_name = 'transfer' if rate.rate_type == 'TRS' else 'cash'
                    
                    formatted_rate = {
                        "currency": rate.ccy,
                        "exchange_type": rate_name,
                    }
                    
                    # Add buy_rate only for TRS and CA1
                    if rate.rate_type in ['TRS', 'CA1']:
                        formatted_rate["buying_price"] = f"{rate.buy_rate} VND"
                    
                    # Add sell_rate only for TRS and CA3
                    if rate.rate_type in ['TRS', 'CA3']:
                        formatted_rate["selling_price"] = f"{rate.sell_rate} VND"
                    
                    formatted_rates.append(formatted_rate)
               
                return formatted_rates
               
            finally:
                session.close() 

    def get_savings_rate(self, 
                         product_type: str,
                         saving_type: str = "ONLINE, COUNTER",                 
                         client_type: str = CLIENT_TYPE_CA_NHAN,
                         term_type: str = "1M",
                         ccy: str = CURRENCY_VND,
                         balance: int = 0
                         ) -> str:

        product_types = [type.strip() for type in product_type.split(',')] if product_type else []
        term_types = [type.strip() for type in term_type.split(',')]
        currencies = [ccy] if ccy else []
        saving_types = [type.strip() for type in saving_type.split(',')]
        int_types: List[str] = []


        # Set default product types if empty
        if not product_types:
            if client_type == self.CLIENT_TYPE_CA_NHAN:
                if self.SAVING_TYPE_TRUC_TUYEN in saving_types:                
                    product_types.append("524")
                if self.SAVING_TYPE_TAI_QUAY in saving_types:                                    
                    product_types.append("TDE")
            else:
                if self.SAVING_TYPE_TRUC_TUYEN in saving_types:                
                    product_types.append("588")
                if self.SAVING_TYPE_TAI_QUAY in saving_types:                                    
                    product_types.append("TDE")

        # Process product types
        i = 0
        while i < len(product_types):
            if product_types[i] == "401":  # automatic
                if client_type == self.CLIENT_TYPE_CA_NHAN and self.SAVING_TYPE_TRUC_TUYEN in saving_types:  
                    int_types.append("SA1")
                else:
                    product_types.pop(i)
                    continue

            elif product_types[i] == "502":  # installment
                if client_type == self.CLIENT_TYPE_CA_NHAN and self.SAVING_TYPE_TAI_QUAY in saving_types:  
                    int_types.append("S")
                else:
                    product_types.pop(i)
                    continue

            elif product_types[i] == "506":  # target
                if not (client_type == self.CLIENT_TYPE_CA_NHAN and self.SAVING_TYPE_TRUC_TUYEN in saving_types):
                    product_types.pop(i)
                    continue

            elif product_types[i] == "524":  # online
                if client_type == self.CLIENT_TYPE_CA_NHAN and self.SAVING_TYPE_TRUC_TUYEN in saving_types:
                    int_types.append("IT1")
                else:
                    product_types.pop(i)
                    continue

            elif product_types[i] == "526":  # front-end interest
                if client_type == self.CLIENT_TYPE_CA_NHAN and self.SAVING_TYPE_TAI_QUAY in saving_types:
                    int_types.append("FT2")
                else:
                    product_types.pop(i)
                    continue

            elif product_types[i] == "528":  # idepo
                if client_type == self.CLIENT_TYPE_CA_NHAN:
                    if self.SAVING_TYPE_TRUC_TUYEN in saving_types:
                        int_types.append("ET3")
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        product_types.append("529")
                        int_types.append("ET2")
                else:
                    product_types.pop(i)
                    continue

            elif product_types[i] == "TDE":  # term deposit
                if client_type == self.CLIENT_TYPE_CA_NHAN:
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        int_types.append("TDE")
                    else:
                        product_types.pop(i)
                        continue
                else:
                    if self.SAVING_TYPE_TRUC_TUYEN in saving_types:
                        int_types.append("IT8")
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        product_types.append("TMS")
                        int_types.append("TMS")

            elif product_types[i] == "FFD":  # flexible
                if client_type == self.CLIENT_TYPE_CA_NHAN:
                    if not self.SAVING_TYPE_TAI_QUAY in saving_types:
                        product_types.pop(i)
                        continue
                else:
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        product_types[i] = "FBM"
                    else:
                        product_types.pop(i)
                        continue

            elif product_types[i] == "TMS":  # foreign currency
                if client_type == self.CLIENT_TYPE_CA_NHAN:
                    if self.SAVING_TYPE_TRUC_TUYEN in saving_types:
                        product_types[i] = "524"
                        int_types.append("IT1")
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        product_types.append("TDE")
                        int_types.append("TDE")
                    if not ccy:
                        ccy = [self.CURRENCY_USD, self.CURRENCY_EUR, self.CURRENCY_AUD]
                else:
                    if self.SAVING_TYPE_TAI_QUAY in saving_types:
                        int_types.append("TMS")
                        if not ccy:
                            ccy = [self.CURRENCY_USD, self.CURRENCY_EUR]
                        break
                    product_types.pop(i)
                    continue

            else:  # invalid product_type
                product_types.pop(i)
                continue

            i += 1

        if not product_types:
            return ""

        # Set default currency
        if not ccy:
            ccy = [self.CURRENCY_VND]

        has_product_types = len(product_types) > 0
        has_term_types = len(term_types) > 0
        has_currencies = len(currencies) > 0
        has_int_types = len(int_types) > 0

                    
        with self._session_maker() as session:
            try:
                result = session.execute(
                    text("""
                        SELECT 
                            CASE 
                                WHEN product_type IN ('524','528','506','401','TDE','529','502','526','FFD') THEN 'cá nhân'
                                WHEN product_type IN ('588','TMS','FBM') THEN 'doanh nghiệp'
                            END AS client_type,
                            
                            CASE 
                                WHEN product_type IN ('524','528','506','401','588') THEN 'trực tuyến'
                                WHEN product_type IN ('TDE','529','502','526','FFD','TMS','FBM') THEN 'tại quầy'
                            END AS saving_type,
                            
                            CASE product_type 
                                WHEN '524' THEN 
                                    CASE ccy 
                                        WHEN 'VND' THEN 'trực tuyến' 
                                        ELSE 'ngoại tệ' 
                                    END
                                WHEN '528' THEN 'iDepo'
                                WHEN '506' THEN 'mục tiêu'
                                WHEN '401' THEN 'tự động'
                                WHEN 'TDE' THEN 'có kỳ hạn'
                                WHEN '529' THEN 'iDepo'
                                WHEN '502' THEN 'gửi góp'
                                WHEN '526' THEN 'lãi đầu kỳ'
                                WHEN 'FFD' THEN 'linh hoạt'
                                WHEN 'TDE' THEN 'ngoại tệ'
                                WHEN '588' THEN 'có kỳ hạn'
                                WHEN 'TMS' THEN 
                                    CASE ccy 
                                        WHEN 'VND' THEN 'có kỳ hạn' 
                                        ELSE 'ngoại tệ' 
                                    END
                                WHEN 'FBM' THEN 'linh hoạt'
                                ELSE product_type 
                            END AS product_name,
                            
                            REPLACE(
                                REPLACE(
                                    REPLACE(
                                        REPLACE(
                                            REPLACE(
                                                REPLACE(
                                                    term,
                                                    '7D', '1 tuần'
                                                ),
                                                '14D', '2 tuần'
                                            ),
                                            '21D', '3 tuần'
                                        ),
                                        'D', ' ngày'
                                    ),
                                    'M', ' tháng'
                                ),
                                'Y', ' năm'
                            ) AS term_name,
                            
                            regexp_replace(term, '[^0-9]', '', 'g')::numeric * 
                            CASE 
                                WHEN position('D' IN term) > 0 THEN 1
                                WHEN position('W' IN term) > 0 THEN 7
                                WHEN position('M' IN term) > 0 THEN 30
                                ELSE 365 
                            END AS term_day,
                            
                            COALESCE(
                                REPLACE(
                                    REPLACE(cr_int_freq, 'D', ' ngày'),
                                    'T', ' tháng'
                                ),
                                'cuối kỳ'
                            ) AS int_freq,
                            
                            cr_int_freq,
                            actual_rate || '%/năm' AS full_rate,
                            balance,
                            ccy
                        FROM eocetl.tbl_saving_core_rate
                        WHERE 
                            (:check_product = 0 OR product_type = ANY(:product_types)) 
                            AND (:check_int_types = 0 OR int_type = ANY(:int_types)) 
                            AND (:check_term = 0 OR term = ANY(:term_types)) 
                            AND (:check_ccy = 0 OR ccy = ANY(:currencies))
                            AND (ccy != 'VND' OR actual_rate > 0)
                        ORDER BY 
                            product_type, 
                            term_day, 
                            cr_int_freq, 
                            balance;
                    """),
                    {
                       "check_product": 1 if has_product_types else 0,
                       "product_types": product_types,
                       "check_int_types": 1 if has_int_types else 0,
                       "int_types": int_types,
                       "check_term": 1 if has_term_types else 0,
                       "term_types": term_types,
                       "check_ccy": 1 if has_currencies else 0,
                       "currencies": currencies
                    }
                )

                # logger.info(f"[get_saving_rate] get data from DB: {len(result)}")
                # Convert result to list of dictionaries
                rates = []
                for row in result:
                    rates.append({
                        "customer_type": row.client_type,
                        "saving_type": row.saving_type,
                        "product_name": row.product_name,
                        "term": row.term_name,
                        "interest_freq": row.int_freq,
                        "rate": row.full_rate,
                        "currency": row.ccy,
                        "from_amount": f"{row.balance:,} VND" if row.balance else "",
                        "to_amount": ""  # Will be filled in next step
                    })
                # Add to_amount for each tier except the last one
                for i in range(len(rates) - 1):
                    if rates[i]["currency"] == rates[i+1]["currency"]:
                        rates[i]["to_amount"] = f"dưới {rates[i+1]['from_amount']}"
                        
                logger.info(f"[get_saving_rate] get data from DB - PROCESSED: {rates}")
                return rates

            finally:
                session.close()    
