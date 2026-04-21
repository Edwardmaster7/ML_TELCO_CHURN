import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_merge_data(
    path_customers: str,
    path_services: str,
    path_contracts: str
) -> pd.DataFrame:
    """Carrega os dados particionados em 3 tabelas e faz o merge por CustomerID."""
    try:
        df_customers = pd.read_csv(path_customers)
        df_services = pd.read_csv(path_services)
        df_contracts = pd.read_csv(path_contracts)

        df = df_customers.merge(df_services, on="customerID", how="inner")
        df = df.merge(df_contracts, on="customerID", how="inner")

        logger.info(f"Shape final do merge: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        raise
