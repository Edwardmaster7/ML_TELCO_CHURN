"""Contratos de Dados e validação usando Pydantic."""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union

class ChurnPredictionRequest(BaseModel):
    """Schema de requisição contendo as features originais do dataset Telco.

    Valida e tipa os atributos vindos do JSON de request para impedir Data Leakage
    e bugs por entrada malformada, com literais estritos para categóricas.

    Attributes:
        customerID (str): Identificador único do usuário.
        gender (Literal["Male", "Female"]): Gênero do cliente.
        SeniorCitizen (Literal[0, 1]): Indica se é idoso.
        Partner (Literal["Yes", "No"]): Possui parceiro.
        Dependents (Literal["Yes", "No"]): Possui dependentes.
        tenure (int): Meses de permanência na empresa.
        PhoneService (Literal["Yes", "No"]): Assina serviço de telefone.
        MultipleLines (Literal["Yes", "No", "No phone service"]): Múltiplas linhas telefônicas.
        InternetService (Literal["DSL", "Fiber optic", "No"]): Tipo de internet.
        OnlineSecurity (Literal["Yes", "No", "No internet service"]): Possui segurança online.
        OnlineBackup (Literal["Yes", "No", "No internet service"]): Possui backup online.
        DeviceProtection (Literal["Yes", "No", "No internet service"]): Possui proteção de aparelho.
        TechSupport (Literal["Yes", "No", "No internet service"]): Possui suporte técnico.
        StreamingTV (Literal["Yes", "No", "No internet service"]): Assina TV a cabo.
        StreamingMovies (Literal["Yes", "No", "No internet service"]): Assina filmes.
        Contract (Literal["Month-to-month", "One year", "Two year"]): Tipo de contrato.
        PaperlessBilling (Literal["Yes", "No"]): Fatura digital.
        PaymentMethod (Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]): Método de pagamento.
        MonthlyCharges (float): Cobrança mensal.
        TotalCharges (Union[float, str]): Cobrança total, que passará por coerção numérica.
    """
    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: Union[float, str]

    @field_validator('TotalCharges')
    @classmethod
    def coerce_total_charges(cls, v: Union[float, str]) -> float:
        """Coerce campos de TotalCharges vazios para 0.0 seguindo regra de EDA.

        Args:
            v (Union[float, str]): Valor de TotalCharges recebido no payload.

        Returns:
            float: O valor numérico formatado corretamente, garantindo float.

        Raises:
            ValueError: Se a string não puder ser convertida numérico real.
        """
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped:
                return 0.0
            try:
                return float(v_stripped)
            except ValueError:
                raise ValueError("TotalCharges must be a valid float string or empty.")
        return float(v)

class ChurnPredictionResponse(BaseModel):
    """Schema de resposta representando a inferência do modelo campeão.

    Attributes:
        churn_probability (float): A probabilidade (0.0 a 1.0) calculada pela rede neural via sigmoid.
        churn_prediction (Literal[0, 1]): A classe consolidada (1 = sim, 0 = não) baseada em um limiar estrito de 0.5.
    """
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: Literal[0, 1]