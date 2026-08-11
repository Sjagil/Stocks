"""Research-only, multi-asset quant platform.

The package deliberately has no dependency on broker execution modules.  Its
public contracts are shared by the progressively more advanced research
levels, starting with canonical market data and descriptive risk analytics.
"""

from stocks.quant_platform.analytics import PerformanceRiskAnalyzer
from stocks.quant_platform.allocation import (
    BlackLittermanAllocator,
    DynamicMultiAssetAllocator,
    HierarchicalRiskParity,
    PortfolioExposureRiskEngine,
)
from stocks.quant_platform.backtest import (
    BacktestConfig,
    BacktestOrder,
    OrderSide,
    OrderType,
    ProfessionalBacktestEngine,
    orders_from_target_positions,
)
from stocks.quant_platform.capabilities import CAPABILITIES, capability_registry
from stocks.quant_platform.data import (
    CANONICAL_MARKET_DATA_COLUMNS,
    AssetClass,
    CanonicalMarketData,
    clean_market_data,
    resample_market_data,
)
from stocks.quant_platform.explorer import MultiAssetMarketDataExplorer
from stocks.quant_platform.execution import (
    OptimalExecutionEngine,
    TransactionCostModel,
    execution_shortfall,
)
from stocks.quant_platform.factors import (
    DEFAULT_FACTOR_SPECS,
    CrossSectionalFactorEngine,
    FactorSpec,
    build_market_factor_snapshot,
)
from stocks.quant_platform.ml import (
    MetaLabelingEngine,
    ProbabilityCalibratedSignalEngine,
    ReturnPredictionModel,
    TemporalConvolutionalReturnModel,
    build_economic_return_target,
)
from stocks.quant_platform.manager import FullQuantPortfolioManager, manager_feedback
from stocks.quant_platform.intelligence import (
    NewsIntelligenceEngine,
    SecFilingIntelligenceEngine,
    event_study,
)
from stocks.quant_platform.portfolio import PortfolioOptimizer, PortfolioSolution
from stocks.quant_platform.professional import (
    AlphaCombinationEngine,
    ConstrainedQPortfolioPolicy,
    FactorRiskModel,
    PortfolioRLEnvironment,
    RegimeConditionalMixtureOfExperts,
    RewardWeights,
)
from stocks.quant_platform.regime import (
    HiddenMarkovRegimeDetector,
    RuleBasedRegimeDetector,
    StatisticalRegimeDetector,
    StrategyAllocationEngine,
)
from stocks.quant_platform.risk import (
    MonteCarloPortfolioSimulator,
    PortfolioTailRiskEngine,
    VolatilityModelEngine,
)
from stocks.quant_platform.stat_arb import StatisticalArbitrageEngine
from stocks.quant_platform.providers import (
    BitvavoAdapter,
    CoinMarketCapAdapter,
    EodhdAdapter,
    FredAdapter,
    OpenExchangeRatesAdapter,
)
from stocks.quant_platform.storage import MultiAssetStore
from stocks.quant_platform.technical import TechnicalStrategyLab, strategy_signals, technical_features
from stocks.quant_platform.validation import WalkForwardValidator, cost_stress, parameter_sensitivity, stability_report

__all__ = [
    "CANONICAL_MARKET_DATA_COLUMNS",
    "AssetClass",
    "AlphaCombinationEngine",
    "BacktestConfig",
    "BacktestOrder",
    "BlackLittermanAllocator",
    "CAPABILITIES",
    "BitvavoAdapter",
    "CanonicalMarketData",
    "CoinMarketCapAdapter",
    "ConstrainedQPortfolioPolicy",
    "CrossSectionalFactorEngine",
    "DEFAULT_FACTOR_SPECS",
    "EodhdAdapter",
    "DynamicMultiAssetAllocator",
    "FredAdapter",
    "FullQuantPortfolioManager",
    "FactorSpec",
    "FactorRiskModel",
    "MultiAssetMarketDataExplorer",
    "MultiAssetStore",
    "MetaLabelingEngine",
    "MonteCarloPortfolioSimulator",
    "NewsIntelligenceEngine",
    "HierarchicalRiskParity",
    "HiddenMarkovRegimeDetector",
    "OpenExchangeRatesAdapter",
    "OptimalExecutionEngine",
    "OrderSide",
    "OrderType",
    "PerformanceRiskAnalyzer",
    "PortfolioOptimizer",
    "PortfolioExposureRiskEngine",
    "PortfolioRLEnvironment",
    "PortfolioSolution",
    "PortfolioTailRiskEngine",
    "ProbabilityCalibratedSignalEngine",
    "ProfessionalBacktestEngine",
    "RuleBasedRegimeDetector",
    "ReturnPredictionModel",
    "TemporalConvolutionalReturnModel",
    "RegimeConditionalMixtureOfExperts",
    "RewardWeights",
    "StatisticalRegimeDetector",
    "StatisticalArbitrageEngine",
    "SecFilingIntelligenceEngine",
    "StrategyAllocationEngine",
    "TechnicalStrategyLab",
    "TransactionCostModel",
    "WalkForwardValidator",
    "VolatilityModelEngine",
    "build_market_factor_snapshot",
    "build_economic_return_target",
    "capability_registry",
    "cost_stress",
    "event_study",
    "execution_shortfall",
    "manager_feedback",
    "parameter_sensitivity",
    "orders_from_target_positions",
    "clean_market_data",
    "resample_market_data",
    "strategy_signals",
    "stability_report",
    "technical_features",
]
