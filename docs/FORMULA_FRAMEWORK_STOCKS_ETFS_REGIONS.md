# Formula Framework: Stocks, ETFs and Regions

This document converts the stock, ETF and regional-market formula framework into implementation contracts. It is a research and validation layer, not an execution layer.

## Core Rules

```text
normalize before combining unlike signals
work point-in-time
compare within market, sector and region where needed
include FX, liquidity, tax drag and execution costs separately
use simple returns for capital accounting
use log returns for statistics and volatility models
do not activate IBKR data or order requests from this layer
```

Base currency is assumed to be EUR unless configuration later says otherwise.

## Returns

```text
PriceReturn_t = P_t / P_(t-1) - 1
TotalReturn_t = (P_t + D_t - P_(t-1)) / P_(t-1)
LogReturn_t = ln(P_t / P_(t-1))
```

FX-adjusted return:

```text
Return_EUR = (1 + Return_Local) * (1 + Return_FX) - 1
```

Hedged approximation:

```text
HedgedReturn = LocalAssetReturn + HedgeCarry - HedgeCosts
```

## Technical Formula Contracts

Trend and momentum formulas are pure research features:

```text
SMA_n = mean(P_t ... P_(t-n+1))
EMA_t = alpha * P_t + (1 - alpha) * EMA_(t-1)
Momentum_L = P_t / P_(t-L) - 1
RiskAdjustedMomentum = Momentum_L / Volatility_L
RelativeStrength = AssetReturn - BenchmarkReturn
TrendStrength = (EMA_fast - EMA_slow) / ATR
```

Volatility and gap contracts:

```text
TrueRange_t = max(
    High_t - Low_t,
    abs(High_t - Close_(t-1)),
    abs(Low_t - Close_(t-1))
)
ATR_n = EMA(TrueRange, n)
GapReturn_t = Open_t / Close_(t-1) - 1
```

Bollinger contracts:

```text
MiddleBand = SMA_n
UpperBand = SMA_n + k * sigma_n
LowerBand = SMA_n - k * sigma_n
BollingerZ = (P_t - SMA_n) / sigma_n
Bandwidth = (UpperBand - LowerBand) / MiddleBand
PercentB = (P_t - LowerBand) / (UpperBand - LowerBand)
```

These features must be calculated from data available on the decision timestamp. Breakout boundaries use completed historical bars only.

## Fundamental Formula Contracts

Valuation:

```text
MarketCap = SharePrice * SharesOutstanding
FreeFloatMarketCap = MarketCap * FreeFloatPercentage
EnterpriseValue = MarketCap + TotalDebt + PreferredEquity + MinorityInterest - Cash
PE = SharePrice / EPS
EarningsYield = NetIncome / MarketCap
PriceToBook = MarketCap / BookEquity
PriceToSales = MarketCap / Revenue
EVToEBITDA = EnterpriseValue / EBITDA
EVToEBIT = EnterpriseValue / EBIT
FCFYield = FreeCashFlow / EnterpriseValue
```

Shareholder return:

```text
NetDividendYield = DividendYield * (1 - WithholdingTaxRate)
PayoutRatio = Dividends / NetIncome
FCFPayoutRatio = Dividends / FreeCashFlow
BuybackYield = NetShareRepurchases / MarketCap
DebtReductionYield = (Debt_(t-1) - Debt_t) / MarketCap
ShareholderYield = DividendYield + BuybackYield + DebtReductionYield
```

Quality, growth and balance-sheet contracts:

```text
ROE = NetIncome / AverageBookEquity
ROA = NetIncome / AverageTotalAssets
ROIC = NOPAT / InvestedCapital
GrossProfitability = GrossProfit / TotalAssets
Margin = Numerator / Revenue
Growth = CurrentValue / PreviousValue - 1
CAGR = (CurrentValue / PreviousValue)^(1 / Years) - 1
DebtToEquity = TotalDebt / BookEquity
NetDebt = TotalDebt - Cash
NetDebtToEBITDA = NetDebt / EBITDA
InterestCoverage = EBIT / InterestExpense
CurrentRatio = CurrentAssets / CurrentLiabilities
QuickRatio = (Cash + MarketableSecurities + Receivables) / CurrentLiabilities
AssetTurnover = Revenue / AverageTotalAssets
InventoryTurnover = COGS / AverageInventory
Accruals = (NetIncome - OperatingCashFlow) / AverageTotalAssets
CashConversion = OperatingCashFlow / NetIncome
FCFConversion = FreeCashFlow / NetIncome
Dilution = SharesOutstanding_t / SharesOutstanding_(t-1) - 1
```

Revision contracts:

```text
EarningsSurprise = ActualEPS - ConsensusEPS
StandardizedSurprise = (ActualEPS - ConsensusEPS) / abs(ConsensusEPS)
RevisionPercentage = NewConsensusEPS / OldConsensusEPS - 1
RevisionBreadth = (UpRevisions - DownRevisions) / TotalRevisions
```

Fundamental values must be point-in-time before they are used in research. Required timing metadata for later data phases includes `period_end`, `filing_date`, `accepted_at`, `available_at` and restatement/version lineage where available.

## Regional Score Contract

```text
RegionScore_r =
    0.25 * TechnicalScore_r
  + 0.25 * FundamentalScore_r
  + 0.15 * EarningsRevisionScore_r
  + 0.10 * MacroScore_r
  + 0.10 * ValuationScore_r
  + 0.10 * CurrencyScore_r
  + 0.05 * LiquidityScore_r
```

Emerging markets add heavier penalties:

```text
EMRegionScore_r =
    BaseRegionScore_r
  - lambda_fx * FXVolatility_r
  - lambda_political * PoliticalRisk_r
  - lambda_external * ExternalVulnerability_r
  - lambda_liquidity * LiquidityPenalty_r
```

Regional adjustment helpers:

```text
CountryAdjustedMomentum = Momentum - MedianMomentum_country
SectorAdjustedValue = RawValue - MedianValue_sector
FXReturn = EURValueOfCurrency_t / EURValueOfCurrency_(t-1) - 1
CommodityBeta = Cov(Return_country, CommodityReturn) / Var(CommodityReturn)
USDBeta = Cov(Return_country, Return_USD) / Var(Return_USD)
ExternalRiskScore =
    CurrentAccountDeficit / GDP
  + ShortTermExternalDebt / FXReserves
  - FXReserves / MonthlyImports
```

## Stock Score Contract

Research starting point:

```text
CombinedStockScore_i =
    0.18 * z_region,sector(Momentum_12_1,i)
  + 0.10 * z_region,sector(RelativeStrength_i)
  + 0.07 * z_region,sector(TrendStrength_i)
  + 0.12 * z_region,sector(EarningsRevision_i)
  + 0.10 * z_region,sector(ROIC_i)
  + 0.08 * z_region,sector(FCFYield_i)
  + 0.07 * z_region,sector(GrossProfitability_i)
  + 0.06 * z_region,sector(RevenueGrowth_i)
  + 0.05 * z_region,sector(ShareholderYield_i)
  - 0.05 * z_region,sector(Accruals_i)
  - 0.04 * z_region,sector(Dilution_i)
  - 0.04 * z_region,sector(Volatility_i)
  - 0.02 * z_region,sector(LiquidityPenalty_i)
  + 0.04 * RegionScore_r
  - 0.02 * FXRisk_r
```

The weights are defaults for research bootstrapping, not proven optimal parameters.

## Eligibility Gates

Fundamental gate:

```text
ROIC > sector median
FCFYield > 0
InterestCoverage > minimum
Dilution < maximum
```

Technical gate:

```text
Close > SMA200
Momentum_12_1 > 0
RelativeStrength > 0
```

Combined:

```text
Eligible = FundamentalEligible AND TechnicalEligible
```

Fundamentals decide what may be owned. Technicals decide when it can be bought. The risk engine decides size.

## Weighting

Regional score to volatility-adjusted weight:

```text
RawWeight_r = max(RegionScore_r, 0) / ForecastVolatility_r
Weight_r = RawWeight_r / sum(RawWeight_r)
```

Effective volatility with FX:

```text
EffectiveVolatility =
    sqrt(equity_vol^2 + fx_vol^2 + 2 * rho_equity_fx * equity_vol * fx_vol)
```

Stock weight inside region:

```text
StockRawWeight_i = max(CombinedScore_i - Threshold, 0) / EffectiveVolatility_i
StockWeightWithinRegion_i = StockRawWeight_i / sum(StockRawWeight_j in region)
PortfolioWeight_i = RegionWeight_r * StockWeightWithinRegion_i
```

Caps remain mandatory:

```text
single stock cap
sector cap
region cap
emerging market cap
cluster cap
```

## Costs

```text
NetExpectedReturn =
    GrossExpectedReturn
  - Commission
  - Spread
  - Slippage
  - FXConversionCost
  - WithholdingTaxDrag
  - MarketImpact
```

Cost model:

```text
Cost =
    FixedCommission
  + SpreadBps * Notional / 10000
  + ImpactCoefficient * sqrt(Notional / ADV) * Notional
```

No formula in this document grants order authority.
