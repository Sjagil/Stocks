# Phase 4 Freeze Report

Status:

```text
IBKR_PHASE4_HISTORICAL_BARS_V1_FROZEN_GO
```

GO evidence:

```text
instrument_count       19
file_count             19
row_count              51457
duplicate_rows         0
invalid_ohlc_rows      0
timezone_errors        0
contract_mismatches    0
financial_calls        0
```

Source hashes:

```text
src/stocks/data/ibkr_historical.py                                             35AFE18C83323AA4EC14C51FFF61AD1B8F48B5251ABED5B62EE1A68D04CBD326
src/stocks/data/bars.py                                                        525F91270642345DBF0AC693E93B4C54B5E6A902EB2AC5B229F1E584A9BE4387
src/stocks/market/sessions.py                                                  E10E990EFD250A14A1E4E40F27D1D401D42B1ADFABFF57B83B2A880DC9D3E6D0
```

Contract cache hashes:

```text
output/ibkr/contracts/stocks.parquet                                           5166244EB2C665939478105AE17E2E4C18E706EABD02B2AB22C8E57C9F40EA67
output/ibkr/contracts/contract_manifest.json                                   0FEE541231382FACEEFA432465B17ED32CC89BBD928F50B6CDF21625B53693A9
output/ibkr/contracts/contract_requests.jsonl                                  0EBAEE7A6BBA92C50F96EEC688ABA3E570375361633F73F51DD8273579251283
output/ibkr/contracts/contract_errors.jsonl                                    A71D2A52D426756D215D46B115CA6AF6211639DD69B7680AFBA6759512202F9C
```

Session cache hashes:

```text
data/sessions/sessions.parquet                                                 C53E00094A1875F51C495AE448DE09CD59EA9E32EF5D4272B78B7B1B7E831ACE
data/sessions/session_manifest.json                                            C9C3DE7679B5B067B4E1DC693D6EE8FDCBA22206064D27FA146F0A77E4E02E09
data/sessions/session_conflicts.jsonl                                          7B2E15F355BCC1F9645BD27C34FDF298DE96CB59B4B75268F912ECB5F9614008
data/sessions/session_errors.jsonl                                             E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
```

Bar schema hashes:

```text
data/bars/bar_manifest.json                                                    43F6C2E09656D9F0501247D4F73F77B2D708141E24823ED000EC66D3FBB9E1C1
output/ibkr/bars/cache-validation.json                                         AFC0B7CCC817D758ECC75A3E5FB67D8750BD2601214EFF0B6408FD661B105E80
```

Error classification hashes:

```text
output/ibkr/bars/error-classification.json                                     71565A0A093F07D826EC6176EAC7CF92C01956D55AC116CE201502F524309D94
```

Bar data hashes:

```text
data/bars/security_type=STK/con_id=101484826/interval=1d/data_type=TRADES/bars.parquet D067E0275708A6A69E2AA27304437F39E68F18EF8604BA069ABAD46AF2E6670D
data/bars/security_type=STK/con_id=117589399/interval=1d/data_type=TRADES/bars.parquet B739A2E30081048CB436056A1D381EDA861CA4F2B8AE12379A491A21A9AF587B
data/bars/security_type=STK/con_id=15547841/interval=1d/data_type=TRADES/bars.parquet 2FE1808B8D32E21A47E2BCC0B9430A930372292D2194D20FFB45678E980F00F6
data/bars/security_type=STK/con_id=15547844/interval=1d/data_type=TRADES/bars.parquet 26F5DF8E3808A161FEC000CA6AD742321CF309851FBE9B286C5873DE76693A76
data/bars/security_type=STK/con_id=155518918/interval=1d/data_type=TRADES/bars.parquet 7304D01AFD88E798D44DE0FA9449089EA0E5B0EB79FDA978818B72DB411A67EB
data/bars/security_type=STK/con_id=253190534/interval=1d/data_type=TRADES/bars.parquet BC0B2F14DB6F14FCFBC2A8309C5A1C1250648058601FD4353DE1188A66331416
data/bars/security_type=STK/con_id=265598/interval=1d/data_type=TRADES/bars.parquet A80FC6CAD8589F929FDDDBE1F343A37B6708D4638CBE20C3317C720DD48B173C
data/bars/security_type=STK/con_id=296457239/interval=1d/data_type=TRADES/bars.parquet 97AAFBE3A9FB6CD02DEA2B3779AE1F6171DCB020551AB1E581A8D53398797CFD
data/bars/security_type=STK/con_id=319355208/interval=1d/data_type=TRADES/bars.parquet F85278E332B6508E847AAA87E88EBBB8E15C0F67D6929E491C3B7016F0A3CAB9
data/bars/security_type=STK/con_id=319355717/interval=1d/data_type=TRADES/bars.parquet 953EF256D32769AF638447D7EC253EB2B48D115B9D762A9126E84814BBB42E7A
data/bars/security_type=STK/con_id=319355727/interval=1d/data_type=TRADES/bars.parquet 691F88A56DF9AA6E02C3ED3379D77A3A33594A724530E1309F1F7F4C9F0EA1C2
data/bars/security_type=STK/con_id=320227571/interval=1d/data_type=TRADES/bars.parquet F4A10895F09BFF7ED811FA220176DE94D3EE3536EE799105A17C8C850609C479
data/bars/security_type=STK/con_id=39039301/interval=1d/data_type=TRADES/bars.parquet 097B78B7F5477832DFC0B2ECF67C7E6BD5374A6CC962AFD8AD2567CF8DDD11F5
data/bars/security_type=STK/con_id=49954154/interval=1d/data_type=TRADES/bars.parquet 29867D229F76C4E06345DFAF62C52C29C7D91CB8F937FC3AD8F4D202E38A747C
data/bars/security_type=STK/con_id=51529211/interval=1d/data_type=TRADES/bars.parquet 61C56728C73546D09F636C805B9131A0F0629FB751B9AC61507555E9E60D1DB7
data/bars/security_type=STK/con_id=6604766/interval=1d/data_type=TRADES/bars.parquet 1AEC62F3F62E0AB5C452CCD9E4972304E1CF372881090CB6AC6945925576C102
data/bars/security_type=STK/con_id=756733/interval=1d/data_type=TRADES/bars.parquet B04A824A11B34F6C03B819F8AE23DBDF2767F3BE2C627933AA1F27EB489B20C8
data/bars/security_type=STK/con_id=86326988/interval=1d/data_type=TRADES/bars.parquet E80B5BBCE7659F041642701DC89DDA076E973880EA48D3C4EB50EF414C451EDD
data/bars/security_type=STK/con_id=9579970/interval=1d/data_type=TRADES/bars.parquet F2B13BFCF633FA6D00302F14E0E732CBDE211DC29F06E9382D80F7611D44D25B
```

Phase 4 remains read-only. This report does not grant order authority.
