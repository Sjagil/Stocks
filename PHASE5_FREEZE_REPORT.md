# Phase 5 Freeze Report

Status:

```text
IBKR_PHASE5_TOTAL_RETURN_AND_FX_FROZEN_GO
```

Status artifact:

```text
output/ibkr/phase5-total-return-fx-status.json
9B7709A5DBB8C4ECF5DDF35E0D97AD216EC8F0352192AA9C16B5B86A6EA7FA22
```

Source hashes:

```text
main.py                                                                3FD3844A1DEFE60079B6AED061BE633963C0B2575444258CBC8BFE33140A3907
src/stocks/data/phase5_common.py                                       4F85C2D4653F7A23BE728FA86CF9902AC718274868DFD0D6BC902B9373181573
src/stocks/data/corporate_actions.py                                   11352EE4E4CCC0CED09CCCB98A1DCEB2B7566758E08ECFF4087C03142E11BBC5
src/stocks/data/fx.py                                                  9CEE6BB5AD7434724B0244C4D2F0C8C819AA1017FBEBC8AA23475815791E68DF
src/stocks/data/total_returns.py                                       DFA537D203AE49AAA455C88F5BDD178703E7CDED7DF99D907F52059BBA88CE79
tests/test_phase5_total_return.py                                      F1507BE1C0C962521E4B20D32C4A36D766B13F1EA2FD6B1872623D1D864C6334
```

Data hashes:

```text
data/corporate_actions/corporate_actions.parquet                                           CBB93211A4AEFFF04B26095E9487976261E6AA79747756927FC15F8031583E60
data/corporate_actions/dividends.parquet                                                   0B97577CA7795CB2FC151FA9FA071779DCF0CE87AD205AB937F4417F2526BE3F
data/corporate_actions/splits.parquet                                                      E86F4CD4D5AFF00CC9CFCCCDE83AEB95831E811FB267315B5205996D49DB8254
data/corporate_actions/event_manifest.json                                                 B25649B50628588D5ACB58E28518270E1E61CC783C37B593A9C5ED059BE96C1D
data/fx/fx_daily.parquet                                                                   60E838DF8D6340533A1BC28C202D2DF3A4A3308E7BFE5DDEE4E9125712A7B988
data/fx/fx_manifest.json                                                                   AFC6DD345AFF83E8887894F764B2EFB826BA8AF2E279E9F2DF334F9721865F99
data/total_returns/total_return_manifest.json                                              0A9E7B4230C9A046736FB2266F15E42A16F3659D1E8F389DE88B1C9AA83A13D9
data/total_returns/security_type=STK/con_id=101484826/interval=1d/total_returns.parquet    F1E146E8F1D665E89FA36B7106B1D4DF39E03EEF3DF23EE68791C44B5F0EA6E6
data/total_returns/security_type=STK/con_id=15547841/interval=1d/total_returns.parquet     D15D5E6AC9C832A3EF625BEDB95EE2F8481312C643AC5EF50F9CAD1434945ED8
data/total_returns/security_type=STK/con_id=15547844/interval=1d/total_returns.parquet     F3B7660E178FB413B021E9EDC1A71A06D10A36394982D95877B66E17705FC150
data/total_returns/security_type=STK/con_id=155518918/interval=1d/total_returns.parquet    47697A13D524FBFD0D8FDB4326065A333668840BC8483D60C81333FA78B2AB27
data/total_returns/security_type=STK/con_id=253190534/interval=1d/total_returns.parquet    7B708EF8A20273FE222F00D9971B112D2DD42461411CD92FC0A0D93353E84D1E
data/total_returns/security_type=STK/con_id=296457239/interval=1d/total_returns.parquet    14C09E6E58A35A7FDE65270AA088B05E81FC1D6CDBDAFF32D8492408AAF3E46F
data/total_returns/security_type=STK/con_id=319355208/interval=1d/total_returns.parquet    79FF84D728297FAE71FD70AF50E3EEB0BDF395AB914AF2A2DA42B62A6E56CEB9
data/total_returns/security_type=STK/con_id=319355717/interval=1d/total_returns.parquet    BB8A07CD32E896732470801A9EB87380976227538F23259E4A42AD9BF26B589E
data/total_returns/security_type=STK/con_id=319355727/interval=1d/total_returns.parquet    9B9B29B124DC7D5908B51F489261A8700021180EC76E5A34F202E63E3E909B30
data/total_returns/security_type=STK/con_id=320227571/interval=1d/total_returns.parquet    FC3828FECD2738D85B043AE84CB5199D090BAD3D0A1F67AEDDDD36CD5364E9BD
data/total_returns/security_type=STK/con_id=39039301/interval=1d/total_returns.parquet     1C76847485AD0FAD11503415828716A0887D883720EFEF121F38B5506B2E1423
data/total_returns/security_type=STK/con_id=49954154/interval=1d/total_returns.parquet     A7F379C65B9F694C712CC460DFB3EF39F55E1E7E39E5FA02A67323011B883DE3
data/total_returns/security_type=STK/con_id=51529211/interval=1d/total_returns.parquet     36272CB5D60C0DFE1A5D70BAA2623169FEF3AB4E880E3C981D7B7496F9E0AEAA
data/total_returns/security_type=STK/con_id=6604766/interval=1d/total_returns.parquet      362C8DED337361355FD7B887CC596FC0980A328E42EF0E36BDAD48C916C5D369
data/total_returns/security_type=STK/con_id=756733/interval=1d/total_returns.parquet       BCDF2C8A25F1FD4FCA0C39350183062254A15F0EF8BAEC801569698451053C27
data/total_returns/security_type=STK/con_id=86326988/interval=1d/total_returns.parquet     E5838F54079B26471530CC0EE92454C239644E47ABA4EE29C7A6D04D6236125B
data/total_returns/security_type=STK/con_id=9579970/interval=1d/total_returns.parquet      2DA5132FE9D54DA185B8E38BD796EDE2567C3D61B6BB372E7331BC33943F1338
```

Phase 5 remains read-only. This report does not grant order authority.
