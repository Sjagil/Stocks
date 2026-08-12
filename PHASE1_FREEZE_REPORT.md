# Phase 1 Freeze Report

Status:

```text
IBKR_PHASE_1_READ_ONLY_CONNECTION_SERVICE_GO
PHASE1_CONNECTION_SERVICE_FROZEN_GO
```

Verified artifact:

```text
C:\Users\alhar\Documents\Stocks\output\ibkr\phase1-disconnect-drill-20260720-133116.json
```

Artifact SHA256:

```text
D91691AB2A053B61FD0DAA7ED5C5DB6729EBC6A67306ACFEA3976A5FFE22DDD9
```

Evidence:

```text
schema                  ibkr_forced_disconnect_drill_v1
status                  GO
host                    127.0.0.1
port                    7497
client_id               17
disconnect_observed     True
reconnect_successful    True
place_order             0
cancel_order            0
global_cancel           0
```

Frozen Phase 0 hashes:

```text
ibkr_tws_probe.py       CA5B20533C1B6FBE5F0405F2DB94C0E0DD37BFEB16CF19746C613434D1F41F27
requirements.lock.txt   AC7AAE517FED4FF220CF7C424A76874EA0577E11ADF15C553747532D73C0A162
```

Immutable Phase 1 service hashes:

```text
src/stocks/application/config.py   ED69FC59B6AA8597B6F0B54E6F1765CB7BA09A0FF63AB2746CAF8431FFAA2058
src/stocks/ibkr/connection.py      B39184B9E09EB674D71AF35423DCF5A136BF8B9847A9FC6F527242504C34340B
src/stocks/ibkr/client.py          10ED7752FF880A994EC3F219AA2F39CB17D35A6A3D58D45A171949F5BC28658C
src/stocks/ibkr/callbacks.py       B91CDABA6B65A61438185852D35F6FCF292A4E939792B1525DDCD0E6DECFE990
src/stocks/ibkr/errors.py          55D8ADCA07EC1B77FE8289D12D90BB04D1CA7C4A64DBA16746F60F12E34438A2
src/stocks/ibkr/health.py          AC303071D30BB1FB08FAC11390AE2B17EDE91ACDA4FD8268DEB7F675D904B321
```

Mutable application entrypoint hash:

```text
main.py                            B12F5A5E10F120350102107256ED86CA818C6DD032D931241CAEDECFD79881C5
```

Phase 1 remains read-only. This report does not grant order authority.

Dependency-only revalidation on 2026-08-11 added the RL shadow research
runtime (`torch`, `gymnasium`, `stable-baselines3`, `sb3-contrib` and their
transitive packages). The immutable Phase 1 service hashes and the verified
disconnect/reconnect drill are unchanged. The complete product suite passed
1,900 tests, Ruff, compileall and diff checks before this hash was updated.
RL execution authority and broker writes remain `NONE` and `0`.
