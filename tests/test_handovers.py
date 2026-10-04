"""跨盘点轮次的馆藏区域交接：链式交接、单事务核验、并发唯一、归并闭包查询、隔离性。"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

BASE = 1_700_000_000  # 2023-11-14T22:13:20Z


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from app.main import app

    with TestClient(app) as c:
        yield c, db_file


def make_exhibit(client: TestClient, eid: str, name: str | None = None) -> str:
    resp = client.post("/exhibits", json={
        "id": eid, "name": name or eid,
        "dose_limit_luxh": 1000.0, "max_interval_seconds": 3600.0,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def send_readings(client: TestClient, exhibit_id: str, readings: list[dict]) -> None:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text


def ev(exhibit_id: str, area: str, dt: float, rssi: float = -50) -> dict:
    return {"exhibit_id": exhibit_id, "area": area, "ts": iso(dt), "rssi": rssi}


def reconcile(client: TestClient, round_id: str, start: float,
              expected: list[tuple[str, str]], events: list[dict] | None = None,
              status: int = 200) -> dict:
    body = {
        "start_at": iso(start),
        "window_seconds": 60,
        "rssi_delta_threshold": 5,
        "expected": [{"exhibit_id": e, "area": a} for e, a in expected],
        "events": events if events is not None else [
            ev(e, a, start + 5) for e, a in expected
        ],
    }
    resp = client.post(f"/inventory-rounds/{round_id}/reconcile", json=body)
    assert resp.status_code == status, resp.text
    return resp.json()


def hb(key: str, from_round: str, to_round: str, from_area: str, to_area: str) -> dict:
    return {
        "request_key": key,
        "from_round_id": from_round,
        "to_round_id": to_round,
        "from_area": from_area,
        "to_area": to_area,
    }


def handover(client: TestClient, exhibit_id: str, body: dict, status: int = 200) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/zone-handovers", json=body)
    assert resp.status_code == status, resp.text
    return resp.json()


def history(client: TestClient, exhibit_id: str, status: int = 200) -> dict:
    resp = client.get(f"/exhibits/{exhibit_id}/zone-handovers")
    assert resp.status_code == status, resp.text
    return resp.json()


def merge(client: TestClient, source_id: str, target_id: str, status: int = 200) -> dict:
    token = client.post("/merges/preview",
                        json={"source_id": source_id, "target_id": target_id})
    assert token.status_code == 200, token.text
    resp = client.post("/merges/commit",
                       json={"token": token.json()["token"], "resolutions": []})
    assert resp.status_code == status, resp.text
    return resp.json()


# ---------- 基本交接与链式续接 ----------

def test_first_handover_builds_chain(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    r1 = reconcile(client, "R1", BASE, [("EX-1", "A区")])
    r2 = reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])

    out = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))
    assert out["already_committed"] is False
    assert out["requested_id"] == "EX-1" and out["canonical_id"] == "EX-1"
    assert out["from_round_id"] == "R1" and out["to_round_id"] == "R2"
    assert out["from_area"] == "A区" and out["to_area"] == "B区"
    assert out["prev_handover_id"] is None
    # 固化两轮摘要与起始时刻
    assert out["from_round_digest"] == r1["request_digest"]
    assert out["to_round_digest"] == r2["request_digest"]
    assert out["from_round_start_at"] == r1["start_at"]
    assert out["to_round_start_at"] == r2["start_at"]

    hist = history(client, "EX-1")
    assert hist["current"] == {
        "round_id": "R2", "area": "B区",
        "handover_id": out["handover_id"],
        "round_start_at": r2["start_at"],
    }
    assert [h["handover_id"] for h in hist["handovers"]] == [out["handover_id"]]
    # 历史条目是固化的交接记录（不含提交响应才有的 already_committed 标记）
    assert "already_committed" not in hist["handovers"][0]


def test_chain_continuation_and_history_sorted_by_round_time(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 7200, [("EX-1", "C区")])

    h1 = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))
    h2 = handover(client, "EX-1", hb("k2", "R2", "R3", "B区", "C区"))
    assert h2["prev_handover_id"] == h1["handover_id"]

    hist = history(client, "EX-1")
    assert [h["handover_id"] for h in hist["handovers"]] == [
        h1["handover_id"], h2["handover_id"],
    ]
    assert [h["to_round_id"] for h in hist["handovers"]] == ["R2", "R3"]
    assert hist["current"]["round_id"] == "R3"
    assert hist["current"]["area"] == "C区"
    assert hist["current"]["handover_id"] == h2["handover_id"]


def test_continuation_must_follow_tip_round_and_area(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 7200, [("EX-1", "C区")])
    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))

    # 来源轮次不是链尖接收轮次 → 409
    out = handover(client, "EX-1", hb("k2", "R1", "R3", "A区", "C区"), status=409)
    assert "链尖" in out["detail"]
    # 来源区域不是链尖接收区域 → 409
    handover(client, "EX-1", hb("k3", "R2", "R3", "A区", "C区"), status=409)
    # 旧轮次回填（R1→R2 已发生过）→ 409
    handover(client, "EX-1", hb("k4", "R1", "R2", "A区", "B区"), status=409)

    # 全部失败不留记录：链仍只有一笔，当前去向不变
    hist = history(client, "EX-1")
    assert len(hist["handovers"]) == 1
    assert hist["current"]["round_id"] == "R2"

    # 正确的续接仍然可行
    handover(client, "EX-1", hb("k5", "R2", "R3", "B区", "C区"))
    assert history(client, "EX-1")["current"]["round_id"] == "R3"


def test_backfill_cannot_overwrite_current_whereabouts(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 7200, [("EX-1", "B区")])
    handover(client, "EX-1", hb("k1", "R1", "R3", "A区", "B区"))

    # 试图在链尖之前补一笔 R1→R2：来源轮次不是链尖接收轮次 R3 → 409
    handover(client, "EX-1", hb("k2", "R1", "R2", "A区", "B区"), status=409)
    hist = history(client, "EX-1")
    assert len(hist["handovers"]) == 1
    assert hist["current"]["round_id"] == "R3"


def test_no_handover_yet_current_is_null(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    hist = history(client, "EX-1")
    assert hist["current"] is None
    assert hist["handovers"] == []
    assert hist["requested_id"] == "EX-1" and hist["canonical_id"] == "EX-1"


# ---------- 轮次前置条件 ----------

def test_rounds_must_be_solidified(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    # 接收轮次未固化 → 404
    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=404)
    # 来源轮次未固化 → 404
    handover(client, "EX-1", hb("k2", "R9", "R1", "A区", "B区"), status=404)
    assert history(client, "EX-1")["handovers"] == []


def test_round_start_times_must_increase(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 3600, [("EX-1", "C区")])  # 与 R2 同一起始时刻

    # 起始时间倒退 → 422
    handover(client, "EX-1", hb("k1", "R2", "R1", "B区", "A区"), status=422)
    # 起始时间相同（不递增）→ 422
    handover(client, "EX-1", hb("k2", "R2", "R3", "B区", "C区"), status=422)
    # 来源与接收同一轮次 → 422
    handover(client, "EX-1", hb("k3", "R1", "R1", "A区", "A区"), status=422)
    assert history(client, "EX-1")["handovers"] == []


def test_unknown_exhibit_404(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    handover(client, "ghost", hb("k1", "R1", "R2", "A区", "B区"), status=404)
    history(client, "ghost", status=404)


# ---------- 定位证据：缺失 / 错位 / 位置不明 / 未列入预期均不得交接 ----------

def test_exhibit_must_be_in_both_expected_lists(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    make_exhibit(client, "EX-2")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-2", "B区")])  # EX-1 不在 R2 预期清单
    out = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)
    assert "预期清单" in out["detail"]
    # 反向：EX-2 不在 R1 预期清单
    handover(client, "EX-2", hb("k2", "R1", "R2", "A区", "B区"), status=409)
    assert history(client, "EX-1")["handovers"] == []


def test_reported_area_must_match_expected_area(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    # 所报来源区域与轮次预期区域不符 → 409
    handover(client, "EX-1", hb("k1", "R1", "R2", "Z区", "B区"), status=409)
    # 所报接收区域与轮次预期区域不符 → 409
    handover(client, "EX-1", hb("k2", "R1", "R2", "A区", "Z区"), status=409)
    assert history(client, "EX-1")["handovers"] == []


def test_missing_forbids_handover(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    # R2 中 EX-1 在预期清单但全程无事件 → 缺失
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")], events=[])
    out = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)
    assert "缺失" in out["detail"]


def test_misplaced_forbids_handover(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    # R2 中 EX-1 被定位到 C区 而非预期的 B区 → 错位
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")],
              events=[ev("EX-1", "C区", BASE + 3605)])
    out = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)
    assert "错位" in out["detail"]


def test_unknown_position_forbids_handover(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    # R2 最后窗口 B/C 两区信号差 1 < 阈值 5 → 位置不明
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")],
              events=[ev("EX-1", "B区", BASE + 3605, -50),
                      ev("EX-1", "C区", BASE + 3606, -51)])
    out = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)
    assert "位置不明" in out["detail"]


def test_source_round_evidence_also_verified(env):
    """来源轮一侧缺失/错位/不明同样不得交接。"""
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")], events=[])  # 来源轮缺失
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)
    assert history(client, "EX-1")["handovers"] == []


# ---------- 幂等：request_key 重放与换目标 409 ----------

def test_request_key_replay_and_conflict(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 7200, [("EX-1", "C区")])

    first = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))
    replay = handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))
    assert replay["already_committed"] is True
    assert replay["handover_id"] == first["handover_id"]
    assert replay["created_at"] == first["created_at"]
    # 重放不产生新记录
    assert len(history(client, "EX-1")["handovers"]) == 1

    # 同 request_key 换目标（轮次/区域任一不同）→ 409
    handover(client, "EX-1", hb("k1", "R2", "R3", "B区", "C区"), status=409)
    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "Z区"), status=409)

    # 失败的交接不占用 request_key：修复后可重新提交
    make_exhibit(client, "EX-2")
    reconcile(client, "S1", BASE, [("EX-2", "A区")], events=[])
    reconcile(client, "S2", BASE + 3600, [("EX-2", "B区")])
    handover(client, "EX-2", hb("k9", "S1", "S2", "A区", "B区"), status=409)
    # 重新对账 S1 让 EX-2 定位正确（不同请求 → 新轮次 ID）
    reconcile(client, "S1b", BASE, [("EX-2", "A区")])
    ok = handover(client, "EX-2", hb("k9", "S1b", "S2", "A区", "B区"))
    assert ok["already_committed"] is False


# ---------- 并发：最多一笔生效 ----------

def test_concurrent_different_keys_at_most_one_commits(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])

    def post(i: int) -> tuple[int, dict]:
        r = client.post("/exhibits/EX-1/zone-handovers",
                        json=hb(f"k{i}", "R1", "R2", "A区", "B区"))
        return r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(post, range(12)))
    committed = [d for s, d in results if s == 200 and not d["already_committed"]]
    conflicts = [s for s, _ in results if s == 409]
    # 并发交接最多一笔生效，其余全部 409（链尖已被占用）
    assert len(committed) == 1
    assert len(conflicts) == 11
    hist = history(client, "EX-1")
    assert len(hist["handovers"]) == 1
    assert hist["handovers"][0]["handover_id"] == committed[0]["handover_id"]


def test_concurrent_same_key_replays(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])

    def post(_: int) -> tuple[int, dict]:
        r = client.post("/exhibits/EX-1/zone-handovers",
                        json=hb("same-key", "R1", "R2", "A区", "B区"))
        return r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(post, range(12)))
    assert all(s == 200 for s, _ in results)
    committed = [d for _, d in results if not d["already_committed"]]
    replayed = [d for _, d in results if d["already_committed"]]
    assert len(committed) == 1
    assert len(replayed) == 11
    assert {d["handover_id"] for _, d in results} == {committed[0]["handover_id"]}
    assert len(history(client, "EX-1")["handovers"]) == 1


def test_concurrent_continuations_at_most_one_commits(env):
    """链已存在时并发续接：同样最多一笔生效。"""
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    reconcile(client, "R3", BASE + 7200, [("EX-1", "C区")])
    handover(client, "EX-1", hb("k0", "R1", "R2", "A区", "B区"))

    def post(i: int) -> tuple[int, dict]:
        r = client.post("/exhibits/EX-1/zone-handovers",
                        json=hb(f"c{i}", "R2", "R3", "B区", "C区"))
        return r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(post, range(10)))
    committed = [d for s, d in results if s == 200 and not d["already_committed"]]
    assert len(committed) == 1
    assert sum(1 for s, _ in results if s == 409) == 9
    hist = history(client, "EX-1")
    assert len(hist["handovers"]) == 2
    assert hist["current"]["round_id"] == "R3"


# ---------- 别名与归并 ----------

def test_handover_via_alias_and_replay_after_merge(env):
    client, _ = env
    make_exhibit(client, "src")
    make_exhibit(client, "tgt")
    send_readings(client, "src", [{"id": "r1", "ts": iso(BASE), "lux": 1}])
    merge(client, "src", "tgt")

    # 归并后的轮次：预期清单用别名 src，解析到规范档案 tgt
    reconcile(client, "R1", BASE, [("src", "A区")])
    reconcile(client, "R2", BASE + 3600, [("src", "B区")])

    # 用别名提交交接：解析到规范档案 tgt
    out = handover(client, "src", hb("k1", "R1", "R2", "A区", "B区"))
    assert out["requested_id"] == "src"
    assert out["canonical_id"] == "tgt"

    # 用目标 ID 重放同一 request_key → 命中同一笔
    replay = handover(client, "tgt", hb("k1", "R1", "R2", "A区", "B区"))
    assert replay["already_committed"] is True
    assert replay["handover_id"] == out["handover_id"]

    # 历史用别名或目标 ID 查询一致（仅 requested_id 随请求写法不同）
    hist_alias = history(client, "src")
    hist_target = history(client, "tgt")
    assert hist_alias["requested_id"] == "src"
    assert hist_target["requested_id"] == "tgt"
    assert {k: v for k, v in hist_alias.items() if k != "requested_id"} == \
        {k: v for k, v in hist_target.items() if k != "requested_id"}


def test_merge_keeps_one_sided_chain_queryable_by_target_and_alias(env):
    client, _ = env
    make_exhibit(client, "EX-S")
    make_exhibit(client, "EX-T")
    reconcile(client, "R1", BASE, [("EX-S", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-S", "B区")])
    h1 = handover(client, "EX-S", hb("k1", "R1", "R2", "A区", "B区"))

    # 仅来源侧有交接链：归并照常进行
    merge(client, "EX-S", "EX-T")

    # 归并后单侧已有的交接链仍可用目标 ID 或别名查询
    for eid in ("EX-T", "EX-S"):
        hist = history(client, eid)
        assert hist["canonical_id"] == "EX-T"
        assert [h["handover_id"] for h in hist["handovers"]] == [h1["handover_id"]]
        assert hist["current"]["round_id"] == "R2"
        assert hist["current"]["area"] == "B区"
        # 历史记录保持写入时的规范档案 ID，不改写
        assert hist["handovers"][0]["canonical_id"] == "EX-S"

    # 归并后用别名重放同一 request_key 仍命中原记录
    replay = handover(client, "EX-S", hb("k1", "R1", "R2", "A区", "B区"))
    assert replay["already_committed"] is True
    assert replay["handover_id"] == h1["handover_id"]


def test_merge_with_chains_on_both_sides_rejected(env):
    client, _ = env
    make_exhibit(client, "EX-S")
    make_exhibit(client, "EX-T")
    reconcile(client, "R1", BASE, [("EX-S", "A区"), ("EX-T", "M区")])
    reconcile(client, "R2", BASE + 3600, [("EX-S", "B区"), ("EX-T", "N区")])
    handover(client, "EX-S", hb("ks", "R1", "R2", "A区", "B区"))
    handover(client, "EX-T", hb("kt", "R1", "R2", "M区", "N区"))

    # 两侧均有交接链 → 归并 409 拒绝
    token = client.post("/merges/preview",
                        json={"source_id": "EX-S", "target_id": "EX-T"})
    assert token.status_code == 200, token.text
    resp = client.post("/merges/commit",
                       json={"token": token.json()["token"], "resolutions": []})
    assert resp.status_code == 409, resp.text
    assert "交接链" in resp.json()["detail"]

    # 不改动任何记录：两档案仍在、别名未建立、两条链原样
    assert client.get("/exhibits/EX-S").status_code == 200
    assert client.get("/exhibits/EX-T").status_code == 200
    assert client.get("/exhibits/EX-S").json()["canonical_id"] == "EX-S"
    assert len(history(client, "EX-S")["handovers"]) == 1
    assert len(history(client, "EX-T")["handovers"]) == 1
    assert history(client, "EX-S")["current"]["area"] == "B区"
    assert history(client, "EX-T")["current"]["area"] == "N区"


def test_merge_rejected_by_both_sided_chains_leaves_readings_untouched(env):
    client, _ = env
    make_exhibit(client, "EX-S")
    make_exhibit(client, "EX-T")
    send_readings(client, "EX-S", [{"id": "r1", "ts": iso(BASE), "lux": 10}])
    send_readings(client, "EX-T", [{"id": "r9", "ts": iso(BASE), "lux": 20}])
    reconcile(client, "R1", BASE, [("EX-S", "A区"), ("EX-T", "M区")])
    reconcile(client, "R2", BASE + 3600, [("EX-S", "B区"), ("EX-T", "N区")])
    handover(client, "EX-S", hb("ks", "R1", "R2", "A区", "B区"))
    handover(client, "EX-T", hb("kt", "R1", "R2", "M区", "N区"))

    readings_s = client.get("/exhibits/EX-S/readings").json()
    readings_t = client.get("/exhibits/EX-T/readings").json()
    merge(client, "EX-S", "EX-T", status=409)
    assert client.get("/exhibits/EX-S/readings").json() == readings_s
    assert client.get("/exhibits/EX-T/readings").json() == readings_t


def test_chain_tip_spans_alias_closure_after_merge(env):
    """归并后链尖查询跨别名闭包：续接仍须从既有链的链尖出发。"""
    client, _ = env
    make_exhibit(client, "EX-S")
    make_exhibit(client, "EX-T")
    reconcile(client, "R1", BASE, [("EX-S", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-S", "B区")])
    handover(client, "EX-S", hb("k1", "R1", "R2", "A区", "B区"))
    merge(client, "EX-S", "EX-T")

    # 归并后新轮次用目标 ID 对账；续接从链尖（R2, B区）出发，用别名提交
    reconcile(client, "R3", BASE + 7200, [("EX-T", "C区")])
    # 不从链尖出发 → 409（链尖查询跨闭包找到了 R2/B区）
    handover(client, "EX-T", hb("k2", "R1", "R3", "A区", "C区"), status=409)
    # 从链尖续接 → 200，新记录挂在当前规范档案下
    h2 = handover(client, "EX-S", hb("k3", "R2", "R3", "B区", "C区"))
    assert h2["canonical_id"] == "EX-T"
    assert h2["requested_id"] == "EX-S"
    assert h2["prev_handover_id"] is not None
    hist = history(client, "EX-T")
    assert len(hist["handovers"]) == 2
    assert hist["current"]["round_id"] == "R3"
    assert hist["current"]["area"] == "C区"


# ---------- 隔离性：不改盘点结果、读数与收据 ----------

def test_handover_does_not_modify_rounds_readings_or_receipts(env):
    client, db_file = env
    make_exhibit(client, "EX-1")
    send_readings(client, "EX-1", [
        {"id": "r1", "ts": iso(BASE), "lux": 100},
        {"id": "r2", "ts": iso(BASE + 1800), "lux": 120},
    ])
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])

    round1_before = client.get("/inventory-rounds/R1").json()
    round2_before = client.get("/inventory-rounds/R2").json()
    readings_before = client.get("/exhibits/EX-1/readings").json()
    receipts_before = client.get("/exhibits/EX-1/receipts").json()

    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))
    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"))  # 重放也无副作用

    assert client.get("/inventory-rounds/R1").json() == round1_before
    assert client.get("/inventory-rounds/R2").json() == round2_before
    assert client.get("/exhibits/EX-1/readings").json() == readings_before
    assert client.get("/exhibits/EX-1/receipts").json() == receipts_before
    verify = client.get("/exhibits/EX-1/receipts/verify").json()
    assert verify["ok"] is True

    # 交接只写 zone_handovers 表
    conn = sqlite3.connect(str(db_file))
    try:
        assert conn.execute("SELECT COUNT(*) FROM zone_handovers").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM inventory_rounds").fetchone()[0] == 2
    finally:
        conn.close()


def test_failed_handover_leaves_no_partial_record(env):
    """各类失败都不留下半笔记录（随后 GET 仍为空、request_key 不占用）。"""
    client, db_file = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")], events=[])  # EX-1 缺失

    handover(client, "EX-1", hb("k1", "R1", "R2", "A区", "B区"), status=409)  # 缺失
    handover(client, "EX-1", hb("k2", "R2", "R1", "B区", "A区"), status=422)  # 时间倒退
    handover(client, "EX-1", hb("k3", "R1", "R9", "A区", "B区"), status=404)  # 轮次未固化

    hist = history(client, "EX-1")
    assert hist["handovers"] == [] and hist["current"] is None
    conn = sqlite3.connect(str(db_file))
    try:
        assert conn.execute("SELECT COUNT(*) FROM zone_handovers").fetchone()[0] == 0
    finally:
        conn.close()


# ---------- 参数校验 ----------

def test_request_validation_422(env):
    client, _ = env
    make_exhibit(client, "EX-1")
    reconcile(client, "R1", BASE, [("EX-1", "A区")])
    reconcile(client, "R2", BASE + 3600, [("EX-1", "B区")])
    assert client.post("/exhibits/EX-1/zone-handovers",
                       json=hb("", "R1", "R2", "A区", "B区")).status_code == 422
    assert client.post("/exhibits/EX-1/zone-handovers",
                       json=hb("k", "", "R2", "A区", "B区")).status_code == 422
    assert client.post("/exhibits/EX-1/zone-handovers",
                       json=hb("k", "R1", "R2", "", "B区")).status_code == 422
    body = hb("k", "R1", "R2", "A区", "B区")
    del body["request_key"]
    assert client.post("/exhibits/EX-1/zone-handovers", json=body).status_code == 422
