"""跨盘点轮次的馆藏区域交接：链式交接、单事务核验固化、归并别名闭包查询。

设计要点：
- **交接**：提交展品 ID（可用归并别名）、来源与接收 ``round_id`` 及两端区域。
  服务在**一个 SQLite 写事务**（``BEGIN IMMEDIATE``）内完成全部核验并落库：
  两轮均已固化、来源轮次起始时刻**早于**接收轮次（起始时间递增）、展品列在
  两轮预期清单中且在各自轮次**明确定位于所报区域**（缺失 / 错位 / 位置不明 /
  不在预期清单均不得交接）、以及链式续接核验。任一失败整体回滚，
  **不留下半笔记录**；核验与写入同事务加上写锁串行化，**并发交接最多一笔
  生效**（竞争方在锁内重读链尖后得到 409 或按 request_key 回放）。
- **链式续接**：首次交接以来源轮建链（``prev_handover_id`` 为空）；此后每笔
  交接的（来源轮次, 来源区域）必须等于链尖的（接收轮次, 接收区域）——只能从
  上一笔的接收轮和区域续接，**旧轮次不能回填覆盖当前去向**。
- **固化**：每笔交接记录两轮的固化摘要（``request_digest``）与起始时刻，
  记录落库后绝不改写；``request_key`` 为幂等键，同键同目标重放首次结果
  （``already_committed=true``），换目标 → 409。
- **查询**：按展品给出**当前区域**（链尖接收轮次与区域）与**按轮次时间
  排序**的交接历史。交接记录按写入时的规范档案 ID 落库、归并后不改写；
  查询经**别名闭包**（规范 ID + 全部别名）取链，因此归并后单侧已有的
  交接链仍可用目标 ID 或别名查询；核验轮次证据时同样把轮次内的档案 ID
  经当前别名表解析（归并不改变展品的物理身份），交接链在归并后仍可
  续接；两侧均有交接链时归并被拒绝（见 merge.py，且不改动任何记录）。
- 本模块只读写 ``zone_handovers``：不改写盘点轮次、照度读数与存证收据链。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from .inventory import load_round
from .merge import aliases_of, resolve_canonical
from .schemas import (
    InventoryRoundOut,
    ZoneCurrentOut,
    ZoneHandoverCommitOut,
    ZoneHandoverHistoryOut,
    ZoneHandoverOut,
    ZoneHandoverRequest,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


# ---------- 链读取（别名闭包） ----------

def _chain_rows(conn: sqlite3.Connection, canonical_id: str) -> list[sqlite3.Row]:
    """取出该规范档案别名闭包内的全部交接记录，按轮次时间升序。

    记录按写入时的规范档案 ID 落库、归并后不改写，因此一条逻辑交接链可能
    分散在历次归并前的规范 ID 下；闭包 = 规范 ID + 全部别名（别名表始终
    只有一跳，闭包即该档案的全部历史身份）。链式续接保证接收轮次起始时刻
    沿链严格递增，故按轮次时间排序即链接顺序。
    """
    ids = [canonical_id, *aliases_of(conn, canonical_id)]
    marks = ",".join("?" for _ in ids)
    return conn.execute(
        f"SELECT * FROM zone_handovers WHERE canonical_id IN ({marks}) "
        "ORDER BY to_round_start_s, from_round_start_s, handover_id",
        ids,
    ).fetchall()


def _chain_tip(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    """链尖：没有被任何后续交接引用为 prev 的那一笔；空链返回 None。"""
    referenced = {r["prev_handover_id"] for r in rows if r["prev_handover_id"] is not None}
    tips = [r for r in rows if r["handover_id"] not in referenced]
    if not tips:
        return None
    # 链式续接保证链尖唯一；防御性地取轮次时间最晚者
    return max(
        tips,
        key=lambda r: (r["to_round_start_s"], r["from_round_start_s"], r["handover_id"]),
    )


# ---------- 核验 ----------

def _current_canonical(conn: sqlite3.Connection, canonical_id: str) -> str:
    """轮次固化内容记录的是对账时的规范档案 ID；归并后同一展品的历史身份
    经别名解析到当前规范档案（档案只会因归并而消失，解析不到时返回原 ID）。"""
    return resolve_canonical(conn, canonical_id) or canonical_id


def _verify_round_location(
    conn: sqlite3.Connection,
    round_out: InventoryRoundOut,
    canonical_id: str,
    area: str,
    label: str,
) -> None:
    """核验展品在某轮次中列于预期清单且明确定位于所报区域；不满足 → 409。

    缺失 / 错位 / 位置不明 / 不在预期清单（含意外出现）均不得交接。
    轮次内的档案 ID 先经当前别名表解析（归并不改变展品的物理身份），
    与链读取的别名闭包口径一致：归并后交接链仍可续接。
    """
    rid = round_out.round_id
    # 预期区域按当前规范档案归集；归并后多条历史预期指向同一档案且区域
    # 不一致时视为无法确认（None），不得交接。
    expected: dict[str, str | None] = {}
    for e in round_out.expectations:
        current = _current_canonical(conn, e.canonical_id)
        if current not in expected:
            expected[current] = e.expected_area
        elif expected[current] != e.expected_area:
            expected[current] = None
    if canonical_id not in expected:
        raise HTTPException(
            status_code=409,
            detail=f"展品未列在{label}轮次 {rid} 的预期清单中，不得交接",
        )
    if expected[canonical_id] is None:
        raise HTTPException(
            status_code=409,
            detail=f"展品在{label}轮次 {rid} 中有多条历史预期且区域不一致"
                   "（归并后指向同一规范档案），无法确认预期区域，不得交接",
        )
    if expected[canonical_id] != area:
        raise HTTPException(
            status_code=409,
            detail=f"展品在{label}轮次 {rid} 的预期区域为 {expected[canonical_id]}，"
                   f"与所报区域 {area} 不符，不得交接",
        )
    for m in round_out.missing:
        if _current_canonical(conn, m.canonical_id) == canonical_id:
            raise HTTPException(
                status_code=409,
                detail=f"展品在{label}轮次 {rid} 中缺失"
                       f"（预期区域 {m.expected_area} 全程无有效读头事件），不得交接",
            )
    for m in round_out.misplaced:
        if _current_canonical(conn, m.canonical_id) == canonical_id:
            raise HTTPException(
                status_code=409,
                detail=f"展品在{label}轮次 {rid} 中错位"
                       f"（预期 {m.expected_area}，实际定位 {m.actual_area}），不得交接",
            )
    for u in round_out.unknown_position:
        if _current_canonical(conn, u.canonical_id) == canonical_id:
            raise HTTPException(
                status_code=409,
                detail=f"展品在{label}轮次 {rid} 中位置不明"
                       "（最后窗口多区域信号差不足阈值），不得交接",
            )


def _check_replay(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    requested_id: str,
    payload: ZoneHandoverRequest,
) -> ZoneHandoverCommitOut:
    """相同 request_key 重放：目标（规范档案/轮次对/两端区域）一致则回放，否则 409。"""
    # 请求 ID（可能是别名）必须仍解析到规范档案；档案不存在 → 404。
    canonical = resolve_canonical(conn, requested_id)
    if canonical is None:
        raise HTTPException(status_code=404, detail=f"展品不存在: {requested_id}")
    # 首次交接时的规范档案可能已归并：解析其当前归属后再比较，
    # 归并后用目标 ID 或别名重放同一 request_key 仍命中同一笔交接。
    stored_canonical = resolve_canonical(conn, row["canonical_id"]) or row["canonical_id"]
    same_target = (
        stored_canonical == canonical
        and row["from_round_id"] == payload.from_round_id
        and row["to_round_id"] == payload.to_round_id
        and row["from_area"] == payload.from_area
        and row["to_area"] == payload.to_area
    )
    if not same_target:
        raise HTTPException(
            status_code=409,
            detail=f"request_key {payload.request_key} 已绑定另一笔交接"
                   "（展品/轮次/区域不同），不能换目标重复提交",
        )
    return ZoneHandoverCommitOut(**json.loads(row["result_json"]), already_committed=True)


# ---------- 交接提交与固化 ----------

def commit_handover(
    conn: sqlite3.Connection,
    requested_id: str,
    payload: ZoneHandoverRequest,
) -> ZoneHandoverCommitOut:
    """在单个 SQLite 写事务内核验并固化一笔区域交接；幂等。

    别名解析、轮次读取、定位证据核验、链尖续接核验与落库全部在同一个
    ``BEGIN IMMEDIATE`` 事务内完成：任一校验失败整体回滚、不留半笔记录；
    并发交接由写锁串行化，最多一笔生效。
    """
    # 幂等回放不依赖写锁之外的状态：先读一次（无行则进入写事务）。
    existing = conn.execute(
        "SELECT * FROM zone_handovers WHERE request_key = ?",
        (payload.request_key,),
    ).fetchone()
    if existing is not None:
        return _check_replay(conn, existing, requested_id, payload)

    conn.execute("BEGIN IMMEDIATE")
    try:
        # 锁内重读：并发的相同 request_key 可能已完成落库
        existing = conn.execute(
            "SELECT * FROM zone_handovers WHERE request_key = ?",
            (payload.request_key,),
        ).fetchone()
        if existing is not None:
            conn.commit()  # 尚未写入，COMMIT 仅释放 IMMEDIATE 锁
            return _check_replay(conn, existing, requested_id, payload)

        canonical = resolve_canonical(conn, requested_id)
        if canonical is None:
            raise HTTPException(status_code=404, detail=f"展品不存在: {requested_id}")

        if payload.from_round_id == payload.to_round_id:
            raise HTTPException(status_code=422, detail="来源轮次与接收轮次不能相同")

        from_round = load_round(conn, payload.from_round_id)
        if from_round is None:
            raise HTTPException(
                status_code=404,
                detail=f"盘点轮次不存在或尚未对账: {payload.from_round_id}",
            )
        to_round = load_round(conn, payload.to_round_id)
        if to_round is None:
            raise HTTPException(
                status_code=404,
                detail=f"盘点轮次不存在或尚未对账: {payload.to_round_id}",
            )

        # 轮次起始时间必须递增：来源轮次起始时刻严格早于接收轮次
        from_start_s = _parse_iso_epoch(from_round.start_at)
        to_start_s = _parse_iso_epoch(to_round.start_at)
        if from_start_s >= to_start_s:
            raise HTTPException(
                status_code=422,
                detail=f"交接轮次起始时间必须递增：来源轮次 {payload.from_round_id} "
                       f"起始时刻 {from_round.start_at} 不早于接收轮次 "
                       f"{payload.to_round_id} 起始时刻 {to_round.start_at}",
            )

        # 证据核验：展品须列在两轮预期清单中，并在各轮明确定位于所报区域
        _verify_round_location(conn, from_round, canonical, payload.from_area, "来源")
        _verify_round_location(conn, to_round, canonical, payload.to_area, "接收")

        # 链式续接：只能从上一笔的接收轮和区域续接；旧轮次不能回填覆盖当前去向
        rows = _chain_rows(conn, canonical)
        tip = _chain_tip(rows)
        prev_id: str | None = None
        if tip is not None:
            if tip["to_round_id"] != payload.from_round_id or tip["to_area"] != payload.from_area:
                raise HTTPException(
                    status_code=409,
                    detail=f"交接链只能从链尖续接：当前去向为轮次 {tip['to_round_id']} "
                           f"区域 {tip['to_area']}（交接 {tip['handover_id']}），本次来源"
                           f"轮次 {payload.from_round_id} 区域 {payload.from_area} 与链尖"
                           "不符，旧轮次不能回填覆盖当前去向",
                )
            prev_id = tip["handover_id"]

        handover_id = uuid.uuid4().hex
        created_at = _utc_now_iso()
        result = ZoneHandoverOut(
            handover_id=handover_id,
            request_key=payload.request_key,
            requested_id=requested_id,
            canonical_id=canonical,
            from_round_id=payload.from_round_id,
            to_round_id=payload.to_round_id,
            from_area=payload.from_area,
            to_area=payload.to_area,
            from_round_digest=from_round.request_digest,
            to_round_digest=to_round.request_digest,
            from_round_start_at=from_round.start_at,
            to_round_start_at=to_round.start_at,
            prev_handover_id=prev_id,
            created_at=created_at,
        )
        conn.execute(
            """INSERT INTO zone_handovers
               (handover_id, request_key, canonical_id, requested_id,
                from_round_id, to_round_id, from_area, to_area,
                from_round_digest, to_round_digest,
                from_round_start_s, to_round_start_s,
                prev_handover_id, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                handover_id, payload.request_key, canonical, requested_id,
                payload.from_round_id, payload.to_round_id,
                payload.from_area, payload.to_area,
                from_round.request_digest, to_round.request_digest,
                from_start_s, to_start_s, prev_id,
                json.dumps(result.model_dump(), ensure_ascii=False),
                created_at,
            ),
        )
        conn.commit()
        return ZoneHandoverCommitOut(**result.model_dump(), already_committed=False)
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise


# ---------- 当前区域与交接历史 ----------

def load_history(
    conn: sqlite3.Connection,
    requested_id: str,
    canonical_id: str,
) -> ZoneHandoverHistoryOut:
    """当前区域 + 按轮次时间排序的交接历史（别名闭包取链；只读）。

    历史条目回放落库时固化的记录原文；当前区域取链尖的接收轮次与区域。
    """
    rows = _chain_rows(conn, canonical_id)
    entries = [ZoneHandoverOut(**json.loads(r["result_json"])) for r in rows]
    tip = _chain_tip(rows)
    current: ZoneCurrentOut | None = None
    if tip is not None:
        current = ZoneCurrentOut(
            round_id=tip["to_round_id"],
            area=tip["to_area"],
            handover_id=tip["handover_id"],
            round_start_at=_to_iso(tip["to_round_start_s"]),
        )
    return ZoneHandoverHistoryOut(
        requested_id=requested_id,
        canonical_id=canonical_id,
        current=current,
        handovers=entries,
    )
