"""Agent 双通道编排：意图识别 → 实时通道（TMS/OMS/规则引擎）→ 结构化证据。"""
from __future__ import annotations

import logging
import re

from app.tools.tms_client import OMSClient, RuleEngine, TMSClient, extract_waybill_no

logger = logging.getLogger(__name__)

# 实时意图模式
_INTENT_TRACK = re.compile(r"(运单|快递|包裹|件).{0,6}(到哪|在哪|状态|轨迹|进度)|查.{0,4}(单|件|物流)")
_INTENT_ORDER = re.compile(r"(订单|下单|揽收|预约|取件|费用|运费).{0,8}(查|多少|什么|怎么样)")
_INTENT_PRICE = re.compile(r"(运费|多少钱|收费|价格).{0,10}(kg|公斤|斤|重)?", re.IGNORECASE)
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|公斤|千克|斤)", re.IGNORECASE)
_LANE_RE = re.compile(r"(同城|省内|跨省|偏远)")


class LogisticsAgent:
    """RAG 之外的实时业务通道，输出结构化证据供融合生成。"""

    def __init__(self) -> None:
        self.tms = TMSClient()
        self.oms = OMSClient()
        self.rules = RuleEngine()

    async def detect_and_execute(self, query: str) -> list[dict]:
        """返回结构化实时证据列表 [{type, data, summary}]。"""
        evidences: list[dict] = []

        # 1) 运单号直接查询（最高优先级）
        waybill = extract_waybill_no(query)
        if waybill:
            info = await self.tms.get_waybill(waybill)
            if info:
                evidences.append({"type": "tms_waybill", "data": info,
                                  "summary": self._fmt_waybill(info)})
            order = await self.oms.get_order(waybill)
            if order:
                evidences.append({"type": "oms_order", "data": order,
                                  "summary": self._fmt_order(order)})
            return evidences

        # 2) 运单状态意图（无单号 → 提示需要单号）
        if _INTENT_TRACK.search(query):
            evidences.append({"type": "need_waybill_no", "data": {},
                              "summary": "用户想查询运单进度但未提供运单号，需要向用户澄清询问运单号。"})

        # 3) 运费估算意图
        if _INTENT_PRICE.search(query):
            m_w, m_l = _WEIGHT_RE.search(query), _LANE_RE.search(query)
            weight = float(m_w.group(1)) if m_w else 1.0
            if m_w and m_w.group(2).lower() == "斤":
                weight /= 2
            lane = m_l.group(1) if m_l else "跨省"
            est = self.rules.estimate_freight(lane, weight)
            evidences.append({"type": "freight_estimate", "data": est,
                              "summary": (f"按{lane}、{weight}kg 估算：首重{est['first_weight_fee']}元，"
                                          f"续重{est['additional_fee_per_kg']}元/kg，"
                                          f"预计运费约 {est['estimated_freight']} 元（仅供参考）。")})
        return evidences

    @staticmethod
    def _fmt_waybill(info: dict) -> str:
        tl = "；".join(f"{e['time']} {e['event']}（{e['location']}）"
                      for e in info.get("timeline", [])[-3:])
        return (f"运单 {info['waybill_no']}（{info['carrier']}）{info['origin']}→{info['destination']}，"
                f"当前状态：{info['status']}。最新动态：{info.get('latest_event', '')}。"
                f"预计送达：{info.get('estimated_delivery', '待更新')}。近期轨迹：{tl}")

    @staticmethod
    def _fmt_order(order: dict) -> str:
        return (f"订单状态：{order.get('order_status', '未知')}，"
                f"运费：{order.get('freight', '-')} 元，"
                f"代收货款：{order.get('cod_amount', 0)} 元，"
                f"揽收情况：{order.get('pickup', '未知')}。")
