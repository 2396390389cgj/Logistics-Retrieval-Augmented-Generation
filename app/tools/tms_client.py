"""TMS / OMS 客户端：实时运单状态、轨迹、费用查询。

- 配置了 TMS_BASE_URL/OMS_BASE_URL 时走真实 HTTP 接口；
- 未配置时使用内置 Mock 数据（演示/开发/评测均可离线运行）。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# \b 在中文语境（\w 含汉字）不生效，改用显式前后断言
WAYBILL_RE = re.compile(r"(?<![A-Za-z0-9])(SF|JD|ZTO|DB)\d{10,14}(?![0-9])", re.IGNORECASE)

# 内置 Mock 运单库
_MOCK_WAYBILLS = {
    "SF1234567890": {
        "waybill_no": "SF1234567890", "carrier": "顺丰",
        "origin": "上海", "destination": "北京", "status": "运输中",
        "latest_event": "快件已从上海转运中心发出",
        "timeline": [
            {"time": "2026-11-12 09:20", "event": "顺丰已揽收", "location": "上海浦东新区张江营业点"},
            {"time": "2026-11-12 21:40", "event": "到达上海转运中心", "location": "上海"},
            {"time": "2026-11-13 06:10", "event": "快件已从上海转运中心发出", "location": "上海"},
        ],
        "freight": 23.0, "cod_amount": 0.0,
        "estimated_delivery": "2026-11-14",
    },
    "JD9876543210": {
        "waybill_no": "JD9876543210", "carrier": "京东物流",
        "origin": "广州", "destination": "成都", "status": "派送中",
        "latest_event": "快件正在派送中，请保持电话畅通",
        "timeline": [
            {"time": "2026-11-11 15:00", "event": "京东物流已揽收", "location": "广州天河区"},
            {"time": "2026-11-12 19:30", "event": "到达成都分拨中心", "location": "成都"},
            {"time": "2026-11-13 08:55", "event": "快件正在派送中", "location": "成都高新区"},
        ],
        "freight": 15.0, "cod_amount": 299.0,
        "estimated_delivery": "2026-11-13",
    },
}


def extract_waybill_no(text: str) -> str | None:
    m = WAYBILL_RE.search(text or "")
    return m.group(0).upper() if m else None


class TMSClient:
    """运输管理系统：运单状态 / 轨迹。"""

    async def get_waybill(self, waybill_no: str) -> dict | None:
        if settings.TMS_BASE_URL:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{settings.TMS_BASE_URL}/api/tms/waybill/{waybill_no}")
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:  # noqa: BLE001
                logger.warning("TMS 接口调用失败，回退 Mock：%s", e)
        return _MOCK_WAYBILLS.get(waybill_no.upper())


class OMSClient:
    """订单管理系统：订单状态 / 揽收 / 费用。"""

    async def get_order(self, waybill_no: str) -> dict | None:
        wb = _MOCK_WAYBILLS.get(waybill_no.upper())
        if wb is None and settings.OMS_BASE_URL:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{settings.OMS_BASE_URL}/api/oms/order/{waybill_no}")
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:  # noqa: BLE001
                logger.warning("OMS 接口调用失败：%s", e)
        if wb is None:
            return None
        return {
            "waybill_no": wb["waybill_no"],
            "order_status": "已发货" if wb["status"] != "已签收" else "已完成",
            "pickup": "已揽收",
            "freight": wb["freight"],
            "cod_amount": wb["cod_amount"],
            "pickup_appointment": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
        }


class RuleEngine:
    """实时费用/时效规则计算（Agent 工具之一，补充静态知识库的实时性）。"""

    PRICES = {  # 首重/kg，续重/kg（示意）
        "同城": (8, 1), "省内": (10, 2), "跨省": (12, 4), "偏远": (15, 8),
    }

    def estimate_freight(self, lane_type: str, weight_kg: float) -> dict:
        first, add = self.PRICES.get(lane_type, self.PRICES["跨省"])
        weight = max(float(weight_kg), 1.0)
        extra = max(weight - 1.0, 0.0)
        fee = first + extra * add
        return {"lane_type": lane_type, "weight_kg": weight,
                "first_weight_fee": first, "additional_fee_per_kg": add,
                "estimated_freight": round(fee, 1)}
