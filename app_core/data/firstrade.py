"""Firstrade session and option helpers."""

from __future__ import annotations

import datetime as dt
import time
import typing as T
from urllib.parse import urlparse

import requests

Formatter = T.Callable[[dict[str, T.Any], tuple[str, ...]], str]


def has_valid_session(session_state: dict[str, T.Any] | None) -> bool:
    """Return ``True`` when the cached Firstrade session contains SID and FTAT."""

    if not isinstance(session_state, dict):
        return False
    sid = str(session_state.get("sid") or "").strip()
    ftat = str(session_state.get("ftat") or "").strip()
    return bool(sid and ftat)


def sample_session_rows(
    ft_session: dict[str, T.Any] | None,
    *,
    formatter: Formatter | None = None,
) -> list[dict[str, str]]:
    """Summarise the cached Firstrade session for diagnostic displays."""

    if not isinstance(ft_session, dict) or not ft_session:
        return [{"label": "缓存会话", "detail": "尚未登录或无会话信息", "raw": ""}]

    keys = ", ".join(sorted(map(str, ft_session.keys())))
    detail = f"字段：{keys}" if keys else "会话字段为空"
    raw = ""
    if formatter:
        try:
            raw = formatter(ft_session, ("sid", "ftat", "timestamp", "accounts"))
        except Exception:
            raw = ""
    else:
        subset = {key: ft_session.get(key) for key in ("sid", "ftat", "timestamp", "accounts") if key in ft_session}
        raw = str(subset)

    return [
        {
            "label": "缓存会话",
            "detail": detail,
            "raw": raw,
        }
    ]


class FTClient:
    LOGIN_URL = "https://api3x.firstrade.com/sess/login"
    VERIFY_URL = "https://api3x.firstrade.com/sess/verify_pin"
    OPTIONS_URL = "https://api3x.firstrade.com/public/oc"
    OPTION_QUOTE_URL = "https://api3x.firstrade.com/market/quote/option"
    OPTION_PREVIEW_URL = "https://api3x.firstrade.com/trade/options/preview"
    OPTION_PLACE_URL = "https://api3x.firstrade.com/trade/options/place"
    SESSION_HEADERS = {
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
        "User-Agent": "okhttp/4.9.2",
        "access-token": "833w3XuIFycv18ybi",
    }

    def __init__(
        self,
        username: str,
        password: str,
        twofa_code: str | None = None,
        session_state: dict[str, T.Any] | None = None,
        login: bool = True,
        logger: T.Optional[T.Callable[[str], None]] = None,
    ):
        self.username = username or ""
        self.password = password or ""
        self.twofa_code = (twofa_code or "").strip()
        self.session: T.Optional[requests.Session] = None
        self.enabled = False
        self.error: str | None = None
        self._login_json: dict[str, T.Any] | None = None
        self.session_state: dict[str, T.Any] = {}
        self._logger = logger
        self.account_ids: list[str] = []
        self._default_host = urlparse(self.LOGIN_URL).netloc

        if session_state:
            self._restore_session(session_state)
        elif login:
            self._init()

    def _log(self, message: str) -> None:
        if self._logger:
            try:
                self._logger(message)
            except Exception:
                pass

    def _init(self) -> None:
        try:
            sess = requests.Session()
            sess.headers.update(self.SESSION_HEADERS)
            if self._default_host:
                sess.headers["Host"] = self._default_host
            self._log("正在初始化 Firstrade 会话并发送登录请求。")
            login_resp = sess.post(
                self.LOGIN_URL,
                data={"username": self.username, "password": self.password},
                timeout=20,
            )
            login_data = self._safe_json(login_resp)
            self._login_json = login_data

            if login_resp.status_code != 200:
                detail = ""
                if isinstance(login_data, dict):
                    detail = str(login_data.get("message") or login_data.get("error") or "").strip()
                self._log(
                    f"Firstrade 登录出现 HTTP {login_resp.status_code}{f'（{detail}）' if detail else ''} 错误。"
                )
                self.error = f"登录 HTTP {login_resp.status_code}{f'（{detail}）' if detail else ''}"
                return
            if not isinstance(login_data, dict):
                self._log("Firstrade 登录返回异常数据。")
                self.error = "登录响应异常"
                return

            err_msg = str(login_data.get("error") or "").strip()
            if err_msg:
                self._log(f"Firstrade 登录返回错误：{err_msg}")
                self.error = err_msg
                return

            sid = login_data.get("sid")
            ftat = login_data.get("ftat")
            t_token = login_data.get("t_token")
            verification_sid = login_data.get("verificationSid")
            requires_mfa = bool(login_data.get("mfa"))

            if sid:
                sess.headers["sid"] = sid
                self._log("已从 Firstrade 登录响应中获取会话 ID。")

            if requires_mfa:
                self._log("Firstrade 登录需要多重验证。")
                if not self.twofa_code:
                    self.error = "Firstrade 登录需要输入双重验证码（短信或动态口令）。"
                    return

                verify_payload: dict[str, T.Any] = {
                    "remember_for": "30",
                }
                if t_token:
                    verify_payload["t_token"] = t_token

                code = self.twofa_code
                if verification_sid:
                    sess.headers["sid"] = verification_sid
                    self._log("使用验证 SID 进行一次性验证码校验。")
                    verify_payload["verificationSid"] = verification_sid
                    verify_payload["otpCode"] = code
                else:
                    self._log("提交认证器验证码进行校验。")
                    verify_payload["mfaCode"] = code

                verify_resp = sess.post(self.VERIFY_URL, data=verify_payload, timeout=20)
                verify_data = self._safe_json(verify_resp)
                if verify_resp.status_code != 200:
                    detail = ""
                    if isinstance(verify_data, dict):
                        detail = str(verify_data.get("message") or verify_data.get("error") or "").strip()
                    self._log(
                        f"Firstrade 多重验证失败，HTTP {verify_resp.status_code}{f'（{detail}）' if detail else ''}。"
                    )
                    self.error = f"双重验证 HTTP {verify_resp.status_code}{f'（{detail}）' if detail else ''}"
                    return
                if not isinstance(verify_data, dict):
                    self._log("Firstrade 多重验证返回异常数据。")
                    self.error = "双重验证响应异常"
                    return
                err_msg = str(verify_data.get("error") or "").strip()
                if err_msg:
                    self._log(f"Firstrade 多重验证报错：{err_msg}")
                    self.error = err_msg
                    return
                ftat = verify_data.get("ftat", ftat)
                sid = verify_data.get("sid") or verify_data.get("verificationSid") or verification_sid or sid

            if ftat:
                sess.headers["ftat"] = ftat
            if sid:
                sess.headers["sid"] = sid

            self.session = sess
            self.enabled = bool(ftat and sid)
            if self.enabled:
                self._update_accounts(login_data)
                self.session_state = {
                    "ftat": ftat,
                    "sid": sid,
                    "timestamp": time.time(),
                    "accounts": list(self.account_ids),
                }
                self._log("Firstrade 会话建立成功。")
            if not self.enabled and not self.error:
                self._log("Firstrade 会话缺少 SID/FTAT。")
                self.error = "Firstrade 会话缺少 SID/FTAT"
        except Exception as exc:
            self._log(f"Firstrade 登录出现意外错误：{exc}")
            self.error = str(exc)
            return

    def _restore_session(self, session_state: dict[str, T.Any]) -> None:
        ftat = session_state.get("ftat")
        sid = session_state.get("sid")
        if not ftat or not sid:
            self._log("无法恢复 Firstrade 会话：缺少 SID/FTAT。")
            self.error = "缺少会话令牌"
            self.enabled = False
            self.session_state = {}
            return

        sess = requests.Session()
        sess.headers.update(self.SESSION_HEADERS)
        host = urlparse(self.LOGIN_URL).netloc or self._default_host
        if host:
            sess.headers["Host"] = host
        sess.headers["ftat"] = ftat
        sess.headers["sid"] = sid
        self.session = sess
        self.enabled = True
        self.error = None
        self.session_state = {
            "ftat": ftat,
            "sid": sid,
            "timestamp": session_state.get("timestamp", time.time()),
            "accounts": list(session_state.get("accounts") or []),
        }
        self.account_ids = [
            str(a).strip()
            for a in session_state.get("accounts", [])
            if isinstance(a, str) and str(a).strip()
        ]
        self._log("已通过缓存令牌恢复 Firstrade 会话。")

    @staticmethod
    def _safe_json(resp: requests.Response) -> T.Any:
        try:
            return resp.json()
        except Exception:
            return None

    def export_session_state(self) -> dict[str, T.Any]:
        state = dict(self.session_state)
        if self.account_ids:
            state.setdefault("accounts", list(self.account_ids))
        return state

    def _update_accounts(self, payload: dict[str, T.Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        accounts: list[str] = []

        def add(value: T.Any) -> None:
            if value is None:
                return
            text = str(value).strip()
            if text and text not in accounts:
                accounts.append(text)

        candidate_keys = [
            "accounts",
            "accountList",
            "acctList",
            "account_list",
            "acctNoList",
            "data",
        ]
        for key in candidate_keys:
            items = payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        add(item.get("account"))
                        add(item.get("acctNo"))
                        add(item.get("accountNumber"))
                    else:
                        add(item)

        for key in ("account", "acctNo", "accountNumber", "primaryAccount"):
            add(payload.get(key))

        if accounts:
            self.account_ids = accounts

    def has_weekly_expiring_on(
        self, symbol: str, expiry: dt.date
    ) -> tuple[T.Optional[bool], T.Optional[str], bool]:
        """Check whether a symbol lists the target Friday contract in Firstrade."""

        if not self.enabled or not self.session:
            return None, None, False

        try:
            self._log(f"正在请求 {symbol.upper()} 的周度期权到期日。")
            resp = self.session.get(
                self.OPTIONS_URL,
                params={"m": "get_exp_dates", "root_symbol": symbol.upper()},
                timeout=20,
            )
            data = self._safe_json(resp)
            if resp.status_code == 401:
                self.enabled = False
                self.error = "Firstrade 会话已过期"
                self._log("查询期权时 Firstrade 会话已过期（HTTP 401）。")
                return None, None, False
            if resp.status_code != 200 or not isinstance(data, dict):
                self._log(f"查询 {symbol.upper()} 的期权失败，HTTP {resp.status_code}。")
                return None, None, False
            err_msg = str(data.get("error") or "").strip()
            if err_msg:
                self._log(f"查询 {symbol.upper()} 的期权返回错误：{err_msg}")
                return None, None, False
            items = data.get("items")
            if not isinstance(items, list):
                return None, None, False

            target = expiry.strftime("%Y%m%d")
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                exp_date = str(entry.get("exp_date", "")).strip()
                if exp_date != target:
                    continue
                exp_type = str(entry.get("exp_type", "")).upper()
                if exp_type == "W":
                    self._log(
                        f"找到 {symbol.upper()} 在 {target} 的到期日 {exp_date}（类型 {exp_type}）。"
                    )
                    return True, exp_type, True
                self._log(
                    f"找到 {symbol.upper()} 在 {target} 的到期日 {exp_date}（类型 {exp_type}，非周度期权）。"
                )
                return False, exp_type or None, True
            self._log(f"未找到 {symbol.upper()} 在 {target} 的匹配到期日。")
            return False, None, False
        except Exception:
            self._log(f"获取 {symbol.upper()} 期权到期日时出现异常。")
            return None, None, False


__all__ = ["FTClient", "has_valid_session", "sample_session_rows"]
