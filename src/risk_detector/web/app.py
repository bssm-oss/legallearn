from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

from risk_detector.paths import METADATA_PATH, MODEL_PATH, PROJECT_ROOT
from risk_detector.risk.scorer import RiskScorer


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_css() -> str:
    return (STATIC_DIR / "styles.css").read_text(encoding="utf-8")


def _model_summary() -> dict[str, str]:
    if not METADATA_PATH.exists():
        return {"rows": "-", "macro_f1": "-", "holdout": "-", "trained_at": "-"}
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metrics = metadata.get("metrics", {})
        return {
            "rows": f"{int(metadata.get('rows', 0)):,}",
            "macro_f1": str(metrics.get("macro_f1", "-")),
            "holdout": f"{int(metadata.get('holdout_rows', 0)):,}",
            "trained_at": str(metadata.get("trained_at", "-"))[:10],
        }
    except Exception:
        return {"rows": "-", "macro_f1": "-", "holdout": "-", "trained_at": "-"}


def _field_value(data: dict[str, list[str]], key: str, default: str = "") -> str:
    values = data.get(key)
    return values[0] if values else default


def _checkbox(data: dict[str, list[str]], key: str) -> str:
    return "true" if key in data else "false"


def _severity_label(severity: str) -> str:
    return {"safe": "안전", "warning": "주의", "danger": "위험"}.get(severity, severity)


def render_page(result: dict | None = None, error: str | None = None) -> str:
    model_summary = _model_summary()
    result_html = ""
    if error:
        result_html = f"<section class='result danger'><h2>실행 오류</h2><p>{error}</p></section>"
    elif result:
        grade_class = {"안전": "safe", "주의": "warning", "위험": "danger"}.get(result["risk_grade"], "warning")
        reasons = "".join(f"<li>{reason}</li>" for reason in result["reasons"])
        probs = "".join(
            f"<span><strong>{label}</strong> {prob:.1%}</span>"
            for label, prob in result["model_probabilities"].items()
        )
        checks = "".join(
            f"<tr><td><span class='severity {item['severity']}'>{_severity_label(item['severity'])}</span></td><td>{item['signal']}</td><td>{item['value']}</td></tr>"
            for item in result["checks"][:12]
        )
        result_html = f"""
        <section class="result {grade_class}">
          <div>
            <p class="eyeless">AI 2차 확인 결과</p>
            <h2>{result['risk_grade']} · {result['risk_score']}점</h2>
            <p>{result['legal_notice']}</p>
          </div>
          <div class="probabilities">{probs}</div>
          <div class="result-grid">
            <div>
              <h3>주요 위험 근거</h3>
              <ul>{reasons}</ul>
            </div>
            <div>
              <h3>모델/규칙 점수</h3>
              <dl>
                <dt>Bagging 모델 예측</dt><dd>{result['model_predicted_grade']}</dd>
                <dt>모델 점수 성분</dt><dd>{result['model_score_component']}</dd>
                <dt>규칙 점수 성분</dt><dd>{result['rule_score_component']}</dd>
              </dl>
            </div>
          </div>
          <table>
            <thead><tr><th>강도</th><th>점검 항목</th><th>값</th></tr></thead>
            <tbody>{checks}</tbody>
          </table>
        </section>
        """
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>부동산 계약 위험 탐지</title>
  <style>{_load_css()}</style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark"></span><span>Real Estate Risk AI</span></div>
      <a class="health-link" href="/api/health">health</a>
    </header>
    <section class="hero">
      <div>
        <h1>계약 전 위험 신호를 Bagging AI로 2차 확인</h1>
        <p>전세가율, 근저당, 압류, 신탁, 위반건축물, 중개사 설명, 특약 내용을 함께 분석해 안전·주의·위험 등급과 근거를 제공합니다.</p>
      </div>
      <div class="hero-side">
        <img src="/static/risk-illustration.svg" alt="부동산 계약 위험 분석 일러스트" class="hero-visual" />
        <div class="model-card">
          <span>학습 모델</span>
          <strong>sklearn BaggingClassifier</strong>
          <dl>
            <dt>rows</dt><dd>{model_summary['rows']}</dd>
            <dt>holdout F1</dt><dd>{model_summary['macro_f1']}</dd>
            <dt>holdout</dt><dd>{model_summary['holdout']}</dd>
            <dt>trained</dt><dd>{model_summary['trained_at']}</dd>
          </dl>
          <small>{MODEL_PATH.name}</small>
        </div>
      </div>
    </section>
    {result_html}
    <form method="post" action="/predict" class="panel" autocomplete="off">
      <div class="form-grid">
        <label>계약 유형<select name="contract_type" autocomplete="off"><option value="jeonse">전세</option><option value="monthly_rent">월세</option><option value="sale">매매</option></select></label>
        <label>주택 유형<select name="property_type" autocomplete="off"><option value="villa">빌라/다세대</option><option value="apartment">아파트</option><option value="officetel">오피스텔</option><option value="multi_family">다가구</option><option value="commercial">상가</option></select></label>
        <label>지역<select name="region" autocomplete="off"><option>수도권</option><option>광역시</option><option>비수도권</option><option>지방중소도시</option></select></label>
        <label>보증금/계약금(백만원)<input name="deposit_million" autocomplete="off" type="number" step="1" value="270"></label>
        <label>월세(백만원)<input name="monthly_rent_million" autocomplete="off" type="number" step="0.1" value="0"></label>
        <label>매매가(백만원)<input name="sale_price_million" autocomplete="off" type="number" step="1" value="0"></label>
        <label>추정 시세(백만원)<input name="estimated_market_price_million" autocomplete="off" type="number" step="1" value="300"></label>
        <label>근저당(백만원)<input name="mortgage_million" autocomplete="off" type="number" step="1" value="80"></label>
        <label>선순위채권/보증금(백만원)<input name="senior_claim_million" autocomplete="off" type="number" step="1" value="30"></label>
        <label>주변 시세 괴리율(%)<input name="nearby_market_gap_percent" autocomplete="off" type="number" step="0.1" value="12"></label>
        <label>계약기간(개월)<input name="contract_period_months" autocomplete="off" type="number" step="1" value="24"></label>
      </div>
      <fieldset>
        <legend>등기부·건축물·중개 위험 신호</legend>
        <label><input type="checkbox" name="seizure"> 압류 있음</label>
        <label><input type="checkbox" name="provisional_seizure" checked> 가압류/가처분 있음</label>
        <label><input type="checkbox" name="trust_registered"> 신탁등기 있음</label>
        <label><input type="checkbox" name="illegal_building"> 위반건축물</label>
        <label><input type="checkbox" name="landlord_multiple_properties" checked> 임대인 다주택/동시반환 부담</label>
        <label><input type="checkbox" name="landlord_prior_incidents"> 임대인 보증사고/분쟁 이력</label>
        <label><input type="checkbox" name="broker_unregistered"> 무등록/중개보조 단독 진행 의심</label>
        <label><input type="checkbox" name="broker_advertising_issue" checked> 허위·과장 광고 의심</label>
        <label><input type="checkbox" name="suspicious_special_clause" checked> 수상한 특약</label>
      </fieldset>
      <fieldset>
        <legend>보호 요건</legend>
        <label><input type="checkbox" name="guarantee_insurance_available"> 보증보험 가능</label>
        <label><input type="checkbox" name="fixed_date_ready" checked> 확정일자 가능</label>
        <label><input type="checkbox" name="move_in_ready" checked> 전입/점유 가능</label>
        <label><input type="checkbox" name="broker_explained_rights"> 중개대상물 확인설명 충분</label>
      </fieldset>
      <label>특약/계약서 내용<textarea name="special_clause_text" autocomplete="off">임차인은 임대인의 담보 제공 및 채권양도에 이의를 제기하지 않는다.</textarea></label>
      <label>사용자 상황<textarea name="user_situation_text" autocomplete="off">사회초년생이 시세보다 높은 보증금의 신축 빌라 전세계약을 앞두고 있다.</textarea></label>
      <button type="submit">위험도 분석</button>
    </form>
  </main>
</body>
</html>"""


class RiskRequestHandler(BaseHTTPRequestHandler):
    scorer: RiskScorer | None = None

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        relative = unquote(path.removeprefix("/static/"))
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents or not candidate.is_file():
            self._send_json({"ok": False, "error": "static file not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json({"ok": True, "model_exists": MODEL_PATH.exists(), "project_root": str(PROJECT_ROOT)})
            return
        if self.path.startswith("/static/"):
            self._send_static(self.path)
            return
        self._send_html(render_page())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        if "application/json" in self.headers.get("Content-Type", ""):
            payload = json.loads(raw or "{}")
        else:
            form = parse_qs(raw)
            payload = {
                "contract_type": _field_value(form, "contract_type", "jeonse"),
                "property_type": _field_value(form, "property_type", "villa"),
                "region": _field_value(form, "region", "수도권"),
                "deposit_million": _field_value(form, "deposit_million", "0"),
                "monthly_rent_million": _field_value(form, "monthly_rent_million", "0"),
                "sale_price_million": _field_value(form, "sale_price_million", "0"),
                "estimated_market_price_million": _field_value(form, "estimated_market_price_million", "1"),
                "mortgage_million": _field_value(form, "mortgage_million", "0"),
                "senior_claim_million": _field_value(form, "senior_claim_million", "0"),
                "nearby_market_gap_percent": _field_value(form, "nearby_market_gap_percent", "0"),
                "contract_period_months": _field_value(form, "contract_period_months", "24"),
                "special_clause_text": _field_value(form, "special_clause_text", ""),
                "user_situation_text": _field_value(form, "user_situation_text", ""),
                "seizure": _checkbox(form, "seizure"),
                "provisional_seizure": _checkbox(form, "provisional_seizure"),
                "trust_registered": _checkbox(form, "trust_registered"),
                "illegal_building": _checkbox(form, "illegal_building"),
                "landlord_multiple_properties": _checkbox(form, "landlord_multiple_properties"),
                "landlord_prior_incidents": _checkbox(form, "landlord_prior_incidents"),
                "broker_unregistered": _checkbox(form, "broker_unregistered"),
                "broker_advertising_issue": _checkbox(form, "broker_advertising_issue"),
                "suspicious_special_clause": _checkbox(form, "suspicious_special_clause"),
                "guarantee_insurance_available": _checkbox(form, "guarantee_insurance_available"),
                "fixed_date_ready": _checkbox(form, "fixed_date_ready"),
                "move_in_ready": _checkbox(form, "move_in_ready"),
                "broker_explained_rights": _checkbox(form, "broker_explained_rights"),
            }
        try:
            if self.scorer is None:
                self.scorer = RiskScorer()
            result = self.scorer.score(payload)
            if self.path == "/api/predict":
                self._send_json(result)
            else:
                self._send_html(render_page(result=result))
        except Exception as exc:  # pragma: no cover - web error path
            if self.path == "/api/predict":
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self._send_html(render_page(error=str(exc)), HTTPStatus.INTERNAL_SERVER_ERROR)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), RiskRequestHandler)
    print(f"웹 데모 실행: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
