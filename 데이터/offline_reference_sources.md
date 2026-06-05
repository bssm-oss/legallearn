# 오프라인 참고 공개자료와 위험 신호 매핑

이 프로젝트는 실행 시 외부 API에 의존하지 않는다. 아래 자료는 모델 피처와 규칙 설계에 반영한 공개 기준이며, 실제 앱 런타임은 로컬 CSV와 학습 아티팩트만 사용한다.

## 공식 참고 자료

1. 국토교통부 실거래가 공개시스템
   - URL: https://rt.molit.go.kr/pt/info/info.do?mobileAt=v
   - 활용: 주변 실거래가와 입력 보증금/매매가의 괴리율 피처 설계
   - 확인 내용: 부동산 거래신고제를 통해 수집된 실거래 자료를 공개한다.

2. 공공데이터포털 `국토교통부_아파트 매매 실거래가 자료`
   - URL: https://www.data.go.kr/data/15126469/openapi.do
   - 활용: 오프라인 확장 시 지역·기간별 매매 신고정보를 수집할 수 있는 공개 API 후보

3. 공공데이터포털 `국토교통부_건축HUB_건축물대장정보 서비스`
   - URL: https://www.data.go.kr/data/15134735/openapi.do
   - 활용: 위반건축물 여부, 건축물대장 속성 확인 피처의 공개자료 후보

4. 공공데이터포털 `국토교통부_공동주택 기본 정보제공 서비스`
   - URL: https://www.data.go.kr/data/15058453/openapi.do
   - 활용: 공동주택 기본정보와 주변 시설·관리정보 확장 후보

5. HUG 전세보증금반환보증 상품개요
   - URL: https://m.khug.or.kr/hug/web/ig/dr/igdr000001.jsp?tabMenu=Y
   - 활용: 전세보증금+선순위채권이 주택가격×90% 이내인지, 등기부 권리침해사항·선순위채권·신탁/담보 제한 특약을 위험 신호로 반영

6. HUG 전세 및 임대보증 공공데이터
   - URL: https://khug.or.kr/houstar/web/p03/01/p030105.jsp?articleId=34712&currentPage=1&mode=S
   - 활용: 지역별 전세보증사고 현황을 향후 오프라인 데이터로 확장할 수 있는 후보

7. 대법원 인터넷등기소
   - URL: https://www.iros.go.kr
   - 활용: 압류, 가압류, 가처분, 근저당, 신탁 등 등기부 권리침해 신호 설계

8. 국가법령정보센터 판례/법령
   - URL: https://www.law.go.kr
   - 활용: 판례와 법령상 사기, 공인중개사 표시광고, 임대차 보호요건 참조
   - 수집 메모: OPEN API는 등록 IP/도메인 검증이 필요해 로컬 대량 호출은 실패했고, 접근 가능한 공개 페이지와 기존 판례 CSV를 오프라인 참조로 사용했다.

9. 한국경제 대법원 보증채무금 보도
   - URL: https://www.hankyung.com/article/202506220823i
   - 활용: 전세보증금 부풀림/허위 전세계약 위험 유형을 공개자료 기반 파생 사례로 반영

## 모델 피처 반영

- `jeonse_ratio`: 보증금 / 추정 시세
- `debt_ratio`: (보증금 + 근저당 + 선순위채권) / 추정 시세
- `seizure`, `provisional_seizure`: 갑구 권리침해 신호
- `trust_registered`: 신탁등기 및 신탁원부 확인 필요 신호
- `illegal_building`: 건축물대장상 위반건축물 위험
- `broker_unregistered`, `broker_advertising_issue`: 중개사/광고 관련 위험
- `guarantee_insurance_available`: 보증보험 가능 여부
- `public_risk_indicators.csv`: 공식 공개자료 기준 위험 지표
- `external_case_references.csv`: 추가 판례/보도 참조 메타데이터

## 한계

- 실제 등기부등본·건축물대장·실거래가 원문을 자동 조회하지 않는다.
- 국가법령정보 OPEN API는 등록 검증 없이는 현재 로컬에서 직접 대량 호출되지 않았다.
- 파생 계약 예시는 판례와 공개 기준을 바탕으로 만든 학습용 예시이며 실제 피해자 기록이 아니다.
- 출력은 계약 전 위험 신호 탐지이며 법적 판단이 아니다.
