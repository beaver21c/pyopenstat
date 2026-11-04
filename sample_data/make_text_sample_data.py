"""
Generate synthetic sample datasets for the text-mining and relational-network modules.
All sentences and organisation names are synthetic (templates + random combinations, seed fixed).

- Text_PolicyDocs.csv      : 150 policy-style documents in 5 fields (문서ID, 분야, 연도, 제목, 본문)
- Network_Collaboration.csv: collaboration edge list among 30 organisations (출발기관, 도착기관, 협력횟수, 협력유형)

Run: python sample_data/make_text_sample_data.py
"""

import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(2026)

REGIONS = ["서울", "부산", "대구", "광주", "대전", "강원", "충남", "전북", "경북", "제주", "농어촌", "도시 외곽"]
COMMON = ["지방자치단체", "중앙정부", "예산", "전달체계", "성과평가", "데이터", "협력체계", "시범사업", "제도개선", "현장"]

FIELDS = {
    "돌봄": {
        "targets": ["노인", "장애인", "독거노인", "고령자", "보호자"],
        "services": ["방문요양", "통합돌봄", "재가서비스", "주간보호", "돌봄인력", "케어매니저"],
        "policies": ["지역사회 통합돌봄", "장기요양보험", "돌봄수당", "돌봄 바우처"],
        "issues": ["돌봄 공백", "가족 부담", "인력 부족", "서비스 연계"],
    },
    "고용": {
        "targets": ["청년", "중장년", "여성", "구직자", "취약계층"],
        "services": ["직업훈련", "취업알선", "고용센터", "일자리", "직무교육", "구직급여"],
        "policies": ["국민취업지원제도", "고용보험", "청년일자리 사업", "직업능력개발"],
        "issues": ["실업률", "고용 불안", "임금 격차", "노동시장 이탈"],
    },
    "주거": {
        "targets": ["무주택 가구", "청년 가구", "신혼부부", "주거취약계층", "임차인"],
        "services": ["공공임대", "주거급여", "전세자금", "주택개량", "임대료 지원", "주거상담"],
        "policies": ["주거복지 로드맵", "공공임대주택 공급", "주거급여 제도", "전월세 지원"],
        "issues": ["주거비 부담", "주택 노후화", "전세 사기", "주거 불안정"],
    },
    "보건의료": {
        "targets": ["만성질환자", "환자", "지역주민", "고위험군", "응급환자"],
        "services": ["건강검진", "일차의료", "방문진료", "보건소", "응급의료", "만성질환 관리"],
        "policies": ["건강보험 보장성 강화", "공공의료 확충", "일차의료 시범사업", "의료전달체계 개편"],
        "issues": ["의료 접근성", "의료비 부담", "의료인력 부족", "지역 의료 격차"],
    },
    "아동가족": {
        "targets": ["아동", "영유아", "한부모가족", "다문화가족", "부모"],
        "services": ["보육", "어린이집", "아동수당", "육아휴직", "방과후 돌봄", "아동보호"],
        "policies": ["저출산 대응 정책", "아동수당 제도", "부모급여", "아동학대 예방"],
        "issues": ["양육 부담", "출산율 하락", "보육 공백", "아동학대"],
    },
}

TEMPLATES = [
    "{region} 지역 {target}의 {service} 이용을 늘리기 위해 {policy} 확대가 추진된다.",
    "{issue} 문제가 커지면서 {target} 대상 {service} 수요가 증가하였다.",
    "정부는 {policy}을 통해 {target}의 {issue}을 완화하고 {service} 품질을 높일 계획이다.",
    "{region}에서는 {service}와 {service2}를 연계하는 {common} 구축이 진행 중이다.",
    "{target} 지원 강화를 위해 {common} 정비와 {service} 인력 확충이 필요하다.",
    "전문가들은 {issue} 해소를 위해 {policy}의 {common} 개선을 제안하였다.",
    "{service} 이용 실태 조사 결과 {target}의 만족도는 높았으나 {issue}은 여전하였다.",
    "{policy} 시행 이후 {region} {target}의 {service} 접근성이 개선되었다.",
]


def make_doc(field: str) -> str:
    f = FIELDS[field]
    lines = []
    for _ in range(rng.randint(4, 7)):
        # 15% of sentences borrow vocabulary from another field, so clusters overlap realistically
        g = FIELDS[rng.choice(list(FIELDS))] if rng.random() < 0.15 else f
        sv = rng.sample(g["services"], 2)
        lines.append(rng.choice(TEMPLATES).format(
            region=rng.choice(REGIONS), target=rng.choice(g["targets"]), service=sv[0], service2=sv[1],
            policy=rng.choice(g["policies"]), issue=rng.choice(g["issues"]), common=rng.choice(COMMON)))
    return " ".join(lines)


def make_text_docs(n_per_field: int = 30) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        for _ in range(n_per_field):
            f = FIELDS[field]
            title = f"{rng.choice(REGIONS)} {rng.choice(f['targets'])} {rng.choice(f['services'])} 현황과 과제"
            rows.append({"분야": field, "연도": rng.choice([2022, 2023, 2024, 2025]), "제목": title, "본문": make_doc(field)})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "문서ID", range(1, len(df) + 1))
    return df


def make_network() -> pd.DataFrame:
    groups = {
        "중앙부처": ["보건복지부", "고용노동부", "국토교통부", "교육부", "행정안전부", "여성가족부", "기획재정부"],
        "연구기관": ["사회정책연구원", "보건정책연구소", "노동정책연구원", "주거정책연구소", "아동정책연구소", "지역정책연구원", "데이터연구센터"],
        "지자체": ["가람시", "누리군", "다온시", "마루시", "바다군", "새솔시", "아라도", "하늬도"],
        "민간기관": ["복지재단", "돌봄협동조합", "청년지원센터", "주거상담센터", "의료협동조합", "아동권리기관", "사회적기업협회", "자원봉사센터"],
    }
    kinds = ["공동연구", "위탁사업", "자문", "자료공유"]
    edges = {}

    def add(a, b, w):
        key = tuple(sorted((a, b)))
        edges[key] = edges.get(key, 0) + w

    names = {g: v for g, v in groups.items()}
    for g, members in names.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if rng.random() < 0.42:
                    add(a, b, rng.randint(2, 9))
    all_nodes = [(g, n) for g, v in names.items() for n in v]
    while len(edges) < 86:
        (g1, a), (g2, b) = rng.sample(all_nodes, 2)
        if g1 != g2 and rng.random() < 0.5:
            add(a, b, rng.randint(1, 4))
    rows = [{"출발기관": a, "도착기관": b, "협력횟수": w, "협력유형": rng.choice(kinds)} for (a, b), w in edges.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    docs = make_text_docs()
    docs.to_csv(os.path.join(HERE, "Text_PolicyDocs.csv"), index=False, encoding="utf-8-sig")
    net = make_network()
    net.to_csv(os.path.join(HERE, "Network_Collaboration.csv"), index=False, encoding="utf-8-sig")
    print(f"Text_PolicyDocs.csv: {len(docs)} docs | Network_Collaboration.csv: {len(net)} edges")
