from __future__ import annotations

DEFAULT_OPENSEARCH_URL = "https://localhost:9200"

DEFAULT_OPENSEARCH_USERNAME = "admin"

DEFAULT_OPENSEARCH_PASSWORD_ENV = "OPENSEARCH_PASSWORD"

DEFAULT_CHUNK_INDEX = "caselaw_benchmark_chunks_v1"

DEFAULT_CASE_INDEX = "caselaw_benchmark_cases_v1"

DEFAULT_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_EMBEDDING_KEY_ENV = "SILICONFLOW_API_KEY"

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"

SECTION_WEIGHTS = {
    "fine_tags": 1.60,
    "fine_rule": 1.55,
    "focus_tags": 1.50,
    "focus_analysis": 1.45,
    "reasoning": 1.20,
    "facts": 1.00,
    "claims": 0.70,
    "defense": 0.85,
    "judgment": 0.75,
    "header": 0.40,
    "statutes": 0.45,
}

KEY_SECTION_TYPES = {"fine_tags", "fine_rule", "focus_tags", "focus_analysis", "reasoning", "facts", "defense"}

LEGAL_RERANK_SECTION_TYPES = {"fine_tags", "fine_rule", "focus_tags", "focus_analysis", "reasoning"}

CASE_KEY_SECTION_TYPES = {"fine_tags", "fine_rule", "focus_tags", "focus_analysis", "reasoning", "facts"}

CASE_RERANK_SECTION_ORDER = [
    "fine_tags",
    "fine_rule",
    "focus_tags",
    "focus_analysis",
    "reasoning",
    "facts",
    "claims",
    "judgment",
]

CASE_RERANK_SECTION_BUDGETS = {
    "fine_tags": 300,
    "fine_rule": 500,
    "focus_tags": 300,
    "focus_analysis": 700,
    "reasoning": 900,
    "facts": 320,
    "judgment": 240,
    "claims": 220,
    "defense": 220,
}

CASE_RERANK_SECTION_GROUPS = {
    "fine_tags": "【核心争议】",
    "fine_rule": "【核心争议】",
    "focus_tags": "【核心争议】",
    "focus_analysis": "【核心争议】",
    "reasoning": "【裁判规则】",
    "facts": "【关键事实】",
    "judgment": "【裁判结果】",
    "claims": "【诉请摘要】",
    "defense": "【抗辩摘要】",
}

CASE_RERANK_MAX_SELECTED_CHUNKS = 6

RERANK_GUARDRAIL_TEXT_LIMIT = 5000

RERANK_GUARDRAIL_MAX_CHUNKS = 12

RERANK_NEGATION_LOOKBACK = 18

RERANK_NEGATION_CUES = [
    "完全未涉及",
    "并未涉及",
    "未涉及",
    "不涉及",
    "未提及",
    "未载明",
    "未显示",
    "未体现",
    "未说明",
    "未论及",
    "未主张",
    "未请求",
    "没有",
    "并无",
    "均无",
    "不存在",
    "不包含",
    "未见",
    "未发现",
]

RERANK_SINGLE_CHAR_NEGATION_CUES = ["无"]

RERANK_REQUIRED_FACTORS = [
    {
        "name": "ownership_retention",
        "query_any": ["所有权保留", "取回权", "留置所有权"],
        "doc_any": ["所有权保留", "取回权", "返还货物", "留置所有权"],
        "penalty": 0.18,
    },
    {
        "name": "deposit_penalty",
        "query_any": ["定金罚则", "成约定金", "违约定金", "返还定金"],
        "doc_any": ["定金罚则", "成约定金", "违约定金", "返还定金", "没收定金", "双倍返还定金"],
        "penalty": 0.12,
    },
    {
        "name": "invoice_dispute",
        "query_any": [
            "发票争议",
            "未开票",
            "未开发票",
            "未开具发票",
            "拒开发票",
            "开票义务",
            "开具发票",
            "补开发票",
            "发票问题",
        ],
        "doc_any": ["发票争议", "未开票", "开票", "发票", "增值税发票"],
        "penalty": 0.06,
    },
    {
        "name": "third_party_supply",
        "query_any": ["第三方供货", "第三方代为供货", "第三人供货", "代为供货", "指示第三方"],
        "doc_any": ["第三方供货", "第三方代为供货", "第三人供货", "代为供货", "指示第三方", "第三人履行"],
        "penalty": 0.10,
    },
    {
        "name": "reconciliation_silence",
        "query_all": ["对账单"],
        "query_any": ["未在合理期限", "未提出异议", "未及时提出异议", "视为认可", "结算依据"],
        "doc_any": ["对账单", "未提出异议", "未在合理期限", "结算依据", "对账", "结算单"],
        "penalty": 0.08,
    },
    {
        "name": "seal_dispute",
        "query_any": ["偷盖", "冒盖", "私盖", "收货确认单", "合同外供货"],
        "doc_any": ["偷盖", "冒盖", "私盖", "印章", "收货确认单", "合同外供货", "盖章", "公章"],
        "penalty": 0.10,
    },
    {
        "name": "oral_contract_evidence",
        "query_any": ["口头买卖", "微信聊天记录", "仅凭微信", "无书面合同"],
        "doc_any": ["口头买卖", "微信聊天记录", "无书面合同", "聊天记录", "微信"],
        "penalty": 0.10,
    },
    {
        "name": "termination_time",
        "query_any": ["解除时间", "解除时间如何认定", "起诉状副本送达", "送达时解除"],
        "doc_any": ["解除时间", "起诉状副本送达", "送达时解除", "解除通知", "合同解除时间"],
        "penalty": 0.10,
    },
    {
        "name": "agency_or_third_payment",
        "query_any": ["委托他人", "代付", "案外人", "以自己名义"],
        "doc_any": ["委托他人", "代付", "案外人", "以自己名义", "第三人付款", "第三人代付", "代为支付"],
        "penalty": 0.04,
    },
]

RERANK_CONFLICT_FACTORS = [
    {
        "name": "defective_delivery_vs_non_delivery",
        "query_any": ["瑕疵", "异物", "碎骨", "淤血", "淋巴", "质量"],
        "doc_any": ["未交货", "未发货", "未交付", "迟延交货", "逾期交货", "未履行交货"],
        "doc_required_absent": ["瑕疵", "异物", "碎骨", "淤血", "淋巴", "质量"],
        "penalty": 0.14,
    },
    {
        "name": "non_delivery_vs_defective_delivery",
        "query_any": ["未交货", "未发货", "未交付"],
        "doc_any": ["质量异议", "瑕疵", "异物", "质量问题"],
        "doc_required_absent": ["未交货", "未发货", "未交付"],
        "penalty": 0.10,
    },
    {
        "name": "buyer_default_vs_seller_default",
        "query_any": ["未按约提车", "拒收", "价格过高", "买方拒收", "买方未提货", "买方未按约"],
        "doc_any": ["卖方未交货", "出卖人未交货", "卖方未发货", "出卖人未发货", "卖方根本违约"],
        "penalty": 0.08,
    },
    {
        "name": "collateral_invoice_only",
        "query_any": ["定金罚则", "所有权保留", "解除时间", "第三方供货"],
        "doc_any": ["发票", "开票"],
        "doc_required_absent": ["定金罚则", "所有权保留", "解除时间", "第三方供货", "第三方代为供货"],
        "penalty": 0.04,
    },
]

CASE_RERANK_HYBRID_WEIGHT = 0.65

CASE_RERANK_MODEL_WEIGHT = 0.25

CASE_RERANK_TEXT_LIMIT = 3600

DEFAULT_RERANK_MIN_INTERVAL_MS = 1200

DEFAULT_RERANK_MAX_RETRIES = 3

DEFAULT_RERANK_RANK_SAFE = True

DEFAULT_RERANK_MAX_RANK_PROMOTION = 20

DEFAULT_SOURCE_FIELDS = [
    "chunk_id",
    "doc_id",
    "case_name",
    "reason",
    "trial_level",
    "court_name",
    "judge_date",
    "section_type",
    "section_title",
    "chunk_text",
    "embedding_text",
    "char_start",
    "char_end",
    "line_start",
    "line_end",
    "prev_chunk_id",
    "next_chunk_id",
    "chunk_index_in_case",
    "chunk_index_in_section",
    "statutes",
    "section_weight",
]

DEFAULT_CASE_FIELDS = [
    "doc_id",
    "case_name",
    "reason",
    "trial_level",
    "court_name",
    "judge_date",
    "publish_date",
    "full_text",
    "full_text_hash",
    "litigants",
    "statutes",
]

