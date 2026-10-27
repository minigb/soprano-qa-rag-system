"""Post-hoc unchanged-source check of the observed p/PP disagreement."""
import copy
import json
from evaluation.source_priority_matched import run


def prepare():
    from soprano_qa.dense import LlamaCppQwen3Embedder, _normalized_vector, dense_query_text
    s=run.load_settings()
    main=json.loads((run.ROOT / "frozen.json").read_text())
    source=next(c for c in main["cases"] if c["id"]=="conflict-attack")
    records=[x["record"] for x in source["pool"] if not x["record"]["id"].startswith("synthetic-")]
    root=run.ROOT / "observed_disagreement"
    root.mkdir(exist_ok=True)
    model=LlamaCppQwen3Embedder(s["embedding_model_path"],query_instruction=s["retrieval"]["query_instruction"],n_gpu_layers=-1)
    vectors=[_normalized_vector(v) for v in model.embed_documents([r["answer"] for r in records])]
    questions=[
        "처음 반주의 다이나믹을 p라고 설명한 자료와 PP라고 설명한 자료가 있는데, 이 차이를 어떻게 이해해야 하나요?",
        "첫 반주의 p와 PP 설명이 서로 다른 이유를 확정할 수 있나요? 두 설명에서 공통으로 권하는 첫소리도 알려 주세요.",
    ]
    cases=[]
    for i,q in enumerate(questions,1):
        v=_normalized_vector(model.embed_query(dense_query_text(q,"nella-fantasia")))
        pool=[{"record":r,"relevance":sum(a*b for a,b in zip(v,d)),"scope":"unspecified_scope" if r["evidence_type"]=="sumijo_feedback" else "local_example"}
              for r,d in zip(records,vectors)]
        cases.append({"id":f"observed-p-pp-{i}","kind":"observed_disagreement","piece":"nella-fantasia",
            "question":q,"target_id":"sumijo-nella-fantasia-036-ku-01","measure_ranges":[],"budgets":[2],
            "cluster":"observed-p-pp","observed_conflict":True,
            "reference_requirements":"Distinguish Sumi Jo's PP from the expert's p; do not invent an edition/error explanation or claim to have resolved the score marking. Both advise a quiet/soft opening.",
            "reference_answer":"Sumi Jo source says PP; expert source says p. The corpus alone does not establish why they differ or confirm the score. Both recommend avoiding an overly loud first sound.",
            "pool":pool,"pool_hash":run.digest(pool)})
    data=copy.deepcopy(main)
    data.update(parent_contract_id=main["contract_id"],cases=cases,
        sensitivity="Post-hoc observed source disagreement; no synthetic claim or altered canonical answer. Two phrasings share one source cluster; not independent test cases.")
    data.pop("contract_id");data["contract_id"]=run.digest(data)
    run.write(root / "frozen.json",data)
    return root


if __name__ == "__main__":
    import sys
    if sys.argv[1]=="prepare":
        print(prepare())
    else:
        run.ROOT=run.ROOT / "observed_disagreement"
        run.generate_all()
