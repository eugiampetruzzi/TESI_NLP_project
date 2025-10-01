#!/usr/bin/env python3
# Author: Eugenia Giampetruzzi
# Date: 2025-10-01
# Description: Sentence-level LDA (~500 topics) + participant aggregation + PCA (<=200)

import argparse, os, re, glob, sqlite3, subprocess, pandas as pd, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def build_db(src, db):
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r'^(ELS[_-]?(\d+))[_-].*participant[_-]?parsing', re.I)
    rows = []
    for fp in sorted(glob.glob(os.path.join(src, "ELS_*participant_parsing*.txt"))):
        fn = os.path.basename(fp); m = pat.match(os.path.splitext(fn)[0])
        if not m: continue
        txt = open(fp, encoding="utf-8", errors="ignore").read().strip().lower()
        # very simple sentence split; fine for LDA pre-chunking
        sents = re.split(r'(?<=[.!?])\s+|\n+', txt)
        for i, sent in enumerate(s for s in (s.strip() for s in sents) if s):
            rows.append(dict(
                message_id=f"{fn}_s{i}",
                user_id=m.group(1),
                session_id="stress_interview",
                message=sent
            ))
    con = sqlite3.connect(db)
    pd.DataFrame(rows).to_sql("msgs", con, if_exists="replace", index=False)
    con.close()
    print(f"[db] loaded {len(rows)} sentences -> {db}")

def run_dlatk_ngrams(dbstem):
    run([
        "dlatkInterface.py","--db_engine","sqlite",
        "-d",dbstem,"-t","msgs","-c","message_id",
        "--group_freq_thresh","0","--add_ngrams","-n","1"
    ])

def run_dlatk_lda(dbstem, mallet, k, alpha, stop, lex, outdir):
    run([
        "dlatkInterface.py","--db_engine","sqlite",
        "-d",dbstem,"-t","msgs","-c","message_id",
        "--feat_table","feat$1gram$msgs$message_id",
        "--estimate_lda_topics",
        "--num_stopwords",str(stop),
        "--num_topics",str(k),
        "--lda_alpha",str(alpha),
        "--lexicondb",dbstem,
        "--lda_lexicon_name",lex,
        "--save_lda_files",outdir,
        "--mallet_path",mallet
    ])

def parse_doctopics_to_df(path_txt, k):
    """
    Robust MALLET doctopics parser.
    Produces DataFrame with columns: message_id, user_id?, topic0..topic{k-1}
    """
    rows = []
    with open(path_txt, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): 
                continue
            parts = line.strip().split()
            # common MALLET format: <docid> <source> <t0> <p0> <t1> <p1> ...
            # sometimes: <docid> <source> <p0> <p1> ... <pK-1>
            doc = parts[1] if len(parts) > 1 else parts[0]
            tail = parts[2:] if len(parts) > 2 else []
            vec = np.zeros(k, dtype=float)
            if any(":" in t for t in tail):
                # key:value format (rare)
                for kv in tail:
                    if ":" not in kv: 
                        continue
                    tid, val = kv.split(":")
                    tid = int(tid)
                    if 0 <= tid < k:
                        vec[tid] = float(val)
            else:
                # either alternating topic prob pairs, or just probs
                if len(tail) == 2*k and all(_is_int(tail[i]) for i in range(0, 2*k, 2)):
                    for i in range(0, 2*k, 2):
                        tid = int(tail[i]); val = float(tail[i+1])
                        if 0 <= tid < k:
                            vec[tid] = val
                else:
                    # assume straight list of K probs
                    for tid, val in enumerate(tail[:k]):
                        vec[tid] = float(val)
            s = vec.sum()
            if s > 0: vec = vec / s
            rows.append((doc, *vec.tolist()))
    cols = ["message_id"] + [str(i) for i in range(k)]
    return pd.DataFrame(rows, columns=cols)

def _is_int(x):
    try:
        int(x); return True
    except: 
        return False

def pca_reduce(doc_topics_df, db_path, out_csv, n_comp=200):
    # attach user_id from sqlite (works for our sentence IDs too)
    con = sqlite3.connect(db_path)
    ids = pd.read_sql('select message_id,user_id from msgs', con)
    con.close()
    df = doc_topics_df.merge(ids, on="message_id", how="left")
    meta = df[["message_id","user_id"]]
    X = df.drop(columns=["message_id","user_id"], errors="ignore")
    n = min(n_comp, max(1, min(X.shape[0]-1, X.shape[1])))
    pca = PCA(n_components=n)
    Z = pca.fit_transform(X.values)
    out = pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(Z, columns=[f"pc{i+1}" for i in range(Z.shape[1])])],
        axis=1
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[pca] wrote {out_csv} with {Z.shape[1]} components; var_explained={pca.explained_variance_ratio_.sum():.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_text")
    ap.add_argument("--db", default="sqlite_data/child_stress.db")
    ap.add_argument("--db_stem", default="child_stress")
    ap.add_argument("--mallet", default=os.path.expanduser("~/.local/mallet/bin/mallet"))
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--stop", type=int, default=200)
    ap.add_argument("--lex", default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    lex = args.lex or f"SENT_{args.k}_a{int(args.alpha)}_s{args.stop}"

    # 1) sentence db
    build_db(args.src, args.db)
    # 2) 1-grams
    run_dlatk_ngrams(args.db_stem)
    # 3) LDA ~500 topics
    run_dlatk_lda(args.db_stem, args.mallet, args.k, args.alpha, args.stop, lex, args.outdir)

    # 4) parse doctopics -> csv + PCA
    doctopics = os.path.join(args.outdir, "a29fc_doctopics.txt")
    raw_out = f"exports/{lex}_sent_doc_topics.csv"
    if not os.path.exists(doctopics):
        raise SystemExit(f"[error] missing {doctopics}")
    dt = parse_doctopics_to_df(doctopics, args.k)
    Path("exports").mkdir(exist_ok=True)
    dt.to_csv(raw_out, index=False)
    print(f"[export] saved {raw_out}  shape={dt.shape}")
    pca_reduce(dt, args.db, f"exports/{lex}_sent_doc_topics_pca.csv", n_comp=200)

if __name__ == "__main__":
    main()
