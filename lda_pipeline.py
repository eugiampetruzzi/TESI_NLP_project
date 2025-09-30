
#!/usr/bin/env python3
# Author: Eugenia Giampetruzzi
# Date: 2025-09-30
# Description: LDA pipeline with DLATK + MALLET

import argparse, os, re, glob, sqlite3, subprocess, pandas as pd
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def build_db(src, db):
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r'^(ELS[_-]?(\d+))[_-].*participant[_-]?parsing', re.I)
    rows = []
    for fp in sorted(glob.glob(os.path.join(src, "ELS_*participant_parsing*.txt"))):
        fn = os.path.basename(fp); m = pat.match(os.path.splitext(fn)[0])
        if not m: continue
        txt = " ".join(open(fp, encoding="utf-8", errors="ignore").read().split()).lower()
        rows.append(dict(message_id=fn, user_id=m.group(1), session_id="stress_interview", message=txt))
    con = sqlite3.connect(db)
    pd.DataFrame(rows).drop_duplicates("message_id").to_sql("msgs", con, if_exists="replace", index=False)
    con.close()
    print(f"[db] loaded {len(rows)} docs -> {db}")

def run(cmd):
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True)

def run_dlatk_ngrams(dbstem):
    run(["dlatkInterface.py","--db_engine","sqlite","-d",dbstem,"-t","msgs","-c","message_id","--group_freq_thresh","0","--add_ngrams","-n","1"])

def run_dlatk_lda(dbstem, mallet, k, alpha, stop, lex, outdir):
    run(["dlatkInterface.py","--db_engine","sqlite","-d",dbstem,"-t","msgs","-c","message_id",
         "--feat_table","feat$1gram$msgs$message_id","--estimate_lda_topics",
         "--num_stopwords",str(stop),"--num_topics",str(k),"--lda_alpha",str(alpha),
         "--lexicondb",dbstem,"--lda_lexicon_name",lex,"--save_lda_files",outdir,"--mallet_path",mallet])

def parse_mallet_doctopics(path_txt, export_csv, db):
    rows=[]
    with open(path_txt, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts=line.strip().split(); 
            if len(parts)<3: continue
            doc=parts[1]; tail=parts[2:]; d={"message_id":doc}
            if any(':' in x for x in tail):
                it=iter(tail)
                for k,v in zip(it,it):
                    k=int(k.rstrip(':')); d[k]=float(v)
            else:
                for k,v in enumerate(tail): d[k]=float(v)
            rows.append(d)
    df=pd.DataFrame(rows).set_index("message_id").sort_index()
    df=df.div(df.sum(1), axis=0).fillna(0.0)
    Path(export_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(export_csv)
    # with ids
    con=sqlite3.connect(db)
    ids=pd.read_sql('select message_id,user_id from msgs', con).set_index("message_id")
    con.close()
    out=df.join(ids, how="left").reset_index()[["message_id","user_id"]+list(df.columns)]
    out.to_csv(export_csv.replace(".csv","_with_ids.csv"), index=False)
    print(f"[export] {export_csv}  shape={df.shape}")

def export_topic_words_from_sqlite(db, lex_freq_table, export_csv):
    con=sqlite3.connect(db)
    cols=[r[1] for r in con.execute(f'pragma table_info("{lex_freq_table}")')]
    term_col=next((c for c in ['word','ngram','feat','token','term'] if c in cols), None)
    if not term_col: raise SystemExit(f"no term-like col in {lex_freq_table}: {cols}")
    q=f'''select category as topic, {term_col} as term, weight
          from "{lex_freq_table}"
          order by cast(replace(category,'topic_','') as int), weight desc'''
    df=pd.read_sql(q, con); con.close()
    Path(export_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(export_csv, index=False)
    print(f"[export] {export_csv}  rows={len(df)}")

def make_wordclouds(topic_words_csv, outdir, grid=True):
    lex=pd.read_csv(topic_words_csv)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    topics=sorted(lex["topic"].unique(), key=lambda x: int(str(x).replace("topic_","")) if isinstance(x,str) else int(x))
    for t in topics:
        sub=lex[lex["topic"]==t]
        freqs=dict(zip(sub["term"], sub["weight"]))
        wc=WordCloud(width=800, height=400, background_color="white", colormap="Blues").generate_from_frequencies(freqs)
        fig=plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.title(f"topic {t}")
        fig.tight_layout(pad=0.5); fig.savefig(f"{outdir}/topic_{t}.png", dpi=150); plt.close(fig)
    if grid:
        import math
        n=len(topics); cols=5 if n>=15 else (4 if n>=9 else (3 if n>=5 else max(1,n))); rows=math.ceil(n/cols)
        fig,axes=plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False)
        for i,t in enumerate(topics):
            r,c=divmod(i,cols)
            sub=lex[lex["topic"]==t]; freqs=dict(zip(sub["term"], sub["weight"]))
            wc=WordCloud(width=600, height=300, background_color="white", colormap="Blues").generate_from_frequencies(freqs)
            ax=axes[r][c]; ax.imshow(wc, interpolation="bilinear"); ax.set_axis_off(); ax.set_title(f"topic {t}", fontsize=10)
        for k in range(len(topics), rows*cols): axes[divmod(k,cols)].set_visible(False)
        fig.tight_layout(pad=1.0); fig.savefig(f"{outdir}/topics_grid.png", dpi=200); plt.close(fig)
    print(f"[viz] saved wordclouds -> {outdir}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_text")
    ap.add_argument("--db", default="sqlite_data/child_stress.db")
    ap.add_argument("--db_stem", default="child_stress")
    ap.add_argument("--mallet", default=os.path.expanduser("~/.local/mallet/bin/mallet"))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--stop", type=int, default=200)
    ap.add_argument("--lex", default=None)
    ap.add_argument("--outdir", default="outputs")
    args=ap.parse_args()
    lex = args.lex or f"ELS_{args.k}_a{int(args.alpha)}_s{args.stop}"
    # 1) build db
    build_db(args.src, args.db)
    # 2) 1grams
    run_dlatk_ngrams(args.db_stem)
    # 3) lda
    run_dlatk_lda(args.db_stem, args.mallet, args.k, args.alpha, args.stop, lex, args.outdir)
    # 4) exports
    parse_mallet_doctopics(os.path.join(args.outdir, "a29fc_doctopics.txt"),
                           f"exports/{lex}_doc_topics.csv", args.db)
    export_topic_words_from_sqlite(args.db, f"{lex}_freq_t50ll", f"exports/{lex}_topic_words.csv")
    # 5) wordclouds
    make_wordclouds(f"exports/{lex}_topic_words.csv", f"{args.outdir}/wordclouds_py")

if __name__ == "__main__":
    main()

