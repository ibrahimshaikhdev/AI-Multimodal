"""v2: builds train.jsonl / val.jsonl for ONE model doing all jobs. Prompts match app.py exactly.
Jobs: report summary (up to 20 params), research RAG (3-6 longer sources), research comparison (2-4 papers, long text),
long-paper map-reduce (section extraction, merge, compare-from-facts), and the project/patient assistant.
Edit KB below to match your real project. Run: python make_dataset.py"""
import json, random
rnd = random.Random(42)
SYSTEM = ("You are the built-in assistant of the project 'Multimodal AI Research & Medical Report Intelligence'. "
          "Answer questions about the project using ONLY the PROJECT KNOWLEDGE provided. For questions about a patient, use ONLY the "
          "PATIENT DATA provided. If the answer is not in the provided context, say you don't have that information. "
          "Never diagnose or prescribe; for personal medical decisions tell the user to consult a doctor. Be concise.")
TESTS = [("Hemoglobin","g/dL",13,17),("Fasting Blood Sugar","mg/dL",70,100),("HbA1c","%",4.0,5.6),("LDL Cholesterol","mg/dL",0,100),
 ("Serum Creatinine","mg/dL",0.6,1.3),("SGPT (ALT)","U/L",7,56),("Vitamin D","ng/mL",30,100),("Total Cholesterol","mg/dL",0,200),
 ("Triglycerides","mg/dL",0,150),("Blood Urea","mg/dL",15,40),("WBC Count","/uL",4000,11000),("Platelet Count","lakh/uL",1.5,4.1),
 ("RBC Count","million/uL",4.5,5.5),("Hematocrit (PCV)","%",40,50),("SGOT (AST)","U/L",10,40),("HDL Cholesterol","mg/dL",40,60),
 ("Post-Prandial Sugar","mg/dL",70,140),("TSH","uIU/mL",0.4,4.0),("Sodium","mmol/L",135,145),("Potassium","mmol/L",3.5,5.0)]
NAMES = ["Fahad Shaikh","Rohan Rao","Meera Reddy","Kabir Patel","Vihaan Sharma","Ananya Iyer","Aarav Mehta","Atharva Patil",
 "Ibrahim Shaikh","Sara Khan","Neha Joshi","Arjun Nair","Diya Kulkarni","Imran Ansari","Pooja Desai","Karan Malhotra"]
DATES = ["2026-03-12","2026-06-20","2026-10-02","2025-11-05","2026-01-18"]
ex = []
def add(user, assistant, system=None):
    ex.append({"messages": ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user},{"role":"assistant","content":assistant}]})
def flag(v, lo, hi): return "low" if v < lo else "high" if v > hi else "normal"
def val(lo, hi, state):
    if state == "high": return round(hi * rnd.uniform(1.06, 1.6), 1)
    if state == "low" and lo > 0: return round(lo * rnd.uniform(0.5, 0.93), 1)
    return round(rnd.uniform(lo + 0.1*(hi-lo), hi - 0.1*(hi-lo)) if lo > 0 else rnd.uniform(0.3*hi, 0.92*hi), 1)
def rstate(): return rnd.choices(["normal","high","low"], [60,25,15])[0]

# ---------- Job 1: report comparison summary (handles long diffs by condensing) ----------
def summarize(diff):
    imp, wor, sta = [], [], []
    for r in diff:
        o, n, nm, a, b, u = r["old_flag"], r["new_flag"], r["name"], r["old"], r["new"], r["unit"]; span = f"({a} to {b} {u})"
        if o == n == "normal": sta.append((nm, f"* **{nm}:** Stable, remaining in the normal range {span}."))
        elif o == n:
            if a == b: sta.append((nm, f"* **{nm}:** Unchanged and still {o} {span}."))
            elif (o == "high" and b < a) or (o == "low" and b > a): imp.append(f"* **{nm}:** Still {o} but moving toward normal {span}.")
            else: wor.append(f"* **{nm}:** Still {o} and moving further from normal {span}.")
        elif n == "normal": imp.append(f"* **{nm}:** Improved from {o} to normal {span}.")
        elif o == "normal": wor.append(f"* **{nm}:** Worsened from normal to {n} {span}.")
        else: wor.append(f"* **{nm}:** Changed from {o} to {n} {span}.")
    out = []
    for title, rows, cap in (("Improved", imp, 4), ("Worsened", wor, 6)):
        if rows:
            out += [f"**{title}:**"] + rows[:cap]
            if len(rows) > cap: out.append(f"* ...and {len(rows)-cap} more {title.lower()} parameters.")
            out.append("")
    if sta:
        out.append("**Stable:**")
        if len(sta) <= 3: out += [l for _, l in sta]
        else: out.append(f"* {len(sta)} parameters stayed stable: " + ", ".join(n for n, _ in sta[:5]) + (f" and {len(sta)-5} more." if len(sta) > 5 else "."))
        out.append("")
    return "\n".join(out + ["This is a data summary, not a diagnosis."])
for _ in range(1500):
    k = rnd.choice([rnd.randint(3, 8)]*7 + [rnd.randint(9, 15)]*2 + [len(TESTS)]); diff = []
    for nm, u, lo, hi in rnd.sample(TESTS, k):
        a, b = val(lo, hi, rstate()), val(lo, hi, rstate())
        if rnd.random() < .15: b = a
        diff.append(dict(name=nm, unit=u, old=a, new=b, delta=round(b-a, 2), old_flag=flag(a, lo, hi), new_flag=flag(b, lo, hi)))
    add("Summarize changes between two lab reports (earlier -> later) in under 150 words. "
        "Use ONLY this data; say improved/worsened/stable; no diagnosis.\n" + json.dumps(diff), summarize(diff))

# ---------- synthetic papers (sentences tagged by field) ----------
MODELS = ["ResNet-50","EfficientNet-B3","DenseNet-121","BioBERT","ClinicalBERT","a 3-layer CNN","a Vision Transformer","XGBoost","a Random Forest","U-Net"]
DATA = ["a public chest X-ray dataset","a synthetic lab-report corpus","a retinal image collection","an ICU vitals dataset","a clinical notes corpus","a skin-lesion image set"]
METRICS = ["accuracy","F1-score","AUROC","precision","recall"]
LIMITS = ["the dataset is small","the data comes from a single hospital","the model was not validated externally","labels were noisy","class imbalance remains an issue"]
FUTURE = ["validate on multi-center data","add explainability methods","test larger models","extend to more modalities","deploy a clinical pilot"]
FILL = ["Images were resized to {r} x {r} pixels before training.","Intensity values were normalised to zero mean and unit variance.",
 "Random flips and small rotations were used for augmentation.","A fixed random seed ({r}) was used to support reproducibility.",
 "Ethical approval for data use was obtained from the institutional review board.","Related work on this topic is discussed in the previous section.",
 "The source code and configuration files are available on request.","Missing values were handled by median imputation.",
 "Three annotators reviewed a random sample of {r} cases for quality control.","Baseline methods from earlier studies were re-implemented for comparison.",
 "All experiments were repeated {r} times and averaged.","Statistical significance was assessed with a paired test."]
F = ["methodology","dataset","model","results","limitations","future_work"]
def paper(i, extra):
    p = dict(id=i, model=rnd.choice(MODELS), data=rnd.choice(DATA), n=rnd.choice([800,1500,5000,12000,48000]), metric=rnd.choice(METRICS),
             score=round(rnd.uniform(.72,.97), 2), lim=rnd.choice(LIMITS), fut=rnd.choice(FUTURE), split=rnd.choice(["80/20","70/30","5-fold"]))
    p["title"] = f"{p['model'].replace('a ','').title()} for {p['data'].split()[-2].title()} Analysis {rnd.randint(2019,2026)}"
    p["has_lim"], p["has_fut"] = rnd.random() > .3, rnd.random() > .3
    s = [("mm", f"We train {p['model']} with a {p['split']} split."), ("dataset", f"The study uses {p['data']} with {p['n']} samples."),
         ("results", f"The approach reaches a {p['metric']} of {p['score']}.")]
    if p["has_lim"]: s.append(("limitations", f"A limitation is that {p['lim']}."))
    if p["has_fut"]: s.append(("future_work", f"Future work will {p['fut']}."))
    s += [(None, rnd.choice(FILL).format(r=rnd.choice([3,5,10,42,128,224,256]))) for _ in range(extra)]
    rnd.shuffle(s); p["sents"] = s; p["text"] = " ".join(t for _, t in s); return p
def fv(p):
    return {"methodology": f"{p['model']} trained with a {p['split']} split", "dataset": f"{p['data']} ({p['n']} samples)",
            "model": p["model"][2:] if p["model"].startswith("a ") else p["model"], "results": f"{p['metric']} of {p['score']}",
            "limitations": p["lim"] if p["has_lim"] else "Not stated", "future_work": p["fut"] if p["has_fut"] else "Not stated"}
def pj(p): return {"id": p["id"], "title": p["title"], **fv(p)}
schema = '{"papers":[{"id":int,"title":str,' + ",".join(f'"{x}":str' for x in F) + '}],"overall":str}'
def overall(ps): return "; ".join(f"Paper {p['id']} uses {p['model']} on {p['data']} ({p['metric']} {p['score']})" for p in ps) + "."

# ---------- Job 2: research RAG (3-6 sources) ----------
for _ in range(900):
    t = paper(0, rnd.randint(1, 4)); order = [t] + [paper(i, rnd.randint(1, 4)) for i in range(1, rnd.randint(3, 6))]
    rnd.shuffle(order); idx = order.index(t) + 1
    ctx = "\n\n".join(f"[S{i+1}] ({q['title']}, p.{rnd.randint(1,9)}) {q['text']}" for i, q in enumerate(order))
    kind = rnd.choice(["model","data","score","lim","fut","none","none"])
    if kind == "model": q, a = f"Which model did '{t['title']}' use?", f"It used {t['model']} [S{idx}]."
    elif kind == "data": q, a = f"How large was the dataset in '{t['title']}'?", f"The dataset contained {t['n']} samples [S{idx}]."
    elif kind == "score": q, a = f"What {t['metric']} did '{t['title']}' report?", f"It reported a {t['metric']} of {t['score']} [S{idx}]."
    elif kind == "lim" and t["has_lim"]: q, a = f"What limitation does '{t['title']}' mention?", f"It notes that {t['lim']} [S{idx}]."
    elif kind == "fut" and t["has_fut"]: q, a = f"What future work does '{t['title']}' plan?", f"It plans to {t['fut']} [S{idx}]."
    else: q, a = rnd.choice(["What learning rate was used?","How many GPUs were used for training?","Who funded the study?","What was the batch size?"]), "The provided sources do not contain this information."
    add(f"Answer ONLY from the sources below. Cite like [S1]. If they do not contain the answer, say so.\n\nSOURCES:\n{ctx}\n\nQUESTION: {q}", a)

# ---------- Job 3: research comparison, 2-4 long papers (<= 3500 chars each, like the app) ----------
for _ in range(500):
    ids = rnd.sample(range(1, 13), rnd.randint(2, 4)); ps = [paper(i, rnd.randint(6, 26)) for i in ids]
    docs = "\n\n".join(f"### PAPER {p['id']}: {p['title']}\n{p['text']}" for p in ps)
    add(f'Compare these papers using ONLY their text. Return ONLY JSON like {schema}. Write "Not stated" when absent.\n\n{docs}',
        json.dumps({"papers": [pj(p) for p in ps], "overall": overall(ps)}))

# ---------- Job 4: long-paper map-reduce ----------
fschema = '{' + ",".join(f'"{x}":str' for x in F) + '}'
for _ in range(600):   # map: facts from ONE section
    p = paper(0, rnd.randint(12, 30)); sec = sorted(rnd.sample(range(len(p["sents"])), rnd.randint(5, min(14, len(p["sents"])))))
    facts = {x: "Not stated" for x in F}; full = fv(p)
    for j in sec:
        f = p["sents"][j][0]
        if f == "mm": facts["methodology"], facts["model"] = full["methodology"], full["model"]
        elif f: facts[f] = full[f]
    add(f'Extract facts from this section of a research paper using ONLY the text. Return ONLY JSON like {fschema}. Write "Not stated" when absent.\n\nSECTION:\n' +
        " ".join(p["sents"][j][1] for j in sec), json.dumps(facts))
for _ in range(300):   # merge partial extractions of one paper
    p = paper(0, 3); full = fv(p); m = rnd.randint(2, 4); parts = [{x: "Not stated" for x in F} for _ in range(m)]
    for x in F:
        if full[x] != "Not stated":
            for k in rnd.sample(range(m), rnd.randint(1, min(2, m))): parts[k][x] = full[x]
    add('Merge these partial extractions of ONE paper into a single JSON object. For each field keep the value that is not "Not stated"; '
        f'if every partial says "Not stated" keep "Not stated". Return ONLY JSON like {fschema}.\n\nPARTIAL EXTRACTIONS:\n' + json.dumps(parts), json.dumps(full))
for _ in range(300):   # reduce: compare papers from their merged facts
    ids = rnd.sample(range(1, 13), rnd.randint(2, 4)); ps = [paper(i, 2) for i in ids]
    docs = "\n\n".join(f"### PAPER {p['id']}: {p['title']}\n{json.dumps(fv(p))}" for p in ps)
    add(f'Compare these papers using ONLY their extracted facts. Return ONLY JSON like {schema}. Write "Not stated" when absent.\n\n{docs}',
        json.dumps({"papers": [pj(p) for p in ps], "overall": overall(ps)}))

# ---------- Job 5: assistant (project Q&A + lab terms + patient data) ----------
KB = [  # EDIT to match your real project: (topic, text, [question variants])
 ("purpose","The project is a Flask web app that organises medical reports and research papers, extracts structured data with OCR, compares reports over time, and answers questions about research papers using RAG. It supports information retrieval and does not diagnose.",
  ["What does this project do?","Explain the project in short.","What is this project about?","What is the purpose of this app?"]),
 ("report_compare","Report comparison: two reports of one patient are ordered by date. Parameter differences (old value, new value, change, normal/high/low) are computed in Python; the AI only writes a short summary.",
  ["How does report comparison work?","Who computes the differences in report comparison?","What does the AI do in report comparison?"]),
 ("research_compare","Research comparison compares uploaded papers on methodology, dataset, model, results, limitations and future work and shows them in a table. Long papers are read in sections and merged before comparing.",
  ["What does research comparison do?","Which fields are compared between papers?","How are research papers compared?","How are long papers handled?"]),
 ("rag","Research RAG splits uploaded papers into chunks, retrieves the most relevant chunks for a question, and answers only from them with citations like [S1].",
  ["What is research RAG in this project?","How does the RAG feature answer questions?","Why does the RAG page show sources?"]),
 ("stack","Tech stack: Python, Flask, PostgreSQL, SQLAlchemy, PyMuPDF, Tesseract OCR, scikit-learn TF-IDF retrieval, JWT authentication, and a plain HTML/JavaScript frontend.",
  ["What technologies does the project use?","Which database is used?","What is the tech stack?","Which OCR is used?"]),
 ("privacy","Privacy: the project runs its own fine-tuned open model locally, so patient and project data is not sent to third-party AI services. Only synthetic or public data is used for training.",
  ["How is data privacy handled?","Does patient data go to a third party?","Which model does the project use?"]),
 ("limits","Limits: AI output can be wrong and needs human review. The project is a research prototype and is not for clinical use or diagnosis.",
  ["Can this app diagnose diseases?","Is the AI output always correct?","Is this app ready for clinical use?"]),
 ("ocr","OCR: PDFs are read with PyMuPDF, and scanned pages or images (JPG, PNG) fall back to Tesseract OCR. The extracted text is stored in the database.",
  ["How is text extracted from reports?","Which file types can be uploaded?","What happens to scanned documents?"]),
 ("timeline","Health timeline: a patient's documents and extracted events are arranged chronologically.",["What is the health timeline?","How are reports ordered over time?"]),
]
J = "\n---\n"
for i, (_, text, qs) in enumerate(KB):
    for q in qs:
        for _ in range(5):
            chunks = [text] + rnd.sample([t for j, (_, t, _) in enumerate(KB) if j != i], rnd.randint(0, 3)); rnd.shuffle(chunks)
            add(f"PROJECT KNOWLEDGE:\n{J.join(chunks)}\n\nPATIENT DATA:\n(none)\n\nQUESTION: {q}", text, SYSTEM)
for q in ["What is the capital of France?","Who won the last cricket match?","What is the weather today?","Write me a poem.","What is the price of Bitcoin?"]:
    for _ in range(8): add(f"PROJECT KNOWLEDGE:\n{rnd.choice(KB)[1]}\n\nPATIENT DATA:\n(none)\n\nQUESTION: {q}", "I don't have that information in the project knowledge.", SYSTEM)
TERMS = {"Hemoglobin":"Hemoglobin is the protein in red blood cells that carries oxygen around the body.","HbA1c":"HbA1c reflects average blood sugar over roughly the past two to three months.",
 "LDL Cholesterol":"LDL is often called 'bad' cholesterol because high levels are linked to a higher risk of heart disease.","Serum Creatinine":"Creatinine is a waste product filtered by the kidneys, so it is used to assess kidney function.",
 "SGPT (ALT)":"SGPT (ALT) is a liver enzyme; raised levels can indicate liver stress or injury.","Vitamin D":"Vitamin D helps the body absorb calcium and supports bone health.",
 "Fasting Blood Sugar":"Fasting blood sugar is the glucose level measured after not eating for several hours.","Triglycerides":"Triglycerides are a type of fat in the blood; high levels are linked to heart-disease risk."}
for nm, d in TERMS.items():
    for q in (f"What is {nm}?", f"Explain {nm} in simple words.", f"What does {nm} measure?"):
        add(f"PROJECT KNOWLEDGE:\n(nothing relevant found)\n\nPATIENT DATA:\n(none)\n\nQUESTION: {q}", d + " This is general information, not a diagnosis.", SYSTEM)
for _ in range(600):
    nm = rnd.choice(NAMES); tests = rnd.sample(TESTS, rnd.randint(4, 12)); vals = {t: val(lo, hi, rstate()) for t, u, lo, hi in tests}
    ctx = f"{nm}, latest report ({rnd.choice(DATES)}):\nMediCare Diagnostics Lab\nPatient: {nm}\n" + "\n".join(f"{t} {vals[t]} {u} {lo} - {hi}" for t, u, lo, hi in tests)
    t, u, lo, hi = rnd.choice(tests); v = vals[t]; f = flag(v, lo, hi); first = nm.split()[0]; kind = rnd.choice(["value","value","diag","abn"])
    if kind == "value": q, a = f"What is {first}'s {t} value?", f"{nm}'s {t} is {v} {u} (reference {lo} - {hi}), which is {f}."
    elif kind == "diag": q, a = f"Does {first} have a disease based on this report?", f"I can't diagnose. The report shows {t} at {v} {u}, which is {f} (reference {lo} - {hi}). Please consult a doctor to interpret the results."
    else:
        ab = [f"{x} ({vals[x]} {uu}, {flag(vals[x], l, h)})" for x, uu, l, h in tests if flag(vals[x], l, h) != "normal"]
        q, a = f"Which values are abnormal for {first}?", (f"Outside the reference range: {', '.join(ab)}." if ab else "All listed values are within the reference range.")
    add(f"PROJECT KNOWLEDGE:\n(nothing relevant found)\n\nPATIENT DATA:\n{ctx}\n\nQUESTION: {q}", a, SYSTEM)

rnd.shuffle(ex); cut = int(len(ex) * .95)
for fn, part in (("train.jsonl", ex[:cut]), ("val.jsonl", ex[cut:])):
    with open(fn, "w", encoding="utf-8") as f:
        for e in part: f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f"train={cut} val={len(ex)-cut}")
