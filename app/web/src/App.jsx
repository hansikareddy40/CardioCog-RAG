import React, { useEffect, useMemo, useRef, useState } from "react";

const pct = (p) => (p < 0.01 ? "under 1%" : `${Math.round(p * 100)}%`);

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.json();
}

function Field({ field, value, onChange, typical }) {
  if (field.type === "checkbox") {
    return (
      <label className="tick">
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        <span>{field.label}</span>
      </label>
    );
  }
  return (
    <label className="field">
      <span className="field-label">{field.label}</span>
      {field.type === "select" ? (
        <select value={value ?? field.default} onChange={(e) => onChange(e.target.value)}>
          {field.options.map((o) => <option key={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type="number" min={field.min} max={field.max} step={field.step} value={value ?? ""} placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        />
      )}
      {typical !== undefined && <span className="hint">typical for this person: about {typical}</span>}
    </label>
  );
}

function FieldGroups({ fields, values, set, typical }) {
  // Fields that share a group title are shown together under that title.
  const groups = [];
  for (const f of fields) {
    const last = groups[groups.length - 1];
    if (last && last.title === (f.group || null)) last.fields.push(f);
    else groups.push({ title: f.group || null, fields: [f] });
  }
  return groups.map((g, i) => (
    <div key={i} className="group">
      {g.title && <div className="group-title">{g.title}</div>}
      <div className={g.fields.every((f) => f.type === "checkbox") ? "grid ticks" : "grid"}>
        {g.fields.map((f) => (
          <Field key={f.id} field={f} value={values[f.id]} onChange={(v) => set(f.id, v)} typical={typical?.[f.id]} />
        ))}
      </div>
    </div>
  ));
}

function Section({ number, section, on, toggle, values, set, typical }) {
  const [open, setOpen] = useState(false);
  const main = section.fields.filter((f) => !f.more);
  const more = section.fields.filter((f) => f.more);
  const active = section.required || on;
  return (
    <section className={active ? "card" : "card off"}>
      <header>
        <div>
          <h2><span className="num">{number}</span>{section.title}</h2>
          <p className="note">{section.note}</p>
        </div>
        {!section.required && (
          <label className="switch">
            <input type="checkbox" checked={on} onChange={(e) => toggle(e.target.checked)} />
            <span>{on ? "Available" : "Not available"}</span>
          </label>
        )}
      </header>
      {active && (
        <>
          <FieldGroups fields={main} values={values} set={set} typical={typical} />
          {more.length > 0 && (
            <>
              <button className="link" onClick={() => setOpen(!open)}>{open ? "Hide" : "Show"} {more.length} optional items</button>
              {open && (
                <div className="more">
                  {section.more_note && <p className="note">{section.more_note}</p>}
                  <FieldGroups fields={more} values={values} set={set} typical={typical} />
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}

function Bars({ groups }) {
  const max = Math.max(0.5, ...groups.map((g) => Math.abs(g.effect)));
  return (
    <div className="bars">
      {[...groups].sort((a, b) => b.effect - a.effect).map((g) => (
        <div className="bar-row" key={g.name}>
          <span className="bar-name">{g.name}</span>
          <div className="bar-track">
            <div className="bar-mid" />
            <div
              className={g.effect >= 0 ? "bar up" : "bar down"}
              style={{ width: `${(Math.abs(g.effect) / max) * 50}%`, [g.effect >= 0 ? "left" : "right"]: "50%" }}
            />
          </div>
          <span className="bar-dir">{Math.abs(g.effect) < 0.05 ? "no effect" : g.effect > 0 ? "raises" : "lowers"}</span>
        </div>
      ))}
    </div>
  );
}

function Result({ res }) {
  const blocking = res.checks.filter((c) => c.level === "check");
  const notes = res.checks.filter((c) => c.level === "note");
  const s = res.similar;
  return (
    <div className="result">
      {blocking.length > 0 && (
        <div className="alert stop">
          <strong>Check these entries before reading the estimate.</strong>
          <ul>{blocking.map((c, i) => <li key={i}>{c.message}</li>)}</ul>
        </div>
      )}
      <div className="numbers">
        <div className={blocking.length ? "big doubtful" : "big"}>
          <span className="label">Estimated 3-year dementia risk</span>
          <span className="value">{pct(res.risk)}</span>
        </div>
        <div className="big ref">
          <span className="label">What happened to similar participants</span>
          <span className="value">{pct(s.rate)}</span>
        </div>
      </div>
      <p className="note">
        Right: of {s.n.toLocaleString()} research participants aged {s.age_group.toLowerCase()}
        {s.status ? `, status ${s.status}` : ", any status"}, {s.events.toLocaleString()} were diagnosed with dementia within 3 years.
        Across everyone without dementia it was {pct(res.cohort_rate)}.
      </p>
      {res.risk_without_pet !== null && (
        <p className="note">Without the PET numbers: {pct(res.risk_without_pet)}. With them: {pct(res.risk)}. Treat the two as similar.</p>
      )}
      {notes.map((c, i) => <div className="alert info" key={i}>{c.message}</div>)}
      {res.no_cognitive_information && (
        <div className="alert warn">
          Without any thinking or memory information the estimate is little more than an average for the person's age. Do not read it as an
          individual risk.
        </div>
      )}

      <h3>Information used</h3>
      <p className="chips">
        {res.used.map((u) => <span className="chip" key={u}>{u}</span>)}
        {res.missing.map((u) => <span className="chip none" key={u}>{u}: not available</span>)}
      </p>
      <p className="note">{res.features_filled} of the model's {res.features_total} inputs are filled in.</p>

      <h3>What moved this estimate</h3>
      <Bars groups={res.groups} />
      <p className="note">
        How much each kind of information pushed the estimate up or down compared with an average participant. This describes what the model
        used. It does not show causes.
      </p>

      {res.domain_scores.length > 0 && (
        <>
          <h3>Test scores compared with healthy people of the same age, sex and education</h3>
          <table>
            <tbody>
              {res.domain_scores.map((d) => (
                <tr key={d.name}>
                  <td>{d.name}</td>
                  <td className="r">{d.z > 0 ? "+" : ""}{d.z.toFixed(1)}</td>
                  <td>{d.z <= -2 ? "well below expected" : d.z <= -1 ? "below expected" : d.z < 1 ? "as expected" : "above expected"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">0 is typical; each step of 1 is one standard deviation. These, not the raw scores, are what the model sees.</p>
        </>
      )}

      {res.items.length > 0 && (
        <>
          <h3>Single items with the largest effect</h3>
          <table>
            <tbody>
              {res.items.map((it) => (
                <tr key={it.name}><td>{it.name}</td><td className="r">{it.value}</td><td>{it.effect > 0 ? "raises" : "lowers"}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function Evidence({ context }) {
  const [q, setQ] = useState("");
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const ask = async (e) => {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    try { setOut(await post("/api/ask", { question: q, context })); } catch (err) { setOut({ answer: String(err), sources: [] }); }
    setBusy(false);
  };
  return (
    <section className="card evidence">
      <h2>Ask the evidence library</h2>
      <p className="note">
        Answers come only from a small library of open-access papers and are shown with their sources. This is general published information.
        It is not advice about this person, and it cannot recommend treatment.
      </p>
      <form onSubmit={ask}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="For example: Is midlife hypertension associated with later dementia?" />
        <button disabled={busy}>{busy ? "Searching" : "Ask"}</button>
      </form>
      {out && (
        <div className="answer">
          <p>{out.answer}</p>
          {out.mode === "extractive" && <p className="note">Shown as direct quotations from the sources.</p>}
          {out.sources.map((s) => (
            <details key={s.n}>
              <summary>[{s.n}] {s.short} ({s.year}){s.section ? `, section: ${s.section}` : ""}</summary>
              <p>{s.passage}</p>
              <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [schema, setSchema] = useState(null);
  const [values, setValues] = useState({});
  const [on, setOn] = useState({});
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  const latest = useRef(0);

  useEffect(() => {
    fetch("/api/schema").then((r) => r.json()).then((s) => {
      const v = {}, o = {};
      for (const sec of s.sections) {
        o[sec.id] = !!sec.default_on;
        for (const f of sec.fields) v[f.id] = f.default ?? null;
      }
      setSchema(s); setValues(v); setOn(o);
    }).catch((e) => setError(String(e)));
  }, []);

  const ready = schema && values.age != null && values.educ_years != null;
  useEffect(() => {
    if (!ready) return;
    const id = ++latest.current;
    const t = setTimeout(() => {
      post("/api/predict", { values, on })
        .then((r) => { if (id === latest.current) { setRes(r); setError(null); } })
        .catch((e) => setError(String(e)));
    }, 200);
    return () => clearTimeout(t);
  }, [values, on, ready]);

  const context = useMemo(() => {
    if (!res) return "";
    const top = [...res.groups].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).slice(0, 2).map((g) => g.name);
    return `estimated 3-year dementia risk ${Math.round(res.risk * 100)}%; categories with the largest influence: ${top.join(", ")}`;
  }, [res]);

  if (error && !schema) return <main><div className="alert stop">Cannot reach the model API. Start it with: python -m uvicorn app.api:app --port 8000<br />{error}</div></main>;
  if (!schema) return <main><p>Loading</p></main>;

  const set = (id, v) => setValues((old) => ({ ...old, [id]: v }));
  return (
    <main>
      <h1>CardioCog: 3-year dementia risk estimate</h1>
      <div className="alert banner">
        <strong>Research prototype. Not for clinical use.</strong> For a person who does <strong>not</strong> have dementia today, it
        estimates the chance of a dementia diagnosis within 3 years, based on volunteers in the NACC research cohort (median age 71, mostly
        White, highly educated). Tested on past research data only. It does not diagnose and does not recommend treatment.
      </div>
      <p className="note lead">
        The sections below are the kinds of information the model was trained on, and nothing else. Fill in section 1, mark the other
        sections as available or not, and leave unknown boxes empty. Empty boxes are passed to the model as "not available".
      </p>
      <div className="layout">
        <div className="inputs">
          {schema.sections.map((sec, i) => (
            <Section
              key={sec.id} number={i + 1} section={sec} on={!!on[sec.id]} toggle={(v) => setOn((o) => ({ ...o, [sec.id]: v }))}
              values={values} set={set} typical={sec.id === "tests" ? res?.typical_scores : undefined}
            />
          ))}
        </div>
        <aside>
          <div className="card sticky">
            <h2>Estimate</h2>
            {!ready && <div className="alert warn">Enter age and years of education.</div>}
            {error && <div className="alert stop">{error}</div>}
            {ready && res && <Result res={res} />}
          </div>
        </aside>
      </div>
      <Evidence context={context} />
      <footer className="note">
        Model: gradient-boosted trees trained on {schema.reference.n_train.toLocaleString()} research participants who did not have dementia
        at their first visit, with whole sections hidden at random during training so that it works with any combination. Tested on held-out
        participants: AUROC 0.94 for everyone together and 0.95 at nine held-out centres; about 0.84 among people with MCI and about 0.88
        among cognitively normal people (few cases). No entries are stored and nothing leaves this computer.
      </footer>
    </main>
  );
}
