# The whole project, explained simply

This note explains everything in the project from the data to the final tool, in plain words. Each part says **what we did**, **why**, and **what we found**. Technical words are explained the first time they appear.

---

## 1. What is this project about?

Some older people slowly lose their memory and thinking ability. When it becomes bad enough that they cannot manage daily life on their own, doctors call it **dementia**. Alzheimer's disease is the most common cause.

**Our question:** if a person does *not* have dementia today, can a computer look at their health information and estimate the chance that they will have dementia **within the next 3 years**?

We also wanted to know **which kinds of information actually help** to make that estimate, and to build a simple screen that a doctor could use.

**One important thing to remember from the start:** this is a research project. It is not a medical product. It cannot diagnose anyone and it does not suggest any medicine.

---

## 2. The data

### Where it comes from

The data come from **NACC**, a large research programme in the United States. About 46 research centres invite volunteers to come in roughly once a year. At every visit they record the same kinds of information.

### How much there is

| What | How many |
|---|---|
| People | 57,038 |
| Visits (one person comes many times) | 217,598 |
| People who came back at least once | 39,798 |
| People with brain-scan **numbers** | about 5,100 |
| People with actual brain-scan **images** | 446 |

### The five kinds of information

| Kind | Examples | Simple meaning |
|---|---|---|
| **Demographics** | Age, sex, years of education | Basic facts about the person |
| **Heart and blood vessel health** (cardiovascular) | Blood pressure, diabetes, stroke, heart disease, smoking | How healthy the heart and blood vessels are |
| **Thinking tests** (cognitive) | Memory tests, naming animals in one minute, connecting dots quickly | How well the brain is working right now |
| **Genes** | APOE | One gene that changes the risk of Alzheimer's |
| **Brain scans** (PET) | Amyloid and tau scans | Pictures showing harmful proteins that build up in the brain |

There is also a doctor's rating called **CDR**. It scores how much a person's memory and daily life are affected, from 0 (no problem) upward.

### Rules we followed with the data

* We never changed the original files. We only read them.
* The data are private. Nothing about any single person appears in our reports, only totals and averages.

---

## 3. Getting to know the data (Notebooks 01 to 04)

Before building anything, we looked carefully at what we had. Three things mattered most.

**1. "Missing" does not always mean the same thing.**
In this data the number 88 can be a real age (88 years old), but for "years of smoking" 88 means "does not apply, never smoked". If we treated every 88 as missing, we would delete every 88-year-old. So we wrote down the correct meaning for each item separately.

**2. The thinking tests changed in 2015.**
Before 2015 the centres used one set of tests, and after 2015 a different set. It is like a school changing its exam paper halfway through. We had to make the old and new scores comparable (explained in step 4).

**3. Brain scans mean two different things here.**
* For about 5,100 people we have scan **numbers**, already measured by experts.
* For only 446 people we have the actual scan **images**.

And of those 446 people, only **5** later developed dementia. This turned out to be very important (see step 10).

---

## 4. Cleaning the data and setting up a fair test (Notebook 05)

### Cleaning

* **Fixing missing-value codes** using the correct meaning for each item.
* **Making old and new tests comparable.** For every test we asked: "How does this person's score compare with a healthy person of the same age, sex and education?" The answer is a number called a **z-score**. Zero means "typical for a healthy person". Negative means "worse than expected". This works for any test, old or new.
* **Health history.** If a person once had a stroke, they have always "had a stroke" from then on. We carried such facts forward in time.

### Who we studied and what we predicted

* **Who:** people who did **not** have dementia at the visit where we make the prediction.
* **What:** did they get a dementia diagnosis within the next **3 years**? Yes or no.
* People we could not follow for 3 years were left out of this yes/no question, because we simply do not know their answer.

That left **19,470 people**, of whom **2,818** developed dementia within 3 years.

### A fair test

Think of a student preparing for an exam. If the student has already seen the exam questions, a high mark means nothing. It is the same for a computer model.

So we split the people into groups:

| Group | People | Used for |
|---|---|---|
| Training | 10,879 | The model learns from these |
| Validation | 2,241 | Used to adjust settings |
| Test | 2,286 | The "exam": never seen during learning |
| External | 4,064 | People from **9 whole centres** the model never saw |

The last group is the hardest and most honest test. It checks whether the model still works at hospitals it has never seen.

---

## 5. Teaching computers to predict (Notebook 06)

A **model** is a computer program that learns patterns from examples. We tried five kinds, from very simple to very modern:

1. **Logistic regression**: the simplest, like a weighted checklist.
2. **Random forest**: many small decision trees voting together.
3. **XGBoost**: decision trees built one after another, each fixing the last one's mistakes.
4. **Neural network**: a basic "deep learning" model.
5. **FT-Transformer**: a modern deep-learning model, related to the technology behind chatbots.

### How we scored them

We used a score called **AUROC**. Imagine picking one person who later got dementia and one who did not. AUROC is how often the model gives the higher risk to the right person.

* 0.5 = no better than tossing a coin
* 1.0 = perfect

### What we found

| Model | Score on the test group |
|---|---|
| Logistic regression | 0.936 |
| Random forest | 0.932 |
| XGBoost | 0.938 |
| Neural network | 0.938 |
| FT-Transformer | 0.939 |

**All five are basically the same.** The fancy modern models did not beat the simple one. On the 9 unseen centres all of them scored about 0.945.

**An honest warning about that 0.94.** Part of it is easy. People who already have mild memory problems convert to dementia far more often (41%) than healthy people (1.6%). Just knowing which group someone is in already helps a lot. When we look only inside the group with mild memory problems, the score is about **0.84**. That is the fairer number.

---

## 6. Which kind of information helps most?

We removed one kind of information at a time and watched how much the score dropped.

| Kind of information | How much it adds |
|---|---|
| Thinking tests | The most |
| Doctor's rating (CDR) and current diagnosis | A lot |
| Demographics (mainly age) | Some |
| Gene (APOE) | A little |
| Heart and blood vessel health | **Almost nothing** (about 0.001) |

### The main answer to our research question

Once we know how well a person's brain is working today, their heart and blood-vessel records add almost nothing to a 3-year prediction.

**Be careful how you say this.** It does **not** mean heart health is unimportant for the brain. It means:

* The people in this study joined at around age 71. Damage from high blood pressure usually happens in middle age, which we cannot see in this data.
* Any damage already done is probably already showing up in the thinking-test scores.

---

## 7. Does a person's history help? (Notebook 06)

We asked: if we look at how a person's scores changed over several past visits, do we predict better than using only today's visit?

We tried two ways, including a model called a **GRU** that reads visits in order, like reading a story.

**Result: history adds very little** (about 0.003 to 0.005). The reason makes sense. If someone has been getting worse, today's scores are already low. Knowing *how* they got there adds little once you know *where* they are.

---

## 8. Explaining the predictions (Notebook 07)

A doctor will not trust a number without a reason. We used a method called **SHAP**, which shows how much each piece of information pushed a prediction up or down.

What the model relies on:

| Kind of information | Share |
|---|---|
| Doctor's rating and diagnosis | about 41% |
| Thinking tests | about 35% |
| Demographics (mainly age) | about 15% |
| Heart and blood vessel health | 5 to 10% |
| Gene (APOE) | under 5% |

So about three quarters of the prediction comes from how the person's brain is doing right now.

**Important:** SHAP explains what the *model* looked at. It does not prove what *causes* dementia. Example: people with lower body weight had higher predicted risk. That is not because being thin causes dementia. People tend to lose weight in the years *before* a dementia diagnosis.

---

## 9. Do brain-scan numbers help? (Notebook 08)

**Amyloid** is a harmful protein that builds up in the brain many years before Alzheimer's symptoms appear. A PET scan can measure it.

* **On its own, amyloid matters a lot.** Among people with mild memory problems, about 34% of those with amyloid developed dementia within 2 years, compared with about 7% of those without.
* **But added to everything else, it did not clearly improve the prediction.** The model had already picked up most of that information from the memory tests.

**The honest way to say it:** "We could not show that PET improves the prediction." We cannot say "PET is useless", because only a small number of scanned people have been followed long enough.

---

## 10. Working with the actual scan images (Notebook 11)

We had 599 raw scans straight from hospital scanners, about 100 GB, from around 20 different scanner models.

### Step 1: Turning raw scanner files into usable pictures

* Each scan was hundreds or thousands of small files. We combined them into **one 3D picture per scan**.
* Every head was in a different position and size. We **lined them all up** so that the same spot means the same part of the brain in every picture. This is called **registration**.

### Step 2: Proving we did it correctly

From our pictures we measured a standard number called **SUVR** (how much amyloid signal is in the outer brain compared with a reference area).

Experts at NACC had already measured the same number for the same scans, using a different method. We compared the two:

**They matched very closely: a correlation of 0.98 across 318 scans** (1.0 would be a perfect match).

This is strong proof that our image pipeline is correct.

### Step 3: A deep-learning model on the images

We trained a **3D CNN** (a deep-learning model for 3D pictures) to say whether a scan shows amyloid or not.

| Method | Score |
|---|---|
| The single simple SUVR number | 0.98 |
| Our 3D CNN | 0.96 |

The simple number beat the deep-learning model. We also used **Grad-CAM**, a method meant to show where in the picture the model is looking. It gave a blurry map that did not point clearly at the right brain regions. We reported that as it is.

### What we did NOT do, and why

We did **not** try to predict dementia directly from the images. Only 5 of the 446 people with images later developed dementia. A computer cannot learn a pattern from 5 examples. Saying so openly is more honest, and safer in a viva, than pretending.

---

## 11. What if a doctor does not have all the information? (Notebook 10)

In real life, not every patient has had every test.

**The problem:** a normal model breaks badly when a whole section is missing. With no thinking tests at all, its risk numbers became worse than a blind guess.

**The fix:** during training, we randomly hid whole sections of information, again and again, so the model practised working with gaps. This is called **modality dropout**.

| Situation: no thinking tests at all | Error (lower is better) |
|---|---|
| Normal model | 0.44 (very bad) |
| Our model trained with hidden sections | 0.12 (good) |
| A model built specially for that case | 0.11 (best possible) |

One model now handles any combination of available information. This is one place where an idea from deep learning clearly helped.

---

## 12. Checking the model properly (Notebook 09)

A single score is not enough. We also checked:

* **Is it fair across groups?** It worked about equally well for men and women, for different races, and for different education levels.
* **Where is it weaker?** For people aged 85 and over.
* **Are the risk numbers believable?** When the model said "40% risk", about 40% of those people really did develop dementia. At the 9 unseen centres it guessed slightly too high (15.0% predicted, 13.4% actual).
* **Does the answer change with a different method?** We repeated the analysis with a method that includes everyone, even people followed for a short time. Same conclusions.

---

## 13. The doctor's screen (the interface)

We built a simple web page (`app/app.py`).

1. The doctor **ticks what information is available** for this patient.
2. Only those sections ask for input.
3. The page shows:
   * the estimated 3-year risk,
   * the average for comparison,
   * which information was used and which was missing,
   * what pushed the risk up or down,
   * the change since the previous visit.

A clear notice says: **research prototype, not for clinical use.**

It does **not** recommend medicines or treatment. Our data only records what happened to people. It cannot tell us what *would* have happened with a different treatment, so any such advice would be a guess, and a dangerous one.

---

## 14. The evidence chatbot (Notebook 12)

A normal chatbot can sound confident and still be wrong. Ours is built differently, using a method called **RAG** (retrieval-augmented generation):

1. We gave it a small library of **11 trusted, freely available research papers**.
2. When asked a question, it first **searches the library** for the most relevant paragraphs.
3. It answers **using only those paragraphs** and shows which paper each sentence came from.
4. If the library has nothing relevant, it says **"I could not find evidence for that."**
5. If someone asks "what medicine should my father take?", it **refuses to give personal advice**.

Everything runs on the laptop. Nothing is sent to the internet.

**Honest limits:** the library is small. In our tests it found the right paper every time, but we wrote those test questions ourselves, so it was an easy test. The small language model on the laptop wrote a properly cited answer only about 1 time in 5; the rest of the time the chatbot simply quotes the papers directly, which is the safer behaviour anyway.

---

## 15. Checking our own work

Because a mentor will examine everything, we wrote a program (`scripts/verify_results.py`) that runs **17 checks**. Some examples:

* The original data files were never changed.
* No person is in both the learning group and the test group.
* The answers ("developed dementia or not") were recalculated a second, separate way: 0 mismatches in 11,403 visits.
* **A scrambled-answers test.** We shuffled the answers randomly and trained a model. It should then be no better than a coin toss. It scored 0.51. Good. (If it had scored high, the model would be cheating somehow.)
* **A "no peeking into the future" test.** We deleted all later visits and checked that the information used at a visit stayed exactly the same.

### The checks found two mistakes, and we fixed them

1. For about 0.1% of people, a missing education value at an early visit had been filled in from a **later** visit. That is using the future, which is not allowed.
2. Many brain scans had been matched to a visit that happened shortly **before** the scan. Again, that uses information from slightly in the future.

We fixed both and re-ran every model. The main results changed only in the third decimal place.

Finding and fixing your own mistakes is a strength. Tell your mentor about it yourself.

---

## 16. Everything we found, in one table

| Question | Answer |
|---|---|
| Can we predict dementia within 3 years? | Yes: about 0.94 overall, about 0.84 among people with mild memory problems |
| Does it work at hospitals it has never seen? | Yes: about 0.95 at 9 unseen centres |
| Is deep learning better than simple models? | No, they are equal on this kind of data |
| Does heart and blood-vessel information help? | Almost not at all, once thinking tests are known |
| Does past history help? | Very little |
| Do brain-scan numbers help? | Not shown, though a small benefit cannot be ruled out |
| Did our image pipeline work? | Yes: it matches the experts' numbers (0.98) |
| Is the 3D image model better than one simple number? | No (0.96 against 0.98) |
| Can the model cope with missing information? | Yes, after training with hidden sections |

---

## 17. What this project cannot claim

* It is **not** tested on real patients in a real clinic. It was tested on past research data only.
* It does **not** diagnose and does **not** recommend treatment.
* The volunteers are mostly highly educated and mostly White, with an average age around 71. Results may differ for other groups.
* About half of the eligible people were not followed long enough to be included.
* We cannot see people's health in middle age, when heart and blood-vessel problems matter most for the brain.
* Brain-scan follow-up is short, and very few healthy people in the study developed dementia.
* The chatbot's library is small.

---

## 18. Where to find everything

| What | Where |
|---|---|
| Step-by-step explanations with charts | `notebooks/01` to `notebooks/12`, read in order |
| Likely viva questions with answers | `docs/viva_notes.md` |
| The full plan and what was done | `docs/project_plan.md` |
| Short results summaries | `reports/` |
| All the code | `scripts/` |
| The doctor's screen | `app/app.py` |
| The self-check program | `scripts/verify_results.py` |

---

## 19. Word list

| Word | Simple meaning |
|---|---|
| **Dementia** | Loss of memory and thinking severe enough to affect daily life |
| **MCI** | Mild cognitive impairment: memory problems that are noticeable but not yet dementia |
| **Cognitive** | To do with thinking and memory |
| **Cardiovascular** | To do with the heart and blood vessels |
| **APOE** | A gene; one version of it raises Alzheimer's risk |
| **PET scan** | A brain scan that shows where a tracer substance collects |
| **Amyloid / tau** | Two harmful proteins that build up in the brain in Alzheimer's disease |
| **SUVR** | A number measuring how much tracer signal a brain area has compared with a reference area |
| **CDR** | A doctor's rating of how much memory and daily life are affected |
| **Model** | A computer program that learns patterns from examples |
| **Training / test** | Examples the model learns from / examples kept hidden to check it fairly |
| **AUROC** | A score from 0.5 (coin toss) to 1.0 (perfect) for how well the model ranks people by risk |
| **z-score** | How far a score is from what is expected for a healthy person; 0 is typical |
| **Deep learning** | Models built from many layers of simple units; good with images and text |
| **CNN** | A deep-learning model for pictures |
| **GRU** | A deep-learning model that reads things in order, such as visits over time |
| **SHAP** | A method that shows how much each input pushed a prediction up or down |
| **Grad-CAM** | A method that tries to show where in a picture a model is looking |
| **RAG** | A chatbot design that looks up trusted documents first and answers only from them |
| **Leakage** | When a model accidentally sees information it should not have, making it look better than it is |
| **Calibration** | Whether predicted risks match what really happens |
