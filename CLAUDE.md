# League Objective Setup Analytics Project

## Project goal

This project studies objective setup in League of Legends using Riot Match-V5 match and timeline data.

The goal is not just to build a prediction engine. The goal is to create an analytics framework that explains what makes an objective setup good or bad, and how those factors differ by rank, objective type, and game state.

The core research question:

What conditions make contesting, starting, trading, or giving an objective a good decision?

## Project framing

The model should be used as an analytical tool, not as the final product.

The final output should help answer questions like:

- Are teams losing objectives because they are weaker, late, dead, lacking vision, or choosing bad contests?
- How much does death timing matter compared to gold difference?
- How early do good teams begin setting up for objectives?
- When is giving an objective better than contesting?
- Do low-rank teams over-contest bad objective states?
- Do high-rank teams trade objectives more cleanly?
- Does objective setup quality explain outcomes better than raw gold lead?

## Unit of analysis

The main row should be:

match_id + objective_instance + team_id

Example:

NA1_12345, dragon_1, blue_team  
NA1_12345, dragon_1, red_team

Do not make the main unit of analysis a whole game. Do not make it only one minute. The project should focus on objective windows.

## Objective windows

For each objective, create windows around objective take time:

- T-120 to T-90: early setup
- T-90 to T-30: rotation and setup phase
- T-30 to T+30: contest or take phase
- T+30 to T+120: aftermath

Use only information available before or during the relevant prediction point.

For pre-objective prediction, features should generally come from before T or before T-30, depending on the target.

## Objectives to start with

Start with Dragon 1 and Dragon 2 only.

Reasons:

- frequent sample size
- clear setup patterns
- bot, mid, jungle, and support all matter
- highly relevant for low and mid elo improvement
- simpler than Baron or Elder

After the pipeline works, expand to:

- Voidgrubs
- Herald
- Baron
- Soul
- Elder

## Outcome labels

Avoid only predicting "got objective."

That is too shallow because sometimes giving an objective and trading elsewhere is correct.

Create a richer objective outcome taxonomy.

Possible labels:

- clean_take
- clean_give
- good_trade
- bad_contest
- won_fight_lost_objective
- lost_fight_got_objective
- objective_steal
- coinflip
- throw_setup
- no_meaningful_contest

Also create a net objective value label.

Example scoring framework:

+3 objective secured  
+2 won fight near objective  
+2 took meaningful opposite-side objective  
+1 enemy jungler killed  
+1 enemy support killed  
+1 tower or plates gained elsewhere  
-3 lost fight near objective  
-2 allied jungler died before objective  
-2 allied support died before objective  
-2 major tower lost elsewhere  
-1 objective taken but team lost more gold afterward  

The exact scoring can change, but it must be documented and tested.

## Setup quality framework

Analyze objective setup through these dimensions:

### 1. Numbers advantage

Features may include:

- allied champions alive
- enemy champions alive
- jungler alive
- support alive
- deaths in last 30, 60, and 90 seconds
- death timers near objective
- whether a key role died before spawn

Core question:

Are teams losing because they are outnumbered before the objective even begins? How does this change as a game goes on (ex, do toplaners dying impact 1st and 2nd drake less?)

### 2. Tempo and arrival

Features may include:

- allied champions near objective at T-90, T-60, T-30
- enemy champions near objective at T-90, T-60, T-30
- jungler distance to objective
- support distance to objective
- mid laner distance to objective
- movement toward or away from objective
- whether team arrived first

Core question:

Are teams losing because they arrive late or enter after control is already lost? What kind of champions suffer the most from losing control (eg, control mages?)

### 3. Combat power

Features may include:

- team gold difference
- nearby gold difference
- jungler level difference
- average nearby level difference
- completed item proxies
- champion scaling or archetype tags
- objective history before current objective

Core question:

How much combat disadvantage can good setup overcome?

### 4. Lane priority proxy

Wave state is not directly available from Riot Match-V5 timeline data, so use proxies.

Features may include:

- laner position before objective
- CS or gold change in prior minute
- tower state
- whether laner moved first
- whether enemy laner was forced away
- whether laner died before objective

Core question:

Can lane priority be approximated through movement, position, and timing?

### 5. Vision and control

Features may include:

- ward placements near objective before spawn
- ward kills near objective before spawn
- support vision events
- jungle vision events
- facecheck deaths
- vision activity in river/jungle/objective area

Core question:

Does vision matter directly, or mostly because it prevents bad deaths and late arrivals?

### 6. Trade availability

Features may include:

- opposite-side objective alive
- opposite-side tower state
- side lane pressure proxy
- number of enemies committed to objective side
- allied position opposite side
- tower, plates, camps, or objective gained after giving

Core question:

When is giving the objective the correct decision?

## Modeling goals

The model should predict one or more of:

- net objective value
- objective outcome category
- probability of bad contest
- probability of clean setup
- probability of good trade
- probability of throw setup

Do not only predict final game win.

Do not only predict objective secured.

## Preferred model progression

Start simple and interpretable:

1. descriptive statistics
2. logistic regression
3. decision tree
4. random forest or gradient boosting
5. SHAP or permutation importance if useful

The model should support analysis such as:

- feature importance by rank
- marginal effects
- interaction effects
- setup profile clustering
- rank-specific differences
- objective-specific differences

## Analysis outputs

The project should produce findings like:

- "Low-rank teams lose more objectives from deaths 60-90 seconds before spawn than from raw gold deficits."
- "Support and jungle arrival timing is more important for Dragon 2 than Dragon 1."
- "Teams down moderate gold can still take good objective fights if they arrive first and have numbers advantage."
- "Some objective losses are correct gives when the team trades opposite side."
- "Bad contests are more common when teams have no nearby jungler or support but still walk into river."

These are examples only. Do not force these conclusions if the data does not support them.

## Setup profiles

Create setup profile categories, either rule-based or model-assisted.

Possible profiles:

### Clean setup

Team arrives early, has jungler alive, support nearby, vision activity, and no recent deaths.

### Coinflip setup

Both teams are late, both junglers alive, objective is started without clear control.

### Forced contest

Team is behind or late but contests anyway.

### Good give or trade

Team gives objective but gains tower, camps, plates, or opposite-side objective.

### Throw setup

Team has a playable or winning state but loses a member shortly before objective.

### No setup

Team neither contests nor trades meaningfully.

## Data leakage rules

This is extremely important.

At time T, features may only use information available at or before T.

Never use these as pre-objective features:

- final postgame stats
- final vision score
- final damage dealt
- final gold earned
- final KDA
- final item state
- final objective totals not yet known at T
- events after the prediction point

Allowed examples:

- gold at T-60
- position at T-30
- kills before objective
- deaths before objective
- ward events before objective
- previous objectives before current objective
- tower state before objective
- champion and role metadata
- rank bucket
- patch

Train/test splits must be by match_id, never by objective row.

Bad:

minute/objective rows from the same match split across train and test.

Good:

all rows from a match are entirely in train, validation, or test.

## Rank buckets

Use these default rank buckets unless changed later:

- low: Iron, Bronze, Silver
- mid: Gold, Platinum
- high: Emerald, Diamond
- elite: Master+

Always consider rank-specific analysis. Do not assume the same factors matter equally across all ranks.

## Data source assumptions

Use Riot Match-V5 match and timeline data.

The public timeline data is good enough for coarse objective setup analysis, but not perfect replay reconstruction.

Be careful with:

- exact player pathing
- exact wave state
- causal claims
- role detection
- missing timeline data
- patch differences
- sample selection bias

Use conservative language.

Prefer:

"likely setup pattern"  
"proxy for priority"  
"associated with objective outcome"  
"predictive of bad contest"

Avoid:

"proves this caused the loss"  
"this was definitely the correct play"  
"the model knows the best decision"

## Feature engineering priorities

Start with features that are feasible and interpretable.

Priority features:

- team gold diff at T-90, T-60, T-30
- nearby champion count at T-90, T-60, T-30
- jungler alive at T-90, T-60, T-30
- support alive at T-90, T-60, T-30
- jungler distance to objective
- support distance to objective
- mid laner distance to objective
- deaths in last 30, 60, 90 seconds
- kills in last 30, 60, 90 seconds
- ward placements near objective before spawn
- ward kills near objective before spawn
- previous dragon count
- tower state near objective side
- objective-side laner position proxy
- opposite-side trade result after objective

## Coding style

Use Python.

Preferred libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- requests
- python-dotenv
- pytest

Use type hints when practical.

Keep code modular.

Do not put large exploratory logic into production modules. Use notebooks for exploration and src/ for reusable code.

## Suggested repo structure

src/
  lolobj/
    __init__.py
    config.py
    riot_client.py
    ingest/
      download_matches.py
      storage.py
    parsing/
      timeline_parser.py
      objective_events.py
      positions.py
    features/
      objective_windows.py
      setup_features.py
      vision_features.py
      trade_features.py
      rank_features.py
    labels/
      objective_outcomes.py
      net_value.py
    models/
      train_baseline.py
      train_objective_model.py
      evaluate.py
      interpret.py
    analysis/
      rank_comparison.py
      setup_profiles.py
    viz/
      plots.py

tests/
  fixtures/
  test_objective_windows.py
  test_no_leakage.py
  test_objective_labels.py
  test_setup_features.py

notebooks/
  01_data_audit.ipynb
  02_objective_windows.ipynb
  03_baseline_models.ipynb
  04_rank_comparison.ipynb

data/
  raw/
  interim/
  processed/

## Testing expectations

For every major feature transformation, add a small test with synthetic data.

Important tests:

- objective window creation works
- future events do not affect past features
- train/test split is by match_id
- death-before-objective features only count deaths before objective
- objective outcome labels match toy examples
- trade labels do not count events before the objective incorrectly

Before claiming the pipeline works, run tests.

Use commands like:

pytest

and, once implemented:

python -m lolobj.features.objective_windows --sample

## Working style for Claude Code

When asked to implement a change:

1. inspect the relevant files
2. make a short implementation plan
3. edit the smallest number of files necessary
4. add or update tests
5. run tests
6. summarize what changed
7. mention any limitations or assumptions

Do not silently make large architecture changes.

Do not invent Riot API fields without checking existing data structures or fixtures.

If a field is uncertain, write the code defensively and document the assumption.

## First milestone

The first milestone is not a model.

The first milestone is:

A clean objective-window table for Dragon 1 and Dragon 2.

Each row should represent:

match_id  
team_id  
objective_type  
objective_number  
objective_time  
rank_bucket  
patch  
features before objective  
outcome label  
net objective value label  

Once this table exists, modeling and analysis can begin.

## Second milestone

Build a baseline analytical report answering:

- How often does each setup profile occur?
- How often does each outcome type occur?
- How often do teams contest from bad setup states?
- How often do teams give and trade successfully?
- How do these rates differ by rank?

Only after this should a more complex ML model be built.

## Third milestone

Build a model that predicts net objective value or bad contest probability.

Use the model to explain:

- strongest predictors
- rank differences
- feature interactions
- setup profile differences
- situations where raw gold lead is misleading

The final deliverable should be analytical, interpretable, and player-facing.