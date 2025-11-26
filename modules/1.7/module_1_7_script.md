# Module 1.7: Understanding the M5 Dataset
## Video Tutorial Script — Detailed Narration

---

## Opening

> **[On camera or voiceover]**

Welcome to Module 1.7 — Understanding the M5 Dataset.

In Module 1.6, we ran first-contact checks on the M5 data. We loaded it, cleaned it, aggregated to weekly, and assessed whether it could support our 5Q Framework.

But we used a helper function — `tsf.load_m5()` — that did a lot of work behind the scenes. Today, we're going to look under the hood.

> **[Open the Module 1.7 HTML file in browser]**

I've created an interactive explorer for the M5 dataset. Open the HTML file and follow along — I'll tell you where to click as we go.

### Why This Matters

You might be thinking: "If the helper function works, why do I need to understand the raw data?"

Two reasons:

First, **not every dataset will have a helper**. When you get data from your company's warehouse or a new client, there's no `load_company_data()` function waiting for you. You need to understand data structures so you can wrangle any dataset into forecasting-ready format.

Second, **understanding prevents leakage**. The M5 dataset has a critical trap — prices that look like they should be features but are actually unknown at forecast time. If you don't understand which fields are safe to use when, you'll build models that look perfect in validation but fail completely in production.

So let's dig in.

---

## The Raw M5 Structure

> **[In the HTML: Click the "📖 Overview" tab if not already selected]**

> **[In the HTML: Notice the "Reading This Guide: Raw vs. Derived" legend box — this shows you how to distinguish between original data and transformed fields throughout the explorer]**

The raw M5 dataset from Kaggle has three files:

| File | What It Contains | Rows |
|------|------------------|------|
| `sales_train.csv` | Daily unit sales (the target) + hierarchy IDs | 30,490 |
| `calendar.csv` | Date attributes, events, SNAP schedules | 1,969 |
| `sell_prices.csv` | Weekly prices by item-store | 6.8M |

> **[In the HTML: Scroll down to see "The Three Files" section with the icons]**

You can see each file has a specific role. Sales is your target. Calendar is your known-at-time features. Prices is where the leakage trap lives.

Now here's something important to understand: **what you see depends on how you load the data**.

### M5 Files

> **[Show side-by-side comparison]**

The raw `sales_train.csv` from Kaggle is in **wide format**:

```
item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, d_3, ... d_1941
FOODS_3_090, FOODS_3, FOODS, CA_3, CA, 3, 0, 5, ...
```

Each row is one item-store combination. Each column `d_1` through `d_1941` is a day's sales. That's 1,941 columns of sales data.

But when you use `tsf.load_m5()` or load from Nixtla's `datasetsforecast` package, you get **long format**:

```
unique_id, ds, y
FOODS_3_090_CA_3, 2011-01-29, 3
FOODS_3_090_CA_3, 2011-01-30, 0
FOODS_3_090_CA_3, 2011-01-31, 5
```

Each row is one observation — one item, one store, one date, one sales value.

**This is the format Nixtla and most forecasting libraries expect.** The transformation from wide to long is handled automatically by the helper.

> **[In the HTML: Scroll down in the Overview tab to see the "Format Transformation: Raw → Modeling-Ready" diagram — this shows exactly what changes]**

### What About unique_id?

> **[Highlight the unique_id field]**

In the raw data, there's no `unique_id` column. You have `item_id` and `store_id` separately.

The helper creates `unique_id` by concatenating them: `FOODS_3_090` + `CA_3` = `FOODS_3_090_CA_3`.

When you use `include_hierarchy=True` in `tsf.load_m5()`, you get both:
- The combined `unique_id` for Nixtla compatibility
- The separate hierarchy columns (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`) for analysis

This is what you saw in Module 1.6 when we loaded with `include_hierarchy=True`.

---

## The Three Files in Detail

> **[In the HTML: Click the "📊 Tables" tab]**

Let's walk through each file and classify every field. You'll see three cards — one for each file. Click on each to explore the fields.

### File 1: sales_train.csv (Target + Static IDs)

> **[In the HTML: Click the blue "sales_train.csv" card]**

> **[In the HTML: Notice the warning boxes now have 🔧 icons for derived fields — these help you distinguish what's in the raw data vs what you need to create]**

This is your target data. In raw format:

**Shape:** 30,490 rows × 1,947 columns
- 6 identifier columns
- 1,941 daily sales columns (d_1 through d_1941)

**The identifier columns are all STATIC:**

| Column | Classification | Description |
|--------|---------------|-------------|
| `item_id` | Static Identifier | 3,049 unique products |
| `dept_id` | Static Feature | 7 departments |
| `cat_id` | Static Feature | 3 categories (FOODS, HOBBIES, HOUSEHOLD) |
| `store_id` | Static Identifier | 10 stores |
| `state_id` | Static Feature | 3 states (CA, TX, WI) |

**Static means these never change over time.** An item is always in the same department. A store is always in the same state. These are always safe to use as features.

**The d_X columns are the TARGET:**

| Column | Classification | Description |
|--------|---------------|-------------|
| `d_1, d_2, ... d_1941` | Target Variable | Daily unit sales starting 2011-01-29 |

**Critical point:** The target is what you're predicting. You can use PAST values as lag features, but never future values.

> **[Show warning box]**

**LEAKAGE RISK:** When forecasting d_100, you can use d_1 through d_99 as features. You cannot use d_100 itself — that's what you're trying to predict.

---

### A Note on Hierarchy

> **[In the HTML: Scroll down within the sales card to see the warning boxes, or go back to the Overview tab and scroll to "M5 Data Hierarchy"]**

> **[In the HTML: In the hierarchy diagram, notice the unique_id section has an orange "🔧 DERIVED — NOT IN RAW DATA" badge — this is the key field you must create yourself]**

You might notice the hierarchy is embedded right in the sales data. You don't need a separate metadata file.

**Product Hierarchy:**
```
Category (3) → Department (7) → Item (3,049)
   FOODS   →    FOODS_3    →  FOODS_3_090
```

**Location Hierarchy:**
```
State (3) → Store (10)
   CA    →   CA_3
```

**Combined:** 3,049 items × 10 stores = 30,490 unique item-store combinations.

This hierarchy matters for two reasons:

First, you can **aggregate** to any level. Don't want to forecast 30,000 series? Aggregate to department-store level and you have 70 series. Or category-state level for 9 series.

Second, you can use **hierarchical reconciliation**. Forecast at multiple levels and ensure they're mathematically consistent. We'll cover this in later modules.

---

### File 2: calendar.csv (Known-At-Time Features)

> **[In the HTML: Click the purple "calendar.csv" card]**

**Shape:** 1,969 rows × 14 columns (one row per day)

This is your date reference table. **Every field here is safe to use for forecasting** because they're all known in advance.

**Date identifiers:**

| Column | Classification | Description |
|--------|---------------|-------------|
| `date` | Time Identifier | Date in YYYY-MM-DD format |
| `d` | Time Identifier | Matches d_X columns in sales (d_1, d_2, etc.) |
| `wm_yr_wk` | Time Grouping | Walmart week ID — use this to join prices |

**Calendar features (all known-at-time):**

| Column | Classification | Description |
|--------|---------------|-------------|
| `weekday` | Known-At Feature | Day name (Saturday, Sunday, etc.) |
| `wday` | Known-At Feature | Day number (1-7, starting Saturday) |
| `month` | Known-At Feature | Month (1-12) |
| `year` | Known-At Feature | Year |

**Event features (all known-at-time):**

| Column | Classification | Description |
|--------|---------------|-------------|
| `event_name_1` | Known-At Feature | Primary event name (SuperBowl, Christmas, etc.) |
| `event_type_1` | Known-At Feature | Event type (Sporting, Cultural, Religious, National) |
| `event_name_2` | Known-At Feature | Secondary event name (some dates have 2 events) |
| `event_type_2` | Known-At Feature | Secondary event type |

**SNAP features (all known-at-time):**

| Column | Classification | Description |
|--------|---------------|-------------|
| `snap_CA` | Known-At Feature | SNAP eligible in California (0/1) |
| `snap_TX` | Known-At Feature | SNAP eligible in Texas (0/1) |
| `snap_WI` | Known-At Feature | SNAP eligible in Wisconsin (0/1) |

> **[Emphasize this point]**

**Why are these "known-at-time"?** Because they come from calendars and government schedules. You know Christmas is December 25th. You know the SNAP schedule for next month. These are predetermined — not observed from actual sales data.

**Important distinction:** You know the event NAME (SuperBowl is on February 7th). You do NOT know the event IMPACT (how much sales will spike). The event is known; its effect must be learned from historical data.

---

### File 3: sell_prices.csv (THE LEAKAGE TRAP)

> **[In the HTML: Click the orange "sell_prices.csv" card]**

This is where most people make mistakes. Notice the card says "DYNAMIC FEATURES (UNKNOWN)" — that's the warning sign.

**Shape:** 6,841,121 rows × 4 columns

| Column | Classification | Description |
|--------|---------------|-------------|
| `store_id` | Join Key | Store identifier |
| `item_id` | Join Key | Product identifier |
| `wm_yr_wk` | Join Key (Time-Varying) | Walmart week — join to calendar |
| `sell_price` | **DYNAMIC (UNKNOWN)** | Weekly average price |

> **[Show the critical warning]**

**THE CRITICAL LEAKAGE RISK:**

Prices are DYNAMIC. They change week to week. And critically — **you do NOT know next week's actual price when making a forecast**.

Think about it: when you're forecasting next week's sales, what price will you use? You don't have next week's actual price yet. It hasn't happened.

**Common mistake:**
```python
# ❌ WRONG: Joining future prices
df = df.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'])
# This gives you next week's actual price to predict next week's sales
# That's leakage!
```

**What you CAN do instead:**

1. **Use lagged prices:** Last week's price as a feature
2. **Use planned prices:** If you have a promotional calendar with planned prices
3. **Forecast prices separately:** Build a price prediction model

> **[Show leakage scenario]**

**Why does this destroy your model?**

In backtesting, you JOIN prices by date. For historical dates, you have the actual price — so the model learns "when price drops, sales spike." Looks great!

In production, you're forecasting next week. You don't have next week's actual price. You have to guess. Now your model breaks because it's missing the feature it relied on most.

This is why leakage causes "perfect scores in notebooks, total failure in production."

---

## The 5Q Classification Summary

> **[In the HTML: Click the "💡 Key Concepts" tab]**

Let's tie this back to our 5Q Framework. This tab shows you the five core concepts you need to internalize.

> **[In the HTML: Look at the purple "Core Concepts" banner at the top]**

| Field Type | 5Q Connection | M5 Examples | Safe to Use? |
|------------|---------------|-------------|--------------|
| **Static** | Q3 (Structure) | item_id, store_id, dept_id, cat_id, state_id | ✓ Always |
| **Known-At-Time** | Q4 (Drivers) | weekday, month, events, SNAP | ✓ Always |
| **Target** | Q1 (Target) | d_1...d_1941 (sales) | ✓ Past only |
| **Dynamic Unknown** | Q4 (Drivers) | sell_price | ⚠ Lagged only |

> **[In the HTML: Scroll down in the Concepts tab to see the five concept cards — STATIC, DYNAMIC, KNOWN-AT-TIME, TARGET, and DATA LEAKAGE]**

Each card shows the definition, M5 examples, usage guidance, and watch-outs. Take time to read through each one.

The critical question for any feature: **"Will I actually have this data when I need to make the forecast?"**

> **[In the HTML: Scroll down to see "The Leakage Trap" red warning box — read this carefully]**

> **[In the HTML: Click the "📋 Cheat Sheet" tab for a complete column reference]**

The cheat sheet gives you every single column in M5, classified by type. Bookmark this — you'll refer back to it when building features.

---

## What TSForge Does For You

> **[Show what the helper handles]**

Now you understand what's in the raw M5 data. Here's what `tsf.load_m5()` does automatically:

1. **Downloads** the data from datasetsforecast (or uses cache)
2. **Reshapes** from wide (d_X columns) to long (unique_id, ds, y)
3. **Creates** unique_id from item_id + store_id
4. **Converts** d_X column names to actual dates
5. **Optionally expands** hierarchy columns with `include_hierarchy=True`

In Module 1.8, we'll show how to merge calendar and price features while avoiding leakage. But first, you needed to understand what each field represents and why some are dangerous.

---

## Key Takeaways

> **[Summary slide — or revisit each HTML tab as you summarize]**

**The HTML explorer has four tabs — here's what each covers:**
- **📖 Overview:** The three files, hierarchy diagram, why this matters
- **📊 Tables:** Click each file card to see every field classified
- **💡 Key Concepts:** The five classifications (Static, Dynamic, Known-At-Time, Target, Leakage)
- **📋 Cheat Sheet:** Complete column reference — bookmark this

**Three files, three roles:**
- `sales_train.csv` — Target + static hierarchy
- `calendar.csv` — Known-at-time features (safe)
- `sell_prices.csv` — Dynamic features (leakage risk)

**The format transformation:**
- Raw M5: Wide format with d_X columns
- TSForge/Nixtla: Long format with (unique_id, ds, y)
- Helper functions handle this automatically

**The classification framework:**
- STATIC: Never changes — always safe
- KNOWN-AT-TIME: From calendars/schedules — always safe
- TARGET: What you predict — use past values only
- DYNAMIC UNKNOWN: Changes over time, not known ahead — use lags only

**The critical question:**
> "Will I actually have this data when I need to make the forecast?"

---

## Next Steps

> **[Preview upcoming modules]**

Now that you understand the M5 structure:

- **Module 1.8:** Diagnostics — profile volatility, seasonality, trend across all series
- **Module 1.9:** Portfolio Analysis with GenAI — scale understanding to 30K series
- **Module 1.10:** Data Preparation — merge calendar and prices safely, avoiding leakage

We've laid the foundation. You know what the data contains, how it's structured, and where the traps are. Next, we'll actually work with these files.

See you in Module 1.8.

---

*End of Script*
