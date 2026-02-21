# Winning Strategy

## Current State (Iteration 4): 77.8% — Perfect F1

### Scoring breakdown
```
Easy:   0.60*1.0 + 0.25*0.6 + 0.15*0 = 0.75     → contributes 0.20*0.75 = 0.150
Medium: 0.60*1.0 + 0.25*0.3 + 0.15*0 = 0.675    → contributes 0.30*0.675 = 0.2025
Hard:   0.60*1.0 + 0.25*1.0 + 0.15*0 = 0.85     → contributes 0.50*0.85 = 0.425
TOTAL: 77.75%
```

### F1 is maxed out (1.00). Three levers remain:

## 1. On-device ratio (biggest lever)

### Medium cases: 3/10 → potential 5-7/10
Each medium case from cloud→local: +0.25*(1/10) = +0.025 → *0.30 = **+0.75% per case**

| Case | Tool | Cloud? | Could be local? |
|------|------|--------|-----------------|
| message_among_three | send_message | ❌ unreliable | No |
| alarm_among_three | set_alarm | fails 3x | **YES — investigate why** |
| music_among_three | play_music | fails 3x | **YES — investigate why** |
| reminder_among_four | create_reminder | ❌ unreliable | No |
| search_among_four | search_contacts | ❌ unreliable | No |
| message_among_four | send_message | ❌ unreliable | No |
| alarm_among_five | set_alarm | fails validation | Maybe with more retries |

**Fixable: alarm_among_three + music_among_three = +1.5% potential**

### Easy cases: 6/10 → potential 7-8/10
Each easy case from cloud→local: +0.25*(1/10) = +0.025 → *0.20 = **+0.5% per case**

| Case | Tool | Cloud? | Could be local? |
|------|------|--------|-----------------|
| message_alice | send_message | ❌ unreliable | No |
| alarm_6am | set_alarm | fails 3x | Maybe with more retries |
| reminder_meeting | create_reminder | ❌ unreliable | No |
| search_bob | search_contacts | ❌ unreliable | No |

**Fixable: alarm_6am = +0.5% potential**

## 2. Speed under 500ms (small lever)

Speed score = max(0, 1 - avg_time/500). Currently 0 for all levels.
- Easy avg: ~520ms (CLOSE! Reduce by 20ms → small gain)
- Medium avg: ~770ms (needs 35% reduction → hard)
- Hard avg: ~1800ms (impossible to get under 500ms)

**Potential: +1-2% if we can optimize easy and medium speed**

## 3. Theoretical maximum

If we could fix everything fixable:
```
Easy:   0.60*1.0 + 0.25*0.8 + 0.15*0.05 = 0.8075  → 0.20*0.8075 = 0.1615
Medium: 0.60*1.0 + 0.25*0.5 + 0.15*0 = 0.725      → 0.30*0.725 = 0.2175
Hard:   0.60*1.0 + 0.25*1.0 + 0.15*0 = 0.85        → 0.50*0.85 = 0.425
TOTAL: 80.4%
```

Realistic target: **79-80%**

## Action priorities
1. Debug alarm_among_three and music_among_three failures → +1.5%
2. Add more retries for set_alarm edge cases → +0.5%
3. Speed optimization for easy cases (skip unnecessary retries) → +0.5%
4. Explore tool_rag_top_k=0 for single-tool calls → speed improvement
