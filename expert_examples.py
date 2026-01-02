"""
Expert Examples for Chain-of-Thought Generation
AI Telco Troubleshooting Challenge - Track 3

These 8 examples (one per root cause C1-C8) are based on the validated 
reasoning approach that achieved 14/16 (87.5%) accuracy on sample questions.
"""

# Root Cause Definitions (from competition)
ROOT_CAUSE_DEFINITIONS = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput."
}

# Expert reasoning examples - one per root cause
EXPERT_EXAMPLES = {
    "C1": """### Example (Root Cause: C1 - Excessive Downtilt)

**User Plane Data Summary:**
- Serving PCI: 418
- RSRP: Drops rapidly from -88 dBm to -94 dBm over short distance
- SINR: 8-10 dB (acceptable)
- Throughput: Drops to 268 Mbps
- Speed: 15-33 km/h (normal)
- RBs: ~200 (sufficient)

**Engineering Parameters for Serving Cell 418:**
- Mechanical Downtilt: 35°
- Digital Tilt: 9°
- Total Downtilt: 44°
- Height: 51.3m

<reasoning>
1. **Symptom Identification**: Throughput drops to 268 Mbps despite acceptable SINR (8-10 dB).

2. **Metric Extraction**:
   - GPS Speed: 15-33 km/h → Below 40 km/h threshold ✓
   - DL RB Num: ~200 → Above 160 threshold ✓
   - RSRP: Drops from -88 to -94 dBm rapidly over short distance
   - Serving Cell 418 Engineering: Mech Tilt 35° + Digital Tilt 9° = **44° Total**

3. **Hypothesis Testing**:
   - C7 (Speed > 40): Max speed 33 km/h < 40. **Ruled out.**
   - C8 (RBs < 160): RBs ~200 > 160. **Ruled out.**
   - C6 (PCI Collision): No neighbor shares PCI mod 30 with 418. **Ruled out.**
   - C5 (Frequent HO): Only 1-2 handovers in window. **Ruled out.**
   - C1 (Downtilt too large): Total downtilt of 44° is extremely excessive (normal is 5-15°).

4. **Root Cause Analysis**:
   A 44° downtilt from a 51m tower causes the main beam to hit the ground very close to the antenna.
   This creates a severe coverage drop-off zone. The rapid RSRP degradation (-6 dB over ~50m) 
   confirms the UE is moving out of the narrow beam footprint.

5. **Conclusion**: The excessive downtilt (44°) causes weak coverage at the cell edge, 
   degrading throughput despite normal speed and resource allocation.
</reasoning>

\\boxed{C1}""",

    "C2": """### Example (Root Cause: C2 - Overshooting/Coverage Distance > 1km)

**User Plane Data Summary:**
- Serving PCI: 134
- Location: Longitude 128.1915
- RSRP: -80 to -88 dBm
- SINR: 8-11 dB
- Throughput: Drops to 221 Mbps
- Speed: 4-38 km/h

**Engineering Parameters for Serving Cell 134:**
- Location: Longitude 128.2213, Latitude 32.5771
- Distance from UE: ~2.8 km (calculated from coordinates)

<reasoning>
1. **Symptom Identification**: Throughput drops to 221 Mbps while connected to cell 134.

2. **Metric Extraction**:
   - GPS Speed: 4-38 km/h → Below 40 km/h threshold ✓
   - DL RB Num: ~207 → Above 160 threshold ✓
   - Serving Cell 134 Location: (128.2213, 32.5771)
   - UE Location: (128.1915, 32.5778)
   - Longitude difference: 0.0298° ≈ 2.8 km

3. **Distance Calculation**:
   Using approximate conversion (1° longitude ≈ 95 km at this latitude):
   Distance = 0.0298 × 95 ≈ **2.83 km**

4. **Hypothesis Testing**:
   - C7 (Speed > 40): Max 38 km/h < 40. **Ruled out.**
   - C8 (RBs < 160): RBs sufficient. **Ruled out.**
   - C2 (Distance > 1km): Distance is 2.83 km >> 1 km. **Confirmed.**

5. **Root Cause Analysis**:
   The UE is connected to a cell 2.8 km away (overshooting scenario).
   Despite acceptable RSRP (possible LOS condition), the large distance causes:
   - Higher path loss requiring more power
   - Reduced spectral efficiency
   - Timing advance challenges
   This results in poor throughput even with good signal quality metrics.

6. **Conclusion**: Serving cell coverage distance exceeds 1km, causing overshooting.
</reasoning>

\\boxed{C2}""",

    "C3": """### Example (Root Cause: C3 - Neighbor Provides Higher Throughput)

**User Plane Data Summary:**
- Serving PCI: 432
- Serving RSRP: -86 to -88 dBm
- SINR: 16-17 dB (good)
- Throughput: 574-594 Mbps (below 600)
- Top Neighbor PCI: 441
- Neighbor RSRP: -89 dBm

**After Handover to 441:**
- Throughput jumps to 1393 Mbps

<reasoning>
1. **Symptom Identification**: Throughput on serving cell 432 is ~590 Mbps, just below threshold.
   Immediately after handover to neighbor 441, throughput doubles to 1393 Mbps.

2. **Metric Extraction**:
   - GPS Speed: 10-36 km/h → Normal ✓
   - DL RB Num: ~210 → Sufficient ✓
   - Serving 432 RSRP: -86 to -88 dBm
   - Neighbor 441 RSRP: -89 dBm (slightly weaker in RSRP)
   - SINR: 16-17 dB (good, no interference issue)

3. **Hypothesis Testing**:
   - C7/C8: Speed and RBs normal. **Ruled out.**
   - C6 (PCI Collision): 432 % 30 = 12, 441 % 30 = 21. **No collision.**
   - C4 (Overlap): SINR is good (16-17 dB). No interference pattern. **Ruled out.**
   - C3 (Neighbor Better): Despite similar RSRP, neighbor 441 provides 2x throughput.

4. **Root Cause Analysis**:
   The neighbor cell 441 provides significantly higher throughput despite similar signal strength.
   This indicates:
   - Cell 441 has better load conditions (fewer users)
   - Or better backhaul capacity
   - Or better interference environment in its direction
   The UE would benefit from earlier handover to 441.

5. **Conclusion**: Neighboring cell 441 provides higher throughput than serving cell 432.
</reasoning>

\\boxed{C3}""",

    "C4": """### Example (Root Cause: C4 - Overlapping Coverage/Interference)

**User Plane Data Summary:**
- Serving PCI: 44
- Serving RSRP: -82 dBm
- SINR: 13 dB (moderate)
- Throughput: Drops to 341 Mbps
- Top Neighbor PCI: 462
- Neighbor RSRP: -85 dBm (only 3 dB weaker)
- Other neighbors also strong: 835 at -107 dBm

<reasoning>
1. **Symptom Identification**: Throughput drops to 341 Mbps despite reasonable SINR.

2. **Metric Extraction**:
   - GPS Speed: 19 km/h → Normal ✓
   - DL RB Num: ~210 → Sufficient ✓
   - Serving 44 RSRP: -82.98 dBm
   - Neighbor 462 RSRP: -85.77 dBm
   - **RSRP Delta: Only 2.8 dB** (very close!)

3. **PCI Mod 30 Check**:
   - 44 % 30 = 14
   - 462 % 30 = 12
   - **No collision.** C6 ruled out.

4. **Overlap Analysis**:
   - Serving to Top Neighbor delta: 2.8 dB
   - When delta < 3-5 dB, the UE is in a "pilot pollution" zone
   - Multiple cells compete for the connection
   - Causes scheduling inefficiency and interference

5. **Hypothesis Testing**:
   - C7/C8: Normal. **Ruled out.**
   - C6: No PCI mod 30 match. **Ruled out.**
   - C4 (Overlapping Coverage): Neighbor within 3 dB of serving. **Confirmed.**

6. **Root Cause Analysis**:
   The non-colocated cell 462 is nearly as strong as the serving cell 44.
   This creates severe overlapping coverage where:
   - Neither cell is clearly dominant
   - Inter-cell interference reduces effective SINR
   - Handover decisions become unstable

7. **Conclusion**: Non-colocated co-frequency cells cause severe overlapping coverage.
</reasoning>

\\boxed{C4}""",

    "C5": """### Example (Root Cause: C5 - Frequent Handovers)

**User Plane Data Summary:**
- PCI sequence over 10 samples: 477 → 477 → 477 → 477 → 412 → 477 → 412 → 412 → 412 → 412
- Throughput: Unstable, drops to 304-337 Mbps during transitions
- Both cells have similar RSRP (~-83 to -86 dBm)
- SINR fluctuates: 0 to 3 dB during transitions

<reasoning>
1. **Symptom Identification**: Throughput drops to ~320 Mbps during periods of instability.

2. **Handover Analysis**:
   Tracking serving PCI across timestamps:
   - 11:29:15 → 477
   - 11:29:16 → 477
   - 11:29:17 → 477
   - 11:29:18 → 477
   - 11:29:19 → 412 ← Handover
   - 11:29:20 → 477 ← Handover back (Ping-pong!)
   - 11:29:21 → 412 ← Handover again
   - 11:29:22 → 412
   
   **4 handovers in 10 seconds = Frequent handovers!**

3. **Impact of Frequent Handovers**:
   - Each handover causes brief service interruption (~50-100ms)
   - TCP congestion window resets
   - Scheduler loses context about channel conditions
   - Combined effect: Severe throughput degradation

4. **Hypothesis Testing**:
   - C7 (Speed): 0-36 km/h, max < 40. **Ruled out.**
   - C8 (RBs): ~200, sufficient. **Ruled out.**
   - C6 (PCI Collision): 477%30=27, 412%30=22. **No collision.**
   - C5 (Frequent HO): 4 handovers in 10 samples. **Confirmed.**

5. **Conclusion**: Frequent handovers (ping-pong between 477 and 412) degrade performance.
</reasoning>

\\boxed{C5}""",

    "C6": """### Example (Root Cause: C6 - PCI Mod 30 Collision)

**User Plane Data Summary:**
- Serving PCI: 567
- Serving RSRP: -88 dBm
- SINR: **3.68 dB** (critically low!)
- Throughput: 256 Mbps
- Top Neighbor PCI: 687
- Neighbor RSRP: -88.23 dBm (nearly identical to serving)

<reasoning>
1. **Symptom Identification**: Very low SINR (3.68 dB) despite adequate RSRP.
   Throughput severely degraded to 256 Mbps.

2. **Metric Extraction**:
   - GPS Speed: 5-33 km/h → Normal ✓
   - DL RB Num: ~211 → Sufficient ✓
   - SINR: 3.68 dB → **Critically low** (should be >10 for good throughput)
   - Serving 567 vs Neighbor 687 RSRP: Nearly identical (-88 vs -88)

3. **PCI Mod 30 Analysis**:
   - Serving PCI 567: 567 % 30 = **27**
   - Neighbor PCI 687: 687 % 30 = **27**
   - **MATCH! PCI Mod 30 collision detected.**

4. **Why PCI Mod 30 Matters**:
   - 5G NR uses PCI mod 30 for reference signal sequence generation
   - When two cells share the same mod 30 value, their reference signals overlap
   - This causes reference signal interference
   - The receiver cannot properly estimate the channel → low SINR
   - Low SINR → poor modulation and coding scheme → low throughput

5. **Hypothesis Testing**:
   - C4 (Overlap): Cells overlap, but the **root cause** is the PCI collision
   - C6 (PCI Collision): Confirmed by mod 30 match and critically low SINR

6. **Conclusion**: PCI mod 30 collision between 567 and 687 causes reference signal interference.
</reasoning>

\\boxed{C6}""",

    "C7": """### Example (Root Cause: C7 - High Vehicle Speed)

**User Plane Data Summary:**
- Serving PCI: 482, then 462
- RSRP: -75 to -88 dBm (acceptable)
- SINR: 17-25 dB (good)
- GPS Speed: **56, 58, 67, 70 km/h** (exceeds 40 km/h!)
- Throughput: Drops to 386-434 Mbps during high speed
- RBs: ~210 (sufficient)

<reasoning>
1. **Symptom Identification**: Throughput drops to ~400 Mbps despite good SINR (17-25 dB).

2. **Metric Extraction**:
   - GPS Speed: 56, 58, 67, 70 km/h → **All exceed 40 km/h threshold**
   - DL RB Num: ~210 → Sufficient ✓
   - RSRP: -75 to -88 dBm → Acceptable ✓
   - SINR: 17-25 dB → Good ✓

3. **Speed Impact Analysis**:
   At speeds above 40 km/h:
   - Doppler shift affects channel estimation accuracy
   - Channel coherence time decreases
   - Scheduler cannot adapt fast enough to rapid channel changes
   - Higher layer protocols (TCP) suffer from increased latency variance

4. **Hypothesis Testing**:
   - C8 (RBs < 160): RBs ~210. **Ruled out.**
   - C1/C2/C3/C4/C5/C6: Signal quality metrics are good. **Ruled out.**
   - C7 (Speed > 40): Speed reaches 70 km/h. **Confirmed.**

5. **Conclusion**: Vehicle speed exceeding 40 km/h (up to 70 km/h) impacts throughput
   due to Doppler effects and channel estimation challenges.
</reasoning>

\\boxed{C7}""",

    "C8": """### Example (Root Cause: C8 - Low Scheduled RBs)

**User Plane Data Summary:**
- Serving PCI: 330, then 591
- RSRP: -85 to -88 dBm (acceptable)
- SINR: 18-22 dB (excellent!)
- GPS Speed: 4-39 km/h (normal)
- DL RB Num: **101, 102, 103** (below 160 threshold!)
- Throughput: Drops to 257-316 Mbps

<reasoning>
1. **Symptom Identification**: Throughput drops to ~290 Mbps despite excellent SINR (18-22 dB).

2. **Metric Extraction**:
   - GPS Speed: 4-39 km/h → Below 40 km/h ✓
   - SINR: 18-22 dB → Excellent ✓
   - RSRP: -85 to -88 dBm → Acceptable ✓
   - DL RB Num: 101, 102, 103 → **Below 160 threshold!**

3. **RB Analysis**:
   - Normal RB allocation for high throughput: 200-210 RBs
   - Current allocation: ~102 RBs
   - This is only ~50% of normal capacity
   - With 5G NR 100 MHz bandwidth, 160 RBs is minimum for 600 Mbps

4. **Why RBs are Low**:
   Possible causes (not our job to fix, just diagnose):
   - Cell is heavily loaded with other users
   - Backhaul limitation
   - Scheduler configuration issue
   - Inter-cell interference coordination reserving RBs

5. **Hypothesis Testing**:
   - C7 (Speed): Max 39 km/h < 40. **Ruled out.**
   - SINR is excellent, so C4/C6 interference causes **ruled out.**
   - C8 (RBs < 160): RBs ~102 < 160. **Confirmed.**

6. **Conclusion**: Average scheduled RBs (102) are below 160, directly limiting throughput.
</reasoning>

\\boxed{C8}"""
}

# Related root causes for dynamic few-shot selection
RELATED_CAUSES = {
    "C1": ["C2"],  # Both coverage-related
    "C2": ["C1"],
    "C3": ["C4"],  # Both involve neighbor cells
    "C4": ["C3", "C6"],  # C4 and C6 both involve interference
    "C5": ["C7"],  # Both mobility-related
    "C6": ["C4"],
    "C7": ["C5"],
    "C8": []  # Unique - resource allocation
}

def get_few_shot_examples(target_cause: str, num_examples: int = 2) -> str:
    """
    Get relevant few-shot examples for a given target root cause.
    Returns the target cause example + related examples.
    """
    examples = [EXPERT_EXAMPLES[target_cause]]
    
    # Add related examples
    for related in RELATED_CAUSES.get(target_cause, []):
        if len(examples) < num_examples:
            examples.append(EXPERT_EXAMPLES[related])
    
    # If still need more, add a different one
    if len(examples) < num_examples:
        for cause in ["C5", "C7", "C8"]:  # Easy to distinguish
            if cause != target_cause and EXPERT_EXAMPLES[cause] not in examples:
                examples.append(EXPERT_EXAMPLES[cause])
                break
    
    return "\n\n---\n\n".join(examples)


# System prompt for generation
SYSTEM_PROMPT = """You are a senior 5G network engineer specializing in Root Cause Analysis (RCA).
Your task is to diagnose why throughput dropped below 600Mbps in a 5G drive-test scenario.

You must analyze:
1. User plane KPIs (RSRP, SINR, throughput, RBs, speed)
2. Engineering parameters (downtilt, cell locations, PCI)
3. Handover patterns

Always structure your response with:
- <reasoning> tags containing step-by-step analysis
- Check ALL potential root causes systematically
- End with \\boxed{CX} where X is 1-8

The 8 root causes are:
C1: Downtilt too large (total tilt > 25°)
C2: Coverage distance > 1km (overshooting)
C3: Neighbor provides higher throughput
C4: Non-colocated co-frequency overlap (cells within 3-5dB)
C5: Frequent handovers (3+ in 10 samples)
C6: PCI mod 30 collision (serving % 30 == neighbor % 30)
C7: Speed > 40 km/h
C8: Scheduled RBs < 160"""


def build_generation_prompt(question: str, answer: str) -> str:
    """
    Build the full prompt for generating a reasoning trace.
    Uses few-shot examples relevant to the target answer.
    """
    few_shot = get_few_shot_examples(answer, num_examples=2)
    
    prompt = f"""{SYSTEM_PROMPT}

Here are examples of expert diagnostic reasoning:

{few_shot}

---

Now analyze this case. The correct root cause is {answer}.
Explain step-by-step how you would diagnose this from the data.

{question}

Provide your diagnostic reasoning:

<reasoning>"""
    
    return prompt


if __name__ == "__main__":
    # Test the examples
    print("Expert Examples loaded successfully!")
    print(f"Number of examples: {len(EXPERT_EXAMPLES)}")
    
    # Test few-shot selection
    for cause in ["C1", "C4", "C6", "C8"]:
        examples = get_few_shot_examples(cause)
        print(f"\nFew-shot for {cause}: {len(examples.split('---'))} examples")
