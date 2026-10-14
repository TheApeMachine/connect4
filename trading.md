### 1. Known Component Ideas

*   **Perpetual Swap Funding Settlement & TWAP Mechanics:** Crypto perpetual futures contracts lack expiration and maintain peg to spot indices via periodic funding payments (e.g., 8-hour cycles on major CEXs or hourly/continuous cycles on DEXs). The funding rate $F$ is calculated from the Time-Weighted Average Price (TWAP) premium/discount of the perpetual contract mark price relative to the underlying spot index over an observation window leading up to settlement time $t_k$.
*   **Hawkes Point Processes for Self-Exciting Order Flow:** In market microstructure, aggressive order arrivals (market orders sweeping the limit order book) are modeled as point processes with conditional intensity:
    $$\lambda(t) = \mu(t) + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}$$
    where $\mu(t)$ is the baseline exogenous arrival rate, $\alpha$ is the excitation amplitude, and $\beta$ is the exponential decay rate. The system branching ratio $\eta = \alpha / \beta$ measures self-excitation: when $\eta \to 1$, the system approaches a "critical" state where a single trade triggers an avalanche of follow-on trades (liquidity cascades).
*   **Transient Market Impact & Order Book Resiliency (Kyle–Obizhaeva / Farmer Dynamics):** Transient price impact from aggressive order flow decays as limit orders replenish the order book. When aggressive trade arrivals occur faster than the replenishment rate, price impact scales non-linearly with flow velocity rather than linearly with aggregate volume.

---

### 2. Hidden or Under-Used Link

Standard execution and carry-trading literature treats crypto perpetual funding capture as a static, linear carry yield, assuming execution costs and market impact around funding timestamps are simple functions of average bid-ask spreads or daily volume profiles. Separately, Hawkes process literature in microstructure typically models baseline intensity $\mu(t)$ as constant or slowly varying with diurnal human activity patterns.

The under-explored connection is the **endogenous coupling between deterministic TWAP funding calculation windows and Hawkes process criticality dynamics**:

1. Market participants (hedging desks, delta-neutral basis traders, miners) face strong financial incentives to adjust positions within the TWAP calculation window immediately preceding discrete funding timestamp $t_k$.
2. This concentrated incentive creates a deterministic, time-dependent spike in the Hawkes baseline intensity $\mu(t)$ as $t \to t_k$.
3. In a self-exciting Hawkes system, increasing the baseline rate $\mu(t)$ while intrinsic feedback mechanisms $(\alpha, \beta)$ are present multiplies the expected cluster size $E[N] = \frac{1}{1 - \eta}$.
4. As a result, funding settlement windows do not simply experience elevated volume; they trigger a structural regime shift where order flow approaches **Hawkes Criticality ($\eta \to 1$)**. During these windows, transient order book decay dominates, causing adverse selection and price impact per unit of executed volume to scale non-linearly relative to non-funding periods.

---

### 3. Resulting Trading-Relevant Mechanism

**Hawkes-Criticality-Aware Execution & Market-Making Controller:**

*   **Real-Time Parameter Calibration:** The execution system continuously fits a high-frequency Hawkes process to tick-level trade arrivals (separated by buy/sell initiative) on the perpetual L2/L3 order book, calculating real-time estimates of baseline intensity $\mu(t)$, excitation $\alpha(t)$, decay $\beta(t)$, and branching ratio $\eta(t) = \alpha(t) / \beta(t)$.
*   **Dynamic Execution Scheduling (Toxicity Avoidance):** For an algorithmic trader unwinding or establishing a basis hedge around funding time $t_k$, the scheduler monitors $\eta(t)$ during the pre-funding window $[t_k - \Delta t, t_k]$. 
    *   If $\eta(t) > \eta_{\text{critical}}$ (e.g., $0.85$), the scheduler temporarily pauses passive limit execution and defers market orders to an earlier or later window outside the TWAP calculation period—even if doing so incurs a slight adverse funding payment—because expected slippage during the Hawkes cascade exceeds the marginal funding carry.
*   **Adaptive Liquidity Provisioning:** Market-making algorithms widen quotes or reduce passive bid/ask sizes non-linearly as $\eta(t) \to 1$, scaling spread width proportionally to $(1 - \eta(t))^{-1}$ to protect against flow toxicity (adverse selection) during self-exciting sweeps.

---

### 4. Concrete Testable Prediction

Across major high-liquidity crypto perpetual pairs (e.g., BTC-USDT, ETH-USDT, SOL-USDT):

1. In funding settlement periods where absolute funding rate $|F_{t_k}| > 0.03\%$ (approx. 33% annualized), the estimated Hawkes branching ratio $\eta(t)$ of aggressive market orders will systematically increase during the 15-minute window prior to settlement ($t_k - 15\text{m} \le t < t_k$), approaching values near $0.85 - 0.95$, compared to baseline values ($0.40 - 0.60$) observed mid-cycle ($t_k + 3\text{h} \le t < t_k + 5\text{h}$).
2. Passive limit order fill toxicity—measured as the 5-second post-trade markout loss $\Delta P_{t+5\text{s}} = \text{sign}(\text{trade}) \cdot (P_{t+5\text{s}} - P_t)$—will be at least $2.5\times$ higher per unit of volume executed in the 10 minutes prior to funding settlement compared to the 10 minutes immediately following settlement, controlling for ambient broad-market volatility ($\sigma$).

---

### 5. Falsification Test

*   **Data Requirements:** Microsecond-stamped tick-by-tick order book (L2/L3) and trade flow data for the top 10 perpetual swap contracts by open interest across a 12-month period.
*   **Procedure:**
    1. Divide trade data into non-overlapping 10-minute rolling windows. Estimate Hawkes parameters $(\mu, \alpha, \beta)$ via Maximum Likelihood Estimation (MLE) for each window.
    2. Compute the 5-second markout loss for all passive limit order fills within each window.
    3. Stratify windows into two subsets for periods where $|F_{t_k}| > 0.03\%$:
       *   **Set A (Pre-Settlement):** $[t_k - 15\text{m}, t_k]$
       *   **Set B (Post-Settlement Baseline):** $[t_k + 15\text{m}, t_k + 30\text{m}]$
    4. Run a two-sample Mann-Whitney U test comparing the branching ratios $\eta_A$ vs. $\eta_B$ and unit fill toxicity between Set A and Set B.
*   **Falsification Criteria:**
    *   The hypothesis is **falsified** if:
        1. The median branching ratio $\eta_A$ in Set A is not statistically significantly greater than $\eta_B$ in Set B ($p \ge 0.01$).
        2. Average 5-second post-fill markout loss per unit volume in Set A is less than or equal to $1.2\times$ that of Set B.
        3. An execution strategy using an adaptive Hawkes-criticality execution threshold fails to achieve lower net slippage + funding cost relative to a baseline TWAP strategy across 1,000 backtested execution runs.

---

### 6. Nearest Known Alternative Explanation

*   **Linear Volume Seasonality / Static Rebalancing Hypothesis:**
    *   *Alternative View:* Conventional quantitative models attribute pre-settlement price movements and spread widening strictly to a deterministic increase in aggregate trading volume as basis arbitrageurs execute rebalancing trades at funding boundaries.
    *   *Key Difference:* The linear volume hypothesis predicts that execution cost and price impact scale predictably as a function of aggregate volume ($I \propto \sqrt{V}$). In contrast, the **Hawkes Criticality Mechanism** demonstrates that costs are driven by *self-exciting trade clustering dynamics* ($\eta \to 1$), where transient order book depletion causes non-linear price impact cascades. This implies execution performance is optimized by tracking point-process criticality metrics ($\eta_t$) rather than simple volume-clock TWAP schedules.

---

### 1. Known Component Ideas

*   **COIN-Margined (Inverse) vs. USDT-Margined (Linear) Perpetual Swap Microstructure:** 
    Crypto derivatives platforms offer two distinct collateral structures: Linear contracts (USDT-M or USDC-M), where margin and PnL are settled in stablecoins, and Inverse/Coin-Margined contracts (COIN-M), where margin and PnL are settled in the base asset (e.g., BTC or ETH). In COIN-M contracts, position collateral value is endogenously coupled to the underlying asset price $S_t$, creating a quadratic margin equity curve: $E_t = N \cdot S_t + \text{PnL}(S_t)$. Consequently, downward price shocks decrease position equity and collateral value simultaneously, causing COIN-M long positions to reach forced liquidation thresholds non-linearly faster than USDT-M positions.
*   **Hawkes Point Processes & Branching Ratio Criticality:** 
    Order arrivals, market sweeps, and forced liquidations can be modeled as self-exciting point processes with conditional intensity:
    $$\lambda_i(t) = \mu_i(t) + \sum_{j} \int_0^t \alpha_{ij} e^{-\beta_{ij}(t-s)} dN_j(s)$$
    where $\mu_i(t)$ is baseline exogenous arrival intensity, $\alpha_{ij}$ is the cross-excitation amplitude, and $\beta_{ij}$ is the decay rate. The spectral radius of the matrix $\boldsymbol{\Gamma} = (\alpha_{ij}/\beta_{ij})$ defines the system branching ratio $\eta$. When $\eta \to 1$, the order flow reaches a critical regime where a single market order triggers an explosive avalanche of follow-on market orders and liquidations.
*   **Cross-Venue Index Arbitrage & Transient Price Impact:** 
    High-frequency statistical arbitrageurs continuously maintain mark-to-index price alignment between COIN-M and USDT-M order books. An aggressive liquidity sweep on one venue propagates order flow into the other venue via automated index-arbitrage execution, depleting available order book depth.

---

### 2. The Hidden or Under-Used Link

Standard crypto quantitative literature treats COIN-M and USDT-M perpetuals as near-identical instruments whose price differences reflect simple interest rate differentials or static borrow fee dynamics. Separately, Hawkes process applications in order book microstructure typically treat excitation parameters $(\alpha_{ij}, \beta_{ij})$ as invariant to collateral type or constant across directional price movements.

The overlooked link is that **base-asset collateral convexity in COIN-M perps induces a price-direction-dependent asymmetry in Hawkes process self-excitation, causing the cross-venue funding rate spread ($F_{\text{COIN-M}} - F_{\text{USDT-M}}$) to act as a real-time proxy for differential Hawkes branching ratios ($\eta_{\text{COIN-M}} - \eta_{\text{USDT-M}}$)**:

1. During negative price drift ($\Delta S_t < 0$), the collateral value of COIN-M long positions deteriorates alongside paper losses, creating a non-linear surge in the conditional probability of forced engine liquidations per unit price decline.
2. This collateral degradation elevates the self-excitation parameter $\alpha_{\text{COIN}}(S_t)$ specifically on downside trade flows, driving the branching ratio $\eta_{\text{COIN}}$ toward criticality ($\eta \to 1$) while $\eta_{\text{USDT}}$ remains sub-critical ($\eta \ll 1$).
3. Because index arbitrageurs tightly link both order books, an aggressive sell flow arriving on the USDT-M venue when $\eta_{\text{COIN}} \approx 1$ propagates into the COIN-M venue and triggers an explosive, self-sustaining liquidation cascade that feeds back into spot price indices.
4. The funding rate spread $F_{\text{COIN-M}} - F_{\text{USDT-M}}$ adjusts to compensate market participants for taking on this asymmetric liquidation cascade risk, making it an observable state variable for differential order flow criticality.

---

### 3. Resulting Trading-Relevant Mechanism

**Cross-Collateral Hawkes Criticality Execution & Market-Making Engine:**

1.  **Real-Time Criticality Estimation:**
    The system fits a bivariate Hawkes process on microsecond-stamped tick trade data (distinguishing organic trades from exchange liquidation messages) for corresponding COIN-M and USDT-M perpetual contracts. It tracks live estimates of branching ratios $\eta_{\text{USDT}}(t)$ and $\eta_{\text{COIN}}(t)$, along with the real-time funding spread $\Delta F(t) = F_{\text{COIN-M}}(t) - F_{\text{USDT-M}}(t)$.
2.  **Cascading Liquidity Shift Signal:**
    When directional price drift is negative ($\Delta S_t < 0$) and $\eta_{\text{COIN}}(t) > \eta_{\text{threshold}}$ (e.g., $0.85$) while $\eta_{\text{USDT}}(t) < 0.60$, the system enters a "Collateral Cascade State." In this regime, sell orders executed on USDT-M perps have a high conditional probability of inducing non-linear price overshoot on COIN-M perps.
3.  **Algorithmic Execution & Market-Making Response:**
    *   *Optimal Execution:* Algorithms executing buy orders temporarily defer execution on USDT-M perps and submit passive buy limit orders deeply below the bid on COIN-M perps, capturing execution discount during predicted liquidation sweeps.
    *   *Adaptive Market Making:* Market makers providing liquidity on COIN-M perps dynamically widen downside bid spreads by a factor proportional to $(1 - \eta_{\text{COIN}}(t))^{-1}$, mitigating adverse selection risk during cascading liquidations.

---

### 4. Concrete Testable Prediction

On major crypto exchanges offering both COIN-M and USDT-M perpetual contracts (e.g., Binance, Bybit):

1.  In time windows where 1-hour price drift is negative ($\Delta S_{1\text{h}} < -2\%$) and open interest in COIN-M contracts is high (top quartile relative to 30-day moving average), the ratio of Hawkes branching ratios $\eta_{\text{COIN}} / \eta_{\text{USDT}}$ will exceed $1.4$ with statistical significance ($p < 0.001$), whereas during positive price drift regimes ($\Delta S_{1\text{h}} > +2\%$), the ratio $\eta_{\text{COIN}} / \eta_{\text{USDT}}$ will remain close to $1.0$.
2.  An aggressive market sell order of fixed size $V$ executed on USDT-M perps when $\eta_{\text{COIN}} > 0.85$ will generate a 1-minute price impact on the spot index that is at least $2.2\times$ larger than an identical market sell order executed when $\eta_{\text{COIN}} < 0.50$, holding USDT-M order book depth constant.

---

### 5. Falsification Test

*   **Dataset:** Tick-by-tick order book (L2/L3) and trade flow data (including forced liquidation messages) for BTC-USDT and BTC-USD (COIN-M) perpetual contracts across Binance and Bybit over a 24-month historical period.
*   **Procedure:**
    1. Estimate bivariate Hawkes process parameters $(\mu, \alpha, \beta)$ on 15-minute rolling windows across the full dataset to construct continuous time series for $\eta_{\text{COIN}}(t)$ and $\eta_{\text{USDT}}(t)$.
    2. Segment windows into subsets based on price drift sign ($\Delta S < 0$ vs $\Delta S > 0$) and COIN-M open interest levels.
    3. Calculate 1-minute post-trade markout losses and cross-venue price impact following large market sell orders executed on USDT-M perps.
*   **Falsification Criteria:**
    The hypothesis is **falsified** if:
    1. Estimated branching ratio $\eta_{\text{COIN}}$ fails to demonstrate statistically significant asymmetric elevation during downward price drift regimes relative to upward price drift regimes ($p \ge 0.01$, Wilcoxon signed-rank test).
    2. Transient market impact of USDT-M sell flow on COIN-M mark price during high $\eta_{\text{COIN}}$ regimes is statistically indistinguishable ($p \ge 0.05$) from impact during low $\eta_{\text{COIN}}$ regimes.
    3. An execution strategy utilizing this cross-collateral criticality signal achieves no statistically significant reduction in implementation shortfall compared to a standard VWAP algorithm across 500 simulated execution runs.

---

### 6. Nearest Known Alternative Explanation

*   **Generic Market Volatility / Gamma Hedging Hypothesis:**
    *   *Alternative View:* Standard quantitative literature attributes cross-venue price overshoots during market downturns to generic market-wide volatility spikes ($\sigma$) or option market makers dynamically delta-hedging long/short gamma positions in spot markets.
    *   *Key Difference:* The generic volatility hypothesis predicts symmetric price impact and order book depletion across all derivative contract types regardless of collateral structure. In contrast, the **Cross-Collateral Hawkes Criticality Mechanism** explicitly predicts a venue-specific asymmetry tied to collateral type: self-excitation ($\eta_{\text{COIN}} \to 1$) is concentrated in base-asset margined contracts due to collateral value erosion during price drops. This allows trading algorithms to exploit differential excitation between venues rather than treating volatility as uniform across derivatives products.

---

### Research Design: Mark-Lag Cross-Excited Liquidation & Inventory Control (ML-XMM)

---

### 1. Known Component Ideas

*   **Component A: Derivative Exchange Mark Price Index Smoothing (Exchange Microstructure)**
    Major crypto perpetual swap venues (e.g., Binance, Bybit, OKX) do not trigger forced liquidations based on instantaneous perpetual order book prices. To prevent manipulation, liquidations are triggered by a calculated **Mark Price**, typically defined as a combination of a composite Spot Price Index (a weighted basket across multiple spot exchanges) and a smoothed moving average (e.g., TWAP or EWMA) of the local perpetual basis ($P_{\text{perp, mid}} - P_{\text{spot, index}}$).
*   **Component B: Cross-Excitation Hawkes Processes (Point Process Microstructure)**
    Hawkes processes are self- and cross-exciting point processes where the arrival of an event (e.g., a large trade or cancellation) transiently increases the conditional intensity $\lambda(t)$ (hazard rate) of future events on the same venue or related venues.
*   **Component C: Cross-Venue Collateral Mobility Latency ($T_{\text{settle}}$) & Inventory Limits (Execution Friction)**
    Cross-venue arbitrageurs and market makers (MMs) operate under strict capital allocation constraints. Transferring margin collateral between isolated exchanges or moving funds via on-chain transactions or custodial settlement networks (e.g., Fireblocks, ClearLoop) incurs a latency $T_{\text{settle}}$ (ranging from seconds to minutes). Consequently, an MM’s local capacity to absorb one-sided inventory without breaching exchange maintenance margin rules is fixed over short time horizons.

---

### 2. The Under-Used / Hidden Link

Standard high-frequency market-making and signal processing models treat order flow toxicity or inventory risk as a continuous function of local order book imbalance (OBI) or realized trade volatility. They treat forced liquidations as exogenous, high-volatility trade bursts.

The overlooked link emerges from the structural interaction between **Mark Price calculation delays** and **cross-venue collateral settlement latencies ($T_{\text{settle}}$)**:

1. When a sharp, directional liquidity shock originates on a leading spot exchange (e.g., Coinbase spot), the underlying Spot Index updates almost instantly, but the perpetual's **Mark Price** responds with a deterministic time lag $\tau$, dictated by the smoothing filter (e.g., TWAP window).
2. During this delay window $\Delta t \in (0, \tau)$, the perpetual mid-price may adjust due to speculative order flow, but **forced liquidations cannot execute until the Mark Price itself crosses the liquidation thresholds of leveraged accounts**.
3. If $T_{\text{settle}} > \tau$, market makers on the perpetual exchange know that a deterministic, high-volume wave of forced market orders (liquidations) is imminent once the Mark Price filter catches up. However, MMs cannot move physical collateral to the perpetual exchange fast enough ($T_{\text{settle}}$) to increase their local balance sheet capacity.

Thus, liquidation hazard rate is not merely cross-excited by generic market velocity; it is a deterministic function of the **integrated gap between the leading spot micro-price and the lagged perpetual Mark Price**, modulated by the ratio of estimated liquidation volume to local MM collateral buffer.

---

### 3. Resulting Trading-Relevant Mechanism

#### Signal Formulation
Define the instantaneous **Mark-Lag Divergence ($G(t)$)**:
$$G(t) = P_{\text{spot, micro}}^{\text{leading}}(t) - P_{\text{perp, mark}}(t)$$

Model the conditional intensity $\lambda_{\text{liq}}(t)$ of forced liquidation arrivals on the perpetual exchange as a cross-excited Hawkes process parameterized by $G(t)$:
$$\lambda_{\text{liq}}(t) = \mu_0 + \sum_{T_k < t} \alpha \, e^{-\beta (t - T_k)} + \gamma \cdot \max\left(0, |G(t)| - \theta_{\text{margin}}\right)$$
where:
*   $\mu_0$ is the baseline liquidation intensity.
*   $\alpha, \beta$ govern the self-excitation decay of liquidation cascades.
*   $\theta_{\text{margin}}$ represents the estimated aggregate distance to liquidation for high-leverage Open Interest clusters.
*   $\gamma$ scales the cross-excitation forced by the spot-mark divergence.

#### Execution Logic for Automated Market Makers
Instead of adjusting bid-ask spreads symmetrically via standard inventory control models (e.g., Avellaneda-Stoikov), the strategy operates as follows:

1. **Preemptive Depth Asymmetry:** When $|G(t)| > \theta_{\text{margin}}$ and $\frac{d|G|}{dt} > 0$, compute $T_{\text{eta}}$, the estimated time remaining before $P_{\text{perp, mark}}$ crosses the liquidation threshold.
2. **Dynamic Withdrawal:** If $T_{\text{eta}} < T_{\text{settle}}$, the market maker algorithmically cancels resting limit orders on the side that would absorb liquidations (e.g., pulling bids during a downward spot shock) *before* any forced liquidation orders are generated by the matching engine.
3. **Cross-Venue Offset:** Simultaneously, the algorithm submits passive maker orders on the leading spot venue, accumulating inventory on the venue where settlement and collateral constraints are not binding, capturing the index lag delta.

---

### 4. Concrete Market Prediction

1. **Bifurcated Depth Decay:** On perpetual swap exchanges using smoothed Mark Price indexes, order book depth at levels 5–20 ($L_5 - L_{20}$) on the perp book will experience an exponential decay **prior to** the execution of the first forced liquidation trade, correlated specifically with $G(t)$ crossing $\theta_{\text{margin}}$.
2. **Liquidation Peak Lag:** Peak forced liquidation volume on derivative venues will systematically lag the peak velocity ($\max |\frac{dP}{dt}|$) of leading spot exchanges by an interval equal to the Mark Price filter time constant $\tau$, rather than coinciding with peak spot velocity.

---

### 5. Falsification Test & Experimental Protocol

#### Data Requirements
*   100ms-resolution order book snapshots (L2, top 20 levels) and tick trade feeds for BTC-USDT and ETH-USDT across:
    *   *Leading Spot Venues:* Coinbase, Kraken, Binance Spot.
    *   *Perpetual Derivatives Venues:* Binance Futures, Bybit Perp.
*   Real-time Mark Price update feeds and WebSocket forced-liquidation execution streams over a continuous 12-month historical window.

#### Testing Protocol
1. **Event Selection:** Isolate all regime shifts where the 5-second spot price volatility exceeds $3\sigma$ relative to a rolling 1-hour window.
2. **Model Comparison:** Fit two competing statistical models to predict liquidation arrival intensity $\lambda_{\text{liq}}(t)$ and order book depth decay rate $\frac{\partial D_{L5-L20}}{\partial t}$:
    *   **Null Model (Baseline Microstructure):** Multi-variate Hawkes model using local perpetual trade velocity, local Order Book Imbalance (OBI), and local realized volatility.
    *   **Hypothesis Model (ML-XMM):** Multi-variate Hawkes model incorporating the Mark-Lag Divergence $G(t)$ and the Mark Price filter parameter $\tau$.
3. **Evaluation Metrics:**
    *   Out-of-sample log-likelihood of liquidation order arrival times.
    *   Receiver Operating Characteristic / Area Under Curve (ROC-AUC) for predicting $L_5 - L_{20}$ depth collapse events 500ms to 2000ms prior to occurrence.

#### Falsification Criteria
The hypothesis is **falsified** if:
1. The ML-XMM model fails to demonstrate a statistically significant improvement in out-of-sample log-likelihood over the Null Model ($p > 0.01$ via Likelihood Ratio Test).
2. The lead time between $G(t)$ expansion and derivative depth decay exhibits zero statistically significant correlation ($r \approx 0$), indicating MMs do not adjust quotes until liquidations actually hit the local matching engine.

---

### 6. Nearest Known Alternative Explanations

1. **Standard Adverse Selection / VPIN (Volume-Synchronized Probability of Informed Trading):**
   *   *Alternative Argument:* MMs withdraw liquidity solely because high-volume market orders from informed traders exhaust the order book, causing passive liquidity removal through execution rather than anticipatory quote pulling.
   *   *Distinction:* The proposed ML-XMM model explicitly predicts quote pulling *before* order execution occurs on the perp venue, driven by the deterministic calculation lag of $P_{\text{perp, mark}}$.

2. **Generic Volatility-Based Inventory Control (VaR Skew):**
   *   *Alternative Argument:* MMs pull quotes because short-term realized price volatility breaches internal Risk-at-Risk (VaR) or variance exposure limits.
   *   *Distinction:* Generic VaR limits respond symmetrically to price volatility on the local book, whereas ML-XMM predicts directional, asymmetric depth removal linked specifically to external spot index divergence and mark price calculation mechanics.

---

*Disclaimer: This documentation is framed strictly for academic research design and quantitative modeling purposes. It does not constitute financial or trading advice.*

---

### 1. Known Component Ideas

1. **Multivariate Hawkes Processes in Microstructure (Self- and Cross-Excitation):**
   A point process framework where order arrivals (e.g., spot buys, spot sells, perpetual buys, perpetual sells) have conditional intensities $\lambda_i(t) = \mu_i + \sum_{j} \int_0^t \alpha_{ij} e^{-\beta_{ij}(t-s)} dN_j(s)$. The spectral radius $\rho(\mathbf{\Gamma})$ of the excitation matrix $\mathbf{\Gamma}_{ij} = \alpha_{ij}/\beta_{ij}$ measures order flow reflexivity and proximity to endogenous liquidity cascades ($\rho \to 1$).
2. **Perpetual Swap Funding Rate Mechanics:**
   Crypto perpetual swap contracts maintain peg to spot through periodic (e.g., 8-hour) or continuous cash transfers between long and short positions based on the premium rate $F_t = \frac{P_{\text{perp}} - P_{\text{spot}}}{P_{\text{spot}}}$. At discrete settlement timestamps $T_{\text{fund}}$, net position holders pay or receive $F_t \times \text{Position Size}$.
3. **Inventory-Dependent Optimal Execution & Market Making (Avellaneda–Stoikov Framework):**
   Optimal quoting/execution models set reservation prices $r(s, q, t) = s - q \gamma \sigma^2 (T-t)$ and bid/ask spreads based on current inventory $q$, risk aversion $\gamma$, volatility $\sigma$, and local market order arrival intensities.

---

### 2. The Hidden / Under-Used Link

In quant finance, Hawkes processes are typically calibrated on high-frequency order book data assuming either time-homogeneity or simple intraday seasonality. Conversely, crypto perpetual funding rates are usually analyzed as low-frequency carry yield or discrete arbitrage opportunities.

**The overlooked link is that deterministic funding settlement countdowns ($\tau = T_{\text{fund}} - t$) act as exogenous boundary conditions that systematically drive the cross-venue Hawkes excitation matrix $\mathbf{\Gamma}(\tau, F_t)$ into asymmetric, near-critical regimes ($\rho \to 1$).**

As $\tau \to 0$, position holders face a binary choice: hold through settlement and incur/collect funding fee $F_t$, or close/hedge positions in the limit order book. When $|F_t|$ is large relative to venue taker fees, impatient positioning triggers a directional cascade where perpetual market orders excite spot market orders ($\alpha_{\text{perp} \to \text{spot}} \gg \alpha_{\text{spot} \to \text{perp}}$). Consequently, the microsecond-level order flow reflexivity $\rho(\mathbf{\Gamma})$ is direct, deterministic function of the funding countdown and predicted funding rate $F_t$.

---

### 3. Resulting Trading-Relevant Mechanism

**Funding-Hawkes Adaptive Execution and Quote Skewing (FH-AEQS):**

Instead of relying on static inventory parameters ($\gamma$) or empirical time-of-day scaling, an algorithmic execution engine or market-making system continuously estimates a time-varying 4D Hawkes process (Spot Buy/Sell, Perp Buy/Sell) parameterized by funding boundary dynamics:

1. **Spectral Risk Scaling:** The algorithm calculates the cross-venue spectral radius $\rho(\mathbf{\Gamma}(\tau, F_t))$. As $\rho \to 1$, market reflexivity increases, signifying that incoming limit orders face severe adverse selection from funding-driven cascades.
2. **Dynamic Reservation Price Adjustment:** The traditional Avellaneda–Stoikov reservation price is modified to incorporate the Hawkes reflexivity multiplier and directional funding alignment:
   $$r(s, q, \tau) = s - q \cdot \gamma \cdot \sigma^2 \cdot \left( \frac{1}{1 - \rho(\mathbf{\Gamma}(\tau, F_t))} \right) \cdot \Big[1 + \text{sign}(q \cdot F_t) \cdot \kappa\Big]$$
   where $\kappa > 0$ penalizes holding inventory aligned with the toxic, funding-driven exit flow.
3. **Execution Routing:** When $\rho(\mathbf{\Gamma})$ exceeds a critical threshold ($\rho > 0.85$) prior to $T_{\text{fund}}$, parent order execution shifts dynamically from passive limit order posting to cross-venue, low-latency sweeping (or liquidity taking on the unpopulated venue) before spread widening and book depletion occur.

---

### 4. Concrete Prediction

In major crypto currency trading pairs (e.g., BTC/USDT, ETH/USDT) across major exchanges, during the 15-minute window preceding funding settlement ($\tau < 900\text{s}$):

* When absolute predicted funding $|F_t| > 0.03\%$, the cross-venue excitation parameter $\alpha_{\text{perp} \to \text{spot}}$ will exhibit an exponential increase as $\tau \to 0$, causing the Hawkes spectral radius $\rho(\mathbf{\Gamma})$ to approach 1.
* A market-making strategy utilizing FH-AEQS will reduce adverse selection costs (measured as 1-second post-fill price decay) by **15–30%** during pre-funding windows compared to standard time-invariant or pure volatility-scaled inventory models.

---

### 5. Falsification Test

**Setup:**
* **Data:** Tick-by-tick Level-3 or L2 order book updates and trade prints for BTC/USDT Spot and Perpetual contracts across Binance and Bybit over 12 consecutive months.
* **Methodology:**
  1. Fit a 4-dimensional Hawkes process across moving 5-minute rolling windows.
  2. Estimate parameters ($\mu_i, \alpha_{ij}, \beta_{ij}$) during high-funding pre-settlement windows ($\tau < 15\text{ min}, |F_t| > 0.03\%$) versus control windows ($\tau > 2\text{ hours}$ or $|F_t| \approx 0$).
  3. Run a simulated optimal-execution backtest (e.g., liquidating 100 BTC over 10 minutes) comparing FH-AEQS against a standard Volatility-Scaled TWAP and static Avellaneda–Stoikov strategy.

**Falsification Criteria:**
The hypothesis is **falsified** if:
1. A Likelihood Ratio Test shows no statistically significant improvement ($p > 0.05$) when modeling excitation parameters $\mathbf{\Gamma}$ as a function of $(\tau, F_t)$ versus a state-independent baseline Hawkes model.
2. The implementation shortfall of FH-AEQS shows no statistically significant improvement ($p > 0.05$) over standard volatility-scaled TWAP during high-funding pre-settlement intervals, demonstrating that funding-driven order flow does not generate structured, cross-venue self-excitation.

---

### 6. Nearest Known Alternative Explanation

**Diurnal / Scheduled Volatility & Volume Scaling:**
Standard quant execution literature attributes order book degradation and wider spreads prior to known events (such as macro releases or funding settlement) to simple exogenous volume spikes or diurnal volatility profiles ($\sigma(t)$). Under this traditional view, price impact increases purely because aggregate trading volume rises, which can be mitigated using a standard volume-profile TWAP or scaling bid/ask spreads by implied volatility, without accounting for directional cross-venue point-process excitation or spectral radius criticality.

---

### Candidate Concept: Funding-Modulated Order Flow Imbalance (FM-OFI) Engine for Microstructure Execution and Regime Detection

---

### 1. Known Component Ideas

1. **Order Flow Imbalance (OFI) & Linear Price Impact (Market Microstructure)**  
   *Cont, Kukanov, and Stoikov (2014)* established that short-term price movements in limit order books (LOB) are driven by Order Flow Imbalance ($OFI_t$), which aggregates net changes in bid and ask depth alongside aggressive order arrivals at the top L2 levels. In classical microstructure models, price impact per unit of $OFI$ ($\lambda$) is assumed to be symmetric between buy and sell flows under stationary conditions.

2. **Perpetual Swap Funding Carry Kinetics (Crypto Microstructure)**  
   Crypto perpetual futures use a periodic or continuous funding rate mechanism ($f_t$) to anchor perpetual contract prices to the underlying spot index. Position holders on the dominant side pay the counterparty a rate proportional to their notionally leveraged position at discrete settlement boundaries $T_{\text{settle}}$ (e.g., every 8 hours on Binance/Bybit) or continuously (e.g., Hyperliquid/dYdX).

3. **Inventory Reservation Price Decay (Optimal Execution & Control Theory)**  
   *Avellaneda and Stoikov (2008)* and *Guéant, Tapia, and Lehalle (2012)* showed that market participants subject to inventory risk alter their reservation prices ($\hat{s}_t = s_t - q \gamma \sigma^2 \Delta t$), creating asymmetric quote placements to avoid holding unhedged positions against adverse inventory drift.

---

### 2. The Overlooked Link

In traditional equities or spot FX, inventory holding costs (e.g., overnight repo, dividend yields) change slowly and are negligible on sub-second or intraday execution timescales. In crypto perpetual markets, however, high funding rates ($f_t$) combined with high leverage create an impending, deterministic inventory carry penalty that intensifies as the settlement boundary approaches.

The overlooked link is that **perpetual funding rate kinetics alter the directional liquidity elasticity of the limit order book itself**. 

When perpetual funding is strongly positive ($f_t \gg 0$) and the settlement epoch boundary $T_{\text{settle}}$ approaches:
* Leveraged long position holders face an increasing, deterministic cash penalty to maintain positions past $T_{\text{settle}}$.
* Their inventory reservation prices shift downward, leading to aggressive bid cancellations and lower limit order queue replenishment on the bid side.
* Market makers, anticipating long positioning unwind, widen bid spreads and reduce bid depth while tightening ask depth.

Consequently, price impact per unit of order flow is **not static or symmetric**. Sell-side order flow ($OFI^-$) produces non-linear, disproportionately larger downside price jumps as $t \to T_{\text{settle}}$, whereas buy-side order flow ($OFI^+$) experiences compressed price impact. When funding is negative ($f_t \ll 0$), this elasticity asymmetry cleanly flips.

---

### 3. The Composed Trading-Relevant Mechanism

#### **The Funding-Modulated OFI (FM-OFI) Model**

Instead of modeling price impact with a single symmetric parameter $\Delta P_t = \lambda \cdot OFI_t + \epsilon_t$, we decompose impact into state-dependent asymmetric channels modulated by funding carry kinetics:

$$\Delta P_t = \lambda_t^+ \cdot OFI_t^+ - \lambda_t^- \cdot OFI_t^- + \epsilon_t$$

Where directional price impact sensitivities are formulated as:

$$\lambda_t^- = \lambda_0 \cdot \left(1 + \theta \cdot \max(0, f_t) \cdot \frac{1}{\tau_t + \delta} \cdot \frac{\text{OpenInterest}_t}{\overline{\text{Depth}}_t}\right)$$

$$\lambda_t^+ = \lambda_0 \cdot \left(1 + \theta \cdot \max(0, -f_t) \cdot \frac{1}{\tau_t + \delta} \cdot \frac{\text{OpenInterest}_t}{\overline{\text{Depth}}_t}\right)$$

* $\lambda_0$: Baseline linear price impact coefficient.
* $f_t$: Predicted or prevailing funding rate for the current epoch.
* $\tau_t = T_{\text{settle}} - t$: Time remaining until the next funding settlement timestamp.
* $\frac{\text{OpenInterest}_t}{\overline{\text{Depth}}_t}$: Ratio of exchange open interest to average L2 top-5 book depth (leverage pressure ratio).
* $\theta, \delta$: Positive scaling and regularization parameters.

#### **Algorithmic Crypto Trading Applications**
1. **Adaptive Execution Scheduling**: When executing large sell orders in positive funding regimes ($f_t \gg 0$), the execution algorithm front-loads aggressive taker orders early in the window or shifts away from passive limit posting before $\tau_t$ enters the high-fragility threshold ($\tau_t < 30\text{ min}$).
2. **Automated Market Making (AMM / Centralized Market Making)**: Market makers dynamically adjust their quote asymmetry, widening bid-side quotes and shifting the mid-price reference lower by a factor proportional to $\frac{\lambda_t^-}{\lambda_t^+}$, preventing toxic adverse selection from long-deleveraging cascades prior to funding payout.

---

### 4. Concrete Testable Prediction

In high-frequency order book data for liquid crypto perpetual swaps (e.g., BTC-USDT, ETH-USDT, SOL-USDT):

1. **Impact Asymmetry**: During periods where $|f_t| > 0.03\%$ per 8-hour epoch and $\tau_t < 45\text{ minutes}$, the ratio of downside price impact to upside price impact ($\frac{\lambda_t^-}{\lambda_t^+}$ when $f_t > 0$, or $\frac{\lambda_t^+}{\lambda_t^-}$ when $f_t < 0$) will be strictly greater than $1.5$ ($p < 0.01$).
2. **Monotonic Convergence**: The ratio $\frac{\lambda_t^-}{\lambda_t^+}$ will exhibit a monotonic inverse relationship with time-to-settlement $\tau_t$, reaching its peak during the 5-minute window immediately preceding $T_{\text{settle}}$.
3. **Neutral Baseline**: During baseline neutral funding periods ($|f_t| < 0.005\%$), the impact ratio $\frac{\lambda_t^-}{\lambda_t^+}$ will remain statistically indistinguishable from $1.0$.

---

### 5. Falsification Test

#### **Dataset & Setup**
* High-frequency L2 order book quotes (100ms snapshots) and tick trades for BTC-USDT and ETH-USDT perpetual swaps across major CEXs (Binance, Bybit, OKX) over a continuous 12-month historical window.

#### **Regression Specification**
Estimate a rolling panel regression across 1-minute aggregations:

$$\Delta P_{i, t+\Delta t} = \beta_0 + \beta_1 OFI_{i,t}^+ + \beta_2 OFI_{i,t}^- + \beta_3 \left(OFI_{i,t}^- \cdot f_{i,t} \cdot \tau_{i,t}^{-1}\right) + \beta_4 \left(OFI_{i,t}^+ \cdot f_{i,t} \cdot \tau_{i,t}^{-1}\right) + \boldsymbol{\gamma} \mathbf{X}_{i,t} + \epsilon_{i,t}$$

Where $\mathbf{X}_{i,t}$ controls for contemporaneous volume, realized volatility, and index spot returns.

#### **Falsification Criteria**
The hypothesis is **falsified** if either of the following occurs:
1. **Statistical Insignificance**: The coefficient $\beta_3$ on the triple interaction term $(OFI_{i,t}^- \cdot f_{i,t} \cdot \tau_{i,t}^{-1})$ is not statistically significant ($p \ge 0.05$) or exhibits a sign opposite to theoretical prediction.
2. **Out-of-Sample Simulator Failure**: In a paper-trading simulation including realistic execution fees and latencies (50ms), an adaptive execution agent using the FM-OFI model fails to achieve lower overall implementation shortfall (slippage + adverse selection cost) compared to a standard baseline Cont et al. static $OFI$ model.

---

### 6. Nearest Known Alternative Explanation

* **Generic Volume/Volatility Settlement Seasonality**: Standard microstructure literature attributes price volatility and spread widening around discrete funding hours (UTC 00:00, 08:00, 16:00) to generic diurnal seasonality, institutional trading shifts, or exchange batch processing.
* **Why FM-OFI is Distinct**: Generic diurnal seasonality predicts *symmetric* volatility expansion and spread widening regardless of the funding rate's direction or magnitude. FM-OFI specifically predicts a **directional impact asymmetry** whose orientation reverses deterministically based on whether $f_t > 0$ or $f_t < 0$.

---

### 7. Uncertainty & Boundary Conditions

This concept is presented as a testable hypothesis rather than an established market law. Its effectiveness depends on market structure parameters, such as the proportion of unhedged directional retail leverage versus delta-neutral basis arbitrageurs. If delta-neutral arbitrageurs with low balance-sheet costs absorb directional inventory near $T_{\text{settle}}$ efficiently, the predicted order book liquidity asymmetry may be attenuated.

---

### Candidate Proposal: Basis-Depletion Flow Coupling (BDFC) for Adaptive Crypto Execution and Market Making

---

### 1. Known Component Ideas

1. **Order Flow Imbalance (OFI) and Multi-Level Queue Dynamics** *(Microstructure / Quantitative Finance; Cont, Kukanov, & Stoikov, 2014)*: 
   OFI measures the net change in aggressive liquidity demand across limit order book (LOB) levels over discrete time windows. It provides short-term linear predictive power for price impact by evaluating how changes in bid/ask queue sizes affect immediate price adjustments.

2. **Perpetual Swap Basis Arbitrage & Deterministic Funding Mechanics** *(Crypto Market Microstructure)*: 
   Crypto perpetual futures incorporate a periodic funding rate mechanism to anchor perpetual prices ($P_{\text{mark}}$) to underlying index/spot prices ($P_{\text{index}}$). Algorithmic basis arbitrageurs execute deterministic cash-and-carry or reverse cash-and-carry trades (taking perps and hedging spot) whenever the basis exceeds transaction and capital costs ($|P_{\text{mark}} - P_{\text{index}}| > c_{\text{frictions}}$).

3. **Self-Exciting Point Processes with State-Dependent Intensities (Hawkes Processes)** *(Statistics / Signal Processing; Bacry et al., 2015)*: 
   A class of stochastic point processes where the instantaneous rate (intensity $\lambda_t$) of event arrivals (e.g., market orders) increases transiently in response to recent past events, modeling order flow clustering and transient liquidity exhaustion.

---

### 2. The Overlooked Link

While market microstructurists use OFI to model generic short-term order pressure, and crypto quant desks track funding rates to time cash-and-carry trades, standard systems treat these phenomena as separate: OFI is treated as an unobserved stochastic process, while basis arbitrage is modeled as a macroscopic daily/hourly yield balance.

The overlooked mechanistic link lies in the **spatial distribution of L2 order book queue depth relative to the deterministic execution volume threshold of automated basis arbitrageurs**. 

Programmatic basis arbitrageurs execute market orders when basis diverges past $c_{\text{frictions}}$. However, these execution bots typically target fixed monetary sizes or constrained TWAP windows. If the cumulative available limit order depth $D_{\Delta P}(t) = \sum_{k=1}^m v_k(t)$ within the price distance $\Delta P = |P_{\text{mark}} - P_{\text{index}}|$ is *smaller* than the incoming deterministic arbitrage execution size $V_{\text{arb}}$, the arbitrage market order will completely sweep the local order book queues. 

This queue exhaustion creates a localized "liquidity hole." In crypto markets, sweeping these levels pushes the mark price directly into clusters of exchange maintenance margin thresholds, turning deterministic basis-arbitrage flow into a self-exciting Hawkes liquidation cascade. The ratio of *deterministic arbitrage volume* to *local L2 queue depth* acts as an observable regime-switching control parameter that converts linear price impact into non-linear, self-exciting feedback loops.

---

### 3. Resulting Trading-Relevant Mechanism: Basis-Depletion Flow Coupling (BDFC)

The BDFC system modifies high-frequency market making quote-placing algorithms and institutional execution algorithms (TWAP/VWAP) by overlaying a state-dependent hazard rate for directional liquidity sweeps.

#### A. Real-Time Hazard Rate Formulation
At time $t$, compute the normalized liquidity exhaustion ratio $\rho(t)$:
$$\rho(t) = \frac{\widehat{V}_{\text{arb}}(t)}{\sum_{k=1}^{K} D_k(t)}$$

Where:
* $\widehat{V}_{\text{arb}}(t)$ is the estimated capital size of active basis arbitrageurs (derived from Open Interest changes during recent basis divergence events).
* $\sum_{k=1}^{K} D_k(t)$ is the aggregate L2 order book depth up to depth level $K$ corresponding to $|P_{\text{mark}} - P_{\text{index}}|$.

The instantaneous hazard rate $\lambda_{\text{sweep}}(t)$ for a liquidity-exhausting sweep is modeled as:
$$\lambda_{\text{sweep}}(t) = \lambda_0 + \alpha \cdot \max\left(0, \rho(t) - 1\right) \cdot \mathbb{I}_{\left\{|P_{\text{mark}} - P_{\text{index}}| > c_{\text{frictions}}\right\}} + \int_{0}^t \beta e^{-\gamma(t-s)} dN(s)$$

Where $dN(s)$ tracks aggressive market order fills that deplete order book levels.

#### B. Algorithmic Response Logic
* **For Market Makers (Inventory & Spread Control)**:
  When $\lambda_{\text{sweep}}(t)$ exceeds a critical threshold on the ask side (perpetuals trading at a premium):
  1. **Asymmetric Spread Expansion**: Expand ask quotes ($\delta_a$) non-linearly to avoid adverse fills from the impending arbitrage sweep.
  2. **Post-Cascade Liquidity Provision**: Skew bid quotes ($\delta_b$) lower and expand bid order sizes below the expected liquidation cluster to capture discounted mean-reversion fills once the queue depletes and arbitrage flows cease.
* **For Execution Desks (TWAP/VWAP Order Routing)**:
  When buying perpetuals, if $\lambda_{\text{sweep}}(t)$ spikes, execution urgency is paused (holding market orders) right before the expected arbitrage sweep and resumed immediately after to avoid paying peak implementation shortfall inside the transient liquidity hole.

---

### 4. Concrete Predictions

1. **Non-Linear Impact Threshold**: In high-frequency perpetual swap markets (e.g., BTC-USDT or ETH-USDT on major exchanges), aggressive order impact will deviate from square-root law dynamics: when $\rho(t) > 1.0$ during basis divergence, short-term price impact per unit volume will be $\ge 2.5\times$ higher than during periods with identical OFI but where $\rho(t) < 1.0$.
2. **Adverse Selection Reduction**: A perpetual swap market-making strategy utilizing BDFC dynamic quote asymmetry will achieve a statistically significant reduction in Adverse Selection Loss (measured via 5-second and 30-second markout PnL: $L_{\text{markout}} = q_t (P_{t+\tau} - P_{\text{fill}})$) compared to standard inventory-based models (e.g., classical Avellaneda-Stoikov or linear OFI models).

---

### 5. Falsification Test

#### Test Setup & Dataset
* **Data**: High-frequency tick-by-tick trades, top 20 L2 order-book snapshots (100ms frequency), mark prices, index prices, and open interest for BTC-USDT and ETH-USDT perpetual swaps across major exchanges (e.g., Binance, Bybit, OKX) over a 12-month period.
* **Simulated Environment**: An event-driven backtest or live paper-trading simulation featuring queue-position tracking and realistic maker/taker fees, latency (e.g., 20ms–50ms execution delay), and exchange matching engine rules.

#### Model Comparison
1. **Model A (Baseline - Standard Avellaneda-Stoikov)**: Adjusts quotes based purely on inventory $q_t$ and rolling mid-price volatility $\sigma$.
2. **Model B (Control - OFI-Augmented AS)**: Adjusts quotes using standard linear Order Flow Imbalance and VPIN metrics.
3. **Model C (Treatment - BDFC Model)**: Adjusts quotes using the hazard-rate formulation $\lambda_{\text{sweep}}(t)$ coupling basis divergence to L2 queue depth limits.

#### Metrics
* **Markout PnL ($M_\tau$)**: $\frac{1}{N}\sum_{i=1}^N \text{side}_i \cdot (P_{i+\tau} - P_{\text{fill}, i})$ at $\tau = 5\text{s}$ and $\tau = 30\text{s}$.
* **Sharpe Ratio** of realized strategy PnL.
* **Maximum Adverse Excursion (MAE)** following quote executions.

#### Null Hypothesis ($H_0$)
Under periods of basis divergence ($|P_{\text{mark}} - P_{\text{index}}| > c_{\text{frictions}}$), Model C produces no statistically significant difference in 5-second Markout PnL or strategy Sharpe ratio compared to Model B ($p > 0.01$ via block bootstrap resampling / Diebold-Mariano test).

#### Falsification Criterion
If Model C fails to show a statistically significant reduction in adverse markout loss ($M_{5\text{s}}$) or an improved Sharpe ratio over Model B during high-basis regimes across out-of-sample data, the hypothesis that basis-depletion coupling provides superior predictive/control signal over generic OFI is falsified.

---

### 6. Nearest Known Alternative Explanations

1. **Standard Order Flow Imbalance (OFI) / VPIN (Volume-Synchronized Probability of Toxicity)**:
   * *Alternative View*: High adverse selection for market makers is driven by generic "informed trading" or toxicity in the order flow stream.
   * *Mechanistic Difference*: Standard OFI/VPIN treats order flow as an unobserved, state-agnostic stochastic process driven by homogeneous informed traders. BDFC identifies a specific, deterministic structural driver—basis-arbitrage orders interacting with finite queue depth boundaries—explaining *why* and *when* order flow toxicity spikes non-linearly.

2. **Standard Exchange Liquidation Models**:
   * *Alternative View*: Micro-cascades and price jumps are caused strictly by exchange risk engines executing forced market orders at liquidation thresholds.
   * *Mechanistic Difference*: Standard liquidation models focus on the distribution of leverage and maintenance margin boundaries. BDFC models the *trigger mechanism*: how predictable basis-arbitrage flows consume the intermediate order book queue depth, pushing the mark price into those liquidation boundaries in the first place.