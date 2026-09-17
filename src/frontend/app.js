// ============================================================
// SupplyShield AI - Frontend Application
// Complete app.js
// ============================================================

const API_BASE = "http://127.0.0.1:5000/api";

// ------------------------------------------------------------
// Global state
// ------------------------------------------------------------

let currentFacilityId = null;
let dashboardData = null;

// Risk distribution used by the visible facility alerts.
// Risk Overview is synchronized with this population.
let alertRiskDistribution = {
    HIGH: 0,
    MODERATE: 0,
    LOW: 0,
    VERY_LOW: 0
};


// ============================================================
// DOM HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {
    const el = $(id);

    if (el) {
        el.textContent = value;
    }
}


function escapeHTML(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function formatNumber(value, decimals = 0) {
    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString("en-IN", {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}


function formatPercent(value, decimals = 1) {
    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }

    const number = Number(value);

    const percent =
        Math.abs(number) <= 1
            ? number * 100
            : number;

    return `${percent.toFixed(decimals)}%`;
}


function formatScore(value, decimals = 3) {
    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toFixed(decimals);
}


function formatDate(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleDateString("en-IN", {
        year: "numeric",
        month: "short",
        day: "numeric"
    });
}


// ============================================================
// API HELPER
// ============================================================

async function apiRequest(endpoint, options = {}) {

    const response =
        await fetch(
            `${API_BASE}${endpoint}`,
            {
                headers: {
                    "Content-Type": "application/json",
                    ...(options.headers || {})
                },
                ...options
            }
        );


    if (!response.ok) {

        let message =
            `HTTP ${response.status}`;


        try {

            const errorData =
                await response.json();


            if (errorData.message) {

                message =
                    errorData.message;

            } else if (
                errorData.error
            ) {

                message =
                    errorData.error;
            }

        } catch (_) {
            // Keep default HTTP message.
        }


        throw new Error(message);
    }


    return response.json();
}


// ============================================================
// BACKEND STATUS
// ============================================================

async function checkBackendStatus() {

    const statusEl =
        $("backend-status");


    if (!statusEl) {
        return null;
    }


    try {

        const data =
            await apiRequest(
                "/health"
            );


        statusEl.textContent =
            "Backend Online";


        statusEl.classList.remove(
            "offline"
        );


        statusEl.classList.add(
            "online"
        );


        return data;


    } catch (error) {

        console.error(
            "Backend health check failed:",
            error
        );


        statusEl.textContent =
            "Backend Offline";


        statusEl.classList.remove(
            "online"
        );


        statusEl.classList.add(
            "offline"
        );


        return null;
    }
}


// ============================================================
// DASHBOARD SUMMARY
// ============================================================

async function loadSummary() {

    try {

        const data =
            await apiRequest(
                "/summary"
            );


        dashboardData =
            data;


        console.log(
            "Summary:",
            data
        );


        const summary =
            data.summary ||
            data.data ||
            data;


        // ----------------------------------------------------
        // Core dashboard metrics
        // ----------------------------------------------------

        const facilities =
            summary.facilities ??
            summary.facility_count ??
            summary.total_facilities ??
            0;


        const alerts =
            summary.alerts ??
            summary.alert_count ??
            summary.total_alerts ??
            0;


        const highRisk =
            summary.high_risk ??
            summary.high_risk_count ??
            summary.highRisk ??
            0;


        const regional =
            summary.regional_signals ??
            summary.regional ??
            summary.regional_events ??
            summary.regional_count ??
            summary.spatial_alerts ??
            0;


        setText(
            "facilities-count",
            formatNumber(
                facilities
            )
        );


        setText(
            "alerts-count",
            formatNumber(
                alerts
            )
        );


        setText(
            "high-risk-count",
            formatNumber(
                highRisk
            )
        );


        setText(
            "regional-count",
            formatNumber(
                regional
            )
        );


        // ----------------------------------------------------
        // Latest model assessment
        // ----------------------------------------------------

        updateLatestAssessment(
            summary
        );


        // ----------------------------------------------------
        // Risk overview
        // ----------------------------------------------------

        updateRiskOverview(
            summary
        );


        // ----------------------------------------------------
        // Regional signal validation
        // ----------------------------------------------------

        updateRegionalSignal(
            summary
        );


        return data;


    } catch (error) {

        console.error(
            "Failed to load summary:",
            error
        );


        setText(
            "facilities-count",
            "—"
        );


        setText(
            "alerts-count",
            "—"
        );


        setText(
            "high-risk-count",
            "—"
        );


        setText(
            "regional-count",
            "—"
        );


        updateLatestAssessment(
            null
        );


        updateRegionalSignal(
            null
        );


        return null;
    }
}


// ============================================================
// LATEST MODEL ASSESSMENT
// ============================================================

function updateLatestAssessment(summary) {

    if (!summary) {

        setText(
            "latest-risk-date",
            "—"
        );


        setText(
            "risk-records",
            "—"
        );


        setText(
            "spatial-records",
            "—"
        );


        return;
    }


    const latestDate =
        summary.latest_risk_date ??
        summary.assessment_date ??
        summary.date ??
        null;


    const riskRecords =
        summary.risk_records ??
        summary.risk_record_count ??
        0;


    const spatialRecords =
        summary.spatial_records ??
        summary.spatial_record_count ??
        0;


    setText(
        "latest-risk-date",
        latestDate
            ? formatDate(
                latestDate
            )
            : "—"
    );


    setText(
        "risk-records",
        formatNumber(
            riskRecords
        )
    );


    setText(
        "spatial-records",
        formatNumber(
            spatialRecords
        )
    );


    console.log(
        "Latest assessment:",
        {
            date: latestDate,
            riskRecords,
            spatialRecords
        }
    );
}


// ============================================================
// REGIONAL SIGNAL VALIDATION
// ============================================================

function updateRegionalSignal(summary) {

    if (!summary) {

        renderRegionalSignalUnavailable(
            "Historical regional validation data unavailable."
        );

        return;
    }


    const validation =
        summary.regional_validation ||
        summary.regional_validation_metrics ||
        summary.spatial_validation ||
        null;


    if (!validation) {

        renderRegionalSignalUnavailable(
            "Historical regional validation data unavailable."
        );

        return;
    }


    const signalRecords =
        validation.regional_signal_records ??
        validation.signal_records ??
        validation.records_with_signal ??
        null;


    const signalStockoutRate =
        validation.regional_signal_stockout_rate ??
        validation.stockout_rate_signal ??
        validation.signal_stockout_rate ??
        null;


    const noSignalStockoutRate =
        validation.no_signal_stockout_rate ??
        validation.stockout_rate_no_signal ??
        validation.no_signal_rate ??
        null;


    const signalRatio =
        validation.regional_signal_ratio ??
        validation.signal_ratio ??
        validation.ratio ??
        null;


    if (
        signalRecords === null &&
        signalStockoutRate === null &&
        noSignalStockoutRate === null &&
        signalRatio === null
    ) {

        renderRegionalSignalUnavailable(
            "Historical regional validation data unavailable."
        );

        return;
    }


    setText(
        "regional-signal-count",
        signalRecords !== null
            ? formatNumber(
                signalRecords
            )
            : "—"
    );


    setText(
        "regional-stockout-signal",
        signalStockoutRate !== null
            ? formatPercent(
                signalStockoutRate
            )
            : "—"
    );


    setText(
        "regional-stockout-no-signal",
        noSignalStockoutRate !== null
            ? formatPercent(
                noSignalStockoutRate
            )
            : "—"
    );


    setText(
        "regional-signal-ratio",
        signalRatio !== null
            ? `${Number(signalRatio).toFixed(2)}×`
            : "—"
    );


    setText(
        "regional-pressure-title",
        "Detecting spatial pressure"
    );


    const badge =
        $("regional-pressure-badge");


    if (badge) {

        badge.textContent =
            signalRatio !== null
                ? "VALIDATED SIGNAL"
                : "HISTORICAL VALIDATION";


        badge.classList.remove(
            "inactive",
            "warning",
            "positive"
        );


        badge.classList.add(
            "positive"
        );
    }


    const description =
        $("regional-pressure-description");


    if (description) {

        if (
            signalStockoutRate !== null &&
            noSignalStockoutRate !== null &&
            signalRatio !== null
        ) {

            description.textContent =
                `Historical observations show a ${formatPercent(
                    signalStockoutRate
                )} stockout rate when a regional signal is present, compared with ${formatPercent(
                    noSignalStockoutRate
                )} when no signal is present — a ${Number(
                    signalRatio
                ).toFixed(
                    2
                )}× observed rate ratio.`;

        } else {

            description.textContent =
                "Historical regional validation data is partially available.";
        }
    }


    console.log(
        "Regional signal validation:",
        {
            signalRecords,
            signalStockoutRate,
            noSignalStockoutRate,
            signalRatio
        }
    );
}


// ------------------------------------------------------------
// Regional signal unavailable state
// ------------------------------------------------------------

function renderRegionalSignalUnavailable(
    message
) {

    setText(
        "regional-signal-count",
        "—"
    );


    setText(
        "regional-stockout-signal",
        "—"
    );


    setText(
        "regional-stockout-no-signal",
        "—"
    );


    setText(
        "regional-signal-ratio",
        "—"
    );


    setText(
        "regional-pressure-title",
        "Detecting spatial pressure"
    );


    const badge =
        $("regional-pressure-badge");


    if (badge) {

        badge.textContent =
            "UNAVAILABLE";


        badge.classList.remove(
            "positive",
            "warning"
        );


        badge.classList.add(
            "inactive"
        );
    }


    const description =
        $("regional-pressure-description");


    if (description) {

        description.textContent =
            message ||
            "Historical regional validation data unavailable.";
    }
}


// ============================================================
// RISK OVERVIEW
// ============================================================

function updateRiskOverview(summary) {

    const risk =
        summary?.risk_distribution ||
        summary?.risk ||
        summary?.risk_overview ||
        {};


    const high =
        risk.HIGH ??
        risk.high ??
        summary?.high_risk ??
        summary?.high_risk_count ??
        0;


    const moderate =
        risk.MODERATE ??
        risk.moderate ??
        summary?.moderate_risk ??
        summary?.moderate_risk_count ??
        0;


    const low =
        risk.LOW ??
        risk.low ??
        summary?.low_risk ??
        summary?.low_risk_count ??
        0;


    const veryLow =
        risk.VERY_LOW ??
        risk["VERY LOW"] ??
        risk.very_low ??
        summary?.very_low_risk ??
        summary?.very_low_risk_count ??
        0;


    updateRiskOverviewValues(
        high,
        moderate,
        low,
        veryLow
    );
}


// ------------------------------------------------------------
// Update risk overview from explicit values
// ------------------------------------------------------------

function updateRiskOverviewValues(
    high,
    moderate,
    low,
    veryLow
) {

    high =
        Number(high) || 0;

    moderate =
        Number(moderate) || 0;

    low =
        Number(low) || 0;

    veryLow =
        Number(veryLow) || 0;


    const total =
        high +
        moderate +
        low +
        veryLow;


    setText(
        "risk-high-count",
        formatNumber(
            high
        )
    );


    setText(
        "risk-high-percent",
        total > 0
            ? formatPercent(
                high / total
            )
            : "0%"
    );


    setText(
        "risk-moderate-count",
        formatNumber(
            moderate
        )
    );


    setText(
        "risk-moderate-percent",
        total > 0
            ? formatPercent(
                moderate / total
            )
            : "0%"
    );


    setText(
        "risk-low-count",
        formatNumber(
            low
        )
    );


    setText(
        "risk-low-percent",
        total > 0
            ? formatPercent(
                low / total
            )
            : "0%"
    );


    setText(
        "risk-very-low-count",
        formatNumber(
            veryLow
        )
    );


    setText(
        "risk-very-low-percent",
        total > 0
            ? formatPercent(
                veryLow / total
            )
            : "0%"
    );


    setWidth(
        "risk-high-bar",
        total > 0
            ? (high / total) * 100
            : 0
    );


    setWidth(
        "risk-moderate-bar",
        total > 0
            ? (moderate / total) * 100
            : 0
    );


    setWidth(
        "risk-low-bar",
        total > 0
            ? (low / total) * 100
            : 0
    );


    setWidth(
        "risk-very-low-bar",
        total > 0
            ? (veryLow / total) * 100
            : 0
    );


    setText(
        "high-risk-count",
        formatNumber(
            high
        )
    );
}


// ------------------------------------------------------------
// Set progress-bar width safely
// ------------------------------------------------------------

function setWidth(
    id,
    percentage
) {

    const el =
        $(id);


    if (!el) {
        return;
    }


    const safePercentage =
        Math.min(
            100,
            Math.max(
                0,
                Number(
                    percentage
                ) || 0
            )
        );


    el.style.width =
        `${safePercentage}%`;
}


// ============================================================
// BUILD RISK DISTRIBUTION FROM VISIBLE ALERTS
// ============================================================

function calculateAlertRiskDistribution(
    alerts
) {

    const distribution = {
        HIGH: 0,
        MODERATE: 0,
        LOW: 0,
        VERY_LOW: 0
    };


    alerts.forEach(
        alert => {

            const rawRisk =
                alert.risk_level ??
                alert.risk ??
                alert.risk_category ??
                alert.risk_class ??
                "";


            const normalized =
                String(rawRisk)
                    .trim()
                    .toUpperCase()
                    .replace(/[\s-]+/g, "_");


            if (
                normalized === "HIGH"
            ) {

                distribution.HIGH++;

            } else if (
                normalized === "MODERATE"
            ) {

                distribution.MODERATE++;

            } else if (
                normalized === "LOW"
            ) {

                distribution.LOW++;

            } else if (
                normalized === "VERY_LOW"
            ) {

                distribution.VERY_LOW++;
            }

        }
    );


    return distribution;
}


// ============================================================
// APPLY ALERT-BASED RISK OVERVIEW
// ============================================================

function updateRiskOverviewFromAlerts(
    alerts
) {

    const distribution =
        calculateAlertRiskDistribution(
            alerts
        );


    alertRiskDistribution =
        distribution;


    console.log(
        "Risk distribution from displayed alerts:",
        distribution
    );


    updateRiskOverviewValues(
        distribution.HIGH,
        distribution.MODERATE,
        distribution.LOW,
        distribution.VERY_LOW
    );
}


// ============================================================
// ALERTS
// ============================================================

async function loadAlerts() {

    const container =
        $("alerts-container");


    if (!container) {

        console.error(
            "Alerts container not found."
        );

        return;
    }


    container.innerHTML = `
        <div class="loading-state">
            Loading shortage alerts...
        </div>
    `;


    try {

        const data =
            await apiRequest(
                "/alerts"
            );


        console.log(
            "Alerts:",
            data
        );


        let alerts = [];


        if (
            Array.isArray(data)
        ) {

            alerts =
                data;

        } else if (
            Array.isArray(
                data.alerts
            )
        ) {

            alerts =
                data.alerts;

        } else if (
            Array.isArray(
                data.data
            )
        ) {

            alerts =
                data.data;
        }


        if (
            alerts.length === 0
        ) {

            alertRiskDistribution = {
                HIGH: 0,
                MODERATE: 0,
                LOW: 0,
                VERY_LOW: 0
            };


            updateRiskOverviewValues(
                0,
                0,
                0,
                0
            );


            container.innerHTML = `
                <div class="empty-state">

                    <div class="empty-icon">
                        ✓
                    </div>

                    <h3>
                        No shortage alerts
                    </h3>

                    <p>
                        No facilities currently meet
                        the configured alert criteria.
                    </p>

                </div>
            `;


            return;
        }


        updateRiskOverviewFromAlerts(
            alerts
        );


        const enrichedAlerts =
            await Promise.all(
                alerts.map(
                    enrichAlertWithDistrict
                )
            );


        container.innerHTML =
            enrichedAlerts
                .map(
                    renderAlertCard
                )
                .join("");


        attachInvestigateHandlers();


    } catch (error) {

        console.error(
            "Failed to load alerts:",
            error
        );


        container.innerHTML = `
            <div class="error-state">

                <h3>
                    Unable to load alerts
                </h3>

                <p>
                    ${escapeHTML(
                        error.message
                    )}
                </p>

            </div>
        `;
    }
}


// ============================================================
// ENRICH ALERT WITH DISTRICT
// ============================================================

async function enrichAlertWithDistrict(
    alert
) {

    const existingDistrict =
        alert.district ??
        alert.district_name ??
        alert.region ??
        alert.region_name ??
        alert.location?.district ??
        alert.location?.district_name;


    if (
        existingDistrict !== null &&
        existingDistrict !== undefined &&
        String(
            existingDistrict
        ).trim() !== ""
    ) {

        return {
            ...alert,
            district: existingDistrict
        };
    }


    const facilityId =
        alert.facility_id ??
        alert.id ??
        alert.facility;


    if (
        facilityId === null ||
        facilityId === undefined ||
        facilityId === "—"
    ) {

        return {
            ...alert,
            district: "—"
        };
    }


    try {

        const response =
            await apiRequest(
                `/facility/${encodeURIComponent(
                    facilityId
                )}`
            );


        const data =
            response?.data ||
            response ||
            {};


        const risk =
            Array.isArray(
                data.risk
            ) &&
            data.risk.length > 0
                ? data.risk[0]
                : {};


        const explanations =
            Array.isArray(
                data.explanations
            ) &&
            data.explanations.length > 0
                ? data.explanations[0]
                : {};


        const spatial =
            Array.isArray(
                data.spatial
            ) &&
            data.spatial.length > 0
                ? data.spatial[0]
                : {};


        const district =
            risk.district ??
            risk.district_name ??
            explanations.district ??
            explanations.district_name ??
            spatial.district ??
            spatial.district_name ??
            data.district ??
            data.district_name ??
            data.facility?.district ??
            data.facility?.district_name ??
            "—";


        return {
            ...alert,
            district
        };


    } catch (error) {

        console.warn(
            `Could not retrieve district for facility ${facilityId}:`,
            error
        );


        return {
            ...alert,
            district: "—"
        };
    }
}


// ============================================================
// FACILITY ALERT CARD
// ============================================================

function renderAlertCard(
    alert
) {

    const facilityId =
        alert.facility_id ??
        alert.id ??
        alert.facility ??
        "—";


    const rawRiskLevel =
        alert.risk_level ??
        alert.risk ??
        "UNKNOWN";


    const riskLevel =
        String(rawRiskLevel)
            .trim()
            .toUpperCase();


    const score =
        alert.final_shortage_score ??
        alert.shortage_score ??
        alert.score ??
        null;


    const mlRisk =
        alert.ml_shortage_risk ??
        alert.prediction_stockout_risk ??
        alert.ml_risk ??
        null;


    const spatialPressure =
        alert.spatial_shortage_pressure ??
        alert.local_shortage_pressure ??
        alert.spatial_pressure ??
        null;


    const evidenceCount =
        alert.evidence_count ??
        alert.signals ??
        0;


    const date =
        alert.date ??
        alert.timestamp ??
        alert.assessment_date ??
        null;


    const district =
        alert.district ??
        alert.district_name ??
        alert.region ??
        alert.region_name ??
        "—";


    const evidence =
        alert.evidence_summary ??
        alert.xai_primary_reason ??
        alert.primary_reason ??
        "Shortage risk detected.";


    const riskClass =
        String(riskLevel)
            .toLowerCase()
            .replace(/[\s_]+/g, "-");


    return `

        <article
            class="facility-alert-card ${riskClass}"
        >

            <div class="facility-alert-card__top">

                <div class="facility-alert-card__identity">

                    <span
                        class="facility-alert-card__severity ${riskClass}"
                    >
                        ${escapeHTML(
                            riskLevel
                        )}
                    </span>

                    <h3>
                        Facility
                        ${escapeHTML(
                            facilityId
                        )}
                    </h3>

                    <p>
                        ${escapeHTML(
                            district
                        )}
                    </p>

                </div>


                <div class="facility-alert-card__score">

                    <span>
                        Risk Score
                    </span>

                    <strong>
                        ${
                            score !== null
                                ? formatScore(
                                    score
                                )
                                : "—"
                        }
                    </strong>

                </div>

            </div>


            <div class="facility-alert-card__metrics">

                <div
                    class="facility-alert-card__metric"
                >

                    <span>
                        ML Risk
                    </span>

                    <strong>
                        ${formatPercent(
                            mlRisk
                        )}
                    </strong>

                </div>


                <div
                    class="facility-alert-card__metric"
                >

                    <span>
                        Spatial Pressure
                    </span>

                    <strong>
                        ${formatPercent(
                            spatialPressure
                        )}
                    </strong>

                </div>


                <div
                    class="facility-alert-card__metric"
                >

                    <span>
                        Evidence
                    </span>

                    <strong>
                        ${formatNumber(
                            evidenceCount
                        )}
                    </strong>

                </div>

            </div>


            <div class="facility-alert-card__evidence">

                <span>
                    PRIMARY EVIDENCE
                </span>

                <p>
                    ${escapeHTML(
                        evidence
                    )}
                </p>

            </div>


            <div class="facility-alert-card__footer">

                <span>
                    ${formatDate(
                        date
                    )}
                </span>


                <button
                    class="investigate-btn"
                    data-facility-id="${escapeHTML(
                        facilityId
                    )}"
                    type="button"
                >
                    Investigate
                    <span>→</span>
                </button>

            </div>

        </article>
    `;
}


// ============================================================
// INVESTIGATE HANDLERS
// ============================================================

function attachInvestigateHandlers() {

    const buttons =
        document.querySelectorAll(
            ".investigate-btn"
        );


    buttons.forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    const facilityId =
                        button.dataset.facilityId;


                    if (
                        !facilityId ||
                        facilityId === "—"
                    ) {

                        console.error(
                            "Invalid facility ID."
                        );

                        return;
                    }


                    navigateToFacility(
                        facilityId
                    );
                }
            );
        }
    );
}


// ============================================================
// PRIORITY ALERT INVESTIGATION
// ============================================================
//
// The priority alert is currently hardcoded in index.html and
// its Investigate link uses:
//
//     href="#"
//
// Therefore the normal .investigate-btn handler does not catch
// it.
//
// This handler specifically connects the priority alert to the
// facility investigation page.
//
// It first looks for a data-facility-id attribute. If one is not
// present, it extracts the facility ID from the visible title:
//
//     Facility 10166
//
// This means the current priority alert works without requiring
// an HTML rewrite.
// ============================================================

function setupPriorityAlertHandler() {

    const priorityCard =
        $("priority-alert-card");


    if (!priorityCard) {

        console.log(
            "Priority alert card not found."
        );

        return;
    }


    const priorityLink =
        priorityCard.querySelector(
            'a[href="#"], .priority-investigate, .investigate-btn'
        );


    if (!priorityLink) {

        console.log(
            "Priority alert Investigate link not found."
        );

        return;
    }


    priorityLink.addEventListener(
        "click",
        event => {

            event.preventDefault();


            // ------------------------------------------------
            // First choice:
            // Explicit data attribute.
            // ------------------------------------------------

            let facilityId =
                priorityLink.dataset.facilityId ||
                priorityCard.dataset.facilityId ||
                null;


            // ------------------------------------------------
            // Second choice:
            // Extract ID from priority alert title.
            //
            // Example:
            // "Facility 10166"
            // ------------------------------------------------

            if (!facilityId) {

                const titleElement =
                    $("priority-alert-title");


                const title =
                    titleElement?.textContent ||
                    priorityCard.textContent ||
                    "";


                const match =
                    title.match(
                        /facility\s+([A-Za-z0-9_-]+)/i
                    );


                if (match) {

                    facilityId =
                        match[1];
                }
            }


            // ------------------------------------------------
            // Final fallback:
            // Current priority alert is Facility 10166.
            //
            // This is only used if the card does not expose
            // the facility ID in either HTML or title text.
            // ------------------------------------------------

            if (!facilityId) {

                facilityId =
                    "10166";
            }


            if (
                !facilityId ||
                facilityId === "—"
            ) {

                console.error(
                    "Priority alert does not contain a valid facility ID."
                );

                return;
            }


            console.log(
                `Opening priority facility investigation: ${facilityId}`
            );


            navigateToFacility(
                facilityId
            );
        }
    );
}


// ============================================================
// CENTRAL FACILITY NAVIGATION
// ============================================================
//
// All Investigate buttons now use this function so navigation
// behaves consistently.
// ============================================================

function navigateToFacility(
    facilityId
) {

    if (
        facilityId === null ||
        facilityId === undefined ||
        String(facilityId).trim() === "" ||
        String(facilityId) === "—"
    ) {

        console.error(
            "Cannot navigate to facility: invalid facility ID.",
            facilityId
        );

        return;
    }


    const encodedFacilityId =
        encodeURIComponent(
            String(facilityId)
        );


    window.location.href =
        `facility.html?id=${encodedFacilityId}`;
}


// ============================================================
// LEGACY FACILITY PANEL FUNCTIONS
// ============================================================

async function investigateFacility(
    facilityId
) {

    currentFacilityId =
        facilityId;


    console.log(
        `Investigating facility ${facilityId}`
    );


    const panel =
        $("facility-details-panel");


    const details =
        $("facility-details");


    if (
        !panel ||
        !details
    ) {

        console.log(
            "Dedicated facility page is being used."
        );


        navigateToFacility(
            facilityId
        );


        return;
    }


    panel.style.display =
        "block";


    details.innerHTML = `
        <div class="loading-state">

            <div class="loading-spinner"></div>

            Loading facility investigation...

        </div>
    `;


    panel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });


    try {

        const data =
            await apiRequest(
                `/facility/${encodeURIComponent(
                    facilityId
                )}`
            );


        renderFacilityDetails(
            data
        );


    } catch (error) {

        console.error(
            `Failed to investigate facility ${facilityId}:`,
            error
        );


        details.innerHTML = `
            <div class="error-state">

                <h3>
                    Investigation failed
                </h3>

                <p>
                    ${escapeHTML(
                        error.message
                    )}
                </p>

            </div>
        `;
    }
}


// ============================================================
// FACILITY DETAILS
// ============================================================

function renderFacilityDetails(
    response
) {

    const container =
        $("facility-details");


    if (!container) {

        console.error(
            "Facility details container not found."
        );

        return;
    }


    const data =
        response?.data ||
        response ||
        {};


    const risk =
        Array.isArray(
            data.risk
        ) &&
        data.risk.length > 0
            ? data.risk[0]
            : {};


    const explanation =
        Array.isArray(
            data.explanations
        ) &&
        data.explanations.length > 0
            ? data.explanations[0]
            : {};


    const spatialRows =
        Array.isArray(
            data.spatial
        )
            ? data.spatial
            : [];


    const redistribution =
        Array.isArray(
            data.redistribution
        )
            ? data.redistribution
            : [];


    const facility = {
        ...risk,
        ...explanation
    };


    const facilityId =
        facility.facility_id ??
        currentFacilityId;


    const riskLevel =
        facility.risk_level ??
        "UNKNOWN";


    const riskClass =
        String(riskLevel)
            .toLowerCase()
            .replace(/\s+/g, "-");


    const district =
        facility.district ??
        facility.district_name ??
        spatialRows[0]?.district ??
        spatialRows[0]?.district_name ??
        "—";


    const finalScore =
        facility.final_shortage_score;


    const mlRisk =
        facility.ml_shortage_risk ??
        facility.prediction_stockout_risk;


    const spatialPressure =
        facility.spatial_shortage_pressure;


    const evidenceCount =
        facility.evidence_count;


    const currentStockout =
        Number(
            facility.current_stockout
        ) === 1;


    const regionalEvent =
        Number(
            facility.regional_event
        ) === 1;


    const anomalyFlag =
        Number(
            facility.anomaly_flag
        ) === 1;


    const anomalyEvidence =
        facility.anomaly_evidence;


    const consumption =
        facility.consumption;


    const inventory =
        facility.closing_inventory;


    const spatialRow =
        spatialRows[0] || {};


    const nearbyFacilities =
        spatialRow.nearby_facilities;


    const nearbyStockouts =
        spatialRow.nearby_stockouts;


    const nearbyStockoutRate =
        spatialRow.nearby_stockout_rate;


    const localPressure =
        spatialRow.local_shortage_pressure;


    const overallStockoutRate =
        spatialRow.overall_stockout_rate;


    const primaryReason =
        facility.xai_primary_reason ??
        "Shortage evidence detected.";


    const modelEvidence =
        facility.xai_model_evidence ??
        "";


    const observedEvidence =
        facility.xai_observed_evidence ??
        "";


    const xaiExplanation =
        facility.xai_explanation ??
        facility.evidence_summary ??
        "";


    container.innerHTML = `

        <div class="facility-header">

            <div>

                <div class="facility-eyebrow">
                    FACILITY INVESTIGATION
                </div>

                <h2>
                    Facility
                    ${escapeHTML(
                        facilityId
                    )}
                </h2>

                <p>
                    ${escapeHTML(
                        district
                    )}
                </p>

            </div>


            <div class="facility-risk-display">

                <span class="risk-badge ${riskClass}">
                    ${escapeHTML(
                        riskLevel
                    )}
                </span>

                <div class="facility-score">
                    ${
                        finalScore !== undefined
                            ? formatScore(
                                finalScore
                            )
                            : "—"
                    }
                </div>

                <span>
                    fused shortage score
                </span>

            </div>

        </div>


        <div class="investigation-grid">

            <div class="investigation-card primary">

                <div class="investigation-card-title">
                    <span>
                        Prediction
                    </span>
                </div>

                <div class="metric-large">
                    ${formatPercent(
                        mlRisk
                    )}
                </div>

                <div class="metric-label">
                    ML-predicted shortage risk
                </div>

                <div class="metric-bar">

                    <div
                        class="metric-bar-fill"
                        style="
                            width:${Math.min(
                                100,
                                Math.max(
                                    0,
                                    Number(
                                        mlRisk || 0
                                    ) * 100
                                )
                            )}%
                        "
                    ></div>

                </div>

            </div>


            <div class="investigation-card">

                <div class="investigation-card-title">
                    <span>
                        Spatial pressure
                    </span>
                </div>

                <div class="metric-large">
                    ${formatPercent(
                        spatialPressure
                    )}
                </div>

                <div class="metric-label">
                    Nearby shortage pressure
                </div>

            </div>


            <div class="investigation-card">

                <div class="investigation-card-title">
                    <span>
                        Evidence
                    </span>
                </div>

                <div class="metric-large">
                    ${formatNumber(
                        evidenceCount
                    )}
                </div>

                <div class="metric-label">
                    shortage signals detected
                </div>

            </div>


            <div class="investigation-card">

                <div class="investigation-card-title">
                    <span>
                        Anomaly
                    </span>
                </div>

                <div class="metric-large">
                    ${
                        anomalyFlag
                            ? "YES"
                            : "NO"
                    }
                </div>

                <div class="metric-label">

                    ${
                        anomalyFlag
                            ? `Evidence ${formatScore(
                                anomalyEvidence,
                                3
                            )}`
                            : "No anomaly detected"
                    }

                </div>

            </div>

        </div>


        <div class="section-divider"></div>


        <section class="investigation-section">

            <div class="section-heading">

                <div>

                    <span class="section-kicker">
                        WHY THIS FACILITY IS FLAGGED
                    </span>

                    <h3>
                        Evidence breakdown
                    </h3>

                </div>

            </div>


            <div class="evidence-list">

                ${renderEvidenceItem(
                    "ML shortage prediction",
                    mlRisk !== undefined
                        ? `ML-predicted shortage risk is ${formatPercent(
                            mlRisk
                        )}.`
                        : "ML shortage risk detected.",
                    Number(
                        mlRisk || 0
                    ) > 0.5
                )}


                ${renderEvidenceItem(
                    "Current stock status",
                    currentStockout
                        ? "Current stockout is observed in the dataset."
                        : "No current stockout is observed.",
                    currentStockout
                )}


                ${renderEvidenceItem(
                    "Nearby facilities",
                    spatialPressure !== undefined
                        ? `Nearby shortage pressure is ${formatPercent(
                            spatialPressure
                        )}.`
                        : "Nearby shortage pressure detected.",
                    Number(
                        spatialPressure || 0
                    ) > 0.5
                )}


                ${renderEvidenceItem(
                    "Inventory behaviour",
                    anomalyFlag
                        ? `Unusual consumption/inventory behaviour was detected (anomaly evidence ${formatScore(
                            anomalyEvidence,
                            3
                        )}).`
                        : "No unusual inventory behaviour was detected.",
                    anomalyFlag
                )}


                ${renderEvidenceItem(
                    "Regional signal",
                    regionalEvent
                        ? "The facility belongs to a period with a detected regional shortage event."
                        : "No regional shortage event is currently associated.",
                    regionalEvent
                )}

            </div>

        </section>


        <section class="investigation-section">

            <div class="section-heading">

                <div>

                    <span class="section-kicker">
                        LOCAL CONDITIONS
                    </span>

                    <h3>
                        Facility and neighbourhood data
                    </h3>

                </div>

            </div>


            <div class="detail-grid">

                ${detailMetric(
                    "Current inventory",
                    inventory !== undefined
                        ? formatNumber(
                            inventory,
                            1
                        )
                        : "—"
                )}


                ${detailMetric(
                    "Consumption",
                    consumption !== undefined
                        ? formatNumber(
                            consumption,
                            0
                        )
                        : "—"
                )}


                ${detailMetric(
                    "Nearby facilities",
                    nearbyFacilities !== undefined
                        ? formatNumber(
                            nearbyFacilities
                        )
                        : "—"
                )}


                ${detailMetric(
                    "Nearby stockouts",
                    nearbyStockouts !== undefined
                        ? formatNumber(
                            nearbyStockouts
                        )
                        : "—"
                )}


                ${detailMetric(
                    "Nearby stockout rate",
                    formatPercent(
                        nearbyStockoutRate
                    )
                )}


                ${detailMetric(
                    "Local pressure",
                    formatPercent(
                        localPressure
                    )
                )}


                ${detailMetric(
                    "Overall stockout rate",
                    formatPercent(
                        overallStockoutRate
                    )
                )}


                ${detailMetric(
                    "Regional event",
                    regionalEvent
                        ? "Detected"
                        : "Not detected"
                )}

            </div>

        </section>


        <section class="investigation-section xai-section">

            <div class="section-heading">

                <div>

                    <span class="section-kicker">
                        EXPLAINABLE AI
                    </span>

                    <h3>
                        Model reasoning
                    </h3>

                </div>

            </div>


            <div class="xai-primary">

                <span class="xai-label">
                    PRIMARY REASON
                </span>

                <p>
                    ${escapeHTML(
                        primaryReason
                    )}
                </p>

            </div>


            ${
                modelEvidence
                    ? `
                        <div class="xai-block">

                            <span>
                                MODEL EVIDENCE
                            </span>

                            <p>
                                ${escapeHTML(
                                    modelEvidence
                                )}
                            </p>

                        </div>
                    `
                    : ""
            }


            ${
                observedEvidence
                    ? `
                        <div class="xai-block">

                            <span>
                                OBSERVED EVIDENCE
                            </span>

                            <p>
                                ${escapeHTML(
                                    observedEvidence
                                )}
                            </p>

                        </div>
                    `
                    : ""
            }


            ${
                xaiExplanation
                    ? `
                        <div class="xai-explanation">

                            <span>
                                SUMMARY
                            </span>

                            <p>
                                ${escapeHTML(
                                    xaiExplanation
                                )}
                            </p>

                        </div>
                    `
                    : ""
            }

        </section>


        <section class="investigation-section">

            <div class="section-heading">

                <div>

                    <span class="section-kicker">
                        RESPONSE OPTIONS
                    </span>

                    <h3>
                        Redistribution opportunities
                    </h3>

                </div>

            </div>

            ${renderRedistribution(
                redistribution,
                facility
            )}

        </section>


        ${
            spatialRows.length > 0
                ? `
                    <section class="investigation-section">

                        <div class="section-heading">

                            <div>

                                <span class="section-kicker">
                                    SPATIAL SIGNAL
                                </span>

                                <h3>
                                    Recent spatial observations
                                </h3>

                            </div>

                        </div>

                        ${renderSpatialTable(
                            spatialRows
                        )}

                    </section>
                `
                : ""
        }

    `;
}


// ============================================================
// EVIDENCE ITEM
// ============================================================

function renderEvidenceItem(
    title,
    text,
    active
) {

    return `
        <div class="evidence-item ${
            active
                ? "active"
                : "inactive"
        }">

            <div class="evidence-icon">
                ${active ? "✓" : "–"}
            </div>

            <div>

                <strong>
                    ${escapeHTML(
                        title
                    )}
                </strong>

                <p>
                    ${escapeHTML(
                        text
                    )}
                </p>

            </div>

        </div>
    `;
}


// ============================================================
// DETAIL METRIC
// ============================================================

function detailMetric(
    label,
    value
) {

    return `
        <div class="detail-metric">

            <span>
                ${escapeHTML(
                    label
                )}
            </span>

            <strong>
                ${escapeHTML(
                    value
                )}
            </strong>

        </div>
    `;
}


// ============================================================
// REDISTRIBUTION
// ============================================================

function renderRedistribution(
    options,
    facility
) {

    if (
        !options ||
        options.length === 0
    ) {

        return `
            <div class="empty-state compact">

                <div class="empty-icon">
                    —
                </div>

                <h4>
                    No redistribution option available
                </h4>

                <p>
                    No feasible donor is available
                    in the current simulated donor network.
                </p>

            </div>
        `;
    }


    const bestDonor =
        facility.best_donor_facility_id;


    const bestDistance =
        facility.best_donor_distance_km;


    const bestSurplus =
        facility.best_donor_surplus;


    const bestFeasibility =
        facility.best_feasibility_score;


    const bestTransit =
        facility.best_transit_probability;


    return `

        <div class="redistribution-summary">

            <div class="redistribution-highlight">

                <span>
                    BEST SIMULATED DONOR
                </span>

                <strong>
                    ${
                        bestDonor !== undefined
                            ? `Facility ${escapeHTML(
                                bestDonor
                            )}`
                            : "No donor"
                    }
                </strong>

            </div>


            <div class="detail-grid compact-grid">

                ${detailMetric(
                    "Distance",
                    bestDistance !== undefined
                        ? `${formatNumber(
                            bestDistance,
                            1
                        )} km`
                        : "—"
                )}


                ${detailMetric(
                    "Available surplus",
                    bestSurplus !== undefined
                        ? formatNumber(
                            bestSurplus,
                            1
                        )
                        : "—"
                )}


                ${detailMetric(
                    "Feasibility",
                    formatPercent(
                        bestFeasibility
                    )
                )}


                ${detailMetric(
                    "Transit probability",
                    formatPercent(
                        bestTransit
                    )
                )}

            </div>

        </div>


        <div class="donor-list">

            ${
                options
                    .slice(0, 8)
                    .map(
                        renderDonorOption
                    )
                    .join("")
            }

        </div>


        <div class="simulation-note">

            <strong>
                Prototype note:
            </strong>

            Redistribution results are based on the
            current simulated donor network and
            feasibility assumptions. They are not
            observed supplier or transport records.

        </div>
    `;
}


function renderDonorOption(
    option
) {

    const donorId =
        option.donor_facility_id ??
        option.facility_id ??
        option.id ??
        "—";


    const distance =
        option.distance_km ??
        option.best_donor_distance_km;


    const surplus =
        option.surplus ??
        option.donor_surplus ??
        option.best_donor_surplus;


    const feasibility =
        option.feasibility_score ??
        option.best_feasibility_score;


    const transit =
        option.transit_probability ??
        option.best_transit_probability;


    const category =
        option.feasibility_category ??
        option.category ??
        "—";


    return `

        <div class="donor-option">

            <div class="donor-main">

                <span class="donor-id">
                    Facility
                    ${escapeHTML(
                        donorId
                    )}
                </span>

                <span class="donor-category">
                    ${escapeHTML(
                        category
                    )}
                </span>

            </div>


            <div class="donor-metrics">

                <span>

                    <small>
                        Distance
                    </small>

                    ${formatNumber(
                        distance,
                        1
                    )} km

                </span>


                <span>

                    <small>
                        Surplus
                    </small>

                    ${formatNumber(
                        surplus,
                        1
                    )}

                </span>


                <span>

                    <small>
                        Feasibility
                    </small>

                    ${formatPercent(
                        feasibility
                    )}

                </span>


                <span>

                    <small>
                        Transit
                    </small>

                    ${formatPercent(
                        transit
                    )}

                </span>

            </div>

        </div>
    `;
}


// ============================================================
// SPATIAL TABLE
// ============================================================

function renderSpatialTable(
    rows
) {

    const limitedRows =
        rows.slice(0, 10);


    return `

        <div class="table-wrapper">

            <table class="data-table">

                <thead>

                    <tr>

                        <th>
                            Date
                        </th>

                        <th>
                            Nearby Facilities
                        </th>

                        <th>
                            Nearby Stockouts
                        </th>

                        <th>
                            Stockout Rate
                        </th>

                        <th>
                            Local Pressure
                        </th>

                        <th>
                            Regional Event
                        </th>

                    </tr>

                </thead>


                <tbody>

                    ${
                        limitedRows
                            .map(
                                row => `

                                    <tr>

                                        <td>
                                            ${formatDate(
                                                row.date
                                            )}
                                        </td>

                                        <td>
                                            ${formatNumber(
                                                row.nearby_facilities
                                            )}
                                        </td>

                                        <td>
                                            ${formatNumber(
                                                row.nearby_stockouts
                                            )}
                                        </td>

                                        <td>
                                            ${formatPercent(
                                                row.nearby_stockout_rate
                                            )}
                                        </td>

                                        <td>
                                            ${formatPercent(
                                                row.local_shortage_pressure
                                            )}
                                        </td>

                                        <td>

                                            ${
                                                Number(
                                                    row.regional_event
                                                ) === 1

                                                    ? `
                                                        <span class="status-positive">
                                                            Detected
                                                        </span>
                                                    `

                                                    : `
                                                        <span class="status-neutral">
                                                            No
                                                        </span>
                                                    `
                                            }

                                        </td>

                                    </tr>

                                `
                            )
                            .join("")
                    }

                </tbody>

            </table>

        </div>
    `;
}


// ============================================================
// CLOSE FACILITY DETAILS
// ============================================================

function closeFacilityDetails() {

    const panel =
        $("facility-details-panel");


    if (!panel) {
        return;
    }


    panel.style.display =
        "none";


    currentFacilityId =
        null;
}


// ============================================================
// CASCADE SIMULATION
// ============================================================

async function loadCascadeSimulation() {

    try {

        const data =
            await apiRequest(
                "/cascade"
            );


        console.log(
            "Cascade API response:",
            data
        );


        renderCascade(
            data
        );


        return data;


    } catch (error) {

        console.error(
            "Failed to load cascade simulation:",
            error
        );


        renderCascadeError(
            error
        );


        return null;
    }
}


function renderCascade(
    data
) {

    const rows =
        Array.isArray(
            data?.cascade
        )
            ? data.cascade
            : [];


    if (
        rows.length === 0
    ) {

        renderCascadeError(
            new Error(
                "No cascade records returned by backend."
            )
        );


        return;
    }


    const origin =
        rows.find(
            row =>
                row.origin_facility_id !== null &&
                row.origin_facility_id !== undefined
        )?.origin_facility_id ??
        "—";


    const facilitiesSet =
        new Set(
            rows
                .map(
                    row =>
                        row.facility_id
                )
                .filter(
                    value =>
                        value !== null &&
                        value !== undefined
                )
        );


    const facilities =
        facilitiesSet.size;


    const records =
        Number(
            data.count
        ) ||
        rows.length;


    const maxStep =
        Math.max(
            ...rows.map(
                row =>
                    Number(
                        row.cascade_step
                    ) || 0
            )
        );


    const propagationPressures =
        rows
            .map(
                row =>
                    Number(
                        row.propagation_pressure
                    )
            )
            .filter(
                value =>
                    Number.isFinite(
                        value
                    )
            );


    const averagePressure =
        propagationPressures.length > 0
            ? propagationPressures.reduce(
                (
                    sum,
                    value
                ) =>
                    sum + value,
                0
            ) /
            propagationPressures.length
            : null;


    const maximumPressure =
        propagationPressures.length > 0
            ? Math.max(
                ...propagationPressures
            )
            : null;


    setSystemInfoValue(
        "cascade-origin",
        origin
    );


    setSystemInfoValue(
        "cascade-facilities",
        formatNumber(
            facilities
        )
    );


    setSystemInfoValue(
        "cascade-records",
        formatNumber(
            records
        )
    );


    setSystemInfoValue(
        "cascade-max-step",
        formatNumber(
            maxStep
        )
    );


    setSystemInfoValue(
        "cascade-avg-pressure",
        formatPercent(
            averagePressure
        )
    );


    setSystemInfoValue(
        "cascade-max-pressure",
        formatPercent(
            maximumPressure
        )
    );


    const cascadeContainer =
        $("cascade-container");


    if (
        cascadeContainer
    ) {

        cascadeContainer.innerHTML =
            rows
                .slice(0, 20)
                .map(
                    renderCascadeRow
                )
                .join("");
    }
}


function renderCascadeRow(
    row
) {

    const pressure =
        row.propagation_pressure ??
        row.new_pressure ??
        null;


    return `

        <div class="cascade-row">

            <div>

                <strong>
                    Facility
                    ${escapeHTML(
                        row.facility_id ??
                        "—"
                    )}
                </strong>

                <span>
                    Step
                    ${escapeHTML(
                        row.cascade_step ??
                        "—"
                    )}
                </span>

            </div>


            <div>

                ${formatPercent(
                    pressure
                )}

            </div>

        </div>
    `;
}


function renderCascadeError(
    error
) {

    setSystemInfoValue(
        "cascade-origin",
        "Unavailable"
    );


    setSystemInfoValue(
        "cascade-facilities",
        "—"
    );


    setSystemInfoValue(
        "cascade-records",
        "—"
    );


    setSystemInfoValue(
        "cascade-max-step",
        "—"
    );


    setSystemInfoValue(
        "cascade-avg-pressure",
        "—"
    );


    setSystemInfoValue(
        "cascade-max-pressure",
        "—"
    );


    const cascadeContainer =
        $("cascade-container");


    if (
        cascadeContainer
    ) {

        cascadeContainer.innerHTML = `

            <div class="error-state">

                ${escapeHTML(
                    error.message
                )}

            </div>
        `;
    }
}


// ============================================================
// INTERVENTION SIMULATION
// ============================================================

async function loadInterventionSimulation() {

    try {

        const data =
            await apiRequest(
                "/intervention"
            );


        console.log(
            "Intervention API response:",
            data
        );


        renderIntervention(
            data
        );


        return data;


    } catch (error) {

        console.error(
            "Failed to load intervention simulation:",
            error
        );


        renderInterventionError(
            error
        );


        return null;
    }
}


function renderIntervention(
    data
) {

    const intervention =
        Array.isArray(
            data?.interventions
        ) &&
        data.interventions.length > 0
            ? data.interventions[0]
            : null;


    if (!intervention) {

        renderInterventionError(
            new Error(
                "No intervention record returned by backend."
            )
        );


        return;
    }


    const recipient =
        intervention.recipient_facility_id ??
        "—";


    const donor =
        intervention.donor_facility_id ??
        "—";


    const distance =
        intervention.distance_km;


    const before =
        intervention.recipient_pressure_before;


    const after =
        intervention.recipient_pressure_after;


    const reduction =
        intervention.pressure_reduction;


    const transit =
        intervention.transit_feasibility_probability;


    const donorSafety =
        Number(
            intervention.donor_has_safety_surplus
        ) === 1;


    const status =
        intervention.intervention_status ??
        "SIMULATED";


    setSystemInfoValue(
        "intervention-recipient",
        recipient
    );


    setSystemInfoValue(
        "intervention-donor",
        donor
    );


    setSystemInfoValue(
        "intervention-distance",
        `${formatNumber(
            distance,
            1
        )} km`
    );


    setSystemInfoValue(
        "intervention-before",
        formatPercent(
            before
        )
    );


    setSystemInfoValue(
        "intervention-after",
        formatPercent(
            after
        )
    );


    setSystemInfoValue(
        "intervention-reduction",
        formatPercent(
            reduction
        )
    );


    setSystemInfoValue(
        "intervention-transit",
        formatPercent(
            transit
        )
    );


    setSystemInfoValue(
        "intervention-safety",
        donorSafety
            ? "YES"
            : "NO"
    );


    setSystemInfoValue(
        "intervention-status",
        status
    );


    const interventionContainer =
        $("intervention-container");


    if (
        interventionContainer
    ) {

        interventionContainer.innerHTML = `

            <div class="intervention-result">

                <div>

                    <span>
                        Recipient
                    </span>

                    <strong>
                        ${escapeHTML(
                            recipient
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Donor
                    </span>

                    <strong>
                        ${escapeHTML(
                            donor
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Distance
                    </span>

                    <strong>
                        ${formatNumber(
                            distance,
                            1
                        )} km
                    </strong>

                </div>


                <div>

                    <span>
                        Pressure reduction
                    </span>

                    <strong>
                        ${formatPercent(
                            reduction
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Transit probability
                    </span>

                    <strong>
                        ${formatPercent(
                            transit
                        )}
                    </strong>

                </div>


                <div>

                    <span>
                        Donor safety
                    </span>

                    <strong>
                        ${
                            donorSafety
                                ? "YES"
                                : "NO"
                        }
                    </strong>

                </div>


                <div>

                    <span>
                        Status
                    </span>

                    <strong>
                        ${escapeHTML(
                            status
                        )}
                    </strong>

                </div>

            </div>
        `;
    }
}


// ============================================================
// INTERVENTION ERROR
// ============================================================

function renderInterventionError(
    error
) {

    const fields = [

        "intervention-recipient",

        "intervention-donor",

        "intervention-distance",

        "intervention-before",

        "intervention-after",

        "intervention-reduction",

        "intervention-transit",

        "intervention-safety",

        "intervention-status"

    ];


    fields.forEach(
        id => {

            setSystemInfoValue(
                id,
                "—"
            );
        }
    );


    const container =
        $("intervention-container");


    if (
        container
    ) {

        container.innerHTML = `

            <div class="error-state">

                ${escapeHTML(
                    error.message
                )}

            </div>
        `;
    }
}


// ============================================================
// SYSTEM INFORMATION
// ============================================================

function findSystemInfoContainer() {

    return $("system-info");
}


function setSystemInfoValue(
    id,
    value
) {

    const el =
        $(id);


    if (el) {

        el.textContent =
            value;
    }
}


function renderInitialSystemInfo() {

    const container =
        findSystemInfoContainer();


    if (!container) {

        console.warn(
            "System information container not found."
        );

        return;
    }


    const hasKnownFields =
        $("cascade-origin") ||
        $("intervention-recipient");


    if (hasKnownFields) {
        return;
    }


    container.innerHTML = `

        <div class="system-info-grid">

            <div class="system-info-group">

                <span class="system-info-heading">
                    CASCADE SIMULATION
                </span>


                ${systemInfoItem(
                    "cascade-origin",
                    "Cascade origin",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "cascade-facilities",
                    "Facilities",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "cascade-records",
                    "Records",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "cascade-max-step",
                    "Maximum step",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "cascade-avg-pressure",
                    "Average pressure",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "cascade-max-pressure",
                    "Maximum pressure",
                    "Loading..."
                )}

            </div>


            <div class="system-info-group">

                <span class="system-info-heading">
                    INTERVENTION SIMULATION
                </span>


                ${systemInfoItem(
                    "intervention-recipient",
                    "Recipient",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-donor",
                    "Donor",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-distance",
                    "Distance",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-before",
                    "Pressure before",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-after",
                    "Pressure after",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-reduction",
                    "Pressure reduction",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-transit",
                    "Transit probability",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-safety",
                    "Donor safety",
                    "Loading..."
                )}


                ${systemInfoItem(
                    "intervention-status",
                    "Status",
                    "Loading..."
                )}

            </div>

        </div>
    `;
}


function systemInfoItem(
    id,
    label,
    value
) {

    return `

        <div class="system-info-item">

            <span>
                ${escapeHTML(
                    label
                )}
            </span>

            <strong id="${escapeHTML(
                id
            )}">
                ${escapeHTML(
                    value
                )}
            </strong>

        </div>
    `;
}


// ============================================================
// CLOSE BUTTON
// ============================================================

function setupCloseButton() {

    const closeButton =
        $("close-details");


    if (!closeButton) {
        return;
    }


    closeButton.addEventListener(
        "click",
        closeFacilityDetails
    );
}


// ============================================================
// NAVIGATION
// ============================================================

function setupNavigation() {

    const links =
        document.querySelectorAll(
            "[data-scroll-to]"
        );


    links.forEach(
        link => {

            link.addEventListener(
                "click",
                event => {

                    event.preventDefault();


                    const targetId =
                        link.dataset.scrollTo;


                    const target =
                        $(targetId);


                    if (target) {

                        target.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }

                }
            );
        }
    );
}


// ============================================================
// KEYBOARD SUPPORT
// ============================================================

function setupKeyboardShortcuts() {

    document.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Escape"
            ) {

                closeFacilityDetails();

            }

        }
    );
}


// ============================================================
// REFRESH DASHBOARD
// ============================================================

async function refreshDashboard() {

    await Promise.all([

        checkBackendStatus(),

        loadSummary(),

        loadAlerts()

    ]);
}


// ============================================================
// APPLICATION INITIALIZATION
// ============================================================

async function initializeApp() {

    console.log(
        "SupplyShield AI frontend starting..."
    );


    renderInitialSystemInfo();


    setupCloseButton();

    setupNavigation();

    setupKeyboardShortcuts();

    // IMPORTANT:
    // Connect the priority alert Investigate button.
    setupPriorityAlertHandler();


    await refreshDashboard();


    await Promise.all([

        loadCascadeSimulation(),

        loadInterventionSimulation()

    ]);


    console.log(
        "SupplyShield AI frontend ready."
    );
}


// ============================================================
// START APPLICATION
// ============================================================

if (
    document.readyState === "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeApp
    );

} else {

    initializeApp();

}