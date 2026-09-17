// ============================================================
// SupplyShield AI - Facility Investigation Page
// facility.js
// ============================================================

const API_BASE = "http://127.0.0.1:5000/api";


// ============================================================
// DOM HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {

    const element = $(id);

    if (element) {
        element.textContent = value;
    }
}


function escapeHTML(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// ============================================================
// FORMATTING
// ============================================================

function formatNumber(
    value,
    decimals = 0
) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "—";
    }

    return Number(value).toLocaleString(
        "en-IN",
        {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }
    );
}


function formatPercent(
    value,
    decimals = 1
) {

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


function formatScore(
    value,
    decimals = 3
) {

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

    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return value;
    }

    return date.toLocaleDateString(
        "en-IN",
        {
            year: "numeric",
            month: "short",
            day: "numeric"
        }
    );
}


// ============================================================
// API
// ============================================================

async function apiRequest(
    endpoint
) {

    const response =
        await fetch(
            `${API_BASE}${endpoint}`
        );


    if (!response.ok) {

        let message =
            `HTTP ${response.status}`;

        try {

            const error =
                await response.json();

            if (error.message) {
                message =
                    error.message;
            }

            else if (error.error) {
                message =
                    error.error;
            }

        }

        catch (_) {
            // Keep HTTP error.
        }

        throw new Error(
            message
        );
    }


    return response.json();
}


// ============================================================
// URL PARAMETER
// ============================================================

function getFacilityId() {

    const params =
        new URLSearchParams(
            window.location.search
        );


    /*
     * Primary format:
     *
     * facility.html?id=10166
     */

    let facilityId =
        params.get("id");


    /*
     * Also support:
     *
     * facility.html?facility_id=10166
     *
     * This makes the page tolerant of either
     * URL format.
     */

    if (!facilityId) {

        facilityId =
            params.get(
                "facility_id"
            );
    }


    return facilityId;
}


// ============================================================
// BACKEND STATUS
// ============================================================

async function checkBackendStatus() {

    const status =
        $("backend-status");


    if (!status) {
        return;
    }


    try {

        await apiRequest(
            "/health"
        );


        status.textContent =
            "Backend Online";


        status.classList.remove(
            "offline"
        );

        status.classList.add(
            "online"
        );

    }

    catch (error) {

        console.error(
            "Backend health check failed:",
            error
        );


        status.textContent =
            "Backend Offline";


        status.classList.remove(
            "online"
        );

        status.classList.add(
            "offline"
        );
    }
}


// ============================================================
// LOAD FACILITY
// ============================================================

async function loadFacility() {

    const facilityId =
        getFacilityId();


    console.log(
        "Facility ID:",
        facilityId
    );


    // --------------------------------------------------------
    // No facility ID
    // --------------------------------------------------------

    if (
        !facilityId ||
        facilityId === "—"
    ) {

        showError(
            "No facility ID was provided in the URL."
        );

        return;
    }


    try {

        const response =
            await apiRequest(
                `/facility/${encodeURIComponent(
                    facilityId
                )}`
            );


        console.log(
            "Facility API response:",
            response
        );


        renderFacility(
            response,
            facilityId
        );

    }

    catch (error) {

        console.error(
            "Facility loading failed:",
            error
        );


        showError(
            error.message
        );
    }
}


// ============================================================
// RENDER FACILITY
// ============================================================

function renderFacility(
    response,
    facilityId
) {

    const loading =
        $("facility-loading");

    const content =
        $("facility-content");

    const error =
        $("facility-error");


    if (loading) {
        loading.style.display =
            "none";
    }


    if (error) {
        error.style.display =
            "none";
    }


    if (content) {
        content.style.display =
            "block";
    }


    /*
     * Your backend returns something similar to:
     *
     * {
     *   data: {
     *      risk: [...],
     *      explanations: [...],
     *      spatial: [...],
     *      redistribution: [...]
     *   }
     * }
     */

    const data =
        response?.data ||
        response ||
        {};


    const risk =
        Array.isArray(data.risk) &&
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


    /*
     * Merge the most useful fields.
     */

    const facility = {
        ...risk,
        ...explanation
    };


    renderHeader(
        facility,
        spatialRows,
        facilityId
    );


    renderMetrics(
        facility,
        spatialRows
    );


    renderEvidence(
        facility,
        spatialRows
    );


    renderLocalConditions(
        facility,
        spatialRows
    );


    renderXAI(
        facility
    );


    renderRedistribution(
        redistribution,
        facility
    );


    renderSpatial(
        spatialRows
    );
}


// ============================================================
// FACILITY HEADER
// ============================================================

function renderHeader(
    facility,
    spatialRows,
    facilityId
) {

    const id =
        facility.facility_id ??
        facilityId;


    const riskLevel =
        facility.risk_level ??
        "UNKNOWN";


    const score =
        facility.final_shortage_score;


    const district =
        facility.district ??
        facility.district_name ??
        spatialRows[0]?.district ??
        spatialRows[0]?.district_name ??
        "—";


    const date =
        facility.date ??
        facility.timestamp ??
        spatialRows[0]?.date;


    setText(
        "facility-id",
        id
    );


    setText(
        "facility-location",
        district
    );


    setText(
        "facility-date",
        `Risk assessment date: ${formatDate(date)}`
    );


    setText(
        "facility-score",
        formatScore(score)
    );


    const badge =
        $("facility-risk-badge");


    if (badge) {

        badge.textContent =
            riskLevel;


        const riskClass =
            String(
                riskLevel
            )
                .toLowerCase()
                .replace(
                    /\s+/g,
                    "-"
                );


        badge.className =
            `risk-badge ${riskClass}`;
    }
}


// ============================================================
// KEY METRICS
// ============================================================

function renderMetrics(
    facility,
    spatialRows
) {

    const spatial =
        spatialRows[0] ||
        {};


    const mlRisk =
        facility.ml_shortage_risk ??
        facility.prediction_stockout_risk ??
        facility.ml_risk;


    const spatialPressure =
        facility.spatial_shortage_pressure ??
        facility.local_shortage_pressure ??
        facility.spatial_pressure;


    const evidenceCount =
        facility.evidence_count ??
        facility.signals ??
        0;


    const currentStockout =
        Number(
            facility.current_stockout
        ) === 1;


    // ML risk

    setText(
        "metric-ml-risk",
        formatPercent(
            mlRisk
        )
    );


    const mlBar =
        $("ml-risk-bar");


    if (mlBar) {

        let percentage =
            Number(mlRisk);


        if (
            Number.isFinite(
                percentage
            )
        ) {

            if (
                Math.abs(
                    percentage
                ) <= 1
            ) {
                percentage *= 100;
            }

            percentage =
                Math.max(
                    0,
                    Math.min(
                        100,
                        percentage
                    )
                );

        }

        else {
            percentage = 0;
        }


        mlBar.style.width =
            `${percentage}%`;
    }


    // Spatial

    setText(
        "metric-spatial",
        formatPercent(
            spatialPressure
        )
    );


    // Evidence

    setText(
        "metric-evidence",
        formatNumber(
            evidenceCount
        )
    );


    // Current stockout

    setText(
        "metric-stockout",
        currentStockout
            ? "YES"
            : "NO"
    );


    setText(
        "metric-stockout-label",
        currentStockout
            ? "Current stockout observed"
            : "No current stockout observed"
    );
}


// ============================================================
// EVIDENCE
// ============================================================

function renderEvidence(
    facility,
    spatialRows
) {

    const container =
        $("evidence-list");


    if (!container) {
        return;
    }


    const spatial =
        spatialRows[0] ||
        {};


    const mlRisk =
        facility.ml_shortage_risk ??
        facility.prediction_stockout_risk;


    const spatialPressure =
        facility.spatial_shortage_pressure ??
        facility.local_shortage_pressure;


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


    const nearbyStockouts =
        spatial.nearby_stockouts;


    const nearbyFacilities =
        spatial.nearby_facilities;


    const nearbyRate =
        spatial.nearby_stockout_rate;


    const evidence = [];


    // --------------------------------------------------------
    // ML prediction
    // --------------------------------------------------------

    if (
        mlRisk !== null &&
        mlRisk !== undefined
    ) {

        evidence.push({
            title:
                "ML shortage prediction",

            text:
                `The model estimates a ${formatPercent(
                    mlRisk
                )} probability of next-period stockout.`,

            active:
                Number(
                    mlRisk
                ) >= 0.5
        });
    }


    // --------------------------------------------------------
    // Current stockout
    // --------------------------------------------------------

    evidence.push({
        title:
            "Current stock status",

        text:
            currentStockout
                ? "A current stockout is observed in the dataset."
                : "No current stockout is observed in the latest available record.",

        active:
            currentStockout
    });


    // --------------------------------------------------------
    // Nearby facilities
    // --------------------------------------------------------

    if (
        spatialPressure !== null &&
        spatialPressure !== undefined
    ) {

        let text =
            `Nearby shortage pressure is ${formatPercent(
                spatialPressure
            )}.`;


        if (
            nearbyFacilities !== undefined
        ) {

            text +=
                ` The spatial window contains ${formatNumber(
                    nearbyFacilities
                )} nearby facilities.`;
        }


        if (
            nearbyStockouts !== undefined
        ) {

            text +=
                ` ${formatNumber(
                    nearbyStockouts
                )} nearby facilities are currently in stockout.`;
        }


        if (
            nearbyRate !== undefined
        ) {

            text +=
                ` The nearby stockout rate is ${formatPercent(
                    nearbyRate
                )}.`;
        }


        evidence.push({
            title:
                "Neighbourhood pressure",

            text,

            active:
                Number(
                    spatialPressure
                ) >= 0.5
        });
    }


    // --------------------------------------------------------
    // Inventory anomaly
    // --------------------------------------------------------

    if (
        anomalyFlag
    ) {

        evidence.push({
            title:
                "Inventory behaviour anomaly",

            text:
                `Unusual consumption or inventory behaviour was detected. Anomaly evidence score: ${formatScore(
                    anomalyEvidence
                )}.`,

            active:
                true
        });

    }

    else {

        evidence.push({
            title:
                "Inventory behaviour",

            text:
                "No significant inventory behaviour anomaly was detected.",

            active:
                false
        });
    }


    // --------------------------------------------------------
    // Regional signal
    // --------------------------------------------------------

    evidence.push({
        title:
            "Regional shortage signal",

        text:
            regionalEvent
                ? "A regional shortage signal is associated with this facility and period."
                : "No regional shortage event is associated with this facility and period.",

        active:
            regionalEvent
    });


    container.innerHTML =
        evidence
            .map(
                item =>
                    `
                    <div class="evidence-item ${
                        item.active
                            ? "active"
                            : "inactive"
                    }">

                        <div class="evidence-icon">
                            ${
                                item.active
                                    ? "✓"
                                    : "–"
                            }
                        </div>

                        <div>

                            <strong>
                                ${escapeHTML(
                                    item.title
                                )}
                            </strong>

                            <p>
                                ${escapeHTML(
                                    item.text
                                )}
                            </p>

                        </div>

                    </div>
                    `
            )
            .join("");
}


// ============================================================
// LOCAL CONDITIONS
// ============================================================

function renderLocalConditions(
    facility,
    spatialRows
) {

    const container =
        $("facility-metrics");


    if (!container) {
        return;
    }


    const spatial =
        spatialRows[0] ||
        {};


    const inventory =
        facility.closing_inventory;


    const consumption =
        facility.consumption;


    const nearbyFacilities =
        spatial.nearby_facilities;


    const nearbyStockouts =
        spatial.nearby_stockouts;


    const nearbyRate =
        spatial.nearby_stockout_rate;


    const localPressure =
        spatial.local_shortage_pressure ??
        facility.spatial_shortage_pressure;


    const overallRate =
        spatial.overall_stockout_rate;


    const regionalEvent =
        Number(
            facility.regional_event
        ) === 1;


    const anomalyFlag =
        Number(
            facility.anomaly_flag
        ) === 1;


    container.innerHTML = `

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
                nearbyRate
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
                overallRate
            )
        )}


        ${detailMetric(
            "Regional event",
            regionalEvent
                ? "Detected"
                : "Not detected"
        )}


        ${detailMetric(
            "Inventory anomaly",
            anomalyFlag
                ? "Detected"
                : "Not detected"
        )}

    `;
}


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
// XAI
// ============================================================

function renderXAI(
    facility
) {

    const container =
        $("xai-content");


    if (!container) {
        return;
    }


    const primaryReason =
        facility.xai_primary_reason ??
        "";


    const modelEvidence =
        facility.xai_model_evidence ??
        "";


    const observedEvidence =
        facility.xai_observed_evidence ??
        "";


    const explanation =
        facility.xai_explanation ??
        facility.evidence_summary ??
        "";


    let html = "";


    if (primaryReason) {

        html += `

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

        `;
    }


    if (modelEvidence) {

        html += `

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

        `;
    }


    if (observedEvidence) {

        html += `

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

        `;
    }


    if (explanation) {

        html += `

            <div class="xai-explanation">

                <span>
                    SUMMARY
                </span>

                <p>
                    ${escapeHTML(
                        explanation
                    )}
                </p>

            </div>

        `;
    }


    if (!html) {

        html = `

            <div class="empty-state compact">

                <p>
                    No additional explainability information
                    is available for this facility.
                </p>

            </div>

        `;
    }


    container.innerHTML =
        html;
}


// ============================================================
// REDISTRIBUTION
// ============================================================

function renderRedistribution(
    options,
    facility
) {

    const container =
        $("redistribution-content");


    if (!container) {
        return;
    }


    if (
        !options ||
        options.length === 0
    ) {

        container.innerHTML = `

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

        return;
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


    container.innerHTML = `

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
                    formatNumber(
                        bestDistance,
                        1
                    ) + " km"
                )}


                ${detailMetric(
                    "Available surplus",
                    formatNumber(
                        bestSurplus,
                        1
                    )
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
                    .slice(
                        0,
                        8
                    )
                    .map(
                        renderDonor
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


function renderDonor(
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

function renderSpatial(
    rows
) {

    const container =
        $("spatial-content");


    if (!container) {
        return;
    }


    if (
        !rows ||
        rows.length === 0
    ) {

        container.innerHTML = `

            <div class="empty-state compact">

                <p>
                    No spatial observations are available
                    for this facility.
                </p>

            </div>

        `;

        return;
    }


    const limitedRows =
        rows.slice(
            0,
            15
        );


    container.innerHTML = `

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
// ERROR
// ============================================================

function showError(
    message
) {

    const loading =
        $("facility-loading");

    const content =
        $("facility-content");

    const error =
        $("facility-error");

    const errorMessage =
        $("facility-error-message");


    if (loading) {
        loading.style.display =
            "none";
    }


    if (content) {
        content.style.display =
            "none";
    }


    if (error) {
        error.style.display =
            "block";
    }


    if (errorMessage) {

        errorMessage.textContent =
            message ||
            "Unable to retrieve facility information.";
    }
}


// ============================================================
// INITIALIZATION
// ============================================================

async function initializeFacilityPage() {

    console.log(
        "SupplyShield AI facility page starting..."
    );


    await checkBackendStatus();


    await loadFacility();


    console.log(
        "Facility investigation page ready."
    );
}


// ============================================================
// START
// ============================================================

if (
    document.readyState === "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeFacilityPage
    );

}

else {

    initializeFacilityPage();

}