document.getElementsByName("table-search-input-box").forEach(input => {
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            var formId = input.id.substring(0, input.id.lastIndexOf("-input-box")) + "-form";
            document.getElementById(formId).submit();
        }
    })
});

// Field configurations with their allowed operators and value types
const fieldConfigs = {
    // General fields
    'name': {
        operators: ['=', '!=', '=~', '!~'],
        type: 'string',
        placeholder: 'Enter name...'
    },
    'cost': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'number',
        placeholder: 'Enter cost (e.g., 5.00)...'
    },
    'duration': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'number',
        placeholder: 'Enter duration in seconds...'
    },
    'start_time': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'datetime',
        placeholder: 'Enter date (ISO format, e.g., 2025-02-27T17:15:25Z)...'
    },
    'end_time': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'datetime',
        placeholder: 'Enter date (ISO format, e.g., 2025-02-27T17:15:25Z)...'
    },
    'state': {
        operators: ['=', '!='],
        type: 'string',
        placeholder: 'Select state...'
    },

    // Batch-specific fields
    'batch_id': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'number',
        placeholder: 'Enter batch ID...'
    },
    'user': {
        operators: ['=', '!='],
        type: 'string',
        placeholder: 'Enter username...'
    },
    'billing_project': {
        operators: ['=', '!='],
        type: 'string',
        placeholder: 'Enter billing project...'
    },

    // Job-specific fields
    'job_id': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'number',
        placeholder: 'Enter job ID...'
    },
    'instance': {
        operators: ['=', '!=', '=~', '!~'],
        type: 'string',
        placeholder: 'Enter instance name...'
    },
    'instance_collection': {
        operators: ['=', '!=', '=~', '!~'],
        type: 'string',
        placeholder: 'Enter instance collection...'
    },
    'exit_code': {
        operators: ['=', '!=', '>', '>=', '<', '<='],
        type: 'number',
        placeholder: 'Enter exit code (e.g., 0, 1)...'
    }
};

// Special state values
const stateValues = {
    batch: ['running', 'complete', 'success', 'failure', 'cancelled', 'open', 'closed'],
    job: ['pending', 'ready', 'creating', 'running', 'live', 'cancelled', 'error', 'failed', 'bad', 'success', 'done']
};

// Global variable for search context
let isJobSearch;
let currentMode = 'dropdown'; // 'dropdown' or 'textbox'

// Cached clone of the initial structured criterion row. Captured on DOMContentLoaded
// so we can reconstruct a blank structured row even when no `.search-criterion` is
// currently in the DOM (e.g. after parseQuery() produced only free-text rows).
let structuredRowTemplate = null;

// Update operators based on selected field
function updateOperators(selectElement) {
    const field = selectElement.value;
    const operatorSelect = selectElement.parentElement.querySelector('.operator-select');
    const valueInput = selectElement.parentElement.querySelector('.value-input');

    // Clear existing options
    operatorSelect.innerHTML = '<option value="">Select operator...</option>';

    if (field && Object.hasOwn(fieldConfigs, field)) {
        const config = fieldConfigs[field];

        // Add operator options
        config.operators.forEach(op => {
            const option = document.createElement('option');
            option.value = op;
            option.textContent = op;
            operatorSelect.appendChild(option);
        });

        // Update value input
        valueInput.placeholder = config.placeholder;

        // Handle state fields via dropdown
        if (field === 'state') {
            valueInput.style.display = 'none';
            if (!selectElement.parentElement.querySelector('.state-select')) {
                const stateSelect = document.createElement('select');
                stateSelect.className = 'state-select p-2 bg-white rounded border';
                stateSelect.innerHTML = '<option value="">Select state...</option>';

                const values = isJobSearch ? stateValues.job : stateValues.batch;
                values.forEach(state => {
                    const option = document.createElement('option');
                    option.value = state;
                    option.textContent = state;
                    stateSelect.appendChild(option);
                });

                valueInput.parentElement.insertBefore(stateSelect, valueInput);
            }
        } else {
            valueInput.style.display = 'block';
            const stateSelect = selectElement.parentElement.querySelector('.state-select');
            if (stateSelect) {
                stateSelect.remove();
            }
        }
    }
}

// Add new search criterion. Clones the first existing structured row when available;
// falls back to `structuredRowTemplate` when the container currently holds only
// free-text rows.
function addCriterion() {
    const container = document.getElementById('search-criteria-container');
    const source = container.querySelector('.search-criterion') || structuredRowTemplate;
    const newCriterion = source.cloneNode(true);

    // Reset values
    newCriterion.querySelector('.criterion-select').value = '';
    newCriterion.querySelector('.criterion-select').addEventListener('change', function (_e) { updateOperators(this); });
    newCriterion.querySelector('.operator-select').innerHTML = '<option value="">Select operator...</option>';
    newCriterion.querySelector('.value-input').value = '';
    newCriterion.querySelector('.value-input').placeholder = 'Enter value...';
    newCriterion.querySelector('.remove-criterion').addEventListener('click', function (_e) { removeCriterion(this); });

    // Remove any state select
    const stateSelect = newCriterion.querySelector('.state-select');
    if (stateSelect) {
        stateSelect.remove();
    }

    container.appendChild(newCriterion);
    updateRemoveButtonsVisibility(container);
}

// Add a free-text row (clones the <template id='search-freetext-template'>)
function addFreeTextCriterion(initialValue) {
    const container = document.getElementById('search-criteria-container');
    const tpl = document.getElementById('search-freetext-template');
    const row = tpl.content.firstElementChild.cloneNode(true);
    row.querySelector('.freetext-input').value = initialValue == null ? '' : initialValue;
    row.querySelector('.remove-criterion').addEventListener('click', function () { removeCriterion(this); });
    container.appendChild(row);
    updateRemoveButtonsVisibility(container);
    return row;
}

// Update visibility of remove buttons based on total row count (structured + free-text)
function updateRemoveButtonsVisibility(container) {
    const rows = container.querySelectorAll('.search-criterion, .search-freetext');
    const removeButtons = container.querySelectorAll('.remove-criterion');

    removeButtons.forEach(button => {
        if (rows.length <= 1) {
            button.style.display = 'none';
        } else {
            button.style.display = 'block';
        }
    });
}

// Remove a row (structured or free-text). Prevents removing the last remaining row.
function removeCriterion(button) {
    const container = button.closest('[id$="-criteria-container"]');
    const rows = container.querySelectorAll('.search-criterion, .search-freetext');

    // Don't remove if it's the last remaining row
    if (rows.length > 1) {
        const row = button.closest('.search-criterion, .search-freetext');
        if (row) {
            row.remove();
        }
        updateRemoveButtonsVisibility(container);
    }
}

// Populate a structured criterion row (already in the DOM) with field/operator/value.
function populateStructuredRow(criterion, field, operator, value) {
    criterion.querySelector('.criterion-select').value = field;
    updateOperators(criterion.querySelector('.criterion-select'));
    criterion.querySelector('.operator-select').value = operator;

    if (field === 'state') {
        const stateSelect = criterion.querySelector('.state-select');
        if (stateSelect) {
            stateSelect.value = value;
        }
    } else {
        criterion.querySelector('.value-input').value = value;
    }
}

// Build query string from all rows (structured + free-text), preserving DOM order.
function buildQuery() {
    const container = document.getElementById('search-criteria-container');
    const rows = container.querySelectorAll('.search-criterion, .search-freetext');
    const queryParts = [];

    rows.forEach(row => {
        if (row.classList.contains('search-freetext')) {
            const value = row.querySelector('.freetext-input').value.trim();
            if (value) {
                queryParts.push(value);
            }
            return;
        }

        const field = row.querySelector('.criterion-select').value;
        const operator = row.querySelector('.operator-select').value;
        let value = row.querySelector('.value-input').value;

        // Check for state select
        const stateSelect = row.querySelector('.state-select');
        if (stateSelect && stateSelect.style.display !== 'none') {
            value = stateSelect.value;
        }

        if (field && operator && value) {
            queryParts.push(`${field} ${operator} ${value}`);
        }
    });

    return queryParts.join('\n');
}

// Classify a query line: return {kind: 'structured', field, operator, value}
// if it matches "<known-field> <valid-op> <value>"; otherwise return {kind: 'freetext'}.
function classifyQueryLine(line) {
    const match = line.trim().match(/^(\w+)\s*(=~|!~|<=|>=|!=|==|=|<|>)\s*(.+)$/);
    if (match) {
        const [, field, operator, value] = match;
        if (Object.hasOwn(fieldConfigs, field)) {
            const config = fieldConfigs[field];
            if (config.operators.includes(operator)) {
                return { kind: 'structured', field, operator, value };
            }
        }
    }
    return { kind: 'freetext' };
}

// Parse an existing query and populate the rows. Each non-blank line becomes either
// a structured row (if it matches "<known-field> <valid-op> <value>") or a free-text
// row (otherwise). Unknown-field terms like `custom_attr = foo` render verbatim as
// free-text and round-trip losslessly.
function parseQuery(query) {
    if (!query) return;

    const container = document.getElementById('search-criteria-container');

    // Preserve the first structured row's DOM shape as a template for reconstruction,
    // then remove ALL existing rows (both kinds). We rebuild from scratch below. Fall
    // back to the module-level `structuredRowTemplate` if no structured row is present.
    const firstStructured = container.querySelector('.search-criterion');
    const structuredTemplate = firstStructured
        ? firstStructured.cloneNode(true)
        : structuredRowTemplate;
    container.querySelectorAll('.search-criterion, .search-freetext').forEach(row => { row.remove(); });

    const lines = query.split('\n').filter(line => line.trim());

    if (lines.length === 0) {
        // Nothing parseable — restore a single blank structured starter row.
        if (structuredTemplate) {
            appendBlankStructuredRow(container, structuredTemplate);
        }
        updateRemoveButtonsVisibility(container);
        return;
    }

    lines.forEach(line => {
        const classification = classifyQueryLine(line);
        if (classification.kind === 'structured' && structuredTemplate) {
            const criterion = appendBlankStructuredRow(container, structuredTemplate);
            populateStructuredRow(
                criterion,
                classification.field,
                classification.operator,
                classification.value,
            );
        } else {
            addFreeTextCriterion(line.trim());
        }
    });

    updateRemoveButtonsVisibility(container);
}

// Clone `structuredTemplate`, reset its inputs, wire up its listeners, and append it
// to `container`. Returns the newly appended row.
function appendBlankStructuredRow(container, structuredTemplate) {
    const newCriterion = structuredTemplate.cloneNode(true);

    // Reset values
    const criterionSelect = newCriterion.querySelector('.criterion-select');
    if (criterionSelect) {
        criterionSelect.value = '';
        criterionSelect.addEventListener('change', function () { updateOperators(this); });
    }
    const operatorSelect = newCriterion.querySelector('.operator-select');
    if (operatorSelect) {
        operatorSelect.innerHTML = '<option value="">Select operator...</option>';
    }
    const valueInput = newCriterion.querySelector('.value-input');
    if (valueInput) {
        valueInput.value = '';
        valueInput.placeholder = 'Enter value...';
        valueInput.style.display = 'block';
    }
    const removeButton = newCriterion.querySelector('.remove-criterion');
    if (removeButton) {
        removeButton.addEventListener('click', function () { removeCriterion(this); });
    }

    // Remove any state select carried over from the clone
    const stateSelect = newCriterion.querySelector('.state-select');
    if (stateSelect) {
        stateSelect.remove();
    }

    container.appendChild(newCriterion);
    return newCriterion;
}

// Toggle between dropdown and textbox input modes
function toggleInputMode() {
    const dropdownContainer = document.getElementById('search-criteria-container');
    const textboxContainer = document.getElementById('search-textbox-container');
    const toggleButton = document.getElementById('search-toggle-mode');
    const addButton = document.getElementById('search-add-criterion');
    const addFreeTextButton = document.getElementById('search-add-freetext');
    const textbox = document.getElementById('search-input-box');
    const hiddenInput = document.getElementById('search-query');

    if (currentMode === 'dropdown') {
        // Switch to textbox mode and sync dropdown to textbox
        const query = buildQuery();
        textbox.value = query;
        hiddenInput.value = query;

        dropdownContainer.style.display = 'none';
        textboxContainer.style.display = 'block';
        toggleButton.textContent = 'Switch to drop-down query input';
        addButton.style.display = 'none';
        addFreeTextButton.style.display = 'none';

        currentMode = 'textbox';
    } else {
        // Switch to dropdown mode and sync textbox to dropdown
        const query = textbox.value.trim();
        hiddenInput.value = query;

        dropdownContainer.style.display = 'block';
        textboxContainer.style.display = 'none';
        toggleButton.textContent = 'Switch to textbox query input';
        addButton.style.display = 'block';
        addFreeTextButton.style.display = 'block';

        // Parse the textbox query back into dropdowns
        if (query) {
            parseQuery(query);
        } else {
            // If textbox is empty, reset to single empty criterion
            const container = document.getElementById('search-criteria-container');
            const criteria = container.querySelectorAll('.search-criterion');
            // Remove extra criteria, keep only first one
            for (let i = criteria.length - 1; i > 0; i--) {
                criteria[i].remove();
            }
            // Reset first criterion
            const firstCriterion = container.querySelector('.search-criterion');
            if (firstCriterion) {
                firstCriterion.querySelector('.criterion-select').value = '';
                firstCriterion.querySelector('.operator-select').innerHTML = '<option value="">Select operator...</option>';
                firstCriterion.querySelector('.value-input').value = '';
                const stateSelect = firstCriterion.querySelector('.state-select');
                if (stateSelect) stateSelect.remove();
            }
            updateRemoveButtonsVisibility(container);
        }

        currentMode = 'dropdown';
    }

    // Add real-time sync for textbox changes
    if (currentMode === 'textbox') {
        // Remove previous listener if any, then add
        textbox.removeEventListener('input', syncTextboxToHiddenInput);
        textbox.addEventListener('input', syncTextboxToHiddenInput);
    }
}
// Handler for syncing textbox to hidden input
function syncTextboxToHiddenInput() {
    const hiddenInput = document.getElementById('search-query');
    const textbox = document.getElementById('search-input-box');
    hiddenInput.value = textbox.value;
}

// Form submission handler
document.getElementById('search-form').addEventListener('submit', function(_e) {
    let query;
    if (currentMode === 'dropdown') {
        query = buildQuery();
    } else {
        query = document.getElementById('search-input-box').value;
    }
    document.getElementById('search-query').value = query;
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize search context
    isJobSearch = document.getElementById('search-form').getAttribute('data-is-job-search') === 'true';

    const container = document.getElementById('search-criteria-container');

    // Capture the initial-load structured row shape so we can rebuild blank structured
    // rows even when the container currently holds no `.search-criterion` element
    // (e.g. after parseQuery() produces only free-text rows).
    const initialFirst = container.querySelector('.search-criterion');
    if (initialFirst) {
        structuredRowTemplate = initialFirst.cloneNode(true);
    }

    const existingQuery = document.getElementById('search-query').value;
    if (existingQuery) {
        parseQuery(existingQuery);
    } else {
        // Update remove button visibility on initial load
        updateRemoveButtonsVisibility(container);
    }
});

// Keyboard shortcut support
document.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        let query;
        if (currentMode === 'dropdown') {
            query = buildQuery();
        } else {
            query = document.getElementById('search-input-box').value;
        }
        document.getElementById('search-query').value = query;
        document.getElementById('search-form').submit();
    }
});

// Add initial event listeners
document.getElementsByName('remove-criterion').forEach(button => {
    button.addEventListener('click', function (_e) { removeCriterion(this) });
} );
document.getElementsByName('criterion-select').forEach(selector => {
    selector.addEventListener('change', function (_e) { updateOperators(this) });
} );
document.getElementById('search-add-criterion').addEventListener('click', _e => addCriterion());
document.getElementById('search-add-freetext').addEventListener('click', () => addFreeTextCriterion(''));
document.getElementById('search-toggle-mode').addEventListener('click', _e => toggleInputMode());
