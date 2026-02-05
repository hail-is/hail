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

// Add new search criterion
function addCriterion() {
    const container = document.getElementById('search-criteria-container');
    const newCriterion = container.querySelector('.search-criterion').cloneNode(true);

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

// Update visibility of remove buttons based on criteria count
function updateRemoveButtonsVisibility(container) {
    const criteria = container.querySelectorAll('.search-criterion');
    const removeButtons = container.querySelectorAll('.remove-criterion');

    removeButtons.forEach(button => {
        if (criteria.length <= 1) {
            button.style.display = 'none';
        } else {
            button.style.display = 'block';
        }
    });
}

// Remove search criterion
function removeCriterion(button) {
    const container = button.closest('[id$="-criteria-container"]');
    const criteria = container.querySelectorAll('.search-criterion');

    // Don't remove if it's the last criterion
    if (criteria.length > 1) {
        button.closest('.search-criterion').remove();
        updateRemoveButtonsVisibility(container);
    }
}

// Build query string from criteria
function buildQuery() {
    const container = document.getElementById('search-criteria-container');
    const criteria = container.querySelectorAll('.search-criterion');
    const queryParts = [];

    criteria.forEach(criterion => {
        const field = criterion.querySelector('.criterion-select').value;
        const operator = criterion.querySelector('.operator-select').value;
        let value = criterion.querySelector('.value-input').value;

        // Check for state select
        const stateSelect = criterion.querySelector('.state-select');
        if (stateSelect && stateSelect.style.display !== 'none') {
            value = stateSelect.value;
        }

        if (field && operator && value) {
            queryParts.push(`${field} ${operator} ${value}`);
        }
    });

    return queryParts.join('\n');
}

// Parse existing query and populate dropdowns
function parseQuery(query) {
    if (!query) return;

    const container = document.getElementById('search-criteria-container');
    const existingCriteria = container.querySelectorAll('.search-criterion');

    // Remove extra criteria, keeping only the first one
    for (let i = Number(existingCriteria.length - 1); i > 0; i--) {
        existingCriteria[i].remove();
    }

    const lines = query.split('\n').filter(line => line.trim());

    lines.forEach((line, index) => {
        let criterion;

        if (index === 0) {
            // Use the existing first criterion
            criterion = container.querySelector('.search-criterion');
        } else {
            // Add additional criteria
            addCriterion();
            const criteria = container.querySelectorAll('.search-criterion');
            criterion = criteria[criteria.length - 1];
        }

        // Parse line and populate the criterion
        const parts = line.trim().match(/^(\w+)\s*(=~|!~|<=|>=|!=|==|=|<|>)\s*(.+)$/);

        if (parts && criterion) {
            const [, field, operator, value] = parts;

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
    });

    updateRemoveButtonsVisibility(container);
}

// Toggle between dropdown and textbox input modes
function toggleInputMode() {
    const dropdownContainer = document.getElementById('search-criteria-container');
    const textboxContainer = document.getElementById('search-textbox-container');
    const toggleButton = document.getElementById('search-toggle-mode');
    const addButton = document.getElementById('search-add-criterion');
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

        currentMode = 'textbox';
    } else {
        // Switch to dropdown mode and sync textbox to dropdown
        const query = textbox.value.trim();
        hiddenInput.value = query;

        dropdownContainer.style.display = 'block';
        textboxContainer.style.display = 'none';
        toggleButton.textContent = 'Switch to textbox query input';
        addButton.style.display = 'block';

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
document.getElementById('search-toggle-mode').addEventListener('click', _e => toggleInputMode());
