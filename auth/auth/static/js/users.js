const showActiveCheckbox = document.getElementById('show-active');
showActiveCheckbox.addEventListener('change', (_e) => { filterTable(); })
const showDevelopersCheckbox = document.getElementById('show-developers');
showDevelopersCheckbox.addEventListener('change', (_e) => { filterTable(); })

const reactivateUserForm = document.getElementById('reactivate-user-form');
reactivateUserForm.addEventListener('submit', (_e) => { validateReactivateForm(); })

function filterTable() {
    const showActive = document.getElementById('show-active').checked;
    const showDevelopers = document.getElementById('show-developers').checked;

    const rows = document.querySelectorAll('#users-table tbody tr');

    rows.forEach(row => {
        const state = row.querySelector('[data-state]').dataset.state.trim();
        const isDeveloper = row.querySelector('[data-developer]').dataset.developer.trim();

        let show = true;
        if (showActive && state !== 'active') show = false;
        if (showDevelopers && isDeveloper !== '1') show = false;

        row.style.display = show ? '' : 'none';
    });
}
function validateReactivateForm() {
    const userId = document.getElementById('inactive-id').value.trim();
    const username = document.getElementById('inactive-username').value.trim();

    if (!userId && !username) {
        alert('Please provide either a User ID or a Username.');
        return false;
    }
    return true;
}
