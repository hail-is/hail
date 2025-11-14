const csrfToken = document.getElementById('_csrf').value;
const basePath = document.getElementById('_base_path').value;

document.getElementsByName('remove-role-button').forEach(button => {
        button.addEventListener('click', (e) => { removeRole(button.dataset.user, button.dataset.role); })
});

document.getElementsByName('add-role-button').forEach(button => {
        button.addEventListener('click', (e) => { addRole(button.dataset.role); })
});

async function removeRole(username, roleName) {
    if (!confirm(`Are you sure you want to remove the '${roleName}' role from user '${username}'?`)) {
        return;
    }

    try {
        const response = await fetch(`${basePath}/api/v1alpha/system_roles/${username}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            body: JSON.stringify({
                role_removal: roleName
            })
        });

        if (response.ok) {
            // Refresh the page to show updated data
            window.location.reload();
        } else {
            const errorData = await response.json().catch(() => ({}));
            alert(`Failed to remove role: ${errorData.message || response.statusText}`);
        }
    } catch (error) {
        console.error('Error removing role:', error);
        alert('Failed to remove role. Please try again.');
    }
}

async function addRole(roleName) {
    const inputId = `add-user-${roleName}`;
    const username = document.getElementById(inputId).value.trim();

    if (!username) {
        alert('Please enter a username');
        return;
    }

    try {
        const response = await fetch(`${basePath}/api/v1alpha/system_roles/${username}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            body: JSON.stringify({
                role_addition: roleName
            })
        });

        if (response.ok) {
            // Clear the input and refresh the page
            document.getElementById(inputId).value = '';
            window.location.reload();
        } else {
            const errorData = await response.json().catch(() => ({}));
            alert(`Failed to add role: ${errorData.message || response.statusText}`);
        }
    } catch (error) {
        console.error('Error adding role:', error);
        alert('Failed to add role. Please try again.');
    }
}
