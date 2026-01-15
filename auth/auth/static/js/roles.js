const csrfToken = document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') || '';

document.getElementsByName('remove-role-button').forEach(button => {
        button.addEventListener('click', (_e) => { removeRole(button.dataset.user, button.dataset.role); })
});

document.getElementsByName('add-role-button').forEach(button => {
        button.addEventListener('click', (_e) => { addRole(button.dataset['username-input-id'], button.dataset.role); })
});

async function removeRole(username, roleName) {
    if (!confirm(`Are you sure you want to remove the '${roleName}' role from user '${username}'?`)) {
        return;
    }

    try {
        const response = await fetch("{{ base_path }}/api/v1alpha/system_roles/${username}", {
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

async function addRole(usernameInputId, roleName) {
    const username = document.getElementById(usernameInputId).value.trim();

    if (!username) {
        alert('Please enter a username');
        return;
    }

    try {
        const response = await fetch("{{ base_path }}/api/v1alpha/system_roles/${username}", {
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
            document.getElementById(usernameInputId).value = '';
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
