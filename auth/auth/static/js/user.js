const tosCheckbox = document.getElementById('tosCheckbox');
tosCheckbox.addEventListener('load', (_e) => { updateButtons(); })
tosCheckbox.addEventListener('change', (_e) => { updateButtons(); })

function updateButtons() {
    const checkbox = document.getElementById('tosCheckbox');
    const signupButton = document.getElementById('signupButton');
    const loginButton = document.getElementById('loginButton');

    signupButton.disabled = !checkbox.checked;
    loginButton.disabled = !checkbox.checked;

    // Update button styles based on state
    [signupButton, loginButton].forEach(button => {
        if (button.disabled) {
            button.classList.add('opacity-50', 'cursor-not-allowed');
            button.classList.remove('hover:bg-blue-700');
        } else {
            button.classList.remove('opacity-50', 'cursor-not-allowed');
            button.classList.add('hover:bg-blue-700');
        }
    });
}
