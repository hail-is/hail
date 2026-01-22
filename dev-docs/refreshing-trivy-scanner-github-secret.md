# Refreshing the trivy-scanner GitHub Secret

The `GOOGLE_GAR_CREDENTIALS` GitHub secret is used by the Trivy Security Scan workflow
(`.github/workflows/trivy-scan.yml`) to authenticate with Google Cloud and access
images from Google Artifact Registry (GAR) for vulnerability scanning.

This secret contains a Google Service Account (GSA) key in JSON format. Service account
keys should be rotated periodically for security. This guide walks through the process
of creating a new GSA key and updating the GitHub secret.

## Prerequisites

- Access to the Google Cloud Console for the relevant project
- Admin access to the GitHub repository to update secrets
- Knowledge of which Google Service Account is used for the trivy-scanner workflow

## Creating a New GSA Key in Google Cloud Console

1. **Navigate to the Google Cloud Console**
   - Go to https://console.cloud.google.com
   - Select the appropriate project (e.g., `hail-vdc`)

2. **Open the Service Accounts page**
   - In the left navigation menu, go to **IAM & Admin** → **Service Accounts**
   - Or navigate directly to: https://console.cloud.google.com/iam-admin/serviceaccounts

3. **Find the Service Account**
   - Locate the `trivy-scanner` service account.
   - Click on the service account name to open its details page

4. **Create a New Key**
   - Click on the **KEYS** tab
   - Click **ADD KEY** → **Create new key**
   - Select **JSON** as the key type
   - Click **CREATE**
   - The JSON key file will automatically download to your computer

5. **Save the Key Securely**
   - The downloaded file contains sensitive credentials
   - Store it temporarily in a secure location until you've updated the GitHub secret

## Copying the Key to GitHub Secret

1. **Open the Key File**
   - Open the downloaded JSON key file in a text editor
   - The file contains a JSON object with fields like `type`, `project_id`, `private_key_id`,
     `private_key`, `client_email`, etc.

2. **Copy the Entire Contents**
   - Select all contents of the JSON file (the entire JSON object)
   - Copy it to your clipboard
   - Ensure you copy the complete JSON, including all opening and closing braces

3. **Navigate to GitHub Repository Settings**
   - Go to the repository: https://github.com/hail-is/hail
   - Click on **Settings** (requires admin access)
   - In the left sidebar, click **Secrets and variables** → **Actions**

4. **Update the Secret**
   - Find the `GOOGLE_GAR_CREDENTIALS` secret in the list
   - Click **Update** (or if it doesn't exist, click **New repository secret**)
   - Paste the entire JSON key contents into the **Secret** field
   - Click **Update secret** (or **Add secret**)
   - You might need to authenticate with 2FA to update the secret

5. **Verify the Update**
   - The secret should now show as "Last updated: now"
   - You can verify it works by running the Trivy Security Scan workflow manually
   - Go to **Actions** → **Trivy Security Scan** → **Run workflow**

## Cleanup

After successfully updating the GitHub secret:

1. **Delete the Old Keys**
   - Return to the Google Cloud Console Service Accounts page
   - Open the service account's **KEYS** tab
   - Identify the old key (it will have an older creation date)
   - Click the **Delete** icon (trash can) next to the old key
   - Confirm the deletion

2. **Delete the Local Key File**
   - Securely delete the downloaded JSON key file from your local machine
   - Consider using `shred` or secure deletion tools if the file contained highly sensitive data

## Troubleshooting

- **Authentication errors in workflow**: Verify the JSON was copied completely and correctly
- **Permission denied errors**: Ensure the service account has the necessary Artifact Registry permissions
- **Key not found**: Verify you're using the correct service account for the trivy-scanner workflow
