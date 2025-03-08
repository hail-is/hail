# Trivy Security Scanner Workflow

This GitHub Action workflow allows you to run Trivy security scans on Docker images stored in Google Artifact Registry. The workflow can be triggered manually or via API, and the results are uploaded as GitHub artifacts and to the GitHub Security tab.

## Prerequisites

1. A Google Cloud Service Account with permissions to access your Google Artifact Registry
2. An access token set up in the repository under actions / secrets / GOOGLE_GAR_CREDENTIALS
3. Repository must have GitHub Advanced Security enabled to view security alerts

## Configuration

### GitHub Secrets Required

- `GOOGLE_GAR_CREDENTIALS`: The JSON key file for a Google Cloud service account with access to your Artifact Registry
- `GITHUB_TOKEN`: Automatically provided by GitHub, used for uploading scan results

### Required Permissions

The workflow requires the following permissions:
- `security-events: write`: To upload security scan results
- `contents: read`: For reading back action artifacts

## Usage

### Manual Trigger

You can trigger the workflow manually from the GitHub Actions UI:

1. Go to the "Actions" tab in your repository
2. Select "Trivy Security Scan" from the workflows list
3. Click "Run workflow"
4. Fill in the required parameters:
   - `images`: Comma-separated list of image names to scan (e.g., "us-docker.pkg.dev/project/repo/image1:tag,us-docker.pkg.dev/project/repo/image2:tag")
   - `branch`: The branch name to associate the scan results with
   - `commit_hash`: The commit hash to associate the scan results with

### API Trigger

You can trigger the workflow via GitHub's REST API. Here's an example using curl:

```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/OWNER/REPO/actions/workflows/trivy-scan.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "images": "us-docker.pkg.dev/project/repo/image1:tag,us-docker.pkg.dev/project/repo/image2:tag",
      "branch": "main",
      "commit_hash": "abc123def456"
    }
  }'
```

Replace:
- `YOUR_GITHUB_TOKEN` with a valid GitHub token
- `OWNER` with the repository owner
- `REPO` with the repository name

## Workflow Output

The workflow produces the following outputs:

1. **GitHub Security Alerts**:
   - All vulnerabilities are uploaded to the Security tab as code scanning alerts
   - View results by going to Security > Code scanning > Trivy Container Scan
   - Alerts are associated with the specified branch and commit

2. **Scan Results Artifacts**: 
   - SARIF format report per image: `trivy-scan-<image-name>.sarif`
   - Human-readable reports per image: `trivy-scan-<image-name>.txt`
   - Combined SARIF format results file: `trivy-combined-results`

## Accessing Results

### Security Tab
1. Go to the repository's "Security" tab
2. Click on "Code scanning" in the left sidebar
3. Filter alerts by the "trivy-container-scan" tool
4. View detailed vulnerability information and recommendations

### Artifacts
1. Go to the workflow run in GitHub Actions
2. Click on the "Artifacts" section
3. Download the "trivy-scan-results" artifact
4. Extract the ZIP file to view the SARIF and text reports

## Customization

The workflow can be customized by:

1. Modifying the Trivy scan parameters in the workflow file
2. Adding additional output formats
3. Customizing the SARIF report categorization
4. Adjusting the severity levels reported to GitHub Security

## Troubleshooting

Common issues and solutions:

1. **Authentication Errors**:
   - Verify that the `GOOGLE_GAR_CREDENTIALS` secret is properly set
   - Ensure the service account has proper permissions

2. **Image Not Found**:
   - Verify the image names are correct and include the full path
   - Check if the images exist in the specified registry

3. **Security Alerts Not Appearing**:
   - Verify that GitHub Advanced Security is enabled for the repository
   - Check that the workflow has `security-events: write` permission
   - Ensure the SARIF file is properly formatted and valid

4. **Workflow Failures**:
   - Check the workflow logs for detailed error messages
   - Verify all required inputs are provided correctly 