resource "random_string" "name_suffix" {
  length = 3
  special = false
  upper = false
  lower = false
  lifecycle {
    ignore_changes = [
      special,
      upper,
      lower,
    ]
  }
}

resource "google_service_account" "service_account" {
  display_name = "${var.name}-${random_string.name_suffix.result}"
  account_id = "${var.name}-${random_string.name_suffix.result}"
  timeouts {}
}

resource "google_project_iam_member" "iam_member" {
  for_each = toset(var.iam_roles)

  project = var.project
  role = "roles/${each.key}"
  member = "serviceAccount:${google_service_account.service_account.email}"
}
