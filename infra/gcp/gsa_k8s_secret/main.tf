resource "random_id" "name_suffix" {
  byte_length = 2
}

resource "google_service_account" "service_account" {
  account_id = "${var.name}-${random_id.name_suffix.hex}"
}

resource "google_service_account_key" "service_account_key" {
  service_account_id = google_service_account.service_account.name
}

resource "kubernetes_secret" "k8s_key" {
  metadata {
    name = "${var.name}-gsa-key"
  }

  data = {
    "key.json" = base64decode(google_service_account_key.service_account_key.private_key)
  }
}

resource "google_project_iam_member" "iam_member" {
  for_each = toset(var.iam_roles)

  project = var.project
  role = "roles/${each.key}"
  member = "serviceAccount:${google_service_account.service_account.email}"
}
