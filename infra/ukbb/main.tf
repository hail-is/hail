resource "kubernetes_namespace" "ukbb_rg" {
  metadata {
    name = "ukbb-rg"
  }
}

resource "kubernetes_service" "ukbb_rb_browser" {
  metadata {
    name = "ukbb-rg-browser"
    namespace = kubernetes_namespace.ukbb_rg.metadata[0].name
    labels = {
      app = "ukbb-rg-browser"
    }
  }
  spec {
    port {
      port = 80
      protocol = "TCP"
      target_port = 80
    }
    selector = {
      app = "ukbb-rg-browser"
    }
  }
}

resource "kubernetes_service" "ukbb_rb_static" {
  metadata {
    name = "ukbb-rg-static"
    namespace = kubernetes_namespace.ukbb_rg.metadata[0].name
    labels = {
      app = "ukbb-rg-static"
    }
  }
  spec {
    port {
      port = 80
      protocol = "TCP"
      target_port = 80
    }
    selector = {
      app = "ukbb-rg-static"
    }
  }
}
