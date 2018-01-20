package is.hail.utils

object PrintConfig {
  val default: PrintConfig = PrintConfig(missing="NA", boolTrue = "true", boolFalse = "false",
    floatFormat = "%.4e")
}

case class PrintConfig(missing: String, boolTrue: String, boolFalse: String, floatFormat: String)
