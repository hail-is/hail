package millbuild

import upickle.default.{ReadWriter, readwriter}

enum DeployEnvironment(val scalaVersion: String, val sparkVersion: String):
  case Generic(a: String, b: String) extends DeployEnvironment(a, b)
  case `dataproc-2.3.x` extends DeployEnvironment("2.12.18", "3.5.3")
  case `dataproc-3.0.x` extends DeployEnvironment("2.13.14", "4.0.0")
  case `hdinsight-5.1` extends DeployEnvironment("2.12.15", "3.3.1")

  override def toString: String =
    this match
      case Generic(a, b) => f"spark_$a-$b"
      case default => default.productPrefix

object DeployEnvironment:
  implicit lazy val DeployTargetRW: ReadWriter[DeployEnvironment] =
    readwriter[String].bimap(_.toString, read)

  private lazy val Generic_ = raw"spark_([^-]+)-([\S]+)".r

  def unapply(s: String): Option[DeployEnvironment] =
    s match
      case Generic_(scala, spark) => Some(Generic(scala, spark))
      case "dataproc-2.3.x" => Some(`dataproc-2.3.x`)
      case "dataproc-3.0.x" => Some(`dataproc-3.0.x`)
      case "hdinsight-5.1" => Some(`hdinsight-5.1`)
      case _ => None

  def read(s: String): DeployEnvironment =
    unapply(s).getOrElse {
      throw new IllegalArgumentException(f"Unknown target: '$s'.")
    }
