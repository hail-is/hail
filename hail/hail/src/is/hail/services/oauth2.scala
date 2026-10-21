package is.hail.services

import is.hail.utils.{jsonToBytes, using}

import java.io.{ByteArrayInputStream, Serializable}
import java.nio.file.{Files, Path}

import com.google.auth.oauth2.GoogleCredentials
import org.json4s.{DefaultFormats, Formats, JValue}
import org.json4s.jackson.JsonMethods

object oauth2 {

  sealed trait CloudCredentials extends Product with Serializable {
    def accessToken: String
    def scoped(scopes: Array[String]): CloudCredentials
  }

  implicit lazy val fmts: Formats = DefaultFormats

  def HailCredentials(env: Map[String, String] = sys.env): Option[CloudCredentials] =
    for {
      config <-
        env
          .get("XDG_CONFIG_HOME")
          .map(Path.of(_))
          .orElse(env.get("HOME").map(Path.of(_, ".config")))

      identity = config.resolve("hail/identity.json").toFile
      if identity.exists()

      jvalue <- JsonMethods.parseOpt(identity)
    } yield (jvalue \ "idp").extract[String] match {
      case "Google" => GoogleCloudCredentials.fromJson(jvalue \ "credentials")
      case other => throw new IllegalArgumentException(s"Unknown identity provider: '$other'")
    }

  def CloudCredentials(keyPath: Option[Path], env: Map[String, String] = sys.env)
    : CloudCredentials =
    env.get("HAIL_CLOUD") match {
      case None | Some("gcp") => GoogleCloudCredentials(keyPath)
      case Some(cloud) => throw new IllegalArgumentException(s"Unknown cloud: '$cloud'")
    }

  case class GoogleCloudCredentials(value: GoogleCredentials) extends CloudCredentials {
    override def accessToken: String = {
      value.refreshIfExpired()
      value.getAccessToken.getTokenValue
    }

    override def scoped(scopes: Array[String]): GoogleCloudCredentials =
      GoogleCloudCredentials(value.createScoped(scopes: _*))
  }

  object GoogleCloudCredentials {
    def fromJson(jv: JValue): GoogleCloudCredentials =
      GoogleCloudCredentials {
        GoogleCredentials.fromStream(
          new ByteArrayInputStream(jsonToBytes(jv))
        )
      }

    def apply(keyPath: Option[Path]): GoogleCloudCredentials =
      GoogleCloudCredentials {
        keyPath match {
          case Some(path) =>
            using(Files.newInputStream(path))(GoogleCredentials.fromStream)
          case None =>
            GoogleCredentials.getApplicationDefault
        }
      }
  }
}
