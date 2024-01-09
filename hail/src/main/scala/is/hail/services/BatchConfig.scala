package is.hail.services

import is.hail.utils._

import org.json4s._
import org.json4s.jackson.JsonMethods

import java.io.{File, FileInputStream}

import org.apache.log4j.Logger

object BatchConfig {
  private[this] val log = Logger.getLogger("BatchConfig")

  def fromConfigFile(file: String): Option[BatchConfig] =
    if (new File(file).exists()) {
      using(new FileInputStream(file))(in => Some(fromConfig(JsonMethods.parse(in))))
    } else {
      None
    }

  def fromConfig(config: JValue): BatchConfig = {
    implicit val formats: Formats = DefaultFormats
    new BatchConfig((config \ "batch_id").extract[Int])
  }
}

class BatchConfig(val batchId: Long)
