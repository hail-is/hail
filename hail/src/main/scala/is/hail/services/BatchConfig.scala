package is.hail.services

import is.hail.utils._
import org.json4s._
import org.json4s.jackson.JsonMethods

import java.nio.file.{Files, Path}

object BatchConfig {
  def fromConfigFile(file: Path): Option[BatchConfig] =
    if (!file.toFile.exists()) None
    else using(Files.newInputStream(file))(in => Some(fromConfig(JsonMethods.parse(in))))

  def fromConfig(config: JValue): BatchConfig = {
    implicit val formats: Formats = DefaultFormats
    new BatchConfig((config \ "batch_id").extract[Int], (config \ "job_group_id").extract[Int])
  }
}

case class BatchConfig(batchId: Int, jobGroupId: Int)
