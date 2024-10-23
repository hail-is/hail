package is.hail.services

import is.hail.utils._

import java.nio.file.{Files, Path}

import org.json4s._
import org.json4s.jackson.JsonMethods

object BatchConfig {
  def fromConfigFile(file: Path): BatchConfig =
    using(Files.newInputStream(file))(in => fromConfig(JsonMethods.parse(in)))

  def fromConfig(config: JValue): BatchConfig = {
    implicit val formats: Formats = DefaultFormats
    new BatchConfig((config \ "batch_id").extract[Int], (config \ "job_group_id").extract[Int])
  }
}

case class BatchConfig(batchId: Int, jobGroupId: Int)
