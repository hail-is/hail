package is.hail.services

import is.hail.expr.ir.ByteArrayBuilder
import is.hail.services.requests.{BatchServiceRequester, Requester}
import is.hail.utils._
import org.apache.http.entity.ByteArrayEntity
import org.apache.http.entity.ContentType.APPLICATION_JSON
import org.json4s.JsonAST.{JArray, JBool}
import org.json4s.jackson.JsonMethods
import org.json4s.{CustomSerializer, DefaultFormats, Extraction, Formats, JInt, JObject, JString}

import java.nio.charset.StandardCharsets
import java.nio.file.Path
import scala.util.Random

case class BatchRequest(
  billing_project: String,
  n_jobs: Int,
  token: String,
  attributes: Map[String, String] = Map.empty,
)

case class JobGroupRequest(
  job_group_id: Int,
  absolute_parent_id: Int,
  attributes: Map[String, String] = Map.empty,
)

case class JobRequest(
  job_id: Int,
  always_run: Boolean,
  in_update_job_group_id: Int,
  in_update_parent_ids: Array[Int],
  process: JobProcess,
  resources: Option[JobResources] = None,
  regions: Option[Array[String]] = None,
  cloudfuse: Option[Array[CloudfuseConfig]] = None,
  attributes: Map[String, String] = Map.empty,
)

sealed trait JobProcess

case class BashJob(
  image: String,
  command: Array[String],
) extends JobProcess

case class JvmJob(
  command: Array[String],
  jar_url: String,
  profile: Boolean,
) extends JobProcess

case class JobResources(
  preemptible: Boolean,
  cpu: Option[String],
  memory: Option[String],
  storage: Option[String],
)

case class CloudfuseConfig(
  bucket: String,
  mount_path: String,
  read_only: Boolean,
)

case class JobGroupResponse(
  batch_id: Int,
  job_group_id: Int,
  state: JobGroupState,
  complete: Boolean,
  n_jobs: Int,
  n_completed: Int,
  n_succeeded: Int,
  n_failed: Int,
  n_cancelled: Int,
)

sealed trait JobGroupState extends Product with Serializable

object JobGroupStates {
  case object Failure extends JobGroupState
  case object Cancelled extends JobGroupState
  case object Success extends JobGroupState
  case object Running extends JobGroupState
}

object BatchClient {
  def apply(deployConfig: DeployConfig, credentialsFile: Path, env: Map[String, String] = sys.env)
  : BatchClient =
    new BatchClient(BatchServiceRequester(deployConfig, credentialsFile, env))
}

case class BatchClient private (req: Requester) extends Logging with AutoCloseable {

  implicit private[this] val fmts: Formats =
    DefaultFormats +
      JobProcessRequestSerializer +
      JobGroupStateDeserializer +
      JobGroupResponseDeserializer

  def newBatch(createRequest: BatchRequest): Int = {
    val response = req.post("/api/v1alpha/batches/create", Extraction.decompose(createRequest))
    val batchId = (response \ "id").extract[Int]
    log.info(s"run: created batch $batchId")
    batchId
  }

  def newJobGroup(
    batchId: Int,
    token: String,
    jobGroup: JobGroupRequest,
    jobs: IndexedSeq[JobRequest],
  ): (Int, Int) = {

    val updateJson = JObject(
      "n_jobs" -> JInt(jobs.length),
      "n_job_groups" -> JInt(1),
      "token" -> JString(token),
    )

    val jobGroupSpec = getJsonBytes(jobGroup)
    val jobBunches = createBunches(jobs)
    val updateIDAndJobGroupId =
      if (jobBunches.length == 1 && jobBunches(0).length + jobGroupSpec.length < 1024 * 1024) {
        val b = new ByteArrayBuilder()
        b ++= "{\"job_groups\":".getBytes(StandardCharsets.UTF_8)
        addBunchBytes(b, Array(jobGroupSpec))
        b ++= ",\"bunch\":".getBytes(StandardCharsets.UTF_8)
        addBunchBytes(b, jobBunches(0))
        b ++= ",\"update\":".getBytes(StandardCharsets.UTF_8)
        b ++= JsonMethods.compact(updateJson).getBytes(StandardCharsets.UTF_8)
        b += '}'
        val data = b.result()
        val resp = req.post(
          s"/api/v1alpha/batches/$batchId/update-fast",
          new ByteArrayEntity(data, APPLICATION_JSON),
        )
        b.clear()
        ((resp \ "update_id").extract[Int], (resp \ "start_job_group_id").extract[Int])
      } else {
        val resp = req.post(s"/api/v1alpha/batches/$batchId/updates/create", updateJson)
        val updateID = (resp \ "update_id").extract[Int]
        val startJobGroupId = (resp \ "start_job_group_id").extract[Int]

        val b = new ByteArrayBuilder()
        b ++= "[".getBytes(StandardCharsets.UTF_8)
        b ++= jobGroupSpec
        b ++= "]".getBytes(StandardCharsets.UTF_8)
        req.post(
          s"/api/v1alpha/batches/$batchId/updates/$updateID/job-groups/create",
          new ByteArrayEntity(b.result(), APPLICATION_JSON),
        )

        b.clear()
        var i = 0
        while (i < jobBunches.length) {
          addBunchBytes(b, jobBunches(i))
          val data = b.result()
          req.post(
            s"/api/v1alpha/batches/$batchId/updates/$updateID/jobs/create",
            new ByteArrayEntity(data, APPLICATION_JSON),
          )
          b.clear()
          i += 1
        }

        req.patch(s"/api/v1alpha/batches/$b/updates/$updateID/commit")
        (updateID, startJobGroupId)
      }

    log.info(s"run: created update $updateIDAndJobGroupId for batch $batchId")
    updateIDAndJobGroupId
  }

  def run(batchRequest: BatchRequest, jobGroup: JobGroupRequest, jobs: IndexedSeq[JobRequest])
    : JobGroupResponse = {
    val batchID = newBatch(batchRequest)
    val (_, jobGroupId) = newJobGroup(batchID, batchRequest.token, jobGroup, jobs)
    waitForJobGroup(batchID, jobGroupId)
  }

  def waitForJobGroup(batchID: Int, jobGroupId: Int): JobGroupResponse = {

    Thread.sleep(600) // it is not possible for the batch to be finished in less than 600ms

    val start = System.nanoTime()

    while (true) {
      val jobGroup = req
        .get(s"/api/v1alpha/batches/$batchID/job-groups/$jobGroupId")
        .extract[JobGroupResponse]

      if (jobGroup.complete)
        return jobGroup

      // wait 10% of duration so far
      // at least, 50ms
      // at most, 5s
      val now = System.nanoTime()
      val elapsed = now - start
      val d = math.max(
        math.min(
          (0.1 * (0.8 + Random.nextFloat() * 0.4) * (elapsed / 1000.0 / 1000)).toInt,
          5000,
        ),
        50,
      )
      Thread.sleep(d)
    }

    throw new AssertionError("unreachable")
  }

  private def createBunches(jobs: IndexedSeq[JobRequest]): BoxedArrayBuilder[Array[Array[Byte]]] = {
    val bunches = new BoxedArrayBuilder[Array[Array[Byte]]]()
    val bunchb = new BoxedArrayBuilder[Array[Byte]]()

    var i = 0
    var size = 0
    while (i < jobs.length) {
      val jobBytes = getJsonBytes(jobs(i))
      if (size + jobBytes.length > 1024 * 1024) {
        bunches += bunchb.result()
        bunchb.clear()
        size = 0
      }
      bunchb += jobBytes
      size += jobBytes.length
      i += 1
    }
    assert(bunchb.size > 0)

    bunches += bunchb.result()
    bunchb.clear()
    bunches
  }

  private def getJsonBytes(obj: Any): Array[Byte] =
    JsonMethods.compact(Extraction.decompose(obj)).getBytes(StandardCharsets.UTF_8)

  private def addBunchBytes(b: ByteArrayBuilder, bunch: Array[Array[Byte]]): Unit = {
    var j = 0
    b += '['
    while (j < bunch.length) {
      if (j > 0)
        b += ','
      b ++= bunch(j)
      j += 1
    }
    b += ']'
  }

  override def close(): Unit =
    req.close()

  private[this] object JobProcessRequestSerializer
      extends CustomSerializer[JobProcess](_ =>
        (
          PartialFunction.empty,
          {
            case BashJob(image, command) =>
              JObject(
                "type" -> JString("docker"),
                "image" -> JString(image),
                "command" -> JArray(command.map(JString).toList),
              )
            case JvmJob(command, url, profile) =>
              JObject(
                "type" -> JString("jvm"),
                "command" -> JArray(command.map(JString).toList),
                "jar_spec" -> JObject("type" -> JString("jar_url"), "value" -> JString(url)),
                "profile" -> JBool(profile),
              )
          },
        )
      )

  private[this] object JobGroupStateDeserializer
      extends CustomSerializer[JobGroupState](_ =>
        (
          {
            case JString("failure") => JobGroupStates.Failure
            case JString("cancelled") => JobGroupStates.Cancelled
            case JString("success") => JobGroupStates.Success
            case JString("running") => JobGroupStates.Running
          },
          PartialFunction.empty,
        )
      )

  private[this] object JobGroupResponseDeserializer
      extends CustomSerializer[JobGroupResponse](implicit fmts =>
        (
          {
            case o: JObject =>
              JobGroupResponse(
                batch_id = (o \ "batch_id").extract[Int],
                job_group_id = (o \ "job_group_id").extract[Int],
                state = (o \ "state").extract[JobGroupState],
                complete = (o \ "complete").extract[Boolean],
                n_jobs = (o \ "n_jobs").extract[Int],
                n_completed = (o \ "n_completed").extract[Int],
                n_succeeded = (o \ "n_succeeded").extract[Int],
                n_failed = (o \ "n_failed").extract[Int],
                n_cancelled = (o \ "n_failed").extract[Int],
              )
          },
          PartialFunction.empty,
        )
      )
}
