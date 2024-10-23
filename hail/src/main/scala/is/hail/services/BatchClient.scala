package is.hail.services

import is.hail.expr.ir.ByteArrayBuilder
import is.hail.services.BatchClient.BunchMaxSizeBytes
import is.hail.services.oauth2.CloudCredentials
import is.hail.services.requests.Requester
import is.hail.utils._

import scala.util.Random

import java.net.URL
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path

import org.apache.http.entity.ByteArrayEntity
import org.apache.http.entity.ContentType.APPLICATION_JSON
import org.json4s.{CustomSerializer, DefaultFormats, Extraction, Formats, JInt, JObject, JString}
import org.json4s.JsonAST.{JArray, JBool}
import org.json4s.jackson.JsonMethods

case class BatchRequest(
  billing_project: String,
  token: String,
  n_jobs: Int,
  attributes: Map[String, String] = Map.empty,
)

case class JobGroupRequest(
  batch_id: Int,
  absolute_parent_id: Int,
  token: String,
  attributes: Map[String, String] = Map.empty,
  jobs: IndexedSeq[JobRequest] = FastSeq(),
)

case class JobRequest(
  always_run: Boolean,
  process: JobProcess,
  attributes: Map[String, String] = Map.empty,
  cloudfuse: Option[Array[CloudfuseConfig]] = None,
  resources: Option[JobResources] = None,
  regions: Option[Array[String]] = None,
)

sealed trait JobProcess
case class BashJob(image: String, command: Array[String]) extends JobProcess
case class JvmJob(command: Array[String], spec: JarSpec, profile: Boolean) extends JobProcess

sealed trait JarSpec
case class GitRevision(sha: String) extends JarSpec
case class JarUrl(url: String) extends JarSpec

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

  private[this] def BatchServiceScopes(env: Map[String, String]): Array[String] =
    env.get("HAIL_CLOUD") match {
      case Some("gcp") =>
        Array(
          "https://www.googleapis.com/auth/userinfo.profile",
          "https://www.googleapis.com/auth/userinfo.email",
          "openid",
        )
      case Some("azure") =>
        env.get("HAIL_AZURE_OAUTH_SCOPE").toArray
      case Some(cloud) =>
        throw new IllegalArgumentException(s"Unknown cloud: '$cloud'.")
      case None =>
        throw new IllegalArgumentException(s"HAIL_CLOUD must be set.")
    }

  def apply(deployConfig: DeployConfig, credentialsFile: Path, env: Map[String, String] = sys.env)
    : BatchClient =
    new BatchClient(Requester(
      new URL(deployConfig.baseUrl("batch")),
      CloudCredentials(credentialsFile, BatchServiceScopes(env), env),
    ))

  private val BunchMaxSizeBytes: Int = 1024 * 1024
}

case class BatchClient private (req: Requester) extends Logging with AutoCloseable {

  implicit private[this] val fmts: Formats =
    DefaultFormats +
      JobProcessRequestSerializer +
      JobGroupStateDeserializer +
      JobGroupResponseDeserializer +
      JarSpecSerializer

  def newBatch(createRequest: BatchRequest): Int = {
    val response = req.post("/api/v1alpha/batches/create", Extraction.decompose(createRequest))
    val batchId = (response \ "id").extract[Int]
    log.info(s"Created batch $batchId")
    batchId
  }

  def newJobGroup(req: JobGroupRequest): Int = {
    val nJobs = req.jobs.length
    val (updateId, startJobGroupId) = beginUpdate(req.batch_id, req.token, nJobs)
    log.info(s"Began update '$updateId' for batch '${req.batch_id}'.")

    createJobGroup(updateId, req)
    log.info(s"Created job group $startJobGroupId for batch ${req.batch_id}")

    createJobsIncremental(req.batch_id, updateId, req.jobs)
    log.info(s"Submitted $nJobs in job group $startJobGroupId for batch ${req.batch_id}")

    commitUpdate(req.batch_id, updateId)
    log.info(s"Committed update $updateId for batch ${req.batch_id}.")

    startJobGroupId
  }

  def getJobGroup(batchId: Int, jobGroupId: Int): JobGroupResponse =
    req
      .get(s"/api/v1alpha/batches/$batchId/job-groups/$jobGroupId")
      .extract[JobGroupResponse]

  def waitForJobGroup(batchId: Int, jobGroupId: Int): JobGroupResponse = {
    val start = System.nanoTime()

    while (true) {
      val jobGroup = getJobGroup(batchId, jobGroupId)

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

  override def close(): Unit =
    req.close()

  private[this] def createJobsIncremental(
    batchId: Int,
    updateId: Int,
    jobs: IndexedSeq[JobRequest],
  ): Unit = {
    val buff = new ByteArrayBuilder(BunchMaxSizeBytes)
    var sym = "["

    def flush(): Unit = {
      buff ++= "]".getBytes(UTF_8)
      req.post(
        s"/api/v1alpha/batches/$batchId/updates/$updateId/jobs/create",
        new ByteArrayEntity(buff.result(), APPLICATION_JSON),
      )
      buff.clear()
      sym = "["
    }

    for ((job, idx) <- jobs.zipWithIndex) {
      val jobPayload = jobToJson(job, idx).getBytes(UTF_8)

      if (buff.size + jobPayload.length > BunchMaxSizeBytes) {
        flush()
      }

      buff ++= sym.getBytes(UTF_8)
      buff ++= jobPayload
      sym = ","
    }

    if (buff.size > 0) { flush() }
  }

  private[this] def jobToJson(j: JobRequest, jobIdx: Int): String =
    JsonMethods.compact {
      Extraction.decompose(j)
        .asInstanceOf[JObject]
        .merge(
          JObject(
            "job_id" -> JInt(jobIdx + 1),
            "in_update_job_group_id" -> JInt(1),
          )
        )
    }

  private[this] def beginUpdate(batchId: Int, token: String, nJobs: Int): (Int, Int) =
    req
      .post(
        s"/api/v1alpha/batches/$batchId/updates/create",
        JObject(
          "token" -> JString(token),
          "n_jobs" -> JInt(nJobs),
          "n_job_groups" -> JInt(1),
        ),
      )
      .as { case obj: JObject =>
        (
          (obj \ "update_id").extract[Int],
          (obj \ "start_job_group_id").extract[Int],
        )
      }

  private[this] def commitUpdate(batchId: Int, updateId: Int): Unit =
    req.patch(s"/api/v1alpha/batches/$batchId/updates/$updateId/commit")

  private[this] def createJobGroup(updateId: Int, jobGroup: JobGroupRequest): Unit =
    req.post(
      s"/api/v1alpha/batches/${jobGroup.batch_id}/updates/$updateId/job-groups/create",
      JArray(List(
        JObject(
          "job_group_id" -> JInt(1), // job group id relative to the update
          "absolute_parent_id" -> JInt(jobGroup.absolute_parent_id),
          "attributes" -> Extraction.decompose(jobGroup.attributes),
        )
      )),
    )

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
            case JvmJob(command, jarSpec, profile) =>
              JObject(
                "type" -> JString("jvm"),
                "command" -> JArray(command.map(JString).toList),
                "jar_spec" -> Extraction.decompose(jarSpec),
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

  private[this] object JarSpecSerializer
      extends CustomSerializer[JarSpec](_ =>
        (
          PartialFunction.empty,
          {
            case JarUrl(url) =>
              JObject("type" -> JString("jar_url"), "value" -> JString(url))
            case GitRevision(sha) =>
              JObject("type" -> JString("git_revision"), "value" -> JString(sha))
          },
        )
      )
}
