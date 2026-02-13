package is.hail.services

import is.hail.collection.{ByteArrayBuilder, FastSeq}
import is.hail.services.BatchClient.{
  BunchMaxSizeBytes, JarSpecSerializer, JobGroupResponseDeserializer, JobGroupStateDeserializer,
  JobListEntryDeserializer, JobProcessRequestSerializer, JobStateDeserializer,
}
import is.hail.services.JobGroupStates.isTerminal
import is.hail.services.oauth2.CloudCredentials
import is.hail.services.requests.{ClientResponseException, Requester}
import is.hail.utils._

import scala.collection.compat.immutable.LazyList
import scala.util.Random

import java.net.{URL, URLEncoder}
import java.nio.charset.StandardCharsets.UTF_8

import org.apache.http.entity.ByteArrayEntity
import org.apache.http.entity.ContentType.APPLICATION_JSON
import org.json4s.{
  CustomSerializer, DefaultFormats, Extraction, Formats, JInt, JNull, JObject, JString,
}
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
  cancel_after_n_failures: Option[Int] = None,
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
  cpu: Option[String] = None,
  memory: Option[String] = None,
  storage: Option[String] = None,
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

  def isTerminal(s: JobGroupState): Boolean =
    s != Running
}

sealed trait JobState extends Product with Serializable

object JobStates {
  case object Pending extends JobState
  case object Ready extends JobState
  case object Creating extends JobState
  case object Running extends JobState
  case object Cancelled extends JobState
  case object Error extends JobState
  case object Failed extends JobState
  case object Success extends JobState
}

case class JobListEntry(
  batch_id: Int,
  job_id: Int,
  state: JobState,
  exit_code: Int,
)

object BatchClient {
  object RequiredOAuth2Scopes {
    private[this] val Google: Array[String] =
      Array(
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
      )

    private[this] val Microsoft: Array[String] =
      Array(".default")

    def apply(env: Map[String, String] = sys.env): Array[String] =
      env.get("HAIL_CLOUD") match {
        case None | Some("gcp") => Google
        case Some("azure") => env.get("HAIL_AZURE_OAUTH_SCOPE").map(Array(_)).getOrElse(Microsoft)
        case None => throw new IllegalArgumentException(s"'HAIL_CLOUD' must be set.")
      }
  }

  val BunchMaxSizeBytes: Int = 1024 * 1024

  def apply(
    deployConfig: DeployConfig,
    credentials: CloudCredentials,
    env: Map[String, String] = sys.env,
  ): BatchClient =
    new BatchClient(
      Requester(
        new URL(deployConfig.baseUrl("batch")),
        credentials.scoped(RequiredOAuth2Scopes(env)),
      )
    )

  object JobProcessRequestSerializer extends CustomSerializer[JobProcess](implicit fmts =>
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

  object JobGroupStateDeserializer extends CustomSerializer[JobGroupState](_ =>
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

  object JobGroupResponseDeserializer extends CustomSerializer[JobGroupResponse](implicit fmts =>
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

  object JarSpecSerializer extends CustomSerializer[JarSpec](_ =>
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

  object JobStateDeserializer
      extends CustomSerializer[JobState](_ =>
        (
          {
            case JString("Pending") => JobStates.Pending
            case JString("Ready") => JobStates.Ready
            case JString("Creating") => JobStates.Creating
            case JString("Running") => JobStates.Running
            case JString("Cancelled") => JobStates.Cancelled
            case JString("Error") => JobStates.Error
            case JString("Failed") => JobStates.Failed
            case JString("Success") => JobStates.Success
          },
          PartialFunction.empty,
        )
      )

  object JobListEntryDeserializer
      extends CustomSerializer[JobListEntry](implicit fmts =>
        (
          {
            case o: JObject =>
              JobListEntry(
                batch_id = (o \ "batch_id").extract[Int],
                job_id = (o \ "job_id").extract[Int],
                state = (o \ "state").extract[JobState],
                exit_code = (o \ "exit_code").extract[Int],
              )
          },
          PartialFunction.empty,
        )
      )
}

case class BatchClient private (req: Requester) extends Logging with AutoCloseable {

  implicit private[this] val fmts: Formats =
    DefaultFormats +
      JobProcessRequestSerializer +
      JobGroupStateDeserializer +
      JobGroupResponseDeserializer +
      JarSpecSerializer +
      JobStateDeserializer +
      JobListEntryDeserializer

  private[this] def paginated[S, A](s0: S)(f: S => (A, S)): LazyList[A] = {
    val (a, s1) = f(s0)
    LazyList.cons(a, paginated(s1)(f))
  }

  def newBatch(createRequest: BatchRequest): Int = {
    val response = req.post("/api/v1alpha/batches/create", Extraction.decompose(createRequest))
    val batchId = (response \ "id").extract[Int]
    logger.info(s"Created batch $batchId")
    batchId
  }

  def newJobGroup(req: JobGroupRequest): (Int, Int) =
    retryable { attempts =>
      try {
        val batchId = req.batch_id
        val nJobs = req.jobs.length
        val (updateId, startJobGroupId, startJobId) = beginUpdate(batchId, req.token, nJobs)
        logger.info(s"Began update '$updateId' for batch '$batchId'.")

        createJobGroup(updateId, req)
        logger.info(s"Created job group '$startJobGroupId' for batch '$batchId'.")

        createJobsIncremental(batchId, updateId, req.jobs)
        logger.info(s"Created '$nJobs' jobs in job group '$startJobGroupId' for batch '$batchId'.")

        commitUpdate(batchId, updateId)
        logger.info(s"Committed update '$updateId' for batch '$batchId'.")

        (startJobGroupId, startJobId)
      } catch {
        case e: ClientResponseException
            if e.status == 400
              && e.getMessage.contains("job group specs were not submitted in order") =>
          val delay = delayMsForTry(attempts + 1)
          logger.warn(
            f"Tried to update batch '${req.batch_id}' before another process could commit " +
              "an earlier update. This is most likely caused by running parallel query pipelines " +
              "in the same batch. Batch does not yet support out-of-order updates. Sleeping for " +
              f"'$delay' ms to allow the other update complete.",
            e,
          )
          Thread.sleep(delay.toLong)
          retry
      }
    }

  def getJobGroup(batchId: Int, jobGroupId: Int): JobGroupResponse =
    req
      .get(s"/api/v1alpha/batches/$batchId/job-groups/$jobGroupId")
      .extract[JobGroupResponse]

  def getJobGroupJobs(batchId: Int, jobGroupId: Int, status: Option[JobState] = None)
    : LazyList[IndexedSeq[JobListEntry]] = {
    val q = status.map(s => s"state=${s.toString.toLowerCase}").getOrElse("")
    paginated(Some(0): Option[Int]) {
      case Some(jobId) =>
        req.get(
          s"/api/v2alpha/batches/$batchId/job-groups/$jobGroupId/jobs?q=${URLEncoder.encode(q, UTF_8)}&last_job_id=$jobId"
        )
          .as { case obj: JObject =>
            (
              (obj \ "jobs").extract[IndexedSeq[JobListEntry]],
              (obj \ "last_job_id").extract[Option[Int]],
            )
          }
      case None =>
        (IndexedSeq.empty, None)
    }
      .takeWhile(_.nonEmpty)
  }

  def cancelJobGroup(batchId: Int, jobGroupId: Int): Unit =
    req.patch(s"/api/v1alpha/batches/$batchId/job-groups/$jobGroupId/cancel"): Unit

  def waitForJobGroup(batchId: Int, jobGroupId: Int): JobGroupResponse = {
    val start = System.nanoTime()

    while (true) {
      val jobGroup = getJobGroup(batchId, jobGroupId)

      if (isTerminal(jobGroup.state)) return jobGroup

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
      Thread.sleep(d.toLong)
    }

    unreachable
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
      ): Unit
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
            // Batch allows clients to create multiple job groups in an update.
            // For each table stage, we create one job group in an update; all jobs in
            // that update belong to that one job group. This allows us to abstract updates
            // from the case class used by the ServiceBackend but that information needs to
            // get added back here.
            "in_update_job_group_id" -> JInt(1),
          )
        )
    }

  private[this] def beginUpdate(batchId: Int, token: String, nJobs: Int): (Int, Int, Int) =
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
          (obj \ "start_job_id").extract[Int],
        )
      }

  private[this] def commitUpdate(batchId: Int, updateId: Int): Unit =
    req.patch(s"/api/v1alpha/batches/$batchId/updates/$updateId/commit"): Unit

  private[this] def createJobGroup(updateId: Int, jobGroup: JobGroupRequest): Unit =
    req.post(
      s"/api/v1alpha/batches/${jobGroup.batch_id}/updates/$updateId/job-groups/create",
      JArray(List(
        JObject(
          "job_group_id" -> JInt(1), // job group id relative to the update
          "absolute_parent_id" -> JInt(jobGroup.absolute_parent_id),
          "cancel_after_n_failures" -> jobGroup.cancel_after_n_failures.map(JInt(_)).getOrElse(
            JNull
          ),
          "attributes" -> Extraction.decompose(jobGroup.attributes),
        )
      )),
    ): Unit
}
