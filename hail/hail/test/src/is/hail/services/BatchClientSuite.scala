package is.hail.services

import is.hail.{services, Revision}
import is.hail.backend.service.Main
import is.hail.collection.FastSeq
import is.hail.services.JobGroupStates.Failure
import is.hail.services.oauth2.CloudCredentials
import is.hail.utils._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

import java.nio.file.Path

class BatchClientSuite extends munit.FunSuite {

  private[this] var client: BatchClient = _
  private[this] var batchId: Int = _
  private[this] var parentJobGroupId: Int = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    client =
      BatchClient(
        DeployConfig.default,
        CloudCredentials(Some(Path.of("/test-gsa-key/key.json"))),
      )

    batchId =
      client.newBatch(
        BatchRequest(
          billing_project = "test",
          n_jobs = 0,
          token = tokenUrlSafe,
          attributes = Map("name" -> s"${getClass.getName}"),
        )
      )
  }

  override def beforeEach(context: BeforeEach): Unit = {
    super.beforeEach(context)
    parentJobGroupId = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = 0,
        token = tokenUrlSafe,
        attributes = Map("name" -> context.test.name),
        jobs = FastSeq(),
      )
    )._1
  }

  override def afterAll(): Unit = {
    client.close()
    super.afterAll()
  }

  test("CancelAfterNFailures") {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        cancel_after_n_failures = Some(1),
        token = tokenUrlSafe,
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "sleep 5m"),
            ),
            resources = Some(JobResources(preemptible = true)),
          ),
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 1"),
            ),
          ),
        ),
      )
    )
    val result = client.waitForJobGroup(batchId, jobGroupId)
    assertEquals(result.state, Failure)
    assertEquals(result.n_jobs, 2)
    assertEquals(result.n_failed, 1)
  }

  test("GetJobGroupJobsByState") {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        token = tokenUrlSafe,
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 0"),
            ),
          ),
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 1"),
            ),
          ),
        ),
      )
    )
    client.waitForJobGroup(batchId, jobGroupId): Unit
    Array(JobStates.Failed, JobStates.Success).foreach { state =>
      client.getJobGroupJobs(batchId, jobGroupId, Some(state)).foreach { jobs =>
        assertEquals(jobs.length, 1)
        assertEquals(jobs(0).state, state)
        assert(jobs.head.end_time.isDefined)
      }
    }
  }

  test("NewJobGroup") {
    (1 to 2).foreach { i =>
      val (jobGroupId, _) = client.newJobGroup(
        req = JobGroupRequest(
          batch_id = batchId,
          absolute_parent_id = parentJobGroupId,
          token = tokenUrlSafe,
          attributes = Map("name" -> s"JobGroup$i"),
          jobs = (1 to i).map { k =>
            JobRequest(
              always_run = false,
              process = BashJob(
                image = "ubuntu:24.04",
                command = Array("/bin/bash", "-c", s"echo 'job $k'"),
              ),
            )
          },
        )
      )

      val result = client.getJobGroup(batchId, jobGroupId)
      assertEquals(result.n_jobs, i)
    }
  }

  test("JvmJob") {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        token = tokenUrlSafe,
        attributes = Map("name" -> "TableStage"),
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = JvmJob(
              command = Array(Main.TEST),
              spec = GitRevision(Revision),
              profile = false,
            ),
          )
        ),
      )
    )

    val result = client.getJobGroup(batchId, jobGroupId)
    assertEquals(result.n_jobs, 1)
  }

  test("WaitForCancelledJobGroup") { // #15118
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        cancel_after_n_failures = Some(1),
        token = tokenUrlSafe,
        jobs = 0 until 10 map { _ =>
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "sleep 5s"),
            ),
            resources = Some(JobResources(preemptible = true)),
          )
        },
      )
    )

    Future {
      Thread.sleep(100)
      client.cancelJobGroup(batchId, jobGroupId)
    }: Unit

    val jg = client.waitForJobGroup(batchId, jobGroupId)
    assertEquals(jg.state, services.JobGroupStates.Cancelled)
  }
}
